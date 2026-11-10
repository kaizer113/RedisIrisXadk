#!/usr/bin/env python3
"""Exercise the live port-8083 app using synthetic members and real Iris/Vertex services.

Run with the onprem Python environment. This writes demo conversations and one
explicit shopping preference. Receipts contain synthetic answers but no keys/cookies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def json_request(client: httpx.AsyncClient, method: str, path: str, **kwargs):
    response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


async def sse_events(response: httpx.Response):
    event: dict = {}
    data: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("id:"):
            event["id"] = line[3:].strip()
        elif line.startswith("event:"):
            event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and (event or data):
            if data:
                event["data"] = json.loads("\n".join(data))
            yield event
            event, data = {}, []


async def snapshot(client: httpx.AsyncClient, run_id: str, timeout: float = 180) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        if response.status_code == 200:
            value = response.json()
            if value["status"] in ("completed", "failed"):
                require(value["status"] == "completed", "The live agent turn failed")
                require(bool(value.get("answer")), "Completed turn has no retained answer")
                return value
        else:
            require(response.status_code == 404, "Unexpected snapshot response")
        await asyncio.sleep(0.3)
    raise RuntimeError("Live turn did not complete before the smoke deadline")


async def turn(client: httpx.AsyncClient, message: str) -> tuple[dict, dict, dict]:
    body = {"message": message, "run_id": str(uuid4())}
    started = await json_request(client, "POST", "/api/chat", json=body)
    require(started["created"] is True, "A fresh run did not create a producer")
    saved = await snapshot(client, body["run_id"])
    return body, started, saved


async def run(base_url: str, receipts: dict) -> None:
    timeout = httpx.Timeout(200, connect=10)
    async with (
        httpx.AsyncClient(base_url=base_url, timeout=timeout) as first,
        httpx.AsyncClient(base_url=base_url, timeout=timeout) as other,
    ):
        health = await json_request(first, "GET", "/api/health")
        receipts["health"] = health
        require(health.get("ready") is True, "The live app is not ready")
        available = await json_request(first, "GET", "/api/members")
        known = {member["id"] for member in available["members"]}
        require({"member-1001", "member-1002"} <= known, "Expected synthetic members missing")
        receipts["sessions"] = []
        for browser, member in ((first, "member-1001"), (other, "member-1002")):
            created = await json_request(
                browser, "POST", "/api/session", json={"member_id": member}
            )
            receipts["sessions"].append(created)

        faq_body, _, faq = await turn(first, "How does warehouse pickup work?")
        receipts["faq"] = {"run_id": faq_body["run_id"], **faq}
        require(faq.get("model_calls") == 0, "Published FAQ did not bypass the model")
        require(
            bool((faq.get("published") or {}).get("entry_id")), "FAQ has no published provenance"
        )

        # Attach early, close the subscriber, retry through a new connection, then resume
        # from the actual last event id. This does not identify which Uvicorn worker served it.
        body = {
            "message": "Find dark chocolate and check current inventory at my home warehouse.",
            "run_id": str(uuid4()),
        }
        started = await json_request(first, "POST", "/api/chat", json=body)
        require(started["created"] is True, "Shopping run did not start")
        first_event = None
        async with first.stream("GET", started["events_url"]) as response:
            response.raise_for_status()
            async for event in sse_events(response):
                if event.get("id") and event.get("event") == "trace":
                    first_event = event
                    break
        require(first_event is not None, "No trace event was received before disconnect")
        during = await json_request(first, "GET", f"/api/runs/{body['run_id']}")
        require(during["status"] == "running", "Turn finished before disconnect could be tested")
        async with httpx.AsyncClient(
            base_url=base_url, cookies=first.cookies, timeout=timeout
        ) as reconnected:
            retry = await json_request(reconnected, "POST", "/api/chat", json=body)
            require(retry["created"] is False, "Retry created another producer")
            resumed = []
            async with reconnected.stream(
                "GET", started["events_url"], headers={"Last-Event-ID": first_event["id"]}
            ) as response:
                response.raise_for_status()
                async for event in sse_events(response):
                    resumed.append(event)
                    require(
                        event.get("id") != first_event["id"], "Resume replayed the cursor event"
                    )
                    if event.get("event") in ("completed", "failed"):
                        break
            require(any(e.get("event") == "completed" for e in resumed), "SSE did not complete")
            shopping = await snapshot(reconnected, body["run_id"])
            require(shopping.get("model_calls", 0) > 0, "Shopping skipped actual inference")
            steps = [e["step"] for e in shopping["events"] if e.get("type") == "trace"]
            require(
                any(
                    s.get("service") == "context_gateway"
                    and s.get("status") == "completed"
                    and "__" in s.get("title", "")
                    for s in steps
                ),
                "Shopping completed without a successful discovered business tool call",
            )
            retained_retry = await json_request(reconnected, "POST", "/api/chat", json=body)
            require(retained_retry["created"] is False, "Retained run restarted")
            same = await snapshot(reconnected, body["run_id"])
            require(same == shopping, "Retry changed the retained result")
        for path in (started["events_url"], f"/api/runs/{body['run_id']}"):
            denied = await other.get(path)
            require(denied.status_code == 404, "Another member could read the run")
        receipts["shopping_mesh"] = {
            "run_id": body["run_id"],
            "disconnected_while_running": True,
            "cursor": first_event["id"],
            "resumed_events": resumed,
            "retry_created": retry["created"],
            "retained_snapshot_unchanged": True,
            "cross_member_status": 404,
            "worker_assignment": "Not observed; no cross-worker claim",
            "snapshot": shopping,
        }

        marker = "demo-preference-" + uuid4().hex[:12]
        receipts["memory"] = {"marker": marker, "memory_ids": []}
        preference_body, _, preference = await turn(
            first,
            "Remember that I prefer dark chocolate and avoid milk chocolate. "
            f"Save this shopping preference with the exact note marker {marker} in its text.",
        )
        own = await json_request(first, "GET", "/api/memory")
        receipts["memory"]["memory_ids"] = [
            row["memory_id"] for row in own["memories"] if marker in row.get("text", "")
        ]
        require(marker in json.dumps(own["memories"]), "Requested preference was not persisted")
        foreign = await json_request(other, "GET", "/api/memory")
        require(
            marker not in json.dumps(foreign["memories"]), "Preference leaked to another member"
        )
        created = await json_request(
            first, "POST", "/api/session", json={"member_id": "member-1001"}
        )
        receipts["sessions"].append(created)
        continued = await json_request(first, "GET", "/api/memory")
        require(
            marker in json.dumps(continued["memories"]), "Preference did not survive a new session"
        )
        receipts["memory"] = {
            "run_id": preference_body["run_id"],
            "marker": marker,
            "memory_ids": [
                row["memory_id"] for row in own["memories"] if marker in row.get("text", "")
            ],
            "snapshot": preference,
            "persisted": True,
            "new_session_recalled": True,
            "cross_member_absent": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path, help="Private JSON receipt file")
    args = parser.parse_args()
    os.umask(0o077)
    receipts = {"base_url": args.base_url, "passed": False}
    try:
        asyncio.run(run(args.base_url, receipts))
        receipts["passed"] = True
    except Exception as exc:
        receipts["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.output.write_text(json.dumps(receipts, indent=2) + "\n")
        args.output.chmod(0o600)
        print(json.dumps({"passed": receipts["passed"], "receipt": str(args.output)}))


if __name__ == "__main__":
    main()
