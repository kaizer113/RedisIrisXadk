"""Real Redis native-idempotency and HTTP replay across two application replicas."""

import asyncio
import json
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import uvicorn
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis_fastapi.config import get_settings

from valuewholesale_onprem import api
from valuewholesale_onprem.config import Settings


@asynccontextmanager
async def http_server(application):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(application, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Test HTTP server stopped during startup")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()


async def sse_events(response):
    event = {}
    async for line in response.aiter_lines():
        if line.startswith("id:"):
            event["id"] = line[3:].strip()
        elif line.startswith("event:"):
            event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            event["data"] = json.loads(line[5:].strip())
        elif not line and event:
            yield event
            event = {}


async def test_native_mesh_disconnect_retry_and_cross_replica_replay(tmp_path, monkeypatch):
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("Install Redis 8.6.2+ locally to run the native Mesh integration test")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    proc = subprocess.Popen(
        [
            binary,
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
            "--protected-mode",
            "yes",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/0"
    redis = Redis.from_url(url, decode_responses=True)
    get_settings.cache_clear()
    monkeypatch.setenv("REDIS_URL", url)
    monkeypatch.setenv("REDIS_STREAM_DONE_TTL", "60")
    try:
        async with asyncio.timeout(8):
            while True:
                try:
                    await redis.ping()
                    break
                except (OSError, RedisConnectionError):
                    await asyncio.sleep(0.02)
        version = tuple(
            int(part) for part in (await redis.info("server"))["redis_version"].split(".")
        )
        if version < (8, 6, 2):
            pytest.skip("Local Redis predates native producer idempotency (requires 8.6.2+)")

        class Producer:
            def __init__(self, services, identity, run_id, message, enabled, emit, model=None):
                self.services, self.emit = services, emit

            async def run(self):
                await self.services.redis.incr("native-test-producer-count")
                for index in range(4):
                    await self.emit(
                        {
                            "type": "trace",
                            "step": {
                                "service": "agent_mesh",
                                "title": f"step-{index}",
                                "status": "completed",
                            },
                        }
                    )
                    await asyncio.sleep(0.12)
                await self.emit({"type": "answer", "answer": "Retained answer"})

        monkeypatch.setattr(api, "Turn", Producer)
        settings = Settings(
            redis_url=url,
            onprem_session_secret="native-test-secret-" * 3,
            iris_agent_memory_url="http://unused",
            iris_agent_memory_store_id="unused",
            iris_agent_memory_api_key="unused",
            iris_agent_memory_namespace_id="unused",
            iris_agent_playbooks_url="http://unused",
            iris_agent_playbooks_playbook_id="unused",
            iris_agent_playbooks_api_key="unused",
            iris_intent_router_url="http://unused",
            iris_intent_router_id="unused",
            iris_intent_router_api_key="unused",
            iris_context_gateway_url="http://unused",
            iris_context_gateway_api_key="unused",
            google_cloud_project="unused",
        )

        def service_factory(settings, redis):
            return SimpleNamespace(
                settings=settings, redis=redis, open=AsyncMock(), close=AsyncMock()
            )

        app_a = api.create_app(settings, services_factory=service_factory)
        app_b = api.create_app(settings, services_factory=service_factory)
        async with http_server(app_a) as first, http_server(app_b) as second:
            async with httpx.AsyncClient(base_url=first, timeout=10) as browser:
                assert (
                    await browser.post("/api/session", json={"member_id": "member-1001"})
                ).is_success
                body = {"message": "Native stream test", "run_id": str(uuid4())}
                start = await browser.post("/api/chat", json=body)
                assert start.is_success and start.json()["created"]
                path = start.json()["events_url"]
                async with browser.stream("GET", path) as response:
                    assert response.is_success
                    async for event in sse_events(response):
                        if event.get("event") == "trace":
                            cursor = event["id"]
                            first_title = event["data"]["step"]["title"]
                            break
                # The HTTP subscriber is now closed; the producer still runs on replica A.
                async with httpx.AsyncClient(
                    base_url=second, cookies=browser.cookies, timeout=10
                ) as reconnected:
                    retry = await reconnected.post("/api/chat", json=body)
                    assert retry.is_success and retry.json()["created"] is False
                    received = []
                    async with reconnected.stream(
                        "GET", path, headers={"Last-Event-ID": cursor}
                    ) as response:
                        assert response.is_success
                        async for event in sse_events(response):
                            received.append(event)
                            if event.get("event") == "completed":
                                break
                    assert not any(
                        event.get("data", {}).get("step", {}).get("title") == first_title
                        for event in received
                    )
                    assert any(
                        event.get("data", {}).get("answer") == "Retained answer"
                        for event in received
                    )
                    snapshot = await reconnected.get(f"/api/runs/{body['run_id']}")
                    assert snapshot.json()["status"] == "completed"
                    assert await redis.get("native-test-producer-count") == "1"
                    await reconnected.post("/api/session", json={"member_id": "member-1002"})
                    assert (await reconnected.get(path)).status_code == 404
    finally:
        await redis.aclose()
        proc.terminate()
        proc.wait(timeout=5)
        get_settings.cache_clear()
