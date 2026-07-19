from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import vertexai
from redis_agent_memory import AgentMemory

from valuewholesale_agent.config import Settings

APP_NAME = "valuewholesale-shopping-agent"
REDIS_BATCH_SIZE = 100
REDIS_TIMEOUT_MS = 120_000
VERTEX_WRITES_PER_MINUTE = 80
VERTEX_HTTP_TIMEOUT_MS = 60_000


def load_memories(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_redis(settings: Settings, memories: list[dict[str, Any]]) -> tuple[int, int]:
    if not settings.memory_configured:
        raise RuntimeError("Redis Agent Memory settings are incomplete")
    created = 0
    errors = 0
    with AgentMemory(
        settings.agent_memory_base_url,
        store_id=settings.agent_memory_store_id,
        api_key=settings.agent_memory_api_key,
        timeout_ms=REDIS_TIMEOUT_MS,
    ) as client:
        for start in range(0, len(memories), REDIS_BATCH_SIZE):
            batch = memories[start : start + REDIS_BATCH_SIZE]
            for attempt in range(3):
                try:
                    response = client.bulk_create_long_term_memories(
                        memories=batch,
                        timeout_ms=REDIS_TIMEOUT_MS,
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            created += len(response.created)
            errors += len(response.errors or [])
    return created, errors


def _scope_dict(scope: Any) -> dict[str, str]:
    if isinstance(scope, dict):
        return {str(key): str(value) for key, value in scope.items()}
    if hasattr(scope, "items"):
        return {str(key): str(value) for key, value in scope.items()}
    return {}


def seed_vertex(settings: Settings, memories: list[dict[str, Any]]) -> tuple[int, int]:
    if not settings.vertex_memory_configured:
        raise RuntimeError("GOOGLE_AGENT_ENGINE_ID is not configured")
    client = vertexai.Client(
        project=settings.google_cloud_project,
        location=settings.google_memory_location,
        http_options={"timeout": VERTEX_HTTP_TIMEOUT_MS},
    )
    name = (
        f"projects/{settings.google_cloud_project}/locations/{settings.google_memory_location}"
        f"/reasoningEngines/{settings.google_agent_engine_id}"
    )
    existing = {
        (memory.fact, tuple(sorted(_scope_dict(memory.scope).items())))
        for memory in client.agent_engines.memories.list(name=name)
    }
    created = 0
    skipped = 0
    pending_operations: list[Any] = []
    last_create_started = 0.0

    def finish_pending() -> None:
        nonlocal created
        for operation in pending_operations:
            if hasattr(operation, "result"):
                operation.result()
            created += 1
        pending_operations.clear()

    for memory in memories:
        scope = {"app_name": APP_NAME, "user_id": memory["owner_id"]}
        identity = (memory["text"], tuple(sorted(scope.items())))
        if identity in existing:
            skipped += 1
            continue
        minimum_interval = 60 / VERTEX_WRITES_PER_MINUTE
        elapsed = time.monotonic() - last_create_started
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        for attempt in range(5):
            try:
                last_create_started = time.monotonic()
                operation = client.agent_engines.memories.create(
                    name=name,
                    fact=memory["text"],
                    scope=scope,
                )
                break
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == 4:
                    raise
                time.sleep(65)
        pending_operations.append(operation)
        if len(pending_operations) == 5:
            finish_pending()
            if created % 25 == 0:
                print(f"ADK Memory Bank: {created} new memories completed", flush=True)
        existing.add(identity)
    finish_pending()
    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed identical Value Wholesale facts into both managed memory providers."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/generated/memory_seeds.jsonl"),
    )
    parser.add_argument(
        "--owner-id",
        help="Seed only memories belonging to one member ID.",
    )
    parser.add_argument(
        "--provider",
        choices=("both", "redis", "vertex"),
        default="both",
        help="Limit seeding to one managed provider.",
    )
    args = parser.parse_args()
    settings = Settings()
    memories = load_memories(args.data)
    if args.owner_id:
        memories = [memory for memory in memories if memory["owner_id"] == args.owner_id]
    if not memories:
        raise RuntimeError("No memories matched the requested seed scope")
    redis_created = redis_errors = 0
    vertex_created = vertex_skipped = 0
    if args.provider in {"both", "redis"}:
        redis_created, redis_errors = seed_redis(settings, memories)
        print(
            f"Redis Agent Memory: {redis_created} created, {redis_errors} errors",
            flush=True,
        )
    if args.provider in {"both", "vertex"}:
        vertex_created, vertex_skipped = seed_vertex(settings, memories)
        print(
            f"ADK Memory Bank: {vertex_created} created, "
            f"{vertex_skipped} already present",
            flush=True,
        )
    if args.provider in {"both", "redis"} and redis_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
