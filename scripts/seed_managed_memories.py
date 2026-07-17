from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import vertexai
from redis_agent_memory import AgentMemory

from valueharbor_agent.config import Settings

APP_NAME = "valueharbor-shopping-agent"


def load_memories(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_redis(settings: Settings, memories: list[dict[str, Any]]) -> tuple[int, int]:
    if not settings.memory_configured:
        raise RuntimeError("Redis Agent Memory settings are incomplete")
    with AgentMemory(
        settings.agent_memory_base_url,
        store_id=settings.agent_memory_store_id,
        api_key=settings.agent_memory_api_key,
    ) as client:
        response = client.bulk_create_long_term_memories(memories=memories)
    return len(response.created), len(response.errors or [])


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
    for memory in memories:
        scope = {"app_name": APP_NAME, "user_id": memory["owner_id"]}
        identity = (memory["text"], tuple(sorted(scope.items())))
        if identity in existing:
            skipped += 1
            continue
        operation = client.agent_engines.memories.create(
            name=name,
            fact=memory["text"],
            scope=scope,
        )
        if hasattr(operation, "result"):
            operation.result()
        created += 1
        existing.add(identity)
    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed identical ValueHarbor facts into both managed memory providers."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/generated/memory_seeds.jsonl"),
    )
    args = parser.parse_args()
    settings = Settings()
    memories = load_memories(args.data)
    redis_created, redis_errors = seed_redis(settings, memories)
    vertex_created, vertex_skipped = seed_vertex(settings, memories)
    print(
        f"Redis Agent Memory: {redis_created} created, {redis_errors} errors; "
        f"ADK Memory Bank: {vertex_created} created, {vertex_skipped} already present"
    )
    if redis_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
