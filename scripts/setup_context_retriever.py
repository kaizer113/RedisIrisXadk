"""Create or update the existing shopping Context Surface using ctxctl."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from context_surfaces import UnifiedClient
from dotenv import dotenv_values

from scripts.generate_dataset import records_for_experience
from valuewholesale_agent.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "valuewholesale_agent" / "context_models.py"


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            output.append(line)
            continue
        key = line.split("=", 1)[0]
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in updates.items() if key not in seen)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def ctxctl(*args: str, admin_key: str | None = None) -> Any:
    command = ["uv", "run", "ctxctl", "--no-color", "-o", "json", *args]
    if admin_key:
        command.extend(["--admin-key", admin_key])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ctxctl command failed")
    return json.loads(result.stdout) if result.stdout.strip() else None


def redis_connection(redis_url: str) -> tuple[str, str, str, bool]:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(redis_url)
    if not parsed.hostname or not parsed.port:
        raise ValueError("REDIS_URL must include a hostname and port")
    return (
        f"{parsed.hostname}:{parsed.port}",
        unquote(parsed.username or "default"),
        unquote(parsed.password or ""),
        parsed.scheme == "rediss",
    )


def ensure_surface(
    env: dict[str, str],
    settings: Settings,
    env_path: Path,
    *,
    force_agent_key: bool,
) -> tuple[str, str]:
    admin_key = env.get("CTX_ADMIN_KEY", "")
    if not admin_key:
        raise SystemExit("CTX_ADMIN_KEY is required in .env")
    redis_url = env.get("REDIS_URL", "")
    if not redis_url:
        raise SystemExit("REDIS_URL is required in .env")

    surface_id = env.get("CTX_SURFACE_ID", "")
    if surface_id:
        try:
            ctxctl("surface", "describe", surface_id, admin_key=admin_key)
        except RuntimeError:
            surface_id = ""

    if not surface_id:
        for surface in ctxctl("surface", "list", admin_key=admin_key) or []:
            if surface.get("name") == settings.effective_context_surface_name:
                surface_id = str(surface["id"])
                break

    if surface_id:
        ctxctl(
            "surface",
            "update",
            surface_id,
            "--name",
            settings.effective_context_surface_name,
            "--description",
            f"Governed live ecommerce context for the "
            f"{settings.experience.brand_name} ADK shopping agent.",
            "--models",
            str(MODELS_PATH),
            admin_key=admin_key,
        )
        print(f"Updated Context Surface {surface_id}")
    else:
        address, username, password, tls_enabled = redis_connection(redis_url)
        create_args = [
            "surface",
            "create",
            "--name",
            settings.effective_context_surface_name,
            "--description",
            f"Governed live ecommerce context for the "
            f"{settings.experience.brand_name} ADK shopping agent.",
            "--models",
            str(MODELS_PATH),
            "--redis-addr",
            address,
            "--redis-username",
            username,
            "--redis-password",
            password,
        ]
        if tls_enabled:
            create_args.append("--redis-tls")
        payload = ctxctl(*create_args, admin_key=admin_key)
        surface_id = str(payload["id"])
        print(f"Created Context Surface {surface_id}")

    agent_key = "" if force_agent_key else env.get("MCP_AGENT_KEY", "")
    if not agent_key:
        payload = ctxctl(
            "agent",
            "create",
            "--surface-id",
            surface_id,
            "--name",
            settings.effective_context_agent_name,
            "--description",
            settings.effective_context_agent_display_name,
            admin_key=admin_key,
        )
        agent_key = str(payload["key"])
        print("Created a new Context Retriever agent key")

    upsert_env(env_path, {"CTX_SURFACE_ID": surface_id, "MCP_AGENT_KEY": agent_key})
    return surface_id, agent_key


async def import_records(
    surface_id: str,
    admin_key: str,
    settings: Settings,
) -> None:
    from valuewholesale_agent.context_models import (
        Inventory,
        Member,
        Order,
        OrderItem,
        Product,
        Warehouse,
    )

    datasets = records_for_experience(settings.experience_id)
    entities = {
        Product: datasets["products"],
        Warehouse: datasets["warehouses"],
        Inventory: datasets["inventory"],
        Member: datasets["members"],
        Order: datasets["orders"],
        OrderItem: datasets["order_items"],
    }
    async with UnifiedClient() as client:
        for model, rows in entities.items():
            result = await client.import_data(
                admin_key=admin_key,
                context_surface_id=surface_id,
                records=[model(**row) for row in rows],
                on_conflict="overwrite",
                on_error="fail_fast",
            )
            print(f"{model.__name__}: imported={result.imported}, failed={result.failed}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate-agent-key", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    env_path = args.env_file.resolve()
    raw_env = dotenv_values(env_path)
    env = {key: str(value or "") for key, value in raw_env.items()}
    os.environ.update(env)
    get_settings.cache_clear()
    settings = Settings(_env_file=env_path)
    surface_id, agent_key = ensure_surface(
        env,
        settings,
        env_path,
        force_agent_key=args.rotate_agent_key,
    )
    await import_records(surface_id, env["CTX_ADMIN_KEY"], settings)
    tools = await UnifiedClient().list_tools(agent_key)
    print(f"Context Retriever ready with {len(tools)} generated tools")


if __name__ == "__main__":
    asyncio.run(main())
