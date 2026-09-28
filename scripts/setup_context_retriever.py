"""Create or update the existing shopping Context Surface using ctxctl."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from context_surfaces import UnifiedClient
from context_surfaces.constants import DEFAULT_API_URL, DEFAULT_MCP_URL
from dotenv import dotenv_values

from scripts.generate_dataset import records_for_experience
from valuewholesale_agent.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "valuewholesale_agent" / "context_models.py"
SEMANTIC_SEARCH_FLAG = "CONTEXT_SEMANTIC_SEARCH_ENABLED"
DEFAULT_SEMANTIC_EMBEDDING_MODEL = "text-embedding-005"
DEFAULT_SEMANTIC_EMBEDDING_DIMENSIONS = 768


def env_flag(env: dict[str, str], name: str, *, default: bool = False) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def semantic_embedding_config(env: dict[str, str]) -> tuple[str, int]:
    model = env.get("CONTEXT_SEMANTIC_EMBEDDING_MODEL", "").strip()
    model = model or DEFAULT_SEMANTIC_EMBEDDING_MODEL
    raw_dimensions = env.get("CONTEXT_SEMANTIC_EMBEDDING_DIMENSIONS", "").strip()
    dimensions = int(raw_dimensions or DEFAULT_SEMANTIC_EMBEDDING_DIMENSIONS)
    if dimensions <= 0:
        raise ValueError("CONTEXT_SEMANTIC_EMBEDDING_DIMENSIONS must be positive")
    return model, dimensions


@contextmanager
def surface_model_args(
    env: dict[str, str],
    settings: Settings,
) -> Iterator[list[str]]:
    if not env_flag(env, SEMANTIC_SEARCH_FLAG):
        yield ["--models", str(MODELS_PATH)]
        return

    from context_surfaces.context_model import export_data_model

    from valuewholesale_agent.context_models import (
        Inventory,
        Member,
        Order,
        OrderItem,
        Product,
        Warehouse,
    )

    model, dimensions = semantic_embedding_config(env)
    description = (
        f"Governed live ecommerce context for the "
        f"{settings.experience.brand_name} ADK shopping agent."
    )
    data_model = export_data_model(
        title=settings.effective_context_surface_name,
        description=description,
        entities=[Product, Warehouse, Inventory, Member, Order, OrderItem],
    )
    product = next(entity for entity in data_model["entities"] if entity["name"] == "Product")
    embedding = next(
        field for field in product["fields"] if field["name"] == "semantic_embedding"
    )
    embedding["redis_indices"] = [
        {
            "type": "vector",
            "vector_dim": dimensions,
            "distance_metric": "cosine",
            "source_field": "description",
            "embedding_model": model,
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as data_model_file:
        json.dump(data_model, data_model_file)
        data_model_file.flush()
        yield ["--datamodel", data_model_file.name]


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

    api_url = os.getenv("CTX_API_URL", "").strip()
    mcp_url = os.getenv("CTX_MCP_URL", "").strip()
    if api_url or mcp_url:
        config = {
            "default_profile": "deployment",
            "profiles": {
                "deployment": {
                    "api_url": api_url or DEFAULT_API_URL,
                    "mcp_url": mcp_url or DEFAULT_MCP_URL,
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            command[3:3] = ["--config", config_file.name]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    else:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ctxctl command failed")
    output = result.stdout.strip()
    if not output:
        return None
    if output == "No context surfaces found":
        return []
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ctxctl returned unexpected output: {output}") from exc


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

    with surface_model_args(env, settings) as model_args:
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
                *model_args,
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
                *model_args,
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
    env: dict[str, str],
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
    if env_flag(env, SEMANTIC_SEARCH_FLAG):
        from google import genai
        from google.genai import types

        project = env.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = env.get("GOOGLE_CLOUD_LOCATION", "").strip() or "global"
        if not project:
            raise SystemExit(
                "GOOGLE_CLOUD_PROJECT is required when Context Retriever semantic search is enabled"
            )
        model, dimensions = semantic_embedding_config(env)
        products = datasets["products"]
        client = genai.Client(
            enterprise=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=60000),
        )
        result = client.models.embed_content(
            model=model,
            contents=[str(product["description"]) for product in products],
            config=types.EmbedContentConfig(
                output_dimensionality=dimensions,
                task_type="SEMANTIC_SIMILARITY",
            ),
        )
        vectors = result.embeddings or []
        if len(vectors) != len(products):
            raise RuntimeError("Vertex AI returned an unexpected number of product embeddings")
        for product, embedding in zip(products, vectors, strict=True):
            values = list(embedding.values or [])
            if len(values) != dimensions:
                raise RuntimeError("Vertex AI returned an embedding with unexpected dimensions")
            product["semantic_embedding"] = values
        print(f"Product: generated={len(products)} semantic embeddings")
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
    # Explicit process-level overrides are useful when a private deployment is reached
    # through a temporary tunnel while the saved runtime MCP URL stays VM-private.
    for key, value in env.items():
        os.environ.setdefault(key, value)
    get_settings.cache_clear()
    settings = Settings(_env_file=env_path)
    surface_id, agent_key = ensure_surface(
        env,
        settings,
        env_path,
        force_agent_key=args.rotate_agent_key,
    )
    await import_records(surface_id, env["CTX_ADMIN_KEY"], settings, env)
    tools = await UnifiedClient().list_tools(agent_key)
    print(f"Context Retriever ready with {len(tools)} generated tools")


if __name__ == "__main__":
    asyncio.run(main())
