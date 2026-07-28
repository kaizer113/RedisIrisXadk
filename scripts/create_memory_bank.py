from __future__ import annotations

import argparse
import os
from pathlib import Path

import vertexai

from valuewholesale_agent.config import Settings

DISPLAY_NAME = "valuewholesale-memory-bank"


def memory_bank_config() -> dict:
    return {
        "context_spec": {
            "memory_bank_config": {
                "customization_configs": [
                    {
                        "scope_keys": ["app_name", "user_id"],
                        "memory_topics": [
                            {"managed_memory_topic": {"managed_topic_enum": "USER_PERSONAL_INFO"}},
                            {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
                            {
                                "managed_memory_topic": {
                                    "managed_topic_enum": "KEY_CONVERSATION_DETAILS"
                                }
                            },
                            {
                                "managed_memory_topic": {
                                    "managed_topic_enum": "EXPLICIT_INSTRUCTIONS"
                                }
                            },
                        ],
                    }
                ]
            }
        }
    }


def save_env_id(path: Path, memory_bank_id: str) -> None:
    """Update only the non-secret Memory Bank ID while preserving the env file."""
    lines = path.read_text().splitlines() if path.exists() else []
    replacement = f"GOOGLE_AGENT_ENGINE_ID={memory_bank_id}"
    for index, line in enumerate(lines):
        if line.startswith("GOOGLE_AGENT_ENGINE_ID="):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an experience Vertex Memory Bank.")
    parser.add_argument("--project")
    parser.add_argument("--location")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    settings = Settings(_env_file=args.env_file)
    project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or settings.google_cloud_project
    location = (
        args.location or os.getenv("GOOGLE_MEMORY_LOCATION") or settings.google_memory_location
    )
    display_name = settings.effective_memory_bank_display_name
    if not project:
        parser.error("--project or GOOGLE_CLOUD_PROJECT is required")
    if not location:
        parser.error("--location or GOOGLE_MEMORY_LOCATION is required")

    client = vertexai.Client(project=project, location=location)
    memory_bank = next(
        (
            engine
            for engine in client.agent_engines.list()
            if getattr(engine.api_resource, "display_name", "") == display_name
        ),
        None,
    )
    if memory_bank is None:
        memory_bank = client.agent_engines.create(
            config={
                "display_name": display_name,
                "description": (
                    f"ADK memory for the {settings.experience.brand_name} shopping agent."
                ),
                "labels": {"app": settings.redis_namespace},
                **memory_bank_config(),
            }
        )
    else:
        memory_bank = client.agent_engines.update(
            name=memory_bank.api_resource.name,
            config={
                "display_name": display_name,
                "description": (
                    f"ADK memory for the {settings.experience.brand_name} shopping agent."
                ),
                "labels": {"app": settings.redis_namespace},
                **memory_bank_config(),
            },
        )
    resource_name = memory_bank.api_resource.name
    memory_bank_id = resource_name.rsplit("/", 1)[-1]
    save_env_id(args.env_file, memory_bank_id)
    print(resource_name)
    print(f"GOOGLE_AGENT_ENGINE_ID={memory_bank_id}")


if __name__ == "__main__":
    main()
