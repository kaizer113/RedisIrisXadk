from __future__ import annotations

import argparse
from pathlib import Path

import vertexai

DISPLAY_NAME = "valueharbor-memory-bank"


def memory_bank_config() -> dict:
    return {
        "context_spec": {
            "memory_bank_config": {
                "customization_configs": [
                    {
                        "scope_keys": ["app_name", "user_id"],
                        "memory_topics": [
                            {
                                "managed_memory_topic": {
                                    "managed_topic_enum": "USER_PREFERENCES"
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
    parser = argparse.ArgumentParser(description="Create the ValueHarbor Vertex Memory Bank.")
    parser.add_argument("--project", default="central-beach-194106")
    parser.add_argument("--location", default="us-east4")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    client = vertexai.Client(project=args.project, location=args.location)
    memory_bank = next(
        (
            engine
            for engine in client.agent_engines.list()
            if getattr(engine.api_resource, "display_name", "") == DISPLAY_NAME
        ),
        None,
    )
    if memory_bank is None:
        memory_bank = client.agent_engines.create(
            config={
                "display_name": DISPLAY_NAME,
                "description": "ADK memory for the ValueHarbor shopping agent.",
                "labels": {"owner": "lionel_giavelli", "app": "valueharbor"},
                **memory_bank_config(),
            }
        )
    else:
        memory_bank = client.agent_engines.update(
            name=memory_bank.api_resource.name,
            config={
                "display_name": DISPLAY_NAME,
                "description": "ADK memory for the ValueHarbor shopping agent.",
                "labels": {"owner": "lionel_giavelli", "app": "valueharbor"},
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
