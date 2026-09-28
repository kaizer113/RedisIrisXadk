from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import setup_context_retriever


def test_semantic_search_flag_validation() -> None:
    assert setup_context_retriever.env_flag({}, "FEATURE") is False
    assert setup_context_retriever.env_flag({"FEATURE": "yes"}, "FEATURE") is True
    assert setup_context_retriever.env_flag({"FEATURE": "OFF"}, "FEATURE") is False
    with pytest.raises(ValueError, match="must be a boolean"):
        setup_context_retriever.env_flag({"FEATURE": "sometimes"}, "FEATURE")


def test_semantic_surface_model_links_product_description() -> None:
    settings = SimpleNamespace(
        effective_context_surface_name="Semantic test",
        experience=SimpleNamespace(brand_name="Value Wholesale"),
    )
    env = {
        "CONTEXT_SEMANTIC_SEARCH_ENABLED": "true",
        "CONTEXT_SEMANTIC_EMBEDDING_MODEL": "text-embedding-005",
        "CONTEXT_SEMANTIC_EMBEDDING_DIMENSIONS": "768",
    }

    with setup_context_retriever.surface_model_args(env, settings) as args:
        assert args[0] == "--datamodel"
        data_model = json.loads(Path(args[1]).read_text(encoding="utf-8"))

    product = next(entity for entity in data_model["entities"] if entity["name"] == "Product")
    embedding = next(
        field for field in product["fields"] if field["name"] == "semantic_embedding"
    )
    assert embedding["redis_indices"] == [
        {
            "type": "vector",
            "vector_dim": 768,
            "distance_metric": "cosine",
            "source_field": "description",
            "embedding_model": "text-embedding-005",
        }
    ]


def test_ctxctl_uses_explicit_self_managed_endpoints(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        config_path = command[command.index("--config") + 1]
        captured["command"] = command
        captured["config"] = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(command, 0, "No context surfaces found\n", "")

    monkeypatch.setenv("CTX_API_URL", "http://context-admin:8080")
    monkeypatch.setenv("CTX_MCP_URL", "http://context-mcp:8081/mcp")
    monkeypatch.setattr(setup_context_retriever.subprocess, "run", fake_run)

    result = setup_context_retriever.ctxctl("surface", "list", admin_key="test-key")

    assert result == []
    assert captured["config"] == {
        "default_profile": "deployment",
        "profiles": {
            "deployment": {
                "api_url": "http://context-admin:8080",
                "mcp_url": "http://context-mcp:8081/mcp",
            }
        },
    }
    assert captured["command"][-4:] == ["surface", "list", "--admin-key", "test-key"]
