from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import setup_context_retriever


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
