import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fakeredis.aioredis import FakeRedis

from valuewholesale_onprem import api
from valuewholesale_onprem.config import Settings
from valuewholesale_onprem.identity import Identity


@pytest.mark.parametrize(
    ("exception", "step", "code", "message_fragment"),
    [
        (TimeoutError("private provider detail"), None, "request_timed_out", "time limit"),
        (
            RuntimeError("private provider detail"),
            {"service": "context_gateway", "title": "Catalog lookup", "status": "error"},
            "service_call_failed",
            "Context Gateway",
        ),
        (
            RuntimeError("private provider detail"),
            {"service": "gemini", "title": "Reasoning", "status": "completed"},
            "request_failed",
            "before producing an answer",
        ),
    ],
)
async def test_failed_run_retains_safe_diagnostic_without_claiming_an_outage(
    monkeypatch, caplog, exception, step, code, message_fragment
):
    class FailedTurn:
        def __init__(self, *args, **kwargs):
            self.emit = args[5]

        async def run(self):
            if step:
                await self.emit({"type": "trace", "step": step})
            await self.emit({"type": "lens", "trace_id": "known-trace"})
            raise exception

    monkeypatch.setattr(api, "Turn", FailedTurn)
    redis = FakeRedis(decode_responses=True)
    settings = Settings()
    identity = Identity("member-1001", "verification-session")
    request = api.ChatRequest(message="Find chocolate", run_id=uuid4())
    writer = SimpleNamespace(append=AsyncMock(), finish=AsyncMock())
    await api.produce(
        SimpleNamespace(settings=settings, redis=redis), identity, request, writer, "lock"
    )
    saved = json.loads(
        await redis.get(api.run_key(settings, identity, str(request.run_id), "snapshot"))
    )
    assert saved["status"] == "failed" and saved["error_code"] == code
    assert message_fragment in saved["error"]
    assert saved["trace_id"] == "known-trace"
    emitted = writer.finish.call_args.args[1]
    assert emitted["message"] == saved["error"] and emitted["error_code"] == code
    assert str(request.run_id) in caplog.text and "known-trace" in caplog.text
    assert "private provider detail" not in json.dumps(saved) + caplog.text
    assert "unavailable" not in saved["error"]
    await redis.aclose()
