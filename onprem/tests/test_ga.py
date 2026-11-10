import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.responses import JSONResponse
from google.genai import types
from pydantic import SecretStr
from redis_agent_memory import models
from redis_iris.agent_playbooks.values import DiscoveredEntry, Discovery, Faq
from redis_iris.context_gateway.values import GatewayTool, Handle, ToolResult
from redis_iris.intent_router.values import Match, Matches, Target

from valuewholesale_onprem.api import create_app, produce, run_key
from valuewholesale_onprem.config import Settings
from valuewholesale_onprem.engine import Turn
from valuewholesale_onprem.identity import Identity, issue, verify
from valuewholesale_onprem.services import (
    Memory,
    Services,
    member_arguments,
    member_bound,
    published_direct_answer,
    read_tool,
)


@pytest.fixture
def settings():
    return Settings(
        redis_url="redis://localhost:6379",
        onprem_session_secret="x" * 40,
        iris_agent_memory_url="http://memory",
        iris_agent_memory_store_id="ga-store",
        iris_agent_memory_api_key="scoped",
        iris_agent_memory_namespace_id="ga-namespace",
        iris_agent_playbooks_url="http://playbook",
        iris_agent_playbooks_playbook_id="approved",
        iris_agent_playbooks_api_key="read-only",
        iris_intent_router_url="http://router",
        iris_intent_router_id="shopping",
        iris_intent_router_api_key="read-only",
        iris_context_gateway_url="http://gateway/mcp",
        iris_context_gateway_api_key="use-only",
        google_cloud_project="demo",
    )


async def test_ga_memory_serialization_preserves_both_scope_filters(settings):
    client = SimpleNamespace(
        search_long_term_memory_async=AsyncMock(return_value=SimpleNamespace(items=[])),
        bulk_create_long_term_memories_async=AsyncMock(),
    )
    memory = Memory(settings, client)
    identity = Identity("member-1001", "session-a")
    await memory.recall(identity, "olive oil")
    request = client.search_long_term_memory_async.call_args.kwargs["request"]
    # Inspect the actual generated GA model's wire output, not a mocked input dictionary.
    wire = json.loads(request.model_dump_json(by_alias=True, exclude_none=True))
    assert wire["query"] == "olive oil"
    assert wire["filter"] == {
        "op": "ALL",
        "ownerId": {"eq": identity.owner},
        "namespaceId": {"eq": "ga-namespace"},
    }
    await memory.remember(identity, "Prefers olive oil")
    record = client.bulk_create_long_term_memories_async.call_args.kwargs["memories"][0]
    wire = json.loads(record.model_dump_json(by_alias=True, exclude_none=True))
    assert wire["memoryId"]
    assert wire["namespaceId"] == "ga-namespace"
    assert wire["ownerId"] == identity.owner
    assert "id" not in wire and "namespace" not in wire


async def test_memory_recall_normalizes_actual_sdk_model_and_verifies_scope(settings):
    identity = Identity("member-1001", "session-a")
    now = datetime.now(UTC)
    item = models.MemoryRecord(
        memory_id="actual-id",
        text="Dark chocolate preference",
        owner_id=identity.owner,
        namespace_id="ga-namespace",
        created_at=now,
        updated_at=now,
    )
    client = SimpleNamespace(
        search_long_term_memory_async=AsyncMock(return_value=SimpleNamespace(items=[item]))
    )
    memory = Memory(settings, client)
    assert (await memory.recall(identity))[0]["memory_id"] == "actual-id"
    item.owner_id = "another-member"
    with pytest.raises(PermissionError):
        await memory.recall(identity)


async def test_session_transcript_checks_owner_and_namespace(settings):
    identity = Identity("member-1001", "session-a")
    client = SimpleNamespace(
        get_session_memory_async=AsyncMock(
            return_value=SimpleNamespace(
                owner_id="another-member",
                namespace_id="ga-namespace",
                events=[],
                summary=None,
            )
        )
    )
    with pytest.raises(PermissionError):
        await Memory(settings, client).transcript(identity)
    client.get_session_memory_async.return_value.owner_id = identity.owner
    client.get_session_memory_async.return_value.namespace_id = "other-application"
    with pytest.raises(PermissionError):
        await Memory(settings, client).transcript(identity)


def test_session_cookie_cannot_be_relabelled_or_moved_between_apps():
    identity, cookie = issue("member-1001", "x" * 40)
    assert verify(cookie, "x" * 40) == identity
    with pytest.raises(ValueError):
        verify(cookie + "0", "x" * 40)
    with pytest.raises(ValueError):
        verify(cookie, "y" * 40)


def test_explicit_non_readonly_annotation_overrides_a_read_like_tool_name():
    tool = GatewayTool(
        "shop__get_or_create_order",
        "Create if absent",
        {"type": "object"},
        original_name="get_or_create_order",
        annotations={"readOnlyHint": False},
    )
    assert read_tool(tool) is False
    read = GatewayTool(
        "shop__get_product",
        "Read product",
        {"type": "object"},
        original_name="get_product",
        annotations={"readOnlyHint": True},
    )
    assert read_tool(read) is True


async def test_surface_order_filter_forces_member_and_excludes_foreign_rows(settings):
    # This is the schema actually advertised by the GA Context Surface.
    tool = GatewayTool(
        "shopping__filter_order",
        "Filter orders",
        {
            "type": "object",
            "properties": {
                "tag_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"enum": ["member_id", "status"]},
                            "value": {"type": "string"},
                            "exclude": {"type": "boolean"},
                        },
                    },
                }
            },
        },
        original_name="filter_order",
        annotations={"readOnlyHint": True},
    )
    identity = Identity("member-1001", "session-a")
    args = member_arguments(
        tool,
        {
            "tag_conditions": [
                {"field": "member_id", "value": "member-1002", "exclude": True},
                {"field": "status", "value": "delivered"},
            ]
        },
        identity,
    )
    assert args["tag_conditions"] == [
        {"field": "status", "value": "delivered"},
        {"field": "member_id", "value": "member-1001"},
    ]
    assert member_bound(tool)
    services = Services(settings, None)
    services.gateway = SimpleNamespace(
        acall=AsyncMock(
            side_effect=[
                ToolResult(
                    "",
                    {
                        "count": 1,
                        "total_count": 1,
                        "results": [
                            {"order_id": "own", "member_id": "member-1001"},
                        ],
                    },
                ),
                ToolResult(
                    "",
                    {
                        "count": 1,
                        "results": [
                            {"order_id": "foreign", "member_id": "member-1002"},
                        ],
                    },
                ),
            ]
        )
    )
    assert (await services.gateway_call(tool, {}, identity))["ok"]
    assert not (await services.gateway_call(tool, {}, identity))["ok"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"set": {"entity": "Member"}, "relationship": "orders"},
        {"sets": [{"entity": "Product"}, {"entity": "Order"}]},
        {"set": {"entity": "Product"}, "relationship": "order_items"},
        {"sets": [{"entity": "Product", "expand_to": "order_items"}]},
    ],
)
async def test_generic_surface_sets_reject_accounts_before_call(settings, arguments):
    tool = GatewayTool(
        "shopping__expand_results",
        "Set operations",
        {},
        original_name="expand_results",
        annotations={"readOnlyHint": True},
    )
    services = Services(settings, None)
    services.gateway = SimpleNamespace(acall=AsyncMock())
    with pytest.raises(PermissionError):
        await services.gateway_call(tool, arguments, Identity("member-1001", "session-a"))
    services.gateway.acall.assert_not_awaited()


def test_generic_surface_sets_allow_only_public_commerce_joins():
    tool = GatewayTool(
        "shopping__expand_results", "Set operations", {}, original_name="expand_results"
    )
    args = {"set": {"entity": "Product", "text_query": "chocolate"}, "relationship": "inventory"}
    assert member_arguments(tool, args, Identity("member-1001", "session-a")) == args


def test_namespaced_generic_tool_without_original_name_still_checks_entity():
    tool = GatewayTool("shopping__expand_results", "Set operations", {})
    with pytest.raises(PermissionError):
        member_arguments(tool, {"set": {"entity": "Member"}}, Identity("member-1001", "s"))


async def test_gateway_core_helpers_cannot_bypass_current_turn_handle_boundary(settings):
    tool = GatewayTool("page_results", "Page a handle", {}, annotations={"readOnlyHint": True})
    assert not read_tool(tool)
    services = Services(settings, None)
    services.gateway = SimpleNamespace(acall=AsyncMock())
    result = await services.gateway_call(tool, {"handle": "foreign"}, Identity("member-1001", "s"))
    assert not result["ok"]
    services.gateway.acall.assert_not_awaited()


def faq():
    return DiscoveredEntry(
        "entry-1",
        "faq",
        3,
        0.98,
        entry_key="return-policy",
        faq=Faq("What is the returns policy?", "The published demo policy."),
    )


def fake_services(settings, redis=None, discovery=None, target="published_guidance"):
    return SimpleNamespace(
        settings=settings,
        redis=redis,
        router=SimpleNamespace(
            amatch=AsyncMock(
                return_value=Matches((Match("policy", 0.99, target=Target("handler", target)),))
            )
        ),
        playbooks=SimpleNamespace(adiscover=AsyncMock(return_value=discovery or Discovery())),
        memory=SimpleNamespace(
            transcript=AsyncMock(return_value=[]),
            append=AsyncMock(),
            recall=AsyncMock(return_value=[]),
        ),
        gateway=None,
        model=None,
        lens_enabled=False,
        open=AsyncMock(),
        close=AsyncMock(),
    )


async def test_published_faq_bypasses_generation_and_keeps_version_provenance(settings):
    services = fake_services(settings, discovery=Discovery(discovered=(faq(),)))
    events = []

    async def emit(event):
        events.append(event)

    turn = Turn(
        services,
        Identity("member-1001", "session-a"),
        str(uuid4()),
        "Explain the returns policy",
        True,
        emit,
    )
    turn.generate = AsyncMock(side_effect=AssertionError("Published FAQ must bypass the model"))
    await turn.run()
    answer = next(event for event in events if event["type"] == "answer")
    assert answer["answer"] == faq().faq.answer and answer["model_calls"] == 0
    assert answer["published"]["version"] == 3
    assert services.memory.append.await_count == 2


async def test_uncovered_or_current_data_request_does_not_become_a_published_answer(settings):
    match = Match("policy", 0.99, target=Target("handler", "published_guidance"))
    assert (
        published_direct_answer("What is in stock now?", match, Discovery(discovered=(faq(),)))
        is None
    )
    services = fake_services(settings)
    events = []

    async def emit(event):
        events.append(event)

    turn = Turn(
        services,
        Identity("member-1001", "session-a"),
        str(uuid4()),
        "Explain a new shopping topic",
        True,
        emit,
    )
    turn.generate = AsyncMock(return_value="A newly generated answer")
    await turn.run()
    turn.generate.assert_awaited_once()
    # There is no runtime content-writing or publish operation on the playbook client.
    assert vars(services.playbooks).keys() == {"adiscover"}
    assert next(event for event in events if event["type"] == "answer")["published"] is None


async def test_gateway_inventory_calls_are_live_and_member_argument_is_bound(settings):
    tool = GatewayTool(
        "shop__get_member_inventory",
        "Live member inventory",
        {
            "type": "object",
            "properties": {"member_id": {"type": "string"}},
            "required": ["member_id"],
        },
        original_name="get_member_inventory",
        upstream_id="shop",
    )
    gateway = SimpleNamespace(
        acall=AsyncMock(
            side_effect=[
                ToolResult("", {"quantity": 2}),
                ToolResult("", {"quantity": 1}),
            ]
        )
    )
    services = Services.__new__(Services)
    services.gateway = gateway
    identity = Identity("member-1001", "session-a")
    a = await services.gateway_call(tool, {"member_id": "member-1002"}, identity)
    b = await services.gateway_call(tool, {"member_id": "member-1002"}, identity)
    assert (a["data"]["quantity"], b["data"]["quantity"]) == (2, 1)
    assert gateway.acall.await_count == 2
    assert gateway.acall.call_args.args[1]["member_id"] == "member-1001"


async def test_generic_order_lookup_fails_closed_for_foreign_or_ownerless_records(settings):
    tool = GatewayTool(
        "shop__get_order_by_id",
        "Read order",
        {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        original_name="get_order_by_id",
        upstream_id="shop",
    )
    services = Services.__new__(Services)
    services.gateway = SimpleNamespace(
        acall=AsyncMock(
            side_effect=[
                ToolResult(
                    "foreign detail must not enter prompt",
                    {"order_id": "other", "member_id": "member-1002"},
                ),
                ToolResult("ownerless detail", {"order_id": "unknown", "total": 50}),
                ToolResult(
                    "text may disagree",
                    {
                        "orders": [
                            {"order_id": "own", "member_id": "member-1001"},
                            {"order_id": "other", "member_id": "member-1002"},
                        ]
                    },
                ),
                ToolResult(
                    "unbounded aggregate",
                    {"total": 100, "orders": [{"order_id": "own", "member_id": "member-1001"}]},
                ),
                ToolResult(
                    "untrusted parallel text", {"order_id": "own", "member_id": "member-1001"}
                ),
            ]
        )
    )
    identity = Identity("member-1001", "session-a")
    for _ in range(4):
        result = await services.gateway_call(tool, {"id": "arbitrary"}, identity)
        assert not result["ok"] and result["data"] is None and result["text"] == ""
    own = await services.gateway_call(tool, {"id": "own"}, identity)
    assert own["ok"] and own["data"]["member_id"] == "member-1001" and own["text"] == ""


async def test_unbounded_order_handle_is_rejected_before_gateway_execution(settings):
    services = fake_services(settings)
    services.gateway = SimpleNamespace(acall_as_handle=AsyncMock())
    turn = Turn(
        services,
        Identity("member-1001", "session-a"),
        str(uuid4()),
        "show orders",
        True,
        AsyncMock(),
    )
    tool = GatewayTool(
        "shop__query_orders",
        "Read orders",
        {
            "type": "object",
            "properties": {"filter": {"type": "string"}},
        },
        original_name="query_orders",
        upstream_id="shop",
    )
    turn.tools[tool.name] = tool
    result = await turn.call(
        "business_result_handle", {"name": tool.name, "arguments": {"filter": "all"}}
    )
    assert not result["ok"]
    services.gateway.acall_as_handle.assert_not_awaited()


async def test_gemini_loop_uses_discovered_gateway_schema_and_real_response_types(settings):
    services = fake_services(settings, target="shopping")
    definition = GatewayTool(
        "catalog__get_inventory",
        "Live inventory",
        {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
        original_name="get_inventory",
        upstream_id="catalog",
    )
    services.tools = AsyncMock(return_value=[definition])
    services.gateway = object()
    services.gateway_call = AsyncMock(
        return_value={"ok": True, "data": {"sku": "VH-1001", "quantity": 2}}
    )

    def response(part):
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=12, candidates_token_count=7
            ),
        )

    generate = AsyncMock(
        side_effect=[
            response(
                types.Part.from_function_call(
                    name="discover_business_tools", args={"query": "live inventory"}
                )
            ),
            response(
                types.Part.from_function_call(
                    name="call_business_tool",
                    args={"name": definition.name, "arguments": {"sku": "VH-1001"}},
                )
            ),
            response(types.Part.from_text(text="The live tool reports two units.")),
        ]
    )
    services.model = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    events = []

    async def emit(event):
        events.append(event)

    turn = Turn(
        services,
        Identity("member-1001", "session-a"),
        str(uuid4()),
        "Check current olive oil stock",
        True,
        emit,
    )
    await turn.run()
    assert turn.model_calls == 3
    services.gateway_call.assert_awaited_once()
    assert services.gateway_call.call_args.args[0].name == definition.name
    assert next(event for event in events if event["type"] == "answer")["model_calls"] == 3
    assert "VH-1001" in turn.skus
    assert all(
        event["step"]["duration_ms"] >= 0
        for event in events
        if event["type"] == "trace" and "duration_ms" in event["step"]
    )


async def test_gateway_handle_is_bound_to_this_turn_and_uses_discovered_schema(settings):
    services = fake_services(settings)
    schema = {
        "type": "object",
        "properties": {
            "handle": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "maximum": 100},
        },
        "required": ["handle", "offset", "limit"],
    }
    services.gateway = SimpleNamespace(
        atools=AsyncMock(return_value=[GatewayTool("page_results", "Page", schema)]),
        apage=AsyncMock(return_value=ToolResult("", [{"sku": "VH-1001"}])),
        aquery=AsyncMock(),
        acount=AsyncMock(),
        aaggregate=AsyncMock(),
        adescribe=AsyncMock(),
        arelease=AsyncMock(),
        acall_as_handle=AsyncMock(return_value=Handle("known-handle")),
    )
    turn = Turn(
        services, Identity("member-1001", "session-a"), str(uuid4()), "compare", True, AsyncMock()
    )
    denied = await turn.call(
        "handle_operation",
        {"handle": "foreign-handle", "operation": "page", "arguments": {"offset": 0, "limit": 5}},
    )
    assert denied["ok"] is False
    services.gateway.apage.assert_not_awaited()
    turn.handles.add("known-handle")
    result = await turn.call(
        "handle_operation",
        {"handle": "known-handle", "operation": "page", "arguments": {"offset": 0, "limit": 5}},
    )
    assert result["ok"] and "VH-1001" in turn.skus


class FakeStreams:
    """The documented backend boundary; native replay is a separate live acceptance check."""

    def __init__(self):
        self.producers = {}
        self.last_read = None

    async def start(self, fn, *, scope, stream_id):
        key = (scope, stream_id)
        created = key not in self.producers
        if created:
            self.producers[key] = fn
        return SimpleNamespace(created=created)

    async def sse(self, stream_id, *, request, scope, after):
        self.last_read = (scope, stream_id, after)
        return JSONResponse({"cursor": after})


async def test_run_retries_replay_cursor_and_member_session_boundary(settings):
    redis = FakeRedis(decode_responses=True)
    streams = FakeStreams()
    services = fake_services(settings, redis)
    application = create_app(
        settings, redis_client=redis, services_factory=lambda *_: services, stream_backend=streams
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://demo"
        ) as browser:
            assert (await browser.get("/api/session")).status_code == 401
            session = (await browser.post("/api/session", json={"member_id": "member-1001"})).json()
            run_id = str(uuid4())
            body = {"run_id": run_id, "message": "Find olive oil"}
            first = await browser.post("/api/chat", json=body)
            retry = await browser.post("/api/chat", json=body)
            assert first.status_code == retry.status_code == 200
            assert first.json()["created"] and not retry.json()["created"]
            assert len(streams.producers) == 1
            changed = await browser.post("/api/chat", json={**body, "message": "Change my order"})
            assert changed.status_code == 409
            replay = await browser.get(
                first.json()["events_url"], headers={"Last-Event-ID": "123-4"}
            )
            assert replay.json()["cursor"] == "123-4"
            assert streams.last_read[0].endswith(session["session_id"])
            await browser.post("/api/session", json={"member_id": "member-1002"})
            assert (await browser.get(first.json()["events_url"])).status_code == 404
            assert (await browser.get(f"/api/runs/{run_id}")).status_code == 404
            assert len(streams.producers) == 1
    await redis.aclose()


async def test_retained_snapshot_prevents_restarting_a_completed_run(settings):
    redis = FakeRedis(decode_responses=True)
    streams = FakeStreams()
    services = fake_services(settings, redis)
    application = create_app(
        settings, redis_client=redis, services_factory=lambda *_: services, stream_backend=streams
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://demo"
        ) as browser:
            selected = (
                await browser.post("/api/session", json={"member_id": "member-1001"})
            ).json()
            identity = Identity(selected["member_id"], selected["session_id"])
            run_id = str(uuid4())
            body = {"run_id": run_id, "message": "Find olive oil"}
            await browser.post("/api/chat", json=body)
            await redis.set(
                run_key(settings, identity, run_id, "snapshot"),
                json.dumps({"status": "completed", "answer": "Saved result"}),
            )
            streams.producers.clear()  # Simulate stream retention ending on another worker.
            retry = await browser.post("/api/chat", json=body)
            assert retry.json()["created"] is False and not streams.producers
            assert (await browser.get(f"/api/runs/{run_id}")).json()["answer"] == "Saved result"
    await redis.aclose()


async def test_producer_persists_result_independently_of_sse_subscriber(settings):
    redis = FakeRedis(decode_responses=True)
    services = fake_services(settings, redis, Discovery(discovered=(faq(),)))
    identity, run_id = Identity("member-1001", "session-a"), str(uuid4())
    from valuewholesale_onprem.api import ChatRequest

    writer = SimpleNamespace(append=AsyncMock(), finish=AsyncMock())
    await produce(
        services, identity, ChatRequest(run_id=run_id, message="Explain returns"), writer, "lock"
    )
    snapshot = json.loads(await redis.get(run_key(settings, identity, run_id, "snapshot")))
    assert snapshot["status"] == "completed" and snapshot["answer"] == faq().faq.answer
    assert snapshot["published"] == {
        "entry_id": "entry-1",
        "key": "return-policy",
        "version": 3,
        "score": 0.98,
    }
    assert snapshot["model_calls"] == 0
    writer.finish.assert_awaited_once_with("completed", {"type": "completed"})
    # No request object or connected subscriber is needed by the actual producer.
    await redis.aclose()


async def test_unconfigured_app_reports_degraded_instead_of_mock_success():
    settings = Settings(onprem_session_secret=SecretStr("x" * 40))
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://demo"
        ) as browser:
            response = (await browser.get("/api/health")).json()
            assert response["ready"] is False
            assert not any(response["services"].values())
            assert response["checks"]["gemini"]["detail"] == "Vertex client is not configured"
            assert response["checks"]["lens"]["detail"] == "Lens SDK is not initialized"
