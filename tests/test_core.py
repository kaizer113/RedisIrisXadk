from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from redis_agent_memory import models

from scripts.generate_dataset import records
from valueharbor_agent.api import (
    _tool_summary,
    app,
    member_profile_for_session,
    trace_event,
    warmup_redis_services,
)
from valueharbor_agent.config import Settings
from valueharbor_agent.services import (
    PUBLIC_POLICY_ROUTE,
    CatalogService,
    ContextRetrieverService,
    LangCacheService,
    MemoryService,
    SemanticRouterService,
    _retrieval_quality,
    memory_snippets,
    safe_id,
    services,
)


def test_safe_id_and_service_configuration() -> None:
    assert safe_id("member/1001@example.com", "fallback") == "member-1001-example-com"
    settings = Settings(
        _env_file=None,
        redis_url="",
        agent_memory_base_url="",
        agent_memory_store_id="",
        agent_memory_api_key="",
        google_agent_engine_id="",
    )
    assert settings.google_cloud_location == "us-east4"
    assert settings.google_memory_location == "us-east4"
    assert settings.available_google_models == ("gemini-2.5-flash", "gemini-2.5-pro")
    assert not settings.memory_configured


def test_fixture_catalog_search_and_inventory() -> None:
    catalog = CatalogService(Settings(_env_file=None))
    products = catalog.search_products("fragrance free laundry", limit=3)
    assert products[0]["sku"] == "VH-2002"
    inventory = catalog.check_inventory("VH-2002", "portland")
    assert inventory["availability"] == "out_of_stock"


def test_redis_search_response_normalization() -> None:
    redis_8_reply = {
        b"results": [
            {
                b"id": b"valueharbor:product:VH-1001",
                b"extra_attributes": {b"sku": b"VH-1001", b"price": b"21.99"},
            }
        ]
    }
    legacy_reply = [
        1,
        b"valueharbor:product:VH-1001",
        [b"sku", b"VH-1001", b"price", b"21.99"],
    ]
    expected = [{"sku": "VH-1001", "price": "21.99"}]
    assert CatalogService._search_result_maps(redis_8_reply) == expected
    assert CatalogService._search_result_maps(legacy_reply) == expected


def test_retrieval_quality_is_explicit_ground_truth() -> None:
    quality = _retrieval_quality(
        [{"text": "Prefers fragrance-free products and Portland pickup."}],
        ["fragrance-free", "Portland", "vegan"],
    )
    assert quality == {"precision_at_k": 1.0, "recall_at_k": 0.667}


def test_agent_memory_sdk_request_matches_installed_sdk() -> None:
    captured = {}

    class FakeMemoryClient:
        def search_long_term_memory(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(items=[])

    memory = MemoryService(Settings(_env_file=None))
    memory.client = FakeMemoryClient()
    memory.models = models
    assert memory.recall("member-1001", "pickup", 3) == []
    request = captured["request"]
    assert request["filter_"]["owner_id"] == {"eq": "member-1001"}
    assert request["filter_"]["memory_type"] == {"in_": ["semantic", "episodic"]}


def test_semantic_router_applies_guardrails_and_positive_route() -> None:
    settings = Settings(
        _env_file=None,
        redis_url="redis://configured",
        google_cloud_project="example-project",
    )
    router = SemanticRouterService(settings)
    router.configured = True
    router._router = lambda _: SimpleNamespace(name=PUBLIC_POLICY_ROUTE, distance=0.31)

    public = router.route("Could you explain the electronics returns rules?")
    assert public["eligible"] is True
    assert public["decision_source"] == "redisvl"
    assert public["distance"] == 0.31

    personalized = router.route("Where is my pickup order?")
    assert personalized["eligible"] is False
    assert personalized["decision_source"] == "guardrail"
    assert personalized["reason"] == "member-specific request"

    live = router.route("Is detergent in stock at the Portland warehouse?")
    assert live["eligible"] is False
    assert live["reason"] == "live or time-sensitive commerce data"


def test_unconfigured_semantic_router_fails_safe() -> None:
    router = SemanticRouterService(Settings(_env_file=None))
    decision = router.route("What is the electronics return policy?")
    assert decision["eligible"] is False
    assert decision["decision_source"] == "fail-safe"


async def test_langcache_public_cache_does_not_send_undeclared_attributes(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs["json"]))
            body = {"data": [{"response": "cached"}]} if url.endswith("/search") else {}
            return FakeResponse(body)

    monkeypatch.setattr("valueharbor_agent.services.httpx.AsyncClient", FakeAsyncClient)
    cache = LangCacheService(
        Settings(
            _env_file=None,
            langcache_host="https://langcache.example",
            langcache_cache_id="public-policy",
            langcache_api_key="test-key",
        )
    )

    assert await cache.search("Return policy?", "public-policy") == {"response": "cached"}
    assert await cache.warmup("Return policy?") is True
    assert await cache.store("Return policy?", "Thirty days.", "public-policy") is True
    assert all("attributes" not in body for _, body in calls)


async def test_warmup_pings_five_redis_services(monkeypatch) -> None:
    async def list_tools():
        return [{"name": "get_inventory", "description": "Inventory lookup"}]

    async def warm_langcache(_prompt):
        return True

    async def ping_memory():
        return True

    monkeypatch.setattr(services.catalog, "ping", lambda: True)
    monkeypatch.setattr(services.context, "list_tools", list_tools)
    monkeypatch.setattr(
        services.semantic_router,
        "route",
        lambda _message: {"decision_source": "redisvl", "route": PUBLIC_POLICY_ROUTE},
    )
    monkeypatch.setattr(services.langcache, "warmup", warm_langcache)
    monkeypatch.setattr(services.memory, "ping", ping_memory)

    result = await warmup_redis_services()

    assert result["ok"] is True
    assert set(result["services"]) == {
        "redis_database",
        "context_retriever",
        "semantic_router",
        "langcache",
        "redis_agent_memory",
    }
    assert result["services"]["context_retriever"]["tools"][0]["name"] == "get_inventory"


async def test_context_retriever_discovers_member_profile_tool(monkeypatch) -> None:
    context = ContextRetrieverService(Settings(_env_file=None, mcp_agent_key="test"))

    async def list_tools():
        return [
            {
                "name": "get_member_by_id",
                "inputSchema": {"required": ["id"], "properties": {"id": {}}},
            }
        ]

    async def call(name, arguments):
        assert name == "get_member_by_id"
        assert arguments == {"id": "member-1001"}
        return {"member_id": "member-1001", "name": "Alex Rivera"}

    monkeypatch.setattr(context, "list_tools", list_tools)
    monkeypatch.setattr(context, "call", call)
    profile = await context.get_member_profile("member-1001")
    assert profile["name"] == "Alex Rivera"


async def test_member_profile_reuses_shared_adk_session_state(monkeypatch) -> None:
    profile_context = '{"member_id":"member-1001","name":"Alex Rivera"}'

    class FakeSessionService:
        async def get_session(self, **_kwargs):
            return SimpleNamespace(state={"member_profile_context": profile_context})

    async def unexpected_fetch(_member_id):
        raise AssertionError("Context Retriever should not be called for a hydrated session")

    monkeypatch.setattr("valueharbor_agent.api.session_service", FakeSessionService())
    monkeypatch.setattr(services.context, "get_member_profile", unexpected_fetch)

    result = await member_profile_for_session("member-1001", "session-1")

    assert result == {"context": profile_context, "source": "adk_session_state"}


def test_live_trace_formats_memory_and_mcp_results() -> None:
    snippets = memory_snippets(
        [
            {"content": {"parts": [{"text": "Prefers Portland pickup."}]}},
            {"memory": {"fact": "Uses fragrance-free detergent."}},
        ]
    )
    assert snippets == ["Prefers Portland pickup.", "Uses fragrance-free detergent."]
    summary, details = _tool_summary(
        "query_context_retriever",
        {"result": {"sku": "VH-1001", "quantity": 42}},
    )
    assert summary == "VH-1001 · quantity 42"
    assert details == []
    event = trace_event("total", "Total request", duration_ms=1200, summary="Completed")
    assert event["step"]["duration_ms"] == 1200


def test_generated_dataset_has_valid_relationships_and_totals() -> None:
    dataset = records()
    assert {name: len(items) for name, items in dataset.items()} == {
        "products": 10,
        "warehouses": 3,
        "inventory": 30,
        "members": 4,
        "orders": 6,
        "order_items": 12,
        "policies": 3,
        "memory_seeds": 16,
        "memory_evaluations": 7,
    }

    product_ids = {item["sku"] for item in dataset["products"]}
    warehouse_ids = {item["warehouse_id"] for item in dataset["warehouses"]}
    member_ids = {item["member_id"] for item in dataset["members"]}
    order_ids = {item["order_id"] for item in dataset["orders"]}
    memory_by_id = {item["id"]: item for item in dataset["memory_seeds"]}

    assert all(item["sku"] in product_ids for item in dataset["inventory"])
    assert all(item["warehouse_id"] in warehouse_ids for item in dataset["inventory"])
    assert all(order["member_id"] in member_ids for order in dataset["orders"])
    assert all(item["order_id"] in order_ids for item in dataset["order_items"])
    for case in dataset["memory_evaluations"]:
        assert all(
            memory_by_id[memory_id]["owner_id"] == case["member_id"]
            for memory_id in case["relevant_memory_ids"]
        )
        assert all(
            memory_by_id[memory_id]["owner_id"] == case["member_id"]
            for memory_id in case.get("distractor_memory_ids", [])
        )

    items_by_order: dict[str, list[dict]] = {}
    for item in dataset["order_items"]:
        items_by_order.setdefault(item["order_id"], []).append(item)
    for order in dataset["orders"]:
        computed_total = sum(
            item["quantity"] * item["unit_price"] for item in items_by_order[order["order_id"]]
        )
        assert round(computed_total, 2) == order["total"]


def test_health_and_unconfigured_memory_comparison() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["cloud_run_location"] == "us-east4"
        assert health.json()["default_model"] == "gemini-2.5-flash"
        assert health.json()["models"] == ["gemini-2.5-flash", "gemini-2.5-pro"]
        assert "semantic_router" in health.json()["services"]
        response = client.post(
            "/api/memory/compare",
            json={"query": "pickup preference", "expected_terms": ["Portland"], "runs": 2},
        )
        assert response.status_code == 200
        assert set(response.json()["providers"]) == {
            "redis_agent_memory",
            "vertex_adk_memory_bank",
        }
