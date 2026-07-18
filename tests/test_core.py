from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from google.genai import types
from redis_agent_memory import models
from redisvl.query import TextQuery, VectorQuery

from scripts import seed_managed_memories as managed_seed
from scripts.generate_dataset import records
from valueharbor_agent import api as api_module
from valueharbor_agent.agent import build_agent, build_greeting_agent
from valueharbor_agent.api import (
    _chat_events,
    _context_result_session_event,
    _greeting_events,
    _tool_label,
    _tool_summary,
    app,
    member_profile_cache,
    member_profile_for_session,
    trace_event,
    warmup_redis_services,
)
from valueharbor_agent.config import Settings
from valueharbor_agent.services import (
    ECOMMERCE_REFERENCES,
    ECOMMERCE_ROUTE,
    OUT_OF_DOMAIN_ROUTE,
    PRODUCT_EDUCATION_ROUTE,
    PRODUCT_INDEX_NAME,
    PUBLIC_POLICY_ROUTE,
    REDIS_CONNECTION_KWARGS,
    SHOPPING_GUIDE_ROUTE,
    CatalogService,
    ContextRetrieverService,
    LangCacheService,
    LocalEmbeddingService,
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
    assert settings.valueharbor_embedding_model == "redis/langcache-embed-v3-small"
    assert settings.valueharbor_vector_search_enabled is True
    assert settings.semantic_router_configured is False
    assert not settings.memory_configured
    assert REDIS_CONNECTION_KWARGS["socket_keepalive"] is True
    assert REDIS_CONNECTION_KWARGS["health_check_interval"] == 30


def test_fixture_catalog_search_and_inventory() -> None:
    catalog = CatalogService(
        Settings(_env_file=None, valueharbor_vector_search_enabled=False)
    )
    products = catalog.search_products("fragrance free laundry", limit=3)
    assert products[0]["sku"] == "VH-2002"
    inventory = catalog.check_inventory("VH-2002", "portland")
    assert inventory["availability"] == "out_of_stock"


def test_catalog_product_embedding_text_includes_retrieval_signals() -> None:
    text = CatalogService.product_embedding_text(
        {
            "name": "Clear Tide Laundry Pods",
            "description": "Free-and-clear detergent.",
            "category": "household",
            "tags": ["fragrance-free", "laundry"],
        }
    )
    assert text == (
        "Clear Tide Laundry Pods. Free-and-clear detergent. "
        "Category: household. Keywords: fragrance-free, laundry."
    )


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


def test_catalog_search_uses_redisvl_text_query() -> None:
    captured = {}

    class FakeIndex:
        def query(self, query):
            captured["query"] = query
            return [
                {
                    "id": "valueharbor:product:VH-1001",
                    "sku": "VH-1001",
                    "name": "Olive Oil Twin Pack",
                    "category": "pantry",
                    "price": "24.99",
                    "member_price": "21.99",
                    "description": "Cold-pressed olive oil.",
                }
            ]

    catalog = CatalogService(
        Settings(_env_file=None, valueharbor_vector_search_enabled=False)
    )
    catalog.redis = SimpleNamespace()
    catalog._product_index = FakeIndex()

    products = catalog.search_products("olive oil", category="pantry", limit=3)

    query = captured["query"]
    assert isinstance(query, TextQuery)
    assert "@category:{pantry}" in str(query.filter)
    assert products[0]["member_price"] == 21.99
    assert "id" not in products[0]


def test_catalog_search_uses_shared_local_vectorizer() -> None:
    captured = {}

    class FakeEmbeddings:
        def embed(self, text, *, as_buffer=False):
            assert text == "olive oil"
            assert as_buffer is True
            return b"local-vector"

    class FakeIndex:
        def query(self, query):
            captured["query"] = query
            return [
                {
                    "id": "valueharbor:product:VH-1001",
                    "sku": "VH-1001",
                    "name": "Olive Oil Twin Pack",
                    "category": "pantry",
                    "price": "24.99",
                    "member_price": "21.99",
                    "description": "Cold-pressed olive oil.",
                    "vector_distance": "0.12",
                }
            ]

    catalog = CatalogService(Settings(_env_file=None), FakeEmbeddings())
    catalog.redis = SimpleNamespace()
    catalog._product_index = FakeIndex()

    products = catalog.search_products("olive oil", category="pantry", limit=3)

    query = captured["query"]
    assert isinstance(query, VectorQuery)
    assert query._vector == b"local-vector"
    assert products[0]["score"] == 0.12
    assert PRODUCT_INDEX_NAME == "idx:valueharbor:products-v2"


def test_local_embedding_service_reuses_one_384_dimension_vectorizer() -> None:
    calls = []

    class FakeVectorizer:
        dims = 384

        def embed(self, text, *, as_buffer=False):
            calls.append((text, as_buffer))
            return b"buffer" if as_buffer else [0.0] * 384

        def embed_many(self, texts, *, as_buffer=False):
            return [self.embed(text, as_buffer=as_buffer) for text in texts]

    embeddings = LocalEmbeddingService(Settings(_env_file=None))
    embeddings._vectorizer = FakeVectorizer()

    assert len(embeddings.embed("first")) == 384
    assert embeddings.embed("second", as_buffer=True) == b"buffer"
    assert len(embeddings.embed_many(["third", "fourth"])) == 2
    assert embeddings.loaded is True
    assert calls == [
        ("first", False),
        ("second", True),
        ("third", False),
        ("fourth", False),
    ]


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


def test_managed_memory_seed_batches_at_api_limit(monkeypatch) -> None:
    batch_sizes = []

    class FakeAgentMemory:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def bulk_create_long_term_memories(self, *, memories, **_kwargs):
            batch_sizes.append(len(memories))
            return SimpleNamespace(
                created=[memory["id"] for memory in memories],
                errors=[],
            )

    monkeypatch.setattr(managed_seed, "AgentMemory", FakeAgentMemory)
    settings = Settings(
        _env_file=None,
        agent_memory_base_url="https://memory.example",
        agent_memory_store_id="store-id",
        agent_memory_api_key="api-key",
    )
    memories = [{"id": f"memory-{index}"} for index in range(205)]

    created, errors = managed_seed.seed_redis(settings, memories)

    assert (created, errors) == (205, 0)
    assert batch_sizes == [50, 50, 50, 50, 5]


def test_semantic_router_applies_guardrails_and_positive_route() -> None:
    assert "What pasta products do you sell?" in ECOMMERCE_REFERENCES
    settings = Settings(
        _env_file=None,
        redis_url="redis://configured",
        google_cloud_project="example-project",
    )
    router = SemanticRouterService(settings)
    router.configured = True

    class FakeEmbeddings:
        def __init__(self):
            self.embedded = []

        def embed(self, message):
            self.embedded.append(message)
            return [0.1, 0.2]

    fake_embeddings = FakeEmbeddings()
    router.embeddings = fake_embeddings

    class FakeRouter:
        def __init__(self, name, distance):
            self.name = name
            self.distance = distance

        def __call__(self, *, vector):
            assert vector == [0.1, 0.2]
            return SimpleNamespace(name=self.name, distance=self.distance)

    fake_router = FakeRouter(PUBLIC_POLICY_ROUTE, 0.31)
    router._router = fake_router

    public = router.route("Could you explain the electronics returns rules?")
    assert public["eligible"] is True
    assert public["cache_read"] is True
    assert public["cache_write"] is True
    assert public["blocked"] is False
    assert public["decision_source"] == "redisvl"
    assert public["distance"] == 0.31
    assert public["redisvl_duration_ms"] is not None
    assert public["cache_scope"] == "policy:v1"

    followup = router.route(
        "even without a receipt?",
        "What is the electronics return policy? Returns are accepted under policy.",
    )
    assert followup["blocked"] is False
    assert followup["cache_read"] is False
    assert followup["cache_write"] is False
    assert followup["reason"] == "contextual ecommerce follow-up"
    assert "Previous shopping conversation" in fake_embeddings.embedded[-1]
    assert "even without a receipt?" in fake_embeddings.embedded[-1]

    personalized = router.route("Where is my pickup order?")
    assert personalized["eligible"] is False
    assert personalized["blocked"] is False
    assert personalized["decision_source"] == "guardrail"
    assert personalized["reason"] == "member-specific request"
    assert personalized["redisvl_duration_ms"] is None

    for prompt in (
        "Who am I?",
        "What do you know about me?",
        "Do I have a recent order ready for pickup, and where should I collect it?",
        "Plan the best pantry purchase under $40 using my preferences and explain the trade-off.",
    ):
        decision = router.route(prompt)
        assert decision["action"] == "allow"
        assert decision["blocked"] is False
        assert decision["cache_read"] is False
        assert decision["cache_write"] is False
        assert decision["decision_source"] == "guardrail"
        assert decision["reason"] == "member-specific request"
        assert decision["redisvl_duration_ms"] is None

    live = router.route("Is detergent in stock at the Portland warehouse?")
    assert live["eligible"] is False
    assert live["blocked"] is False
    assert live["reason"] == "live or time-sensitive commerce data"

    fake_router.name = ECOMMERCE_ROUTE
    fake_router.distance = 0.24
    ecommerce = router.route("Find family-size pantry staples under thirty dollars.")
    assert ecommerce["action"] == "allow"
    assert ecommerce["cache_read"] is False
    assert ecommerce["cache_write"] is False
    assert ecommerce["blocked"] is False

    fake_router.name = PRODUCT_EDUCATION_ROUTE
    product_education = router.route("What flavor notes does your medium roast have?")
    assert product_education["cache_read"] is True
    assert product_education["cache_scope"] == "product-education:catalog-v1"

    fake_router.name = SHOPPING_GUIDE_ROUTE
    shopping_guide = router.route("How should I keep bulk oats fresh?")
    assert shopping_guide["cache_write"] is True
    assert shopping_guide["cache_scope"] == "shopping-guide:v1"

    fake_router.name = OUT_OF_DOMAIN_ROUTE
    fake_router.distance = 0.19
    out_of_domain = router.route("Where is Dagestan?")
    assert out_of_domain["action"] == "block"
    assert out_of_domain["blocked"] is True

    fake_router.name = None
    fake_router.distance = None
    no_match = router.route("Discuss an unrelated topic.")
    assert no_match["action"] == "block"
    assert no_match["blocked"] is True


def test_unconfigured_semantic_router_fails_safe() -> None:
    router = SemanticRouterService(Settings(_env_file=None))
    decision = router.route("What is the electronics return policy?")
    assert decision["eligible"] is False
    assert decision["blocked"] is False
    assert decision["action"] == "allow"
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
    assert all(
        body["prompt"].startswith("scope:")
        for _, body in calls
    )
    assert calls[0][1]["prompt"] == "scope:public-policy\nReturn policy?"


async def test_langcache_clear_flushes_configured_cache(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr("valueharbor_agent.services.httpx.AsyncClient", FakeAsyncClient)
    cache = LangCacheService(
        Settings(
            _env_file=None,
            langcache_host="https://langcache.example",
            langcache_cache_id="demo-cache",
            langcache_api_key="test-key",
        )
    )

    assert await cache.clear() is True
    assert calls == [
        (
            "https://langcache.example/v1/caches/demo-cache/flush",
            {"headers": {"Authorization": "Bearer test-key"}},
        )
    ]


def test_langcache_demo_default_accepts_documented_paraphrases() -> None:
    assert Settings(_env_file=None).langcache_similarity_threshold == 0.80


async def test_warmup_pings_six_redis_services(monkeypatch) -> None:
    async def list_tools(*, force_refresh=False):
        assert force_refresh is True
        return [{"name": "get_inventory", "description": "Inventory lookup"}]

    async def warm_langcache(_prompt):
        return True

    async def ping_memory():
        return True

    monkeypatch.setattr(services.catalog, "ping", lambda: True)
    monkeypatch.setattr(
        services.embeddings,
        "warmup",
        lambda: {
            "model": "redis/langcache-embed-v3-small",
            "dimensions": 384,
            "device": "cpu",
            "duration_ms": 5.0,
        },
    )
    monkeypatch.setattr(services.context, "list_tools", list_tools)
    monkeypatch.setattr(
        services.embeddings,
        "cache_probe",
        lambda: (
            True,
            "RedisVL EmbeddingsCache ready",
            {"cache_name": "valueharbor-embeddings-v1"},
        ),
    )
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
        "embedding_cache",
        "langcache",
        "redis_agent_memory",
    }
    assert result["services"]["context_retriever"]["tools"][0]["name"] == "get_inventory"
    assert result["services"]["semantic_router"]["embedding"]["dimensions"] == 384
    assert result["services"]["embedding_cache"]["ok"] is True


async def test_keepalive_returns_compact_warmup_result(monkeypatch) -> None:
    async def fake_warmup():
        return {
            "ok": True,
            "duration_ms": 12.5,
            "services": {"context_retriever": {"tools": [{"name": "tool"}]}},
        }

    monkeypatch.setattr(api_module, "warmup_redis_services", fake_warmup)

    assert await api_module.keepalive() == {"ok": True, "duration_ms": 12.5}


async def test_context_tool_catalog_is_reused_until_forced_refresh(monkeypatch) -> None:
    calls = 0

    class FakeUnifiedClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self, _agent_key):
            nonlocal calls
            calls += 1
            return [{"name": f"tool_version_{calls}"}]

    monkeypatch.setitem(
        sys.modules,
        "context_surfaces",
        SimpleNamespace(UnifiedClient=FakeUnifiedClient),
    )
    context = ContextRetrieverService(Settings(_env_file=None, mcp_agent_key="test"))

    first, first_cached = await context.get_tools()
    second, second_cached = await context.get_tools()
    refreshed, refreshed_cached = await context.get_tools(force_refresh=True)

    assert first == second == [{"name": "tool_version_1"}]
    assert first_cached is False
    assert second_cached is True
    assert refreshed == [{"name": "tool_version_2"}]
    assert refreshed_cached is False
    assert calls == 2


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


async def test_member_profile_reuses_application_session_cache(monkeypatch) -> None:
    profile_context = '{"member_id":"member-1001","name":"Alex Rivera"}'

    async def unexpected_fetch(_member_id):
        raise AssertionError("Context Retriever should not be called for a hydrated session")

    member_profile_cache[("member-1001", "session-1")] = profile_context
    monkeypatch.setattr(services.context, "get_member_profile", unexpected_fetch)

    result = await member_profile_for_session("member-1001", "session-1")

    assert result == {"context": profile_context, "source": "application_session_cache"}
    member_profile_cache.clear()


def test_agent_excludes_adk_memory_but_keeps_redis_memory_context() -> None:
    agent = build_agent("gemini-2.5-flash")
    assert agent.include_contents == "none"
    assert "{redis_short_term_context}" in agent.instruction
    assert "{redis_long_term_context}" in agent.instruction
    assert "vertex_long_term_context" not in agent.instruction


def test_greeting_agent_reuses_profile_and_can_choose_redis_memory() -> None:
    agent = build_greeting_agent("gemini-2.5-flash")
    assert agent.include_contents == "none"
    assert [tool.__name__ for tool in agent.tools] == [
        "recall_redis_shopping_memory",
    ]
    assert "{member_profile_context}" in agent.instruction
    assert "do not retrieve\nthe profile again" in agent.instruction
    assert "at most 18 words" in agent.instruction


def test_shopping_agent_has_cache_safety_instruction() -> None:
    agent = build_agent("gemini-2.5-flash")
    assert "{cache_safety_context}" in agent.instruction
    assert "omit prices, availability" in agent.instruction
    assert "consult the signed-in member's recent order history" in agent.instruction
    assert "reuse it and do not call Context Retriever again" in agent.instruction
    assert "Recommend or name only products returned by search_catalog" in agent.instruction
    assert "never invent an additional product" in agent.instruction


def test_context_order_result_becomes_invisible_session_context() -> None:
    result = _context_result_session_event(
        "query_context_retriever",
        {"tool_name": "get_orders_by_member_id"},
        {"result": {"orders": [{"order_id": "VH-ORD-1048"}]}},
    )

    assert result is not None
    text, metadata = result
    assert "Context Retriever order-history snapshot" in text
    assert "VH-ORD-1048" in text
    assert metadata == {
        "kind": "context_retriever_order_history",
        "tool_name": "get_orders_by_member_id",
        "visibility": "agent_context_only",
    }
    assert (
        _context_result_session_event(
            "query_context_retriever",
            {"tool_name": "get_inventory_by_id"},
            {"result": {"quantity": 31}},
        )
        is None
    )


async def test_greeting_generation_uses_an_isolated_session(monkeypatch) -> None:
    captured = {}

    async def get_profile(member_id):
        assert member_id == "member-1005"
        return {
            "member_id": member_id,
            "name": "Jordan Lee",
            "home_warehouse": "portland",
        }

    class FakeCallEvent:
        content = types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="recall_redis_shopping_memory",
                    args={"query": "shopping preferences"},
                )
            ],
        )

        @staticmethod
        def get_function_calls():
            return [
                SimpleNamespace(
                    name="recall_redis_shopping_memory",
                    args={"query": "shopping preferences"},
                    id="memory-call",
                )
            ]

        @staticmethod
        def get_function_responses():
            return []

        @staticmethod
        def is_final_response():
            return False

    class FakeResponseEvent:
        content = None

        @staticmethod
        def get_function_calls():
            return []

        @staticmethod
        def get_function_responses():
            return [
                SimpleNamespace(
                    name="recall_redis_shopping_memory",
                    response={"memories": ["Prefers decaf coffee."]},
                    id="memory-call",
                )
            ]

        @staticmethod
        def is_final_response():
            return False

    class FakeEvent:
        content = types.Content(role="model", parts=[types.Part(text="Ready for a fresh find?")])

        @staticmethod
        def get_function_calls():
            return []

        @staticmethod
        def get_function_responses():
            return []

        @staticmethod
        def is_final_response():
            return True

    class FakeRunner:
        async def run_async(self, **kwargs):
            captured.update(kwargs)
            yield FakeCallEvent()
            yield FakeResponseEvent()
            yield FakeEvent()

    monkeypatch.setitem(api_module.greeting_runners, "gemini-2.5-flash", FakeRunner())
    monkeypatch.setattr(services.context, "get_member_profile", get_profile)
    member_profile_cache.clear()
    stream = _greeting_events(
        api_module.GreetingRequest(
            member_id="member-1005",
            session_id="shopping-session",
            model="gemini-2.5-flash",
        )
    )
    events = [await anext(stream), await anext(stream)]
    await asyncio.sleep(0.05)
    events.extend([event async for event in stream])

    assert captured["user_id"] == "member-1005"
    assert captured["session_id"] == "shopping-session-greeting"
    profile_context = captured["state_delta"]["member_profile_context"]
    assert json.loads(profile_context)["home_warehouse"] == "portland"
    assert member_profile_cache[("member-1005", "shopping-session")] == profile_context
    tool_events = [
        event
        for event in events
        if event["type"] == "trace"
        and event["step"]["id"] == "greeting-tool-memory-call"
    ]
    assert [event["step"]["status"] for event in tool_events] == ["running", "done"]
    greeting_trace = next(
        event["step"]
        for event in reversed(events)
        if event["type"] == "trace" and event["step"]["id"] == "greeting-generation"
    )
    assert greeting_trace["label"] == "ADK Greeting (2 llm calls)"
    assert greeting_trace["duration_ms"] < 50
    assert "Context used: Redis Agent Memory" in greeting_trace["summary"]
    assert greeting_trace["details"] == ["Redis Agent Memory: 1 relevant memories found"]
    assert events[-1] == {"type": "greeting", "greeting": "Ready for a fresh find?"}
    member_profile_cache.clear()


def test_member_selector_displays_names_and_requests_generated_greeting() -> None:
    html = (api_module.STATIC_DIR / "index.html").read_text()
    assert '>Google ADK × Redis Iris</a>' in html
    assert "RedisIrisXadk/blob/main/ARCHITECTURE.md" in html
    assert "RedisIrisXadk/blob/main/docs/demo.md" in html
    assert 'id="reset-demo"' in html
    assert "fetch('/api/reset-demo',{method:'POST'})" in html
    assert 'target="_blank" rel="noopener noreferrer"' in html
    assert '<details class="panel side service-panel" open>' in html
    assert '<summary><h2>Redis Iris services</h2></summary>' in html
    assert "embedding_cache:'Embedding Cache'" in html
    assert 'class="panel side trace-panel"' in html
    assert "What flavor notes does Rain City Medium Roast Coffee have?" in html
    assert "How should I store a large bag of rolled oats after opening?" in html
    assert "option.textContent=member.name" in html
    assert "${member.name} · ${member.member_id}" not in html
    assert "fetch('/api/greeting/stream'" in html
    assert (
        "await warmupOnLoad();setInterval(keepServicesWarm,KEEPALIVE_INTERVAL_MS);"
        "await selectMember()"
    ) in html
    assert "fetch('/api/keepalive',{method:'POST'})" in html
    assert "KEEPALIVE_INTERVAL_MS=120000" in html
    assert "What can I help you find?" not in html


def test_demo_reset_flushes_langcache(monkeypatch) -> None:
    cleared = []

    async def clear():
        cleared.append(True)
        return True

    monkeypatch.setattr(services.langcache, "clear", clear)
    with TestClient(app) as client:
        response = client.post("/api/reset-demo")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "LangCache flushed"}
    assert cleared == [True]


async def test_adk_memory_telemetry_does_not_block_generation(monkeypatch) -> None:
    captured_state = {}
    redis_recall_args = {}

    class FakeEvent:
        content = types.Content(role="model", parts=[types.Part(text="Generated first")])

        @staticmethod
        def get_function_calls():
            return []

        @staticmethod
        def get_function_responses():
            return []

        @staticmethod
        def is_final_response():
            return True

    class FakeRunner:
        async def run_async(self, **kwargs):
            captured_state.update(kwargs["state_delta"])
            yield FakeEvent()

    class SlowSessionService:
        async def get_session(self, **_kwargs):
            await asyncio.sleep(0.05)
            return None

    async def slow_vertex_recall(_member_id, _query):
        await asyncio.sleep(0.05)
        return [{"text": "ADK-only fact"}]

    async def profile(_member_id, _session_id):
        return {
            "context": '{"name":"Alex Rivera"}',
            "source": "application_session_cache",
        }

    def recall(member_id, query, limit):
        redis_recall_args.update(member_id=member_id, query=query, limit=limit)
        return [{"text": "Redis fact"}]

    monkeypatch.setattr(api_module, "session_service", SlowSessionService())
    monkeypatch.setattr(api_module, "member_profile_for_session", profile)
    monkeypatch.setitem(api_module.runners, "gemini-2.5-flash", FakeRunner())
    monkeypatch.setattr(services.memory, "short_term", lambda *_args: [{"text": "Redis turn"}])
    monkeypatch.setattr(services.memory, "recall", recall)
    monkeypatch.setattr(services.memory, "add_event", lambda *_args: True)
    monkeypatch.setattr(services.vertex_memory, "recall", slow_vertex_recall)
    monkeypatch.setattr(
        services.semantic_router,
        "route",
        lambda _message: {
            "eligible": False,
            "decision_source": "guardrail",
            "threshold": 0.48,
            "route": None,
            "reason": "personalized request",
        },
    )

    events = [
        event
        async for event in _chat_events(
            api_module.ChatRequest(
                message="What do I prefer?",
                member_id="member-1001",
                session_id="nonblocking-test",
                model="gemini-2.5-flash",
            )
        )
    ]

    answer_index = next(index for index, event in enumerate(events) if event["type"] == "answer")
    adk_done_indexes = [
        index
        for index, event in enumerate(events)
        if event["type"] == "trace"
        and event["step"]["id"] in {"adk-short-term", "vertex-long-term"}
        and event["step"]["status"] == "done"
    ]
    assert adk_done_indexes and all(answer_index < index for index in adk_done_indexes)
    assert captured_state["redis_short_term_context"] == "Redis turn"
    assert captured_state["redis_long_term_context"] == "Redis fact"
    assert redis_recall_args == {
        "member_id": "member-1001",
        "query": "What do I prefer?",
        "limit": 4,
    }
    assert "vertex_long_term_context" not in captured_state
    assert not any(
        event["type"] == "trace" and event["step"]["id"] == "member-profile"
        for event in events
    )
    first_trace_ids = []
    for event in events:
        if event["type"] != "trace":
            continue
        step_id = event["step"]["id"]
        if step_id not in first_trace_ids:
            first_trace_ids.append(step_id)
    assert first_trace_ids.index("adk-short-term") == (
        first_trace_ids.index("redis-short-term") + 1
    )
    assert first_trace_ids.index("vertex-long-term") == (
        first_trace_ids.index("redis-long-term") + 1
    )
    total_trace = next(
        event["step"]
        for event in events
        if event["type"] == "trace" and event["step"]["id"] == "total"
    )
    assert total_trace["label"] == "Total request (1 llm call)"


async def test_scoped_langcache_hit_skips_adk_runner(monkeypatch) -> None:
    searched = {}

    class UnexpectedRunner:
        async def run_async(self, **_kwargs):
            raise AssertionError("ADK must not run on a LangCache hit")
            yield

    class EmptySessionService:
        async def get_session(self, **_kwargs):
            return None

    async def cache_search(prompt, scope):
        searched.update(prompt=prompt, scope=scope)
        return {
            "prompt": (
                "scope:product-education:catalog-v1\n"
                "What does Rain City Medium Roast taste like?"
            ),
            "response": "Cocoa and caramel notes.",
        }

    async def empty_vertex_recall(_member_id, _query):
        return []

    async def profile(_member_id, _session_id):
        return {"context": '{"name":"Alex Rivera"}', "source": "test"}

    monkeypatch.setitem(api_module.runners, "gemini-2.5-flash", UnexpectedRunner())
    monkeypatch.setattr(api_module, "session_service", EmptySessionService())
    monkeypatch.setattr(api_module, "member_profile_for_session", profile)
    monkeypatch.setattr(
        services.semantic_router,
        "route",
        lambda _message: {
            "eligible": True,
            "cache_read": True,
            "cache_write": True,
            "cache_scope": "product-education:catalog-v1",
            "blocked": False,
            "decision_source": "redisvl",
            "redisvl_duration_ms": 1.1,
            "threshold": 0.48,
            "route": PRODUCT_EDUCATION_ROUTE,
            "reason": "reusable product education",
        },
    )
    monkeypatch.setattr(services.langcache, "search", cache_search)
    monkeypatch.setattr(services.memory, "short_term", lambda *_args: [])
    monkeypatch.setattr(services.memory, "recall", lambda *_args: [])
    monkeypatch.setattr(services.memory, "add_event", lambda *_args: True)
    monkeypatch.setattr(services.vertex_memory, "recall", empty_vertex_recall)

    events = [
        event
        async for event in _chat_events(
            api_module.ChatRequest(
                message="What flavor notes does the medium roast have?",
                member_id="member-1001",
                session_id="scoped-cache-test",
                model="gemini-2.5-flash",
            )
        )
    ]

    assert searched == {
        "prompt": "What flavor notes does the medium roast have?",
        "scope": "product-education:catalog-v1",
    }
    answer = next(event for event in events if event["type"] == "answer")
    assert answer == {
        "type": "answer",
        "answer": "Cocoa and caramel notes.",
        "cache_hit": True,
    }
    traces = {
        event["step"]["id"]: event["step"]
        for event in events
        if event["type"] == "trace"
    }
    assert traces["langcache"]["summary"] == "Hit · product-education:catalog-v1"
    assert traces["langcache"]["details"] == [
        "Current query: What flavor notes does the medium roast have?",
        "Cached query: What does Rain City Medium Roast taste like?",
    ]
    assert traces["generation"]["summary"] == "Skipped · response served by LangCache"
    assert traces["total"]["label"] == "Total request (0 llm calls)"


async def test_semantic_router_blocks_out_of_domain_before_cache_memory_and_adk(
    monkeypatch,
) -> None:
    recorded_events = []

    class UnexpectedRunner:
        async def run_async(self, **_kwargs):
            raise AssertionError("ADK must not run for a blocked request")
            yield

    async def unexpected_async(*_args, **_kwargs):
        raise AssertionError("downstream retrieval must not run for a blocked request")

    def unexpected_sync(*_args, **_kwargs):
        raise AssertionError("downstream retrieval must not run for a blocked request")

    monkeypatch.setitem(api_module.runners, "gemini-2.5-flash", UnexpectedRunner())
    monkeypatch.setattr(
        services.semantic_router,
        "route",
        lambda _message: {
            "eligible": False,
            "cache_read": False,
            "cache_write": False,
            "blocked": True,
            "action": "block",
            "decision_source": "redisvl",
            "redisvl_duration_ms": 1.23,
            "threshold": 0.48,
            "distance": 0.12,
            "route": OUT_OF_DOMAIN_ROUTE,
            "reason": "outside Value Wholesale ecommerce scope",
        },
    )
    monkeypatch.setattr(services.langcache, "search", unexpected_async)
    monkeypatch.setattr(services.langcache, "store", unexpected_async)
    monkeypatch.setattr(services.memory, "short_term", unexpected_sync)
    monkeypatch.setattr(services.memory, "recall", unexpected_sync)
    monkeypatch.setattr(
        services.memory,
        "add_event",
        lambda *args: recorded_events.append(args) or True,
    )
    monkeypatch.setattr(services.vertex_memory, "recall", unexpected_async)
    monkeypatch.setattr(api_module, "member_profile_for_session", unexpected_async)

    events = [
        event
        async for event in _chat_events(
            api_module.ChatRequest(
                message="Where is Dagestan?",
                member_id="member-1001",
                session_id="blocked-test",
                model="gemini-2.5-flash",
            )
        )
    ]

    answer = next(event for event in events if event["type"] == "answer")
    traces = {
        event["step"]["id"]: event["step"]
        for event in events
        if event["type"] == "trace"
    }
    assert answer["blocked"] is True
    assert "Value Wholesale shopping" in answer["answer"]
    assert traces["semantic-router"]["summary"].startswith("Blocked")
    assert traces["langcache"]["summary"] == "Bypassed · request blocked"
    assert traces["generation"]["summary"] == "Skipped · blocked by Semantic Router"
    assert recorded_events == []


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
    assert _tool_label("search_catalog", {}) == "RedisVL Search Catalog"
    event = trace_event("total", "Total request", duration_ms=1200, summary="Completed")
    assert event["step"]["duration_ms"] == 1200


def test_generated_dataset_has_valid_relationships_and_totals() -> None:
    dataset = records()
    assert {name: len(items) for name, items in dataset.items()} == {
        "products": 10,
        "warehouses": 3,
        "inventory": 30,
        "members": 5,
        "orders": 6,
        "order_items": 12,
        "policies": 3,
        "memory_seeds": 516,
        "memory_evaluations": 9,
    }

    product_ids = {item["sku"] for item in dataset["products"]}
    warehouse_ids = {item["warehouse_id"] for item in dataset["warehouses"]}
    member_ids = {item["member_id"] for item in dataset["members"]}
    order_ids = {item["order_id"] for item in dataset["orders"]}
    memory_by_id = {item["id"]: item for item in dataset["memory_seeds"]}
    large_member_memories = [
        memory
        for memory in dataset["memory_seeds"]
        if memory["owner_id"] == "member-1005"
    ]

    assert len(memory_by_id) == len(dataset["memory_seeds"])
    assert len(large_member_memories) == 500
    assert sum(memory["memory_type"] == "semantic" for memory in large_member_memories) == 20
    assert sum(memory["memory_type"] == "episodic" for memory in large_member_memories) == 480

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
        members = client.get("/api/members")
        assert members.status_code == 200
        assert [member["member_id"] for member in members.json()["members"]] == [
            "member-1001",
            "member-1002",
            "member-1003",
            "member-1004",
            "member-1005",
        ]
        response = client.post(
            "/api/memory/compare",
            json={"query": "pickup preference", "expected_terms": ["Portland"], "runs": 2},
        )
        assert response.status_code == 200
        assert set(response.json()["providers"]) == {
            "redis_agent_memory",
            "vertex_adk_memory_bank",
        }
