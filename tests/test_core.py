from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from redis_agent_memory import models

from scripts.generate_dataset import records
from valueharbor_agent.api import _tool_summary, app, cache_eligible, trace_event
from valueharbor_agent.config import Settings
from valueharbor_agent.services import (
    CatalogService,
    MemoryService,
    _retrieval_quality,
    memory_snippets,
    safe_id,
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


def test_only_public_answers_are_cacheable() -> None:
    assert cache_eligible("What is the electronics return policy?")
    assert not cache_eligible("Where is my pickup order?")
    assert not cache_eligible("Remember my return preference")


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
        "memory_seeds": 8,
        "memory_evaluations": 5,
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
        response = client.post(
            "/api/memory/compare",
            json={"query": "pickup preference", "expected_terms": ["Portland"], "runs": 2},
        )
        assert response.status_code == 200
        assert set(response.json()["providers"]) == {
            "redis_agent_memory",
            "vertex_adk_memory_bank",
        }
