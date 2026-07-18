from __future__ import annotations

import asyncio
import json
from typing import Any

from google.adk.tools import ToolContext

from valueharbor_agent.services import compare_memory_retrieval, memory_snippets, services


def _member_id(tool_context: ToolContext) -> str:
    return str(
        tool_context.state.get("member_id")
        or tool_context.state.get("user_id")
        or services.settings.valueharbor_demo_member_id
    )


def _session_id(tool_context: ToolContext) -> str:
    return str(tool_context.state.get("session_id") or _member_id(tool_context))


def search_catalog(
    query: str,
    category: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """Find Value Wholesale products with RedisVL by meaning, keywords, and optional category.

    Args:
        query: What the member wants or the need the product should satisfy.
        category: Optional exact category: pantry, household, beverages, electronics, fresh-food.
        limit: Maximum products to return, from 1 to 10.
    """
    return {"products": services.catalog.search_products(query, category, limit)}


def check_warehouse_inventory(sku: str, warehouse_id: str) -> dict[str, Any]:
    """Check current quantity and availability for a SKU at a Value Wholesale warehouse.

    Args:
        sku: Product SKU such as VH-1001.
        warehouse_id: Warehouse identifier: portland, seattle, or sacramento.
    """
    return services.catalog.check_inventory(sku, warehouse_id)


def get_member_profile(tool_context: ToolContext) -> dict[str, Any]:
    """Get the signed-in member's tier, home warehouse, and reward balance."""
    return services.catalog.member_profile(_member_id(tool_context))


def get_recent_orders(tool_context: ToolContext) -> dict[str, Any]:
    """Get recent orders for the signed-in Value Wholesale member."""
    return {"orders": services.catalog.recent_orders(_member_id(tool_context))}


def search_member_policies(query: str) -> dict[str, Any]:
    """Search grounded Value Wholesale policies for returns, pickup, and member pricing.

    Args:
        query: The member's policy question.
    """
    return {"policies": services.catalog.search_policies(query)}


def add_item_to_cart(sku: str, quantity: int, tool_context: ToolContext) -> dict[str, Any]:
    """Add a known product to the signed-in member's cart after they explicitly ask.

    Args:
        sku: Product SKU to add.
        quantity: Number of units, between 1 and 25.
    """
    return services.cart.add(_member_id(tool_context), sku, quantity)


def view_cart(tool_context: ToolContext) -> dict[str, Any]:
    """View the signed-in member's current cart."""
    return {"items": services.cart.get(_member_id(tool_context))}


async def recall_shopping_memory(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Recall relevant member preferences from both Redis Agent Memory and ADK Memory Bank.

    Use this before personalized recommendations and when the member asks what is remembered.

    Args:
        query: The current shopping need or memory question.
    """
    return await compare_memory_retrieval(query, _member_id(tool_context))


async def recall_redis_shopping_memory(
    query: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Recall Redis Agent Memory facts that could make a member greeting relevant.

    Call this only when remembered preferences or shopping activity would improve the greeting.

    Args:
        query: The kind of member preference or shopping context useful for the greeting.
    """
    memories = await asyncio.to_thread(
        services.memory.recall,
        _member_id(tool_context),
        query,
        5,
    )
    return {"memories": memory_snippets(memories)}


async def compare_memory_systems(
    query: str,
    expected_terms_csv: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Benchmark Redis Agent Memory against ADK Memory Bank for one retrieval query.

    Args:
        query: Identical semantic query sent to both memory systems.
        expected_terms_csv: Comma-separated ground-truth terms expected in relevant results.
    """
    expected = [term.strip() for term in expected_terms_csv.split(",") if term.strip()]
    return await compare_memory_retrieval(query, _member_id(tool_context), expected)


async def remember_shopping_preference(
    preference: str,
    topics_csv: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Persist an explicit member shopping preference to Redis Agent Memory.

    The full ADK conversation is also promoted to Vertex Memory Bank after the turn.
    Call this only for an explicit request to save a new preference, never for recall,
    summarization, or a question about existing memories.

    Args:
        preference: Concise preference fact, with no secrets or payment data.
        topics_csv: Comma-separated tags such as dietary,pickup,brand,household.
    """
    topics = [topic.strip() for topic in topics_csv.split(",") if topic.strip()]
    ok = await asyncio.to_thread(
        services.memory.remember,
        _member_id(tool_context),
        preference,
        topics or ["shopping", "preference"],
    )
    return {
        "redis_agent_memory_saved": ok,
        "vertex_memory_bank": "conversation_promotion_queued_after_turn",
        "preference": preference,
    }


async def list_context_retriever_tools() -> dict[str, Any]:
    """List governed live-data tools exposed by Redis Context Retriever."""
    tools, cached = await services.context.get_tools()
    return {"tools": tools, "source": "server_cache" if cached else "context_retriever"}


async def query_context_retriever(
    tool_name: str,
    arguments_json: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Call a governed Redis Context Retriever tool for live commerce data.

    Call list_context_retriever_tools first. Never invent a tool name or argument.

    Args:
        tool_name: Exact Context Retriever tool name.
        arguments_json: JSON object matching that tool's input schema.
    """
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid_arguments_json: {exc}"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "arguments_json_must_be_an_object"}
    return await services.context.call_for_session(
        _session_id(tool_context),
        tool_name,
        arguments,
    )


ALL_TOOLS = [
    search_catalog,
    search_member_policies,
    add_item_to_cart,
    view_cart,
    remember_shopping_preference,
    list_context_retriever_tools,
    query_context_retriever,
]


GREETING_TOOLS = [
    recall_redis_shopping_memory,
]
