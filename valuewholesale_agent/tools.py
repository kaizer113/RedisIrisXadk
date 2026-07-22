from __future__ import annotations

import asyncio
import json
from typing import Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types

from valuewholesale_agent.services import (
    call_with_timing,
    compare_memory_retrieval,
    memory_snippets,
    services,
)


def _context_retriever_enabled(tool_context: ToolContext) -> bool:
    state = getattr(tool_context, "state", {})
    value = state.get("context_retriever_enabled", True)
    return value is True or str(value).lower() == "true"


def _member_id(tool_context: ToolContext) -> str:
    return str(
        tool_context.state.get("member_id")
        or tool_context.state.get("user_id")
        or services.settings.valuewholesale_demo_member_id
    )


def search_catalog(
    query: str,
    category: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """Find Value Wholesale products with RedisVL by meaning, keywords, and optional category.

    Args:
        query: What the member wants or the need the product should satisfy.
        category: Optional exact category: pantry, household, beverages, electronics, fresh-food.
        limit: Maximum products to return, from 1 to 6.
    """
    normalized_limit = max(1, min(limit, 6))
    (
        products,
        redisvl_duration_ms,
        embedding_duration_ms,
        embedding_cache_hit,
    ) = services.catalog.search_products_with_timing(query, category, normalized_limit)
    return {
        "products": products,
        "redisvl_duration_ms": redisvl_duration_ms,
        "embedding_duration_ms": embedding_duration_ms,
        "embedding_cache_hit": embedding_cache_hit,
    }


def search_product_by_text(
    query: str,
    category: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """Compatibility alias for catalog search; prefer search_catalog.

    Args:
        query: What the member wants or the need the product should satisfy.
        category: Optional exact category: pantry, household, beverages, electronics, fresh-food.
        limit: Maximum products to return, from 1 to 6.
    """
    return search_catalog(query, category, limit)


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
    """Search grounded Value Wholesale policies with RedisVL vector retrieval.

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
    memories, duration_ms = await asyncio.to_thread(
        call_with_timing,
        services.memory.recall,
        _member_id(tool_context),
        query,
        5,
    )
    return {
        "memories": memory_snippets(memories),
        "operation_duration_ms": duration_ms,
    }


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
    ok, duration_ms = await asyncio.to_thread(
        call_with_timing,
        services.memory.remember,
        _member_id(tool_context),
        preference,
        topics or ["shopping", "preference"],
    )
    return {
        "redis_agent_memory_saved": ok,
        "vertex_memory_bank": "conversation_promotion_queued_after_turn",
        "preference": preference,
        "operation_duration_ms": duration_ms,
    }


async def list_context_retriever_tools(tool_context: ToolContext) -> dict[str, Any]:
    """List governed live-data tools exposed by Redis Context Retriever."""
    if not _context_retriever_enabled(tool_context):
        return {"ok": False, "error": "context_retriever_disabled", "tools": []}
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
    if not _context_retriever_enabled(tool_context):
        return {"ok": False, "error": "context_retriever_disabled"}
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid_arguments_json: {exc}"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "arguments_json_must_be_an_object"}
    return await services.context.call(tool_name, arguments)


class ContextRetrieverTool(BaseTool):
    """An ADK-facing governed tool discovered from Context Retriever."""

    def __init__(self, definition: dict[str, Any]) -> None:
        self.definition = definition
        super().__init__(
            name=str(definition["name"]),
            description=str(
                definition.get("description")
                or "Query governed live Value Wholesale context."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        schema = self.definition.get("inputSchema")
        if not isinstance(schema, dict):
            schema = self.definition.get("input_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=schema,
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        if not _context_retriever_enabled(tool_context):
            return {"ok": False, "error": "context_retriever_disabled"}
        return await services.context.call(self.name, args)


class ContextRetrieverToolset(BaseToolset):
    """Expose the live governed catalog as callable ADK tools."""

    def __init__(self, reserved_names: set[str]) -> None:
        super().__init__()
        self.reserved_names = reserved_names

    async def get_tools(self, readonly_context: Any | None = None) -> list[BaseTool]:
        state = getattr(readonly_context, "state", {}) if readonly_context else {}
        enabled = state.get("context_retriever_enabled", True)
        if enabled is not True and str(enabled).lower() != "true":
            return []
        definitions = await services.context.list_tools()
        return [
            ContextRetrieverTool(definition)
            for definition in definitions
            if definition.get("name")
            and str(definition["name"]) not in self.reserved_names
        ]


ALL_TOOLS = [
    search_catalog,
    search_member_policies,
    add_item_to_cart,
    view_cart,
    remember_shopping_preference,
    list_context_retriever_tools,
    query_context_retriever,
]

CONTEXT_RETRIEVER_TOOLSET = ContextRetrieverToolset(
    {tool.__name__ for tool in ALL_TOOLS}
)


GREETING_TOOLS = [
    recall_redis_shopping_memory,
]
