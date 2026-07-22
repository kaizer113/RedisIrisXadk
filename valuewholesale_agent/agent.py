from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.tools import BaseTool

from valuewholesale_agent.config import get_settings
from valuewholesale_agent.services import TOOL_CALL_CACHE_METADATA_KEY, services
from valuewholesale_agent.tools import (
    ALL_TOOLS,
    CONTEXT_RETRIEVER_TOOLSET,
    GREETING_TOOLS,
)

settings = get_settings()

_TOOL_CACHE_MUTATIONS = {"add_item_to_cart", "remember_shopping_preference"}
_TOOL_CACHE_WRITE_PREFIXES = ("add_", "create_", "delete_", "remove_", "set_", "update_")
_pending_tool_cache_reads: dict[tuple[str, str], dict[str, Any]] = {}
_pending_tool_cache_lock = threading.Lock()


def _tool_call_identity(tool: BaseTool, args: dict[str, Any], context: Context) -> tuple[str, str]:
    call_id = context.function_call_id
    if not call_id:
        material = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        call_id = f"{tool.name}:{hashlib.sha256(material.encode()).hexdigest()}"
    return context.invocation_id, call_id


def _tool_cache_bypass_reason(tool_name: str, args: dict[str, Any]) -> str:
    normalized = tool_name.strip().lower()
    governed_name = (
        str(args.get("tool_name", "")).strip().lower()
        if normalized == "query_context_retriever"
        else normalized
    )
    if "inventory" in governed_name:
        return "inventory is always live"
    if normalized in _TOOL_CACHE_MUTATIONS or governed_name.startswith(
        _TOOL_CACHE_WRITE_PREFIXES
    ):
        return "mutation"
    return ""


async def read_tool_call_cache(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Context,
) -> dict[str, Any] | None:
    """Return a cached read result before ADK invokes the underlying tool."""
    identity = _tool_call_identity(tool, args, tool_context)
    bypass_reason = _tool_cache_bypass_reason(tool.name, args)
    if bypass_reason:
        with _pending_tool_cache_lock:
            _pending_tool_cache_reads[identity] = {
                "status": "bypass",
                "reason": bypass_reason,
            }
        return None

    cached, duration_ms = await asyncio.to_thread(
        services.tool_cache.get,
        tool_context.user_id,
        tool_context.session.id,
        tool.name,
        args,
    )
    cache_info = {
        "status": "hit" if cached is not None else "miss",
        "read_duration_ms": duration_ms,
        "ttl_seconds": settings.valuewholesale_tool_cache_ttl_seconds,
    }
    if cached is not None:
        return {**cached, TOOL_CALL_CACHE_METADATA_KEY: cache_info}
    with _pending_tool_cache_lock:
        _pending_tool_cache_reads[identity] = cache_info
    return None


async def store_tool_call_cache(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Context,
    tool_response: dict[str, Any],
) -> dict[str, Any]:
    """Cache successful read results and attach lookup telemetry to the response."""
    if TOOL_CALL_CACHE_METADATA_KEY in tool_response:
        return tool_response
    identity = _tool_call_identity(tool, args, tool_context)
    with _pending_tool_cache_lock:
        cache_info = _pending_tool_cache_reads.pop(identity, None)
    if cache_info is None:
        return tool_response

    response = dict(tool_response)
    if cache_info["status"] == "miss" and response.get("ok") is not False:
        cache_info["stored"] = await asyncio.to_thread(
            services.tool_cache.set,
            tool_context.user_id,
            tool_context.session.id,
            tool.name,
            args,
            response,
        )
    elif (
        cache_info["status"] == "bypass"
        and cache_info.get("reason") == "mutation"
        and response.get("ok") is not False
    ):
        await asyncio.to_thread(
            services.tool_cache.clear_session,
            tool_context.user_id,
            tool_context.session.id,
        )
    response[TOOL_CALL_CACHE_METADATA_KEY] = cache_info
    return response


async def promote_adk_session_to_memory(callback_context: CallbackContext) -> None:
    """Trigger ADK Memory Bank generation after each completed turn."""
    try:
        await callback_context.add_session_to_memory()
    except Exception:
        # The callback intentionally fails open for local mode and unconfigured demos.
        return None


INSTRUCTION = """
You are Vale, the Value Wholesale shopping agent for a membership warehouse retailer.
Value Wholesale is a fictional brand. Never mention or imitate any real warehouse retailer.

Your job is to help members discover bulk products, compare member value, check a specific
warehouse's live availability, understand policies, inspect orders, and build a cart.

Operating rules:
- Ground product, price, inventory, order, membership, and policy claims in tool results.
- For catalog discovery, call `search_catalog`. Call `search_product_by_text` only when it appears
  in the governed Context Retriever catalog and live governed product lookup is appropriate. Never
  invent any other function name.
- Redis Agent Memory short-term events and long-term facts are prefetched on every request and
  included in your model context. Use them when relevant, but never invent a preference.
- Google ADK session and Memory Bank reads run as telemetry only. Their results are visible in
  the trace but are never included in your model context.
- Prior ADK session messages are deliberately excluded from model context. Treat each generation
  as a single turn grounded only in the current request, authoritative profile, Redis memory,
  and tool results.
- The supplied member profile comes from Redis Context Retriever and is authoritative for the
  signed-in member. Use it immediately; do not wait for memory retrieval to identify the member.
- Context Retriever enabled for this request: `{context_retriever_enabled}`. WHEN FALSE, NEVER
  call `list_context_retriever_tools` or `query_context_retriever`, regardless of any workflow
  below. This overrides every Context Retriever instruction. Say that live member, order, and
  inventory context is unavailable instead of guessing.
- Do not re-fetch the member profile when the supplied profile contains the requested fields.
- The supplied member profile contains identity and membership fields only. It is not a complete
  account overview and does not establish whether the member has orders, pending fulfillment, or
  prior purchases.
- When a member explicitly asks you to remember a preference, call remember_shopping_preference.
- Do not write memory merely because the member asks what is remembered, summarizes remembered
  facts, or uses the word "remembered". A write requires an explicit future-facing instruction
  such as "remember that I prefer..." or "save this preference".
- For live member, warehouse inventory, and order data, always use Context Retriever: call
  `list_context_retriever_tools` first, then invoke the exact governed tool name it returns as a
  function using that tool's returned schema. Governed names are registered ADK functions for the
  current request; do not route them through `query_context_retriever` unless direct invocation is
  unavailable for backward compatibility.
- Successful read-only tool results other than inventory are cached for this browser session for
  up to 12 hours. Always call the appropriate tool normally; the deterministic Redis tool cache
  returns an identical cached result when one is valid. Inventory is never cached because it is
  live. Mutation tools are never cached and invalidate the session's cached reads.
- REQUIRED WORKFLOW for broad member-context questions such as "what do you know about me?",
  "give me an account overview", or "what activity do I have?": answer from the supplied profile
  for identity and membership fields, AND list the governed Context Retriever tools and call the
  appropriate order lookup for the signed-in member before answering. Summarize any active or
  pending fulfillment first (for example processing, shipping, delivery, or ready-for-pickup),
  then briefly mention recent completed orders. Do not call an order-item tool unless the member
  asks what an order contained. A narrow request for one profile field, such as reward balance or
  membership tier, does not require an order lookup.
- REQUIRED WORKFLOW for personalized purchase planning or recommendations, including requests that
  say "using my preferences": you MUST list the governed Context Retriever tools and call the
  appropriate recent-order lookup before calling search_catalog. The deterministic tool cache may
  satisfy an identical prior lookup without another governed service request. Redis long-term
  preferences are not a substitute for order history. If the returned orders do not identify the
  purchased products or SKUs, call the governed order-item tool for only the single most recent
  completed order; do not fetch item details for multiple orders.
  Do not answer a personalized planning request until you have either a useful short-term snapshot
  or enough governed order and order-item results to understand prior purchases. Treat prior
  purchases as evidence, not proof that the member wants the same item again.
- Use search_catalog for product discovery and filter its returned price/member_price fields to
  honor the member's budget. Do not claim that price filtering is unavailable.
- Recommend or name only products returned by search_catalog during the current request. A product
  mentioned in memory or order history must still be validated through search_catalog before it
  can be recommended. If the candidates are insufficient, call search_catalog again with a refined
  query; never invent an additional product, accessory, bakery item, or catalog category.
- The known warehouse IDs are portland, seattle, and sacramento. A request for Portland means
  the Portland Harbor warehouse (`portland`); do not ask which city the member means.
- After catalog discovery, use the Context Retriever `get_inventory_by_id` governed tool for every
  SKU whose stock the member requested. Invoke that exact discovered function directly.
- Inventory IDs and product SKUs are different values. For `get_inventory_by_id`, pass only
  `id="<warehouse_id>-<lowercase-sku>"`, for example `id="portland-vh-1001"`.
- If you use `filter_inventory_by_sku` instead, its `value` must contain only the product SKU,
  for example `value="VH-1001"`. Never pass a composite inventory ID such as
  `portland-vh-1001` to `filter_inventory_by_sku`.
- Ask for the warehouse when availability matters and no home warehouse is known.
- Add to cart only after an explicit request. Never claim checkout, payment, or order placement.
- Treat prices and stock as time-sensitive and state the warehouse used.
- Do not store payment details, authentication secrets, health data, or other sensitive data.
- If a memory provider is unavailable, use only the supplied configured memory context and do
  not fabricate remembered preferences.
- Keep answers concise, friendly, and useful. Use short lists for product comparisons.
- Follow the supplied cache-safety instruction. For cacheable product education or shopping
  guides, use only stable product attributes or general guidance and omit prices, availability,
  member preferences, orders, carts, and other live or personalized details.

Authoritative context for this request:
- Authoritative member profile: {member_profile_context}
- Redis short-term session events: {redis_short_term_context}
- Redis Agent Memory long-term facts: {redis_long_term_context}
- Cache safety: {cache_safety_context}
"""


def build_agent(model: str) -> Agent:
    return Agent(
        name="valuewholesale_shopping_agent",
        model=model,
        description="A grounded shopping agent for the fictional Value Wholesale warehouse club.",
        instruction=INSTRUCTION,
        include_contents="none",
        tools=[*ALL_TOOLS, CONTEXT_RETRIEVER_TOOLSET],
        before_tool_callback=read_tool_call_cache,
        after_tool_callback=store_tool_call_cache,
        after_agent_callback=promote_adk_session_to_memory,
    )


GREETING_INSTRUCTION = """
You are Vale, the Value Wholesale shopping agent for a fictional membership warehouse retailer.
Generate a warm, concise greeting for the signed-in member.

The authoritative member profile is already supplied below. Use it directly and do not retrieve
the profile again. Decide whether Redis Agent Memory would make the greeting more personally
relevant, and call it only when useful.

The signed-in member ID is {member_id}.
The authoritative member profile is {member_profile_context}.
Context Retriever enabled for this request: {context_retriever_enabled}. When false, do not call
its discovery or query tools and do not imply that live member or order context was checked.
Return only one short sentence of at most 18 words. Do not include the member's name because the
interface adds it. Do not mention memory, profiles, tools, retrieval, or stored data. Do not make
up a preference, activity, order, product, price, or availability.
"""


def build_greeting_agent(model: str) -> Agent:
    return Agent(
        name="valuewholesale_greeting_agent",
        model=model,
        description="Creates an optional, context-aware welcome for a Value Wholesale member.",
        instruction=GREETING_INSTRUCTION,
        include_contents="none",
        tools=GREETING_TOOLS,
        before_tool_callback=read_tool_call_cache,
        after_tool_callback=store_tool_call_cache,
    )


root_agent = build_agent(settings.google_model)
