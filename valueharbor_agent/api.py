from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk import Runner
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from valueharbor_agent.agent import build_agent, build_greeting_agent
from valueharbor_agent.config import get_settings
from valueharbor_agent.demo_data import MEMBERS, PRODUCTS, WAREHOUSES
from valueharbor_agent.services import (
    compare_memory_retrieval,
    memory_snippets,
    safe_id,
    services,
)

settings = get_settings()
log = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

APP_NAME = "valueharbor-shopping-agent"
GREETING_APP_NAME = "valueharbor-greeting-agent"
STATIC_DIR = Path(__file__).with_name("static")


def build_memory_service() -> InMemoryMemoryService | VertexAiMemoryBankService:
    if settings.vertex_memory_configured:
        return VertexAiMemoryBankService(
            project=settings.google_cloud_project,
            location=settings.google_memory_location,
            agent_engine_id=settings.google_agent_engine_id,
        )
    return InMemoryMemoryService()


def build_session_service() -> InMemorySessionService | VertexAiSessionService:
    if settings.vertex_memory_configured:
        return VertexAiSessionService(
            project=settings.google_cloud_project,
            location=settings.google_memory_location,
            agent_engine_id=settings.google_agent_engine_id,
        )
    return InMemorySessionService()


session_service = build_session_service()
memory_service = build_memory_service()
runners = {
    model: Runner(
        app_name=APP_NAME,
        agent=build_agent(model),
        session_service=session_service,
        memory_service=memory_service,
        auto_create_session=True,
    )
    for model in settings.available_google_models
}
greeting_runners = {
    model: Runner(
        app_name=GREETING_APP_NAME,
        agent=build_greeting_agent(model),
        session_service=session_service,
        memory_service=memory_service,
        auto_create_session=True,
    )
    for model in settings.available_google_models
}
member_profile_cache: dict[tuple[str, str], str] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for model_runner in runners.values():
        await model_runner.close()
    for model_runner in greeting_runners.values():
        await model_runner.close()


app = FastAPI(
    title="Value Wholesale Shopping Agent",
    version="0.1.0",
    description="Google ADK + Redis IRIS ecommerce demonstration.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    member_id: str = Field(default=settings.valueharbor_demo_member_id, max_length=64)
    session_id: str = Field(default=settings.valueharbor_demo_session_id, max_length=64)
    model: str = Field(default=settings.google_model, max_length=100)

    @field_validator("model")
    @classmethod
    def model_must_be_enabled(cls, model: str) -> str:
        if model not in settings.available_google_models:
            raise ValueError(f"model must be one of: {', '.join(settings.available_google_models)}")
        return model


class GreetingRequest(BaseModel):
    member_id: str = Field(default=settings.valueharbor_demo_member_id, max_length=64)
    session_id: str = Field(default=settings.valueharbor_demo_session_id, max_length=64)
    model: str = Field(default=settings.google_model, max_length=100)

    @field_validator("model")
    @classmethod
    def model_must_be_enabled(cls, model: str) -> str:
        if model not in settings.available_google_models:
            raise ValueError(f"model must be one of: {', '.join(settings.available_google_models)}")
        return model


class MemoryCompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    member_id: str = Field(default=settings.valueharbor_demo_member_id, max_length=64)
    expected_terms: list[str] = Field(default_factory=list, max_length=20)
    runs: int = Field(default=1, ge=1, le=10)


def event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    if content is None:
        return ""
    return "\n".join(
        part.text for part in (content.parts or []) if getattr(part, "text", None)
    ).strip()


def trace_event(
    step_id: str,
    label: str,
    *,
    status: str = "done",
    duration_ms: float | None = None,
    summary: str = "",
    details: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "trace",
        "step": {
            "id": step_id,
            "label": label,
            "status": status,
            "duration_ms": duration_ms,
            "summary": summary,
            "details": details or [],
        },
    }


def _tool_label(name: str, arguments: dict[str, Any]) -> str:
    if name == "search_catalog":
        return "RedisVL Search Catalog"
    if name == "list_context_retriever_tools":
        return "Context Retriever · discover MCP tools"
    if name == "query_context_retriever":
        return f"Context Retriever · {arguments.get('tool_name', 'MCP tool')}"
    return f"Agent tool · {name.replace('_', ' ')}"


def _tool_summary(name: str, response: dict[str, Any]) -> tuple[str, list[str]]:
    payload = response.get("result", response)
    if name == "search_catalog" and isinstance(payload, dict):
        products = payload.get("products", [])
        return f"{len(products)} products found", [
            str(product.get("name", product.get("sku", "product"))) for product in products[:5]
        ]
    if name == "check_warehouse_inventory" and isinstance(payload, dict):
        quantity = payload.get("quantity", 0)
        availability = str(payload.get("availability", "checked")).replace("_", " ")
        return f"{availability} · quantity {quantity}", []
    if name == "list_context_retriever_tools" and isinstance(payload, dict):
        tools = payload.get("tools", [])
        return f"{len(tools)} governed tools available", []
    if name == "query_context_retriever" and isinstance(payload, dict):
        if payload.get("ok") is False:
            return "MCP call failed", [str(payload.get("error", "Unknown MCP error"))[:300]]
        if "quantity" in payload:
            sku = payload.get("sku", payload.get("inventory_id", "inventory"))
            return f"{sku} · quantity {payload['quantity']}", []
        compact = json.dumps(payload, default=str, separators=(",", ":"))
        return "MCP response received", [compact[:300]]
    if name == "get_recent_orders" and isinstance(payload, dict):
        return f"{len(payload.get('orders', []))} recent orders found", []
    if name == "search_member_policies" and isinstance(payload, dict):
        return f"{len(payload.get('policies', []))} policy records found", []
    if name == "remember_shopping_preference":
        saved = (
            bool(payload.get("redis_agent_memory_saved"))
            if isinstance(payload, dict)
            else False
        )
        return "Preference saved" if saved else "Preference write unavailable", []
    if name == "recall_redis_shopping_memory" and isinstance(payload, dict):
        memories = [str(item) for item in payload.get("memories", [])]
        return f"{len(memories)} relevant memories found", memories[:5]
    compact = json.dumps(payload, default=str, separators=(",", ":"))
    return "Completed", [compact[:300]] if compact and compact != "{}" else []


async def member_profile_for_session(member_id: str, session_id: str) -> dict[str, Any]:
    """Load the authoritative member profile without depending on ADK session reads."""
    cache_key = (member_id, session_id)
    if existing := member_profile_cache.get(cache_key):
        return {"context": existing, "source": "application_session_cache"}

    profile = await services.context.get_member_profile(member_id)
    if profile.get("ok") is False:
        return {
            "context": json.dumps({"member_id": member_id}, sort_keys=True),
            "source": "member_id_fallback",
            "error": str(profile.get("error", "profile unavailable")),
        }
    context = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    if len(member_profile_cache) >= 1_000:
        member_profile_cache.pop(next(iter(member_profile_cache)))
    member_profile_cache[cache_key] = context
    return {"context": context, "source": "redis_context_retriever"}


async def _chat_events(request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
    """Run one shopping turn and emit observable steps as newline-delimited events."""
    total_started = time.perf_counter()
    member_id = safe_id(request.member_id, settings.valueharbor_demo_member_id)
    session_id = safe_id(request.session_id, settings.valueharbor_demo_session_id)

    yield {"type": "start", "session_id": session_id}

    async def fetch_short_term() -> list[dict[str, Any]]:
        return await asyncio.to_thread(services.memory.short_term, session_id, 5)

    async def fetch_redis_long_term() -> list[dict[str, Any]]:
        return await asyncio.to_thread(services.memory.recall, member_id, request.message, 5)

    async def fetch_vertex_long_term() -> list[dict[str, Any]]:
        return await services.vertex_memory.recall(member_id, request.message)

    async def fetch_adk_short_term() -> list[dict[str, Any]]:
        try:
            session = await session_service.get_session(
                app_name=APP_NAME,
                user_id=member_id,
                session_id=session_id,
            )
        except Exception as exc:
            log.warning("ADK session telemetry read failed open: %s", exc)
            return []
        if session is None:
            return []
        return [
            {
                "text": event_text(event),
                "author": str(getattr(event, "author", "")),
            }
            for event in (getattr(session, "events", None) or [])[-5:]
        ]

    async def fetch_member_profile() -> dict[str, Any]:
        return await member_profile_for_session(member_id, session_id)

    async def timed(step_id: str, awaitable: Any) -> tuple[str, Any, float]:
        started = time.perf_counter()
        result = await awaitable
        return step_id, result, round((time.perf_counter() - started) * 1000, 2)

    route_started = time.perf_counter()
    routing = await asyncio.to_thread(services.semantic_router.route, request.message)
    route_duration = round((time.perf_counter() - route_started) * 1000, 2)
    cache_read = bool(routing.get("cache_read", routing.get("eligible", False)))
    cache_write = bool(routing.get("cache_write", routing.get("eligible", False)))
    blocked = bool(routing.get("blocked", False))
    route_details = [
        f"Decision source: {routing.get('decision_source', 'unknown')}",
        f"Action: {'block' if blocked else 'allow'}",
        f"LangCache read: {'yes' if cache_read else 'no'}",
        f"LangCache write: {'yes' if cache_write else 'no'}",
        f"Threshold: {routing.get('threshold', 'not set')}",
    ]
    if routing.get("distance") is not None:
        route_details.append(f"Cosine distance: {routing['distance']}")
    if blocked:
        route_summary = "Blocked · outside Value Wholesale ecommerce scope"
    elif cache_read or cache_write:
        route_summary = (
            f"{routing.get('route') or 'cacheable ecommerce'} · LangCache read + write"
        )
    else:
        route_summary = f"{routing.get('route') or 'allowed'} · LangCache bypass"
    yield trace_event(
        "semantic-router",
        "Routing with RedisVL Semantic Router",
        duration_ms=route_duration,
        summary=route_summary,
        details=route_details,
    )

    if blocked:
        answer = (
            "I’m focused on Value Wholesale shopping, products, orders, inventory, "
            "membership, and policies. Ask me something in that area and I’ll help."
        )
        yield trace_event(
            "langcache",
            "Checking Redis LangCache",
            duration_ms=0,
            summary="Bypassed · request blocked",
        )
        yield trace_event(
            "generation",
            "ADK Runner + Gemini",
            duration_ms=0,
            summary="Skipped · blocked by Semantic Router",
        )
        yield {"type": "answer", "answer": answer, "cache_hit": False, "blocked": True}
        yield trace_event(
            "total",
            "Total request",
            duration_ms=round((time.perf_counter() - total_started) * 1000, 2),
            summary="Blocked outside ecommerce scope",
        )
        return

    required_tasks = {
        asyncio.create_task(timed("redis-short-term", fetch_short_term())),
        asyncio.create_task(timed("redis-long-term", fetch_redis_long_term())),
        asyncio.create_task(timed("member-profile", fetch_member_profile())),
    }
    if cache_read:
        required_tasks.add(
            asyncio.create_task(
                timed(
                    "langcache",
                    services.langcache.search(request.message, "reusable-ecommerce"),
                )
            )
        )
    else:
        yield trace_event(
            "langcache",
            "Checking Redis LangCache",
            duration_ms=0,
            summary=f"Bypassed · {routing.get('reason', 'router policy')}",
        )
    adk_telemetry_tasks = {
        asyncio.create_task(timed("adk-short-term", fetch_adk_short_term())),
        asyncio.create_task(timed("vertex-long-term", fetch_vertex_long_term())),
    }
    yield trace_event(
        "adk-short-term",
        "ADK short-term session read",
        status="running",
        summary="Telemetry only · excluded from Gemini context",
    )
    yield trace_event(
        "vertex-long-term",
        "ADK Memory Bank search",
        status="running",
        summary="Telemetry only · excluded from Gemini context",
    )
    results: dict[str, Any] = {}
    for task in asyncio.as_completed(required_tasks):
        step_id, result, duration = await task
        results[step_id] = result
        if step_id == "langcache":
            hit = bool(result and result.get("response"))
            yield trace_event(
                "langcache",
                "Checking Redis LangCache",
                duration_ms=duration,
                summary="Hit" if hit else "Miss",
            )
        elif step_id == "redis-short-term":
            snippets = memory_snippets(result)
            yield trace_event(
                step_id,
                "Getting Redis short-term memory",
                duration_ms=duration,
                summary=f"{len(result)} recent session events",
                details=snippets,
            )
        elif step_id == "redis-long-term":
            snippets = memory_snippets(result)
            yield trace_event(
                step_id,
                "Searching Redis long-term memory",
                duration_ms=duration,
                summary=f"{len(result)} relevant memories found",
                details=snippets,
            )
        elif step_id == "member-profile":
            source = result.get("source", "unavailable")
            source_label = {
                "redis_context_retriever": "Loaded from Redis Context Retriever",
                "application_session_cache": "Reused from application session cache",
                "member_id_fallback": "Profile unavailable; using member ID only",
            }.get(source, source)
            yield trace_event(
                step_id,
                "Hydrating authoritative member profile",
                duration_ms=duration,
                summary=source_label,
                details=[result["context"]],
            )

    async def drain_adk_telemetry() -> AsyncIterator[dict[str, Any]]:
        for task in asyncio.as_completed(adk_telemetry_tasks):
            step_id, result, duration = await task
            snippets = memory_snippets(result)
            if step_id == "adk-short-term":
                yield trace_event(
                    step_id,
                    "ADK short-term session read",
                    duration_ms=duration,
                    summary=f"{len(result)} prior events",
                    details=snippets,
                )
            else:
                yield trace_event(
                    step_id,
                    "ADK Memory Bank search",
                    duration_ms=duration,
                    summary=f"{len(result)} memories",
                    details=snippets,
                )

    short_memories = results.get("redis-short-term", [])
    redis_memories = results.get("redis-long-term", [])
    cached = results.get("langcache")
    member_profile = results.get(
        "member-profile",
        {"context": json.dumps({"member_id": member_id}, sort_keys=True)},
    )

    await asyncio.to_thread(
        services.memory.add_event, member_id, session_id, "USER", request.message
    )

    if cached and cached.get("response"):
        answer = str(cached["response"])
        await asyncio.to_thread(
            services.memory.add_event, member_id, session_id, "ASSISTANT", answer
        )
        yield trace_event(
            "generation",
            "ADK Runner + Gemini",
            duration_ms=0,
            summary="Skipped · response served by LangCache",
        )
        yield {"type": "answer", "answer": answer, "cache_hit": True}
        async for telemetry_event in drain_adk_telemetry():
            yield telemetry_event
        yield trace_event(
            "total",
            "Total request",
            duration_ms=round((time.perf_counter() - total_started) * 1000, 2),
            summary="Completed from semantic cache",
        )
        return

    state_delta = {
        "member_id": member_id,
        "user_id": member_id,
        "member_profile_context": member_profile["context"],
        "redis_short_term_context": "\n".join(memory_snippets(short_memories))
        or "No prior session events.",
        "redis_long_term_context": "\n".join(memory_snippets(redis_memories))
        or "No relevant Redis long-term memories.",
    }
    runner_started = time.perf_counter()
    tool_starts: dict[str, tuple[float, str, dict[str, Any], bool]] = {}
    final_answer = ""
    try:
        async with asyncio.timeout(settings.valueharbor_agent_timeout_seconds):
            async for event in runners[request.model].run_async(
                user_id=member_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=request.message)]),
                state_delta=state_delta,
            ):
                for call in event.get_function_calls():
                    name = str(call.name or "tool")
                    arguments = dict(call.args or {})
                    call_id = str(call.id or name)
                    trace_visible = not (
                        name == "list_context_retriever_tools"
                        and services.context.tools_cached
                    )
                    tool_starts[call_id] = (
                        time.perf_counter(),
                        name,
                        arguments,
                        trace_visible,
                    )
                    if trace_visible:
                        yield trace_event(
                            f"tool-{call_id}",
                            _tool_label(name, arguments),
                            status="running",
                            summary="Calling…",
                        )
                for response in event.get_function_responses():
                    call_id = str(response.id or response.name or "tool")
                    started, name, arguments, trace_visible = tool_starts.pop(
                        call_id,
                        (time.perf_counter(), str(response.name or "tool"), {}, True),
                    )
                    duration = round((time.perf_counter() - started) * 1000, 2)
                    summary, details = _tool_summary(name, dict(response.response or {}))
                    if trace_visible:
                        yield trace_event(
                            f"tool-{call_id}",
                            _tool_label(name, arguments),
                            duration_ms=duration,
                            summary=summary,
                            details=details,
                        )
                if event.is_final_response():
                    final_answer = event_text(event)
    except TimeoutError:
        for task in adk_telemetry_tasks:
            task.cancel()
        elapsed = round((time.perf_counter() - runner_started) * 1000, 2)
        yield trace_event(
            "generation",
            "ADK Runner + Gemini",
            status="error",
            duration_ms=elapsed,
            summary="Timed out; retry the request",
        )
        yield {"type": "error", "message": "The model timed out. Please retry."}
        return
    except Exception as exc:
        for task in adk_telemetry_tasks:
            task.cancel()
        log.exception("ADK request failed")
        yield {"type": "error", "message": f"Agent request failed: {exc}"}
        return

    if not final_answer:
        for task in adk_telemetry_tasks:
            task.cancel()
        yield {"type": "error", "message": "Agent returned no final response"}
        return

    runner_duration = round((time.perf_counter() - runner_started) * 1000, 2)
    yield trace_event(
        "generation",
        f"ADK Runner + Gemini · {request.model}",
        duration_ms=runner_duration,
        summary="Response generated and session queued for Memory Bank",
    )

    await asyncio.to_thread(
        services.memory.add_event, member_id, session_id, "ASSISTANT", final_answer
    )
    if cache_write:
        await services.langcache.store(
            request.message, final_answer, "reusable-ecommerce"
        )

    yield {"type": "answer", "answer": final_answer, "cache_hit": False}
    async for telemetry_event in drain_adk_telemetry():
        yield telemetry_event
    yield trace_event(
        "total",
        "Total request",
        duration_ms=round((time.perf_counter() - total_started) * 1000, 2),
        summary="Completed with generation",
    )


async def _greeting_events(request: GreetingRequest) -> AsyncIterator[dict[str, Any]]:
    """Let a separate ADK agent choose whether retrieval would improve its greeting."""
    member_id = safe_id(request.member_id, settings.valueharbor_demo_member_id)
    session_id = safe_id(f"{request.session_id}-greeting", "greeting-session")
    tool_starts: dict[str, tuple[float, str, dict[str, Any], bool]] = {}
    context_sources: list[str] = []
    context_details: list[str] = []
    final_greeting = ""

    yield {"type": "start", "session_id": session_id}
    yield trace_event(
        "greeting-generation",
        "ADK Greeting",
        status="running",
        summary="Selecting optional context and generating a greeting",
    )
    runner_started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.valueharbor_agent_timeout_seconds):
            async for event in greeting_runners[request.model].run_async(
                user_id=member_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Create my brief welcome greeting. Decide whether personal "
                                "memory or live member context would make it more relevant."
                            )
                        )
                    ],
                ),
                state_delta={"member_id": member_id, "user_id": member_id},
            ):
                for call in event.get_function_calls():
                    name = str(call.name or "tool")
                    arguments = dict(call.args or {})
                    call_id = str(call.id or name)
                    trace_visible = not (
                        name == "list_context_retriever_tools"
                        and services.context.tools_cached
                    )
                    tool_starts[call_id] = (
                        time.perf_counter(),
                        name,
                        arguments,
                        trace_visible,
                    )
                    if trace_visible:
                        yield trace_event(
                            f"greeting-tool-{call_id}",
                            _tool_label(name, arguments),
                            status="running",
                            summary="Greeting agent chose to call this service",
                        )
                for response in event.get_function_responses():
                    call_id = str(response.id or response.name or "tool")
                    call_started, name, arguments, trace_visible = tool_starts.pop(
                        call_id,
                        (time.perf_counter(), str(response.name or "tool"), {}, True),
                    )
                    summary, details = _tool_summary(name, dict(response.response or {}))
                    source = {
                        "recall_redis_shopping_memory": "Redis Agent Memory",
                        "list_context_retriever_tools": "Context Retriever catalog",
                        "query_context_retriever": "Context Retriever",
                    }.get(name)
                    if source and source not in context_sources:
                        context_sources.append(source)
                    if source:
                        context_details.append(f"{source}: {summary}")
                    if trace_visible:
                        yield trace_event(
                            f"greeting-tool-{call_id}",
                            _tool_label(name, arguments),
                            duration_ms=round(
                                (time.perf_counter() - call_started) * 1000, 2
                            ),
                            summary=summary,
                            details=details,
                        )
                if event.is_final_response():
                    final_greeting = event_text(event)
    except TimeoutError:
        yield {
            "type": "error",
            "message": "The greeting timed out. Please select the member again.",
        }
        return
    except Exception as exc:
        log.exception("ADK greeting request failed")
        yield {"type": "error", "message": f"Greeting request failed: {exc}"}
        return

    if not final_greeting:
        yield {"type": "error", "message": "The agent returned no greeting."}
        return

    context_summary = (
        f"Context used: {' + '.join(context_sources)}"
        if context_sources
        else "No optional context tools used"
    )
    yield trace_event(
        "greeting-generation",
        "ADK Greeting",
        duration_ms=round((time.perf_counter() - runner_started) * 1000, 2),
        summary=f"Runner only · excludes page warm-up · {context_summary}",
        details=context_details
        or ["The agent chose not to call Redis Agent Memory or Context Retriever."],
    )
    yield {"type": "greeting", "greeting": final_greeting.strip()}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def warmup_redis_services() -> dict[str, Any]:
    """Prime the five Redis integrations used on the shopping request path."""
    started = time.perf_counter()

    async def probe(name: str, operation: Any) -> tuple[str, dict[str, Any]]:
        probe_started = time.perf_counter()
        try:
            ok, summary, details = await operation()
        except Exception as exc:
            log.warning("Warm-up probe failed for %s: %s", name, exc)
            ok, summary, details = False, f"Unavailable ({type(exc).__name__})", {}
        return name, {
            "ok": ok,
            "duration_ms": round((time.perf_counter() - probe_started) * 1000, 2),
            "summary": summary,
            **details,
        }

    async def database_probe() -> tuple[bool, str, dict[str, Any]]:
        ok = await asyncio.to_thread(services.catalog.ping)
        return ok, "Redis PING succeeded" if ok else "Database is not configured", {}

    async def context_probe() -> tuple[bool, str, dict[str, Any]]:
        tools = await services.context.list_tools(force_refresh=True)
        count = len(tools)
        return bool(count), f"{count} governed tools discovered", {"tools": tools}

    async def router_probe() -> tuple[bool, str, dict[str, Any]]:
        decision = await asyncio.to_thread(
            services.semantic_router.route,
            "What is the electronics return policy?",
        )
        ok = decision.get("decision_source") == "redisvl"
        route = decision.get("route") or "no route"
        return ok, f"Semantic route ready · {route}", {}

    async def langcache_probe() -> tuple[bool, str, dict[str, Any]]:
        ok = await services.langcache.warmup("What is the electronics return policy?")
        return ok, "Semantic lookup completed" if ok else "LangCache is not configured", {}

    async def memory_probe() -> tuple[bool, str, dict[str, Any]]:
        ok = await services.memory.ping()
        return ok, "Health check passed" if ok else "Agent Memory is not configured", {}

    results = await asyncio.gather(
        probe("redis_database", database_probe),
        probe("context_retriever", context_probe),
        probe("semantic_router", router_probe),
        probe("langcache", langcache_probe),
        probe("redis_agent_memory", memory_probe),
    )
    service_results = dict(results)
    return {
        "ok": all(result["ok"] for result in service_results.values()),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "services": service_results,
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": APP_NAME,
        "cloud_run_location": settings.google_cloud_location,
        "memory_bank_location": settings.google_memory_location,
        "default_model": settings.google_model,
        "models": settings.available_google_models,
        "services": {
            "redis_database": settings.redis_configured,
            "context_retriever": bool(settings.mcp_agent_key),
            "semantic_router": services.semantic_router.configured,
            "langcache": settings.langcache_configured,
            "redis_agent_memory": services.memory.client is not None,
            "vertex_adk_memory_bank": services.vertex_memory.client is not None,
            "agent_platform_sessions": isinstance(session_service, VertexAiSessionService),
        },
    }


@app.get("/api/context/tools")
async def context_tools() -> dict[str, Any]:
    started = time.perf_counter()
    tools = await services.context.list_tools()
    return {
        "ok": bool(tools),
        "count": len(tools),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "tools": tools,
    }


@app.post("/api/warmup")
async def warmup() -> dict[str, Any]:
    return await warmup_redis_services()


@app.get("/api/catalog")
async def catalog() -> dict[str, Any]:
    return {"products": PRODUCTS, "warehouses": WAREHOUSES}


@app.get("/api/members")
async def members() -> dict[str, Any]:
    return {
        "members": [
            {
                "member_id": member["member_id"],
                "name": member["name"],
                "tier": member["tier"],
                "home_warehouse": member["home_warehouse"],
            }
            for member in MEMBERS.values()
        ]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    trace = []
    answer = ""
    cache_hit = False
    async for event in _chat_events(request):
        if event["type"] == "trace":
            trace.append(event["step"])
        elif event["type"] == "answer":
            answer = event["answer"]
            cache_hit = bool(event.get("cache_hit"))
        elif event["type"] == "error":
            raise HTTPException(status_code=502, detail=event["message"])
    if not answer:
        raise HTTPException(status_code=502, detail="Agent returned no final response")
    return {
        "answer": answer,
        "session_id": safe_id(request.session_id, settings.valueharbor_demo_session_id),
        "cache": {"hit": cache_hit},
        "trace": trace,
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        async for event in _chat_events(request):
            yield json.dumps(event, default=str, separators=(",", ":")) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/greeting/stream")
async def greeting_stream(request: GreetingRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        async for event in _greeting_events(request):
            yield json.dumps(event, default=str, separators=(",", ":")) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/memory/compare")
async def memory_compare(request: MemoryCompareRequest) -> dict[str, Any]:
    samples = []
    for _ in range(request.runs):
        samples.append(
            await compare_memory_retrieval(
                request.query,
                safe_id(request.member_id, settings.valueharbor_demo_member_id),
                request.expected_terms,
            )
        )

    providers: dict[str, dict[str, Any]] = {}
    for provider_index in range(2):
        provider_samples = [sample["results"][provider_index] for sample in samples]
        latencies = sorted(item["latency_ms"] for item in provider_samples)
        median = round(statistics.median(latencies), 2)
        latest = dict(provider_samples[-1])
        latest["latency_samples_ms"] = latencies
        latest["median_latency_ms"] = median
        providers[latest["provider"]] = latest

    return {
        "query": request.query,
        "member_id": request.member_id,
        "runs": request.runs,
        "methodology": {
            "latency": "Client-observed end-to-end wall time; median shown for repeated runs.",
            "precision_at_k": "Fraction of returned memories containing any expected term.",
            "recall_at_k": "Fraction of expected terms found in at least one returned memory.",
        },
        "providers": providers,
    }
