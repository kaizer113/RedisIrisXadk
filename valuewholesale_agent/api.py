from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk import Runner
from google.adk.events import Event
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from valuewholesale_agent.agent import build_agent, build_greeting_agent
from valuewholesale_agent.config import get_settings
from valuewholesale_agent.demo_data import MEMBERS, PRODUCTS, WAREHOUSES
from valuewholesale_agent.services import (
    TOOL_CALL_CACHE_METADATA_KEY,
    call_with_timing,
    compare_memory_retrieval,
    memory_snippets,
    safe_id,
    services,
)
from valuewholesale_agent.tools import is_context_retriever_tool

settings = get_settings()
log = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

APP_NAME = "valuewholesale-shopping-agent"
GREETING_APP_NAME = "valuewholesale-greeting-agent"
TRANSCRIPT_APP_NAME = "valuewholesale-working-memory"
SHORT_TERM_MEMORY_LIMIT = 10
STATIC_DIR = Path(__file__).with_name("static")
MEMORY_SEEDS_PATH = Path(__file__).parent.parent / "data" / "generated" / "memory_seeds.jsonl"
MEMORY_RESETTABLE_MEMBERS = frozenset(
    {"member-1001", "member-1002", "member-1003", "member-1004"}
)


class DemoVertexMemoryBankService(VertexAiMemoryBankService):
    """Tag generated demo memories and keep them separate from seeded facts."""

    async def add_session_to_memory(self, session: Any) -> None:
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            events=session.events,
            custom_metadata={
                "metadata": {"valuewholesale_origin": "demo-created"},
            },
        )


def build_memory_service() -> InMemoryMemoryService | VertexAiMemoryBankService:
    if settings.vertex_memory_configured:
        return DemoVertexMemoryBankService(
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


class LatencyRegistry:
    """Keep post-cold-call service timings for the lifetime of this worker."""

    def __init__(self) -> None:
        self._cold_calls_seen: set[str] = set()
        self._samples: dict[str, list[float]] = defaultdict(list)

    def mark_cold_call_complete(self, service: str) -> None:
        self._cold_calls_seen.add(service)

    def record(self, service: str, duration_ms: float | None) -> None:
        if duration_ms is None or duration_ms < 0:
            return
        if service not in self._cold_calls_seen:
            self._cold_calls_seen.add(service)
            return
        self._samples[service].append(float(duration_ms))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            service: {
                "count": len(values),
                "avg_ms": round(statistics.fmean(values), 2),
                "p50_ms": round(self._percentile(values, 0.50), 2),
                "p95_ms": round(self._percentile(values, 0.95), 2),
                "p99_ms": round(self._percentile(values, 0.99), 2),
            }
            for service, values in self._samples.items()
            if values
        }


latency_registry = LatencyRegistry()
working_memory_tasks: set[asyncio.Task[tuple[bool, bool]]] = set()
working_memory_tails: dict[tuple[str, str], asyncio.Task[tuple[bool, bool]]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.valuewholesale_warmup_on_startup:
        try:
            result = await warmup_redis_services()
            log.info(
                "Worker warm-up completed in %s ms (ok=%s)",
                result["duration_ms"],
                result["ok"],
            )
        except Exception:
            log.exception("Worker warm-up failed; starting worker without primed services")
    yield
    await drain_working_memory_tasks()
    for model_runner in runners.values():
        await model_runner.close()
    for model_runner in greeting_runners.values():
        await model_runner.close()
    await asyncio.gather(
        services.langcache.close(),
        services.context.close(),
        services.memory.close(),
    )


app = FastAPI(
    title="Value Wholesale Shopping Agent",
    version="0.1.0",
    description="Google ADK + Redis IRIS ecommerce demonstration.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    member_id: str = Field(default=settings.valuewholesale_demo_member_id, max_length=64)
    session_id: str = Field(default=settings.valuewholesale_demo_session_id, max_length=64)
    model: str = Field(default=settings.google_model, max_length=100)
    context_retriever_enabled: bool = False

    @field_validator("model")
    @classmethod
    def model_must_be_enabled(cls, model: str) -> str:
        if model not in settings.available_google_models:
            raise ValueError(f"model must be one of: {', '.join(settings.available_google_models)}")
        return model


class GreetingRequest(BaseModel):
    member_id: str = Field(default=settings.valuewholesale_demo_member_id, max_length=64)
    session_id: str = Field(default=settings.valuewholesale_demo_session_id, max_length=64)
    model: str = Field(default=settings.google_model, max_length=100)
    context_retriever_enabled: bool = False

    @field_validator("model")
    @classmethod
    def model_must_be_enabled(cls, model: str) -> str:
        if model not in settings.available_google_models:
            raise ValueError(f"model must be one of: {', '.join(settings.available_google_models)}")
        return model


class MemoryCompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    member_id: str = Field(default=settings.valuewholesale_demo_member_id, max_length=64)
    expected_terms: list[str] = Field(default_factory=list, max_length=20)
    runs: int = Field(default=1, ge=1, le=10)


class MemoryResetRequest(BaseModel):
    member_id: str = Field(default=settings.valuewholesale_demo_member_id, max_length=64)


def event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    if content is None:
        return ""
    return "\n".join(
        part.text for part in (content.parts or []) if getattr(part, "text", None)
    ).strip()


def recent_adk_transcript_events(
    session: Any, limit: int = SHORT_TERM_MEMORY_LIMIT
) -> list[dict[str, str]]:
    """Return the latest displayable prompt/answer events from the ADK transcript."""
    events = []
    for event in getattr(session, "events", None) or []:
        text = event_text(event)
        if text:
            events.append(
                {
                    "text": text,
                    "author": str(getattr(event, "author", "")),
                }
            )
    return events[-max(1, limit) :]


def adk_transcript_session_id(session_id: str) -> str:
    """Keep the canonical transcript separate from the Runner's native session."""
    return f"{session_id}-transcript"


async def append_adk_transcript_event(
    member_id: str,
    session_id: str,
    role: str,
    text: str,
) -> bool:
    """Append one canonical prompt or answer to the dedicated ADK transcript."""
    try:
        session = await session_service.get_session(
            app_name=TRANSCRIPT_APP_NAME,
            user_id=member_id,
            session_id=adk_transcript_session_id(session_id),
        )
        if session is None:
            try:
                session = await session_service.create_session(
                    app_name=TRANSCRIPT_APP_NAME,
                    user_id=member_id,
                    session_id=adk_transcript_session_id(session_id),
                )
            except Exception:
                session = await session_service.get_session(
                    app_name=TRANSCRIPT_APP_NAME,
                    user_id=member_id,
                    session_id=adk_transcript_session_id(session_id),
                )
        if session is None:
            return False
        normalized_role = role.upper()
        await session_service.append_event(
            session,
            Event(
                invocation_id=f"transcript-{time.time_ns()}",
                author=member_id if normalized_role == "USER" else "valuewholesale-agent",
                content=types.Content(
                    role="user" if normalized_role == "USER" else "model",
                    parts=[types.Part(text=text)],
                ),
                custom_metadata={"kind": "working_memory_transcript"},
            ),
        )
        return True
    except Exception as exc:
        log.warning("ADK transcript write failed open: %s", exc)
        return False


async def append_working_memory_event(
    member_id: str,
    session_id: str,
    role: str,
    text: str,
) -> tuple[bool, bool]:
    """Dual-write identical prompt/answer text to Redis and ADK working memory."""
    redis_result, adk_result = await asyncio.gather(
        asyncio.to_thread(services.memory.add_event, member_id, session_id, role, text),
        append_adk_transcript_event(member_id, session_id, role, text),
    )
    return bool(redis_result), bool(adk_result)


def queue_working_memory_event(
    member_id: str,
    session_id: str,
    role: str,
    text: str,
) -> None:
    """Persist one canonical event in the background, ordered within its session."""
    key = (member_id, session_id)
    previous = working_memory_tails.get(key)

    async def persist_after_previous() -> tuple[bool, bool]:
        if previous is not None:
            try:
                await previous
            except Exception:
                # The done callback logs the original failure; later events must still persist.
                pass
        return await append_working_memory_event(member_id, session_id, role, text)

    task = asyncio.create_task(
        persist_after_previous(),
        name=f"working-memory-{member_id}-{session_id}-{role.lower()}",
    )
    working_memory_tasks.add(task)
    working_memory_tails[key] = task

    def complete(completed: asyncio.Task[tuple[bool, bool]]) -> None:
        working_memory_tasks.discard(completed)
        if working_memory_tails.get(key) is completed:
            working_memory_tails.pop(key, None)
        if completed.cancelled():
            log.warning("Working-memory background write was cancelled")
            return
        if exc := completed.exception():
            log.error(
                "Working-memory background write failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return
        redis_ok, adk_ok = completed.result()
        if not redis_ok or not adk_ok:
            log.warning(
                "Working-memory background write incomplete (redis=%s, adk=%s)",
                redis_ok,
                adk_ok,
            )

    task.add_done_callback(complete)


async def drain_working_memory_tasks() -> None:
    """Wait for queued transcript writes before closing shared service clients."""
    while working_memory_tasks:
        await asyncio.gather(*tuple(working_memory_tasks), return_exceptions=True)


def is_llm_response_event(event: Any) -> bool:
    """Return whether an ADK event represents one completed logical LLM call."""
    content = getattr(event, "content", None)
    return bool(
        not getattr(event, "partial", False)
        and (
            getattr(event, "usage_metadata", None) is not None
            or (content is not None and getattr(content, "role", None) == "model")
        )
    )


def llm_count_label(label: str, count: int) -> str:
    unit = "llm call" if count == 1 else "llm calls"
    return f"{label} ({count} {unit})"


def gemini_runner_label(model: str) -> str:
    model_family = "Pro" if "pro" in model.lower() else "Flash"
    return f"ADK Runner + Gemini {model_family}"


def trace_event(
    step_id: str,
    label: str,
    *,
    status: str = "done",
    duration_ms: float | None = None,
    summary: str = "",
    details: list[str] | None = None,
    cache: dict[str, Any] | None = None,
    move_to_end: bool = False,
) -> dict[str, Any]:
    if status == "done":
        if cache and cache.get("read_duration_ms") is not None:
            latency_registry.record("tool_call_cache", float(cache["read_duration_ms"]))
        if duration_ms is not None:
            normalized_id = step_id.lower()
            normalized_label = label.lower()
            normalized_summary = summary.lower()
            service: str | None = None
            if normalized_id == "semantic-router" and "bypassed" not in normalized_summary:
                service = "semantic_router"
            elif normalized_id == "langcache" and "bypassed" not in normalized_summary:
                service = "langcache"
            elif "redis-short-term" in normalized_id:
                service = "redis_agent_memory_short_term"
            elif (
                "redis-long-term" in normalized_id
                or "redis long-term memory" in normalized_label
            ):
                service = "redis_agent_memory_long_term"
            elif "adk-short-term" in normalized_id:
                service = "agent_platform_sessions"
            elif "vertex-long-term" in normalized_id:
                service = "vertex_adk_memory_bank"
            elif "context retriever" in normalized_label:
                service = "context_retriever"
            elif normalized_label.startswith("redisvl search"):
                service = "redis_database"
            elif (
                normalized_id in {"generation", "greeting-generation"}
                and "(0 llm calls)" not in normalized_label
            ):
                service = "gemini_adk_orchestration"
            if service:
                latency_registry.record(service, duration_ms)
            for detail in details or []:
                if detail.startswith("Local embedding:"):
                    try:
                        embedding_ms = float(
                            detail.split(":", 1)[1].split("ms", 1)[0].strip()
                        )
                    except ValueError:
                        continue
                    latency_registry.record("embedding_cache", embedding_ms)
    event = {
        "type": "trace",
        "step": {
            "id": step_id,
            "label": label,
            "status": status,
            "duration_ms": duration_ms,
            "summary": summary,
            "details": details or [],
            "cache": cache,
        },
    }
    if move_to_end:
        event["step"]["move_to_end"] = True
    return event


async def timed_thread_call(
    step_id: str,
    operation: Any,
    *args: Any,
) -> tuple[str, Any, float]:
    """Run and time a synchronous operation within its executor thread."""
    result, duration_ms = await asyncio.to_thread(call_with_timing, operation, *args)
    return step_id, result, duration_ms


def _tool_label(name: str, arguments: dict[str, Any]) -> str:
    if name == "recall_redis_shopping_memory":
        return "Searching Redis long-term memory"
    if name == "search_catalog":
        query = str(arguments.get("query", "")).strip()
        category = str(arguments.get("category", "")).strip() or "all categories"
        limit = arguments.get("limit", 5)
        compact_query = f'"{query[:72]}{"…" if len(query) > 72 else ""}"'
        return f"RedisVL Search Catalog · {compact_query} · {category} · limit {limit}"
    if name == "search_member_policies":
        query = str(arguments.get("query", "")).strip()
        compact_query = f'"{query[:72]}{"…" if len(query) > 72 else ""}"'
        return f"RedisVL Search Policies · {compact_query}"
    if name == "list_context_retriever_tools":
        return "Context Retriever · discover MCP tools"
    if name == "query_context_retriever":
        return f"Context Retriever · {arguments.get('tool_name', 'MCP tool')}"
    if is_context_retriever_tool(name):
        return f"Context Retriever · {name}"
    return f"Agent tool · {name.replace('_', ' ')}"


def _tool_summary(
    name: str,
    response: dict[str, Any],
    *,
    include_timing_details: bool = True,
) -> tuple[str, list[str]]:
    payload = response.get("result", response)
    if name == "search_catalog" and isinstance(payload, dict):
        products = payload.get("products", [])
        summary = f"{len(products)} products found"
        details = [
            str(product.get("name", product.get("sku", "product"))) for product in products[:5]
        ]
        embedding_duration_ms = payload.get("embedding_duration_ms")
        if include_timing_details and isinstance(embedding_duration_ms, (int, float)):
            details.append(f"Local embedding: {embedding_duration_ms} ms")
            cache_hit = payload.get("embedding_cache_hit")
            details.append(
                "Embedding cache: "
                + ("Hit" if cache_hit is True else "Miss" if cache_hit is False else "Unavailable")
            )
        return summary, details
    if name == "check_warehouse_inventory" and isinstance(payload, dict):
        quantity = payload.get("quantity", 0)
        availability = str(payload.get("availability", "checked")).replace("_", " ")
        return f"{availability} · quantity {quantity}", []
    if name == "list_context_retriever_tools" and isinstance(payload, dict):
        tools = payload.get("tools", [])
        return f"{len(tools)} governed tools available", []
    if (
        name == "query_context_retriever" or is_context_retriever_tool(name)
    ) and isinstance(payload, dict):
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
            bool(payload.get("redis_agent_memory_saved")) if isinstance(payload, dict) else False
        )
        return "Preference saved" if saved else "Preference write unavailable", []
    if name == "recall_redis_shopping_memory" and isinstance(payload, dict):
        memories = [str(item) for item in payload.get("memories", [])]
        return f"{len(memories)} relevant memories found", memories[:5]
    compact = json.dumps(payload, default=str, separators=(",", ":"))
    return "Completed", [compact[:300]] if compact and compact != "{}" else []


def _tool_duration(name: str, response: dict[str, Any], elapsed_ms: float) -> float:
    """Use operation-specific timing when a tool reports a narrower boundary."""
    payload = response.get("result", response)
    if name == "search_catalog" and isinstance(payload, dict):
        redisvl_duration_ms = payload.get("redisvl_duration_ms")
        if isinstance(redisvl_duration_ms, (int, float)):
            return float(redisvl_duration_ms)
        return 0.0
    if isinstance(payload, dict):
        operation_duration_ms = payload.get("operation_duration_ms")
        if isinstance(operation_duration_ms, (int, float)):
            return float(operation_duration_ms)
    return elapsed_ms


def _tool_trace_duration(
    name: str,
    response: dict[str, Any],
    elapsed_ms: float,
    cache: dict[str, Any] | None,
) -> float | None:
    """Keep cache lookup time separate and omit tool duration entirely on a hit."""
    if cache and cache.get("status") == "hit":
        return None
    cache_read_ms = (
        float(cache.get("read_duration_ms", 0.0))
        if cache and cache.get("read_duration_ms") is not None
        else 0.0
    )
    return _tool_duration(name, response, max(0.0, elapsed_ms - cache_read_ms))


async def member_profile_for_session(
    member_id: str,
    session_id: str,
    context_retriever_enabled: bool = True,
) -> dict[str, Any]:
    """Load the authoritative member profile without depending on ADK session reads."""
    if not context_retriever_enabled:
        return {
            "context": json.dumps({"member_id": member_id}, sort_keys=True),
            "source": "context_retriever_disabled",
        }
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


def member_profile_source_label(source: str) -> str:
    return {
        "redis_context_retriever": "",
        "application_session_cache": "Reused from application session cache",
        "member_id_fallback": "Profile unavailable; using member ID only",
        "context_retriever_disabled": "Disabled for this session",
    }.get(source, source)


async def _chat_events(request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
    """Run one shopping turn and emit observable steps as newline-delimited events."""
    total_started = time.perf_counter()
    member_id = safe_id(request.member_id, settings.valuewholesale_demo_member_id)
    session_id = safe_id(request.session_id, settings.valuewholesale_demo_session_id)

    yield {"type": "start", "session_id": session_id}

    async def fetch_vertex_long_term() -> list[dict[str, Any]]:
        return await services.vertex_memory.recall(member_id, request.message)

    async def fetch_adk_short_term() -> list[dict[str, Any]]:
        try:
            session = await session_service.get_session(
                app_name=TRANSCRIPT_APP_NAME,
                user_id=member_id,
                session_id=adk_transcript_session_id(session_id),
            )
        except Exception as exc:
            log.warning("ADK session telemetry read failed open: %s", exc)
            return []
        if session is None:
            return []
        return recent_adk_transcript_events(session)

    async def fetch_member_profile() -> dict[str, Any]:
        if request.context_retriever_enabled:
            return await member_profile_for_session(member_id, session_id)
        return await member_profile_for_session(member_id, session_id, False)

    async def timed(step_id: str, awaitable: Any) -> tuple[str, Any, float]:
        started = time.perf_counter()
        result = await awaitable
        return step_id, result, round((time.perf_counter() - started) * 1000, 2)

    prefetched_short_term: tuple[str, Any, float] | None = None
    routing_context = ""
    if services.semantic_router.is_contextual_followup(request.message):
        prefetched_short_term = await timed_thread_call(
            "redis-short-term",
            services.memory.short_term,
            session_id,
            SHORT_TERM_MEMORY_LIMIT,
        )
        routing_context = "\n".join(
            memory_snippets(prefetched_short_term[1], SHORT_TERM_MEMORY_LIMIT)
        )
    route_args = (request.message, routing_context) if routing_context else (request.message,)
    routing = await asyncio.to_thread(services.semantic_router.route, *route_args)
    route_duration = routing.get("route_duration_ms", routing.get("redisvl_duration_ms"))
    redisvl_called = route_duration is not None
    cache_read = bool(routing.get("cache_read", routing.get("eligible", False)))
    cache_write = bool(routing.get("cache_write", routing.get("eligible", False)))
    cache_scope = str(routing.get("cache_scope") or "")
    blocked = bool(routing.get("blocked", False))
    route_details = [
        f"Decision source: {routing.get('decision_source', 'unknown')}",
        f"Action: {'block' if blocked else 'allow'}",
        f"LangCache read: {'yes' if cache_read else 'no'}",
        f"LangCache write: {'yes' if cache_write else 'no'}",
        f"LangCache scope: {cache_scope or 'none'}",
        f"Threshold: {routing.get('threshold', 'not set')}",
    ]
    if routing.get("contextual_followup"):
        route_details.append("Recent Redis session context used for routing")
    if routing.get("distance") is not None:
        route_details.append(f"Cosine distance: {routing['distance']}")
    if routing.get("embedding_duration_ms") is not None:
        route_details.append(
            f"Local embedding: {routing['embedding_duration_ms']} ms · "
            f"{settings.valuewholesale_embedding_model}"
        )
    if routing.get("embedding_cache_hit") is not None:
        route_details.append(
            f"Embedding cache: {'Hit' if routing['embedding_cache_hit'] else 'Miss'}"
        )
    if routing.get("redisvl_duration_ms") is not None:
        route_details.append(f"Redis vector classification: {routing['redisvl_duration_ms']} ms")
    if not redisvl_called:
        route_summary = f"RedisVL bypassed · {routing.get('reason', 'routing guardrail')}"
    elif blocked:
        route_summary = "Blocked · outside Value Wholesale ecommerce scope"
    elif cache_read or cache_write:
        route_summary = "LangCache read + write"
    else:
        route_summary = f"{routing.get('route') or 'allowed'} · LangCache bypass"
    yield trace_event(
        "semantic-router",
        "Routing with RedisVL Semantic Router",
        duration_ms=route_duration if redisvl_called else 0,
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
            llm_count_label(gemini_runner_label(request.model), 0),
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

    cached: dict[str, Any] | None = None
    if cache_read:
        _, cached, cache_duration = await timed(
            "langcache",
            services.langcache.search(request.message, cache_scope),
        )
        hit = bool(cached and cached.get("response"))
        cached_prompt = (
            services.langcache.display_prompt(str(cached.get("prompt", "")))
            if hit
            else ""
        )
        yield trace_event(
            "langcache",
            "Checking Redis LangCache",
            duration_ms=cache_duration,
            summary=("Hit" if hit else "Miss"),
            details=(
                [
                    f"Current query: {request.message}",
                    f"Cached query: {cached_prompt}",
                ]
                if cached_prompt
                else []
            ),
        )
        if hit:
            answer = str(cached["response"])
            queue_working_memory_event(member_id, session_id, "USER", request.message)
            queue_working_memory_event(member_id, session_id, "ASSISTANT", answer)
            yield trace_event(
                "generation",
                llm_count_label(gemini_runner_label(request.model), 0),
                duration_ms=0,
                summary="Skipped · response served by LangCache",
            )
            yield {"type": "answer", "answer": answer, "cache_hit": True}
            yield trace_event(
                "total",
                "Total request",
                duration_ms=round((time.perf_counter() - total_started) * 1000, 2),
                summary="Completed from semantic cache",
            )
            return
    else:
        yield trace_event(
            "langcache",
            "Checking Redis LangCache",
            duration_ms=0,
            summary=f"Bypassed · {routing.get('reason', 'router policy')}",
        )

    async def prefetched_short_term_result() -> tuple[str, Any, float]:
        assert prefetched_short_term is not None
        return prefetched_short_term

    required_tasks = {
        asyncio.create_task(
            prefetched_short_term_result()
            if prefetched_short_term is not None
            else timed_thread_call(
                "redis-short-term",
                services.memory.short_term,
                session_id,
                SHORT_TERM_MEMORY_LIMIT,
            )
        ),
        asyncio.create_task(
            timed_thread_call(
                "redis-long-term",
                services.memory.recall,
                member_id,
                request.message,
                4,
            )
        ),
        asyncio.create_task(timed("member-profile", fetch_member_profile())),
    }
    adk_telemetry_tasks = {
        asyncio.create_task(timed("adk-short-term", fetch_adk_short_term())),
        asyncio.create_task(timed("vertex-long-term", fetch_vertex_long_term())),
    }
    # Insert comparison rows as Redis/ADK pairs. Subsequent completion events update
    # these existing rows without changing their order in the browser trace.
    yield trace_event(
        "redis-short-term",
        "Getting Redis short-term memory",
        status="running",
        summary="Required context for Gemini",
    )
    yield trace_event(
        "adk-short-term",
        "ADK VertexAISession read",
        status="running",
    )
    yield trace_event(
        "redis-long-term",
        "Searching Redis long-term memory",
        status="running",
        summary="Required context for Gemini",
    )
    yield trace_event(
        "vertex-long-term",
        "ADK Memory Bank search",
        status="running",
    )

    def adk_telemetry_event(
        step_id: str, result: list[dict[str, Any]], duration: float
    ) -> dict[str, Any]:
        snippets = memory_snippets(
            result,
            SHORT_TERM_MEMORY_LIMIT if step_id == "adk-short-term" else 5,
        )
        if step_id == "adk-short-term":
            return trace_event(
                step_id,
                "ADK VertexAISession read",
                duration_ms=duration,
                summary=f"{len(result)} recent transcript events",
                details=snippets,
            )
        return trace_event(
            step_id,
            "ADK Memory Bank search",
            duration_ms=duration,
            summary=f"{len(result)} memories",
            details=snippets,
        )

    results: dict[str, Any] = {}
    pending_required = set(required_tasks)
    pending_adk_telemetry = set(adk_telemetry_tasks)
    while pending_required:
        completed, _ = await asyncio.wait(
            pending_required | pending_adk_telemetry,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in completed:
            step_id, result, duration = await task
            if task in pending_adk_telemetry:
                pending_adk_telemetry.remove(task)
                yield adk_telemetry_event(step_id, result, duration)
                continue

            pending_required.remove(task)
            results[step_id] = result
            if step_id == "redis-short-term":
                snippets = memory_snippets(result, SHORT_TERM_MEMORY_LIMIT)
                yield trace_event(
                    step_id,
                    "Getting Redis short-term memory",
                    duration_ms=duration,
                    summary=f"{len(snippets)} recent session events",
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
                if source not in {"application_session_cache", "context_retriever_disabled"}:
                    yield trace_event(
                        step_id,
                        "Context Retriever - get_member_by_id",
                        duration_ms=duration,
                        summary=member_profile_source_label(source),
                        details=[result["context"]],
                    )

    async def drain_adk_telemetry() -> AsyncIterator[dict[str, Any]]:
        for task in asyncio.as_completed(pending_adk_telemetry):
            step_id, result, duration = await task
            yield adk_telemetry_event(step_id, result, duration)
        pending_adk_telemetry.clear()

    short_memories = results.get("redis-short-term", [])
    redis_memories = results.get("redis-long-term", [])
    member_profile = results.get(
        "member-profile",
        {"context": json.dumps({"member_id": member_id}, sort_keys=True)},
    )

    queue_working_memory_event(member_id, session_id, "USER", request.message)

    state_delta = {
        "member_id": member_id,
        "user_id": member_id,
        "member_profile_context": member_profile["context"],
        "context_retriever_enabled": request.context_retriever_enabled,
        "redis_short_term_context": "\n".join(
            memory_snippets(short_memories, SHORT_TERM_MEMORY_LIMIT)
        )
        or "No prior session events.",
        "redis_long_term_context": "\n".join(memory_snippets(redis_memories))
        or "No relevant Redis long-term memories.",
        "cache_safety_context": (
            f"Cacheable reusable scope `{cache_scope}`. Do not include member-specific facts, "
            "current prices, inventory, orders, carts, or other live data in the answer."
            if cache_scope
            else "Not cache eligible; follow the normal grounding and personalization rules."
        ),
    }
    runner_started = time.perf_counter()
    tool_starts: dict[str, tuple[float, str, dict[str, Any], bool]] = {}
    final_answer = ""
    llm_calls = 0
    next_runner_event: asyncio.Task[Any] | None = None

    async def cancel_in_flight_tasks() -> None:
        tasks = list(pending_adk_telemetry)
        if next_runner_event is not None and not next_runner_event.done():
            tasks.append(next_runner_event)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    try:
        async with asyncio.timeout(settings.valuewholesale_agent_timeout_seconds):
            runner_events = (
                runners[request.model]
                .run_async(
                    user_id=member_id,
                    session_id=session_id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=request.message)]
                    ),
                    state_delta=state_delta,
                )
                .__aiter__()
            )
            next_runner_event = asyncio.create_task(anext(runner_events))
            while True:
                completed, _ = await asyncio.wait(
                    {next_runner_event} | pending_adk_telemetry,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for telemetry_task in completed & pending_adk_telemetry:
                    pending_adk_telemetry.remove(telemetry_task)
                    step_id, result, duration = await telemetry_task
                    yield adk_telemetry_event(step_id, result, duration)

                if next_runner_event not in completed:
                    continue
                try:
                    event = next_runner_event.result()
                except StopAsyncIteration:
                    break
                if is_llm_response_event(event):
                    llm_calls += 1
                for call in event.get_function_calls():
                    name = str(call.name or "tool")
                    arguments = dict(call.args or {})
                    call_id = str(call.id or name)
                    trace_visible = True
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
                    response_data = dict(response.response or {})
                    cache_info = response_data.pop(TOOL_CALL_CACHE_METADATA_KEY, None)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                    duration = _tool_trace_duration(
                        name, response_data, elapsed_ms, cache_info
                    )
                    summary, details = _tool_summary(
                        name,
                        response_data,
                        include_timing_details=not (
                            cache_info and cache_info.get("status") == "hit"
                        ),
                    )
                    if trace_visible:
                        yield trace_event(
                            f"tool-{call_id}",
                            _tool_label(name, arguments),
                            duration_ms=duration,
                            summary=summary,
                            details=details,
                            cache=cache_info,
                        )
                if event.is_final_response():
                    final_answer = event_text(event)
                next_runner_event = asyncio.create_task(anext(runner_events))
    except TimeoutError:
        await cancel_in_flight_tasks()
        elapsed = round((time.perf_counter() - runner_started) * 1000, 2)
        yield trace_event(
            "generation",
            llm_count_label(gemini_runner_label(request.model), llm_calls),
            status="error",
            duration_ms=elapsed,
            summary="Timed out; retry the request",
        )
        yield {"type": "error", "message": "The model timed out. Please retry."}
        return
    except Exception as exc:
        await cancel_in_flight_tasks()
        log.exception("ADK request failed")
        yield {"type": "error", "message": f"Agent request failed: {exc}"}
        return

    if not final_answer:
        await cancel_in_flight_tasks()
        yield {"type": "error", "message": "Agent returned no final response"}
        return

    runner_duration = round((time.perf_counter() - runner_started) * 1000, 2)
    yield trace_event(
        "generation",
        llm_count_label(gemini_runner_label(request.model), llm_calls),
        duration_ms=runner_duration,
    )

    queue_working_memory_event(member_id, session_id, "ASSISTANT", final_answer)
    if cache_write:
        await services.langcache.store(request.message, final_answer, cache_scope)

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
    """Hydrate the shopping session, then generate an optionally personalized greeting."""
    member_id = safe_id(request.member_id, settings.valuewholesale_demo_member_id)
    shopping_session_id = safe_id(request.session_id, settings.valuewholesale_demo_session_id)
    session_id = safe_id(f"{shopping_session_id}-greeting", "greeting-session")
    tool_starts: dict[str, tuple[float, str, dict[str, Any], bool]] = {}
    context_sources: list[str] = []
    context_details: list[str] = []
    final_greeting = ""
    llm_calls = 0

    yield {"type": "start", "session_id": session_id}
    profile_started = time.perf_counter()
    member_profile = (
        await member_profile_for_session(member_id, shopping_session_id)
        if request.context_retriever_enabled
        else await member_profile_for_session(member_id, shopping_session_id, False)
    )
    profile_source = str(member_profile.get("source", "unavailable"))
    if profile_source not in {"application_session_cache", "context_retriever_disabled"}:
        yield trace_event(
            "greeting-member-profile",
            "Context Retriever - get_member_by_id",
            duration_ms=round((time.perf_counter() - profile_started) * 1000, 2),
            summary=member_profile_source_label(profile_source),
            details=[member_profile["context"]],
        )
    yield trace_event(
        "greeting-generation",
        "ADK Greeting",
        status="running",
        summary="Selecting optional context and generating a greeting",
    )
    runner_started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.valuewholesale_agent_timeout_seconds):
            async for event in greeting_runners[request.model].run_async(
                user_id=member_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Create my brief welcome greeting using the supplied member "
                                "profile. Decide whether personal long term memory or order "
                                "history would improve it."
                            )
                        )
                    ],
                ),
                state_delta={
                    "member_id": member_id,
                    "user_id": member_id,
                    "member_profile_context": member_profile["context"],
                    "context_retriever_enabled": request.context_retriever_enabled,
                },
            ):
                if is_llm_response_event(event):
                    llm_calls += 1
                for call in event.get_function_calls():
                    name = str(call.name or "tool")
                    arguments = dict(call.args or {})
                    call_id = str(call.id or name)
                    trace_visible = True
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
                    response_data = dict(response.response or {})
                    cache_info = response_data.pop(TOOL_CALL_CACHE_METADATA_KEY, None)
                    summary, details = _tool_summary(
                        name,
                        response_data,
                        include_timing_details=not (
                            cache_info and cache_info.get("status") == "hit"
                        ),
                    )
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
                        elapsed_ms = round(
                            (time.perf_counter() - call_started) * 1000, 2
                        )
                        tool_duration = _tool_trace_duration(
                            name, response_data, elapsed_ms, cache_info
                        )
                        yield trace_event(
                            f"greeting-tool-{call_id}",
                            _tool_label(name, arguments),
                            duration_ms=tool_duration,
                            summary=summary,
                            details=details,
                            cache=cache_info,
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
        else ("The agent chose not to call Redis Agent Memory or Context Retriever.")
    )
    yield trace_event(
        "greeting-generation",
        llm_count_label("ADK Greeting", llm_calls),
        duration_ms=round((time.perf_counter() - runner_started) * 1000, 2),
        summary=context_summary,
        details=context_details,
        move_to_end=True,
    )
    yield {"type": "greeting", "greeting": final_greeting.strip()}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def warmup_redis_services() -> dict[str, Any]:
    """Prime the six Redis integrations used on the shopping request path."""
    started = time.perf_counter()

    async def probe(name: str, operation: Any) -> tuple[str, dict[str, Any]]:
        probe_started = time.perf_counter()
        try:
            ok, summary, details = await operation()
        except Exception as exc:
            log.warning("Warm-up probe failed for %s: %s", name, exc)
            ok, summary, details = False, f"Unavailable ({type(exc).__name__})", {}
        measured_duration_ms = details.pop("_operation_duration_ms", None)
        return name, {
            "ok": ok,
            "duration_ms": (
                measured_duration_ms
                if measured_duration_ms is not None
                else round((time.perf_counter() - probe_started) * 1000, 2)
            ),
            "summary": summary,
            **details,
        }

    async def database_probe() -> tuple[bool, str, dict[str, Any]]:
        ok, duration_ms = await asyncio.to_thread(call_with_timing, services.catalog.ping)
        return ok, "Redis PING succeeded" if ok else "Database is not configured", {
            "_operation_duration_ms": duration_ms
        }

    async def context_probe() -> tuple[bool, str, dict[str, Any]]:
        tools = await services.context.list_tools(force_refresh=True)
        count = len(tools)
        return bool(count), f"{count} governed tools discovered", {"tools": tools}

    async def router_probe() -> tuple[bool, str, dict[str, Any]]:
        def warm_router() -> tuple[dict[str, Any], dict[str, Any]]:
            return (
                services.embeddings.warmup(),
                services.semantic_router.route("What is the electronics return policy?"),
            )

        (embedding, decision), duration_ms = await asyncio.to_thread(
            call_with_timing,
            warm_router,
        )
        ok = decision.get("decision_source") == "redisvl"
        route = decision.get("route") or "no route"
        return ok, f"Semantic route ready · {route}", {
            "embedding": embedding,
            "_operation_duration_ms": duration_ms,
        }

    async def embedding_cache_probe() -> tuple[bool, str, dict[str, Any]]:
        (ok, summary, details), duration_ms = await asyncio.to_thread(
            call_with_timing,
            services.embeddings.cache_probe,
        )
        return ok, summary, {**details, "_operation_duration_ms": duration_ms}

    async def langcache_probe() -> tuple[bool, str, dict[str, Any]]:
        ok = await services.langcache.warmup("What is the electronics return policy?")
        return ok, "Semantic lookup completed" if ok else "LangCache is not configured", {}

    async def memory_probe() -> tuple[bool, str, dict[str, Any]]:
        health_started = time.perf_counter()
        ok = await services.memory.ping()
        health_ms = round((time.perf_counter() - health_started) * 1000, 2)
        if not ok:
            return False, "Agent Memory is not configured", {"health_ms": health_ms}

        async def timed_read(operation: Any) -> float:
            _, duration_ms = await asyncio.to_thread(call_with_timing, operation)
            return duration_ms

        short_term_ms, long_term_ms = await asyncio.gather(
            timed_read(
                lambda: services.memory.short_term(
                    settings.valuewholesale_demo_session_id,
                    1,
                )
            ),
            timed_read(
                lambda: services.memory.recall(
                    settings.valuewholesale_demo_member_id,
                    "shopping preferences",
                    1,
                )
            ),
        )
        return True, "Health check and memory reads passed", {
            "health_ms": health_ms,
            "short_term_ms": short_term_ms,
            "long_term_ms": long_term_ms,
        }

    results = await asyncio.gather(
        probe("redis_database", database_probe),
        probe("context_retriever", context_probe),
        probe("semantic_router", router_probe),
        probe("embedding_cache", embedding_cache_probe),
        probe("langcache", langcache_probe),
        probe("redis_agent_memory", memory_probe),
    )
    service_results = dict(results)
    for service_name, result in service_results.items():
        if result["ok"]:
            if service_name == "redis_agent_memory":
                latency_registry.mark_cold_call_complete("redis_agent_memory_short_term")
                latency_registry.mark_cold_call_complete("redis_agent_memory_long_term")
            else:
                latency_registry.mark_cold_call_complete(service_name)
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
        "embedding_model": settings.valuewholesale_embedding_model,
        "redis_endpoint": settings.redis_endpoint,
        "models": settings.available_google_models,
        "services": {
            "redis_database": settings.redis_configured,
            "context_retriever": bool(settings.mcp_agent_key),
            "semantic_router": services.semantic_router.configured,
            "embedding_cache": services.embeddings.embedding_cache is not None,
            "langcache": settings.langcache_configured,
            "redis_agent_memory": services.memory.client is not None,
            "vertex_adk_memory_bank": services.vertex_memory.client is not None,
            "agent_platform_sessions": isinstance(session_service, VertexAiSessionService),
        },
    }


@app.post("/api/reset-demo")
async def reset_demo() -> dict[str, Any]:
    """Flush shared LangCache state so the scripted demo starts cold."""
    if not settings.langcache_configured:
        raise HTTPException(status_code=503, detail="LangCache is not configured")
    try:
        await services.langcache.clear()
    except Exception as exc:
        log.warning("Demo reset failed: %s", exc)
        raise HTTPException(status_code=502, detail="LangCache reset failed") from exc
    return {"ok": True, "message": "LangCache flushed"}


@app.post("/api/reset-member-memory")
async def reset_member_memory(request: MemoryResetRequest) -> dict[str, Any]:
    """Restore both memory providers for a small demo member to the seed set."""
    if request.member_id not in MEMBERS:
        raise HTTPException(status_code=404, detail="Demo member not found")
    if request.member_id not in MEMORY_RESETTABLE_MEMBERS:
        raise HTTPException(
            status_code=403,
            detail="Memory reset is limited to the four small demo members",
        )
    if services.memory.client is None:
        raise HTTPException(status_code=503, detail="Redis Agent Memory is not configured")
    if services.vertex_memory.client is None:
        raise HTTPException(status_code=503, detail="Vertex ADK Memory Bank is not configured")

    try:
        seeded_memories = [
            json.loads(line)
            for line in MEMORY_SEEDS_PATH.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Memory seed data could not be loaded from %s: %s", MEMORY_SEEDS_PATH, exc)
        raise HTTPException(
            status_code=500,
            detail="Memory seed data is unavailable in this deployment",
        ) from exc
    member_seeds = [
        memory for memory in seeded_memories if memory.get("owner_id") == request.member_id
    ]
    if not member_seeds:
        raise HTTPException(status_code=500, detail="No seeded memories found for demo member")

    await drain_working_memory_tasks()
    try:
        redis_result, vertex_result = await asyncio.gather(
            asyncio.to_thread(
                services.memory.reset_long_term,
                request.member_id,
                member_seeds,
            ),
            asyncio.to_thread(
                services.vertex_memory.reset_long_term,
                request.member_id,
                member_seeds,
            ),
        )
    except Exception as exc:
        log.warning("Member memory reset failed for %s: %s", request.member_id, exc)
        raise HTTPException(status_code=502, detail="Member memory reset failed") from exc
    return {
        "ok": True,
        "member_id": request.member_id,
        "providers": {
            "redis_agent_memory": redis_result,
            "vertex_adk_memory_bank": vertex_result,
        },
    }


@app.get("/api/member-memory")
async def member_memory(member_id: str) -> dict[str, Any]:
    """Return bounded presenter-only inventories without joining the prompt path."""
    if member_id not in MEMBERS:
        raise HTTPException(status_code=404, detail="Demo member not found")

    async def inventory(provider: str, operation: Any) -> tuple[str, dict[str, Any]]:
        if operation is None:
            return provider, {
                "available": False,
                "count": 0,
                "truncated": False,
                "memories": [],
            }
        try:
            result = await asyncio.to_thread(operation, member_id)
            return provider, {"available": True, **result}
        except Exception as exc:
            log.warning("%s memory inventory failed for %s: %s", provider, member_id, exc)
            return provider, {
                "available": False,
                "count": 0,
                "truncated": False,
                "memories": [],
            }

    redis_operation = (
        services.memory.list_long_term if services.memory.client is not None else None
    )
    vertex_operation = (
        services.vertex_memory.list_long_term
        if services.vertex_memory.client is not None
        else None
    )
    providers = dict(
        await asyncio.gather(
            inventory("redis_agent_memory", redis_operation),
            inventory("vertex_adk_memory_bank", vertex_operation),
        )
    )
    return {"ok": True, "member_id": member_id, "providers": providers}


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


@app.get("/api/latency-stats")
async def latency_stats() -> dict[str, Any]:
    return {
        "scope": "current application worker lifetime",
        "cold_call_excluded": True,
        "services": latency_registry.snapshot(),
    }


@app.post("/api/warmup")
async def warmup() -> dict[str, Any]:
    return await warmup_redis_services()


@app.post("/api/keepalive")
async def keepalive() -> dict[str, Any]:
    result = await warmup_redis_services()
    return {
        "ok": result["ok"],
        "duration_ms": result["duration_ms"],
    }


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
                "memory_resettable": member["member_id"] in MEMORY_RESETTABLE_MEMBERS,
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
        "session_id": safe_id(request.session_id, settings.valuewholesale_demo_session_id),
        "cache": {"hit": cache_hit},
        "trace": trace,
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        async for event in _chat_events(request):
            yield json.dumps(event, default=str, separators=(",", ":")) + "\n"
        yield json.dumps(
            {"type": "latency_stats", "services": latency_registry.snapshot()},
            separators=(",", ":"),
        ) + "\n"

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
        yield json.dumps(
            {"type": "latency_stats", "services": latency_registry.snapshot()},
            separators=(",", ":"),
        ) + "\n"

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
                safe_id(request.member_id, settings.valuewholesale_demo_member_id),
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
