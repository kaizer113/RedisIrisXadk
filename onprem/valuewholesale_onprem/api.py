"""Only the port 8083 application uses this entrypoint and dependency environment."""

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import WatchError
from redis_iris.agent_mesh.fastapi_sse import FastAPIRedis, get_stream_backend

from .config import Settings
from .engine import Turn
from .identity import COOKIE, SESSION_SECONDS, Identity, issue, verify
from .services import Services, tool_json

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class SessionRequest(BaseModel):
    member_id: str = Field(min_length=1, max_length=80)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    run_id: UUID
    context_gateway_enabled: bool = True


def run_key(settings: Settings, identity: Identity, run_id: str, kind: str) -> str:
    return f"{settings.onprem_namespace}:{kind}:{identity.scope}:{run_id}"


async def release_lock(redis: Any, key: str, run_id: str) -> None:
    async with redis.pipeline() as pipe:
        try:
            await pipe.watch(key)
            if await pipe.get(key) == run_id:
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
        except WatchError:
            pass  # A newer request owns it; never remove its lock.


async def produce(
    services: Services, identity: Identity, request: ChatRequest, writer: Any, lock_key: str
) -> None:
    run_id = str(request.run_id)
    settings, redis = services.settings, services.redis
    snapshot: dict = {"status": "running", "events": []}
    key = run_key(settings, identity, run_id, "snapshot")

    async def emit(event: dict) -> None:
        snapshot["events"] = [*snapshot["events"][-299:], event]
        if event["type"] == "answer":
            snapshot["answer"] = event["answer"]
            snapshot["published"] = event.get("published")
            snapshot["model_calls"] = event.get("model_calls")
        if event["type"] == "lens":
            snapshot["trace_id"] = event["trace_id"]
        await redis.set(key, json.dumps(snapshot), ex=settings.onprem_run_ttl)
        await writer.append(event["type"], event)

    try:
        await Turn(
            services, identity, run_id, request.message, request.context_gateway_enabled, emit
        ).run()
        snapshot["status"] = "completed"
        await redis.set(key, json.dumps(snapshot), ex=settings.onprem_run_ttl)
        await writer.finish("completed", {"type": "completed"})
    except Exception as exc:
        log.error("GA turn failed (%s)", type(exc).__name__)
        snapshot["status"] = "failed"
        snapshot["error"] = (
            "A required service failed. Inspect the last service step and retry a new turn."
        )
        await redis.set(key, json.dumps(snapshot), ex=settings.onprem_run_ttl)
        await writer.finish("failed", {"type": "failed", "message": snapshot["error"]})
    finally:
        await release_lock(redis, lock_key, run_id)


def create_app(
    settings: Settings | None = None,
    *,
    redis_client: Any = None,
    services_factory: Any = Services,
    stream_backend: Any = None,
) -> FastAPI:
    settings = settings or Settings()
    members = []
    if settings.onprem_members_file.exists():
        members = [
            {"id": row["member_id"], "name": row["name"]}
            for row in (
                json.loads(line)
                for line in settings.onprem_members_file.read_text().splitlines()
                if line.strip()
            )
        ]
    member_ids = {member["id"] for member in members}

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        redis = redis_client
        if redis is None and settings.redis_url.get_secret_value():
            redis = Redis.from_url(
                settings.redis_url.get_secret_value(),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=8,
            )
        services = services_factory(settings, redis)
        application.state.services = services
        if stream_backend is not None:
            application.state.test_stream_backend = stream_backend
        await services.open()
        try:
            yield
        finally:
            await services.close()
            if redis is not None and redis_client is None:
                await redis.aclose()

    application = FastAPI(title="Value Wholesale · Iris GA", lifespan=lifespan)
    # The upstream builder reads REDIS_* from the runtime environment and composes lifespans.
    if settings.redis_url.get_secret_value() and stream_backend is None:
        FastAPIRedis(application).lifespan().streams()

    async def backend(request: Request) -> Any:
        if stream_backend is not None:
            return stream_backend
        if not settings.redis_url.get_secret_value():
            raise HTTPException(503, "Configure REDIS_URL for Agent Mesh")
        return await get_stream_backend(request)

    def identity(request: Request) -> Identity:
        try:
            result = verify(
                request.cookies.get(COOKIE, ""), settings.onprem_session_secret.get_secret_value()
            )
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        if result.member_id not in member_ids:
            raise HTTPException(401, "Unknown demo member")
        return result

    identity_dep = Annotated[Identity, Depends(identity)]
    backend_dep = Annotated[Any, Depends(backend)]

    @application.middleware("http")
    async def private_responses(request: Request, call_next: Any):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @application.get("/api/members")
    async def list_members():
        return {"members": members}

    @application.post("/api/session")
    async def new_session(body: SessionRequest, response: Response):
        if body.member_id not in member_ids:
            raise HTTPException(422, "Choose a known synthetic demo member")
        secret = settings.onprem_session_secret.get_secret_value()
        if len(secret) < 32:
            raise HTTPException(
                503, "Configure a stable ONPREM_SESSION_SECRET of at least 32 characters"
            )
        current, cookie = issue(body.member_id, secret)
        response.set_cookie(
            COOKIE,
            cookie,
            max_age=SESSION_SECONDS,
            httponly=True,
            samesite="lax",
            secure=settings.onprem_cookie_secure,
        )
        return {"member_id": current.member_id, "session_id": current.session_id}

    @application.get("/api/session")
    async def current_session(current: identity_dep):
        return {"member_id": current.member_id, "session_id": current.session_id}

    @application.get("/api/health")
    async def health(request: Request):
        services = request.app.state.services
        checks: dict = {}

        async def check(name: str, fn: Any):
            if not settings.configured[name]:
                checks[name] = {"ok": False, "detail": "Runtime settings missing"}
                return
            try:
                async with asyncio.timeout(8):
                    ok = bool(await fn())
                checks[name] = {"ok": ok, "detail": "Probe passed" if ok else "Probe not ready"}
            except Exception as exc:
                checks[name] = {"ok": False, "detail": "Probe failed: " + type(exc).__name__}

        async def playbook_health():
            result = await services.playbooks.client.get_playbook_health_async()
            return str(getattr(result.status, "value", result.status)) == "READY"

        async def router_health():
            return (await asyncio.to_thread(services.router.health)).ok

        async def mesh_health():
            return await (await backend(request)).detect()

        await asyncio.gather(
            check("agent_memory", lambda: services.memory.health()),
            check("agent_playbooks", playbook_health),
            check("intent_router", router_health),
            check("context_gateway", lambda: services.gateway.atools()),
            check("agent_mesh", mesh_health),
        )
        checks["gemini"] = {
            "ok": services.model is not None,
            "detail": "Vertex client configured; inference checked during a turn"
            if services.model is not None
            else "Vertex client is not configured",
        }
        checks["lens"] = {
            "ok": services.lens_enabled,
            "detail": "SDK initialized; confirm exported turns in Lens"
            if services.lens_enabled
            else "Lens SDK is not initialized",
        }
        return {
            "release": "1.0",
            "services": {key: value["ok"] for key, value in checks.items()},
            "configured": settings.configured,
            "checks": checks,
            "ready": all(value["ok"] for value in checks.values()),
        }

    @application.get("/api/tools")
    async def list_tools(request: Request, _current: identity_dep, query: str = "shopping catalog"):
        services = request.app.state.services
        if not services.gateway:
            raise HTTPException(503, "Configure Context Gateway")
        try:
            return {"tools": [tool_json(tool) for tool in await services.tools(query[:1000])]}
        except Exception as exc:
            raise HTTPException(503, "Context Gateway tool discovery is unavailable") from exc

    @application.get("/api/memory")
    async def memory_inventory(request: Request, current: identity_dep):
        services = request.app.state.services
        if not services.memory:
            raise HTTPException(503, "Configure Agent Memory and its namespace")
        try:
            return {
                "memories": await services.memory.recall(current),
                "namespace_id": settings.iris_agent_memory_namespace_id,
                "limit": 20,
            }
        except Exception as exc:
            raise HTTPException(503, "Member memory is unavailable") from exc

    @application.post("/api/chat")
    async def start_chat(
        body: ChatRequest, request: Request, current: identity_dep, streams: backend_dep
    ):
        services = request.app.state.services
        missing = [
            name
            for name in ("agent_memory", "agent_playbooks", "intent_router", "gemini")
            if not settings.configured[name]
        ]
        if body.context_gateway_enabled and not settings.configured["context_gateway"]:
            missing.append("context_gateway")
        if missing:
            raise HTTPException(503, "Missing GA runtime settings: " + ", ".join(missing))
        redis, run_id = services.redis, str(body.run_id)
        fingerprint = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        binding = run_key(settings, current, run_id, "request")
        created = await redis.set(binding, fingerprint, ex=settings.onprem_run_ttl, nx=True)
        if not created and await redis.get(binding) != fingerprint:
            raise HTTPException(409, "Run ID already belongs to another request")
        events_url = f"/api/runs/{run_id}/events"
        if not created and await redis.exists(run_key(settings, current, run_id, "snapshot")):
            return {"run_id": run_id, "created": False, "events_url": events_url}
        lock_key = f"{settings.onprem_namespace}:active:{current.scope}"
        acquired = await redis.set(lock_key, run_id, nx=True, ex=settings.onprem_turn_timeout + 30)
        if not acquired and await redis.get(lock_key) != run_id:
            raise HTTPException(409, "Another turn is running in this session")
        try:
            writer = await streams.start(
                lambda writer: produce(services, current, body, writer, lock_key),
                scope=current.scope,
                stream_id=run_id,
            )
        except Exception:
            await release_lock(redis, lock_key, run_id)
            raise
        return {"run_id": run_id, "created": writer.created, "events_url": events_url}

    @application.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: UUID,
        request: Request,
        current: identity_dep,
        streams: backend_dep,
        after: str | None = None,
    ):
        services = request.app.state.services
        if not await services.redis.exists(run_key(settings, current, str(run_id), "request")):
            raise HTTPException(404, "Run not found in this session")
        return await streams.sse(
            str(run_id),
            request=request,
            scope=current.scope,
            after=after or request.headers.get("last-event-id"),
        )

    @application.get("/api/runs/{run_id}")
    async def run_snapshot(run_id: UUID, request: Request, current: identity_dep):
        services = request.app.state.services
        raw = await services.redis.get(run_key(settings, current, str(run_id), "snapshot"))
        if raw is None:
            raise HTTPException(404, "No retained result in this session")
        return json.loads(raw)

    return application


app = create_app()
