"""GA service adapters with explicit ownership, publication, and tool boundaries."""

import hashlib
import json
import math
import re
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from jsonschema import validate
from redis_agent_memory import AgentMemory as MemoryClient
from redis_agent_memory import models
from redis_iris import lens
from redis_iris.agent_playbooks import AgentPlaybooks, AgentPlaybooksSettings
from redis_iris.context_gateway import ContextGateway, ContextGatewaySettings
from redis_iris.intent_router import IntentRouter, IntentRouterSettings

from .config import Settings
from .identity import Identity


class Memory:
    def __init__(self, settings: Settings, client: Any = None):
        self.namespace_id = settings.iris_agent_memory_namespace_id
        self.client = client or MemoryClient(
            server_url=settings.iris_agent_memory_url,
            store_id=settings.iris_agent_memory_store_id,
            api_key=settings.iris_agent_memory_api_key.get_secret_value(),
            timeout_ms=10000,
        )

    def search_request(
        self, identity: Identity, query: str | None = None, cursor: str | None = None
    ) -> models.SearchLongTermMemoryRequestContent:
        if not self.namespace_id:
            raise ValueError("A GA Memory namespace ID is required")
        return models.SearchLongTermMemoryRequestContent(
            query=query,
            limit=20,
            cursor=cursor,
            filter_=models.LongTermMemoryFilter(
                op=models.FilterConjunction.ALL,
                owner_id={"eq": identity.owner},
                namespace_id={"eq": self.namespace_id},
            ),
        )

    async def recall(self, identity: Identity, query: str | None = None) -> list[dict]:
        with lens.memory_recall(query=query) as span:
            response = await self.client.search_long_term_memory_async(
                request=self.search_request(identity, query),
            )
            records = []
            for item in response.items:
                if item.owner_id != identity.owner or item.namespace_id != self.namespace_id:
                    raise PermissionError("Recalled memory has a different member or namespace")
                # The generated SDK's custom serializer emits aliases even
                # with by_alias=False. Read public model attributes directly.
                records.append(
                    {
                        "memory_id": item.memory_id,
                        "text": item.text,
                        "owner_id": item.owner_id,
                        "namespace_id": item.namespace_id,
                    }
                )
            span.results([{"id": item["memory_id"], "text": item["text"]} for item in records])
            return records

    async def transcript(self, identity: Identity) -> list[dict]:
        try:
            response = await self.client.get_session_memory_async(
                session_id=identity.session_id,
                include_summarised_events=False,
            )
        except Exception as exc:
            if getattr(getattr(exc, "raw_response", None), "status_code", None) == 404:
                return []
            raise
        if response.owner_id != identity.owner or response.namespace_id != self.namespace_id:
            raise PermissionError("Session does not belong to this member and namespace")
        result = []
        if response.summary:
            result.append(
                {"role": "user", "text": "Earlier conversation: " + response.summary.text}
            )
        for event in response.events[-12:]:
            role = str(getattr(event.role, "value", event.role)).lower()
            if role in ("user", "assistant"):
                result.append({"role": role, "text": " ".join(part.text for part in event.content)})
        return result

    async def append(self, identity: Identity, run_id: str, role: str, text: str) -> None:
        await self.client.add_session_event_async(
            session_id=identity.session_id,
            actor_id=identity.owner if role == "user" else "valuewholesale-onprem-agent",
            role=models.MessageRole[role.upper()],
            content=[{"text": text}],
            namespace_id=self.namespace_id,
            created_at=datetime.now(UTC),
            idempotency_key=f"{run_id}-{role}",
        )

    async def remember(self, identity: Identity, text: str) -> dict:
        record = models.CreateMemoryRecord(
            memory_id=hashlib.sha256(
                f"{self.namespace_id}:{identity.owner}:{text}".encode()
            ).hexdigest(),
            owner_id=identity.owner,
            namespace_id=self.namespace_id,
            memory_type="semantic",
            text=text,
            topics=["shopping-preference", "onprem-demo"],
        )
        await self.client.bulk_create_long_term_memories_async(
            memories=[record],
            idempotency_key=record.memory_id,
        )
        return {"saved": True, "memory_id": record.memory_id}

    async def health(self) -> bool:
        response = await self.client.store_health_async(timeout_ms=5000)
        return str(getattr(response.status, "value", response.status)) == "HEALTHY"


def published_direct_answer(message: str, match: Any, discovery: Any) -> Any:
    """Only approved guidance routes may bypass the model; commerce remains live."""
    if not match or not match.target or match.target.ref != "published_guidance":
        return None
    if re.search(
        r"\b(my|mine|me|today|now|current|inventory|stock|available|price|order|cart)\b",
        message,
        re.I,
    ):
        return None
    best = discovery.best_faq
    return best if best is not None and best.score >= 0.85 else None


def read_tool(tool: Any) -> bool:
    original = original_tool_name(tool)
    if original in GATEWAY_CORE_TOOLS:
        return False
    if "readOnlyHint" in tool.annotations:
        return tool.annotations["readOnlyHint"] is True
    return bool(
        tool.annotations.get("readOnlyHint") is True
        or re.match(
            r"^(get|find|search|list|query|count|aggregate|describe)_",
            original,
        )
    )


def member_arguments(tool: Any, arguments: dict, identity: Identity) -> dict:
    """The model cannot choose another member for account-specific tool calls."""
    args = dict(arguments)
    properties = tool.input_schema.get("properties", {})
    original = original_tool_name(tool)
    if original in GENERIC_SET_TOOLS:
        validate_public_sets(args)
    for key in MEMBER_FIELDS:
        if key in properties or key in args:
            args[key] = identity.member_id
    if re.search(r"(member|customer|user|profile)", original, re.I) and "id" in properties:
        args["id"] = identity.member_id
    field = condition_member_field(tool)
    if field:
        conditions = args.get("tag_conditions", [])
        if not isinstance(conditions, list) or not all(isinstance(c, dict) for c in conditions):
            raise ValueError("Tag conditions must be a list of field/value objects")
        args["tag_conditions"] = [c for c in conditions if c.get("field") not in MEMBER_FIELDS] + [
            {"field": field, "value": identity.member_id}
        ]
    return args


MEMBER_FIELDS = frozenset(
    {
        "member_id",
        "memberId",
        "customer_id",
        "customerId",
        "owner_id",
        "ownerId",
        "user_id",
        "userId",
    }
)

GENERIC_SET_TOOLS = frozenset(
    {"expand_results", "intersect_results", "union_results", "except_results"}
)
GATEWAY_CORE_TOOLS = frozenset(
    {
        "search_tools",
        "run_tool",
        "run_tool_as_handle",
        "page_results",
        "query_results",
        "count_results",
        "aggregate_results",
        "describe_result",
        "release_results",
    }
)
PUBLIC_ENTITIES = frozenset({"Product", "Warehouse", "Inventory"})
PUBLIC_RELATIONSHIPS = frozenset({"inventory", "product", "warehouse"})


def original_tool_name(tool: Any) -> str:
    return tool.original_name or tool.name.rsplit("__", 1)[-1]


def validate_public_sets(value: Any) -> None:
    """Generic set tools may join public commerce data, never account records.

    These tools have entity-neutral names, so classifying only their name would
    miss member/order selections or a public-to-private relationship expansion.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "entity" and child not in PUBLIC_ENTITIES:
                raise PermissionError("Use a member-scoped account tool for account records")
            if key in {"relationship", "expand_to"} and (
                not isinstance(child, str) or child not in PUBLIC_RELATIONSHIPS
            ):
                raise PermissionError("Account relationship expansion is unavailable")
            validate_public_sets(child)
    elif isinstance(value, list):
        for child in value:
            validate_public_sets(child)


def condition_member_field(tool: Any) -> str | None:
    """Recognize the actual GA Surface filter schema's member TAG field."""
    properties = tool.input_schema.get("properties", {})
    condition = properties.get("tag_conditions", {}).get("items", {})
    fields = condition.get("properties", {}).get("field", {}).get("enum", [])
    return next((field for field in fields if field in MEMBER_FIELDS), None)


def account_tool(tool: Any) -> bool:
    return bool(
        re.search(
            r"(member|customer|user|profile|order|reservation|account|reward|balance)",
            original_tool_name(tool),
            re.I,
        )
    )


def member_bound(tool: Any) -> bool:
    properties = tool.input_schema.get("properties", {})
    return bool(
        MEMBER_FIELDS.intersection(properties)
        or condition_member_field(tool)
        or (
            re.search(r"(member|customer|user|profile)", original_tool_name(tool), re.I)
            and "id" in properties
        )
    )


def owned_payload(value: Any, identity: Identity, constrained: bool = False) -> bool:
    """Require a forced member constraint or ownership on every structured account record.

    Generic order IDs/filters are not ownership. A wrapper containing one owned order and one
    ownerless/foreign order fails in full; it is never silently filtered into a partial answer.
    """
    if isinstance(value, dict):
        owners = [str(value[key]) for key in MEMBER_FIELDS.intersection(value)]
        if any(owner != identity.member_id for owner in owners):
            return False
        if constrained or owners:
            return all(owned_payload(child, identity, True) for child in value.values())
        children = [child for child in value.values() if isinstance(child, (dict, list))]
        # An unbounded aggregate/count is not established by finding one owned child record.
        metadata = {"ok", "status"}
        if any(
            key not in metadata and not isinstance(child, (dict, list))
            for key, child in value.items()
        ):
            return False
        return bool(children) and all(owned_payload(child, identity) for child in children)
    if isinstance(value, list):
        return (constrained or bool(value)) and all(
            owned_payload(child, identity, constrained) for child in value
        )
    return constrained


class Services:
    def __init__(self, settings: Settings, redis: Any):
        self.settings = settings
        self.redis = redis
        configured = settings.configured
        self.memory = Memory(settings) if configured["agent_memory"] else None
        self.playbooks = (
            AgentPlaybooks(
                AgentPlaybooksSettings(
                    url=settings.iris_agent_playbooks_url,
                    playbook_id=settings.iris_agent_playbooks_playbook_id,
                    api_key=settings.iris_agent_playbooks_api_key.get_secret_value(),
                )
            )
            if configured["agent_playbooks"]
            else None
        )
        self.router = (
            IntentRouter(
                IntentRouterSettings(
                    url=settings.iris_intent_router_url,
                    router_id=settings.iris_intent_router_id,
                    api_key=settings.iris_intent_router_api_key.get_secret_value(),
                )
            )
            if configured["intent_router"]
            else None
        )
        self.gateway = (
            ContextGateway(
                ContextGatewaySettings(
                    url=settings.iris_context_gateway_url,
                    api_key=settings.iris_context_gateway_api_key.get_secret_value(),
                )
            )
            if configured["context_gateway"]
            else None
        )
        self.model = None
        self.lens_enabled = False

    async def open(self) -> None:
        if self.settings.configured["gemini"]:
            from google import genai

            self.model = genai.Client(
                vertexai=True,
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
        if self.settings.configured["lens"]:
            self.lens_enabled = lens.init(
                endpoint=self.settings.iris_lens_endpoint,
                api_key=self.settings.iris_lens_api_key.get_secret_value(),
                agent_id=self.settings.iris_lens_agent_id,
                service_name="valuewholesale-onprem",
            )

    async def close(self) -> None:
        for client in (self.playbooks, self.router, self.gateway):
            if client is not None:
                await client.__aexit__(None, None, None)
        if self.memory:
            await self.memory.client.__aexit__(None, None, None)
        if self.model:
            await self.model.aio.aclose()
            self.model.close()
        if self.lens_enabled:
            lens.shutdown()

    async def tools(self, query: str = "shopping catalog inventory member orders") -> list:
        if not self.gateway:
            return []
        catalog = await self.gateway.atools()
        if {"search_tools", "run_tool"} <= {tool.name for tool in catalog}:
            catalog = await self.gateway.asearch(query, limit=10)
        return [tool for tool in catalog if read_tool(tool)]

    async def hybrid_products(
        self,
        identity: Identity,
        query: str,
        category: str | None,
        max_member_price: float | None,
        limit: int,
    ) -> dict:
        """Run the actual Product hybrid tool with structured predicates, without an LLM."""
        if self.gateway is None:
            raise RuntimeError("Context Gateway is not configured")
        original = "search_product_by_semantic_embedding_hybrid"
        catalog = await self.gateway.asearch(original, limit=20)
        matches = [
            tool for tool in catalog if original_tool_name(tool) == original and read_tool(tool)
        ]
        if len(matches) != 1:
            raise RuntimeError("A single read-only Product hybrid tool must be available")
        tool = matches[0]
        arguments: dict[str, Any] = {"query_text": query, "top_k": limit, "match_mode": "any"}
        if category:
            arguments["tag_conditions"] = [{"field": "category", "value": category}]
        if max_member_price is not None:
            arguments["numeric_conditions"] = [
                {"field": "member_price", "max_value": max_member_price}
            ]
        validate(arguments, tool.input_schema)
        response = await self.gateway_call(tool, arguments, identity)
        data = response.get("data")
        if not response.get("ok") or not isinstance(data, dict):
            raise RuntimeError("Product hybrid search failed")
        rows = data.get("results")
        if not isinstance(rows, list) or data.get("ranking_method") != "rrf":
            raise RuntimeError("Product hybrid search returned an unsupported result")
        products = []
        for rank, row in enumerate(rows[:limit], start=1):
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) for key in ("sku", "name", "category")
            ):
                raise RuntimeError("Product hybrid search returned a malformed product")
            price = row.get("member_price")
            if category and row["category"] != category:
                raise RuntimeError("Product hybrid search did not honor the category filter")
            if max_member_price is not None and (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(price)
                or price > max_member_price
            ):
                raise RuntimeError("Product hybrid search did not honor the price filter")
            # Return public display fields only, never vector payloads or result handles.
            item = {
                key: row.get(key)
                for key in ("sku", "name", "description", "category", "price", "member_price")
            }
            retrieval = row.get("_retrieval", {})
            item.update(
                hybrid_rank=rank,
                hybrid_score=retrieval.get("hybrid_score") if isinstance(retrieval, dict) else None,
            )
            products.append(item)
        return {
            "products": products,
            "count": len(products),
            "ranking_method": "rrf",
            "model_calls": 0,
        }

    async def cart(self, identity: Identity) -> dict:
        values = await self.redis.hgetall(f"{self.settings.onprem_namespace}:cart:{identity.owner}")
        return {
            "items": [{"sku": sku, "quantity": int(quantity)} for sku, quantity in values.items()],
            "note": "Demo shortlist only. Prices and inventory must be checked live.",
        }

    async def cart_add(self, identity: Identity, sku: str, quantity: int) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", sku) or not 1 <= quantity <= 20:
            raise ValueError("Use a discovered SKU and a quantity between 1 and 20")
        await self.redis.hset(
            f"{self.settings.onprem_namespace}:cart:{identity.owner}", sku, str(quantity)
        )
        return await self.cart(identity)

    async def gateway_call(self, tool: Any, arguments: dict, identity: Identity) -> dict:
        if not read_tool(tool):
            return {"ok": False, "error": "Use the turn-scoped operation for Gateway handles"}
        result = await self.gateway.acall(tool.name, member_arguments(tool, arguments, identity))
        if account_tool(tool):
            if result.is_error or not owned_payload(result.data, identity, member_bound(tool)):
                return {
                    "ok": False,
                    "data": None,
                    "text": "",
                    "error": "Account ownership could not be established for this member.",
                }
            # The independently supplied text might contain records absent from structured data.
            return {
                "ok": True,
                "data": result.data,
                "text": "",
                "upstream_id": result.upstream_id,
                "original_name": result.original_name,
            }
        return {
            "ok": not result.is_error,
            "data": result.data,
            "text": result.text,
            "upstream_id": result.upstream_id,
            "original_name": result.original_name,
        }


def tool_json(tool: Any) -> dict:
    return asdict(tool)


def collect_skus(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = {str(value["sku"])} if "sku" in value else set()
        for item in value.values():
            result.update(collect_skus(item))
        return result
    if isinstance(value, list):
        return set().union(*(collect_skus(item) for item in value)) if value else set()
    return set()


def bounded_result(data: Any, limit: int = 24000) -> dict:
    """Never silently truncate a result into invalid or misleading model context."""
    if len(json.dumps(data, default=str)) > limit:
        return {"ok": False, "error": "Result too large. Use a Gateway result handle and page it."}
    return data
