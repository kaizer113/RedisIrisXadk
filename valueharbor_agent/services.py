from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from array import array
from datetime import UTC, datetime
from typing import Any

import httpx
import redis

from valueharbor_agent.config import Settings, get_settings
from valueharbor_agent.demo_data import INVENTORY, MEMBERS, ORDERS, POLICIES, PRODUCTS, WAREHOUSES

log = logging.getLogger(__name__)

PUBLIC_POLICY_ROUTE = "public_stable_policy"
PUBLIC_POLICY_REFERENCES = [
    "What is the return policy?",
    "How long is the return window?",
    "Can an electronics purchase be returned?",
    "Explain the electronics returns rules.",
    "What does the product warranty cover?",
    "How does warranty coverage work?",
    "What are the curbside pickup rules?",
    "How long is the pickup window?",
    "What happens if a pickup is not collected?",
    "Explain the member pricing policy.",
    "How does membership pricing work?",
    "What are the general membership terms?",
]


def safe_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    return (cleaned or fallback)[:64]


class CatalogService:
    """Redis-backed product and policy retrieval with fixture fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis: redis.Redis | None = None
        if settings.redis_url:
            self.redis = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=False,
                socket_connect_timeout=4,
                socket_timeout=8,
                health_check_interval=30,
            )

    def _embed(self, text: str) -> bytes | None:
        if (
            not self.settings.valueharbor_vector_search_enabled
            or not self.settings.google_cloud_project
        ):
            return None
        try:
            from google import genai

            client = genai.Client(
                vertexai=True,
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
            response = client.models.embed_content(model="text-embedding-005", contents=text)
            return array("f", response.embeddings[0].values).tobytes()
        except Exception as exc:
            log.warning("Vertex embedding unavailable; using lexical retrieval: %s", exc)
            return None

    @staticmethod
    def _decode_map(values: dict[Any, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values.items():
            key = key.decode() if isinstance(key, bytes) else str(key)
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            result[key] = value
        return result

    @staticmethod
    def _escape_tag(value: str) -> str:
        return re.sub(r"([\\,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/ ])", r"\\\1", value)

    @classmethod
    def _search_result_maps(cls, raw: Any) -> list[dict[str, Any]]:
        """Normalize Redis 8 map replies and legacy FT.SEARCH array replies."""
        if isinstance(raw, dict):
            results = raw.get(b"results", raw.get("results", []))
            normalized = []
            for result in results:
                attributes = result.get(b"extra_attributes", result.get("extra_attributes", {}))
                normalized.append(cls._decode_map(attributes))
            return normalized

        normalized = []
        for index in range(2, len(raw), 2):
            values = raw[index]
            normalized.append(cls._decode_map(dict(zip(values[::2], values[1::2], strict=True))))
        return normalized

    def search_products(
        self, query: str, category: str = "", limit: int = 5
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 10))
        if self.redis is not None:
            try:
                vector = self._embed(query)
                category_filter = f"@category:{{{self._escape_tag(category)}}}" if category else "*"
                if vector:
                    raw = self.redis.execute_command(
                        "FT.SEARCH",
                        "idx:valueharbor:products",
                        f"({category_filter})=>[KNN {limit} @embedding $query_vector AS score]",
                        "PARAMS",
                        2,
                        "query_vector",
                        vector,
                        "RETURN",
                        7,
                        "sku",
                        "name",
                        "category",
                        "price",
                        "member_price",
                        "description",
                        "score",
                        "SORTBY",
                        "score",
                        "ASC",
                        "DIALECT",
                        2,
                    )
                else:
                    escaped = re.sub(r"[^A-Za-z0-9 ]", " ", query).strip()
                    terms = "|".join(escaped.split()) or "*"
                    text_query = f"(@name:({terms})|@description:({terms}))"
                    if category:
                        text_query += f" @category:{{{self._escape_tag(category)}}}"
                    raw = self.redis.execute_command(
                        "FT.SEARCH",
                        "idx:valueharbor:products",
                        text_query,
                        "RETURN",
                        6,
                        "sku",
                        "name",
                        "category",
                        "price",
                        "member_price",
                        "description",
                        "LIMIT",
                        0,
                        limit,
                        "DIALECT",
                        2,
                    )
                docs = []
                for mapped in self._search_result_maps(raw):
                    for field in ("price", "member_price", "score"):
                        if field in mapped:
                            mapped[field] = float(mapped[field])
                    docs.append(mapped)
                if docs:
                    return docs
            except Exception as exc:
                log.warning("Redis product search unavailable; using fixtures: %s", exc)

        words = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2}
        ranked: list[tuple[int, dict[str, Any]]] = []
        for product in PRODUCTS:
            if category and product["category"] != category:
                continue
            haystack = " ".join(
                [product["name"], product["description"], product["category"], *product["tags"]]
            ).lower()
            score = sum(1 for word in words if word in haystack)
            ranked.append((score, product))
        ranked.sort(key=lambda item: (-item[0], item[1]["member_price"]))
        return [dict(product) for _, product in ranked[:limit]]

    def search_policies(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked = sorted(
            POLICIES,
            key=lambda policy: (
                -sum(word in f"{policy['title']} {policy['content']}".lower() for word in words)
            ),
        )
        return ranked[: max(1, min(limit, 5))]

    def check_inventory(self, sku: str, warehouse_id: str) -> dict[str, Any]:
        warehouse_id = warehouse_id.lower()
        if self.redis is not None:
            try:
                quantity = self.redis.get(f"valueharbor:inventory:{warehouse_id}:{sku.upper()}")
                if quantity is not None:
                    qty = int(quantity)
                    return self._inventory_result(sku, warehouse_id, qty, "redis")
            except Exception as exc:
                log.warning("Redis inventory lookup unavailable; using fixtures: %s", exc)
        qty = INVENTORY.get(warehouse_id, {}).get(sku.upper())
        if qty is None:
            return {"found": False, "sku": sku.upper(), "warehouse_id": warehouse_id}
        return self._inventory_result(sku, warehouse_id, qty, "fixture")

    @staticmethod
    def _inventory_result(
        sku: str, warehouse_id: str, quantity: int, source: str
    ) -> dict[str, Any]:
        if quantity <= 0:
            availability = "out_of_stock"
        elif quantity <= 5:
            availability = "low_stock"
        else:
            availability = "in_stock"
        return {
            "found": True,
            "sku": sku.upper(),
            "warehouse_id": warehouse_id,
            "warehouse": WAREHOUSES.get(warehouse_id, {}).get("name", warehouse_id),
            "quantity": quantity,
            "availability": availability,
            "source": source,
        }

    def member_profile(self, member_id: str) -> dict[str, Any]:
        if self.redis is not None:
            try:
                raw = self.redis.hgetall(f"valueharbor:member:{safe_id(member_id, 'unknown')}")
                if raw:
                    profile = self._decode_map(raw)
                    profile["reward_balance"] = float(profile["reward_balance"])
                    profile["source"] = "redis"
                    return profile
            except Exception as exc:
                log.warning("Redis member lookup unavailable; using fixtures: %s", exc)
        return dict(MEMBERS.get(member_id, {"found": False, "member_id": member_id}))

    def recent_orders(self, member_id: str) -> list[dict[str, Any]]:
        if self.redis is not None:
            try:
                escaped_member_id = self._escape_tag(safe_id(member_id, "unknown"))
                raw = self.redis.execute_command(
                    "FT.SEARCH",
                    "idx:valueharbor:orders",
                    f"@member_id:{{{escaped_member_id}}}",
                    "RETURN",
                    8,
                    "order_id",
                    "placed_at",
                    "status",
                    "warehouse",
                    "fulfillment",
                    "total",
                    "item_count",
                    "member_id",
                    "LIMIT",
                    0,
                    20,
                    "DIALECT",
                    2,
                )
                orders = []
                for order in self._search_result_maps(raw):
                    order["total"] = float(order["total"])
                    order["item_count"] = int(order["item_count"])
                    order["items"] = self._order_items(order["order_id"])
                    order["source"] = "redis"
                    orders.append(order)
                orders.sort(key=lambda order: order["placed_at"], reverse=True)
                if orders:
                    return orders
            except Exception as exc:
                log.warning("Redis order lookup unavailable; using fixtures: %s", exc)
        return [dict(order) for order in ORDERS.get(member_id, [])]

    def _order_items(self, order_id: str) -> list[dict[str, Any]]:
        if self.redis is None:
            return []
        escaped_order_id = self._escape_tag(order_id)
        raw = self.redis.execute_command(
            "FT.SEARCH",
            "idx:valueharbor:order-items",
            f"@order_id:{{{escaped_order_id}}}",
            "RETURN",
            6,
            "order_item_id",
            "line_number",
            "sku",
            "product_name",
            "quantity",
            "unit_price",
            "LIMIT",
            0,
            50,
            "DIALECT",
            2,
        )
        items = []
        for item in self._search_result_maps(raw):
            item["line_number"] = int(item["line_number"])
            item["quantity"] = int(item["quantity"])
            item["unit_price"] = float(item["unit_price"])
            items.append(item)
        return sorted(items, key=lambda item: item["line_number"])


class CartService:
    def __init__(self, settings: Settings) -> None:
        self.redis = (
            redis.Redis.from_url(settings.redis_url, decode_responses=True)
            if settings.redis_url
            else None
        )
        self._fallback: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def add(self, member_id: str, sku: str, quantity: int) -> dict[str, Any]:
        quantity = max(1, min(quantity, 25))
        sku = sku.upper()
        if not any(product["sku"] == sku for product in PRODUCTS):
            return {"ok": False, "error": "unknown_sku", "sku": sku}
        if self.redis is not None:
            try:
                key = f"valueharbor:cart:{safe_id(member_id, 'anonymous')}"
                new_quantity = self.redis.hincrby(key, sku, quantity)
                self.redis.expire(key, 60 * 60 * 24 * 7)
                return {"ok": True, "sku": sku, "quantity": new_quantity, "source": "redis"}
            except Exception as exc:
                log.warning("Redis cart unavailable; using process-local cart: %s", exc)
        with self._lock:
            cart = self._fallback.setdefault(member_id, {})
            cart[sku] = cart.get(sku, 0) + quantity
            return {"ok": True, "sku": sku, "quantity": cart[sku], "source": "fixture"}

    def get(self, member_id: str) -> dict[str, int]:
        if self.redis is not None:
            try:
                return {
                    key: int(value)
                    for key, value in self.redis.hgetall(
                        f"valueharbor:cart:{safe_id(member_id, 'anonymous')}"
                    ).items()
                }
            except Exception as exc:
                log.warning("Redis cart unavailable; using process-local cart: %s", exc)
        return dict(self._fallback.get(member_id, {}))


class SemanticRouterService:
    """RedisVL cache-eligibility router with conservative deterministic guardrails."""

    _GUARDRAILS = (
        (
            "member-specific request",
            re.compile(
                r"\b(?:my|our)\s+(?:(?:pickup|recent|shopping|member)\s+)?"
                r"(?:order|cart|account|reward|preference|membership|purchase|profile|address)"
                r"\b|\bremember\b|\bi\s+prefer\b",
                re.IGNORECASE,
            ),
        ),
        (
            "live or time-sensitive commerce data",
            re.compile(
                r"\b(?:in stock|inventory|availability|available\s+(?:at|in)|current price|"
                r"today'?s price|warehouse stock|near me)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "sensitive data",
            re.compile(
                r"\b(?:password|passcode|payment|credit card|debit card|card number|"
                r"security code|phone number|email address)\b",
                re.IGNORECASE,
            ),
        ),
    )

    def __init__(
        self,
        settings: Settings,
        redis_client: redis.Redis | None = None,
    ) -> None:
        self.settings = settings
        self.redis = redis_client
        self.configured = bool(redis_client is not None and settings.semantic_router_configured)
        self._router: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def guardrail_reason(cls, message: str) -> str | None:
        for reason, pattern in cls._GUARDRAILS:
            if pattern.search(message):
                return reason
        return None

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        with self._lock:
            if self._router is not None:
                return self._router

            from google import genai
            from redisvl.extensions.router import Route, SemanticRouter
            from redisvl.utils.vectorize import CustomVectorizer

            embedding_client = genai.Client(
                vertexai=True,
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )

            def embed(content: str) -> list[float]:
                response = embedding_client.models.embed_content(
                    model=self.settings.valueharbor_semantic_router_embedding_model,
                    contents=content,
                )
                return list(response.embeddings[0].values)

            def embed_many(contents: list[str]) -> list[list[float]]:
                response = embedding_client.models.embed_content(
                    model=self.settings.valueharbor_semantic_router_embedding_model,
                    contents=contents,
                )
                return [list(item.values) for item in response.embeddings]

            vectorizer = CustomVectorizer(embed=embed, embed_many=embed_many)
            self._router = SemanticRouter(
                name=self.settings.valueharbor_semantic_router_index,
                routes=[
                    Route(
                        name=PUBLIC_POLICY_ROUTE,
                        references=PUBLIC_POLICY_REFERENCES,
                        distance_threshold=self.settings.valueharbor_semantic_router_threshold,
                        metadata={"cache": "public-policy", "safety": "public-only"},
                    )
                ],
                vectorizer=vectorizer,
                redis_client=self.redis,
                overwrite=False,
            )
        return self._router

    def route(self, message: str) -> dict[str, Any]:
        threshold = self.settings.valueharbor_semantic_router_threshold
        guardrail = self.guardrail_reason(message)
        if guardrail:
            return {
                "configured": self.configured,
                "eligible": False,
                "route": None,
                "distance": None,
                "threshold": threshold,
                "reason": guardrail,
                "decision_source": "guardrail",
            }
        if not self.configured:
            return {
                "configured": False,
                "eligible": False,
                "route": None,
                "distance": None,
                "threshold": threshold,
                "reason": "semantic router is not configured",
                "decision_source": "fail-safe",
            }

        try:
            match = self._get_router()(message)
            route_name = getattr(match, "name", None)
            distance = getattr(match, "distance", None)
            eligible = route_name == PUBLIC_POLICY_ROUTE
            return {
                "configured": True,
                "eligible": eligible,
                "route": route_name,
                "distance": round(float(distance), 4) if distance is not None else None,
                "threshold": threshold,
                "reason": "stable public policy route" if eligible else "no safe semantic route",
                "decision_source": "redisvl",
            }
        except Exception as exc:
            log.warning("RedisVL semantic routing failed closed: %s", exc)
            return {
                "configured": True,
                "eligible": False,
                "route": None,
                "distance": None,
                "threshold": threshold,
                "reason": "semantic router unavailable",
                "decision_source": "fail-safe",
            }


class LangCacheService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = (
            f"{settings.langcache_host.rstrip('/')}/v1/caches/{settings.langcache_cache_id}"
            if settings.langcache_configured
            else ""
        )

    async def search(self, prompt: str, _scope: str) -> dict[str, Any] | None:
        if not self.base_url:
            return None
        body = {
            "prompt": prompt,
            "similarityThreshold": self.settings.langcache_similarity_threshold,
            "searchStrategies": ["semantic"],
        }
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.post(
                    f"{self.base_url}/entries/search",
                    headers={"Authorization": f"Bearer {self.settings.langcache_api_key}"},
                    json=body,
                )
                response.raise_for_status()
            entries = response.json().get("data", [])
            return entries[0] if entries else None
        except Exception as exc:
            log.warning("LangCache search failed open: %s", exc)
            return None

    async def store(self, prompt: str, answer: str, _scope: str) -> bool:
        if not self.base_url:
            return False
        body = {
            "prompt": prompt,
            "response": answer,
        }
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.post(
                    f"{self.base_url}/entries",
                    headers={"Authorization": f"Bearer {self.settings.langcache_api_key}"},
                    json=body,
                )
                response.raise_for_status()
            return True
        except Exception as exc:
            log.warning("LangCache store failed open: %s", exc)
            return False


class MemoryService:
    """Official Redis Agent Memory SDK adapter, scoped by member and namespace."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any | None = None
        self.models: Any | None = None
        if settings.memory_configured:
            try:
                from redis_agent_memory import AgentMemory, models

                self.client = AgentMemory(
                    settings.agent_memory_base_url,
                    store_id=settings.agent_memory_store_id,
                    api_key=settings.agent_memory_api_key,
                )
                self.models = models
            except Exception as exc:
                log.warning("Agent Memory SDK initialization failed: %s", exc)

    def add_event(self, member_id: str, session_id: str, role: str, text: str) -> bool:
        if self.client is None or self.models is None:
            return False
        role_enum = getattr(self.models.MessageRole, role.upper())
        try:
            self.client.add_session_event(
                session_id=safe_id(session_id, "shopping-session"),
                actor_id=safe_id(
                    member_id if role.upper() == "USER" else "valueharbor-agent", "actor"
                ),
                role=role_enum,
                content=[{"text": text}],
                created_at=datetime.now(UTC),
                metadata={"channel": "web", "agent": "valueharbor-shopping"},
            )
            return True
        except Exception as exc:
            log.warning("Agent Memory event write failed open: %s", exc)
            return False

    def short_term(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent Redis Agent Memory session events."""
        if self.client is None:
            return []
        try:
            response = self.client.get_session_memory(
                session_id=safe_id(session_id, "shopping-session"),
                include_summarised_events=True,
            )
            events = list(getattr(response, "events", []) or [])[-max(1, min(limit, 20)) :]
            return [
                event.model_dump(mode="json")
                if hasattr(event, "model_dump")
                else dict(event)
                for event in events
            ]
        except Exception as exc:
            # A new browser session has no server-side session record yet.
            if "404" not in str(exc):
                log.warning("Agent Memory session retrieval failed open: %s", exc)
            return []

    def recall(self, member_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self.client is None or self.models is None:
            return []
        try:
            response = self.client.search_long_term_memory(
                request={
                    "text": query,
                    "similarity_threshold": self.settings.agent_memory_similarity_threshold,
                    "filter_op": self.models.FilterConjunction.ALL,
                    "filter_": {
                        "owner_id": {"eq": safe_id(member_id, "anonymous")},
                        "namespace": {"eq": self.settings.agent_memory_namespace},
                        "memory_type": {"in_": ["semantic", "episodic"]},
                    },
                    "limit": max(1, min(limit, 10)),
                },
            )
            return [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
                for item in response.items
            ]
        except Exception as exc:
            log.warning("Agent Memory search failed open: %s", exc)
            return []

    def remember(self, member_id: str, fact: str, topics: list[str] | None = None) -> bool:
        if self.client is None or self.models is None:
            return False
        fact_digest = hashlib.sha256(fact.encode("utf-8")).hexdigest()[:16]
        memory_id = safe_id(f"{member_id}-{fact_digest}", "memory")
        try:
            self.client.bulk_create_long_term_memories(
                memories=[
                    {
                        "id": memory_id,
                        "text": fact,
                        "memory_type": "semantic",
                        "owner_id": safe_id(member_id, "anonymous"),
                        "namespace": self.settings.agent_memory_namespace,
                        "topics": topics or ["shopping", "preference"],
                    }
                ]
            )
            return True
        except Exception as exc:
            log.warning("Agent Memory direct write failed open: %s", exc)
            return False


class ContextRetrieverService:
    def __init__(self, settings: Settings) -> None:
        self.agent_key = settings.mcp_agent_key

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self.agent_key:
            return []
        try:
            from context_surfaces import UnifiedClient

            async with UnifiedClient() as client:
                tools = await client.list_tools(self.agent_key)
            return [tool if isinstance(tool, dict) else tool.model_dump() for tool in tools]
        except Exception as exc:
            log.warning("Context Retriever tool listing failed open: %s", exc)
            return []

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.agent_key:
            return {"ok": False, "error": "context_retriever_not_configured"}
        try:
            from context_surfaces import UnifiedClient

            async with UnifiedClient() as client:
                raw = await client.query_tool(
                    agent_key=self.agent_key,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            if isinstance(raw, dict):
                content = raw.get("content", [])
                if content and content[0].get("type") == "text":
                    return json.loads(content[0].get("text", "{}"))
                return raw
            return {"result": str(raw)}
        except Exception as exc:
            log.warning("Context Retriever call failed open: %s", exc)
            return {"ok": False, "error": str(exc)}


class VertexMemoryService:
    """ADK wrapper for Vertex AI Agent Engine Memory Bank."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any | None = None
        if settings.vertex_memory_configured:
            try:
                from google.adk.memory import VertexAiMemoryBankService

                self.client = VertexAiMemoryBankService(
                    project=settings.google_cloud_project,
                    location=settings.google_memory_location,
                    agent_engine_id=settings.google_agent_engine_id,
                )
            except Exception as exc:
                log.warning("Vertex Memory Bank initialization failed: %s", exc)

    async def recall(self, member_id: str, query: str) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        try:
            response = await self.client.search_memory(
                app_name="valueharbor-shopping-agent",
                user_id=safe_id(member_id, "anonymous"),
                query=query,
            )
            raw_memories = getattr(response, "memories", None)
            if raw_memories is None and isinstance(response, list):
                raw_memories = response
            if raw_memories is None:
                raw_memories = getattr(response, "results", [])
            return [self._serialize(item) for item in raw_memories]
        except Exception as exc:
            log.warning("Vertex Memory Bank search failed open: %s", exc)
            return []

    @staticmethod
    def _serialize(item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if hasattr(item, "to_dict"):
            return item.to_dict()
        if isinstance(item, dict):
            return item
        return {"text": str(item)}


def _memory_text(item: Any) -> str:
    """Extract a readable fact/event from either managed memory provider."""
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return " ".join(filter(None, (_memory_text(value) for value in item))).strip()
    if isinstance(item, dict):
        for key in ("text", "fact"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("memory", "content", "parts"):
            if key in item:
                nested = _memory_text(item[key])
                if nested:
                    return nested
    return ""


def memory_snippets(memories: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Return compact, non-empty memory facts for API traces and agent state."""
    snippets = []
    for memory in memories:
        text = _memory_text(memory)
        if text:
            snippets.append(text[:500])
        if len(snippets) >= limit:
            break
    return snippets


def _retrieval_quality(
    memories: list[dict[str, Any]], expected_terms: list[str]
) -> dict[str, float | None]:
    if not expected_terms:
        return {"precision_at_k": None, "recall_at_k": None}
    expected = [term.lower().strip() for term in expected_terms if term.strip()]
    texts = [_memory_text(memory).lower() for memory in memories]
    relevant_results = sum(any(term in text for term in expected) for text in texts)
    matched_terms = sum(any(term in text for text in texts) for term in expected)
    return {
        "precision_at_k": round(relevant_results / len(texts), 3) if texts else 0.0,
        "recall_at_k": round(matched_terms / len(expected), 3) if expected else None,
    }


async def compare_memory_retrieval(
    query: str,
    member_id: str,
    expected_terms: list[str] | None = None,
) -> dict[str, Any]:
    expected_terms = expected_terms or []

    async def redis_search() -> dict[str, Any]:
        started = time.perf_counter()
        memories = await asyncio.to_thread(services.memory.recall, member_id, query, 5)
        return {
            "provider": "redis_agent_memory",
            "configured": services.memory.client is not None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "count": len(memories),
            "memories": memories,
            **_retrieval_quality(memories, expected_terms),
        }

    async def vertex_search() -> dict[str, Any]:
        started = time.perf_counter()
        memories = await services.vertex_memory.recall(member_id, query)
        return {
            "provider": "vertex_adk_memory_bank",
            "configured": services.vertex_memory.client is not None,
            "location": services.settings.google_memory_location,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "count": len(memories),
            "memories": memories,
            **_retrieval_quality(memories, expected_terms),
        }

    redis_result, vertex_result = await asyncio.gather(redis_search(), vertex_search())
    return {
        "query": query,
        "member_id": member_id,
        "expected_terms": expected_terms,
        "note": (
            "Precision/recall are computed only when expected_terms are supplied. "
            "Latency is client-observed wall time; compare medians across repeated runs."
        ),
        "results": [redis_result, vertex_result],
    }


class Services:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = CatalogService(self.settings)
        self.cart = CartService(self.settings)
        self.semantic_router = SemanticRouterService(self.settings, self.catalog.redis)
        self.langcache = LangCacheService(self.settings)
        self.memory = MemoryService(self.settings)
        self.vertex_memory = VertexMemoryService(self.settings)
        self.context = ContextRetrieverService(self.settings)


services = Services()
