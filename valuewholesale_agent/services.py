from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import re
import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import redis
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.index import SearchIndex
from redisvl.query import TextQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import HFTextVectorizer

from valuewholesale_agent.config import Settings, get_settings
from valuewholesale_agent.demo_data import (
    INVENTORY,
    MEMBERS,
    ORDERS,
    POLICIES,
    PRODUCTS,
    WAREHOUSES,
)

log = logging.getLogger(__name__)

PRODUCT_INDEX_NAME = "idx:valuewholesale:products-v2"
POLICY_INDEX_NAME = "idx:valuewholesale:policies-v2"
LOCAL_EMBEDDING_DIMS = 384
EMBEDDING_CACHE_NAME = "valuewholesale-embeddings-v1"
EMBEDDING_WARMUP_TEXT = "Warm the Value Wholesale semantic embedding model."
AGENT_MEMORY_MAX_CONNECTIONS = 40
AGENT_MEMORY_MAX_KEEPALIVE_CONNECTIONS = 20
TOOL_CALL_CACHE_METADATA_KEY = "_tool_call_cache"
VERTEX_MEMORY_APP_NAME = "valuewholesale-shopping-agent"
MEMORY_INVENTORY_LIMIT = 20

REDIS_KEEPALIVE_OPTIONS = {
    option: value
    for option, value in (
        (getattr(socket, "TCP_KEEPIDLE", None), 60),
        (getattr(socket, "TCP_KEEPINTVL", None), 15),
        (getattr(socket, "TCP_KEEPCNT", None), 4),
    )
    if option is not None
}
REDIS_CONNECTION_KWARGS: dict[str, Any] = {
    "socket_connect_timeout": 4,
    "socket_timeout": 8,
    "health_check_interval": 30,
    "socket_keepalive": True,
    "socket_keepalive_options": REDIS_KEEPALIVE_OPTIONS,
}


def call_with_timing(
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Measure a synchronous operation inside the thread that executes it."""
    started = time.perf_counter()
    result = operation(*args, **kwargs)
    return result, round((time.perf_counter() - started) * 1000, 2)

PUBLIC_POLICY_ROUTE = "reusable_ecommerce"
PRODUCT_EDUCATION_ROUTE = "product_education"
SHOPPING_GUIDE_ROUTE = "shopping_guide"
LANGCACHE_SCOPES = {
    PUBLIC_POLICY_ROUTE: "policy:v1",
    PRODUCT_EDUCATION_ROUTE: "product-education:catalog-v1",
    SHOPPING_GUIDE_ROUTE: "shopping-guide:v1",
}
PUBLIC_POLICY_REFERENCES = [
    "What is the return policy?",
    "What is the electronics return policy?",
    "How long is the return window?",
    "Can an electronics purchase be returned?",
    "Can I return a television after buying it?",
    "Can I return an item without a receipt?",
    "Does the return policy require a receipt?",
    "What if I no longer have the receipt?",
    "Explain the electronics returns rules.",
    "What does the product warranty cover?",
    "How does warranty coverage work?",
    "What are the curbside pickup rules?",
    "How long is the pickup window?",
    "How long will Value Wholesale hold a pickup order after it is ready?",
    "What happens if a pickup is not collected?",
    "Explain the member pricing policy.",
    "How does membership pricing work?",
    "What are the general membership terms?",
    "What payment methods can members use?",
    "How does curbside pickup work?",
    "Do products include a manufacturer warranty?",
]
PRODUCT_EDUCATION_REFERENCES = [
    "Tell me about Rain City Medium Roast Coffee.",
    "What flavor notes does your whole-bean medium roast coffee have?",
    "What are the features of North Trail Organic Oats?",
    "Explain what free-and-clear laundry detergent means.",
    "Tell me about the Harbor Select extra virgin olive oil.",
    "What features does the SummitBook laptop have?",
    "Compare the static features of two products without checking price or stock.",
]
SHOPPING_GUIDE_REFERENCES = [
    "How should I store bulk rolled oats after opening?",
    "What is the best way to keep a large bag of oats fresh?",
    "How do I care for and store bulk pantry products?",
    "How should individually portioned salmon be frozen?",
    "How much pasta should I prepare for twenty people?",
    "Give me a reusable checklist for planning a pantry restock.",
    "How should I organize bulk household supplies?",
]
ECOMMERCE_ROUTE = "ecommerce_request"
ECOMMERCE_REFERENCES = [
    "I want to buy dish soap.",
    "I need to buy a household product.",
    "Help me purchase an item from your catalog.",
    "Find family-size pantry staples under thirty dollars.",
    "Recommend products for my household.",
    "Recommend a good pasta from your catalog.",
    "What pasta products do you sell?",
    "What food and grocery products do you carry?",
    "Help me choose a product that you sell.",
    "Compare these products and prices.",
    "Is this item available at my warehouse?",
    "Check current inventory in Portland.",
    "Add this product to my shopping cart.",
    "Show my recent order and pickup status.",
    "Give me an account overview and tell me if I have anything to pick up.",
    "What household products have I bought in the past?",
    "What products match my preferences?",
    "Help me plan a warehouse shopping trip.",
    "Which membership deal offers the best value?",
    "Do you sell fragrance-free detergent?",
    "What groceries should I buy for a large family?",
]
OUT_OF_DOMAIN_ROUTE = "blocked_out_of_domain"
OUT_OF_DOMAIN_REFERENCES = [
    "Where is Dagestan?",
    "What is the capital of France?",
    "Who won the football game?",
    "Write Python code for me.",
    "Explain quantum physics.",
    "What is the weather today?",
    "Tell me about world history.",
    "Solve this mathematics problem.",
    "Give me medical advice.",
    "Who should I vote for?",
    "Write a poem about the ocean.",
    "Tell me a joke.",
]


def safe_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    return (cleaned or fallback)[:64]


class LocalEmbeddingService:
    """One lazily loaded local embedding model shared by routing and catalog search."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vectorizer: HFTextVectorizer | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.embedding_cache = (
            EmbeddingsCache(
                name=EMBEDDING_CACHE_NAME,
                ttl=settings.valuewholesale_embedding_cache_ttl_seconds,
                redis_url=settings.redis_url,
                connection_kwargs=REDIS_CONNECTION_KWARGS,
            )
            if settings.redis_url
            else None
        )

    @property
    def loaded(self) -> bool:
        return self._vectorizer is not None

    def get_vectorizer(self) -> HFTextVectorizer:
        if self._vectorizer is None:
            with self._lock:
                if self._vectorizer is None:
                    vectorizer = HFTextVectorizer(
                        model=self.settings.valuewholesale_embedding_model,
                        dtype="float32",
                        cache=self.embedding_cache,
                        device=self.settings.valuewholesale_embedding_device,
                        model_kwargs={"dtype": "float32"},
                    )
                    if vectorizer.dims != LOCAL_EMBEDDING_DIMS:
                        raise ValueError(
                            f"Embedding model returned {vectorizer.dims} dimensions; "
                            f"expected {LOCAL_EMBEDDING_DIMS}"
                        )
                    self._vectorizer = vectorizer
        return self._vectorizer

    def embed(self, text: str, *, as_buffer: bool = False) -> list[float] | bytes:
        vectorizer = self.get_vectorizer()
        with self._inference_lock:
            return vectorizer.embed(text, as_buffer=as_buffer)

    def embed_many(
        self, texts: list[str], *, as_buffer: bool = False
    ) -> list[list[float]] | list[bytes]:
        vectorizer = self.get_vectorizer()
        with self._inference_lock:
            return vectorizer.embed_many(texts, as_buffer=as_buffer)

    def is_cached(self, text: str) -> bool | None:
        """Report an embedding-cache hit when the cache can be inspected."""
        if self.embedding_cache is None:
            return None
        try:
            return bool(
                self.embedding_cache.exists(
                    content=text,
                    model_name=self.settings.valuewholesale_embedding_model,
                )
            )
        except Exception as exc:
            log.warning("Embedding cache probe failed open: %s", exc)
            return None

    def cache_probe(self) -> tuple[bool, str, dict[str, Any]]:
        if self.embedding_cache is None:
            return False, "RedisVL EmbeddingsCache is not configured", {}
        vector = self.embed(EMBEDDING_WARMUP_TEXT)
        cached = self.embedding_cache.exists(
            content=EMBEDDING_WARMUP_TEXT,
            model_name=self.settings.valuewholesale_embedding_model,
        )
        return (
            cached,
            "RedisVL EmbeddingsCache ready" if cached else "Embedding was not cached",
            {
                "model": self.settings.valuewholesale_embedding_model,
                "dimensions": len(vector),
                "ttl_seconds": self.settings.valuewholesale_embedding_cache_ttl_seconds,
                "cache_name": EMBEDDING_CACHE_NAME,
            },
        )

    def warmup(self) -> dict[str, Any]:
        started = time.perf_counter()
        vector = self.embed(EMBEDDING_WARMUP_TEXT)
        return {
            "model": self.settings.valuewholesale_embedding_model,
            "dimensions": len(vector),
            "device": self.settings.valuewholesale_embedding_device,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "cache": EMBEDDING_CACHE_NAME if self.embedding_cache else None,
        }


class ToolCallCache:
    """Session-scoped exact-match cache for successful read-only tool results."""

    KEY_PREFIX = "tool-cache"
    SESSION_INDEX_PREFIX = "tool-cache-session"
    SCHEMA_VERSION = "v2"

    def __init__(self, settings: Settings, client: redis.Redis | None) -> None:
        self.settings = settings
        self.redis = client

    @staticmethod
    def _canonical_arguments(arguments: dict[str, Any]) -> str:
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)

    def key(
        self,
        owner_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        owner = safe_id(owner_id, "anonymous")
        session = safe_id(session_id, "shopping-session")
        material = "\n".join(
            (
                self.SCHEMA_VERSION,
                owner,
                session,
                tool_name.strip().lower(),
                self._canonical_arguments(arguments),
            )
        )
        digest = hashlib.sha256(material.encode()).hexdigest()
        return f"{self.KEY_PREFIX}:{digest}"

    def _session_index_key(self, owner_id: str, session_id: str) -> str:
        owner = safe_id(owner_id, "anonymous")
        session = safe_id(session_id, "shopping-session")
        digest = hashlib.sha256(f"{owner}\n{session}".encode()).hexdigest()
        return f"{self.SESSION_INDEX_PREFIX}:{digest}"

    def get(
        self,
        owner_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, float]:
        """Return a cached result and client-observed Redis read latency."""
        started = time.perf_counter()
        if self.redis is None:
            return None, round((time.perf_counter() - started) * 1000, 2)
        try:
            result = self.redis.json().get(
                self.key(owner_id, session_id, tool_name, arguments)
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            if result is None:
                return None, duration_ms
            return (result if isinstance(result, dict) else None), duration_ms
        except Exception as exc:
            log.warning("Tool call cache read failed open: %s", exc)
            return None, round((time.perf_counter() - started) * 1000, 2)

    def set(
        self,
        owner_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        if self.redis is None:
            return False
        cache_value = json.loads(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key != TOOL_CALL_CACHE_METADATA_KEY
                },
                default=str,
            )
        )
        cache_key = self.key(owner_id, session_id, tool_name, arguments)
        session_index_key = self._session_index_key(owner_id, session_id)
        ttl = self.settings.valuewholesale_tool_cache_ttl_seconds
        try:
            pipeline = self.redis.pipeline(transaction=True)
            pipeline.json().set(cache_key, "$", cache_value)
            pipeline.expire(cache_key, ttl)
            pipeline.sadd(session_index_key, cache_key)
            pipeline.expire(session_index_key, ttl)
            results = pipeline.execute()
            return bool(results[0])
        except Exception as exc:
            log.warning("Tool call cache write failed open: %s", exc)
            return False

    def clear_session(self, owner_id: str, session_id: str) -> int:
        """Invalidate cached reads after a successful session mutation."""
        if self.redis is None:
            return 0
        session_index_key = self._session_index_key(owner_id, session_id)
        try:
            keys = list(self.redis.smembers(session_index_key))
            if not keys:
                self.redis.delete(session_index_key)
                return 0
            deleted = int(self.redis.delete(*keys))
            self.redis.delete(session_index_key)
            return deleted
        except Exception as exc:
            log.warning("Tool call cache invalidation failed open: %s", exc)
            return 0


class CatalogService:
    """Redis-backed product and policy retrieval with fixture fallback."""

    def __init__(
        self,
        settings: Settings,
        embeddings: LocalEmbeddingService | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings or LocalEmbeddingService(settings)
        self.redis: redis.Redis | None = None
        self._product_index: SearchIndex | None = None
        self._policy_index: SearchIndex | None = None
        self._product_index_lock = threading.Lock()
        self._policy_index_lock = threading.Lock()
        if settings.redis_url:
            self.redis = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=False,
                **REDIS_CONNECTION_KWARGS,
            )

    def ping(self) -> bool:
        """Perform a real round trip to the configured Redis database."""
        return bool(self.redis is not None and self.redis.ping())

    @staticmethod
    def product_embedding_text(product: dict[str, Any]) -> str:
        """Build a compact product representation for local semantic retrieval."""
        tags = ", ".join(str(tag) for tag in product.get("tags", []))
        description = str(product["description"]).rstrip(". ")
        return (
            f"{product['name']}. {description}. "
            f"Category: {product['category']}. Keywords: {tags}."
        )

    @staticmethod
    def policy_embedding_text(policy: dict[str, Any]) -> str:
        """Build the authoritative policy text used for semantic retrieval."""
        return f"{policy['title']}. {str(policy['content']).rstrip('. ')}."

    def _get_product_index(self) -> SearchIndex:
        """Lazily bind RedisVL to the checked-in product index."""
        if self.redis is None:
            raise RuntimeError("Redis is not configured")
        if self._product_index is None:
            with self._product_index_lock:
                if self._product_index is None:
                    self._product_index = SearchIndex.from_existing(
                        PRODUCT_INDEX_NAME,
                        redis_client=self.redis,
                    )
        return self._product_index

    def _get_policy_index(self) -> SearchIndex:
        """Lazily bind RedisVL to the versioned policy vector index."""
        if self.redis is None:
            raise RuntimeError("Redis is not configured")
        if self._policy_index is None:
            with self._policy_index_lock:
                if self._policy_index is None:
                    self._policy_index = SearchIndex.from_existing(
                        POLICY_INDEX_NAME,
                        redis_client=self.redis,
                    )
        return self._policy_index

    def _embed(self, text: str) -> bytes | None:
        if not self.settings.valuewholesale_vector_search_enabled:
            return None
        try:
            embedding = self.embeddings.embed(text, as_buffer=True)
            assert isinstance(embedding, bytes)
            return embedding
        except Exception as exc:
            log.warning("Local embedding unavailable; using lexical retrieval: %s", exc)
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
        products, _, _, _ = self.search_products_with_timing(query, category, limit)
        return products

    def search_products_with_timing(
        self, query: str, category: str = "", limit: int = 5
    ) -> tuple[list[dict[str, Any]], float | None, float | None, bool | None]:
        """Search products with separate RedisVL and embedding telemetry."""
        limit = max(1, min(limit, 10))
        redisvl_duration_ms: float | None = None
        embedding_duration_ms: float | None = None
        embedding_cache_hit: bool | None = None
        if self.redis is not None:
            try:
                embedding_started = time.perf_counter()
                if self.settings.valuewholesale_vector_search_enabled:
                    cache_probe = getattr(self.embeddings, "is_cached", None)
                    if callable(cache_probe):
                        embedding_cache_hit = cache_probe(query)
                vector = self._embed(query)
                if self.settings.valuewholesale_vector_search_enabled:
                    embedding_duration_ms = round(
                        (time.perf_counter() - embedding_started) * 1000, 2
                    )
                category_filter = Tag("category") == category if category else None
                return_fields = [
                    "sku",
                    "name",
                    "category",
                    "price",
                    "member_price",
                    "description",
                ]
                if vector:
                    redisvl_query = VectorQuery(
                        vector=vector,
                        vector_field_name="embedding",
                        filter_expression=category_filter,
                        return_fields=return_fields,
                        num_results=limit,
                        return_score=True,
                    )
                else:
                    redisvl_query = TextQuery(
                        text=query,
                        text_field_name={"name": 2.0, "description": 1.0},
                        filter_expression=category_filter,
                        return_fields=return_fields,
                        num_results=limit,
                        return_score=False,
                        stopwords=None,
                    )
                product_index = self._get_product_index()
                redisvl_started = time.perf_counter()
                raw_docs = product_index.query(redisvl_query)
                redisvl_duration_ms = round(
                    (time.perf_counter() - redisvl_started) * 1000, 2
                )
                docs = []
                for mapped in raw_docs:
                    mapped.pop("id", None)
                    if "vector_distance" in mapped:
                        mapped["score"] = mapped.pop("vector_distance")
                    for field in ("price", "member_price", "score"):
                        if field in mapped:
                            mapped[field] = float(mapped[field])
                    docs.append(mapped)
                if docs:
                    return (
                        docs,
                        redisvl_duration_ms,
                        embedding_duration_ms,
                        embedding_cache_hit,
                    )
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
        return (
            [dict(product) for _, product in ranked[:limit]],
            redisvl_duration_ms,
            embedding_duration_ms,
            embedding_cache_hit,
        )

    def search_policies(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 5))
        if self.redis is not None:
            try:
                vector = self._embed(query)
                return_fields = ["title", "content"]
                if vector:
                    redisvl_query = VectorQuery(
                        vector=vector,
                        vector_field_name="embedding",
                        return_fields=return_fields,
                        num_results=limit,
                        return_score=True,
                    )
                else:
                    redisvl_query = TextQuery(
                        text=query,
                        text_field_name={"title": 2.0, "content": 1.0},
                        return_fields=return_fields,
                        num_results=limit,
                        return_score=False,
                        stopwords=None,
                    )
                docs = []
                for mapped in self._get_policy_index().query(redisvl_query):
                    mapped.pop("id", None)
                    if "vector_distance" in mapped:
                        mapped["score"] = float(mapped.pop("vector_distance"))
                    docs.append(mapped)
                if docs:
                    return docs
            except Exception as exc:
                log.warning("Redis policy search unavailable; using fixtures: %s", exc)

        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked = sorted(
            POLICIES,
            key=lambda policy: (
                -sum(word in f"{policy['title']} {policy['content']}".lower() for word in words)
            ),
        )
        return [dict(policy) for policy in ranked[:limit]]

    def check_inventory(self, sku: str, warehouse_id: str) -> dict[str, Any]:
        warehouse_id = warehouse_id.lower()
        if self.redis is not None:
            try:
                quantity = self.redis.get(f"valuewholesale:inventory:{warehouse_id}:{sku.upper()}")
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
                raw = self.redis.hgetall(f"valuewholesale:member:{safe_id(member_id, 'unknown')}")
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
                    "idx:valuewholesale:orders",
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
            "idx:valuewholesale:order-items",
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
            redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                **REDIS_CONNECTION_KWARGS,
            )
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
                key = f"valuewholesale:cart:{safe_id(member_id, 'anonymous')}"
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
                        f"valuewholesale:cart:{safe_id(member_id, 'anonymous')}"
                    ).items()
                }
            except Exception as exc:
                log.warning("Redis cart unavailable; using process-local cart: %s", exc)
        return dict(self._fallback.get(member_id, {}))


class SemanticRouterService:
    """RedisVL domain and cache-policy router with deterministic guardrails."""

    _GUARDRAILS = (
        (
            "member-specific request",
            re.compile(
                r"\b(?:my|our)\s+(?:(?:pickup|recent|shopping|member)\s+)?"
                r"(?:orders?|cart|account|rewards?|preferences?|membership|purchases?|profile|address)"
                r"\b|\bwho\s+am\s+i\b|"
                r"\bwhat\s+do\s+you\s+(?:know|remember)\s+about\s+me\b|"
                r"\bdo\s+i\s+have\s+(?:(?:a|any)\s+)?"
                r"(?:(?:recent|open|ready)\s+)?orders?\b|"
                r"\bremember\b|\bi\s+prefer\b",
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
    _CONTEXTUAL_FOLLOWUP = re.compile(
        r"^\s*(?:and\b|also\b|but\b|even\b|what\s+about\b|how\s+about\b|"
        r"what\s+if\b|without\b|with\b|does\s+(?:that|it)\b|"
        r"is\s+(?:that|it)\b|can\s+(?:that|it)\b)",
        re.IGNORECASE,
    )
    _EXPLICIT_SHOPPING_INTENT = re.compile(
        r"\b(?:i\s+(?:want|need|would\s+like)\s+to\s+|help\s+me\s+)"
        r"(?:buy|purchase|shop\s+for|order|get)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        settings: Settings,
        redis_client: redis.Redis | None = None,
        embeddings: LocalEmbeddingService | None = None,
    ) -> None:
        self.settings = settings
        self.redis = redis_client
        self.embeddings = embeddings or LocalEmbeddingService(settings)
        self.configured = bool(redis_client is not None and settings.semantic_router_configured)
        self._router: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def guardrail_reason(cls, message: str) -> str | None:
        for reason, pattern in cls._GUARDRAILS:
            if pattern.search(message):
                return reason
        return None

    @classmethod
    def is_contextual_followup(cls, message: str) -> bool:
        """Identify short utterances that need recent session context to be routed."""
        return bool(
            len(message.split()) <= 12 and cls._CONTEXTUAL_FOLLOWUP.search(message)
        )

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        with self._lock:
            if self._router is not None:
                return self._router

            from redisvl.extensions.router import Route, SemanticRouter

            vectorizer = self.embeddings.get_vectorizer()
            self._router = SemanticRouter(
                name=self.settings.valuewholesale_semantic_router_index,
                routes=[
                    Route(
                        name=PUBLIC_POLICY_ROUTE,
                        references=PUBLIC_POLICY_REFERENCES,
                        distance_threshold=self.settings.valuewholesale_semantic_router_threshold,
                        metadata={"action": "allow", "cache_read": True, "cache_write": True},
                    ),
                    Route(
                        name=ECOMMERCE_ROUTE,
                        references=ECOMMERCE_REFERENCES,
                        distance_threshold=self.settings.valuewholesale_semantic_router_threshold,
                        metadata={"action": "allow", "cache_read": False, "cache_write": False},
                    ),
                    Route(
                        name=PRODUCT_EDUCATION_ROUTE,
                        references=PRODUCT_EDUCATION_REFERENCES,
                        distance_threshold=self.settings.valuewholesale_semantic_router_threshold,
                        metadata={"action": "allow", "cache_read": True, "cache_write": True},
                    ),
                    Route(
                        name=SHOPPING_GUIDE_ROUTE,
                        references=SHOPPING_GUIDE_REFERENCES,
                        distance_threshold=self.settings.valuewholesale_semantic_router_threshold,
                        metadata={"action": "allow", "cache_read": True, "cache_write": True},
                    ),
                    Route(
                        name=OUT_OF_DOMAIN_ROUTE,
                        references=OUT_OF_DOMAIN_REFERENCES,
                        distance_threshold=self.settings.valuewholesale_semantic_router_threshold,
                        metadata={"action": "block", "cache_read": False, "cache_write": False},
                    ),
                ],
                vectorizer=vectorizer,
                redis_client=self.redis,
                overwrite=False,
            )
        return self._router

    def route(self, message: str, recent_context: str = "") -> dict[str, Any]:
        threshold = self.settings.valuewholesale_semantic_router_threshold
        guardrail = self.guardrail_reason(message)
        if guardrail:
            return {
                "configured": self.configured,
                "eligible": False,
                "cache_read": False,
                "cache_write": False,
                "blocked": False,
                "action": "allow",
                "route": None,
                "distance": None,
                "threshold": threshold,
                "reason": guardrail,
                "decision_source": "guardrail",
                "redisvl_duration_ms": None,
            }
        if self._EXPLICIT_SHOPPING_INTENT.search(message):
            return {
                "configured": self.configured,
                "eligible": False,
                "cache_read": False,
                "cache_write": False,
                "blocked": False,
                "action": "allow",
                "route": ECOMMERCE_ROUTE,
                "distance": None,
                "threshold": threshold,
                "reason": "explicit ecommerce request",
                "decision_source": "deterministic",
                "redisvl_duration_ms": None,
            }
        if not self.configured:
            return {
                "configured": False,
                "eligible": False,
                "cache_read": False,
                "cache_write": False,
                "blocked": False,
                "action": "allow",
                "route": None,
                "distance": None,
                "threshold": threshold,
                "reason": "semantic router is not configured",
                "decision_source": "fail-safe",
                "redisvl_duration_ms": None,
            }

        route_started = time.perf_counter()
        redisvl_duration_ms: float | None = None
        embedding_duration_ms: float | None = None
        embedding_cache_hit: bool | None = None
        try:
            router = self._get_router()
            contextual_followup = bool(
                recent_context and self.is_contextual_followup(message)
            )
            routing_statement = (
                f"Previous shopping conversation:\n{recent_context[-1_500:]}\n"
                f"Current follow-up: {message}"
                if contextual_followup
                else message
            )
            embedding_started = time.perf_counter()
            cache_probe = getattr(self.embeddings, "is_cached", None)
            if callable(cache_probe):
                embedding_cache_hit = cache_probe(routing_statement)
            vector = self.embeddings.embed(routing_statement)
            embedding_duration_ms = round(
                (time.perf_counter() - embedding_started) * 1000, 2
            )
            redisvl_started = time.perf_counter()
            try:
                match = router(vector=vector)
            finally:
                redisvl_duration_ms = round(
                    (time.perf_counter() - redisvl_started) * 1000, 2
                )
            route_name = getattr(match, "name", None)
            distance = getattr(match, "distance", None)
            cache_scope = LANGCACHE_SCOPES.get(route_name)
            reusable = cache_scope is not None
            cacheable = reusable and not contextual_followup
            ecommerce = route_name == ECOMMERCE_ROUTE
            blocked = route_name == OUT_OF_DOMAIN_ROUTE or not (reusable or ecommerce)
            reason = (
                "contextual ecommerce follow-up"
                if contextual_followup and (reusable or ecommerce)
                else "reusable policy answer"
                if route_name == PUBLIC_POLICY_ROUTE
                else "reusable product education"
                if route_name == PRODUCT_EDUCATION_ROUTE
                else "reusable shopping guide"
                if route_name == SHOPPING_GUIDE_ROUTE
                else "ecommerce request"
                if ecommerce
                else "outside Value Wholesale ecommerce scope"
            )
            return {
                "configured": True,
                "eligible": cacheable,
                "cache_read": cacheable,
                "cache_write": cacheable,
                "blocked": blocked,
                "action": "block" if blocked else "allow",
                "route": route_name,
                "distance": round(float(distance), 4) if distance is not None else None,
                "threshold": threshold,
                "reason": reason,
                "decision_source": "redisvl",
                "redisvl_duration_ms": redisvl_duration_ms,
                "embedding_duration_ms": embedding_duration_ms,
                "embedding_cache_hit": embedding_cache_hit,
                "route_duration_ms": round(
                    (time.perf_counter() - route_started) * 1000, 2
                ),
                "contextual_followup": contextual_followup,
                "cache_scope": cache_scope if cacheable else None,
            }
        except Exception as exc:
            log.warning("RedisVL semantic routing failed open without caching: %s", exc)
            return {
                "configured": True,
                "eligible": False,
                "cache_read": False,
                "cache_write": False,
                "blocked": False,
                "action": "allow",
                "route": None,
                "distance": None,
                "threshold": threshold,
                "reason": "semantic router unavailable",
                "decision_source": "fail-safe",
                "redisvl_duration_ms": redisvl_duration_ms,
                "embedding_duration_ms": embedding_duration_ms,
                "embedding_cache_hit": embedding_cache_hit,
                "route_duration_ms": round(
                    (time.perf_counter() - route_started) * 1000, 2
                ),
            }


class LangCacheService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = (
            f"{settings.langcache_host.rstrip('/')}/v1/caches/{settings.langcache_cache_id}"
            if settings.langcache_configured
            else ""
        )
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=8,
                        headers={
                            "Authorization": f"Bearer {self.settings.langcache_api_key}"
                        },
                        limits=httpx.Limits(
                            max_connections=AGENT_MEMORY_MAX_CONNECTIONS,
                            max_keepalive_connections=AGENT_MEMORY_MAX_KEEPALIVE_CONNECTIONS,
                            keepalive_expiry=self.settings.langcache_http_keepalive_seconds,
                        ),
                    )
        return self._client

    async def close(self) -> None:
        """Close the shared LangCache HTTP pool owned by this worker."""
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            await client.aclose()

    @staticmethod
    def scoped_prompt(prompt: str, scope: str) -> str:
        """Separate semantic workloads without relying on preview attribute schemas."""
        return f"scope:{scope}\n{prompt.strip()}"

    @staticmethod
    def display_prompt(prompt: str) -> str:
        """Remove the internal cache-scope prefix before showing a stored prompt."""
        first_line, separator, remainder = prompt.partition("\n")
        if separator and first_line.startswith("scope:"):
            return remainder.strip()
        return prompt.strip()

    async def search(self, prompt: str, scope: str) -> dict[str, Any] | None:
        if not self.base_url:
            return None
        body = {
            "prompt": self.scoped_prompt(prompt, scope),
            "similarityThreshold": self.settings.langcache_similarity_threshold,
            "searchStrategies": ["semantic"],
        }
        try:
            client = await self._get_client()
            response = await client.post(f"{self.base_url}/entries/search", json=body)
            response.raise_for_status()
            entries = response.json().get("data", [])
            return entries[0] if entries else None
        except Exception as exc:
            log.warning("LangCache search failed open: %s", exc)
            return None

    async def warmup(self, prompt: str, scope: str = "policy:v1") -> bool:
        """Warm LangCache embeddings with a read-only semantic lookup."""
        if not self.base_url:
            return False
        body = {
            "prompt": self.scoped_prompt(prompt, scope),
            "similarityThreshold": self.settings.langcache_similarity_threshold,
            "searchStrategies": ["semantic"],
        }
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/entries/search", json=body)
        response.raise_for_status()
        return True

    async def store(self, prompt: str, answer: str, scope: str) -> bool:
        if not self.base_url:
            return False
        body = {
            "prompt": self.scoped_prompt(prompt, scope),
            "response": answer,
        }
        try:
            client = await self._get_client()
            response = await client.post(f"{self.base_url}/entries", json=body)
            response.raise_for_status()
            return True
        except Exception as exc:
            log.warning("LangCache store failed open: %s", exc)
            return False

    async def clear(self) -> bool:
        """Flush every entry from the configured demo cache."""
        if not self.base_url:
            return False
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/flush", timeout=15)
        response.raise_for_status()
        return True


class MemoryService:
    """Official Redis Agent Memory SDK adapter, scoped by member and namespace."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any | None = None
        self.models: Any | None = None
        self._http_client: httpx.Client | None = None
        self._async_http_client: httpx.AsyncClient | None = None
        if settings.memory_configured:
            try:
                from redis_agent_memory import AgentMemory, models

                def limits() -> httpx.Limits:
                    return httpx.Limits(
                        max_connections=AGENT_MEMORY_MAX_CONNECTIONS,
                        max_keepalive_connections=AGENT_MEMORY_MAX_KEEPALIVE_CONNECTIONS,
                        keepalive_expiry=settings.agent_memory_http_keepalive_seconds,
                    )

                self._http_client = httpx.Client(
                    follow_redirects=True,
                    limits=limits(),
                )
                self._async_http_client = httpx.AsyncClient(
                    follow_redirects=True,
                    limits=limits(),
                )
                self.client = AgentMemory(
                    settings.agent_memory_base_url,
                    store_id=settings.agent_memory_store_id,
                    api_key=settings.agent_memory_api_key,
                    client=self._http_client,
                    async_client=self._async_http_client,
                )
                self.models = models
            except Exception as exc:
                log.warning("Agent Memory SDK initialization failed: %s", exc)

    async def close(self) -> None:
        """Close the shared Agent Memory HTTP pools owned by this worker."""
        self.client = None
        async_client, self._async_http_client = self._async_http_client, None
        http_client, self._http_client = self._http_client, None
        if async_client is not None and not async_client.is_closed:
            await async_client.aclose()
        if http_client is not None and not http_client.is_closed:
            await asyncio.to_thread(http_client.close)

    def add_event(
        self,
        member_id: str,
        session_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if self.client is None or self.models is None:
            return False
        role_enum = getattr(self.models.MessageRole, role.upper())
        actor_id = (
            member_id
            if role.upper() == "USER"
            else "valuewholesale-context"
            if role.upper() == "SYSTEM"
            else "valuewholesale-agent"
        )
        try:
            self.client.add_session_event(
                session_id=safe_id(session_id, "shopping-session"),
                actor_id=safe_id(actor_id, "actor"),
                role=role_enum,
                content=[{"text": text}],
                created_at=datetime.now(UTC),
                metadata={
                    "channel": "web",
                    "agent": "valuewholesale-shopping",
                    **(metadata or {}),
                },
            )
            return True
        except Exception as exc:
            log.warning("Agent Memory event write failed open: %s", exc)
            return False

    async def ping(self) -> bool:
        """Use the managed Agent Memory health endpoint without reading member data."""
        if self.client is None:
            return False
        await self.client.health_async(timeout_ms=5_000)
        return True

    def short_term(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
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
                        "topics": list(
                            dict.fromkeys([*(topics or ["shopping", "preference"]), "demo-created"])
                        ),
                    }
                ]
            )
            return True
        except Exception as exc:
            log.warning("Agent Memory direct write failed open: %s", exc)
            return False

    def list_long_term(
        self,
        member_id: str,
        limit: int = MEMORY_INVENTORY_LIMIT,
    ) -> dict[str, Any]:
        """List a bounded member inventory without performing semantic retrieval."""
        if self.client is None or self.models is None:
            raise RuntimeError("Redis Agent Memory is not configured")

        display_limit = max(1, min(limit, MEMORY_INVENTORY_LIMIT))
        response = self.client.search_long_term_memory(
            request={
                "filter_op": self.models.FilterConjunction.ALL,
                "filter_": {
                    "owner_id": {"eq": safe_id(member_id, "anonymous")},
                    "namespace": {"eq": self.settings.agent_memory_namespace},
                },
                "limit": display_limit + 1,
            }
        )
        records = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in response.items
        ]
        return {
            "count": min(len(records), display_limit),
            "truncated": len(records) > display_limit,
            "memories": records[:display_limit],
        }

    def reset_long_term(
        self,
        member_id: str,
        seeded_memories: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Preserve seeded records and remove other memories in one member namespace."""
        if self.client is None or self.models is None:
            raise RuntimeError("Redis Agent Memory is not configured")

        owner_id = safe_id(member_id, "anonymous")
        namespace = self.settings.agent_memory_namespace
        existing_memory_ids: set[str] = set()
        page_token: str | None = None
        seen_page_tokens: set[str] = set()

        while True:
            response = self.client.search_long_term_memory(
                request={
                    "filter_op": self.models.FilterConjunction.ALL,
                    "filter_": {
                        "owner_id": {"eq": owner_id},
                        "namespace": {"eq": namespace},
                    },
                    "limit": 100,
                    "page_token": page_token,
                }
            )
            existing_memory_ids.update(str(item.id) for item in response.items)
            next_page_token = getattr(response, "next_page_token", None)
            if not next_page_token:
                break
            if next_page_token in seen_page_tokens:
                raise RuntimeError("Agent Memory returned a repeated page token")
            seen_page_tokens.add(next_page_token)
            page_token = next_page_token

        seed_by_id = {str(memory["id"]): memory for memory in seeded_memories}
        memory_ids_to_delete = sorted(existing_memory_ids - seed_by_id.keys())
        deleted = 0
        for start in range(0, len(memory_ids_to_delete), 100):
            response = self.client.bulk_delete_long_term_memories(
                memory_ids=memory_ids_to_delete[start : start + 100]
            )
            errors = list(getattr(response, "errors", None) or [])
            if errors:
                raise RuntimeError(f"Failed to delete {len(errors)} long-term memories")
            deleted += len(response.deleted)

        missing_seed_ids = seed_by_id.keys() - existing_memory_ids
        records = [
            {
                **memory,
                "owner_id": owner_id,
                "namespace": namespace,
                "topics": list(dict.fromkeys([*(memory.get("topics") or []), "demo-seed"])),
            }
            for memory in seeded_memories
            if memory["id"] in missing_seed_ids
            and safe_id(str(memory.get("owner_id", "")), "anonymous") == owner_id
        ]
        created = 0
        for start in range(0, len(records), 100):
            response = self.client.bulk_create_long_term_memories(
                memories=records[start : start + 100]
            )
            errors = list(getattr(response, "errors", None) or [])
            if errors:
                raise RuntimeError(f"Failed to restore {len(errors)} seeded memories")
            created += len(response.created)

        return {
            "deleted": deleted,
            "restored": created,
            "preserved": len(existing_memory_ids & seed_by_id.keys()),
        }


class ContextRetrieverService:
    def __init__(self, settings: Settings) -> None:
        self.agent_key = settings.mcp_agent_key
        self._tools_cache: list[dict[str, Any]] | None = None
        self._tools_lock = asyncio.Lock()
        self._client: Any | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    from context_surfaces import UnifiedClient

                    client = UnifiedClient()
                    await client.__aenter__()
                    self._client = client
        return self._client

    async def close(self) -> None:
        """Close the shared Context Retriever client owned by this worker."""
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(None, None, None)

    @property
    def tools_cached(self) -> bool:
        return self._tools_cache is not None

    async def get_tools(
        self, *, force_refresh: bool = False
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return the governed catalog and whether it came from the server cache."""
        if not self.agent_key:
            return [], False
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache, True

        async with self._tools_lock:
            if self._tools_cache is not None and not force_refresh:
                return self._tools_cache, True
            try:
                client = await self._get_client()
                tools = await client.list_tools(self.agent_key)
                refreshed = [
                    tool if isinstance(tool, dict) else tool.model_dump() for tool in tools
                ]
                self._tools_cache = refreshed
                return refreshed, False
            except Exception as exc:
                log.warning("Context Retriever tool listing failed open: %s", exc)
                if self._tools_cache is not None:
                    return self._tools_cache, True
                return [], False

    async def list_tools(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        tools, _ = await self.get_tools(force_refresh=force_refresh)
        return tools

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.agent_key:
            return {"ok": False, "error": "context_retriever_not_configured"}
        operation_started: float | None = None
        try:
            client = await self._get_client()
            operation_started = time.perf_counter()
            raw = await client.query_tool(
                agent_key=self.agent_key,
                tool_name=tool_name,
                arguments=arguments,
            )
            operation_duration_ms = round(
                (time.perf_counter() - operation_started) * 1000,
                2,
            )
            if isinstance(raw, dict):
                content = raw.get("content", [])
                if content and content[0].get("type") == "text":
                    result = json.loads(content[0].get("text", "{}"))
                else:
                    result = raw
                if isinstance(result, dict):
                    return {**result, "operation_duration_ms": operation_duration_ms}
                return {
                    "result": result,
                    "operation_duration_ms": operation_duration_ms,
                }
            return {
                "result": str(raw),
                "operation_duration_ms": operation_duration_ms,
            }
        except Exception as exc:
            log.warning("Context Retriever call failed open: %s", exc)
            result: dict[str, Any] = {"ok": False, "error": str(exc)}
            if operation_started is not None:
                result["operation_duration_ms"] = round(
                    (time.perf_counter() - operation_started) * 1000,
                    2,
                )
            return result

    async def get_member_profile(self, member_id: str) -> dict[str, Any]:
        """Discover and call the governed member lookup tool."""
        tools = await self.list_tools()
        lookup = next(
            (tool for tool in tools if tool.get("name") == "get_member_by_id"),
            None,
        )
        if lookup is None:
            return {"ok": False, "error": "member_profile_tool_not_available"}
        schema = lookup.get("inputSchema", {})
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if "id" not in required:
            return {"ok": False, "error": "member_profile_tool_schema_invalid"}
        return await self.call(str(lookup["name"]), {"id": member_id})


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
                app_name="valuewholesale-shopping-agent",
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

    def list_long_term(
        self,
        member_id: str,
        limit: int = MEMORY_INVENTORY_LIMIT,
    ) -> dict[str, Any]:
        """List a bounded member inventory through the underlying Vertex client."""
        if self.client is None:
            raise RuntimeError("Vertex ADK Memory Bank is not configured")

        import vertexai

        display_limit = max(1, min(limit, MEMORY_INVENTORY_LIMIT))
        scope = {
            "app_name": VERTEX_MEMORY_APP_NAME,
            "user_id": safe_id(member_id, "anonymous"),
        }
        resource_name = (
            f"projects/{self.settings.google_cloud_project}"
            f"/locations/{self.settings.google_memory_location}"
            f"/reasoningEngines/{self.settings.google_agent_engine_id}"
        )
        client = vertexai.Client(
            project=self.settings.google_cloud_project,
            location=self.settings.google_memory_location,
            http_options={"timeout": 60_000},
        )
        scope_json = json.dumps(scope, separators=(",", ":"))
        scope_filter = f"scope = {json.dumps(scope_json)}"
        records = list(
            itertools.islice(
                client.agent_engines.memories.list(
                    name=resource_name,
                    config={
                        "page_size": display_limit + 1,
                        "filter": scope_filter,
                        "order_by": "update_time desc",
                    },
                ),
                display_limit + 1,
            )
        )
        return {
            "count": min(len(records), display_limit),
            "truncated": len(records) > display_limit,
            "memories": [self._serialize(item) for item in records[:display_limit]],
        }

    def reset_long_term(
        self,
        member_id: str,
        seeded_memories: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Preserve seeded facts and remove other memories in one member scope."""
        if self.client is None:
            raise RuntimeError("Vertex ADK Memory Bank is not configured")

        import vertexai

        owner_id = safe_id(member_id, "anonymous")
        scope = {"app_name": VERTEX_MEMORY_APP_NAME, "user_id": owner_id}
        resource_name = (
            f"projects/{self.settings.google_cloud_project}"
            f"/locations/{self.settings.google_memory_location}"
            f"/reasoningEngines/{self.settings.google_agent_engine_id}"
        )
        client = vertexai.Client(
            project=self.settings.google_cloud_project,
            location=self.settings.google_memory_location,
            http_options={"timeout": 60_000},
        )
        seed_facts = {str(memory["text"]) for memory in seeded_memories}
        scoped_memories = [
            memory
            for memory in client.agent_engines.memories.list(name=resource_name)
            if getattr(memory, "scope", None) == scope
        ]

        deleted = 0
        existing_seed_facts: set[str] = set()
        for memory in scoped_memories:
            fact = str(getattr(memory, "fact", ""))
            if fact in seed_facts and fact not in existing_seed_facts:
                existing_seed_facts.add(fact)
                continue
            operation = client.agent_engines.memories.delete(name=memory.name)
            if hasattr(operation, "result"):
                operation.result()
            deleted += 1

        restored = 0
        for fact in sorted(seed_facts - existing_seed_facts):
            operation = client.agent_engines.memories.create(
                name=resource_name,
                fact=fact,
                scope=scope,
                config={
                    "metadata": {"valuewholesale_origin": {"string_value": "demo-seed"}}
                },
            )
            if hasattr(operation, "result"):
                operation.result()
            restored += 1

        return {
            "deleted": deleted,
            "restored": restored,
            "preserved": len(existing_seed_facts),
        }

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
        memories, latency_ms = await asyncio.to_thread(
            call_with_timing,
            services.memory.recall,
            member_id,
            query,
            5,
        )
        return {
            "provider": "redis_agent_memory",
            "configured": services.memory.client is not None,
            "latency_ms": latency_ms,
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
        self.embeddings = LocalEmbeddingService(self.settings)
        self.catalog = CatalogService(self.settings, self.embeddings)
        self.tool_cache = ToolCallCache(self.settings, self.catalog.redis)
        self.cart = CartService(self.settings)
        self.semantic_router = SemanticRouterService(
            self.settings,
            self.catalog.redis,
            self.embeddings,
        )
        self.langcache = LangCacheService(self.settings)
        self.memory = MemoryService(self.settings)
        self.vertex_memory = VertexMemoryService(self.settings)
        self.context = ContextRetrieverService(self.settings)


services = Services()
