from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from valuewholesale_agent.experience import ROOT, ExperienceProfile, get_experience_profile


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    experience_id: str = "valuewholesale"
    dataset_dir: str = ""
    static_dir: str = ""
    redis_key_prefix: str = ""
    redis_index_prefix: str = ""
    embedding_cache_name: str = ""
    adk_app_name: str = ""
    adk_greeting_app_name: str = ""
    adk_transcript_app_name: str = ""
    context_surface_name: str = ""
    context_agent_name: str = ""
    context_agent_display_name: str = ""
    memory_bank_display_name: str = ""

    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    google_model: str = "gemini-3.1-flash-lite"
    google_models: str = "gemini-3.1-flash-lite gemini-3.1-pro-preview"
    google_genai_use_vertexai: bool = True
    google_memory_location: str = ""
    google_agent_engine_id: str = ""
    valuewholesale_vector_search_enabled: bool = True
    valuewholesale_embedding_model: str = "redis/langcache-embed-v3-small"
    valuewholesale_embedding_device: str = "cpu"
    valuewholesale_embedding_cache_ttl_seconds: int = Field(default=86_400, ge=60)
    valuewholesale_tool_cache_ttl_seconds: int = Field(default=43_200, ge=60)

    valuewholesale_demo_member_id: str = "member-1001"
    valuewholesale_demo_session_id: str = "shopping-demo-1"

    redis_url: str = ""
    mcp_agent_key: str = ""

    valuewholesale_semantic_router_threshold: float = Field(default=0.48, gt=0, le=2)
    valuewholesale_semantic_router_index: str = "valuewholesale-cache-router-v2"

    langcache_host: str = ""
    langcache_cache_id: str = ""
    langcache_api_key: str = ""
    langcache_similarity_threshold: float = Field(default=0.80, ge=0, le=1)
    langcache_http_keepalive_seconds: float = Field(default=300, ge=5, le=3_600)

    agent_memory_base_url: str = ""
    agent_memory_store_id: str = ""
    agent_memory_api_key: str = ""
    agent_memory_namespace: str = ""
    agent_memory_similarity_threshold: float = Field(default=0.30, ge=0, le=1)
    agent_memory_http_keepalive_seconds: float = Field(default=300, ge=5, le=3_600)

    valuewholesale_agent_timeout_seconds: float = Field(default=90, ge=5, le=120)
    valuewholesale_warmup_on_startup: bool = False

    port: int = 8080
    log_level: str = "INFO"

    @property
    def experience(self) -> ExperienceProfile:
        return get_experience_profile(self.experience_id)

    @property
    def dataset_path(self) -> Path:
        if not self.dataset_dir:
            return self.experience.dataset_path
        path = Path(self.dataset_dir)
        return path if path.is_absolute() else ROOT / path

    @property
    def static_path(self) -> Path:
        if not self.static_dir:
            return self.experience.static_path
        path = Path(self.static_dir)
        return path if path.is_absolute() else ROOT / path

    @property
    def redis_namespace(self) -> str:
        return self.redis_key_prefix.strip(":") or self.experience.redis_key_prefix

    @property
    def index_namespace(self) -> str:
        return self.redis_index_prefix.strip(":") or self.experience.redis_index_prefix

    @property
    def product_index_name(self) -> str:
        return f"{self.index_namespace}:products-v2"

    @property
    def policy_index_name(self) -> str:
        return f"{self.index_namespace}:policies-v2"

    @property
    def member_index_name(self) -> str:
        return f"{self.index_namespace}:members"

    @property
    def order_index_name(self) -> str:
        return f"{self.index_namespace}:orders"

    @property
    def order_item_index_name(self) -> str:
        return f"{self.index_namespace}:order-items"

    @property
    def effective_embedding_cache_name(self) -> str:
        return self.embedding_cache_name or self.experience.embedding_cache_name

    @property
    def effective_semantic_router_index(self) -> str:
        configured = self.valuewholesale_semantic_router_index.strip()
        if self.experience_id == "valuewholesale" or (
            configured and configured != "valuewholesale-cache-router-v2"
        ):
            return configured or self.experience.semantic_router_index
        return self.experience.semantic_router_index

    @property
    def effective_agent_memory_namespace(self) -> str:
        if "agent_memory_namespace" in self.model_fields_set:
            return self.agent_memory_namespace.strip()
        return self.experience.agent_memory_namespace

    @property
    def effective_app_name(self) -> str:
        return self.adk_app_name or self.experience.app_name

    @property
    def effective_greeting_app_name(self) -> str:
        return self.adk_greeting_app_name or self.experience.greeting_app_name

    @property
    def effective_transcript_app_name(self) -> str:
        return self.adk_transcript_app_name or self.experience.transcript_app_name

    @property
    def effective_context_surface_name(self) -> str:
        return self.context_surface_name or self.experience.context_surface_name

    @property
    def effective_context_agent_name(self) -> str:
        return self.context_agent_name or self.experience.context_agent_name

    @property
    def effective_context_agent_display_name(self) -> str:
        return self.context_agent_display_name or self.experience.context_agent_display_name

    @property
    def effective_memory_bank_display_name(self) -> str:
        return self.memory_bank_display_name or self.experience.memory_bank_display_name

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_url)

    @property
    def redis_endpoint(self) -> str:
        """Return the configured Redis host and port without credentials."""
        if not self.redis_url:
            return ""
        parsed = urlparse(self.redis_url)
        if not parsed.hostname:
            return ""
        return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname

    @property
    def langcache_configured(self) -> bool:
        return bool(self.langcache_host and self.langcache_cache_id and self.langcache_api_key)

    @property
    def semantic_router_configured(self) -> bool:
        return bool(self.redis_url)

    @property
    def memory_configured(self) -> bool:
        return bool(
            self.agent_memory_base_url and self.agent_memory_store_id and self.agent_memory_api_key
        )

    @property
    def vertex_memory_configured(self) -> bool:
        return bool(
            self.google_cloud_project
            and self.google_memory_location
            and self.google_agent_engine_id
        )

    @property
    def available_google_models(self) -> tuple[str, str]:
        """The two demo choices: fast and reasoning-heavy."""
        return ("gemini-3.1-flash-lite", "gemini-3.1-pro-preview")


@lru_cache
def get_settings() -> Settings:
    return Settings()
