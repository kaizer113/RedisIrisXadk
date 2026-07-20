from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    agent_memory_base_url: str = ""
    agent_memory_store_id: str = ""
    agent_memory_api_key: str = ""
    agent_memory_namespace: str = "valuewholesale-shopping"
    agent_memory_similarity_threshold: float = Field(default=0.30, ge=0, le=1)

    valuewholesale_agent_timeout_seconds: float = Field(default=90, ge=5, le=120)

    port: int = 8080
    log_level: str = "INFO"

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
