from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_cloud_project: str = ""
    google_cloud_location: str = "us-east4"
    google_model: str = "gemini-2.5-flash"
    google_models: str = "gemini-2.5-flash gemini-2.5-pro"
    google_genai_use_vertexai: bool = True
    google_memory_location: str = "us-east4"
    google_agent_engine_id: str = ""
    valueharbor_vector_search_enabled: bool = False

    valueharbor_demo_member_id: str = "member-1001"
    valueharbor_demo_session_id: str = "shopping-demo-1"

    redis_url: str = ""
    mcp_agent_key: str = ""

    valueharbor_semantic_router_threshold: float = Field(default=0.48, gt=0, le=2)
    valueharbor_semantic_router_index: str = "valueharbor-cache-router-v1"
    valueharbor_semantic_router_embedding_model: str = "text-embedding-005"

    langcache_host: str = ""
    langcache_cache_id: str = ""
    langcache_api_key: str = ""
    langcache_similarity_threshold: float = Field(default=0.92, ge=0, le=1)

    agent_memory_base_url: str = ""
    agent_memory_store_id: str = ""
    agent_memory_api_key: str = ""
    agent_memory_namespace: str = "valueharbor-shopping"
    agent_memory_similarity_threshold: float = Field(default=0.30, ge=0, le=1)

    valueharbor_agent_timeout_seconds: float = Field(default=45, ge=5, le=120)

    port: int = 8080
    log_level: str = "INFO"

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_url)

    @property
    def langcache_configured(self) -> bool:
        return bool(self.langcache_host and self.langcache_cache_id and self.langcache_api_key)

    @property
    def semantic_router_configured(self) -> bool:
        return bool(self.redis_url and self.google_cloud_project)

    @property
    def memory_configured(self) -> bool:
        return bool(
            self.agent_memory_base_url and self.agent_memory_store_id and self.agent_memory_api_key
        )

    @property
    def vertex_memory_configured(self) -> bool:
        return bool(self.google_cloud_project and self.google_agent_engine_id)

    @property
    def available_google_models(self) -> tuple[str, str]:
        """The two demo choices: fast and reasoning-heavy."""
        return ("gemini-2.5-flash", "gemini-2.5-pro")


@lru_cache
def get_settings() -> Settings:
    return Settings()
