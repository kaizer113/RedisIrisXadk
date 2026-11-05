"""GA settings contain scoped runtime credentials only."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    redis_url: SecretStr = SecretStr("")
    onprem_session_secret: SecretStr = SecretStr("")
    onprem_cookie_secure: bool = False
    onprem_namespace: str = "valuewholesale-onprem-ga"
    onprem_members_file: Path = Path(__file__).parent / "members.jsonl"
    onprem_run_ttl: int = Field(default=86400, ge=3600)
    onprem_max_tool_rounds: int = Field(default=6, ge=1, le=12)
    onprem_turn_timeout: int = Field(default=120, ge=10, le=300)

    iris_agent_memory_url: str = ""
    iris_agent_memory_store_id: str = ""
    iris_agent_memory_api_key: SecretStr = SecretStr("")
    iris_agent_memory_namespace_id: str = ""
    iris_agent_playbooks_url: str = ""
    iris_agent_playbooks_playbook_id: str = ""
    iris_agent_playbooks_api_key: SecretStr = SecretStr("")
    iris_intent_router_url: str = ""
    iris_intent_router_id: str = ""
    iris_intent_router_api_key: SecretStr = SecretStr("")
    iris_context_gateway_url: str = ""
    iris_context_gateway_api_key: SecretStr = SecretStr("")
    iris_lens_endpoint: str = ""
    iris_lens_api_key: SecretStr = SecretStr("")
    iris_lens_agent_id: str = "valuewholesale-onprem"
    iris_lens_capture_content: bool = False
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    google_model: str = "gemini-3.6-flash"

    @property
    def configured(self) -> dict[str, bool]:
        return {
            "agent_memory": all(
                (
                    self.iris_agent_memory_url,
                    self.iris_agent_memory_store_id,
                    self.iris_agent_memory_api_key.get_secret_value(),
                    self.iris_agent_memory_namespace_id,
                )
            ),
            "agent_playbooks": all(
                (
                    self.iris_agent_playbooks_url,
                    self.iris_agent_playbooks_playbook_id,
                    self.iris_agent_playbooks_api_key.get_secret_value(),
                )
            ),
            "intent_router": all(
                (
                    self.iris_intent_router_url,
                    self.iris_intent_router_id,
                    self.iris_intent_router_api_key.get_secret_value(),
                )
            ),
            "context_gateway": all(
                (
                    self.iris_context_gateway_url,
                    self.iris_context_gateway_api_key.get_secret_value(),
                )
            ),
            "lens": all((self.iris_lens_endpoint, self.iris_lens_api_key.get_secret_value())),
            "agent_mesh": bool(self.redis_url.get_secret_value()),
            "gemini": bool(self.google_cloud_project),
        }
