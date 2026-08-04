from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "att-langgraph-orchestrator"
    app_env: str = "local"
    log_level: str = "INFO"
    app_api_key: str | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    databricks_host: str
    databricks_token: str
    databricks_policy_index: str = "workspace.staging_ai.ai_knowledge_index"
    databricks_billing_index: str = "workspace.staging_ai.billing_knowledge_index"
    databricks_chat_endpoint: str = "databricks-gpt-oss-120b"

    snowflake_host: str
    snowflake_pat: str
    snowflake_cortex_database: str = "EDU_AI_APP"
    snowflake_cortex_schema: str = "AI"
    snowflake_cortex_service: str = "BILLING_CORTEX_SEARCH"

    llm_enabled: bool = True
    max_results_per_source: int = Field(default=3, ge=1, le=10)

    @field_validator("databricks_host", "snowflake_host")
    @classmethod
    def normalize_host(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("databricks_token", "snowflake_pat")
    @classmethod
    def reject_blank_secrets(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("credential cannot be blank")
        return value.strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

