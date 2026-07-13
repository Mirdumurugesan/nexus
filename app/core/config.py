from pydantic import model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache

_INSECURE_DEFAULT_SECRET = "change-me-in-production-use-secrets-token-hex-32"


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    groq_api_key: str
    primary_llm: str = "gpt-4o"
    # Format: "<provider>/<model-id>". Groq decommissioned llama-3.1-70b-versatile
    # (Jan 2025) and llama-3.3-70b-versatile (Jun 2026, free/dev tiers).
    fallback_llm: str = "groq/openai/gpt-oss-120b"

    # GitHub
    github_token: str

    # Database
    database_url: str

    # Weaviate — local (http://...) or Weaviate Cloud (https://... + api key)
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""

    # Redis (Phase 2+)
    redis_url: str = "redis://localhost:6379"  # not used in Phase 1

    # GitHub Webhook — REQUIRED to enable the webhook endpoint.
    # If empty, POST /api/v1/webhook/github is disabled (503).
    github_webhook_secret: str = ""

    # Auth — generate with: python -c "import secrets; print(secrets.token_hex(32))"
    secret_key: str = _INSECURE_DEFAULT_SECRET

    # CORS — comma-separated list of allowed origins.
    # The dashboard is served by this app itself (same-origin), so the
    # defaults only need to cover local development.
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_tokens_per_task: int = 50000

    @model_validator(mode="after")
    def _no_insecure_defaults_in_production(self) -> "Settings":
        if self.app_env == "production" and self.secret_key == _INSECURE_DEFAULT_SECRET:
            raise ValueError(
                "SECRET_KEY is still the insecure default. Set SECRET_KEY before "
                "deploying with APP_ENV=production "
                '(generate: python -c "import secrets; print(secrets.token_hex(32))")'
            )
        return self

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
