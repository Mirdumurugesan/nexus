from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    groq_api_key: str
    primary_llm: str = "gpt-4o"
    fallback_llm: str = "groq/llama-3.1-70b-versatile"

    # GitHub
    github_token: str

    # Database
    database_url: str

    # Weaviate
    weaviate_url: str = "http://localhost:8080"

    # Redis (Phase 2+)
    redis_url: str = "redis://localhost:6379"  # not used in Phase 1

    # Weaviate API Key
    weaviate_api_key: str = ""

    # GitHub Webhook (optional — for GitHub App integration)
    github_webhook_secret: str = ""

    # Auth — generate with: python -c "import secrets; print(secrets.token_hex(32))"
    secret_key: str = "change-me-in-production-use-secrets-token-hex-32"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_tokens_per_task: int = 50000

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()