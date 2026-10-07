"""
Runtime configuration.

Every external dependency is optional. NEXUS degrades instead of crashing:
  - no LLM keys          -> API still boots; pipeline fails fast with a clear error
  - no Weaviate          -> local in-memory hybrid retriever (RETRIEVER=local)
  - no Postgres          -> SQLite file database
"""
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULT_SECRET = "change-me-in-production-use-secrets-token-hex-32"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # ── LLM chain: "<provider>/<model-id>", providers: groq | google | openai ──
    # Tried in order; a provider whose key is missing is skipped.
    # Groq retired llama-3.1/3.3-70b-versatile on free tiers; gpt-oss-120b replaces them.
    primary_llm: str = "groq/openai/gpt-oss-120b"
    fallback_llm: str = "google/gemini-3.6-flash"
    hyde_llm: str = ""                 # optional cheaper model for HyDE; defaults to the chain
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    llm_timeout_s: float = 90.0
    max_tokens_per_task: int = 50000   # hard budget: the agent loop stops when exhausted

    # ── GitHub ──
    github_token: str = ""
    # Required to enable the webhook endpoint (503 otherwise).
    github_webhook_secret: str = ""

    # ── Storage ──
    database_url: str = "sqlite:///./nexus.db"

    # ── Retrieval: "local" (in-memory BM25 hybrid) or "weaviate" ──
    retriever: str = "local"
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""

    # ── Patch gate ──
    # Optional command run inside the patched checkout, e.g. "python -m pytest -q -x".
    # Empty = only `git apply --check` + Python compile check.
    gate_test_command: str = ""
    gate_test_timeout_s: int = 300

    # ── Agent loop ──
    review_pass_threshold: float = 0.7
    max_reflections: int = 2

    # ── Auth / app ──
    secret_key: str = _INSECURE_DEFAULT_SECRET
    app_env: str = "development"
    # The dashboard is served by this app (same-origin); only add origins if hosting it elsewhere.
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @model_validator(mode="after")
    def _no_insecure_defaults_in_production(self) -> "Settings":
        if not self.secret_key:
            self.secret_key = _INSECURE_DEFAULT_SECRET  # never sign JWTs with an empty key
        if self.is_production and self.secret_key == _INSECURE_DEFAULT_SECRET:
            raise ValueError(
                "SECRET_KEY is still the insecure default. Set SECRET_KEY before deploying "
                'with APP_ENV=production (python -c "import secrets; print(secrets.token_hex(32))")'
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()
