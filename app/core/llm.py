"""
One place for every LLM call.

    llm.call(role, system, user, schema=...)  ->  parsed schema instance (or text)

* Providers come from config as "<provider>/<model-id>" specs
  (PRIMARY_LLM, FALLBACK_LLM). Supported: groq, google (Gemini), openai.
  Model ids may contain slashes ("groq/openai/gpt-oss-120b").
* The chain is the retry policy: a provider with a missing key is skipped, a
  provider that errors hands the same call to the next one. SDK retries stay
  low so a rate limit fails over instead of hanging.
* Every call is tagged with the agent `role`, so tests and the offline demo
  swap in a scripted model (`use_llm`) without patching four modules.
* Token usage and an estimated cost are accumulated per run (`track_usage`);
  the graph uses that to enforce MAX_TOKENS_PER_TASK.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# (role, schema | None, system, user) -> schema instance | str
LLMFn = Callable[[str, Optional[Type[BaseModel]], str, str], Any]

# USD per 1M tokens (input, output). Estimates for reporting only.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICE = (0.0, 0.0)  # free tiers (Groq gpt-oss, Gemini flash) are honestly $0


class LLMUnavailable(RuntimeError):
    """No provider is configured, or every provider in the chain failed."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_provider: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_override: contextvars.ContextVar[Optional[LLMFn]] = contextvars.ContextVar("llm_override", default=None)
_usage: contextvars.ContextVar[Optional[Usage]] = contextvars.ContextVar("llm_usage", default=None)


@contextlib.contextmanager
def use_llm(fn: LLMFn):
    """Route every LLM call in this context to `fn` (tests, offline demo)."""
    token = _override.set(fn)
    try:
        yield
    finally:
        _override.reset(token)


@contextlib.contextmanager
def track_usage():
    usage = Usage()
    token = _usage.set(usage)
    try:
        yield usage
    finally:
        _usage.reset(token)


def current_usage() -> Optional[Usage]:
    return _usage.get()


# ── providers ────────────────────────────────────────────────────────────────

def split_spec(spec: str) -> tuple[str, str]:
    """'groq/openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'); bare id -> openai."""
    provider, _, model = spec.strip().partition("/")
    if not model:
        return "openai", provider
    return provider.lower(), model


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = _PRICES_PER_MTOK.get(model, _DEFAULT_PRICE)
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


def _key_for(provider: str) -> str:
    s = get_settings()
    return {"groq": s.groq_api_key, "google": s.google_api_key, "gemini": s.google_api_key,
            "openai": s.openai_api_key}.get(provider, "")


def build_chat_model(spec: str, temperature: float = 0.1):
    """Construct a LangChain chat model from a spec. Raises if its key is missing."""
    s = get_settings()
    provider, model = split_spec(spec)
    key = _key_for(provider)
    if not key:
        raise LLMUnavailable(f"no API key for {provider} (needed by {spec})")
    common = dict(temperature=temperature, max_retries=1, timeout=s.llm_timeout_s)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=key, **common)
    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, **common)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=key, **common)
    raise ValueError(f"Unknown LLM provider {provider!r} in {spec!r}")


def chain_specs(cheap: bool = False) -> list[str]:
    s = get_settings()
    specs = [s.hyde_llm] if cheap and s.hyde_llm else []
    specs += [s.primary_llm, s.fallback_llm]
    seen, out = set(), []
    for spec in specs:
        if spec and spec not in seen and _key_for(split_spec(spec)[0]):
            seen.add(spec)
            out.append(spec)
    return out


def available() -> bool:
    return _override.get() is not None or bool(chain_specs())


def unpack_structured(result) -> tuple[object, int, int]:
    """Unpack `.with_structured_output(..., include_raw=True)` -> (parsed, in_tokens, out_tokens)."""
    parsed = result["parsed"]
    if parsed is None:
        raise ValueError(f"Structured output parsing failed: {result.get('parsing_error')}")
    usage = getattr(result.get("raw"), "usage_metadata", None) or {}
    return parsed, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def _record(provider: str, model: str = "", in_tok: int = 0, out_tok: int = 0) -> None:
    u = _usage.get()
    if u is None:
        return
    u.calls += 1
    u.by_provider[provider] = u.by_provider.get(provider, 0) + 1
    u.prompt_tokens += in_tok
    u.completion_tokens += out_tok
    u.cost_usd += estimate_cost_usd(model, in_tok, out_tok)


# ── the one entry point ──────────────────────────────────────────────────────

_RATE = re.compile(r"429|rate.?limit|too many requests|resource.?exhausted|quota", re.I)
_WAIT = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.I)


def _rate_limit_wait(errors: list[str]) -> Optional[float]:
    """If any provider was rate-limited, how long it asked us to wait (else None)."""
    if not errors or not any(_RATE.search(e) for e in errors):
        return None
    waits = []
    for e in errors:
        m = _WAIT.search(e)
        if m:
            waits.append(int(m.group(1) or 0) * 60 + float(m.group(2)))
    return min(waits) + 0.5 if waits else 20.0


def call(role: str, system: str, user: str, schema: Optional[Type[T]] = None, cheap: bool = False):
    """
    Run one LLM call through the provider chain. If every provider is rate-limited,
    sleep as long as the API asked (capped) and retry the chain. Other failures
    are not retried. Raises LLMUnavailable if nothing answers.
    """
    s = get_settings()
    for attempt in range(s.llm_rate_limit_retries + 1):
        try:
            return _call_once(role, system, user, schema, cheap)
        except _ChainFailed as failed:
            wait = _rate_limit_wait(failed.errors)
            if wait is None or attempt == s.llm_rate_limit_retries:
                raise LLMUnavailable(f"All LLM providers failed for {role}: " + " | ".join(failed.errors))
            wait = min(wait, s.llm_max_wait_s)
            logger.warning("[llm] %s: every provider rate-limited; waiting %.1fs", role, wait)
            _sleep(wait)


_sleep = time.sleep  # patched in tests


class _ChainFailed(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors


def _call_once(role: str, system: str, user: str, schema, cheap: bool):
    override = _override.get()
    if override is not None:
        _record("override")
        return override(role, schema, system, user)

    specs = chain_specs(cheap)
    if not specs:
        raise LLMUnavailable(
            "No LLM configured. Set GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY "
            "(see PRIMARY_LLM, FALLBACK_LLM in .env.example), or run the offline demo: "
            "python -m app.cli demo"
        )

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    errors: list[str] = []
    for spec in specs:
        provider, model = split_spec(spec)
        try:
            chat = build_chat_model(spec)
            if schema is None:
                msg = chat.invoke(messages)
                _record_msg(provider, model, msg)
                return _text(msg)
            try:
                parsed, in_tok, out_tok = unpack_structured(
                    chat.with_structured_output(schema, include_raw=True).invoke(messages))
                _record(provider, model, in_tok, out_tok)
                return parsed
            except Exception as e:  # noqa: BLE001
                if _RATE.search(str(e)):
                    raise
                # Tool/function calling is the flakiest part of structured output
                # (Groq: "tool_use_failed"). Ask the same model for plain JSON instead.
                logger.warning("[llm] %s via %s: structured call failed (%s: %s); retrying as plain JSON",
                               role, spec, type(e).__name__, str(e)[:160])
                return _json_fallback(chat, provider, model, schema, system, user)
        except Exception as e:  # noqa: BLE001 — any provider failure moves down the chain
            err = f"{spec}: {type(e).__name__}: {str(e)[:300]}"
            errors.append(err)
            u = _usage.get()
            if u is not None:
                u.failures.append(f"[{role}] {err}")
            logger.warning("[llm] %s via %s failed (%s: %s) -> next provider",
                           role, spec, type(e).__name__, str(e)[:200])

    raise _ChainFailed(errors)


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, list):  # some providers return content blocks
        c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c).strip()


def _record_msg(provider: str, model: str, msg) -> None:
    usage = getattr(msg, "usage_metadata", None) or {}
    _record(provider, model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))


def extract_json(text: str) -> str:
    """Pull the outermost JSON object out of a reply (handles ```json fences and prose)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        return fence.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return t[start:end + 1]


def _json_fallback(chat, provider: str, model: str, schema, system: str, user: str):
    import json

    from langchain_core.messages import HumanMessage, SystemMessage
    instr = (
        "\n\nRespond with ONLY one JSON object (no prose, no markdown) that validates against "
        f"this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
    )
    msg = chat.invoke([SystemMessage(content=system + instr), HumanMessage(content=user)])
    _record_msg(provider, model, msg)
    return schema.model_validate_json(extract_json(_text(msg)))
