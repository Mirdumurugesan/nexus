"""
Shared LLM helpers: structured-output unpacking (with token usage) and
cost estimation. Used by all agents so token accounting stays consistent.
"""
import logging

logger = logging.getLogger(__name__)

# USD per 1M tokens: (input, output). Estimates for cost reporting only.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICE = (2.50, 10.00)


def unpack_structured(result) -> tuple[object, int, int]:
    """
    Unpack a `.with_structured_output(..., include_raw=True)` result.
    Returns (parsed_model, prompt_tokens, completion_tokens).
    Raises ValueError if structured-output parsing failed.
    """
    parsed = result["parsed"]
    if parsed is None:
        raise ValueError(f"Structured output parsing failed: {result.get('parsing_error')}")
    usage = getattr(result.get("raw"), "usage_metadata", None) or {}
    return parsed, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Rough $ estimate for a model call. Unknown models use gpt-4o pricing."""
    price_in, price_out = _PRICES_PER_MTOK.get(model, _DEFAULT_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
