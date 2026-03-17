"""Built-in LLM pricing data and cost calculation engine.

All prices are in USD per 1,000,000 tokens (1M tokens).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# fmt: off
# Schema: "canonical_model_key": (input_per_1m_usd, output_per_1m_usd)
_PRICING: Dict[str, Tuple[float, float]] = {
    # ── OpenAI ──────────────────────────────────────────────────────────
    "gpt-4o":              (2.50,   10.00),
    "gpt-4o-mini":         (0.15,    0.60),
    "gpt-4.1":             (2.00,    8.00),
    "gpt-4.1-mini":        (0.40,    1.60),
    "gpt-4.1-nano":        (0.10,    0.40),
    "o3":                  (2.00,    8.00),
    "o4-mini":             (1.10,    4.40),
    "gpt-4-turbo":         (10.00,  30.00),
    "gpt-4":               (30.00,  60.00),
    "gpt-3.5-turbo":       (0.50,    1.50),

    # ── Anthropic ───────────────────────────────────────────────────────
    "claude-opus-4":       (15.00,  75.00),
    "claude-sonnet-4":     (3.00,   15.00),
    "claude-haiku-3.5":    (0.80,    4.00),
    "claude-opus-3":       (15.00,  75.00),
    "claude-sonnet-3.7":   (3.00,   15.00),
    "claude-sonnet-3.5":   (3.00,   15.00),
    "claude-haiku-3":      (0.25,    1.25),

    # ── Google ──────────────────────────────────────────────────────────
    "gemini-2.5-pro":      (1.25,   10.00),
    "gemini-2.5-flash":    (0.30,    2.50),
    "gemini-2.0-flash":    (0.10,    0.40),
    "gemini-1.5-pro":      (1.25,    5.00),
    "gemini-1.5-flash":    (0.075,   0.30),
}
# fmt: on

# Pre-compiled alias patterns: (regex, canonical_key)
# More specific patterns must come before generic ones.
_ALIASES: list[Tuple[re.Pattern, str]] = []

def _build_alias_table() -> None:
    for key in _PRICING:
        # Escape dots in the key so "gpt-4.1" doesn't match "gpt-4X1"
        escaped = re.escape(key)
        # Allow optional date suffix like -2024-12-01, -20250514, or -preview
        pattern = re.compile(
            r"^" + escaped + r"(-\d{4,8}|-preview|-latest)?$",
            re.IGNORECASE,
        )
        _ALIASES.append((pattern, key))


_build_alias_table()


def _match_model(model: str) -> Optional[str]:
    """Return the canonical pricing key for a model string, or None."""
    model_lower = model.lower().strip()
    # Exact match first
    if model_lower in _PRICING:
        return model_lower
    # Alias pattern match
    for pattern, canonical in _ALIASES:
        if pattern.match(model_lower):
            return canonical
    # Fuzzy: find the longest key that is a prefix of model_lower
    best: Optional[str] = None
    for key in _PRICING:
        if model_lower.startswith(key) and (
            best is None or len(key) > len(best)
        ):
            best = key
    return best


def get_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    warn_unknown: bool = True,
) -> Tuple[float, float, float]:
    """Return (input_cost, output_cost, total_cost) in USD.

    Uses 6-decimal precision. Returns (0.0, 0.0, 0.0) for unknown models.
    """
    canonical = _match_model(model)
    if canonical is None:
        if warn_unknown:
            logger.warning(
                "token-meter: unknown model %r — cost recorded as $0.00. "
                "Open an issue at https://github.com/hidearmoon/token-meter "
                "to add pricing.",
                model,
            )
        return 0.0, 0.0, 0.0

    in_rate, out_rate = _PRICING[canonical]
    input_cost = round(input_tokens * in_rate / 1_000_000, 6)
    output_cost = round(output_tokens * out_rate / 1_000_000, 6)
    total_cost = round(input_cost + output_cost, 6)
    return input_cost, output_cost, total_cost


def list_models() -> Dict[str, Tuple[float, float]]:
    """Return a copy of the pricing table (for CLI / debugging)."""
    return dict(_PRICING)


def add_custom_pricing(
    model: str,
    input_per_1m: float,
    output_per_1m: float,
) -> None:
    """Allow users to register pricing for custom / fine-tuned models."""
    key = model.lower().strip()
    _PRICING[key] = (input_per_1m, output_per_1m)
    # Rebuild alias table to include the new key
    _ALIASES.clear()
    _build_alias_table()
