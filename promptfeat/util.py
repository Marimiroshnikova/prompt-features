"""Small shared helpers for feature implementations."""

from __future__ import annotations

from typing import Iterable

from .registry import FeatureResult, span, undefined


def r2(value: float) -> float:
    return round(float(value), 2)


def r3(value: float) -> float:
    return round(float(value), 3)


def ratio(
    numerator: float,
    denominator: float,
    label: str,
    *,
    zero_reason: str,
    digits: int = 3,
) -> tuple[float | None, str, FeatureResult | None]:
    """A guarded division that reports its own arithmetic.

    Returns `(value, step, failure)`. When `denominator` is 0 the value is None
    and `failure` is an `undefined` result carrying `zero_reason`, because a
    ratio with no denominator is not 0.
    """
    if not denominator:
        return None, "", undefined(zero_reason)
    value = round(numerator / denominator, digits)
    step = f"{label} = {_num(numerator)} / {_num(denominator)} = {value}"
    return value, step, None


def _num(value: float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(int(value))


def spans_of(matches: Iterable[dict], label: str = "") -> list[dict]:
    return [span(m["start"], m["end"], label or m.get("lexicon", "")) for m in matches]


def quote_list(items: Iterable[str], limit: int = 8) -> str:
    items = list(items)
    shown = ", ".join(repr(i) for i in items[:limit])
    if len(items) > limit:
        shown += f", … (+{len(items) - limit} more)"
    return shown or "none"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def scale(value: float, lo: float, hi: float) -> float:
    """Map `value` from [lo, hi] onto [0, 1], clamped."""
    if hi == lo:
        return 0.0
    return clamp01((float(value) - lo) / (hi - lo))
