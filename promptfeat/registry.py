"""The feature registry: one declaration per feature, used by everything else.

`extract_features`, `FEATURES.md`, the Markdown report and the web UI are all
generated from this registry, so the spec and the code cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# --- statuses ---------------------------------------------------------------
# A feature that cannot be honestly measured must say so. Writing 0 where the
# truth is "undefined" would teach a downstream model something false.

OK = "ok"
NOT_APPLICABLE = "not_applicable"  # nothing of this kind is present in the prompt
UNDEFINED = "undefined"  # the formula has no value here (e.g. divide by zero)
UNRELIABLE = "unreliable"  # computed, but outside the metric's valid range
UNAVAILABLE = "unavailable"  # the backend needed for it is not installed

STATUS_LABELS = {
    OK: "ok",
    NOT_APPLICABLE: "not applicable",
    UNDEFINED: "undefined",
    UNRELIABLE: "unreliable",
    UNAVAILABLE: "unavailable",
}


@dataclass
class FeatureResult:
    """A value plus everything needed to explain how it was reached."""

    value: Any
    status: str = OK
    reason: str = ""
    steps: list[str] = field(default_factory=list)
    spans: list[dict] = field(default_factory=list)
    lexicon_hits: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def computed(self) -> bool:
        return self.status in (OK, UNRELIABLE)


def span(start: int, end: int, label: str = "") -> dict:
    return {"start": int(start), "end": int(end), "label": label}


def ok(
    value: Any,
    *steps: str,
    spans: Iterable[dict] | None = None,
    hits: Iterable[str] | None = None,
    detail: dict | None = None,
) -> FeatureResult:
    return FeatureResult(
        value=value,
        status=OK,
        steps=[s for s in steps if s],
        spans=list(spans or []),
        lexicon_hits=list(hits or []),
        detail=dict(detail or {}),
    )


def not_applicable(reason: str, *steps: str, value: Any = None) -> FeatureResult:
    return FeatureResult(
        value=value, status=NOT_APPLICABLE, reason=reason, steps=[s for s in steps if s]
    )


def undefined(reason: str, *steps: str) -> FeatureResult:
    return FeatureResult(
        value=None, status=UNDEFINED, reason=reason, steps=[s for s in steps if s]
    )


def unreliable(
    value: Any,
    reason: str,
    *steps: str,
    spans: Iterable[dict] | None = None,
    detail: dict | None = None,
) -> FeatureResult:
    return FeatureResult(
        value=value,
        status=UNRELIABLE,
        reason=reason,
        steps=[s for s in steps if s],
        spans=list(spans or []),
        detail=dict(detail or {}),
    )


def unavailable(backend: str, reason: str = "") -> FeatureResult:
    from . import nlp

    detail_reason = reason or nlp.backend_error(backend)
    text = f"needs the {backend} backend"
    if detail_reason:
        text += f": {detail_reason}"
    text += f". Install it with: {nlp.install_hint(backend)}"
    return FeatureResult(value=None, status=UNAVAILABLE, reason=text)


# --- groups -----------------------------------------------------------------

GROUPS: list[tuple[str, str, str]] = [
    ("size", "Size and shape", "How much text the retriever has to work with."),
    (
        "readability",
        "Readability and lexical difficulty",
        "How hard the wording is, and how varied the vocabulary is.",
    ),
    (
        "rarity",
        "Rarity and vocabulary",
        "Whether the prompt uses words a corpus is likely to share.",
    ),
    (
        "structure",
        "Question structure",
        "What kind of ask it is and how many asks it contains.",
    ),
    (
        "ambiguity",
        "Ambiguity and underspecification",
        "Whether the prompt says enough to be answerable at all.",
    ),
    (
        "anchors",
        "Retrieval anchors",
        "Concrete strings a retriever can latch onto: names, numbers, quotes, ids.",
    ),
    ("temporal", "Temporal constraints", "How the prompt restricts the answer in time."),
    (
        "reasoning",
        "Reasoning and multi-hop",
        "Whether answering needs several documents combined.",
    ),
    ("domain", "Domain and register", "Subject area, language and politeness noise."),
    (
        "promptcraft",
        "Prompt-engineering artifacts",
        "Instruction and example scaffolding wrapped around the real question.",
    ),
    (
        "composite",
        "Composite risk scores",
        "Blends of the features above, each with its terms shown.",
    ),
    (
        "exam",
        "Exam-item traps",
        "Multiple-choice wording that makes an LLM miss: except/NOT, best-answer, long hypos.",
    ),
]

GROUP_ORDER = [key for key, _, _ in GROUPS]
GROUP_TITLES = {key: title for key, title, _ in GROUPS}
GROUP_BLURBS = {key: blurb for key, _, blurb in GROUPS}


# --- features ---------------------------------------------------------------


@dataclass
class Feature:
    name: str
    group: str
    dtype: str
    summary: str  # what we see
    formula: str  # how it is calculated
    why: str  # why it matters for retrieval
    fn: Callable
    backend: str = "python"
    value_range: str = ""
    status_rules: list[str] = field(default_factory=list)
    example: str = ""
    expected: Any = None
    has_expected: bool = False
    tier: int = 2
    rank: int | None = None
    rank_reason: str = ""
    needs: list[str] = field(default_factory=list)

    @property
    def group_title(self) -> str:
        return GROUP_TITLES.get(self.group, self.group)


REGISTRY: dict[str, Feature] = {}
_UNSET = object()


def register(
    name: str,
    *,
    group: str,
    dtype: str,
    summary: str,
    formula: str,
    why: str,
    backend: str = "python",
    value_range: str = "",
    status_rules: Iterable[str] = (),
    example: str = "",
    expected: Any = _UNSET,
    tier: int = 2,
    rank: int | None = None,
    rank_reason: str = "",
    needs: Iterable[str] = (),
) -> Callable:
    """Declare a feature. The decorated function gets `(doc, ctx)`."""

    def decorator(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"duplicate feature name: {name}")
        if group not in GROUP_TITLES:
            raise ValueError(f"unknown group {group!r} for feature {name!r}")
        REGISTRY[name] = Feature(
            name=name,
            group=group,
            dtype=dtype,
            summary=summary,
            formula=formula,
            why=why,
            fn=fn,
            backend=backend,
            value_range=value_range,
            status_rules=list(status_rules),
            example=example,
            expected=None if expected is _UNSET else expected,
            has_expected=expected is not _UNSET,
            tier=tier,
            rank=rank,
            rank_reason=rank_reason,
            needs=list(needs),
        )
        return fn

    return decorator


class Context:
    """Access to already-computed results, for composite features."""

    def __init__(self) -> None:
        self.results: dict[str, FeatureResult] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.results

    def result(self, name: str) -> FeatureResult | None:
        return self.results.get(name)

    def value(self, name: str, default: Any = None) -> Any:
        res = self.results.get(name)
        if res is None or res.value is None:
            return default
        return res.value

    def number(self, name: str, default: float = 0.0) -> float:
        value = self.value(name, default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def flag(self, name: str) -> bool:
        return bool(self.value(name, False))


def feature_names(tier: int | None = None) -> list[str]:
    items = REGISTRY.values()
    if tier is not None:
        items = [f for f in items if f.tier == tier]
    return [f.name for f in _sorted(items)]


def _sorted(items: Iterable[Feature]) -> list[Feature]:
    return sorted(
        items,
        key=lambda f: (GROUP_ORDER.index(f.group) if f.group in GROUP_ORDER else 99,),
    )


def top_features() -> list[Feature]:
    """Tier-1 features in rank order."""
    ranked = [f for f in REGISTRY.values() if f.tier == 1]
    return sorted(ranked, key=lambda f: (f.rank if f.rank is not None else 999, f.name))


def features_by_group() -> list[tuple[str, list[Feature]]]:
    grouped: list[tuple[str, list[Feature]]] = []
    for key in GROUP_ORDER:
        members = [f for f in REGISTRY.values() if f.group == key]
        if members:
            grouped.append((key, members))
    return grouped
