"""Compute and explain features for one prompt."""

from __future__ import annotations

from typing import Any

from . import nlp
from .doc import PromptDoc
from .registry import (
    GROUP_BLURBS,
    GROUP_TITLES,
    OK,
    REGISTRY,
    STATUS_LABELS,
    Context,
    Feature,
    FeatureResult,
    features_by_group,
    top_features,
)

# Feature modules are imported for their registration side effects. The order
# below is only a starting point; _order() then sorts by declared dependencies.
from . import f_size  # noqa: F401
from . import f_lexical  # noqa: F401
from . import f_rarity  # noqa: F401
from . import f_structure  # noqa: F401
from . import f_anchors  # noqa: F401
from . import f_temporal  # noqa: F401
from . import f_reasoning  # noqa: F401
from . import f_domain  # noqa: F401
from . import f_promptcraft  # noqa: F401
from . import f_ambiguity  # noqa: F401
from . import f_composite  # noqa: F401
from . import f_exam  # noqa: F401

_order_cache: list[str] | None = None


def compute_order() -> list[str]:
    """Registration order, adjusted so a feature runs after everything it needs."""
    global _order_cache
    if _order_cache is not None:
        return _order_cache
    pending = list(REGISTRY)
    done: set[str] = set()
    ordered: list[str] = []
    guard = 0
    while pending and guard < 10_000:
        guard += 1
        progressed = False
        for name in list(pending):
            needs = [n for n in REGISTRY[name].needs if n in REGISTRY]
            if all(n in done for n in needs):
                ordered.append(name)
                done.add(name)
                pending.remove(name)
                progressed = True
        if not progressed:
            # A dependency cycle: fall back to registration order for the rest.
            ordered.extend(pending)
            break
    _order_cache = ordered
    return ordered


def compute(prompt: str) -> tuple[PromptDoc, Context]:
    doc = PromptDoc(prompt)
    ctx = Context()
    for name in compute_order():
        feature = REGISTRY[name]
        try:
            result = feature.fn(doc, ctx)
        except Exception as exc:  # a broken feature must not break the rest
            result = FeatureResult(
                value=None,
                status="undefined",
                reason=f"the feature raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(result, FeatureResult):
            result = FeatureResult(value=result)
        ctx.results[name] = result
    return doc, ctx


def extract_features(
    prompt: str, *, tier: int | None = None, with_status: bool = False
) -> dict[str, Any]:
    """Flat dict of feature values, ready for a DataFrame.

    Values are `None` when a feature could not be honestly computed. Pass
    `with_status=True` to also get `<name>__status` and `<name>__reason`.
    """
    _, ctx = compute(prompt)
    names = [n for n in REGISTRY if tier is None or REGISTRY[n].tier == tier]
    out: dict[str, Any] = {}
    for name in names:
        result = ctx.results.get(name)
        if result is None:
            out[name] = None
            continue
        out[name] = result.value
        if with_status:
            out[f"{name}__status"] = result.status
            out[f"{name}__reason"] = result.reason
    return out


def feature_declaration(feature: Feature) -> dict:
    """Everything known about a feature before any prompt is seen."""
    return {
        "name": feature.name,
        "group": feature.group,
        "group_title": feature.group_title,
        "tier": feature.tier,
        "rank": feature.rank,
        "rank_reason": feature.rank_reason,
        "dtype": feature.dtype,
        "backend": feature.backend,
        "value_range": feature.value_range,
        "summary": feature.summary,
        "formula": feature.formula,
        "why": feature.why,
        "status_rules": feature.status_rules,
        "needs": feature.needs,
        "example": feature.example,
        "expected": feature.expected if feature.has_expected else None,
    }


def _feature_payload(feature: Feature, result: FeatureResult) -> dict:
    return {
        **feature_declaration(feature),
        "value": result.value,
        "display_value": _display(result.value),
        "status": result.status,
        "status_label": STATUS_LABELS.get(result.status, result.status),
        "reason": result.reason,
        "steps": result.steps,
        "spans": result.spans,
        "lexicon_hits": result.lexicon_hits,
        "detail": result.detail,
    }


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def explain_prompt(prompt: str) -> dict:
    """Everything the UI and the Markdown report need for one prompt."""
    doc, ctx = compute(prompt)
    features = [
        _feature_payload(REGISTRY[name], ctx.results[name])
        for name in REGISTRY
        if name in ctx.results
    ]
    by_name = {f["name"]: f for f in features}
    groups = []
    for key, members in features_by_group():
        groups.append(
            {
                "key": key,
                "title": GROUP_TITLES[key],
                "blurb": GROUP_BLURBS[key],
                "features": [by_name[f.name] for f in members if f.name in by_name],
            }
        )
    ranked = [by_name[f.name] for f in top_features() if f.name in by_name]
    statuses: dict[str, int] = {}
    for item in features:
        statuses[item["status"]] = statuses.get(item["status"], 0) + 1
    return {
        "prompt": prompt,
        "normalized": doc.text,
        "core_question": doc.core_text,
        "core_spans": [{"start": a, "end": b} for a, b in doc.core_spans],
        "segments": [
            {"kind": s.kind, "start": s.start, "end": s.end, "source": s.source}
            for s in doc.segments
        ],
        "sentences": [
            {"kind": s.kind, "start": s.start, "end": s.end, "text": s.text}
            for s in doc.sentences
        ],
        "features": features,
        "groups": groups,
        "top": ranked,
        "summary": {
            "feature_count": len(features),
            "computed": statuses.get(OK, 0),
            "statuses": statuses,
            "headline": by_name.get("retrieval_difficulty_score", {}).get("value"),
            "band": by_name.get("retrieval_difficulty_score", {})
            .get("detail", {})
            .get("band", ""),
            "category": by_name.get("question_category", {}).get("value"),
            "question_type": by_name.get("question_type", {}).get("value"),
            "words": by_name.get("question_length_words", {}).get("value"),
            "tokens": by_name.get("context_token_count", {}).get("value"),
        },
        "backends": nlp.backend_report(),
    }
