"""Composite risk scores.

Each one is a documented weighted blend of features above, and every term is
listed in the explanation so you can see exactly what drove the number.
"""

from __future__ import annotations

from .registry import ok, register
from .util import clamp01, r3, scale


def _blend(terms: dict[str, tuple[float, float]]):
    """terms maps label -> (weight, raw 0..1 signal)."""
    parts = {}
    total = 0.0
    for label, (weight, signal) in terms.items():
        part = weight * clamp01(signal)
        parts[label] = round(part, 3)
        total += part
    return r3(total), parts


@register(
    "retrieval_anchor_score",
    group="composite",
    dtype="float",
    summary="How well anchored the query is, from 0 (nothing to match on) to 1 (richly anchored).",
    formula=(
        "0.45 * min(1, anchor_count / 2) + 0.30 * min(1, anchor_density / 0.5) + "
        "0.25 * min(1, distinct_entity_types / 3)"
    ),
    why="The positive counterpart of the underspecification score: high values are the cases retrieval usually gets right.",
    value_range="0 to 1",
    example="Who wrote The Hobbit?",
    expected=0.608,
    needs=["anchor_count", "anchor_density", "distinct_entity_types"],
)
def retrieval_anchor_score(doc, ctx):
    value, parts = _blend(
        {
            "anchor_count (0.45)": (0.45, ctx.number("anchor_count") / 2),
            "anchor_density (0.30)": (0.30, ctx.number("anchor_density") / 0.5),
            "distinct_entity_types (0.25)": (0.25, ctx.number("distinct_entity_types") / 3),
        }
    )
    return ok(
        value,
        *[f"{label} = {part}" for label, part in parts.items()],
        f"retrieval_anchor_score = {value}",
        detail={"terms": parts},
    )


@register(
    "multi_need_score",
    group="composite",
    dtype="float",
    summary="How many separate information needs the query bundles, on a 0 to 1 scale.",
    formula=(
        "0.35 * min(1, (sub_question_count - 1) / 2) + 0.25 * is_comparison + "
        "0.25 * min(1, (hop_estimate - 1) / 3) + 0.15 * requires_synthesis_flag"
    ),
    why="One top-k retrieval has a fixed budget; the more needs share it, the more likely one goes unserved.",
    value_range="0 to 1",
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=0.592,
    needs=["sub_question_count", "is_comparison", "hop_estimate", "requires_synthesis_flag"],
)
def multi_need_score(doc, ctx):
    value, parts = _blend(
        {
            "sub_questions (0.35)": (0.35, (ctx.number("sub_question_count") - 1) / 2),
            "is_comparison (0.25)": (0.25, 1.0 if ctx.flag("is_comparison") else 0.0),
            "hops (0.25)": (0.25, (ctx.number("hop_estimate") - 1) / 3),
            "requires_synthesis (0.15)": (
                0.15,
                1.0 if ctx.flag("requires_synthesis_flag") else 0.0,
            ),
        }
    )
    return ok(
        value,
        *[f"{label} = {part}" for label, part in parts.items()],
        f"multi_need_score = {value}",
        detail={"terms": parts},
    )


@register(
    "lexical_rarity_score",
    group="composite",
    dtype="float",
    summary="How likely the query's vocabulary is to be absent from a corpus, on a 0 to 1 scale.",
    formula=(
        "0.40 * (1 - min(1, mean_word_zipf / 5)) + 0.25 * min(1, rare_word_ratio / 0.3) + "
        "0.20 * min(1, (tokens_per_word - 1) / 1.5) + 0.15 * min(1, jargon_ratio / 0.5). "
        "The two ratio terms are multiplied by a confidence factor "
        "min(1, content_word_count / 5), because a ratio measured over two content words "
        "is not evidence of anything. Terms whose inputs are unavailable count as 0."
    ),
    why="Vocabulary mismatch is the main cause of lexical retrieval failure and a significant one for embeddings.",
    value_range="0 to 1",
    example="What is the dose of acetylsalicylic acid?",
    expected=0.246,
    needs=[
        "mean_word_zipf", "rare_word_ratio", "tokens_per_word", "jargon_ratio",
        "content_word_count",
    ],
)
def lexical_rarity_score(doc, ctx):
    mean_zipf = ctx.number("mean_word_zipf", 5.0)
    confidence = clamp01(ctx.number("content_word_count") / 5)
    value, parts = _blend(
        {
            "mean_word_zipf (0.40)": (0.40, 1 - (mean_zipf / 5)),
            "rare_word_ratio (0.25)": (0.25, confidence * ctx.number("rare_word_ratio") / 0.3),
            "tokens_per_word (0.20)": (0.20, (ctx.number("tokens_per_word", 1.0) - 1) / 1.5),
            "jargon_ratio (0.15)": (0.15, confidence * ctx.number("jargon_ratio") / 0.5),
        }
    )
    return ok(
        value,
        f"confidence factor from {int(ctx.number('content_word_count'))} content words = {round(confidence, 2)}",
        *[f"{label} = {part}" for label, part in parts.items()],
        f"lexical_rarity_score = {value}",
        detail={"terms": parts, "confidence": round(confidence, 3)},
    )


@register(
    "temporal_risk_score",
    group="composite",
    dtype="float",
    summary="How much the query depends on the index being current, on a 0 to 1 scale.",
    formula=(
        "0.50 * has_relative_recency + 0.25 * has_temporal_constraint + "
        "0.25 * min(1, year_span / 10), with year_span treated as 0 when no year is named"
    ),
    why="Relative recency is the failure mode no amount of good ranking can fix, because the corpus simply does not contain the newer fact.",
    value_range="0 to 1",
    example="What are the latest treatments?",
    expected=0.75,
    needs=["has_relative_recency", "has_temporal_constraint", "year_span"],
)
def temporal_risk_score(doc, ctx):
    value, parts = _blend(
        {
            "has_relative_recency (0.50)": (0.50, 1.0 if ctx.flag("has_relative_recency") else 0.0),
            "has_temporal_constraint (0.25)": (
                0.25,
                1.0 if ctx.flag("has_temporal_constraint") else 0.0,
            ),
            "year_span (0.25)": (0.25, ctx.number("year_span") / 10),
        }
    )
    return ok(
        value,
        *[f"{label} = {part}" for label, part in parts.items()],
        f"temporal_risk_score = {value}",
        detail={"terms": parts},
    )


@register(
    "retrieval_difficulty_score",
    group="composite",
    dtype="float",
    summary="Overall estimated risk that retrieval fails for this prompt, from 0 to 1.",
    formula=(
        "a noisy-OR over seven independent failure modes: "
        "risk = 1 - product(1 - weight * signal) with "
        "underspecification 0.90, language mismatch 0.70 (1 when the prompt is not in "
        "English), multi-need 0.55, temporal 0.55, lexical rarity 0.50, "
        "boilerplate dilution 0.30 (signal = 1 - core_question_ratio) and "
        "negation-or-exclusion 0.30 (0.5 each). "
        "A weighted average is deliberately not used: these modes are alternatives, not "
        "ingredients, so one severe mode alone should produce a high score. "
        "The language term assumes an English corpus, which is the common case and the "
        "only one the prompt text can speak to; drop that term if your index is "
        "multilingual. Weights are a judgement call, not a fitted model, because no "
        "labelled retrieval-failure data exists for this project yet."
    ),
    why="One headline number for triage and for sorting a dataset by expected difficulty.",
    value_range="0 to 1 (low below 0.33, medium below 0.6, high above)",
    status_rules=["always computed; individual modes fall back to 0 when their input is unavailable"],
    example="What about it?",
    expected=0.638,
    needs=[
        "underspecification_score", "multi_need_score", "lexical_rarity_score",
        "temporal_risk_score", "core_question_ratio", "contains_negation",
        "has_exclusion", "is_english",
    ],
)
def retrieval_difficulty_score(doc, ctx):
    logic_penalty = 0.5 * (1.0 if ctx.flag("contains_negation") else 0.0) + 0.5 * (
        1.0 if ctx.flag("has_exclusion") else 0.0
    )
    english = ctx.result("is_english")
    not_english = 1.0 if (english is not None and english.value is False) else 0.0
    modes = {
        "underspecification (0.90)": (0.90, ctx.number("underspecification_score")),
        "language mismatch (0.70)": (0.70, not_english),
        "multi_need (0.55)": (0.55, ctx.number("multi_need_score")),
        "temporal_risk (0.55)": (0.55, ctx.number("temporal_risk_score")),
        "lexical_rarity (0.50)": (0.50, ctx.number("lexical_rarity_score")),
        "boilerplate dilution (0.30)": (0.30, 1 - ctx.number("core_question_ratio", 1.0)),
        "negation or exclusion (0.30)": (0.30, logic_penalty),
    }
    survival = 1.0
    parts = {}
    for label, (weight, signal) in modes.items():
        mode_risk = weight * clamp01(signal)
        parts[label] = round(mode_risk, 3)
        survival *= 1 - mode_risk
    value = r3(1 - survival)
    band = "low" if value < 0.33 else "medium" if value < 0.6 else "high"
    return ok(
        value,
        *[f"{label} contributes {part}" for label, part in parts.items()],
        f"1 - ({' x '.join(f'{1 - p:.3f}' for p in parts.values())}) = {value}",
        f"retrieval_difficulty_score = {value} ({band} risk)",
        detail={"terms": parts, "band": band},
    )
