"""Temporal-constraint features."""

from __future__ import annotations

from . import lexicons as lex
from .doc import _DATE_LIKE_RE, _YEAR_RE
from .registry import not_applicable, ok, register, span
from .util import quote_list


def _years(doc) -> list[tuple[int, int, int]]:
    found = []
    for match in _YEAR_RE.finditer(doc.text):
        found.append((int(match.group(1)), match.start(), match.end()))
    return found


def _temporal_matches(doc) -> dict[str, list[dict]]:
    return {
        "relative": lex.TEMPORAL_RELATIVE.find(doc.text),
        "range": lex.TEMPORAL_RANGE.find(doc.text),
        "months": lex.MONTHS.find(doc.text),
        "weekdays": lex.WEEKDAYS.find(doc.text),
        "dates": [
            {"start": m.start(), "end": m.end(), "text": m.group(0), "lexicon": "date_pattern"}
            for m in _DATE_LIKE_RE.finditer(doc.text)
        ],
        "years": [
            {"start": s, "end": e, "text": doc.text[s:e], "lexicon": "year_pattern"}
            for _, s, e in _years(doc)
        ],
    }


@register(
    "has_temporal_constraint",
    group="temporal",
    dtype="bool",
    summary="Whether the prompt restricts the answer in time.",
    formula=(
        "true if any of these match: the temporal-range lexicon (before, after, since, "
        "until, between, during, prior to, over the past ...), the relative-recency "
        "lexicon (latest, current, recent, as of ...), a four-digit year from 1000 to "
        "2099 including decades like 1990s, a month or weekday name, or a date pattern "
        "such as 2023-04-01, 4/1/23 or Q3 2024"
    ),
    why="The right topic with the wrong date is one of the most common retrieval misses.",
    value_range="True / False",
    example="Treatments after 2020",
    expected=True,
    tier=1,
    rank=24,
    rank_reason="Time filters are rarely expressed in the chunk text, so the retriever cannot honour them.",
)
def has_temporal_constraint(doc, ctx):
    matches = _temporal_matches(doc)
    active = {kind: hits for kind, hits in matches.items() if hits}
    steps = [
        f"{kind}: {quote_list([h['text'] for h in hits])}" for kind, hits in active.items()
    ]
    all_spans = [
        span(h["start"], h["end"], kind) for kind, hits in active.items() for h in hits
    ]
    return ok(
        bool(active),
        *steps or ["no temporal expression matched"],
        spans=all_spans,
        hits=sorted({h.get("lexicon", "") for hits in active.values() for h in hits}),
    )


@register(
    "temporal_expression_count",
    group="temporal",
    dtype="int",
    summary="How many separate time expressions the prompt contains.",
    formula="total number of matches across the temporal lexicons, year pattern and date patterns",
    why="Several time expressions usually mean a range or a comparison across periods, which needs documents from each.",
    value_range=">= 0",
    example="Compare sales between 2019 and 2023",
    expected=3,
)
def temporal_expression_count(doc, ctx):
    matches = _temporal_matches(doc)
    total = sum(len(hits) for hits in matches.values())
    steps = [f"{kind} = {len(hits)}" for kind, hits in matches.items() if hits]
    return ok(total, *steps, f"total = {total}")


@register(
    "temporal_type",
    group="temporal",
    dtype="label",
    summary="What kind of time constraint the prompt uses.",
    formula=(
        "range if two or more years or a range word plus a year; absolute if a year, "
        "date or month appears; relative if only recency words appear; none otherwise"
    ),
    why="Absolute dates can be matched literally; relative ones depend on when the index was built, which is where staleness bites.",
    value_range="absolute / relative / range / none",
    example="What are the latest treatments?",
    expected="relative",
)
def temporal_type(doc, ctx):
    matches = _temporal_matches(doc)
    years = matches["years"]
    absolute = years or matches["dates"] or matches["months"]
    relative = matches["relative"]
    range_words = matches["range"]
    if len(years) >= 2 or (range_words and absolute):
        value = "range"
        why = f"{len(years)} years and {len(range_words)} range word(s)"
    elif absolute:
        value = "absolute"
        why = "a year, date or month name is present"
    elif relative:
        value = "relative"
        why = f"only recency words: {quote_list([h['text'] for h in relative])}"
    elif range_words:
        value = "relative"
        why = f"range words without a date: {quote_list([h['text'] for h in range_words])}"
    else:
        return not_applicable(
            "the prompt contains no time expression at all", value="none"
        )
    return ok(value, why)


@register(
    "has_relative_recency",
    group="temporal",
    dtype="bool",
    summary="Whether the prompt asks for whatever is newest rather than a fixed date.",
    formula="match against the relative-recency lexicon (latest, newest, current, currently, now, today, recent, as of, up to date, this year ...)",
    why="This is the classic stale-index failure: the corpus cannot know what `latest` means, so it returns whatever was true when it was built.",
    value_range="True / False",
    example="What are the latest treatments?",
    expected=True,
    tier=1,
    rank=23,
    rank_reason="Directly names the failure mode where the index is simply out of date.",
)
def has_relative_recency(doc, ctx):
    matches = lex.TEMPORAL_RELATIVE.find(doc.text)
    return ok(
        bool(matches),
        f"recency words: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "recency") for m in matches],
        hits=[lex.TEMPORAL_RELATIVE.name] if matches else [],
    )


@register(
    "year_count",
    group="temporal",
    dtype="int",
    summary="How many four-digit years the prompt names.",
    formula=r"count of matches of (1[0-9]|20)\d\d, optionally followed by s for decades",
    why="Years are exact filters; a chunk about the right subject in the wrong year does not answer the question.",
    value_range=">= 0",
    example="Compare sales between 2019 and 2023",
    expected=2,
)
def year_count(doc, ctx):
    years = _years(doc)
    return ok(
        len(years),
        f"years: {quote_list([str(y) for y, _, _ in years])}",
        spans=[span(s, e, "year") for _, s, e in years],
    )


@register(
    "year_min",
    group="temporal",
    dtype="int",
    summary="The earliest year named in the prompt.",
    formula="min of the years found",
    why="Sets the lower edge of the time window the retrieval has to respect.",
    value_range="1000 to 2099, or not applicable",
    status_rules=["not applicable when the prompt names no year, because there is no minimum to report"],
    example="Compare sales between 2019 and 2023",
    expected=2019,
)
def year_min(doc, ctx):
    years = _years(doc)
    if not years:
        return not_applicable("the prompt names no year, so there is no earliest year")
    value = min(y for y, _, _ in years)
    return ok(value, f"years found: {sorted(y for y, _, _ in years)}", f"min = {value}")


@register(
    "year_max",
    group="temporal",
    dtype="int",
    summary="The latest year named in the prompt.",
    formula="max of the years found",
    why="Sets the upper edge of the time window; a corpus that ends earlier cannot answer.",
    value_range="1000 to 2099, or not applicable",
    status_rules=["not applicable when the prompt names no year"],
    example="Compare sales between 2019 and 2023",
    expected=2023,
)
def year_max(doc, ctx):
    years = _years(doc)
    if not years:
        return not_applicable("the prompt names no year, so there is no latest year")
    value = max(y for y, _, _ in years)
    return ok(value, f"years found: {sorted(y for y, _, _ in years)}", f"max = {value}")


@register(
    "year_span",
    group="temporal",
    dtype="int",
    summary="How many years the prompt's time window covers.",
    formula="year_max - year_min",
    why="A wide window needs documents from several periods, so one retrieval pass rarely covers all of it.",
    value_range=">= 0, or not applicable",
    status_rules=[
        "not applicable when the prompt names no year: a span of 0 would wrongly claim a single-year window"
    ],
    example="Compare sales between 2019 and 2023",
    expected=4,
    needs=["year_min", "year_max"],
    tier=1,
    rank=25,
    rank_reason="Distinguishes a single-date lookup from a multi-period synthesis.",
)
def year_span(doc, ctx):
    years = _years(doc)
    if not years:
        return not_applicable(
            "the prompt names no year, so the window has no width; reporting 0 here "
            "would look like a single-year constraint"
        )
    lo = min(y for y, _, _ in years)
    hi = max(y for y, _, _ in years)
    return ok(hi - lo, f"year_span = {hi} - {lo} = {hi - lo}")


@register(
    "has_date_like",
    group="temporal",
    dtype="bool",
    summary="Whether the prompt contains a full date or quarter.",
    formula="regex for 2023-04-01, 4/1/23, `April 1, 2023`, `1 April` and Q3 2024",
    why="A precise date is a hard filter that a chunk must state explicitly to satisfy.",
    value_range="True / False",
    example="What happened on 2023-04-01?",
    expected=True,
)
def has_date_like(doc, ctx):
    matches = list(_DATE_LIKE_RE.finditer(doc.text))
    return ok(
        bool(matches),
        f"date patterns: {quote_list([m.group(0) for m in matches])}",
        spans=[span(m.start(), m.end(), "date") for m in matches],
    )
