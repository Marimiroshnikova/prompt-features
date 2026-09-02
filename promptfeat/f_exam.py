"""Exam-item traps: why an LLM misses a multiple-choice question.

These are not retrieval features. They come from reading the 805 misses on
the 280-question MMLU-Pro grid: judgment wording, except/NOT asks, and long
hypos fail more; short definition and formula items fail less.
"""

from __future__ import annotations

import re
import statistics

from .registry import not_applicable, ok, register, span
from .util import quote_list

_OPTION_LINE = re.compile(r"(?m)^(?:Options:\s*)?([A-J])\.\s+(.+?)\s*$")

_EXCEPT_ASK = re.compile(
    r"(?i)\b(?:"
    r"which of the following is not|which is not|is not one of|"
    r"all of the following except|except that|except|"
    r"least likely|least appropriate|incorrect|is false"
    r")\b"
)

_JUDGMENT = re.compile(
    r"(?i)\b(?:"
    r"most likely|best describes|most nearly|most appropriate|"
    r"most accurate|best characterized|best explained|"
    r"most closely|closest to"
    r")\b"
)

_HYPOTHETICAL = re.compile(
    r"(?i)\b(?:"
    r"suppose|assume that|assuming that|"
    r"a defendant|a plaintiff|a prosecutor|"
    r"imaginary"
    r")\b"
    r"|^(?:if)\b"
)

_DEFINITION = re.compile(r"(?i)^(?:what is|what are|what does)\b")

_FORMULA = re.compile(
    r"(?i)\$|%\b|compounded|deposit of|interest rate|how many hours|"
    r"find the (?:number|amount|value|sum)"
)

_CANNOT_DET = re.compile(
    r"(?i)cannot be determined|not enough information|insufficient information|"
    r"none of the above|all of the above"
)


def _split_stem_options(text: str) -> tuple[str, list[tuple[str, str, int, int]]]:
    """Return (stem, [(letter, option_text, start, end), ...])."""
    matches = list(_OPTION_LINE.finditer(text))
    if not matches:
        return text.strip(), []
    stem = text[: matches[0].start()].strip()
    opts = [
        (m.group(1), m.group(2).strip(), m.start(), m.end()) for m in matches
    ]
    return stem, opts


def _stem(doc) -> str:
    stem, _ = _split_stem_options(doc.text)
    return stem


@register(
    "is_except_ask",
    group="exam",
    dtype="bool",
    summary="Whether the question is an except / NOT / least / incorrect item.",
    formula=(
        "true if the stem matches: which of the following is not, which is not, "
        "is not one of, all of the following except, except, least likely, "
        "least appropriate, incorrect, is false"
    ),
    why="The model has to invert the matching option. On this grid, except/NOT items failed more often than ordinary asks.",
    value_range="True / False",
    example="Which of the following is NOT a noble gas?",
    expected=True,
)
def is_except_ask(doc, ctx):
    stem = _stem(doc)
    hits = list(_EXCEPT_ASK.finditer(stem))
    return ok(
        bool(hits),
        f"stem matches: {quote_list([h.group(0) for h in hits]) or 'none'}",
        spans=[span(h.start(), h.end(), "except_ask") for h in hits],
    )


@register(
    "is_best_answer_judgment",
    group="exam",
    dtype="bool",
    summary="Whether the item asks for the best / most likely / most nearly answer.",
    formula=(
        "true if the stem matches most likely, best describes, most nearly, "
        "most appropriate, most accurate, best characterized, best explained, "
        "most closely, closest to"
    ),
    why="There is no unique fact; several options are plausible. These items failed at 44% vs 20% on the pilot grid.",
    value_range="True / False",
    example="Which of the following best describes the role of the Fed?",
    expected=True,
)
def is_best_answer_judgment(doc, ctx):
    stem = _stem(doc)
    hits = list(_JUDGMENT.finditer(stem))
    return ok(
        bool(hits),
        f"judgment phrasing: {quote_list([h.group(0) for h in hits]) or 'none'}",
        spans=[span(h.start(), h.end(), "judgment") for h in hits],
    )


@register(
    "is_hypothetical_scenario",
    group="exam",
    dtype="bool",
    summary="Whether the item is a constructed scenario (if / suppose / a defendant).",
    formula=(
        "true if the stem starts with If, or matches suppose, assume that, "
        "a defendant, a plaintiff, a prosecutor, imaginary"
    ),
    why="The model must track a made-up fact pattern. Long law hypos on this grid were missed by every model we ran.",
    value_range="True / False",
    example="A defendant was arrested and charged with attempted murder. Who must prove insanity?",
    expected=True,
)
def is_hypothetical_scenario(doc, ctx):
    stem = _stem(doc)
    hits = list(_HYPOTHETICAL.finditer(stem))
    return ok(
        bool(hits),
        f"scenario cues: {quote_list([h.group(0) for h in hits]) or 'none'}",
        spans=[span(h.start(), h.end(), "scenario") for h in hits],
    )


@register(
    "is_definition_ask",
    group="exam",
    dtype="bool",
    summary="Whether the item is a short What is / What are definition.",
    formula="true if the stem starts with What is, What are, or What does",
    why="Definition items on this grid failed less (16%) than the rest (21%).",
    value_range="True / False",
    example="What is Market Socialism?",
    expected=True,
)
def is_definition_ask(doc, ctx):
    stem = _stem(doc)
    hit = _DEFINITION.search(stem.strip())
    return ok(
        bool(hit),
        f"stem starts with a definition ask: {bool(hit)}",
        spans=[span(hit.start(), hit.end(), "definition")] if hit else [],
    )


@register(
    "is_formula_setup",
    group="exam",
    dtype="bool",
    summary="Whether the item is a standard formula / deposit / rate calculation.",
    formula="true if the stem contains $, %, compounded, deposit of, interest rate, or find the amount/value/sum",
    why="Closed-form money and rate problems were the easiest items on this grid (every model got the $500 monthly deposit right).",
    value_range="True / False",
    example="If a deposit of $500 is made in an account that pays 8% compounded monthly, what is the amount after five years?",
    expected=True,
)
def is_formula_setup(doc, ctx):
    stem = _stem(doc)
    hits = list(_FORMULA.finditer(stem))
    return ok(
        bool(hits),
        f"formula cues: {quote_list([h.group(0) for h in hits]) or 'none'}",
        spans=[span(h.start(), h.end(), "formula") for h in hits],
    )


@register(
    "stem_word_count",
    group="exam",
    dtype="int",
    summary="Word count of the question stem, ignoring lettered options.",
    formula="len(stem.split()) after stripping A.–J. option lines",
    why="The longest quartile of stems failed at 26%, the second-shortest at 13%. Length here is the hypo, not the options.",
    value_range=">= 0",
    example="What is Market Socialism?\nA. A system\nB. A tax",
    expected=4,
)
def stem_word_count(doc, ctx):
    stem = _stem(doc)
    n = len(stem.split())
    return ok(n, f"stem words = {n}")


@register(
    "is_long_scenario",
    group="exam",
    dtype="bool",
    summary="Whether the stem is a long hypo (80 or more words).",
    formula="stem_word_count >= 80",
    why="Stems of 80+ words failed at 29% vs 19% for shorter ones on this grid.",
    value_range="True / False",
    example="What is 2+2?\nA. 3\nB. 4",
    expected=False,
    needs=["stem_word_count"],
)
def is_long_scenario(doc, ctx):
    n = ctx.number("stem_word_count") or 0
    return ok(n >= 80, f"stem_word_count = {n}; long if >= 80")


@register(
    "mc_option_count",
    group="exam",
    dtype="int",
    summary="How many lettered A–J options are printed.",
    formula="count of lines matching 'X. …' for X in A–J",
    why="More options is more chance to pick a near-miss; reported only when options are actually printed.",
    value_range=">= 2 when options are present",
    status_rules=["not applicable when the prompt has no lettered A–J options"],
    example="What is 2+2?\nA. 3\nB. 4\nC. 5",
    expected=3,
)
def mc_option_count(doc, ctx):
    _, opts = _split_stem_options(doc.text)
    if len(opts) < 2:
        return not_applicable(
            "no lettered A–J options, so there is no option count",
            value=None,
        )
    return ok(len(opts), f"{len(opts)} options: {', '.join(l for l, *_ in opts)}")


@register(
    "option_mean_chars",
    group="exam",
    dtype="float",
    summary="Average character length of the lettered options.",
    formula="mean(len(option_text)) over A–J lines",
    why="Long options are often mini-essays; the model has to compare several near-paragraphs.",
    value_range="> 0 when options are present",
    status_rules=["not applicable when the prompt has no lettered options"],
    example="What is 2+2?\nA. three\nB. four",
    expected=4.5,
)
def option_mean_chars(doc, ctx):
    _, opts = _split_stem_options(doc.text)
    if len(opts) < 2:
        return not_applicable("no lettered options", value=None)
    lengths = [len(t) for _, t, *_ in opts]
    val = sum(lengths) / len(lengths)
    return ok(val, f"lengths {lengths}; mean {val:.2f}")


@register(
    "option_length_spread",
    group="exam",
    dtype="float",
    summary="How uneven option lengths are (sample standard deviation of characters).",
    formula="stdev(len(option_text)); 0 if fewer than 2 options",
    why="One long option next to short ones is a format cue models over-weight.",
    value_range=">= 0 when options are present",
    status_rules=["not applicable when the prompt has no lettered options"],
    example="What?\nA. no\nB. a much longer option text",
    expected=16.263455967290593,
)
def option_length_spread(doc, ctx):
    _, opts = _split_stem_options(doc.text)
    if len(opts) < 2:
        return not_applicable("no lettered options", value=None)
    lengths = [len(t) for _, t, *_ in opts]
    if len(lengths) < 2:
        return not_applicable("need two options to measure spread", value=None)
    val = statistics.stdev(lengths)
    return ok(val, f"lengths {lengths}; stdev {val:.3f}")


@register(
    "has_escape_option",
    group="exam",
    dtype="bool",
    summary="Whether an option is all/none of the above or cannot be determined.",
    formula=(
        "true if any option matches all of the above, none of the above, "
        "cannot be determined, not enough information, insufficient information"
    ),
    why="Escape options are a known trap; the one 'cannot be determined' item on this grid was missed by every model.",
    value_range="True / False",
    status_rules=["not applicable when the prompt has no lettered options"],
    example="What is 2+2?\nA. 4\nB. Cannot be determined",
    expected=True,
)
def has_escape_option(doc, ctx):
    _, opts = _split_stem_options(doc.text)
    if len(opts) < 2:
        return not_applicable("no lettered options", value=None)
    hits = []
    spans = []
    for letter, text, start, end in opts:
        if _CANNOT_DET.search(text):
            hits.append(f"{letter}. {text}")
            spans.append(span(start, end, "escape"))
    return ok(bool(hits), f"escape options: {quote_list(hits) or 'none'}", spans=spans)
