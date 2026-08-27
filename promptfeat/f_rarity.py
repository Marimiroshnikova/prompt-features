"""Rarity and vocabulary-mismatch features.

These answer one question: does this prompt use words a corpus is likely to
share? Retrieval fails when it does not.
"""

from __future__ import annotations

import re

from .registry import not_applicable, ok, register, span, unavailable, undefined
from .util import quote_list, r2, r3, ratio

RARE_ZIPF = 3.0
JARGON_ZIPF = 3.5
JARGON_MIN_LEN = 10

_ACRONYM_RE = re.compile(r"^[A-Z]{2,}(?:s|'s)?$")
_MIXED_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")
_DOTTED_RE = re.compile(r"^[A-Za-z][\w-]*\.[A-Za-z][\w.-]*$")

_ZIPF_SCALE = (
    "Zipf scale: about 7 for 'the', 4 for everyday words, below 3 for rare or "
    "technical words, 0 for words the corpus has never seen"
)


def _scored_words(doc):
    scores = doc.zipf_scores
    if scores is None:
        return None, None
    words = [w for w in doc.alpha_words if w.lower.strip("'-") in scores]
    return words, scores


@register(
    "mean_word_zipf",
    group="rarity",
    dtype="float",
    summary="Average corpus frequency of the words in the prompt.",
    formula="mean(wordfreq.zipf_frequency(word, 'en')) over word tokens",
    why="A low average means the whole query is phrased in vocabulary the index rarely uses, so lexical overlap collapses.",
    backend="wordfreq",
    value_range="0 to about 7 (lower is rarer)",
    status_rules=[
        "not applicable when the prompt has no word tokens",
        "unavailable when wordfreq is not installed",
    ],
    example="Who wrote The Hobbit?",
    expected=5.62,
    tier=1,
    rank=10,
    rank_reason="A real frequency measurement of whether the corpus is likely to use these words at all.",
)
def mean_word_zipf(doc, ctx):
    words, scores = _scored_words(doc)
    if words is None:
        return unavailable("wordfreq")
    if not words:
        return not_applicable("the prompt has no word tokens to score")
    values = [scores[w.lower.strip("'-")] for w in words]
    value = r2(sum(values) / len(values))
    detail = sorted(
        ({"word": w.text, "zipf": scores[w.lower.strip("'-")]} for w in words),
        key=lambda d: d["zipf"],
    )
    return ok(
        value,
        _ZIPF_SCALE,
        f"per-word Zipf: " + ", ".join(f"{d['word']}={d['zipf']}" for d in detail[:10]),
        f"mean = {value}",
        detail={"per_word": detail},
    )


@register(
    "min_word_zipf",
    group="rarity",
    dtype="float",
    summary="Corpus frequency of the rarest word in the prompt.",
    formula="min(wordfreq.zipf_frequency(word, 'en')) over word tokens",
    why="One very rare term often carries the whole information need; if the index lacks it, nothing else in the query will save the retrieval.",
    backend="wordfreq",
    value_range="0 to about 7 (lower is rarer)",
    status_rules=[
        "not applicable when the prompt has no word tokens",
        "unavailable when wordfreq is not installed",
    ],
    example="Who wrote The Hobbit?",
    expected=3.39,
)
def min_word_zipf(doc, ctx):
    words, scores = _scored_words(doc)
    if words is None:
        return unavailable("wordfreq")
    if not words:
        return not_applicable("the prompt has no word tokens to score")
    rarest = min(words, key=lambda w: scores[w.lower.strip("'-")])
    value = r2(scores[rarest.lower.strip("'-")])
    return ok(
        value,
        _ZIPF_SCALE,
        f"rarest word = {rarest.text!r} at Zipf {value}",
        spans=[span(rarest.start, rarest.end, "rarest word")],
    )


@register(
    "rare_word_count",
    group="rarity",
    dtype="int",
    summary=f"How many words have a Zipf frequency below {RARE_ZIPF}.",
    formula=f"count(wordfreq.zipf_frequency(word, 'en') < {RARE_ZIPF})",
    why="Rare words are the ones an embedding model has seen least and a keyword index is least likely to contain verbatim.",
    backend="wordfreq",
    value_range=">= 0",
    status_rules=["unavailable when wordfreq is not installed"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=1,
    tier=1,
    rank=12,
    rank_reason="Counts the specific terms most likely to be absent from the index.",
)
def rare_word_count(doc, ctx):
    words, scores = _scored_words(doc)
    if words is None:
        return unavailable("wordfreq")
    rare = [w for w in words if scores[w.lower.strip("'-")] < RARE_ZIPF]
    listed = ", ".join(f"{w.text}={scores[w.lower.strip(chr(39) + '-')]}" for w in rare)
    return ok(
        len(rare),
        _ZIPF_SCALE,
        f"words below Zipf {RARE_ZIPF}: {listed or 'none'}",
        spans=[span(w.start, w.end, "rare word") for w in rare],
    )


@register(
    "rare_word_ratio",
    group="rarity",
    dtype="float",
    summary=f"Share of words with a Zipf frequency below {RARE_ZIPF}.",
    formula="rare_word_count / number of scored word tokens",
    why="Normalises rarity by length, so a short jargon-heavy query is not masked by a long plain one.",
    backend="wordfreq",
    value_range="0 to 1",
    status_rules=[
        "undefined when the prompt has no scored word tokens",
        "unavailable when wordfreq is not installed",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=0.077,
    needs=["rare_word_count"],
)
def rare_word_ratio(doc, ctx):
    words, scores = _scored_words(doc)
    if words is None:
        return unavailable("wordfreq")
    count = ctx.number("rare_word_count")
    value, step, failure = ratio(
        count,
        len(words),
        "rare_word_ratio",
        zero_reason="the prompt has no word tokens to score",
    )
    return failure or ok(value, step)


@register(
    "jargon_ratio",
    group="rarity",
    dtype="float",
    summary="Share of content words that look like specialist terminology.",
    formula=(
        f"a content word counts as jargon if its Zipf frequency is below {JARGON_ZIPF}, "
        f"or it is at least {JARGON_MIN_LEN} characters long, or it is an all-caps acronym; "
        "ratio = jargon words / content words"
    ),
    why="Jargon-heavy queries need the corpus to use the same terminology; paraphrase-based indexes miss them.",
    backend="wordfreq",
    value_range="0 to 1",
    status_rules=[
        "undefined when the prompt has no content words",
        "unavailable when wordfreq is not installed",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=0.286,
    tier=1,
    rank=14,
    rank_reason="Combines three independent jargon signals into one interpretable share.",
)
def jargon_ratio(doc, ctx):
    scores = doc.zipf_scores
    if scores is None:
        return unavailable("wordfreq")
    content = doc.content_words
    if not content:
        return undefined("the prompt has no content words")
    jargon = []
    for word in content:
        key = word.lower.strip("'-")
        zipf = scores.get(key)
        reasons = []
        if zipf is not None and zipf < JARGON_ZIPF:
            reasons.append(f"Zipf {zipf} < {JARGON_ZIPF}")
        if len(word.text) >= JARGON_MIN_LEN:
            reasons.append(f"{len(word.text)} characters")
        if _ACRONYM_RE.match(word.text):
            reasons.append("acronym")
        if reasons:
            jargon.append((word, "; ".join(reasons)))
    value = r3(len(jargon) / len(content))
    return ok(
        value,
        *[f"{word.text!r}: {reason}" for word, reason in jargon[:10]],
        f"jargon_ratio = {len(jargon)} / {len(content)} = {value}",
        spans=[span(w.start, w.end, "jargon") for w, _ in jargon],
    )


@register(
    "oov_like_count",
    group="rarity",
    dtype="int",
    summary="How many tokens look like identifiers rather than English words.",
    formula=(
        "count of tokens that mix letters and digits, contain an underscore, use "
        "camelCase, join words with a dot (app.py), or are unknown to wordfreq (Zipf 0)"
    ),
    why="Identifier-shaped tokens are matched exactly or not at all, so they either nail the retrieval or break it.",
    backend="wordfreq",
    value_range=">= 0",
    example="Fix the get_user_by_id function in app.py",
    expected=2,
)
def oov_like_count(doc, ctx):
    scores = doc.zipf_scores
    hits: list[tuple] = []
    for word in doc.words:
        text = word.text
        reasons = []
        if "_" in text:
            reasons.append("underscore")
        if _MIXED_RE.match(text) and any(c.isdigit() for c in text) and any(
            c.isalpha() for c in text
        ):
            reasons.append("letters and digits mixed")
        if _CAMEL_RE.search(text):
            reasons.append("camelCase")
        if _DOTTED_RE.match(text):
            reasons.append("dotted identifier")
        if scores is not None and text.isalpha() and len(text) > 2:
            if scores.get(word.lower.strip("'-"), 1.0) == 0.0:
                reasons.append("unknown to wordfreq")
        if reasons:
            hits.append((word, "; ".join(reasons)))
    return ok(
        len(hits),
        *[f"{w.text!r}: {reason}" for w, reason in hits[:10]],
        f"count = {len(hits)}",
        spans=[span(w.start, w.end, "identifier-like") for w, _ in hits],
    )


@register(
    "max_bpe_fertility",
    group="rarity",
    dtype="int",
    summary="The largest number of BPE tokens a single word costs.",
    formula='max(len(cl100k_base.encode(" " + word))) over word tokens',
    why="A word that shatters into many tokens is one the tokenizer has never seen whole, which is a strong hint the corpus has not either.",
    backend="tiktoken",
    value_range=">= 1",
    status_rules=[
        "not applicable when the prompt has no word tokens",
        "unavailable when tiktoken is not installed",
    ],
    example="What is the dose of acetylsalicylic acid?",
    expected=6,
)
def max_bpe_fertility(doc, ctx):
    counts = doc.bpe_per_word
    if counts is None:
        return unavailable("tiktoken")
    if not counts:
        return not_applicable("the prompt has no word tokens")
    word, count = max(counts.items(), key=lambda kv: kv[1])
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    return ok(
        count,
        f"most expensive word = {word!r} at {count} tokens",
        "per-word token cost: " + ", ".join(f"{w}={c}" for w, c in ranked),
        detail={"per_word": [{"word": w, "tokens": c} for w, c in ranked]},
    )
