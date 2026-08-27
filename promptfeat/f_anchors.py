"""Retrieval anchors: the concrete strings a retriever can latch onto."""

from __future__ import annotations

import re

from . import lexicons as lex
from .doc import (
    _ACRONYM_RE,
    _CODE_SPAN_RE,
    _CURRENCY_AMOUNT_RE,
    _EMAIL_RE,
    _ID_LIKE_RE,
    _MATH_RE,
    _NUMERAL_RE,
    _PATH_RE,
    _PERCENT_RE,
    _URL_RE,
)
from .registry import not_applicable, ok, register, span, unavailable, undefined
from .util import quote_list, r3, ratio

NAME_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "WORK_OF_ART",
    "EVENT", "LAW", "LANGUAGE",
}
PERSON_LABELS = {"PERSON"}
ORG_LABELS = {"ORG", "NORP"}
PLACE_LABELS = {"GPE", "LOC", "FAC"}
WORK_LABELS = {"WORK_OF_ART", "PRODUCT", "EVENT", "LAW"}

_TITLE_RUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|the|de|van|von)\s+|\s+)?(?:[A-Z][a-z]+)*")
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?:]\s+|\n\s*)")


def _name_entities(doc) -> list[dict] | None:
    ents = doc.entities
    if ents is None:
        return None
    return [e for e in ents if e["label"] in NAME_LABELS]


def _title_case_fallback(doc) -> list[dict]:
    """Title-Case runs, skipping sentence-initial words and stopwords.

    This is the corrected version of the original heuristic: it discounts the
    first word of *every* sentence rather than only the first word of the
    prompt, collapses `The Hobbit` into one span, and ignores question words.
    """
    starts = {m.end() for m in _SENTENCE_START_RE.finditer(doc.text)}
    spans: list[dict] = []
    current: list = []

    def flush():
        if current:
            first, last = current[0], current[-1]
            spans.append(
                {"text": doc.text[first.start : last.end], "start": first.start, "end": last.end, "label": "TITLE_CASE"}
            )
            current.clear()

    for word in doc.words:
        text = word.text
        titled = bool(re.match(r"^[A-Z][a-z]+$", text)) or bool(_ACRONYM_RE.match(text))
        if not titled:
            flush()
            continue
        if word.start in starts and word.lower in lex.STOPWORDS:
            flush()
            continue
        if word.start in starts and not current:
            # A capitalised sentence opener is usually just orthography.
            continue
        current.append(word)
    flush()
    return spans


@register(
    "named_entity_hint",
    group="anchors",
    dtype="int",
    summary="How many named things (people, organisations, places, works, products) the prompt mentions.",
    formula=(
        "number of spaCy entity spans whose label is a name type "
        "(PERSON, ORG, GPE, LOC, FAC, NORP, PRODUCT, WORK_OF_ART, EVENT, LAW, LANGUAGE); "
        "without the model, Title-Case runs are counted instead, skipping the first "
        "word of every sentence and any stopword"
    ),
    why="Names are the strongest retrieval anchors, and each extra name is another chance to retrieve the wrong entity.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["falls back to a Title-Case heuristic when the spaCy model is missing"],
    example="Who wrote The Hobbit?",
    expected=1,
)
def named_entity_hint(doc, ctx):
    ents = _name_entities(doc)
    if ents is None:
        fallback = _title_case_fallback(doc)
        return ok(
            len(fallback),
            "spaCy model unavailable, using the Title-Case fallback",
            f"spans: {quote_list([s['text'] for s in fallback])}",
            spans=[span(s["start"], s["end"], "title case") for s in fallback],
        )
    listed = ", ".join(f"{e['text']!r} ({e['label']})" for e in ents)
    return ok(
        len(ents),
        f"name entities: {listed or 'none'}",
        f"count = {len(ents)}",
        spans=[span(e["start"], e["end"], e["label"]) for e in ents],
    )


@register(
    "entity_count",
    group="anchors",
    dtype="int",
    summary="Total number of entity spans spaCy finds, including dates and quantities.",
    formula="len(spacy_doc.ents)",
    why="Entities are what a retriever matches on; a prompt with none has nothing concrete to search for.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=1,
    tier=1,
    rank=15,
    rank_reason="The primary anchor count, from a real NER model rather than capitalisation.",
)
def entity_count(doc, ctx):
    ents = doc.entities
    if ents is None:
        return unavailable("spacy")
    return ok(
        len(ents),
        "entities: " + (", ".join(f"{e['text']!r} ({e['label']})" for e in ents) or "none"),
        spans=[span(e["start"], e["end"], e["label"]) for e in ents],
    )


@register(
    "distinct_entity_types",
    group="anchors",
    dtype="int",
    summary="How many different entity types appear.",
    formula="len(set(ent.label_ for ent in spacy_doc.ents))",
    why="A query mixing a person, a place and a date needs a chunk where all three co-occur, which is much rarer than any one of them.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="Did Tolkien publish The Hobbit in 1937 in Britain?",
    expected=3,
    tier=1,
    rank=16,
    rank_reason="Type diversity is a cheap proxy for how specific the co-occurrence requirement is.",
)
def distinct_entity_types(doc, ctx):
    ents = doc.entities
    if ents is None:
        return unavailable("spacy")
    labels = sorted({e["label"] for e in ents})
    return ok(len(labels), f"types: {quote_list(labels)}")


def _label_counter(label_set, name):
    def fn(doc, ctx):
        ents = doc.entities
        if ents is None:
            return unavailable("spacy")
        hits = [e for e in ents if e["label"] in label_set]
        return ok(
            len(hits),
            f"{name} entities: " + (", ".join(f"{e['text']!r}" for e in hits) or "none"),
            spans=[span(e["start"], e["end"], e["label"]) for e in hits],
        )

    return fn


register(
    "entity_person_count",
    group="anchors",
    dtype="int",
    summary="How many people are named.",
    formula="count of spaCy entities labelled PERSON",
    why="Person names are high-precision anchors, but similar names are a classic source of wrong-entity retrieval.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="What did Einstein say to Niels Bohr?",
    expected=2,
)(_label_counter(PERSON_LABELS, "person"))

register(
    "entity_org_count",
    group="anchors",
    dtype="int",
    summary="How many organisations or groups are named.",
    formula="count of spaCy entities labelled ORG or NORP",
    why="Organisation names are often abbreviated differently in the corpus than in the prompt.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="What did NASA and the European Space Agency announce?",
    expected=2,
)(_label_counter(ORG_LABELS, "organisation"))

register(
    "entity_place_count",
    group="anchors",
    dtype="int",
    summary="How many places are named.",
    formula="count of spaCy entities labelled GPE, LOC or FAC",
    why="Place names disambiguate otherwise generic questions, and missing them retrieves the right topic in the wrong country.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="What is the capital of France?",
    expected=1,
)(_label_counter(PLACE_LABELS, "place"))

register(
    "entity_work_count",
    group="anchors",
    dtype="int",
    summary="How many works, products, events or laws are named.",
    formula="count of spaCy entities labelled WORK_OF_ART, PRODUCT, EVENT or LAW",
    why="Titles are usually stated verbatim in exactly one place in a corpus, so getting them right is decisive.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="Who wrote Pride and Prejudice?",
    expected=1,
)(_label_counter(WORK_LABELS, "work or product"))


@register(
    "proper_noun_token_count",
    group="anchors",
    dtype="int",
    summary="How many tokens are proper nouns.",
    formula="count of tokens whose spaCy POS tag is PROPN",
    why="Counts name tokens even when NER fails to group them into an entity, so it degrades more gracefully than entity_count.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="Who wrote The Hobbit?",
    expected=1,
)
def proper_noun_token_count(doc, ctx):
    if not doc.has_spacy:
        return unavailable("spacy")
    hits = [w for w in doc.words if w.pos == "PROPN"]
    return ok(
        len(hits),
        f"proper nouns: {quote_list([w.text for w in hits])}",
        spans=[span(w.start, w.end, "PROPN") for w in hits],
    )


@register(
    "acronym_count",
    group="anchors",
    dtype="int",
    summary="How many all-caps acronyms appear.",
    formula=r"count of matches of \b[A-Z]{2,}(?:s|'s)?\b or dotted forms like U.S.",
    why="Acronyms are exact-match anchors: the corpus either spells them the same way or the retrieval misses entirely.",
    value_range=">= 0",
    example="Drugs that are not NSAIDs",
    expected=1,
)
def acronym_count(doc, ctx):
    matches = list(_ACRONYM_RE.finditer(doc.text))
    return ok(
        len(matches),
        f"acronyms: {quote_list([m.group(0) for m in matches])}",
        spans=[span(m.start(), m.end(), "acronym") for m in matches],
    )


@register(
    "quoted_span_count",
    group="anchors",
    dtype="int",
    summary="How many quoted or backticked phrases the prompt contains.",
    formula="count of \"...\", '...' and `...` spans",
    why="A quoted phrase is an explicit exact-match request, and it fails hard when the corpus words it differently.",
    value_range=">= 0",
    example='Find the paper titled "Attention Is All You Need"',
    expected=1,
    tier=1,
    rank=17,
    rank_reason="The user is naming the exact string to match; nothing else in the query carries that weight.",
)
def quoted_span_count(doc, ctx):
    spans = doc.quoted_spans
    return ok(
        len(spans),
        f"quoted phrases: {quote_list([s['text'] for s in spans])}",
        spans=[span(s["start"], s["end"], "quoted") for s in spans],
    )


@register(
    "quoted_span_word_count",
    group="anchors",
    dtype="int",
    summary="Total number of words inside quoted phrases.",
    formula="sum(len(quoted_text.split()) for each quoted span)",
    why="A long quoted phrase must appear near-verbatim in a chunk, which is much stricter than a single quoted word.",
    value_range=">= 0",
    example='Find the paper titled "Attention Is All You Need"',
    expected=5,
)
def quoted_span_word_count(doc, ctx):
    spans = doc.quoted_spans
    total = sum(len(s["text"].split()) for s in spans)
    return ok(total, f"words inside quotes = {total}")


def _flag(pattern, label, name):
    def fn(doc, ctx):
        matches = list(pattern.finditer(doc.text))
        return ok(
            bool(matches),
            f"{label}: {quote_list([m.group(0) for m in matches])}",
            spans=[span(m.start(), m.end(), name) for m in matches],
        )

    return fn


register(
    "has_url",
    group="anchors",
    dtype="bool",
    summary="Whether the prompt contains a URL.",
    formula=r"regex \b(?:https?://|www\.)\S+",
    why="A URL names the source directly, which either short-circuits retrieval or points somewhere the index does not cover.",
    value_range="True / False",
    example="Summarize https://example.com/report.pdf",
    expected=True,
)(_flag(_URL_RE, "urls", "url"))

register(
    "has_email",
    group="anchors",
    dtype="bool",
    summary="Whether the prompt contains an email address.",
    formula=r"regex \b[\w.+-]+@[\w-]+\.\w{2,}\b",
    why="Email addresses are exact-match tokens that usually appear in private data rather than an indexed corpus.",
    value_range="True / False",
    example="Who owns support@example.com?",
    expected=True,
)(_flag(_EMAIL_RE, "emails", "email"))

register(
    "has_file_path",
    group="anchors",
    dtype="bool",
    summary="Whether the prompt names a file or path.",
    formula="regex for Windows paths, POSIX paths, and bare filenames with a known code or document extension",
    why="File references point at local context that a document index does not contain at all.",
    value_range="True / False",
    example="Fix the bug in src/utils/helpers.py",
    expected=True,
)(_flag(_PATH_RE, "paths", "path"))

register(
    "has_code_span",
    group="anchors",
    dtype="bool",
    summary="Whether the prompt contains code.",
    formula="regex for backticked spans, call syntax like `foo(...)`, and leading keywords (def, class, import, SELECT ...)",
    why="Code queries need code chunks; prose retrieval over code, or the reverse, looks exactly like retrieval failure.",
    value_range="True / False",
    example="Why does my_function(x) raise a TypeError?",
    expected=True,
)(_flag(_CODE_SPAN_RE, "code spans", "code"))

register(
    "has_math_expression",
    group="anchors",
    dtype="bool",
    summary="Whether the prompt contains a mathematical expression.",
    formula="regex for arithmetic operators between numbers, comparisons, function calls like sqrt(, and $...$ segments",
    why="Formulas rarely appear verbatim in prose chunks, so the retrieval has to succeed on the surrounding words alone.",
    value_range="True / False",
    example="Is 2 + 2 = 5 ever true?",
    expected=True,
)(_flag(_MATH_RE, "math expressions", "math"))


@register(
    "id_like_token_count",
    group="anchors",
    dtype="int",
    summary="How many tokens look like identifiers, versions, tickets or hashes.",
    formula="regex for v1.2.3, ABC-123, DOI prefixes, long hex strings and letter+digit codes",
    why="Identifiers are matched exactly; if the corpus stores a different revision or format, retrieval returns nothing useful.",
    value_range=">= 0",
    example="Does v1.2.3 fix JIRA-4821?",
    expected=2,
)
def id_like_token_count(doc, ctx):
    matches = [
        m
        for m in _ID_LIKE_RE.finditer(doc.text)
        if any(ch.isdigit() for ch in m.group(0)) and any(ch.isalpha() for ch in m.group(0))
    ]
    return ok(
        len(matches),
        f"identifier-like tokens: {quote_list([m.group(0) for m in matches])}",
        spans=[span(m.start(), m.end(), "identifier") for m in matches],
    )


def _numeral_matches(doc) -> list:
    """Numerals, excluding digits that belong to identifiers or names.

    The original regex counted `1,000` as two numbers, missed `the 1990s`
    entirely, and counted the `19` in `COVID-19` as a fact.
    """
    blocked: list[tuple[int, int]] = []
    for pattern in (_URL_RE, _EMAIL_RE, _PATH_RE, _ID_LIKE_RE):
        for match in pattern.finditer(doc.text):
            blocked.append((match.start(), match.end()))
    kept = []
    for match in _NUMERAL_RE.finditer(doc.text):
        if any(start <= match.start() and match.end() <= end for start, end in blocked):
            continue
        before = doc.text[: match.start()]
        if before.endswith("-") and len(before) >= 2 and before[-2].isalpha():
            # `COVID-19`: a digit glued to a word is part of a name, not a fact.
            continue
        kept.append(match)
    return kept


@register(
    "numeral_count",
    group="anchors",
    dtype="int",
    summary="How many numbers the prompt states as facts.",
    formula=(
        "regex matching 12, 3.5, 1,000, 12,345.67, 1st and 1990s, then dropping any "
        "match that sits inside a URL, email, file path or identifier, and any digit "
        "glued to a word by a hyphen such as COVID-19"
    ),
    why="Exact-fact queries fail when the retrieved chunk never states that number.",
    value_range=">= 0",
    example="Fever after 2020 in children under 12",
    expected=2,
)
def numeral_count(doc, ctx):
    matches = _numeral_matches(doc)
    return ok(
        len(matches),
        f"numerals: {quote_list([m.group(0) for m in matches])}",
        spans=[span(m.start(), m.end(), "numeral") for m in matches],
    )


@register(
    "percent_count",
    group="anchors",
    dtype="int",
    summary="How many percentages the prompt states.",
    formula=r"regex \d+(\.\d+)?\s*(%|percent|per cent|pct) and `N percentage points`",
    why="A percentage is a precise claim; the right topic with the wrong figure is still a failed retrieval.",
    value_range=">= 0",
    example="Did revenue grow 12% or 15%?",
    expected=2,
)
def percent_count(doc, ctx):
    matches = list(_PERCENT_RE.finditer(doc.text))
    return ok(
        len(matches),
        f"percentages: {quote_list([m.group(0) for m in matches])}",
        spans=[span(m.start(), m.end(), "percent") for m in matches],
    )


@register(
    "currency_count",
    group="anchors",
    dtype="int",
    summary="How many monetary amounts the prompt states.",
    formula="regex for symbol-prefixed amounts ($1.2m) and amounts followed by a currency word or code (12 USD)",
    why="Money figures are usually reported in one specific document, and unit or currency mismatches break the match.",
    value_range=">= 0",
    example="Was the fine $2.5 million or 3 million EUR?",
    expected=2,
)
def currency_count(doc, ctx):
    matches = list(_CURRENCY_AMOUNT_RE.finditer(doc.text))
    return ok(
        len(matches),
        f"amounts: {quote_list([m.group(0).strip() for m in matches])}",
        spans=[span(m.start(), m.end(), "currency") for m in matches],
    )


@register(
    "unit_count",
    group="anchors",
    dtype="int",
    summary="How many measurements with units the prompt states.",
    formula="count of unit-lexicon terms that directly follow a number, within three characters",
    why="A measurement is only retrievable if the chunk uses the same unit; mg versus g is a silent miss.",
    value_range=">= 0",
    example="Is 400 mg every 6 hours safe?",
    expected=2,
)
def unit_count(doc, ctx):
    numerals = _numeral_matches(doc)
    hits = []
    for match in lex.UNITS.find(doc.text):
        gap = doc.text[:match["start"]]
        tail = gap[-3:]
        if any(n.end() >= match["start"] - 3 and n.end() <= match["start"] for n in numerals) and (
            tail.strip() == "" or tail.strip().isdigit() or tail[-1:].isdigit() or tail[-1:] == " "
        ):
            hits.append(match)
    return ok(
        len(hits),
        f"units attached to a number: {quote_list([h['text'] for h in hits])}",
        spans=[span(h["start"], h["end"], "unit") for h in hits],
        hits=[lex.UNITS.name] if hits else [],
    )


@register(
    "anchor_count",
    group="anchors",
    dtype="int",
    summary="Total number of concrete things in the prompt that a retriever can match on.",
    formula=(
        "named_entity_hint + numeral_count + quoted_span_count + acronym_count + "
        "id_like_token_count + percent_count + currency_count + "
        "(1 for each of has_url, has_email, has_file_path that is true)"
    ),
    why="This is the single best summary of whether the query has anything specific to retrieve; zero anchors is the classic failure case.",
    value_range=">= 0",
    example="Who wrote The Hobbit?",
    expected=1,
    needs=[
        "named_entity_hint", "numeral_count", "quoted_span_count", "acronym_count",
        "id_like_token_count", "percent_count", "currency_count", "has_url",
        "has_email", "has_file_path",
    ],
    tier=1,
    rank=1,
    rank_reason="Zero anchors means the retriever has no specific target at all; this is the strongest single failure signal.",
)
def anchor_count(doc, ctx):
    counts = {
        "named_entity_hint": ctx.number("named_entity_hint"),
        "numeral_count": ctx.number("numeral_count"),
        "quoted_span_count": ctx.number("quoted_span_count"),
        "acronym_count": ctx.number("acronym_count"),
        "id_like_token_count": ctx.number("id_like_token_count"),
        "percent_count": ctx.number("percent_count"),
        "currency_count": ctx.number("currency_count"),
        "has_url": 1.0 if ctx.flag("has_url") else 0.0,
        "has_email": 1.0 if ctx.flag("has_email") else 0.0,
        "has_file_path": 1.0 if ctx.flag("has_file_path") else 0.0,
    }
    total = int(sum(counts.values()))
    terms = " + ".join(f"{name}={int(value)}" for name, value in counts.items() if value)
    return ok(
        total,
        f"anchor_count = {terms or '0'} = {total}",
        detail={"terms": {k: int(v) for k, v in counts.items()}},
    )


@register(
    "anchor_density",
    group="anchors",
    dtype="float",
    summary="Anchors per content word.",
    formula="anchor_count / content_word_count",
    why="Separates a specific short query from a long vague one: two anchors in six content words is strong, two in sixty is weak.",
    value_range="0 to 1 or more",
    status_rules=["undefined when the prompt has no content words"],
    example="Who wrote The Hobbit?",
    expected=0.5,
    needs=["anchor_count", "content_word_count"],
    tier=1,
    rank=2,
    rank_reason="Normalises the anchor count so long prompts cannot fake specificity.",
)
def anchor_density(doc, ctx):
    value, step, failure = ratio(
        ctx.number("anchor_count"),
        ctx.number("content_word_count"),
        "anchor_density",
        zero_reason="the prompt has no content words, so density has no denominator",
    )
    return failure or ok(value, step)
