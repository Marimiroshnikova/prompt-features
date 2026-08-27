"""Ambiguity and underspecification: is the prompt answerable at all?"""

from __future__ import annotations

from . import lexicons as lex
from .registry import not_applicable, ok, register, span, unavailable
from .util import clamp01, quote_list, r3, scale

AMBIGUITY_MIN_CONDITIONS = 2
SHORT_CONTENT_WORDS = 5

DEMONSTRATIVES = {"this", "that", "these", "those"}
PRONOUN_DEPS = {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "attr", "dative", "oprd"}
# spaCy tags interrogative and relative wh-words as PRON; they ask for the
# antecedent rather than referring back to one, so they are never "unresolved".
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
WH_WORDS = {"who", "whom", "whose", "what", "when", "where", "why", "how", "which", "that"}


def _pronouns(doc):
    """Pronoun tokens, preferring spaCy's POS tags over a bare word list."""
    if doc.has_spacy:
        found = []
        for word in doc.words:
            if word.tag in WH_TAGS:
                continue
            if word.pos == "PRON":
                found.append(word)
            elif word.lower in DEMONSTRATIVES and word.dep in PRONOUN_DEPS:
                found.append(word)
        return found
    return [
        w
        for w in doc.words
        if w.lower in lex.PRONOUNS.terms and w.lower not in WH_WORDS
    ]


def _unresolved(doc):
    """Pronouns with no candidate antecedent earlier in the prompt."""
    nouns = [
        w
        for w in doc.words
        if (w.pos in ("NOUN", "PROPN")) or (not doc.has_spacy and not w.is_stop and w.is_alpha)
    ]
    unresolved = []
    for pron in _pronouns(doc):
        if pron.lower in ("i", "me", "my", "mine", "we", "us", "our", "you", "your"):
            continue  # speaker and listener need no antecedent
        earlier = [n for n in nouns if n.start < pron.start]
        if not earlier:
            unresolved.append((pron, "no noun appears before it"))
    return unresolved


@register(
    "unresolved_pronoun_count",
    group="ambiguity",
    dtype="int",
    summary="How many pronouns have nothing in the prompt to refer back to.",
    formula=(
        "count pronouns (spaCy POS PRON, plus this/that/these/those used as a subject or "
        "object) that have no noun or proper noun earlier in the prompt; interrogative "
        "wh-words are excluded because they request an antecedent rather than refer to "
        "one, and first and second person pronouns are excluded because they need none"
    ),
    why="A pronoun with no antecedent means the real subject was never named, so the retriever has no topic to search for.",
    backend="spacy",
    value_range=">= 0",
    example="What about it?",
    expected=1,
    tier=1,
    rank=4,
    rank_reason="A dangling pronoun means the information need is literally not in the text.",
)
def unresolved_pronoun_count(doc, ctx):
    unresolved = _unresolved(doc)
    return ok(
        len(unresolved),
        *[f"{p.text!r}: {reason}" for p, reason in unresolved] or ["every pronoun has a candidate antecedent"],
        spans=[span(p.start, p.end, "unresolved pronoun") for p, _ in unresolved],
    )


@register(
    "has_dangling_reference",
    group="ambiguity",
    dtype="bool",
    summary="Whether the prompt points at context that was never included.",
    formula="match against the dangling-reference lexicon (the above, as mentioned, this document, the attached, we discussed, the context ...)",
    why="The prompt assumes text the retriever cannot see, so the query is missing its subject entirely.",
    value_range="True / False",
    example="Summarize the attached document as mentioned above",
    expected=True,
    tier=1,
    rank=5,
    rank_reason="Names a context dependency that no retrieval over a corpus can satisfy.",
)
def has_dangling_reference(doc, ctx):
    matches = lex.DANGLING_REFERENCE.find(doc.text)
    return ok(
        bool(matches),
        f"dangling-reference lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "dangling reference") for m in matches],
        hits=[lex.DANGLING_REFERENCE.name] if matches else [],
    )


@register(
    "vague_term_count",
    group="ambiguity",
    dtype="int",
    summary="How many placeholder words the prompt uses instead of naming things.",
    formula="number of matches against the vague-terms lexicon (thing, stuff, something, whatever, etc, sort of ...)",
    why="Placeholder nouns occupy the slot where a retrievable name should be.",
    value_range=">= 0",
    example="Can you tell me more about that thing we discussed?",
    expected=1,
    tier=1,
    rank=6,
    rank_reason="Placeholders mark exactly the words a retriever cannot match on.",
)
def vague_term_count(doc, ctx):
    matches = lex.VAGUE_TERMS.find(doc.text)
    return ok(
        len(matches),
        f"vague terms: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "vague term") for m in matches],
        hits=[lex.VAGUE_TERMS.name] if matches else [],
    )


@register(
    "vague_quantifiers_count",
    group="ambiguity",
    dtype="int",
    summary="How many imprecise quantifiers the prompt uses.",
    formula="number of matches against the vague-quantifiers lexicon (some, many, several, most, a few, various ...)",
    why="`Several studies` gives no way to know how many chunks would be enough, so retrieval has no stopping point.",
    value_range=">= 0",
    example="Give me several examples of various approaches",
    expected=2,
)
def vague_quantifiers_count(doc, ctx):
    matches = lex.VAGUE_QUANTIFIERS.find(doc.text)
    return ok(
        len(matches),
        f"vague quantifiers: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "vague quantifier") for m in matches],
        hits=[lex.VAGUE_QUANTIFIERS.name] if matches else [],
    )


@register(
    "hedge_word_count",
    group="ambiguity",
    dtype="int",
    summary="How many hedging words the prompt uses.",
    formula="number of matches against the hedges lexicon (maybe, probably, roughly, approximately, I think, typically ...)",
    why="Hedged questions have no crisp answer to retrieve, so any chunk looks partially right and partially wrong.",
    value_range=">= 0",
    example="Roughly how many people probably attended?",
    expected=2,
)
def hedge_word_count(doc, ctx):
    matches = lex.HEDGES.find(doc.text)
    return ok(
        len(matches),
        f"hedges: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "hedge") for m in matches],
        hits=[lex.HEDGES.name] if matches else [],
    )


@register(
    "definite_np_without_entity_count",
    group="ambiguity",
    dtype="int",
    summary="How many `the ...` phrases name no specific entity.",
    formula=(
        "count spaCy noun chunks that start with `the` and contain no proper noun and no "
        "named entity, such as `the report` or `the file`"
    ),
    why="A definite phrase promises a specific document the retriever was never told how to identify.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model, which provides noun chunks"],
    example="Summarize the report and send the file",
    expected=2,
)
def definite_np_without_entity_count(doc, ctx):
    chunks = doc.noun_chunks
    if chunks is None:
        return unavailable("spacy")
    entities = doc.entities or []
    hits = []
    for chunk in chunks:
        text = chunk["text"]
        if not text.lower().startswith("the "):
            continue
        overlaps_entity = any(
            not (e["end"] <= chunk["start"] or e["start"] >= chunk["end"]) for e in entities
        )
        has_propn = any(
            w.pos == "PROPN" and chunk["start"] <= w.start < chunk["end"] for w in doc.words
        )
        if not overlaps_entity and not has_propn:
            hits.append(chunk)
    return ok(
        len(hits),
        f"definite phrases with no entity: {quote_list([c['text'] for c in hits])}",
        spans=[span(c["start"], c["end"], "definite phrase") for c in hits],
    )


@register(
    "missing_subject_flag",
    group="ambiguity",
    dtype="bool",
    summary="Whether the prompt is a fragment with no verb at all.",
    formula="true when no token is tagged VERB or AUX",
    why="A bare noun phrase states a topic but no information need, so the retriever cannot tell what about it matters.",
    backend="spacy",
    value_range="True / False",
    status_rules=["unavailable without the spaCy model"],
    example="best python books",
    expected=True,
)
def missing_subject_flag(doc, ctx):
    if not doc.has_spacy:
        return unavailable("spacy")
    verbs = [w for w in doc.words if w.pos in ("VERB", "AUX")]
    return ok(
        not verbs,
        f"verb or auxiliary tokens: {quote_list([w.text for w in verbs])}",
        spans=[span(w.start, w.end, w.pos) for w in verbs],
    )


@register(
    "anchor_free_flag",
    group="ambiguity",
    dtype="bool",
    summary="Whether the prompt contains no retrieval anchor at all.",
    formula="anchor_count == 0",
    why="With no name, number, quote or identifier, ranking falls back to generic topical similarity, which is where retrieval fails most often.",
    value_range="True / False",
    example="What about it?",
    expected=True,
    needs=["anchor_count"],
)
def anchor_free_flag(doc, ctx):
    count = ctx.number("anchor_count")
    return ok(count == 0, f"anchor_count = {int(count)} -> {count == 0}")


@register(
    "is_ambiguous",
    group="ambiguity",
    dtype="bool",
    summary="Whether the prompt is too underspecified to retrieve for.",
    formula=(
        "true when at least two of these four conditions hold: "
        f"(a) the core question has fewer than {SHORT_CONTENT_WORDS} content words, "
        "(b) anchor_count is 0, "
        "(c) there is an unresolved pronoun or a dangling reference, "
        "(d) vague_term_count is above 0"
    ),
    why="Too little specific text means the index has no clear target; requiring two conditions stops short but perfectly specific questions from being flagged.",
    value_range="True / False",
    status_rules=[],
    example="What about it?",
    expected=True,
    needs=["anchor_count", "unresolved_pronoun_count", "has_dangling_reference", "vague_term_count"],
    tier=1,
    rank=3,
    rank_reason="The headline underspecification flag, now a composite instead of a raw word-count threshold.",
)
def is_ambiguous(doc, ctx):
    core_words = [
        w for w in doc.content_words if any(a <= w.start and w.end <= b for a, b in doc.core_spans)
    ] or doc.content_words
    conditions = {
        f"core question has fewer than {SHORT_CONTENT_WORDS} content words": len(core_words)
        < SHORT_CONTENT_WORDS,
        "no retrieval anchors": ctx.number("anchor_count") == 0,
        "unresolved pronoun or dangling reference": bool(
            ctx.number("unresolved_pronoun_count") > 0 or ctx.flag("has_dangling_reference")
        ),
        "uses vague placeholder terms": ctx.number("vague_term_count") > 0,
    }
    met = [name for name, value in conditions.items() if value]
    value = len(met) >= AMBIGUITY_MIN_CONDITIONS
    steps = [f"content words in the core question = {len(core_words)}"]
    steps += [f"{'yes' if value_ else 'no '}: {name}" for name, value_ in conditions.items()]
    steps.append(
        f"{len(met)} of 4 conditions met, threshold is {AMBIGUITY_MIN_CONDITIONS} -> {value}"
    )
    old = len(doc.whitespace_words) < 5
    steps.append(f"(the old word-count rule would have said {old})")
    return ok(value, *steps, detail={"conditions": conditions})


@register(
    "underspecification_score",
    group="ambiguity",
    dtype="float",
    summary="How underspecified the prompt is, on a 0 to 1 scale.",
    formula=(
        "0.35 * anchor_free + 0.20 * min(1, unresolved_pronouns) + "
        "0.15 * dangling_reference + 0.15 * min(1, vague_terms / 2) + "
        "0.15 * shortness, where shortness = 1 - min(1, content_words / 8)"
    ),
    why="A graded version of is_ambiguous, so a model can learn a threshold instead of inheriting mine.",
    value_range="0 to 1",
    example="What about it?",
    expected=0.7,
    needs=[
        "anchor_free_flag", "unresolved_pronoun_count", "has_dangling_reference",
        "vague_term_count", "content_word_count",
    ],
)
def underspecification_score(doc, ctx):
    shortness = 1 - scale(ctx.number("content_word_count"), 0, 8)
    terms = {
        "anchor_free (0.35)": 0.35 * (1.0 if ctx.flag("anchor_free_flag") else 0.0),
        "unresolved pronouns (0.20)": 0.20 * clamp01(ctx.number("unresolved_pronoun_count")),
        "dangling reference (0.15)": 0.15 * (1.0 if ctx.flag("has_dangling_reference") else 0.0),
        "vague terms (0.15)": 0.15 * clamp01(ctx.number("vague_term_count") / 2),
        "shortness (0.15)": 0.15 * shortness,
    }
    value = r3(sum(terms.values()))
    return ok(
        value,
        *[f"{name} = {round(part, 3)}" for name, part in terms.items()],
        f"underspecification_score = {value}",
        detail={"terms": {k: round(v, 3) for k, v in terms.items()}},
    )
