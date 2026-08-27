"""Reasoning shape: negation, comparison, aggregation and multi-hop depth."""

from __future__ import annotations

from . import lexicons as lex
from .doc import _BOOLEAN_OP_RE
from .registry import ok, register, span, unavailable
from .util import quote_list


def _lexicon_flag(lexicon, doc, extra_steps=()):
    matches = lexicon.find(doc.text)
    steps = [f"{lexicon.name} matched: {quote_list([m['text'] for m in matches])}"]
    steps.extend(extra_steps)
    return matches, steps


@register(
    "contains_negation",
    group="reasoning",
    dtype="bool",
    summary="Whether the prompt uses a negation word.",
    formula=(
        "word-boundary match against the negation lexicon (not, never, no, none, "
        "without, cannot, don't, isn't, hardly, lack of, avoid ...) after folding "
        "curly apostrophes to ASCII"
    ),
    why="Search matches the positive topic and misses the exception, so a negated query often retrieves exactly the wrong chunks.",
    value_range="True / False",
    status_rules=[],
    example="Drugs that are not NSAIDs",
    expected=True,
    tier=1,
    rank=26,
    rank_reason="Negation inverts relevance, and embeddings famously ignore it.",
)
def contains_negation(doc, ctx):
    matches = lex.NEGATION.find(doc.text)
    steps = [f"negation lexicon matched: {quote_list([m['text'] for m in matches])}"]
    if "\u2019" in doc.raw:
        steps.append(
            "note: a curly apostrophe was folded to ASCII first, so forms like "
            "don\u2019t are detected"
        )
    return ok(
        bool(matches),
        *steps,
        spans=[span(m["start"], m["end"], "negation") for m in matches],
        hits=[lex.NEGATION.name] if matches else [],
    )


@register(
    "negation_count",
    group="reasoning",
    dtype="int",
    summary="How many negation words the prompt contains.",
    formula="number of matches against the negation lexicon",
    why="Multiple negations compound the mismatch and often indicate a filtered, exception-heavy request.",
    value_range=">= 0",
    example="Drugs that are not NSAIDs and do not cause drowsiness",
    expected=2,
)
def negation_count(doc, ctx):
    matches = lex.NEGATION.find(doc.text)
    return ok(
        len(matches),
        f"negations: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "negation") for m in matches],
    )


@register(
    "has_exclusion",
    group="reasoning",
    dtype="bool",
    summary="Whether the prompt carves out an exception.",
    formula="match against the exclusion lexicon (except, excluding, other than, besides, apart from, instead of, all but ...)",
    why="An exclusion needs the retriever to find the set and then subtract from it, which a single similarity search cannot express.",
    value_range="True / False",
    example="All EU countries except France",
    expected=True,
    tier=1,
    rank=27,
    rank_reason="Set subtraction is impossible to express in a similarity query, so these fail systematically.",
)
def has_exclusion(doc, ctx):
    matches = lex.EXCLUSION.find(doc.text)
    return ok(
        bool(matches),
        f"exclusion lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "exclusion") for m in matches],
        hits=[lex.EXCLUSION.name] if matches else [],
    )


@register(
    "is_comparison",
    group="reasoning",
    dtype="bool",
    summary="Whether the prompt compares two or more things.",
    formula="match against the comparison lexicon (compare, versus, vs, difference between, better than, pros and cons ...)",
    why="A comparison needs evidence about every side; retrieving only the more popular one looks like a wrong answer.",
    value_range="True / False",
    example="Compare ibuprofen and aspirin for fever in children.",
    expected=True,
    tier=1,
    rank=20,
    rank_reason="Guarantees at least two retrieval targets from a single query.",
)
def is_comparison(doc, ctx):
    matches = lex.COMPARISON.find(doc.text)
    return ok(
        bool(matches),
        f"comparison lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "comparison") for m in matches],
        hits=[lex.COMPARISON.name] if matches else [],
    )


@register(
    "comparison_target_count",
    group="reasoning",
    dtype="int",
    summary="How many things are being compared.",
    formula=(
        "when a comparison is detected, count the coordinated noun phrases joined by "
        "`and`, `or`, `vs` or a comma using the dependency parse (conj links), plus the head itself"
    ),
    why="Each target is a separate document need, so three-way comparisons fail far more often than two-way ones.",
    backend="spacy",
    value_range=">= 0",
    status_rules=[
        "not applicable when the prompt is not a comparison",
        "unavailable without the spaCy model",
    ],
    example="Compare ibuprofen and aspirin for fever in children.",
    expected=2,
    needs=["is_comparison"],
)
def comparison_target_count(doc, ctx):
    from .registry import not_applicable

    if not ctx.flag("is_comparison"):
        return not_applicable("the prompt is not phrased as a comparison")
    sdoc = doc.spacy_doc
    if sdoc is None:
        return unavailable("spacy")
    groups: dict[int, list] = {}
    for token in sdoc:
        if token.dep_ == "conj" and token.pos_ in ("NOUN", "PROPN", "NUM"):
            groups.setdefault(token.head.i, []).append(token)
    if not groups:
        return ok(1, "no coordinated noun phrases found, so only one target is named")
    head_index, members = max(groups.items(), key=lambda kv: len(kv[1]))
    head = sdoc[head_index]
    targets = [head, *members]
    return ok(
        len(targets),
        f"coordinated targets: {quote_list([t.text for t in targets])}",
        spans=[span(t.idx, t.idx + len(t.text), "comparison target") for t in targets],
    )


@register(
    "causal_flag",
    group="reasoning",
    dtype="bool",
    summary="Whether the prompt asks about causes or effects.",
    formula="match against the causal lexicon (because, cause, due to, leads to, effect of, impact of, why ...)",
    why="Causal explanations are usually spread across several passages rather than stated in one chunk.",
    value_range="True / False",
    example="Why did the 2008 crisis happen?",
    expected=True,
)
def causal_flag(doc, ctx):
    matches = lex.CAUSAL.find(doc.text)
    return ok(
        bool(matches),
        f"causal lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "causal") for m in matches],
        hits=[lex.CAUSAL.name] if matches else [],
    )


@register(
    "aggregation_flag",
    group="reasoning",
    dtype="bool",
    summary="Whether the prompt asks for a count, total, ranking or extreme.",
    formula="match against the aggregation lexicon (how many, total, average, number of, most, top, largest, all of ...)",
    why="Aggregates need every relevant record, so a top-k retrieval that misses one produces a confidently wrong number.",
    value_range="True / False",
    example="How many countries joined the EU after 2004?",
    expected=True,
)
def aggregation_flag(doc, ctx):
    matches = lex.AGGREGATION.find(doc.text)
    return ok(
        bool(matches),
        f"aggregation lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "aggregation") for m in matches],
        hits=[lex.AGGREGATION.name] if matches else [],
    )


@register(
    "conditional_flag",
    group="reasoning",
    dtype="bool",
    summary="Whether the prompt is conditional or hypothetical.",
    formula="match against the conditional lexicon (if, unless, provided that, assuming, in case, what if ...)",
    why="Hypotheticals rarely exist verbatim in a corpus; the retriever has to find the underlying rule instead.",
    value_range="True / False",
    example="What happens if the dose is doubled?",
    expected=True,
)
def conditional_flag(doc, ctx):
    matches = lex.CONDITIONAL.find(doc.text)
    return ok(
        bool(matches),
        f"conditional lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "conditional") for m in matches],
        hits=[lex.CONDITIONAL.name] if matches else [],
    )


@register(
    "requires_synthesis_flag",
    group="reasoning",
    dtype="bool",
    summary="Whether answering means combining several sources rather than quoting one.",
    formula="match against the synthesis lexicon (summarize, synthesize, overview, timeline, history of, trends, across ...)",
    why="Synthesis needs broad coverage; a single highly-similar chunk is not enough even when it is retrieved.",
    value_range="True / False",
    example="Give me a timeline of the history of the transistor",
    expected=True,
)
def requires_synthesis_flag(doc, ctx):
    matches = lex.SYNTHESIS.find(doc.text)
    return ok(
        bool(matches),
        f"synthesis lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "synthesis") for m in matches],
        hits=[lex.SYNTHESIS.name] if matches else [],
    )


@register(
    "boolean_operator_count",
    group="reasoning",
    dtype="int",
    summary="How many explicit boolean operators (uppercase AND / OR / NOT) the prompt uses.",
    formula=r"count of matches of (?<!\w)(AND|OR|NOT)(?!\w) in uppercase only",
    why="Uppercase operators mean the user is writing a query language the retriever probably does not honour.",
    value_range=">= 0",
    example="python AND pandas NOT numpy",
    expected=2,
)
def boolean_operator_count(doc, ctx):
    matches = list(_BOOLEAN_OP_RE.finditer(doc.text))
    return ok(
        len(matches),
        f"operators: {quote_list([m.group(0) for m in matches])}",
        spans=[span(m.start(), m.end(), "boolean operator") for m in matches],
    )


@register(
    "nested_modifier_depth",
    group="reasoning",
    dtype="int",
    summary="How deeply noun phrases are nested inside one another.",
    formula=(
        "longest chain of prepositional or relative-clause modifiers in the dependency "
        "parse, following prep / pobj / relcl / acl / poss links, as in "
        "`the director of the film that won the award`"
    ),
    why="Each nesting level is another entity you must resolve before the real target is even identified, which is the definition of a multi-hop query.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model"],
    example="Who directed the film that won the Oscar for best picture in 1994?",
    expected=2,
)
def nested_modifier_depth(doc, ctx):
    sdoc = doc.spacy_doc
    if sdoc is None:
        return unavailable("spacy")
    link_deps = {"prep", "pobj", "relcl", "acl", "poss", "appos", "nmod"}
    best = 0
    best_chain: list[str] = []
    for token in sdoc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue
        depth = 0
        chain = [token.text]
        cursor = token
        guard = 0
        while guard < 32:
            guard += 1
            nxt = None
            for child in cursor.children:
                if child.dep_ in link_deps:
                    nxt = child
                    break
            if nxt is None:
                break
            if nxt.dep_ in ("relcl", "acl", "prep", "poss"):
                depth += 1
            chain.append(nxt.text)
            cursor = nxt
        if depth > best:
            best = depth
            best_chain = chain
    return ok(
        best,
        f"deepest modifier chain: {' -> '.join(best_chain)}" if best_chain else "no noun phrases",
        f"depth = {best}",
    )


@register(
    "hop_estimate",
    group="reasoning",
    dtype="int",
    summary="Estimated number of retrieval hops needed to answer.",
    formula=(
        "1 + nested_modifier_depth + 1 if the prompt is a comparison + 1 if it needs "
        "synthesis + 1 if it aggregates, capped at 5"
    ),
    why="Each hop multiplies the chance of failure, because every hop has to retrieve correctly for the answer to be right.",
    backend="spacy",
    value_range="1 to 5",
    example="Who directed the film that won the Oscar for best picture in 1994?",
    expected=3,
    needs=["nested_modifier_depth", "is_comparison", "requires_synthesis_flag", "aggregation_flag"],
    tier=1,
    rank=21,
    rank_reason="Directly estimates how many successful retrievals the answer depends on.",
)
def hop_estimate(doc, ctx):
    terms = {
        "base": 1.0,
        "nested_modifier_depth": ctx.number("nested_modifier_depth"),
        "is_comparison": 1.0 if ctx.flag("is_comparison") else 0.0,
        "requires_synthesis_flag": 1.0 if ctx.flag("requires_synthesis_flag") else 0.0,
        "aggregation_flag": 1.0 if ctx.flag("aggregation_flag") else 0.0,
    }
    total = min(5, int(sum(terms.values())))
    shown = " + ".join(f"{k}={int(v)}" for k, v in terms.items() if v)
    return ok(total, f"hop_estimate = {shown} = {total} (capped at 5)", detail={"terms": terms})


@register(
    "constraint_count",
    group="reasoning",
    dtype="int",
    summary="How many restrictive conditions the prompt places on a valid answer.",
    formula=(
        "sum of: temporal constraint, negation present, exclusion present, conditional "
        "present, an explicit item count, a length limit, an output-format request, "
        "numerals present, units present"
    ),
    why="Every constraint narrows the set of acceptable chunks, and a retriever ranking by similarity ignores all of them.",
    value_range=">= 0",
    example="List three EU countries except France that joined after 2004 in JSON",
    expected=5,
    needs=[
        "has_temporal_constraint", "contains_negation", "has_exclusion",
        "conditional_flag", "requested_item_count", "has_length_limit",
        "has_output_format_request", "numeral_count", "unit_count",
    ],
    tier=1,
    rank=22,
    rank_reason="Counts the filters a similarity search silently ignores.",
)
def constraint_count(doc, ctx):
    terms = {
        "has_temporal_constraint": 1 if ctx.flag("has_temporal_constraint") else 0,
        "contains_negation": 1 if ctx.flag("contains_negation") else 0,
        "has_exclusion": 1 if ctx.flag("has_exclusion") else 0,
        "conditional_flag": 1 if ctx.flag("conditional_flag") else 0,
        "requested_item_count": 1 if ctx.value("requested_item_count") else 0,
        "has_length_limit": 1 if ctx.flag("has_length_limit") else 0,
        "has_output_format_request": 1 if ctx.flag("has_output_format_request") else 0,
        "numerals": 1 if ctx.number("numeral_count") > 0 else 0,
        "units": 1 if ctx.number("unit_count") > 0 else 0,
    }
    total = sum(terms.values())
    active = [name for name, value in terms.items() if value]
    return ok(
        total,
        f"active constraints: {quote_list(active)}",
        f"constraint_count = {total}",
        detail={"terms": terms},
    )
