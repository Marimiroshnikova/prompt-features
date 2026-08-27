"""Domain, language and register features."""

from __future__ import annotations

from . import lexicons as lex
from . import nlp
from .doc import _EMOJI_RE
from .registry import not_applicable, ok, register, span, unavailable, undefined, unreliable
from .util import quote_list, r2, r3, ratio

LANG_MIN_CHARS = 20

CATEGORY_PRIORITY = (
    "Coding",
    "Translation",
    "Math",
    "Creative",
    "Comparison",
    "Summarization",
    "Fact Retrieval",
    "Definition",
    "Reasoning",
)


def _category_scores(doc, ctx) -> dict[str, dict]:
    """Weighted evidence per category, scored on the core question."""
    text = doc.core_text or doc.text
    qtype = ctx.value("question_type", "other")
    entities = ctx.number("named_entity_hint")

    def hits(lexicon):
        return lexicon.hits(text)

    scores: dict[str, dict] = {}

    def add(category, weight, label, items):
        entry = scores.setdefault(category, {"score": 0.0, "evidence": []})
        if not items:
            return
        count = len(items) if isinstance(items, list) else 1
        entry["score"] += weight * count
        entry["evidence"].append(f"{label}: {quote_list(items) if isinstance(items, list) else items} (+{r2(weight * count)})")

    add("Coding", 1.0, "coding lexicon", hits(lex.CODING))
    if ctx.flag("has_code_span"):
        add("Coding", 2.0, "code span present", "yes")
    if ctx.flag("has_file_path"):
        add("Coding", 1.0, "file path present", "yes")

    add("Creative", 1.5, "creative lexicon", hits(lex.CREATIVE))
    add("Translation", 2.0, "translation lexicon", hits(lex.TRANSLATION))
    add("Math", 1.2, "math lexicon", hits(lex.MATH))
    if ctx.flag("has_math_expression"):
        add("Math", 2.0, "math expression present", "yes")

    add("Summarization", 1.5, "summarization lexicon", hits(lex.SUMMARIZATION))
    add("Comparison", 1.5, "comparison lexicon", hits(lex.COMPARISON))
    add("Definition", 1.0, "definition lexicon", hits(lex.DEFINITION))

    add("Fact Retrieval", 1.5, "fact-retrieval lexicon", hits(lex.FACT_RETRIEVAL))
    if entities:
        add("Fact Retrieval", 0.5, "named entities", f"{int(entities)} found")
    if qtype in ("who", "when", "where", "which", "whose"):
        add("Fact Retrieval", 1.0, "question type", qtype)

    add("Reasoning", 1.0, "causal lexicon", hits(lex.CAUSAL))
    if ctx.flag("conditional_flag"):
        add("Reasoning", 0.5, "conditional phrasing", "yes")
    if qtype == "why":
        add("Reasoning", 1.0, "question type", qtype)
    return scores


def _winning_category(doc, ctx) -> tuple[str, float, dict]:
    scores = _category_scores(doc, ctx)
    best_score = max((entry["score"] for entry in scores.values()), default=0.0)
    if best_score <= 0:
        return "Reasoning", 0.0, scores
    for category in CATEGORY_PRIORITY:
        entry = scores.get(category)
        if entry and entry["score"] >= best_score - 1e-9:
            return category, entry["score"], scores
    return "Reasoning", 0.0, scores


@register(
    "question_category",
    group="domain",
    dtype="label",
    summary="Coarse type of the ask, chosen by weighted evidence rather than first-match-wins.",
    formula=(
        "score every category on the core question: coding lexicon x1.0 (+2.0 for a code "
        "span, +1.0 for a file path), translation x2.0, math x1.2 (+2.0 for an expression), "
        "creative x1.5, comparison x1.5, summarization x1.5, definition x1.0, "
        "fact-retrieval x1.5 (+0.5 for named entities, +1.0 for a who/when/where/which "
        "question), reasoning from the causal lexicon x1.0 (+0.5 conditional, +1.0 for a "
        "why question); the highest score wins, ties break by the priority order "
        "Coding, Translation, Math, Creative, Comparison, Summarization, Fact Retrieval, "
        'Definition, Reasoning; with no evidence at all the value is "Reasoning"'
    ),
    why="Coding, fact and creative asks need completely different document types, and a routing mismatch looks identical to retrieval failure.",
    value_range=" / ".join(CATEGORY_PRIORITY),
    status_rules=['not applicable when no category evidence is found (value falls back to "Reasoning")'],
    example="Who wrote The Hobbit?",
    expected="Fact Retrieval",
    needs=[
        "question_type", "named_entity_hint", "has_code_span", "has_file_path",
        "has_math_expression", "conditional_flag",
    ],
    tier=1,
    rank=30,
    rank_reason="Base rates of retrieval failure differ enormously by ask type, so this is the natural segmentation variable.",
)
def question_category(doc, ctx):
    category, score, scores = _winning_category(doc, ctx)
    if score <= 0:
        return not_applicable(
            "no category lexicon matched, so the category defaults to Reasoning",
            value="Reasoning",
        )
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])
    steps = [f"{name} = {r2(entry['score'])}" for name, entry in ranked if entry["score"] > 0]
    winner = scores[category]
    return ok(
        category,
        *winner["evidence"],
        "scores: " + ", ".join(steps),
        f"winner = {category} at {r2(score)}",
        detail={"scores": {name: r2(entry["score"]) for name, entry in ranked}},
    )


@register(
    "question_category_score",
    group="domain",
    dtype="float",
    summary="How much evidence supported the winning category.",
    formula="the winning score from question_category",
    why="A category picked on one weak keyword is much less trustworthy than one picked on five, and this exposes that difference.",
    value_range=">= 0",
    status_rules=["0.0 when no category evidence was found"],
    example="Who wrote The Hobbit?",
    expected=3.0,
    needs=["question_category"],
)
def question_category_score(doc, ctx):
    _, score, _ = _winning_category(doc, ctx)
    return ok(r2(score), f"winning score = {r2(score)}")


@register(
    "domain_hint",
    group="domain",
    dtype="label",
    summary="Subject area of the prompt, from keyword lexicons.",
    formula=(
        "count matches for each domain lexicon (medical, legal, finance, tech, science, "
        "history, business, everyday) and take the highest"
    ),
    why="Specialised domains need a matching corpus; a general index answers general questions and misses domain ones.",
    value_range="medical / legal / finance / tech / science / history / business / everyday, or not applicable",
    status_rules=["not applicable when no domain lexicon matches"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected="medical",
)
def domain_hint(doc, ctx):
    tallies = []
    for lexicon in lex.DOMAIN_LEXICONS:
        found = lexicon.hits(doc.text)
        if found:
            tallies.append((len(found), lexicon.name.replace("domain_", ""), found, lexicon))
    if not tallies:
        return not_applicable("no domain lexicon matched this prompt")
    tallies.sort(key=lambda t: -t[0])
    count, name, found, lexicon = tallies[0]
    steps = [f"{n} = {c} match(es)" for c, n, _, _ in tallies]
    return ok(
        name,
        f"{name} terms: {quote_list(found)}",
        "domain scores: " + ", ".join(steps),
        spans=[span(m["start"], m["end"], name) for m in lexicon.find(doc.text)],
        hits=[lexicon.name],
    )


def _language(doc):
    if len(doc.stripped) == 0:
        return None
    return nlp.detect_language(doc.stripped)


@register(
    "language_code",
    group="domain",
    dtype="label",
    summary="Detected language of the prompt.",
    formula="langdetect.detect_langs(prompt)[0].lang, with a fixed seed so the result is deterministic",
    why="A query in one language against an index in another is a guaranteed miss, whatever the embedding model claims.",
    backend="langdetect",
    value_range="ISO 639-1 code",
    status_rules=[
        "undefined for an empty prompt",
        f"unreliable below {LANG_MIN_CHARS} characters, where langdetect is close to guessing",
        "unavailable when langdetect is not installed",
    ],
    example="Who wrote The Hobbit?",
    expected="en",
)
def language_code(doc, ctx):
    if doc.is_empty:
        return undefined("the prompt is empty, so there is no language to detect")
    guess = _language(doc)
    if guess is None:
        return unavailable("langdetect")
    code, prob = guess
    step = f"langdetect returned {code!r} with probability {prob}"
    if len(doc.stripped) < LANG_MIN_CHARS:
        return unreliable(
            code,
            f"only {len(doc.stripped)} characters; langdetect needs about {LANG_MIN_CHARS} to be dependable",
            step,
        )
    return ok(code, step)


@register(
    "language_confidence",
    group="domain",
    dtype="float",
    summary="How confident the language detection is.",
    formula="langdetect.detect_langs(prompt)[0].prob",
    why="Low confidence often means mixed-language or code-like text, which retrieves badly in any single-language index.",
    backend="langdetect",
    value_range="0 to 1",
    status_rules=[
        "undefined for an empty prompt",
        f"unreliable below {LANG_MIN_CHARS} characters",
        "unavailable when langdetect is not installed",
    ],
    example="Who wrote The Hobbit?",
    expected=1.0,
)
def language_confidence(doc, ctx):
    if doc.is_empty:
        return undefined("the prompt is empty, so there is no language to detect")
    guess = _language(doc)
    if guess is None:
        return unavailable("langdetect")
    code, prob = guess
    value = r2(prob)
    step = f"probability for {code!r} = {value}"
    if len(doc.stripped) < LANG_MIN_CHARS:
        return unreliable(
            value,
            f"only {len(doc.stripped)} characters; the probability is not meaningful yet",
            step,
        )
    return ok(value, step)


@register(
    "is_english",
    group="domain",
    dtype="bool",
    summary="Whether the prompt is in English.",
    formula='language_code == "en"',
    why="Most indexes are monolingual, so a non-English query is one of the few features that predicts failure almost on its own.",
    backend="langdetect",
    value_range="True / False",
    status_rules=["inherits the status of language_code"],
    example="Who wrote The Hobbit?",
    expected=True,
    needs=["language_code"],
    tier=1,
    rank=11,
    rank_reason="A language mismatch with the index defeats every other signal in the query.",
)
def is_english(doc, ctx):
    result = ctx.result("language_code")
    if result is None or result.value is None:
        return undefined("language detection produced no result")
    value = result.value == "en"
    out = ok(value, f"language_code = {result.value!r} -> {value}")
    if result.status != "ok":
        return unreliable(value, f"language_code was {result.status}: {result.reason}", *out.steps)
    return out


@register(
    "non_ascii_ratio",
    group="domain",
    dtype="float",
    summary="Share of characters outside plain ASCII.",
    formula="count(ord(char) > 127) / len(prompt)",
    why="Non-ASCII text often tokenises into many more tokens and may be absent from an English-only index.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Who wrote The Hobbit?",
    expected=0.0,
)
def non_ascii_ratio(doc, ctx):
    hits = [i for i, ch in enumerate(doc.raw) if ord(ch) > 127]
    value, step, failure = ratio(
        len(hits), len(doc.raw), "non_ascii_ratio", zero_reason="the prompt is empty"
    )
    if failure:
        return failure
    return ok(value, step, spans=[span(i, i + 1, "non-ascii") for i in hits[:50]])


@register(
    "emoji_count",
    group="domain",
    dtype="int",
    summary="How many emoji the prompt contains.",
    formula="count of characters in the emoji and symbol Unicode blocks",
    why="Emoji carry no retrievable content but still consume tokens and shift the embedding.",
    value_range=">= 0",
    example="Who wrote The Hobbit? \U0001F4DA",
    expected=1,
)
def emoji_count(doc, ctx):
    matches = list(_EMOJI_RE.finditer(doc.raw))
    return ok(
        len(matches),
        f"emoji found = {len(matches)}",
        spans=[span(m.start(), m.end(), "emoji") for m in matches],
    )


@register(
    "politeness_filler_count",
    group="domain",
    dtype="int",
    summary="How many courtesy phrases the prompt contains.",
    formula="number of matches against the politeness lexicon (please, thanks, could you, kindly, hello ...)",
    why="Filler dilutes the query embedding with words that appear in every document equally.",
    value_range=">= 0",
    example="Hi, could you please tell me who wrote The Hobbit? Thanks!",
    expected=4,
)
def politeness_filler_count(doc, ctx):
    matches = lex.POLITENESS.find(doc.text)
    return ok(
        len(matches),
        f"politeness phrases: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "politeness") for m in matches],
        hits=[lex.POLITENESS.name] if matches else [],
    )
