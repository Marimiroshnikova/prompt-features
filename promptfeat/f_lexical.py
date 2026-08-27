"""Readability and lexical-diversity features."""

from __future__ import annotations

import math
from collections import Counter

from . import nlp
from .registry import ok, register, span, unavailable, undefined, unreliable
from .util import quote_list, r2, r3, ratio

READABILITY_MIN_WORDS = 12
MTLD_MIN_TOKENS = 50
MTLD_TTR_FLOOR = 0.72

_UNRELIABLE_NOTE = (
    "readability formulas are calibrated on running prose; under "
    f"{READABILITY_MIN_WORDS} words they swing wildly and can go negative"
)


def _readability(doc, fn_name: str, label: str):
    module = nlp.textstat_module()
    if module is None:
        return unavailable("textstat")
    if doc.is_empty:
        return undefined("the prompt is empty, so there is no text to score")
    fn = getattr(module, fn_name, None)
    if fn is None:  # pragma: no cover - depends on textstat version
        return unavailable("textstat", f"textstat has no {fn_name}()")
    try:
        raw = float(fn(doc.text))
    except Exception as exc:  # pragma: no cover - defensive
        return undefined(f"{label} could not be computed ({exc})")
    value = r2(raw)
    n_words = len(doc.whitespace_words)
    step = f"textstat.{fn_name}(prompt) = {value}"
    count_step = f"prompt has {n_words} words (threshold for a reliable score is {READABILITY_MIN_WORDS})"
    if n_words < READABILITY_MIN_WORDS:
        return unreliable(value, _UNRELIABLE_NOTE, step, count_step)
    return ok(value, step, count_step)


@register(
    "question_complexity_score",
    group="readability",
    dtype="float",
    summary="Estimated US school grade level needed to read the prompt.",
    formula="round(textstat.flesch_kincaid_grade(prompt), 2)",
    why="Dense, high-grade wording usually hides a specific technical need that generic chunks miss.",
    backend="textstat",
    value_range="about -3 to 30",
    status_rules=[
        f"unreliable when the prompt has fewer than {READABILITY_MIN_WORDS} words",
        "undefined for an empty prompt",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=5.68,
)
def question_complexity_score(doc, ctx):
    return _readability(doc, "flesch_kincaid_grade", "Flesch-Kincaid grade")


@register(
    "is_readability_reliable",
    group="readability",
    dtype="bool",
    summary="Whether the prompt is long enough for readability scores to mean anything.",
    formula=f"question_length_words >= {READABILITY_MIN_WORDS}",
    why="Lets a model gate the readability columns instead of learning from noise; Flesch-Kincaid returns -3.4 for a one-word prompt.",
    value_range="True / False",
    example="Who wrote The Hobbit?",
    expected=False,
    needs=["question_length_words"],
)
def is_readability_reliable(doc, ctx):
    n_words = len(doc.whitespace_words)
    value = n_words >= READABILITY_MIN_WORDS
    return ok(value, f"{n_words} words >= {READABILITY_MIN_WORDS} is {value}")


@register(
    "flesch_reading_ease",
    group="readability",
    dtype="float",
    summary="Reading-ease score; higher is easier.",
    formula="round(textstat.flesch_reading_ease(prompt), 2)",
    why="A second readability view that reacts to sentence length rather than grade level.",
    backend="textstat",
    value_range="about -50 to 120",
    status_rules=[
        f"unreliable when the prompt has fewer than {READABILITY_MIN_WORDS} words",
        "undefined for an empty prompt",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=66.79,
)
def flesch_reading_ease(doc, ctx):
    return _readability(doc, "flesch_reading_ease", "Flesch reading ease")


@register(
    "gunning_fog",
    group="readability",
    dtype="float",
    summary="Gunning fog index: years of education implied by the wording.",
    formula="round(textstat.gunning_fog(prompt), 2)",
    why="Reacts strongly to polysyllabic jargon, which is exactly what a general index lacks.",
    backend="textstat",
    value_range="about 0 to 30",
    status_rules=[
        f"unreliable when the prompt has fewer than {READABILITY_MIN_WORDS} words",
        "undefined for an empty prompt",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=5.66,
)
def gunning_fog(doc, ctx):
    return _readability(doc, "gunning_fog", "Gunning fog")


@register(
    "coleman_liau_index",
    group="readability",
    dtype="float",
    summary="Readability from letter and sentence counts rather than syllables.",
    formula="round(textstat.coleman_liau_index(prompt), 2)",
    why="Syllable-free, so it stays usable on acronyms and product names that break syllable counting.",
    backend="textstat",
    value_range="about -5 to 25",
    status_rules=[
        f"unreliable when the prompt has fewer than {READABILITY_MIN_WORDS} words",
        "undefined for an empty prompt",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=7.73,
)
def coleman_liau_index(doc, ctx):
    return _readability(doc, "coleman_liau_index", "Coleman-Liau index")


@register(
    "dale_chall_score",
    group="readability",
    dtype="float",
    summary="Dale-Chall score, based on a list of words familiar to fourth-graders.",
    formula="round(textstat.dale_chall_readability_score(prompt), 2)",
    why="Directly measures how much of the prompt is outside everyday vocabulary.",
    backend="textstat",
    value_range="about 0 to 12",
    status_rules=[
        f"unreliable when the prompt has fewer than {READABILITY_MIN_WORDS} words",
        "undefined for an empty prompt",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=8.5,
)
def dale_chall_score(doc, ctx):
    return _readability(doc, "dale_chall_readability_score", "Dale-Chall score")


@register(
    "difficult_word_count",
    group="readability",
    dtype="int",
    summary="How many words are outside the Dale-Chall list of familiar words.",
    formula="textstat.difficult_words(prompt)",
    why="Each unfamiliar word is a term the corpus may phrase differently.",
    backend="textstat",
    value_range=">= 0",
    status_rules=["unavailable when textstat is not installed"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=2,
)
def difficult_word_count(doc, ctx):
    module = nlp.textstat_module()
    if module is None:
        return unavailable("textstat")
    if doc.is_empty:
        return ok(0, "empty prompt has no words")
    try:
        count = int(module.difficult_words(doc.text))
    except Exception as exc:  # pragma: no cover - defensive
        return undefined(f"difficult_words() failed ({exc})")
    try:
        words = sorted(module.difficult_words_list(doc.text))
    except Exception:
        words = []
    steps = [f"Dale-Chall unfamiliar words = {count}"]
    if words:
        steps.append(f"words: {quote_list(words)}")
    return ok(count, *steps)


@register(
    "difficult_word_ratio",
    group="readability",
    dtype="float",
    summary="Share of words that are outside the Dale-Chall familiar list.",
    formula="difficult_word_count / question_length_words",
    why="Normalises unfamiliar vocabulary by length, so a short technical query is not hidden by a long simple one.",
    backend="textstat",
    value_range="0 to 1",
    status_rules=[
        "undefined when the prompt has no words",
        "unavailable when textstat is not installed",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=0.143,
    needs=["difficult_word_count"],
)
def difficult_word_ratio(doc, ctx):
    result = ctx.result("difficult_word_count")
    if result is None or result.value is None:
        return unavailable("textstat")
    value, step, failure = ratio(
        result.value,
        len(doc.whitespace_words),
        "difficult_word_ratio",
        zero_reason="the prompt has no words",
    )
    return failure or ok(value, step)


@register(
    "avg_syllables_per_word",
    group="readability",
    dtype="float",
    summary="Average syllable count per word.",
    formula="sum(textstat.syllable_count(word)) / number of word tokens",
    why="Multi-syllable vocabulary is a reliable marker of specialist language.",
    backend="textstat",
    value_range=">= 1",
    status_rules=[
        "undefined when the prompt has no word tokens",
        "unavailable when textstat is not installed",
    ],
    example="Who wrote The Hobbit?",
    expected=1.25,
)
def avg_syllables_per_word(doc, ctx):
    counts = doc.syllable_counts
    if counts is None:
        return unavailable("textstat")
    words = doc.alpha_words
    total = sum(counts.get(w.lower, 1) for w in words)
    value, step, failure = ratio(
        total,
        len(words),
        "avg_syllables_per_word",
        zero_reason="the prompt has no word tokens",
        digits=2,
    )
    return failure or ok(value, step)


@register(
    "polysyllable_count",
    group="readability",
    dtype="int",
    summary="How many words have three or more syllables.",
    formula="count(textstat.syllable_count(word) >= 3)",
    why="Polysyllabic terms are usually the domain-specific ones a general index handles worst.",
    backend="textstat",
    value_range=">= 0",
    status_rules=["unavailable when textstat is not installed"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=1,
)
def polysyllable_count(doc, ctx):
    counts = doc.syllable_counts
    if counts is None:
        return unavailable("textstat")
    hits = [w for w in doc.alpha_words if counts.get(w.lower, 1) >= 3]
    return ok(
        len(hits),
        f"words with 3+ syllables: {quote_list([w.text for w in hits])}",
        spans=[span(w.start, w.end, "polysyllabic") for w in hits],
    )


@register(
    "type_token_ratio",
    group="readability",
    dtype="float",
    summary="Distinct words divided by total words.",
    formula="len(set(lowercased words)) / len(words)",
    why="Low repetition means every term carries weight; high repetition dilutes the query embedding.",
    value_range="0 to 1",
    status_rules=[
        "undefined when the prompt has no word tokens",
        "note: this measure is length-biased, which is why root_type_token_ratio and mtld are also computed",
    ],
    example="Who wrote The Hobbit?",
    expected=1.0,
)
def type_token_ratio(doc, ctx):
    words = [w.lower for w in doc.alpha_words]
    value, step, failure = ratio(
        len(set(words)),
        len(words),
        "type_token_ratio",
        zero_reason="the prompt has no word tokens",
    )
    return failure or ok(value, step)


@register(
    "root_type_token_ratio",
    group="readability",
    dtype="float",
    summary="Vocabulary richness corrected for length (Guiraud's index).",
    formula="len(set(words)) / sqrt(len(words))",
    why="Plain type-token ratio always falls as text grows, so comparing a 5-word prompt to a 500-word one needs this correction.",
    value_range=">= 0",
    status_rules=["undefined when the prompt has no word tokens"],
    example="Who wrote The Hobbit?",
    expected=2.0,
)
def root_type_token_ratio(doc, ctx):
    words = [w.lower for w in doc.alpha_words]
    if not words:
        return undefined("the prompt has no word tokens")
    types = len(set(words))
    value = r3(types / math.sqrt(len(words)))
    return ok(
        value,
        f"types = {types}, tokens = {len(words)}",
        f"root_type_token_ratio = {types} / sqrt({len(words)}) = {value}",
    )


@register(
    "mtld",
    group="readability",
    dtype="float",
    summary="Measure of Textual Lexical Diversity: mean run length before vocabulary starts repeating.",
    formula=(
        "walk the tokens keeping a running type-token ratio; each time it drops to "
        f"{MTLD_TTR_FLOOR} close a factor and reset; MTLD = tokens / factors, averaged "
        "over a forward and a backward pass"
    ),
    why="Length-independent diversity: a prompt that keeps introducing new terms needs broader retrieval coverage than one circling the same words.",
    value_range=">= 0",
    status_rules=[
        f"unreliable below {MTLD_MIN_TOKENS} tokens, which is the published floor for MTLD",
        "undefined when the prompt has no word tokens",
    ],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=13.0,
)
def mtld(doc, ctx):
    words = [w.lower for w in doc.alpha_words]
    if not words:
        return undefined("the prompt has no word tokens")
    forward = _mtld_pass(words)
    backward = _mtld_pass(list(reversed(words)))
    value = r2((forward + backward) / 2)
    steps = [
        f"forward pass = {r2(forward)}, backward pass = {r2(backward)}",
        f"mtld = ({r2(forward)} + {r2(backward)}) / 2 = {value}",
    ]
    if len(words) < MTLD_MIN_TOKENS:
        return unreliable(
            value,
            f"MTLD needs about {MTLD_MIN_TOKENS} tokens to stabilise; this prompt has {len(words)}",
            *steps,
        )
    return ok(value, *steps)


def _mtld_pass(words: list[str]) -> float:
    factors = 0.0
    types: set[str] = set()
    count = 0
    for word in words:
        count += 1
        types.add(word)
        ttr = len(types) / count
        if ttr <= MTLD_TTR_FLOOR:
            factors += 1
            types.clear()
            count = 0
    if count:
        ttr = len(types) / count
        remainder = (1 - ttr) / (1 - MTLD_TTR_FLOOR)
        factors += max(0.0, min(1.0, remainder))
    if factors <= 0:
        return float(len(words))
    return len(words) / factors


@register(
    "word_entropy",
    group="readability",
    dtype="float",
    summary="Shannon entropy of the word distribution, in bits.",
    formula="-sum(p * log2(p)) over the frequencies of lowercased words",
    why="Measures how much distinct information the prompt actually carries, independent of raw length.",
    value_range="0 to about 10",
    status_rules=["undefined when the prompt has no word tokens"],
    example="Who wrote The Hobbit?",
    expected=2.0,
)
def word_entropy(doc, ctx):
    words = [w.lower for w in doc.alpha_words]
    if not words:
        return undefined("the prompt has no word tokens")
    value, detail = _entropy(words)
    return ok(
        value,
        f"{len(set(words))} distinct words over {len(words)} tokens",
        f"word_entropy = {value} bits",
        detail=detail,
    )


@register(
    "char_entropy",
    group="readability",
    dtype="float",
    summary="Shannon entropy of the character distribution, in bits.",
    formula="-sum(p * log2(p)) over character frequencies of the prompt",
    why="Very low entropy flags padded or repeated text; very high entropy flags identifiers, hashes and code.",
    value_range="0 to about 7",
    status_rules=["undefined for an empty prompt"],
    example="Who wrote The Hobbit?",
    expected=3.56,
)
def char_entropy(doc, ctx):
    chars = list(doc.text)
    if not chars:
        return undefined("the prompt is empty")
    value, detail = _entropy(chars)
    return ok(value, f"char_entropy = {value} bits over {len(chars)} characters", detail=detail)


def _entropy(items: list[str]) -> tuple[float, dict]:
    counts = Counter(items)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    top = counts.most_common(5)
    return r2(entropy), {"most_common": [{"item": k, "count": v} for k, v in top]}


@register(
    "stopword_ratio",
    group="readability",
    dtype="float",
    summary="Share of words that are function words (the, of, is ...).",
    formula="count(token.is_stop) / number of word tokens, using spaCy's stop list (built-in list as fallback)",
    why="A query made mostly of stopwords has little for a retriever to match on.",
    backend="spacy",
    value_range="0 to 1",
    status_rules=["undefined when the prompt has no word tokens"],
    example="What about it?",
    expected=1.0,
)
def stopword_ratio(doc, ctx):
    words = doc.alpha_words
    stops = [w for w in words if w.is_stop]
    value, step, failure = ratio(
        len(stops),
        len(words),
        "stopword_ratio",
        zero_reason="the prompt has no word tokens",
    )
    if failure:
        return failure
    return ok(
        value,
        f"stopwords: {quote_list([w.text for w in stops])}",
        step,
        spans=[span(w.start, w.end, "stopword") for w in stops],
    )


@register(
    "content_word_count",
    group="readability",
    dtype="int",
    summary="How many words carry retrievable content.",
    formula=(
        "count of word tokens that are longer than one character, are not stopwords, "
        "and are not part of a politeness phrase; `Hi`, `please` and `Thanks` are "
        "excluded because padding a prompt with courtesy must not make it look more "
        "specific than it is"
    ),
    why="This is the real payload of the query; everything else is glue.",
    backend="spacy",
    value_range=">= 0",
    example="Who wrote The Hobbit?",
    expected=2,
)
def content_word_count(doc, ctx):
    words = doc.content_words
    return ok(
        len(words),
        f"content words: {quote_list([w.text for w in words])}",
        spans=[span(w.start, w.end, "content word") for w in words],
    )
