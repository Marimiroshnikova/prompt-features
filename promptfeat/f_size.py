"""Size and shape features."""

from __future__ import annotations

import string

from .registry import ok, register, span, unavailable, undefined, unreliable
from .util import quote_list, r2, r3, ratio

PUNCTUATION = set(string.punctuation)


@register(
    "question_length_chars",
    group="size",
    dtype="int",
    summary="Length of the prompt in characters, including spaces and punctuation.",
    formula="len(prompt)",
    why="Very long prompts mix several constraints, and a retriever may match only part of them.",
    value_range=">= 0",
    example="Who wrote The Hobbit?",
    expected=21,
)
def question_length_chars(doc, ctx):
    return ok(len(doc.raw), f"len(prompt) = {len(doc.raw)}")


@register(
    "question_length_words",
    group="size",
    dtype="int",
    summary="Length of the prompt in whitespace-separated words.",
    formula="len(prompt.split())",
    why="Longer or packed queries usually need more than one chunk to answer.",
    value_range=">= 0",
    example="Who wrote The Hobbit?",
    expected=4,
    tier=1,
    rank=7,
    rank_reason="Length is the cheapest proxy for how many distinct needs are packed into one query.",
)
def question_length_words(doc, ctx):
    words = doc.whitespace_words
    return ok(
        len(words),
        f"prompt.split() = {quote_list([w for w in words])}",
        f"count = {len(words)}",
    )


@register(
    "context_token_count",
    group="size",
    dtype="int",
    summary="Length of the prompt in GPT-style BPE tokens.",
    formula='len(tiktoken.get_encoding("cl100k_base").encode(prompt))',
    why="Token length is a better size signal than words when the prompt holds code, numbers or rare terms.",
    backend="tiktoken",
    value_range=">= 0",
    status_rules=["unavailable when tiktoken is not installed"],
    example="Who wrote The Hobbit?",
    expected=6,
    tier=1,
    rank=8,
    rank_reason="The retriever embeds tokens, not words; token count is the true size of the query.",
)
def context_token_count(doc, ctx):
    tokens = doc.bpe_tokens
    if tokens is None:
        return unavailable("tiktoken")
    return ok(len(tokens), f"cl100k_base encoded the prompt into {len(tokens)} tokens")


@register(
    "n_sentences",
    group="size",
    dtype="int",
    summary="How many sentences the prompt contains.",
    formula="len(spaCy sentence boundaries), falling back to a regex split on . ! ? and newlines",
    why="Several sentences usually means several information needs in one query.",
    backend="spacy",
    value_range=">= 0",
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=2,
)
def n_sentences(doc, ctx):
    sents = doc.sentences
    steps = [f"{i + 1}. [{s.kind}] {s.text!r}" for i, s in enumerate(sents[:6])]
    return ok(
        len(sents),
        *steps,
        spans=[span(s.start, s.end, s.kind) for s in sents],
    )


@register(
    "n_lines",
    group="size",
    dtype="int",
    summary="How many non-empty lines the prompt has.",
    formula="count of lines in prompt.split(chr(10)) that contain non-whitespace",
    why="Multi-line prompts usually carry pasted context or instruction blocks around the real question.",
    value_range=">= 0",
    example="Instructions:\nUse the docs only.\nWho wrote The Hobbit?",
    expected=3,
)
def n_lines(doc, ctx):
    lines = [line for line in doc.lines if line.strip()]
    return ok(len(lines), f"non-empty lines = {len(lines)} of {len(doc.lines)} total")


@register(
    "n_clauses",
    group="size",
    dtype="int",
    summary="How many clauses the prompt contains, counted from the dependency parse.",
    formula="count of tokens whose POS is VERB or AUX and whose dependency label is not an auxiliary attachment",
    why="Each clause can hold its own information need, so clause count tracks how much a single retrieval has to cover.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model, which provides the parse"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=2,
)
def n_clauses(doc, ctx):
    sdoc = doc.spacy_doc
    if sdoc is None:
        return unavailable("spacy")
    skip = {"aux", "auxpass", "aux:pass", "neg"}
    heads = [
        token
        for token in sdoc
        if token.pos_ in ("VERB", "AUX") and token.dep_ not in skip
    ]
    return ok(
        len(heads),
        f"clause heads = {quote_list([t.text for t in heads])}",
        spans=[span(t.idx, t.idx + len(t.text), t.dep_) for t in heads],
    )


@register(
    "avg_words_per_sentence",
    group="size",
    dtype="float",
    summary="Average sentence length in words.",
    formula="question_length_words / n_sentences",
    why="Long sentences pack several constraints into one embedding, which blurs the match.",
    value_range=">= 0",
    status_rules=["undefined when the prompt has no sentences (empty prompt)"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=7.0,
)
def avg_words_per_sentence(doc, ctx):
    n_sents = len(doc.sentences)
    value, step, failure = ratio(
        len(doc.whitespace_words),
        n_sents,
        "avg_words_per_sentence",
        zero_reason="the prompt contains no sentences, so there is nothing to average",
        digits=2,
    )
    return failure or ok(value, step)


@register(
    "avg_chars_per_word",
    group="size",
    dtype="float",
    summary="Average word length in characters.",
    formula="sum(len(word) for word in words) / len(words), over word tokens",
    why="Longer words are usually more technical, and technical wording is where lexical retrieval misses.",
    value_range=">= 0",
    status_rules=["undefined when the prompt has no word tokens"],
    example="Who wrote The Hobbit?",
    expected=4.25,
)
def avg_chars_per_word(doc, ctx):
    words = doc.alpha_words
    total = sum(len(w.text) for w in words)
    value, step, failure = ratio(
        total,
        len(words),
        "avg_chars_per_word",
        zero_reason="the prompt contains no word tokens",
        digits=2,
    )
    return failure or ok(value, step)


@register(
    "max_word_len",
    group="size",
    dtype="int",
    summary="Length of the longest word.",
    formula="max(len(word) for word in words)",
    why="A single very long token is often a compound technical term that the index may not contain.",
    value_range=">= 0",
    status_rules=["not applicable when the prompt has no word tokens"],
    example="Who wrote The Hobbit?",
    expected=6,
)
def max_word_len(doc, ctx):
    words = doc.alpha_words
    if not words:
        from .registry import not_applicable

        return not_applicable("the prompt contains no word tokens")
    longest = max(words, key=lambda w: len(w.text))
    return ok(
        len(longest.text),
        f"longest word = {longest.text!r} ({len(longest.text)} characters)",
        spans=[span(longest.start, longest.end, "longest word")],
    )


@register(
    "long_word_ratio",
    group="size",
    dtype="float",
    summary="Share of words that are seven characters or longer.",
    formula="count(len(word) >= 7) / len(words)",
    why="A high share of long words marks a specialised request that generic chunks rarely satisfy.",
    value_range="0 to 1",
    status_rules=["undefined when the prompt has no word tokens"],
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=0.308,
)
def long_word_ratio(doc, ctx):
    words = doc.alpha_words
    long_words = [w for w in words if len(w.text) >= 7]
    value, step, failure = ratio(
        len(long_words),
        len(words),
        "long_word_ratio",
        zero_reason="the prompt contains no word tokens",
    )
    if failure:
        return failure
    return ok(
        value,
        f"words with >= 7 characters: {quote_list([w.text for w in long_words])}",
        step,
        spans=[span(w.start, w.end, "long word") for w in long_words],
    )


@register(
    "tokens_per_word",
    group="size",
    dtype="float",
    summary="BPE tokens per word, i.e. how badly the tokenizer has to split this vocabulary.",
    formula="context_token_count / question_length_words",
    why="Common words cost one token; rare or technical words are split into many. A high value means the query uses vocabulary a corpus is unlikely to share.",
    backend="tiktoken",
    value_range="~1 to 6",
    status_rules=[
        "undefined when the prompt has no words",
        "unavailable when tiktoken is not installed",
    ],
    example="Who wrote The Hobbit?",
    expected=1.5,
    tier=1,
    rank=13,
    rank_reason="A free, model-free rarity signal: 'acetylsalicylic' costs 6 tokens, 'the cat' costs 2.",
)
def tokens_per_word(doc, ctx):
    tokens = doc.bpe_tokens
    if tokens is None:
        return unavailable("tiktoken")
    value, step, failure = ratio(
        len(tokens),
        len(doc.whitespace_words),
        "tokens_per_word",
        zero_reason="the prompt contains no words",
        digits=2,
    )
    return failure or ok(value, step)


@register(
    "punctuation_density",
    group="size",
    dtype="float",
    summary="Share of characters that are punctuation.",
    formula="count(char in string.punctuation) / len(prompt)",
    why="Dense punctuation signals code, lists or templating rather than a natural-language question.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Who wrote The Hobbit?",
    expected=0.048,
)
def punctuation_density(doc, ctx):
    marks = [i for i, ch in enumerate(doc.text) if ch in PUNCTUATION]
    value, step, failure = ratio(
        len(marks),
        len(doc.text),
        "punctuation_density",
        zero_reason="the prompt is empty",
    )
    return failure or ok(value, step)


@register(
    "whitespace_ratio",
    group="size",
    dtype="float",
    summary="Share of characters that are whitespace.",
    formula="count(char.isspace()) / len(prompt)",
    why="Unusually high whitespace means pasted or formatted blocks rather than a compact query.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Who wrote The Hobbit?",
    expected=0.143,
)
def whitespace_ratio(doc, ctx):
    spaces = sum(1 for ch in doc.text if ch.isspace())
    value, step, failure = ratio(
        spaces, len(doc.text), "whitespace_ratio", zero_reason="the prompt is empty"
    )
    return failure or ok(value, step)


@register(
    "uppercase_char_ratio",
    group="size",
    dtype="float",
    summary="Share of letters that are uppercase.",
    formula="count(char.isupper()) / count(char.isalpha())",
    why="Shouting or all-caps text tokenises differently from the cased text in most indexes.",
    value_range="0 to 1",
    status_rules=["undefined when the prompt contains no letters"],
    example="Who wrote The Hobbit?",
    expected=0.176,
)
def uppercase_char_ratio(doc, ctx):
    letters = [ch for ch in doc.text if ch.isalpha()]
    uppers = [ch for ch in letters if ch.isupper()]
    value, step, failure = ratio(
        len(uppers),
        len(letters),
        "uppercase_char_ratio",
        zero_reason="the prompt contains no letters",
    )
    return failure or ok(value, step)


@register(
    "digit_char_ratio",
    group="size",
    dtype="float",
    summary="Share of characters that are digits.",
    formula="count(char.isdigit()) / len(prompt)",
    why="Digit-heavy prompts hinge on exact values, and a chunk that never states the number cannot answer them.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Fever after 2020 in children under 12",
    expected=0.162,
)
def digit_char_ratio(doc, ctx):
    digits = sum(1 for ch in doc.text if ch.isdigit())
    value, step, failure = ratio(
        digits, len(doc.text), "digit_char_ratio", zero_reason="the prompt is empty"
    )
    return failure or ok(value, step)
