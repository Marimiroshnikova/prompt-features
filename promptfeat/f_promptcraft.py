"""Prompt-engineering artifacts: the scaffolding wrapped around the real question."""

from __future__ import annotations

import difflib
import re

from . import lexicons as lex
from .registry import not_applicable, ok, register, span, undefined
from .util import quote_list, r3, ratio

FEW_SHOT_NUMBERED = re.compile(r"(?i)\bexample\s*\d+\s*:")
FEW_SHOT_QA = re.compile(r"(?im)^[ \t]*(?:q|question)\s*:")
FEW_SHOT_ANSWER = re.compile(r"(?im)^[ \t]*(?:a|answer)\s*:")
FEW_SHOT_IO = re.compile(r"(?im)^[ \t]*input\s*:")
FEW_SHOT_IO_OUT = re.compile(r"(?im)^[ \t]*output\s*:")
FEW_SHOT_HASH = re.compile(r"(?m)^[ \t]*#{2,4}[ \t]*\S")
SYSTEM_HEADER = re.compile(r"(?im)^[ \t]*#{0,4}[ \t]*(system|instructions?|role|task|persona)[ \t]*:")

TEMPLATE_MIN_LINES = 3
TEMPLATE_SIMILARITY = 0.8

# "in 200 words", "in two sentences", "one paragraph": a count paired with a
# length unit is a budget on the answer, which the phrase lexicon cannot express.
NUMERIC_LENGTH_LIMIT = re.compile(
    r"(?i)\b(?:\d+|" + "|".join(lex.NUMBER_WORDS) + r")[ \t]+"
    r"(?:words?|sentences?|paragraphs?|characters?|chars?|lines?|pages?|tokens?)\b"
)


def _few_shot_evidence(doc) -> tuple[int, list[str], list[dict]]:
    text = doc.text
    numbered = list(FEW_SHOT_NUMBERED.finditer(text))
    qa = list(FEW_SHOT_QA.finditer(text))
    answers = list(FEW_SHOT_ANSWER.finditer(text))
    inputs = list(FEW_SHOT_IO.finditer(text))
    outputs = list(FEW_SHOT_IO_OUT.finditer(text))
    hashes = list(FEW_SHOT_HASH.finditer(text))

    candidates = [
        (len(numbered), "numbered `Example N:` markers", numbered),
        (min(len(qa), len(answers)) if answers else 0, "`Q:` / `A:` pairs", qa + answers),
        (min(len(inputs), len(outputs)) if outputs else 0, "`Input:` / `Output:` pairs", inputs + outputs),
        (len(hashes) if len(hashes) >= 2 else 0, "repeated `###` section markers", hashes),
    ]
    count, label, matches = max(candidates, key=lambda c: c[0])
    steps = [f"{c[1]}: {c[0]}" for c in candidates if c[0]]
    spans = [span(m.start(), m.end(), "few-shot marker") for m in matches]
    return count, steps or ["no few-shot markers found"], spans


@register(
    "has_few_shot_examples",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt includes worked examples.",
    formula=(
        "true if any of these appear: `Example N:`, at least one `Q:`/`A:` pair, at "
        "least one `Input:`/`Output:` pair, or two or more `###` section markers"
    ),
    why="Examples pull the query embedding toward the example topic instead of the real question.",
    value_range="True / False",
    example="Example 1: Paris. What is the capital of France?",
    expected=True,
)
def has_few_shot_examples(doc, ctx):
    count, steps, spans = _few_shot_evidence(doc)
    return ok(count > 0, *steps, spans=spans)


@register(
    "few_shot_example_count",
    group="promptcraft",
    dtype="int",
    summary="How many worked examples the prompt includes.",
    formula="the largest count among numbered markers, Q/A pairs, Input/Output pairs and ### markers",
    why="More examples mean more off-topic text competing with the question for the retriever's attention.",
    value_range=">= 0",
    example="Example 1: Paris. Example 2: Berlin. What is the capital of France?",
    expected=2,
)
def few_shot_example_count(doc, ctx):
    count, steps, spans = _few_shot_evidence(doc)
    return ok(count, *steps, spans=spans)


@register(
    "template_repetition_score",
    group="promptcraft",
    dtype="float",
    summary="How similar the prompt's lines are to each other.",
    formula=(
        "for each non-empty line, the highest difflib.SequenceMatcher ratio against any "
        "other line; the score is the mean of those maxima"
    ),
    why="A high score means templated few-shot text that no `Example N:` regex would catch, and templates crowd out the real query.",
    value_range="0 to 1",
    status_rules=[f"not applicable with fewer than {TEMPLATE_MIN_LINES} non-empty lines"],
    example="Input: cat\nOutput: animal\nInput: rose\nOutput: plant\nInput: tuna\nOutput:",
    expected=0.729,
)
def template_repetition_score(doc, ctx):
    lines = [line.strip() for line in doc.lines if line.strip()]
    if len(lines) < TEMPLATE_MIN_LINES:
        return not_applicable(
            f"the prompt has {len(lines)} non-empty line(s); at least "
            f"{TEMPLATE_MIN_LINES} are needed to judge repetition"
        )
    maxima = []
    for i, line in enumerate(lines):
        best = 0.0
        for j, other in enumerate(lines):
            if i == j:
                continue
            best = max(best, difflib.SequenceMatcher(None, line, other).ratio())
        maxima.append(best)
    value = r3(sum(maxima) / len(maxima))
    return ok(
        value,
        f"{len(lines)} lines compared pairwise",
        "per-line best match: " + ", ".join(f"{m:.2f}" for m in maxima[:8]),
        f"mean = {value}",
    )


@register(
    "prompt_contains_system_instructions",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt carries a system or instruction block.",
    formula=(
        "true if a line starts with `system:`, `instruction:`, `instructions:`, `role:`, "
        "`task:` or `persona:` (optionally after up to four `#`), or the role-prompt "
        "lexicon matches (`you are a`, `act as`, `your task is` ...)"
    ),
    why="Instruction text is not the information need, and it can outweigh the real query in the embedding.",
    value_range="True / False",
    example="Instructions:\nAnswer using the docs.\nWho wrote The Hobbit?",
    expected=True,
)
def prompt_contains_system_instructions(doc, ctx):
    headers = list(SYSTEM_HEADER.finditer(doc.text))
    roles = lex.ROLE_PROMPT.find(doc.text)
    steps = []
    if headers:
        steps.append(f"section headers: {quote_list([m.group(0).strip() for m in headers])}")
    if roles:
        steps.append(f"role-prompt lexicon: {quote_list([m['text'] for m in roles])}")
    return ok(
        bool(headers or roles),
        *steps or ["no instruction header or role phrasing found"],
        spans=[span(m.start(), m.end(), "instruction header") for m in headers]
        + [span(m["start"], m["end"], "role prompt") for m in roles],
        hits=[lex.ROLE_PROMPT.name] if roles else [],
    )


@register(
    "instruction_line_count",
    group="promptcraft",
    dtype="int",
    summary="How many sentences are instructions rather than the question.",
    formula=(
        "count of sentences classified as instruction: they sit inside an instruction "
        "block, or they match the meta-instruction or role-prompt lexicon and contain no "
        "question mark"
    ),
    why="Counts the scaffolding directly, which is the part of the prompt a retriever should ideally never see.",
    value_range=">= 0",
    example="Instructions:\nUse the docs only.\nWho wrote The Hobbit?",
    expected=1,
)
def instruction_line_count(doc, ctx):
    sents = [s for s in doc.sentences if s.kind == "instruction"]
    return ok(
        len(sents),
        *[f"instruction: {s.text!r}" for s in sents[:6]] or ["no instruction sentences"],
        spans=[span(s.start, s.end, "instruction") for s in sents],
    )


@register(
    "instruction_char_ratio",
    group="promptcraft",
    dtype="float",
    summary="Share of the prompt taken up by instructions.",
    formula="characters in instruction sentences / len(prompt)",
    why="The higher this is, the less of the embedded text describes what to actually find.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Instructions:\nUse the docs only.\nWho wrote The Hobbit?",
    expected=0.593,
)
def instruction_char_ratio(doc, ctx):
    chars = sum(s.end - s.start for s in doc.sentences if s.kind == "instruction")
    value, step, failure = ratio(
        chars, len(doc.text), "instruction_char_ratio", zero_reason="the prompt is empty"
    )
    return failure or ok(value, step)


@register(
    "has_role_prompt",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt assigns the model a persona.",
    formula="match against the role-prompt lexicon (you are a, act as, pretend to be, your role is ...)",
    why="Persona text is pure model instruction; in the retrieval query it is noise.",
    value_range="True / False",
    example="You are an expert librarian. Who wrote The Hobbit?",
    expected=True,
)
def has_role_prompt(doc, ctx):
    matches = lex.ROLE_PROMPT.find(doc.text)
    return ok(
        bool(matches),
        f"role-prompt lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "role prompt") for m in matches],
        hits=[lex.ROLE_PROMPT.name] if matches else [],
    )


@register(
    "has_output_format_request",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt demands a particular answer format.",
    formula="match against the output-format lexicon (json, table, bullet points, one sentence, step by step ...)",
    why="Format demands add tokens unrelated to the topic and are a constraint the retriever cannot satisfy.",
    value_range="True / False",
    example="Who wrote The Hobbit? Answer in JSON.",
    expected=True,
)
def has_output_format_request(doc, ctx):
    matches = lex.FORMAT_LEXICON.find(doc.text)
    return ok(
        bool(matches),
        f"output-format lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "format request") for m in matches],
        hits=[lex.FORMAT_LEXICON.name] if matches else [],
    )


@register(
    "output_format",
    group="promptcraft",
    dtype="label",
    summary="Which answer format the prompt asks for.",
    formula="first output-format lexicon match, mapped to a normalised label (json, table, bullets, steps ...)",
    why="Structured-output demands correlate with extraction tasks, which need precise chunks rather than topical ones.",
    value_range="json / table / bullets / steps / ..., or not applicable",
    status_rules=["not applicable when the prompt requests no particular format"],
    example="Who wrote The Hobbit? Answer in JSON.",
    expected="json",
)
def output_format(doc, ctx):
    matches = lex.FORMAT_LEXICON.find(doc.text)
    if not matches:
        return not_applicable("the prompt does not ask for a particular output format")
    first = sorted(matches, key=lambda m: m["start"])[0]
    key = " ".join(first["text"].lower().split())
    label = lex.FORMAT_REQUEST.get(key, key)
    return ok(
        label,
        f"matched {first['text']!r} -> {label!r}",
        spans=[span(first["start"], first["end"], "format request")],
        hits=[lex.FORMAT_LEXICON.name],
    )


@register(
    "has_length_limit",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt limits the answer length.",
    formula=(
        "match against the length-limit lexicon (no more than, at most, briefly, "
        "concise ...) or a count paired with a length unit (`in 200 words`, `two sentences`)"
    ),
    why="A length cap pushes the answer toward one chunk, so a partially relevant retrieval is more likely to be judged wrong.",
    value_range="True / False",
    example="Who wrote The Hobbit? Answer in no more than 5 words.",
    expected=True,
)
def has_length_limit(doc, ctx):
    matches = lex.LENGTH_LIMIT.find(doc.text)
    numeric = [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in NUMERIC_LENGTH_LIMIT.finditer(doc.text)
    ]
    found = matches + [n for n in numeric if not any(m["start"] <= n["start"] < m["end"] for m in matches)]
    return ok(
        bool(found),
        f"length-limit lexicon matched: {quote_list([m['text'] for m in matches])}",
        f"numeric length budgets matched: {quote_list([n['text'] for n in numeric])}",
        spans=[span(m["start"], m["end"], "length limit") for m in found],
        hits=[lex.LENGTH_LIMIT.name] if matches else [],
    )


@register(
    "has_context_block",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt contains a pasted context block.",
    formula=(
        "true if the prompt has a fenced code block, a tagged block such as "
        "<document>...</document>, a triple-quoted block, or a `Context:` / `Background:` header"
    ),
    why="When context is pasted in, the retriever is often unnecessary or actively harmful, and the query text is dominated by the paste.",
    value_range="True / False",
    example="Context:\nTolkien published The Hobbit in 1937.\nWho wrote it?",
    expected=True,
)
def has_context_block(doc, ctx):
    blocks = [s for s in doc.segments if s.kind == "context"]
    return ok(
        bool(blocks),
        *[f"{s.source}: {s.text[:60]!r}" for s in blocks] or ["no context block found"],
        spans=[span(s.start, s.end, "context block") for s in blocks],
    )


@register(
    "context_block_char_ratio",
    group="promptcraft",
    dtype="float",
    summary="Share of the prompt that is pasted context.",
    formula="characters inside context blocks / len(prompt)",
    why="Quantifies how much of the embedded text is source material rather than a query.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Context:\nTolkien published The Hobbit in 1937.\nWho wrote it?",
    expected=0.133,
)
def context_block_char_ratio(doc, ctx):
    chars = sum(s.end - s.start for s in doc.segments if s.kind == "context")
    value, step, failure = ratio(
        chars, len(doc.text), "context_block_char_ratio", zero_reason="the prompt is empty"
    )
    return failure or ok(value, step)


@register(
    "has_citation_request",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt asks for sources or citations.",
    formula="match against the citation-request lexicon (cite, source, reference, with links, provide evidence ...)",
    why="Citations raise the bar from `topically relevant` to `verifiably supporting`, so borderline retrievals now count as failures.",
    value_range="True / False",
    example="Who wrote The Hobbit? Cite your sources.",
    expected=True,
)
def has_citation_request(doc, ctx):
    matches = lex.CITATION_REQUEST.find(doc.text)
    return ok(
        bool(matches),
        f"citation-request lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "citation request") for m in matches],
        hits=[lex.CITATION_REQUEST.name] if matches else [],
    )


@register(
    "has_chain_of_thought_cue",
    group="promptcraft",
    dtype="bool",
    summary="Whether the prompt asks the model to reason step by step.",
    formula="match against the chain-of-thought lexicon (step by step, show your work, explain your reasoning ...)",
    why="Reasoning cues usually accompany multi-hop questions, which need several correct retrievals rather than one.",
    value_range="True / False",
    example="Think step by step: why did the 2008 crisis happen?",
    expected=True,
)
def has_chain_of_thought_cue(doc, ctx):
    matches = lex.COT_CUE.find(doc.text)
    return ok(
        bool(matches),
        f"chain-of-thought lexicon matched: {quote_list([m['text'] for m in matches])}",
        spans=[span(m["start"], m["end"], "reasoning cue") for m in matches],
        hits=[lex.COT_CUE.name] if matches else [],
    )


@register(
    "core_question_ratio",
    group="promptcraft",
    dtype="float",
    summary="Share of the prompt that is the actual question, after removing instructions, examples and pasted context.",
    formula="len(core question text) / len(prompt.strip())",
    why="This is the fraction of the embedding that describes what to find. When it is low, the retriever is mostly matching on boilerplate that every prompt shares.",
    value_range="0 to 1",
    status_rules=["undefined for an empty prompt"],
    example="Instructions:\nUse the docs only.\nWho wrote The Hobbit?",
    expected=0.389,
    tier=1,
    rank=9,
    rank_reason="Measures signal dilution directly: how much of what gets embedded is actually the query.",
)
def core_question_ratio(doc, ctx):
    if doc.is_empty:
        return undefined("the prompt is empty")
    core = doc.core_text
    value, step, failure = ratio(
        len(core), len(doc.stripped), "core_question_ratio", zero_reason="the prompt is empty"
    )
    if failure:
        return failure
    removed = [s for s in doc.sentences if s.kind != "core"]
    return ok(
        value,
        f"core question = {core[:80]!r}",
        *[f"removed [{s.kind}]: {s.text[:60]!r}" for s in removed[:5]],
        step,
        spans=[span(a, b, "core question") for a, b in doc.core_spans],
    )
