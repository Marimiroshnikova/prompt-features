"""Question-structure features: what kind of ask, and how many asks."""

from __future__ import annotations

import re

from . import lexicons as lex
from .registry import not_applicable, ok, register, span, unavailable
from .util import quote_list

WH_WORDS = ("who", "whom", "whose", "what", "when", "where", "why", "how", "which")
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
_WH_RE = re.compile(r"(?i)\b(" + "|".join(WH_WORDS) + r")\b")
# Enumerable units only. Length units (words, sentences, paragraphs, pages)
# belong to max_length_words: one need can be spread over many sentences,
# whereas each bullet or recommendation needs its own supporting chunk.
_ITEM_NOUNS = (
    r"examples?|ways|reasons|items|options|tips|ideas|steps|points|things|"
    r"alternatives|benefits|differences|features|methods|approaches|names|"
    r"books|papers|tools|questions|facts|bullets|bullet points|takeaways|"
    r"recommendations|suggestions|strategies|sources|citations|references|"
    r"categories|use cases|pros|cons|criteria|factors|causes|advantages|"
    r"disadvantages|findings|insights|metrics|risks|scenarios|quotes|columns|rows"
)
_NUM_WORD_RE = "|".join(lex.NUMBER_WORDS)
_COUNT_BEFORE_RE = re.compile(
    rf"(?i)\b(?:top|best|worst|first|last|give me|list|name)\s+(\d+|{_NUM_WORD_RE})\b"
)
_COUNT_AFTER_RE = re.compile(rf"(?i)\b(\d+|{_NUM_WORD_RE})\s+(?:{_ITEM_NOUNS})\b")


def _wh_tokens(doc, only_core: bool = False):
    """Wh-words, preferring spaCy's interrogative tags over a bare regex."""
    core_spans = doc.core_spans
    sdoc = doc.spacy_doc
    found = []
    if sdoc is not None:
        for token in sdoc:
            if token.lower_ not in WH_WORDS:
                continue
            if token.tag_ not in WH_TAGS:
                continue
            found.append((token.idx, token.idx + len(token.text), token.lower_, token.tag_))
    else:
        for match in _WH_RE.finditer(doc.text):
            found.append((match.start(), match.end(), match.group(1).lower(), "regex"))
    if only_core and core_spans:
        found = [
            f for f in found if any(a <= f[0] and f[1] <= b for a, b in core_spans)
        ]
    return found


@register(
    "question_type",
    group="structure",
    dtype="label",
    summary="The leading question word of the actual question.",
    formula=(
        "leftmost wh-word (who / whom / whose / what / when / where / why / how / which) "
        "that spaCy tags as interrogative (WDT, WP, WP$, WRB) inside the core question; "
        'if there is none the value is "other"'
    ),
    why="`when` and `where` need one exact span, `why` is usually spread over several chunks, and `how` often needs a procedure the index stores elsewhere.",
    backend="spacy",
    value_range="who / whom / whose / what / when / where / why / how / which / other",
    status_rules=['not applicable when the prompt contains no interrogative wh-word (value stays "other")'],
    example="Who wrote The Hobbit?",
    expected="who",
    tier=1,
    rank=28,
    rank_reason="Different question words need structurally different evidence, so failure rates differ sharply by type.",
)
def question_type(doc, ctx):
    found = _wh_tokens(doc, only_core=True) or _wh_tokens(doc)
    if not found:
        return not_applicable(
            "no interrogative wh-word appears in the prompt, so there is no question word to report",
            'value falls back to "other"',
            value="other",
        )
    start, end, word, tag = sorted(found)[0]
    return ok(
        word,
        f"wh-words found: {quote_list([f[2] for f in sorted(found)])}",
        f"leftmost is {word!r} at character {start} (spaCy tag {tag})",
        spans=[span(start, end, "question word")],
    )


@register(
    "question_type_secondary",
    group="structure",
    dtype="label",
    summary="The second distinct question word, when the prompt asks two kinds of thing.",
    formula="second distinct wh-word in reading order",
    why="A prompt that asks both `who` and `when` needs two different kinds of evidence from one retrieval.",
    backend="spacy",
    value_range="a wh-word, or not applicable",
    status_rules=["not applicable when the prompt has fewer than two distinct wh-words"],
    example="Who wrote it and when was it published?",
    expected="when",
)
def question_type_secondary(doc, ctx):
    found = sorted(_wh_tokens(doc))
    seen: list[str] = []
    for _, _, word, _ in found:
        if word not in seen:
            seen.append(word)
    if len(seen) < 2:
        return not_applicable(
            f"the prompt has {len(seen)} distinct wh-word(s), so there is no second one"
        )
    return ok(seen[1], f"distinct wh-words in order: {quote_list(seen)}")


@register(
    "wh_word_count",
    group="structure",
    dtype="int",
    summary="How many interrogative wh-words the prompt contains.",
    formula="count of wh-words tagged WDT / WP / WP$ / WRB by spaCy",
    why="Several wh-words usually mean several information needs sharing one embedding.",
    backend="spacy",
    value_range=">= 0",
    example="Who wrote it and when was it published?",
    expected=2,
)
def wh_word_count(doc, ctx):
    found = _wh_tokens(doc)
    return ok(
        len(found),
        f"wh-words: {quote_list([f[2] for f in found])}",
        spans=[span(f[0], f[1], "wh-word") for f in found],
    )


@register(
    "distinct_wh_count",
    group="structure",
    dtype="int",
    summary="How many different wh-words the prompt uses.",
    formula="len(set(wh-words))",
    why="Distinct question words are a better multi-need signal than raw repetition of the same one.",
    backend="spacy",
    value_range=">= 0",
    example="Who wrote it and when was it published?",
    expected=2,
)
def distinct_wh_count(doc, ctx):
    words = {f[2] for f in _wh_tokens(doc)}
    return ok(len(words), f"distinct wh-words: {quote_list(sorted(words))}")


@register(
    "question_mark_count",
    group="structure",
    dtype="int",
    summary="How many question marks the prompt contains.",
    formula='prompt.count("?")',
    why="The cheapest count of explicit asks in one prompt.",
    value_range=">= 0",
    example="Who wrote it? When was it published?",
    expected=2,
)
def question_mark_count(doc, ctx):
    count = doc.text.count("?")
    spans = [span(i, i + 1, "?") for i, ch in enumerate(doc.text) if ch == "?"]
    return ok(count, f'prompt.count("?") = {count}', spans=spans)


@register(
    "ends_with_question_mark",
    group="structure",
    dtype="bool",
    summary="Whether the prompt ends with a question mark.",
    formula='prompt.strip().endswith("?")',
    why="Distinguishes a genuine question from an instruction block that merely contains one.",
    value_range="True / False",
    example="Who wrote The Hobbit?",
    expected=True,
)
def ends_with_question_mark(doc, ctx):
    value = doc.stripped.endswith("?")
    tail = doc.stripped[-12:]
    return ok(value, f"prompt ends with {tail!r} -> {value}")


@register(
    "is_wh_question",
    group="structure",
    dtype="bool",
    summary="Whether the prompt is a wh-question.",
    formula='question_type != "other"',
    why="Wh-questions target a specific span; other shapes target a document or a task.",
    value_range="True / False",
    example="Who wrote The Hobbit?",
    expected=True,
    needs=["question_type"],
)
def is_wh_question(doc, ctx):
    qtype = ctx.value("question_type", "other")
    value = qtype != "other"
    return ok(value, f"question_type = {qtype!r} -> {value}")


@register(
    "is_yes_no_question",
    group="structure",
    dtype="bool",
    summary="Whether the prompt is a yes/no question.",
    formula=(
        "the core question starts with an auxiliary or modal verb "
        "(do / does / did / is / are / was / were / can / could / will / would / "
        "should / has / have / may / might / must) and the prompt contains a question mark"
    ),
    why="Yes/no questions often need a claim to be confirmed or denied, and a chunk that merely discusses the topic is not enough.",
    value_range="True / False",
    example="Did Tolkien write The Hobbit?",
    expected=True,
)
def is_yes_no_question(doc, ctx):
    from .doc import _YES_NO_START_RE

    target = doc.core_text or doc.stripped
    match = _YES_NO_START_RE.match(target)
    has_q = "?" in doc.text
    value = bool(match) and has_q
    steps = [f"core question starts with {target[:24]!r}"]
    if match:
        steps.append(f"leading auxiliary = {match.group(0).strip()!r}")
    steps.append(f"question mark present = {has_q}")
    return ok(value, *steps)


@register(
    "is_imperative",
    group="structure",
    dtype="bool",
    summary="Whether the prompt is phrased as a command rather than a question.",
    formula=(
        "spaCy: the root of a core sentence is a base-form verb (tag VB) with no "
        "subject; fallback: the first word is a known task verb"
    ),
    why="Commands like `summarize` or `compare` imply a whole document or several documents, not one answer span.",
    backend="spacy",
    value_range="True / False",
    example="Compare ibuprofen and aspirin for fever in children.",
    expected=True,
)
def is_imperative(doc, ctx):
    root = _imperative_root(doc)
    if root is not None:
        return ok(
            True,
            f"root verb {root.text!r} is tagged {root.tag_} with no subject child",
            spans=[span(root.idx, root.idx + len(root.text), "imperative root")],
        )
    first = (doc.core_text or doc.stripped).split()
    lead = first[0].lower().strip(",.:;!?") if first else ""
    if lead in lex.TASK_VERBS:
        return ok(True, f"first word {lead!r} is a task verb (no parse available)")
    return ok(False, "no base-form root verb without a subject was found")


def _imperative_root(doc):
    sdoc = doc.spacy_doc
    if sdoc is None:
        return None
    try:
        sents = list(sdoc.sents)
    except Exception:  # pragma: no cover
        return None
    for sent in sents:
        root = sent.root
        if root.pos_ not in ("VERB", "AUX"):
            continue
        if root.tag_ != "VB":
            continue
        if any(child.dep_ in ("nsubj", "nsubjpass", "expl") for child in root.children):
            continue
        return root
    return None


@register(
    "primary_task_verb",
    group="structure",
    dtype="label",
    summary="The operation the prompt asks for, as a normalised label.",
    formula=(
        "lemma of the imperative root verb mapped through the task-verb lexicon "
        "(compare, summarize, explain, list, write, translate, calculate, define, "
        "implement, ...); without a parse, the first word of a core sentence is used, "
        "so the content verb of a question such as `who wrote X` is not mistaken for a task"
    ),
    why="Each task implies a different unit of evidence: `define` wants one span, `compare` wants two documents, `summarize` wants a whole one.",
    backend="spacy",
    value_range="a task label, or not applicable",
    status_rules=["not applicable when no task verb appears"],
    example="Compare ibuprofen and aspirin for fever in children.",
    expected="compare",
)
def primary_task_verb(doc, ctx):
    root = _imperative_root(doc)
    if root is not None and root.lemma_.lower() in lex.TASK_VERBS:
        label = lex.TASK_VERBS[root.lemma_.lower()]
        return ok(
            label,
            f"imperative root {root.text!r} has lemma {root.lemma_.lower()!r} -> {label!r}",
            spans=[span(root.idx, root.idx + len(root.text), "task verb")],
            hits=[lex.TASK_VERB_LEXICON.name],
        )
    sentence_starts = {s.start for s in doc.sentences if s.kind == "core"}
    for word in doc.words:
        if word.start not in sentence_starts:
            continue
        lemma = (word.lemma or word.lower).strip()
        if lemma in lex.TASK_VERBS:
            label = lex.TASK_VERBS[lemma]
            return ok(
                label,
                f"sentence-initial task verb {word.text!r} -> {label!r}",
                spans=[span(word.start, word.end, "task verb")],
                hits=[lex.TASK_VERB_LEXICON.name],
            )
    return not_applicable(
        "the prompt gives no command, so there is no task verb (a content verb inside a "
        "question does not count)"
    )


@register(
    "sub_question_count",
    group="structure",
    dtype="int",
    summary="How many separate things the prompt asks for.",
    formula=(
        "count core sentences that are questions or imperatives, then add one for "
        "each extra distinct wh-word inside a sentence and one for each verb "
        "coordinated with a sentence root (the `and compare` in `summarize X and "
        "compare Y`)"
    ),
    why="Every extra ask is another set of documents that a single top-k retrieval has to cover at once.",
    backend="spacy",
    value_range=">= 0",
    example="Compare ibuprofen and aspirin for fever in children after 2020. What dose is safe?",
    expected=2,
    tier=1,
    rank=18,
    rank_reason="The most direct measure of asking one query to satisfy several information needs.",
)
def sub_question_count(doc, ctx):
    core_sents = [s for s in doc.sentences if s.kind == "core"] or doc.sentences
    steps: list[str] = []
    spans: list[dict] = []
    total = 0
    sdoc = doc.spacy_doc
    # Scanned once and then filtered per sentence. Scanning inside the loop makes
    # the feature quadratic in prompt length, which is invisible on a one-line
    # question and costs tens of seconds on a pasted page.
    all_wh = _wh_tokens(doc)
    conj_verbs = _coordinated_verbs(sdoc)
    for sent in core_sents:
        unit = 0
        if "?" in sent.text:
            unit = 1
            steps.append(f"question: {sent.text!r} -> 1")
        else:
            lead = sent.text.split()
            lead_word = lead[0].lower().strip(",.:;!?") if lead else ""
            if lead_word in lex.TASK_VERBS:
                unit = 1
                steps.append(f"imperative: {sent.text!r} -> 1")
        if unit == 0:
            continue
        wh_here = {w[2] for w in all_wh if sent.start <= w[0] < sent.end}
        if len(wh_here) > 1:
            unit += len(wh_here) - 1
            steps.append(f"  + {len(wh_here) - 1} for extra wh-words {sorted(wh_here)}")
        if sdoc is not None:
            extra = [t for t in conj_verbs if sent.start <= t.idx < sent.end]
            if extra:
                unit += len(extra)
                steps.append(
                    f"  + {len(extra)} for verbs coordinated with the root: "
                    + quote_list([t.text for t in extra])
                )
                spans.extend(
                    span(t.idx, t.idx + len(t.text), "coordinated ask") for t in extra
                )
        spans.append(span(sent.start, sent.end, "ask"))
        total += unit
    steps.append(f"sub_question_count = {total}")
    return ok(total, *steps, spans=spans)


def _coordinated_verbs(sdoc):
    """Verbs hung off another verb by `and` / `or`, i.e. a second ask."""
    if sdoc is None:
        return []
    return [
        token
        for token in sdoc
        if token.dep_ == "conj"
        and token.pos_ == "VERB"
        and token.head.pos_ in ("VERB", "AUX")
    ]


@register(
    "is_multi_part",
    group="structure",
    dtype="bool",
    summary="Whether the prompt asks more than one thing.",
    formula="sub_question_count > 1",
    why="One query, two information needs: top-k may cover only one of them.",
    value_range="True / False",
    status_rules=[],
    example="Summarize the 2023 filing and compare it to 2022.",
    expected=True,
    needs=["sub_question_count"],
    tier=1,
    rank=19,
    rank_reason="A single robust flag for the multi-need case, now catching imperatives that a question-mark count misses.",
)
def is_multi_part(doc, ctx):
    count = ctx.number("sub_question_count")
    value = count > 1
    marks = doc.text.count("?")
    return ok(
        value,
        f"sub_question_count = {int(count)} -> {value}",
        f'(counting question marks alone would give {marks > 1}, since there are {marks})',
    )


@register(
    "has_enumeration_request",
    group="structure",
    dtype="bool",
    summary="Whether the prompt asks for a set of items rather than a single answer.",
    formula="match against the enumeration-request lexicon (list, examples of, top, ways to, types of ...)",
    why="A list answer needs several supporting chunks, so partial retrieval produces a partial list.",
    value_range="True / False",
    example="List three ways to reduce churn",
    expected=True,
)
def has_enumeration_request(doc, ctx):
    matches = lex.ENUMERATION_REQUEST.find(doc.core_text or doc.text)
    hits = sorted({m["text"].lower() for m in matches})
    return ok(
        bool(matches),
        f"enumeration_request lexicon matched: {quote_list(hits)}",
        hits=[lex.ENUMERATION_REQUEST.name] if matches else [],
    )


@register(
    "requested_item_count",
    group="structure",
    dtype="int",
    summary="How many items the prompt explicitly asks for.",
    formula=(
        "number found in `top N` / `list N` / `give me N` or in `N examples`, "
        "accepting digits and number words (three -> 3)"
    ),
    why="An explicit count sets the bar for retrieval: asking for five distinct facts needs five supported chunks.",
    value_range=">= 1, or not applicable",
    status_rules=["not applicable when the prompt names no count"],
    example="List three ways to reduce churn",
    expected=3,
)
def requested_item_count(doc, ctx):
    text = doc.core_text or doc.text
    for pattern, label in ((_COUNT_BEFORE_RE, "count before the noun"), (_COUNT_AFTER_RE, "count before the item noun")):
        match = pattern.search(text)
        if match:
            token = match.group(1).lower()
            value = int(token) if token.isdigit() else lex.NUMBER_WORDS.get(token, 0)
            if value:
                return ok(
                    value,
                    f"matched {match.group(0)!r} ({label})",
                    f"parsed count = {value}",
                    spans=[span(match.start(), match.end(), "requested count")],
                )
    return not_applicable("the prompt does not state how many items it wants")


@register(
    "max_parse_depth",
    group="structure",
    dtype="int",
    summary="Depth of the deepest dependency tree in the prompt.",
    formula="max over sentences of the longest path from the sentence root to a leaf token",
    why="Syntactic depth measures how much structure the query packs in, and unlike Flesch-Kincaid it stays meaningful on short prompts.",
    backend="spacy",
    value_range=">= 0",
    status_rules=["unavailable without the spaCy model, which provides the parse"],
    example="Who wrote The Hobbit?",
    expected=3,
    tier=1,
    rank=29,
    rank_reason="The complexity measure that works where readability formulas break: short, dense queries.",
)
def max_parse_depth(doc, ctx):
    sdoc = doc.spacy_doc
    if sdoc is None:
        return unavailable("spacy")
    best = 0
    best_path: list[str] = []
    for token in sdoc:
        if token.is_space or token.is_punct:
            continue
        depth = 1
        path = [token.text]
        cursor = token
        guard = 0
        # spaCy returns a fresh Token proxy on each access, so the root has to
        # be detected by index rather than by identity.
        while cursor.head.i != cursor.i and guard < 64:
            cursor = cursor.head
            depth += 1
            path.append(cursor.text)
            guard += 1
        if depth > best:
            best = depth
            best_path = path
    return ok(
        best,
        f"deepest path (leaf to root): {' -> '.join(best_path)}" if best_path else "no tokens",
        f"depth = {best}",
    )
