"""PromptDoc: one shared analysis pass over a prompt.

Every feature reads from this object, so the tokenisation, sentence split,
spaCy parse and BPE encoding happen once per prompt instead of once per
feature. All character offsets are valid in both the raw and the normalised
text, because normalisation only ever swaps one character for one character.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field

from . import lexicons as lex
from . import nlp

# Length-preserving character folding. Curly quotes and exotic spaces are the
# reason `contains_negation` used to miss "don't" pasted from a browser.
_CHAR_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u02bc": "'", "\u2032": "'", "\u00b4": "'", "\u0060": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ",
    "\u202f": " ", "\u3000": " ", "\ufeff": " ", "\u200b": " ",
    "\t": " ",
}
_FOLD_TABLE = str.maketrans(_CHAR_FOLD)

_WORD_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'\-\./_+#]*[A-Za-z0-9%])?|[A-Za-z0-9]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\s\)\"']*\s+|\n+")

_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_TAG_BLOCK_RE = re.compile(
    r"<(document|doc|context|passage|text|article|data)\b.*?</\1>", re.DOTALL | re.IGNORECASE
)
_TRIPLE_QUOTE_RE = re.compile(r'"""(.*?)"""', re.DOTALL)
_SECTION_HEADER_RE = re.compile(
    # Bounded at the first sentence end, so a one-line prompt such as
    # `Instructions: use the docs. Compare A and B.` gives up the real question
    # instead of marking the whole line as instruction.
    r"(?im)^[ \t]*#{0,4}[ \t]*(system|systems|instruction|instructions|role|persona|"
    r"task|rules?|constraints?|guidelines?|requirements?|format|output format|"
    r"response format|context|background|notes?)[ \t]*:[^\n.?!]*[.?!]?"
)
_FEW_SHOT_LINE_RE = re.compile(
    # Stops at the first sentence end so that `Example 1: Paris. What is the
    # capital of France?` gives up the question instead of swallowing it.
    r"(?im)^[ \t]*(?:#{2,4}[ \t]*)?(example[ \t]*\d*|examples|q|a|question|answer|"
    r"input|output|user|assistant)[ \t]*:[^\n.?!]*[.?!]?"
)
_QUOTED_RE = re.compile(r"\"([^\"\n]{2,})\"|'([^'\n]{3,})'|`([^`\n]{2,})`")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"')]+")
_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s]+)|(?:\.{0,2}/[\w.-]+/[\w./-]+)|"
    r"\b[\w-]+\.(?:py|js|ts|tsx|java|cpp|c|h|cs|rb|go|rs|php|sql|json|ya?ml|"
    r"csv|txt|md|html|css|sh|ps1|ipynb|pdf|docx?|xlsx?)\b"
)
_CODE_SPAN_RE = re.compile(
    # Bare keywords like `from` and `return` are ordinary English, so the
    # keyword branch only fires at the start of a line, where code actually
    # sits. Otherwise "how did it differ from 1929" would look like code.
    r"(?m)`[^`\n]+`|\b\w+\([^)\n]*\)|^[ \t]*(?:def|class|import|from|function|const|"
    r"let|var|return|public|private|SELECT|INSERT|UPDATE|DELETE)\b[ \t]+\S+"
)
_MATH_RE = re.compile(
    r"\d+\s*[+\-*/^=<>]\s*\d+|\b\d+\s*%\s*of\b|[=<>]\s*\d+|"
    r"\b(?:sqrt|log|sin|cos|tan|sum|integral|derivative)\s*\(|\$[^$\n]+\$"
)
_ID_LIKE_RE = re.compile(
    r"\b(?:[vV]\d+(?:\.\d+)+|[A-Z]{2,}-\d+|\d+[A-Za-z]-\d+|"
    # `\d+` followed by letters is an identifier like `4chan`, but the negative
    # lookahead keeps decades and ordinals (1990s, 21st) out of it.
    r"(?:[A-Za-z]+\d+[A-Za-z0-9]*|\d+(?!(?:s|st|nd|rd|th)\b)[A-Za-z]+[A-Za-z0-9]*)|"
    r"10\.\d{4,}/\S+|\b\d{3}-\d{3,}|[0-9a-f]{7,40})\b"
)
_ACRONYM_RE = re.compile(r"\b(?:[A-Z]{2,}(?:s|\'s)?|(?:[A-Z]\.){2,})\b")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\U0000FE0F\U00002700-\U000027BF]"
)
_NUMERAL_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,000  12,345.67
    r"|\d+(?:\.\d+)?)"  # 12  3.5
    r"(?:st|nd|rd|th|s)?"  # 1st  1990s
    r"(?!\w)"
)
_YEAR_RE = re.compile(r"(?<![\w.])((?:1[0-9]|20)\d{2})(?:s)?(?!\w)")
_PERCENT_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:%|percent\b|per cent\b|pct\b)|\b\d+\s*(?:percentage points?)\b"
)
_SCALE_WORD = r"(?:k|m|bn|b|billion|million|thousand|trillion)"
_CURRENCY_AMOUNT_RE = re.compile(
    rf"(?i)(?:[$€£¥₹]\s*\d[\d,]*(?:\.\d+)?\s*{_SCALE_WORD}?)"
    rf"|(?:\d[\d,]*(?:\.\d+)?\s*{_SCALE_WORD}?\s*"
    r"(?:usd|eur|gbp|jpy|cny|inr|dollars?|euros?|pounds?|yen|rupees?))"
)
_DATE_LIKE_RE = re.compile(
    r"(?i)\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\bq[1-4]\s*(?:of\s*)?(?:19|20)\d{2}\b"
)
_BOOLEAN_OP_RE = re.compile(r"(?<![\w])(?:AND|OR|NOT)(?![\w])")
_YES_NO_START_RE = re.compile(
    r"(?i)^\s*(?:do|does|did|is|are|was|were|am|can|could|will|would|shall|should|"
    r"has|have|had|may|might|must)\b"
)
_WH_RE = re.compile(r"(?i)\b(who|whom|whose|what|when|where|why|how|which)\b")

SEGMENT_KINDS = ("core", "instruction", "few_shot", "context")


@dataclass
class Word:
    text: str
    start: int
    end: int
    lower: str
    is_stop: bool
    pos: str = ""
    tag: str = ""
    lemma: str = ""
    dep: str = ""
    head: int = -1
    ent_type: str = ""
    index: int = 0

    @property
    def is_alpha(self) -> bool:
        return self.text.isalpha()


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    kind: str = "core"


@dataclass
class Segment:
    kind: str
    start: int
    end: int
    text: str
    source: str = ""


class PromptDoc:
    """Everything the features need, computed once."""

    def __init__(self, raw: str) -> None:
        self.raw = raw if isinstance(raw, str) else ""
        self.text = self.raw.translate(_FOLD_TABLE)
        if len(self.text) != len(self.raw):  # pragma: no cover - guard
            raise AssertionError("normalisation must preserve character offsets")
        self.lower = self.text.lower()
        self.stripped = self.text.strip()
        self.is_empty = not self.stripped

        # Backward-compatible raw counts: these three features were defined on
        # the raw string and must keep their original values.
        self.whitespace_words = self.raw.split()

        self.lines = self.text.split("\n")
        self.segments: list[Segment] = _segment(self.text)
        self.core_text, self.core_spans = _core_text(self.text, self.segments)

    # --- lazy backends ------------------------------------------------------

    @functools.cached_property
    def spacy_doc(self):
        pipeline = nlp.spacy_pipeline()
        if pipeline is None or self.is_empty:
            return None
        try:
            return pipeline(self.text)
        except Exception:  # pragma: no cover - defensive
            return None

    @property
    def has_spacy(self) -> bool:
        return self.spacy_doc is not None

    @functools.cached_property
    def words(self) -> list[Word]:
        sdoc = self.spacy_doc
        words: list[Word] = []
        if sdoc is not None:
            for token in sdoc:
                if token.is_space or token.is_punct:
                    continue
                words.append(
                    Word(
                        text=token.text,
                        start=token.idx,
                        end=token.idx + len(token.text),
                        lower=token.lower_,
                        is_stop=bool(token.is_stop),
                        pos=token.pos_,
                        tag=token.tag_,
                        lemma=token.lemma_.lower(),
                        dep=token.dep_,
                        head=token.head.i,
                        ent_type=token.ent_type_,
                        index=token.i,
                    )
                )
            return words
        for i, match in enumerate(_WORD_RE.finditer(self.text)):
            token_text = match.group(0)
            words.append(
                Word(
                    text=token_text,
                    start=match.start(),
                    end=match.end(),
                    lower=token_text.lower(),
                    is_stop=token_text.lower() in lex.STOPWORDS,
                    lemma=token_text.lower(),
                    index=i,
                )
            )
        return words

    @functools.cached_property
    def alpha_words(self) -> list[Word]:
        return [w for w in self.words if any(ch.isalpha() for ch in w.text)]

    @functools.cached_property
    def politeness_spans(self) -> list[tuple[int, int]]:
        return [(m["start"], m["end"]) for m in lex.POLITENESS.find(self.text)]

    @functools.cached_property
    def content_words(self) -> list[Word]:
        """Words that carry retrievable content.

        Politeness filler is excluded as well as stopwords: `Hi`, `please` and
        `Thanks` are not things to search for, and counting them let a padded
        prompt look more specific than a bare one.
        """
        polite = self.politeness_spans
        return [
            w
            for w in self.alpha_words
            if not w.is_stop
            and len(w.text) > 1
            and not any(a <= w.start and w.end <= b for a, b in polite)
        ]

    @functools.cached_property
    def sentences(self) -> list[Sentence]:
        sdoc = self.spacy_doc
        sents: list[Sentence] = []
        if sdoc is not None:
            try:
                for sent in sdoc.sents:
                    text = sent.text.strip()
                    if text:
                        start = sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
                        sents.append(Sentence(text=text, start=start, end=start + len(text)))
            except Exception:  # pragma: no cover - parser missing
                sents = []
        if not sents:
            cursor = 0
            for piece in _SENT_SPLIT_RE.split(self.text):
                if piece is None:
                    continue
                idx = self.text.find(piece, cursor)
                if idx < 0:
                    idx = cursor
                cursor = idx + len(piece)
                text = piece.strip()
                if text:
                    start = idx + (len(piece) - len(piece.lstrip()))
                    sents.append(Sentence(text=text, start=start, end=start + len(text)))
        for sent in sents:
            sent.kind = _sentence_kind(sent, self.segments)
        return sents

    @functools.cached_property
    def bpe_tokens(self) -> list[int] | None:
        encoding = nlp.bpe_encoding()
        if encoding is None:
            return None
        return encoding.encode(self.raw)

    @functools.cached_property
    def bpe_per_word(self) -> dict[str, int] | None:
        """BPE token count per distinct word, always encoded with a leading
        space so the count matches how the word appears mid-sentence."""
        encoding = nlp.bpe_encoding()
        if encoding is None:
            return None
        counts: dict[str, int] = {}
        for word in self.alpha_words:
            if word.text not in counts:
                counts[word.text] = len(encoding.encode(" " + word.text))
        return counts

    @functools.cached_property
    def zipf_scores(self) -> dict[str, float] | None:
        scores: dict[str, float] = {}
        for word in self.alpha_words:
            key = word.lower.strip("'-")
            if not key or key in scores:
                continue
            value = nlp.zipf(key)
            if value is None:
                return None
            scores[key] = value
        return scores

    @functools.cached_property
    def entities(self) -> list[dict] | None:
        sdoc = self.spacy_doc
        if sdoc is None:
            return None
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in sdoc.ents
        ]

    @functools.cached_property
    def noun_chunks(self) -> list[dict] | None:
        sdoc = self.spacy_doc
        if sdoc is None:
            return None
        try:
            return [
                {"text": chunk.text, "start": chunk.start_char, "end": chunk.end_char}
                for chunk in sdoc.noun_chunks
            ]
        except Exception:  # pragma: no cover - needs the parser
            return None

    # --- cheap derived text -------------------------------------------------

    @functools.cached_property
    def quoted_spans(self) -> list[dict]:
        spans = []
        for match in _QUOTED_RE.finditer(self.text):
            inner = next((g for g in match.groups() if g), "")
            if inner.strip():
                spans.append(
                    {"text": inner, "start": match.start(), "end": match.end()}
                )
        return spans

    @functools.cached_property
    def core_words(self) -> list[str]:
        return self.core_text.split()

    @functools.cached_property
    def syllable_counts(self) -> dict[str, int] | None:
        module = nlp.textstat_module()
        if module is None:
            return None
        counts: dict[str, int] = {}
        for word in self.alpha_words:
            if word.lower not in counts:
                try:
                    counts[word.lower] = int(module.syllable_count(word.text))
                except Exception:
                    counts[word.lower] = 1
        return counts

    def snippet(self, start: int, end: int, pad: int = 0) -> str:
        return self.text[max(0, start - pad) : min(len(self.text), end + pad)]

    def core_question_sentences(self) -> list[Sentence]:
        return [s for s in self.sentences if s.kind == "core"]


# --- segmentation -----------------------------------------------------------


def _add_segment(
    segments: list[Segment], text: str, start: int, end: int, kind: str, source: str
) -> None:
    if end > start:
        segments.append(
            Segment(kind=kind, start=start, end=end, text=text[start:end], source=source)
        )


def _segment(text: str) -> list[Segment]:
    """Structural blocks that are not part of the information need."""
    segments: list[Segment] = []
    for match in _FENCE_RE.finditer(text):
        _add_segment(segments, text, match.start(), match.end(), "context", "fenced block")
    for match in _TAG_BLOCK_RE.finditer(text):
        _add_segment(segments, text, match.start(), match.end(), "context", "tagged block")
    for match in _TRIPLE_QUOTE_RE.finditer(text):
        _add_segment(segments, text, match.start(), match.end(), "context", "triple-quoted block")
    for match in _FEW_SHOT_LINE_RE.finditer(text):
        if _covered(segments, match.start(), match.end()):
            continue
        _add_segment(segments, text, match.start(), match.end(), "few_shot", "example line")
    for match in _SECTION_HEADER_RE.finditer(text):
        if _covered(segments, match.start(), match.end()):
            continue
        kind = "context" if re.match(r"(?i)^[ \t]*#{0,4}[ \t]*(context|background)", match.group(0)) else "instruction"
        _add_segment(segments, text, match.start(), match.end(), kind, "section header")
    return sorted(segments, key=lambda s: s.start)


def _covered(segments: list[Segment], start: int, end: int) -> bool:
    return any(seg.start <= start and seg.end >= end for seg in segments)


def _sentence_kind(sent: Sentence, segments: list[Segment]) -> str:
    for seg in segments:
        overlap = min(sent.end, seg.end) - max(sent.start, seg.start)
        if overlap > 0 and overlap >= 0.6 * max(1, sent.end - sent.start):
            return seg.kind
    if "?" in sent.text:
        return "core"
    if lex.META_INSTRUCTION.matches(sent.text) or lex.ROLE_PROMPT.matches(sent.text):
        return "instruction"
    # A sentence that is nothing but courtesy ("Thanks!") is boilerplate.
    stripped = lex.POLITENESS.pattern.sub(" ", sent.text)
    if not re.search(r"[A-Za-z0-9]", stripped):
        return "instruction"
    return "core"


def _core_text(text: str, segments: list[Segment]) -> tuple[str, list[tuple[int, int]]]:
    """The prompt with instruction, example and context scaffolding removed."""
    doc_sents = _rough_sentences(text)
    spans: list[tuple[int, int]] = []
    for sent in doc_sents:
        if _sentence_kind(sent, segments) == "core":
            spans.append((sent.start, sent.end))
    core = " ".join(text[a:b] for a, b in spans).strip()
    return core, spans


def _rough_sentences(text: str) -> list[Sentence]:
    """Regex sentence split used for segmentation before spaCy is loaded."""
    sents: list[Sentence] = []
    cursor = 0
    for piece in _SENT_SPLIT_RE.split(text):
        if not piece:
            continue
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(piece)
        body = piece.strip()
        if body:
            start = idx + (len(piece) - len(piece.lstrip()))
            sents.append(Sentence(text=body, start=start, end=start + len(body)))
    return sents


def build(prompt: str) -> PromptDoc:
    return PromptDoc(prompt)
