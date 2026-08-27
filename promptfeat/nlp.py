"""Lazy loaders for the optional NLP backends.

Nothing here raises. Every loader returns ``None`` when its backend is missing
so that a feature can report status ``unavailable`` with an install hint
instead of crashing or silently substituting a fake number.
"""

from __future__ import annotations

import functools
from typing import Any

SPACY_MODEL = "en_core_web_sm"

_INSTALL_HINTS = {
    "spacy": "pip install spacy && python -m spacy download en_core_web_sm",
    "wordfreq": "pip install wordfreq",
    "langdetect": "pip install langdetect",
    "tiktoken": "pip install tiktoken",
    "textstat": "pip install textstat",
}

_errors: dict[str, str] = {}


def install_hint(backend: str) -> str:
    return _INSTALL_HINTS.get(backend, f"pip install {backend}")


def backend_error(backend: str) -> str:
    return _errors.get(backend, "")


@functools.lru_cache(maxsize=1)
def spacy_pipeline() -> Any | None:
    """The `en_core_web_sm` pipeline (tagger, parser, NER, lemmatizer)."""
    try:
        import spacy
    except Exception as exc:  # pragma: no cover - depends on environment
        _errors["spacy"] = f"spacy is not installed ({exc})"
        return None
    try:
        return spacy.load(SPACY_MODEL)
    except Exception as exc:
        _errors["spacy"] = f"model {SPACY_MODEL} could not be loaded ({exc})"
        return None


@functools.lru_cache(maxsize=1)
def _zipf_fn() -> Any | None:
    try:
        from wordfreq import zipf_frequency
    except Exception as exc:  # pragma: no cover - depends on environment
        _errors["wordfreq"] = f"wordfreq is not installed ({exc})"
        return None
    return zipf_frequency


def zipf(word: str) -> float | None:
    """Zipf frequency of `word` in English, or None if wordfreq is missing.

    Scale: ~7 for `the`, ~4 for everyday words, <3 for rare/technical words,
    0.0 for words the corpus has never seen.
    """
    fn = _zipf_fn()
    if fn is None:
        return None
    try:
        return float(fn(word, "en"))
    except Exception:
        return 0.0


@functools.lru_cache(maxsize=1)
def _lang_detector() -> Any | None:
    try:
        from langdetect import DetectorFactory, detect_langs
    except Exception as exc:  # pragma: no cover - depends on environment
        _errors["langdetect"] = f"langdetect is not installed ({exc})"
        return None
    # langdetect is randomised by default; a fixed seed makes features stable.
    DetectorFactory.seed = 0
    return detect_langs


def detect_language(text: str) -> tuple[str, float] | None:
    """Best-guess ISO language code and its probability, or None."""
    fn = _lang_detector()
    if fn is None:
        return None
    try:
        guesses = fn(text)
    except Exception:
        return ("unknown", 0.0)
    if not guesses:
        return ("unknown", 0.0)
    best = guesses[0]
    return (str(best.lang), round(float(best.prob), 4))


@functools.lru_cache(maxsize=1)
def bpe_encoding() -> Any | None:
    """The `cl100k_base` byte-pair encoder used by GPT-3.5/4 tokenizers."""
    try:
        import tiktoken
    except Exception as exc:  # pragma: no cover - depends on environment
        _errors["tiktoken"] = f"tiktoken is not installed ({exc})"
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        _errors["tiktoken"] = f"cl100k_base could not be loaded ({exc})"
        return None


@functools.lru_cache(maxsize=1)
def textstat_module() -> Any | None:
    try:
        import textstat
    except Exception as exc:  # pragma: no cover - depends on environment
        _errors["textstat"] = f"textstat is not installed ({exc})"
        return None
    return textstat


def _version(module_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(module_name)
    except Exception:
        return "unknown"


def backend_report() -> dict[str, dict]:
    """Availability of every backend, for the UI banner and the docs header."""
    probes = {
        "spacy": lambda: spacy_pipeline() is not None,
        "wordfreq": lambda: _zipf_fn() is not None,
        "langdetect": lambda: _lang_detector() is not None,
        "tiktoken": lambda: bpe_encoding() is not None,
        "textstat": lambda: textstat_module() is not None,
    }
    report: dict[str, dict] = {}
    for name, probe in probes.items():
        available = bool(probe())
        report[name] = {
            "available": available,
            "version": _version(name) if available else "",
            "error": "" if available else backend_error(name),
            "install": "" if available else install_hint(name),
        }
    if report["spacy"]["available"]:
        report["spacy"]["model"] = SPACY_MODEL
    return report
