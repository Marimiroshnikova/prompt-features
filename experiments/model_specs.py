"""Published specs for the 14 models in results.csv.

Only fields we can document go in as numbers. Unpublished or moving-target
values stay None. Do not invent Galileo/CRAG scores.

Sources (retrieved 2026-09-02):
  Gemini 2.5 Flash / Flash-Lite / Pro
    https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash
    https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite
    1,048,576 input tokens; knowledge cutoff January 2025.
  Gemini 3 / 3.1 / 3.5
    https://ai.google.dev/gemini-api/docs/gemini-3
    https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview
    https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
    1M input / 64k output; knowledge cutoff January 2025.
  Gemma 4 26B A4B and 31B
    https://ai.google.dev/gemma/docs/core/model_card_4
    256K context. No published knowledge-cutoff year.
  *-latest aliases
    Moving snapshots. Window is the current Flash/Pro/Lite 1M default;
    cutoff is unknown because the alias can change under the same id.
  max_tokens_requested
    1024 from the GAIA config/inference.json used to produce results.csv.
"""

from __future__ import annotations

from typing import Any

# Gemini 2.5 / 3.x published input limit
_GEMINI_WINDOW = 1_048_576
_GEMINI_CUTOFF = 2025  # January 2025 → year used for recency_gap
_GEMMA4_WINDOW = 256_000
_MAX_TOKENS_REQUESTED = 1024


def _row(
    *,
    family: str,
    is_preview: bool,
    is_open_source: bool,
    context_window_tokens: int | None,
    knowledge_cutoff_year: int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "model_family": family,
        "is_preview": is_preview,
        "is_open_source": is_open_source,
        "max_tokens_requested": _MAX_TOKENS_REQUESTED,
        "context_window_tokens": context_window_tokens,
        "knowledge_cutoff_year": knowledge_cutoff_year,
        "temperature": None,
        "spec_source": source,
    }


SPECS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash-lite": _row(
        family="gemini-2.5",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite",
    ),
    "gemini-2.5-flash": _row(
        family="gemini-2.5",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash",
    ),
    "gemini-2.5-pro": _row(
        family="gemini-2.5",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/models (Gemini 2.5 Pro, 1M / Jan 2025)",
    ),
    "gemini-3-flash-preview": _row(
        family="gemini-3",
        is_preview=True,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview",
    ),
    "gemini-3.1-flash-lite": _row(
        family="gemini-3.1",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/gemini-3",
    ),
    "gemini-3.1-flash-lite-preview": _row(
        family="gemini-3.1",
        is_preview=True,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/gemini-3 (3.1 Flash-Lite family)",
    ),
    "gemini-3.1-pro-preview": _row(
        family="gemini-3.1",
        is_preview=True,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/gemini-3",
    ),
    "gemini-3.1-pro-preview-customtools": _row(
        family="gemini-3.1",
        is_preview=True,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/gemini-3 (same 3.1 Pro window/cutoff; tools flag from id only)",
    ),
    "gemini-3.5-flash": _row(
        family="gemini-3.5",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=_GEMINI_CUTOFF,
        source="https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5",
    ),
    "gemma-4-26b-a4b-it": _row(
        family="gemma-4",
        is_preview=False,
        is_open_source=True,
        context_window_tokens=_GEMMA4_WINDOW,
        knowledge_cutoff_year=None,
        source="https://ai.google.dev/gemma/docs/core/model_card_4 (256K; cutoff unpublished)",
    ),
    "gemma-4-31b-it": _row(
        family="gemma-4",
        is_preview=False,
        is_open_source=True,
        context_window_tokens=_GEMMA4_WINDOW,
        knowledge_cutoff_year=None,
        source="https://ai.google.dev/gemma/docs/core/model_card_4 (256K; cutoff unpublished)",
    ),
    "gemini-flash-latest": _row(
        family="gemini-latest",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=None,
        source="latest alias: 1M is the current Flash default; cutoff unpublished (moving snapshot)",
    ),
    "gemini-flash-lite-latest": _row(
        family="gemini-latest",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=None,
        source="latest alias: 1M is the current Flash-Lite default; cutoff unpublished (moving snapshot)",
    ),
    "gemini-pro-latest": _row(
        family="gemini-latest",
        is_preview=False,
        is_open_source=False,
        context_window_tokens=_GEMINI_WINDOW,
        knowledge_cutoff_year=None,
        source="latest alias: 1M is the current Pro default; cutoff unpublished (moving snapshot)",
    ),
}


def spec_for(model_id: str) -> dict[str, Any]:
    if model_id not in SPECS:
        raise KeyError(f"No spec row for {model_id!r}")
    return dict(SPECS[model_id])
