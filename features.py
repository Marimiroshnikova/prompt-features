"""Extract prompt features. Spec: FEATURES.md.

This module is the stable public surface. The implementation lives in the
`promptfeat` package, where each feature is declared once in a registry that
also generates FEATURES.md.

    from features import extract_features
    extract_features("Who wrote The Hobbit?")

Every value comes from the prompt text alone: no model name, no retrieved
documents, no gold answer. A feature that cannot be honestly computed for a
given prompt returns None; call with `with_status=True` to see why.
"""

from __future__ import annotations

import json

from promptfeat import (
    FEATURE_NAMES,
    REGISTRY,
    TOP30_FEATURES,
    explain_prompt,
)
from promptfeat import extract_features as _extract
from promptfeat.registry import STATUS_LABELS

__all__ = [
    "extract_features",
    "extract_top_features",
    "explain_prompt",
    "FEATURE_NAMES",
    "TOP30_FEATURES",
    "REGISTRY",
]


def extract_features(prompt: str, *, with_status: bool = False) -> dict:
    """All features for one prompt as a flat dict.

    `with_status=True` adds `<name>__status` and `<name>__reason` columns, which
    say whether a value is `ok`, `unreliable`, `not_applicable`, `undefined` or
    `unavailable`, and why.
    """
    return _extract(prompt, with_status=with_status)


def extract_top_features(prompt: str, *, with_status: bool = False) -> dict:
    """Only the 30 tier-1 features, in rank order."""
    full = _extract(prompt, with_status=with_status)
    out: dict = {}
    for name in TOP30_FEATURES:
        out[name] = full.get(name)
        if with_status:
            out[f"{name}__status"] = full.get(f"{name}__status")
            out[f"{name}__reason"] = full.get(f"{name}__reason")
    return out


def _print_report(prompt: str) -> None:
    report = explain_prompt(prompt)
    summary = report["summary"]
    print("=" * 78)
    print(prompt)
    print("-" * 78)
    print(
        f"difficulty {summary['headline']} ({summary['band']}) | "
        f"{summary['category']} | type {summary['question_type']} | "
        f"{summary['words']} words | {summary['tokens']} tokens"
    )
    if report["core_question"] != prompt.strip():
        print(f"core question: {report['core_question']!r}")
    print(f"top 30 of {summary['feature_count']} features:")
    for item in report["top"]:
        note = ""
        if item["status"] != "ok":
            note = f"   [{STATUS_LABELS[item['status']]}: {item['reason']}]"
        print(f"  {item['rank']:>2}. {item['name']:<32} {item['display_value']:<12}{note}")
    not_computed = [f for f in report["features"] if f["status"] != "ok"]
    if not_computed:
        print(f"{len(not_computed)} feature(s) could not be computed for this prompt:")
        for item in not_computed:
            print(f"  - {item['name']}: {STATUS_LABELS[item['status']]} - {item['reason']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--json":
            print(json.dumps(extract_features(" ".join(sys.argv[2:])), indent=2))
        else:
            _print_report(" ".join(sys.argv[1:]))
    else:
        samples = [
            "Who wrote The Hobbit?",
            "What about it?",
            "Instructions:\nUse the docs only.\n"
            "Compare ibuprofen and aspirin for fever in children after 2020. "
            "What dose is safe? What should be avoided?",
        ]
        for text in samples:
            _print_report(text)
        print("=" * 78)
        print(
            f"{len(FEATURE_NAMES)} features available. "
            "Run `python app.py` for the web view, or `python explain.py \"your prompt\"` "
            "for a full Markdown calculation report."
        )
