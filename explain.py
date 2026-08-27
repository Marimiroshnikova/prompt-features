"""Write a full calculation report for one prompt.

    python explain.py "Who wrote The Hobbit?"
    python explain.py --stdout "Who wrote The Hobbit?"

The report lists every feature with its value, its status, the arithmetic that
produced it and the exact text that matched. The web UI uses the same renderer
for its download button.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from promptfeat import explain_prompt
from promptfeat.registry import OK, STATUS_LABELS

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-") or "prompt")


def _highlight(text: str, spans: list[dict]) -> str:
    """Wrap matched character ranges in backticks, merging overlaps."""
    if not spans or not text:
        return ""
    ranges = sorted(
        ((s["start"], s["end"]) for s in spans if s["end"] > s["start"]),
        key=lambda r: r[0],
    )
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append("**[" + text[start:end] + "]**")
        cursor = end
    out.append(text[cursor:])
    return "".join(out).replace("\n", " ")


def render(report: dict) -> str:
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Prompt calculation report")
    lines.append("")
    lines.append("## Prompt")
    lines.append("")
    lines.append("```")
    lines.append(report["prompt"])
    lines.append("```")
    lines.append("")
    lines.append(
        f"- **Retrieval difficulty:** {summary['headline']} ({summary['band']} risk)"
    )
    lines.append(f"- **Category:** {summary['category']}")
    lines.append(f"- **Question type:** {summary['question_type']}")
    lines.append(f"- **Size:** {summary['words']} words, {summary['tokens']} tokens")
    lines.append(
        f"- **Features:** {summary['feature_count']} total, {summary['computed']} computed normally"
    )
    if report["core_question"] and report["core_question"] != report["prompt"].strip():
        lines.append(f"- **Core question after removing scaffolding:** `{report['core_question']}`")
    lines.append("")

    statuses = {k: v for k, v in summary["statuses"].items() if k != OK}
    if statuses:
        lines.append("### Features not computed normally")
        lines.append("")
        lines.append("| status | count |")
        lines.append("| --- | --- |")
        for status, count in sorted(statuses.items()):
            lines.append(f"| {STATUS_LABELS.get(status, status)} | {count} |")
        lines.append("")

    lines.append("## The 30 most important features")
    lines.append("")
    lines.append("| rank | feature | value | status |")
    lines.append("| --- | --- | --- | --- |")
    for item in report["top"]:
        status = "ok" if item["status"] == OK else STATUS_LABELS.get(item["status"], item["status"])
        lines.append(
            f"| {item['rank']} | `{item['name']}` | {item['display_value']} | {status} |"
        )
    lines.append("")

    for group in report["groups"]:
        lines.append("---")
        lines.append("")
        lines.append(f"## {group['title']}")
        lines.append("")
        lines.append(f"_{group['blurb']}_")
        lines.append("")
        for item in group["features"]:
            lines.extend(_feature_block(item, report))
    return "\n".join(lines)


def _feature_block(item: dict, report: dict) -> list[str]:
    lines = [f"### `{item['name']}` = {item['display_value']}", ""]
    if item["tier"] == 1:
        lines.append(f"> Tier 1, rank {item['rank']} of 30.")
        lines.append("")
    lines.append(f"- **What we see:** {item['summary']}")
    lines.append(f"- **How it is calculated:** {item['formula']}")
    if item["status"] != OK:
        lines.append(
            f"- **Status:** `{item['status']}` ({item['status_label']}) - {item['reason']}"
        )
    if item["steps"]:
        lines.append("- **Calculation for this prompt:**")
        for step in item["steps"]:
            lines.append(f"  - {step}")
    if item["spans"]:
        highlighted = _highlight(report["normalized"], item["spans"])
        if highlighted:
            lines.append(f"- **Matched text:** {highlighted}")
    if item["lexicon_hits"]:
        lines.append(f"- **Lexicons used:** {', '.join(item['lexicon_hits'])}")
    lines.append(f"- **Why retrieval:** {item['why']}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="+", help="the prompt to analyse")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing a file")
    parser.add_argument("--output", type=Path, help="explicit output path")
    args = parser.parse_args()

    prompt = " ".join(args.prompt)
    report = explain_prompt(prompt)
    text = render(report)

    if args.stdout:
        print(text)
        return
    path = args.output or REPORT_DIR / f"{slugify(prompt)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    summary = report["summary"]
    print(f"Wrote {path}")
    print(
        f"difficulty {summary['headline']} ({summary['band']}) | "
        f"{summary['feature_count']} features | {summary['computed']} computed normally"
    )


if __name__ == "__main__":
    main()
