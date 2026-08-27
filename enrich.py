"""Add feature columns from a `question` column. Spec: FEATURES.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from promptfeat import TOP30_FEATURES
from promptfeat import extract_features

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT = DATA_DIR / "example_prompts.csv"
DEFAULT_OUTPUT = DATA_DIR / "example_prompts_enriched.csv"


def enrich(
    df: pd.DataFrame,
    text_col: str = "question",
    *,
    top30: bool = False,
    with_status: bool = False,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Expected a {text_col!r} column")
    rows = (
        df[text_col]
        .fillna("")
        .astype(str)
        .map(lambda text: extract_features(text, with_status=with_status))
    )
    feature_df = pd.DataFrame(list(rows), index=df.index)
    if top30:
        keep = [c for c in feature_df.columns if c.split("__")[0] in TOP30_FEATURES]
        feature_df = feature_df[keep]
    return pd.concat([df, feature_df], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--column", default="question")
    parser.add_argument(
        "--top30", action="store_true", help="write only the 30 tier-1 features"
    )
    parser.add_argument(
        "--with-status",
        action="store_true",
        help="also write <name>__status and <name>__reason columns",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    enriched = enrich(
        df, args.column, top30=args.top30, with_status=args.with_status
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)

    feature_cols = [c for c in enriched.columns if c != args.column]
    print(f"Read {args.input}")
    print(f"Saved {len(enriched)} rows x {len(feature_cols)} feature columns to {args.output}")
    preview = [c for c in TOP30_FEATURES if c in enriched.columns][:8]
    if preview:
        print("\nPreview of the first tier-1 features:")
        print(enriched[[args.column, *preview]].to_string(index=False))
    blanks = {
        col: int(enriched[col].isna().sum())
        for col in feature_cols
        if not col.endswith(("__status", "__reason")) and enriched[col].isna().any()
    }
    if blanks:
        print("\nFeatures that could not be computed for some rows (None cells):")
        for col, count in sorted(blanks.items(), key=lambda kv: -kv[1]):
            print(f"  {col}: {count} of {len(enriched)} rows")
        print("Re-run with --with-status to get the reason in the CSV.")


if __name__ == "__main__":
    main()
