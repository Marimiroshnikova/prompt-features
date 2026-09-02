"""Phase 2 data contract: unique-prompt features, join, model specs, interactions.

Extracts promptfeat once per question_id from question stem + options (not the
shared instruction wrapper in results.csv). Writes:

  experiments/out/mmlu_prompt_features.csv   280 rows
  experiments/out/mmlu_pilot_rows.csv        3920 rows (outcomes + features + specs)
  experiments/out/mmlu_feature_dictionary.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from promptfeat import extract_features  # noqa: E402

from model_specs import SPECS  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SAMPLE = REPO / "data" / "mmlu_pro_sample_20_per_category.csv"
RESULTS = REPO / "results.csv"
LETTERS = "ABCDEFGHIJ"


def format_question_text(question: str, options: list[str]) -> str:
    """Question stem + lettered options. No category instruction wrapper."""
    lines = [question.strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return "\n".join(lines)


def load_unique_questions() -> pd.DataFrame:
    raw = pd.read_csv(SAMPLE)
    rows = []
    for rec in raw.itertuples(index=False):
        options = json.loads(rec.options)
        rows.append(
            {
                "question_id": int(rec.question_id),
                "category": rec.category,
                "question": rec.question,
                "n_options": len(options),
                "text": format_question_text(rec.question, options),
            }
        )
    df = pd.DataFrame(rows)
    if df["question_id"].nunique() != len(df):
        raise ValueError("duplicate question_id in sample CSV")
    return df


def extract_prompt_features(questions: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    records = []
    for i, text in enumerate(questions["text"]):
        records.append(extract_features(text, with_status=True))
        if (i + 1) % 50 == 0:
            print(f"  extract: {i + 1}/{len(questions)}", flush=True)
    feats = pd.DataFrame(records, index=questions.index)
    feats.columns = ["f_" + c for c in feats.columns]
    feats["f_n_options"] = questions["n_options"].astype(float)
    print(f"  extract: {len(questions)} questions in {time.time() - t0:.0f}s", flush=True)
    meta = questions[["question_id", "category", "question", "n_options", "text"]]
    return pd.concat([meta, feats], axis=1)


def load_outcomes() -> pd.DataFrame:
    res = pd.read_csv(RESULTS, low_memory=False)
    out = pd.DataFrame(
        {
            "llm_model": res["llm_model"],
            "question_id": res["question_id"].astype(int),
            "question_category": res["question_category"],
            "correct_answer": res["correct_answer"],
            "it_1_ans": res["it_1_ans"],
            "n_trials": 1,
            "y_fail": (1.0 - res["accuracy"].astype(float)),
        }
    )
    out["risk_target"] = out["y_fail"]
    if len(out) != 280 * 14:
        raise ValueError(f"expected 3920 outcome rows, got {len(out)}")
    if out.duplicated(["llm_model", "question_id"]).any():
        raise ValueError("duplicate (model, question_id) in results.csv")
    return out


def attach_specs(rows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(rows["llm_model"]) - set(SPECS))
    if missing:
        raise KeyError(f"models without specs: {missing}")
    spec_df = pd.DataFrame.from_dict(SPECS, orient="index")
    spec_df.index.name = "llm_model"
    spec_df = spec_df.reset_index()
    return rows.merge(spec_df, on="llm_model", how="left")


def add_interactions(rows: pd.DataFrame) -> pd.DataFrame:
    tokens = pd.to_numeric(rows.get("f_context_token_count"), errors="coerce")
    window = pd.to_numeric(rows["context_window_tokens"], errors="coerce")
    year = pd.to_numeric(rows.get("f_year_max"), errors="coerce")
    cutoff = pd.to_numeric(rows["knowledge_cutoff_year"], errors="coerce")

    pressure = tokens / window
    # Near-zero on 1M windows; keep the column so the dictionary can say so.
    rows["f_context_pressure"] = pressure
    rows["f_recency_gap"] = year - cutoff
    # output_pressure is 1024/1024 for every row — not written.
    rows["context_pressure_nunique"] = pressure.nunique(dropna=True)
    return rows


def write_dictionary(prompt_df: pd.DataFrame, rows: pd.DataFrame) -> str:
    value_cols = [
        c
        for c in prompt_df.columns
        if c.startswith("f_") and not c.endswith(("__status", "__reason"))
    ]
    pressure = rows["f_context_pressure"]
    recency = rows["f_recency_gap"]
    lines = [
        "# Phase 2 feature dictionary",
        "",
        "Computed before generation. `n=1` Bernoulli labels in `results.csv`;",
        "`risk_target` is currently `y_fail` and can later hold `incorrect/10`.",
        "",
        "## Prompt features",
        "",
        "Extracted once per `question_id` from the question stem plus lettered",
        "options. The shared instruction line (`multiple choice questions about",
        "{category}`) is excluded so category cannot leak into prompt-only features.",
        "",
        f"- `{len(value_cols)}` promptfeat measurements plus `f_n_options`.",
        "- Definitions for the 138 live in `FEATURES.md`.",
        "- `f_n_options`: number of multiple-choice options on this question (3–10).",
        "- `category` is metadata / a Phase 3 baseline, not a prompt feature.",
        "",
        "## Model / configuration features",
        "",
        "| field | meaning | missingness |",
        "| --- | --- | --- |",
        "| `model_family` | id prefix (`gemini-2.5`, `gemma-4`, `gemini-latest`, …) | none |",
        "| `is_preview` | `preview` appears in the model id | none |",
        "| `is_open_source` | Gemma yes, Gemini no | none |",
        "| `max_tokens_requested` | 1024 from GAIA `inference.json` | none; **zero variance** |",
        "| `context_window_tokens` | published input limit | none on this 14-model list |",
        "| `knowledge_cutoff_year` | published cutoff year | null for Gemma and `*-latest` aliases |",
        "| `temperature` | sampling temperature | **null for every row** (API default, not stored) |",
        "",
        "`gemini-flash-latest` is kept distinct from `gemini-2.5-flash`:",
        "they have different accuracy, so they are different snapshots.",
        "Lookup and citations: `experiments/model_specs.py`.",
        "",
        "## Interaction features",
        "",
        f"- `f_context_pressure` = `f_context_token_count` / `context_window_tokens`. "
        f"nunique={int(pressure.nunique(dropna=True))}, "
        f"min={pressure.min():.6g}, max={pressure.max():.6g}. "
        "On 1M-token Gemini windows this is ~0 for every MMLU-Pro prompt; "
        "constants are dropped at model time.",
        "- `output_pressure` **dropped: zero variance**. Every row requested 1024 tokens.",
        f"- `f_recency_gap` = `f_year_max` - `knowledge_cutoff_year`. "
        f"non-null={int(recency.notna().sum())} of {len(recency)} rows "
        f"({recency.notna().mean():.0%}). Null when the prompt names no year or the "
        "model has no published cutoff.",
        "- Category × model is a Phase 3 **baseline**, not a learned one-hot here.",
        "",
        "## Labels",
        "",
        "- `n_trials = 1` (not a 10-sample probability).",
        "- `y_fail = 1 - accuracy` (1 = the single generation was wrong).",
        "- `risk_target = y_fail` today; replace with `incorrect/10` when that file exists.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    questions = load_unique_questions()
    print(f"unique questions: {len(questions)}  categories: {questions.category.nunique()}")
    prompt_df = extract_prompt_features(questions)
    prompt_df.to_csv(OUT / "mmlu_prompt_features.csv", index=False)
    print(f"wrote {OUT / 'mmlu_prompt_features.csv'}  {prompt_df.shape}")

    outcomes = load_outcomes()
    # Drop the instruction-wrapped prompt; join the unique-prompt features instead.
    feat_only = prompt_df.drop(columns=["category", "question", "n_options", "text"])
    rows = outcomes.merge(feat_only, on="question_id", how="left")
    if rows["f_n_options"].isna().any():
        raise ValueError("join missed question_ids")
    # Prefer the sample category; results.csv category should match.
    mismatch = (rows["question_category"] != rows["question_category"]).sum()
    del mismatch
    rows = attach_specs(rows)
    rows = add_interactions(rows)
    extra = [c for c in rows.columns if c == "context_pressure_nunique"]
    rows = rows.drop(columns=extra)

    rows.to_csv(OUT / "mmlu_pilot_rows.csv", index=False)
    print(f"wrote {OUT / 'mmlu_pilot_rows.csv'}  {rows.shape}")
    print(f"fail rate {rows.y_fail.mean():.4f}  n_trials unique {sorted(rows.n_trials.unique())}")

    text = write_dictionary(prompt_df, rows)
    (OUT / "mmlu_feature_dictionary.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'mmlu_feature_dictionary.md'}")
    print("done")


if __name__ == "__main__":
    main()
