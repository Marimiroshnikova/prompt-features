"""Attach Phase 2 features to the run-results dataset.

Plan Phase 2: prompt / model-config / interaction columns, using only
information available before generation. Joins onto results_n2.csv.

Writes:
  pilot_results.csv                         repo root (runs + features)
  experiments/out/pilot_results.csv         same copy
  experiments/out/mmlu_feature_dictionary.md  refreshed for n=2
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mmlu_pilot import attach_specs  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
RUNS = REPO / "results_n2.csv"
SAMPLE = REPO / "data" / "mmlu_pro_sample_20_per_category.csv"
PROMPT_FEATS = OUT / "mmlu_prompt_features.csv"
EXAM_FEATS = OUT / "mmlu_exam_features.csv"
DEST_ROOT = REPO / "pilot_results.csv"
DEST_OUT = OUT / "pilot_results.csv"
DEST_QUESTIONS = REPO / "questions_with_features.csv"
DICT_PATH = OUT / "mmlu_feature_dictionary.md"
MAX_GEN_TOKENS = 1024

FRONT = [
    "question_id",
    "question_category",
    "question",
    "options",
    "correct_answer",
    "llm_model",
    "it_1_ans",
    "it_1_raw",
    "it_2_ans",
    "it_2_raw",
    "it_1_parse_ok",
    "it_2_parse_ok",
    "correct_answered_num",
    "accuracy",
    "risk_target",
    "flip",
    "n_trials",
]
MODEL_BLOCK = [
    "model_family",
    "is_preview",
    "is_open_source",
    "has_custom_tools",
    "context_window_tokens",
    "knowledge_cutoff_year",
    "max_tokens_requested",
    "output_token_limit",
    "temperature",
    "top_p",
    "spec_source",
]
INTER_BLOCK = [
    "f_context_pressure",
    "f_output_pressure",
    "f_recency_gap",
    "model_x_category",
    "f_complexity_x_capability",
]


def _prompt_value_cols(feats: pd.DataFrame) -> list[str]:
    return [
        c
        for c in feats.columns
        if c.startswith("f_") and not c.endswith(("__status", "__reason"))
    ]


def main() -> None:
    runs = pd.read_csv(RUNS, low_memory=False)
    if len(runs) != 280 * 14:
        raise ValueError(f"expected 3920 run rows, got {len(runs)}")

    sample = pd.read_csv(SAMPLE)
    questions = sample[["question_id", "question"]].drop_duplicates("question_id")
    runs = runs.merge(questions, on="question_id", how="left", validate="many_to_one")
    if runs["question"].isna().any():
        raise ValueError("sample CSV missed some question_id")

    prompt = pd.read_csv(PROMPT_FEATS, low_memory=False)
    pcols = ["question_id"] + _prompt_value_cols(prompt)
    prompt = prompt[pcols]

    exam = pd.read_csv(EXAM_FEATS, low_memory=False)
    exam_cols = [c for c in exam.columns if c.startswith("f_")]
    exam = exam[["question_id"] + exam_cols]
    # Exam flags already use f_ names; do not duplicate columns that promptfeat now has.
    overlap = [c for c in exam_cols if c in prompt.columns and c != "question_id"]
    if overlap:
        exam = exam.drop(columns=overlap)

    rows = runs.merge(prompt, on="question_id", how="left", validate="many_to_one")
    rows = rows.merge(exam, on="question_id", how="left", validate="many_to_one")
    if rows["f_n_options"].isna().any():
        raise ValueError("prompt features missed some question_id")

    rows = attach_specs(rows)
    rows["top_p"] = pd.NA
    rows["has_custom_tools"] = (
        rows["llm_model"].astype(str).str.contains("customtools", case=False)
    )
    rows["output_token_limit"] = rows["max_tokens_requested"]

    tokens = pd.to_numeric(rows["f_context_token_count"], errors="coerce")
    window = pd.to_numeric(rows["context_window_tokens"], errors="coerce")
    year = pd.to_numeric(rows["f_year_max"], errors="coerce")
    cutoff = pd.to_numeric(rows["knowledge_cutoff_year"], errors="coerce")

    rows["f_context_pressure"] = tokens / window
    rows["f_output_pressure"] = float(MAX_GEN_TOKENS) / float(MAX_GEN_TOKENS)
    rows["f_recency_gap"] = year - cutoff
    rows["model_x_category"] = (
        rows["llm_model"].astype(str) + "|" + rows["question_category"].astype(str)
    )
    # Capability proxy we can cite: published context window, not a quality rank.
    rows["f_complexity_x_capability"] = tokens * (window / 1_000_000.0)

    rows["risk_target"] = 1.0 - rows["accuracy"].astype(float)

    feat_cols = [c for c in rows.columns if c.startswith("f_") and c not in INTER_BLOCK]
    ordered = (
        [c for c in FRONT if c in rows.columns]
        + [c for c in MODEL_BLOCK if c in rows.columns]
        + [c for c in INTER_BLOCK if c in rows.columns]
        + feat_cols
    )
    leftover = [c for c in rows.columns if c not in ordered and c != "prompt"]
    rows = rows[ordered + leftover]

    rows.to_csv(DEST_ROOT, index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(DEST_OUT, index=False)

    q_keep = ["question_id", "question_category", "question", "options", "correct_answer"]
    q_keep += [c for c in rows.columns if c.startswith("f_") and c not in INTER_BLOCK]
    q_keep += [c for c in exam.columns if c.startswith("f_") and c in rows.columns]
    q_keep = list(dict.fromkeys(q_keep))
    one_q = rows.drop_duplicates("question_id")[q_keep]
    one_q.to_csv(DEST_QUESTIONS, index=False)
    print(f"wrote {DEST_QUESTIONS}  {one_q.shape}")

    n_prompt = len(_prompt_value_cols(prompt)) + len(
        [c for c in exam.columns if c.startswith("f_")]
    )
    DICT_PATH.write_text(
        "\n".join(
            [
                "# Phase 2 feature dictionary",
                "",
                "Open `pilot_results.csv`: one row = one question + one model.",
                "Columns start with the question text, gold letter, both answers,",
                "then model specs, interaction features, then every `f_*` measurement.",
                "`questions_with_features.csv` is the same 280 questions once each",
                "(features only, no per-model answers).",
                "Only information available before generation is used.",
                "",
                "Run grid: 280 questions × 14 models × **2** answers (`results_n2.csv`).",
                "`risk_target` = 1 - accuracy, with accuracy in {0, 0.5, 1}.",
                "The plan asked for incorrect/10; we still have 2, not 10.",
                "",
                "## Prompt features",
                "",
                "From question stem + lettered options, not the shared",
                "`multiple choice questions about {category}` wrapper.",
                "",
                f"- {n_prompt} numeric/flag columns (`f_*`), including exam-trap flags.",
                "- Token count: `f_context_token_count`.",
                "- Task type: `question_category` plus `f_question_type`.",
                "- Option count: `f_n_options`.",
                "- Negation / except: `f_contains_negation`, `f_is_except_ask`.",
                "- Constraints: `f_constraint_count`.",
                "- Few-shot: `f_has_few_shot_examples`.",
                "- Calculation / long hypo / best-answer: `f_is_formula_setup`,",
                "  `f_is_long_scenario`, `f_is_best_answer_judgment`.",
                "- Full one-line definitions: `FEATURES.md`.",
                "",
                "## Model / configuration features",
                "",
                "| field | meaning | missingness |",
                "| --- | --- | --- |",
                "| `llm_model` | model id used in the run | none |",
                "| `model_family` | id prefix | none |",
                "| `is_preview` | `preview` in the id | none |",
                "| `is_open_source` | Gemma yes, Gemini no | none |",
                "| `has_custom_tools` | id contains `customtools` | none |",
                "| `context_window_tokens` | published input limit | none on this list |",
                "| `knowledge_cutoff_year` | published cutoff year | null for Gemma and `*-latest` |",
                "| `max_tokens_requested` / `output_token_limit` | 1024 | none; no variance |",
                "| `temperature` | sampling temperature | **null** (not stored in the run) |",
                "| `top_p` | nucleus sampling | **null** (not stored in the run) |",
                "",
                "Citations: `experiments/model_specs.py`.",
                "",
                "## Interaction features",
                "",
                "- `f_context_pressure` = prompt tokens / context window.",
                "- `f_output_pressure` = 1024/1024 = 1 for every row",
                "  (expected output length was not logged).",
                "- `f_recency_gap` = year in prompt − model cutoff year.",
                "- `model_x_category` = task type × model id (string key).",
                "- `f_complexity_x_capability` = prompt tokens × (window / 1e6).",
                "  Window is the published limit, not a quality score.",
                "",
                f"Wrote {len(rows)} rows × {len(rows.columns)} columns.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {DEST_ROOT}  {rows.shape}")
    print(f"wrote {DEST_OUT}")
    print(f"wrote {DICT_PATH}")


if __name__ == "__main__":
    main()
