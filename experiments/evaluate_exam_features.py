"""Measure the new exam-trap features on the 280-question grid."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from promptfeat import extract_features  # noqa: E402
from promptfeat.registry import REGISTRY  # noqa: E402

from evaluate import OUT, prepare_matrix, univariate_spearman  # noqa: E402

EXAM = [n for n, f in REGISTRY.items() if f.group == "exam"]


def main() -> None:
    rows = pd.read_csv(OUT / "mmlu_pilot_rows.csv", low_memory=False)
    prompts = pd.read_csv(OUT / "mmlu_prompt_features.csv", low_memory=False)
    q_fail = rows.groupby("question_id")["y_fail"].mean()

    recs = []
    for rec in prompts.itertuples(index=False):
        feats = extract_features(rec.text)
        recs.append({"question_id": rec.question_id, "category": rec.category, **{
            "f_" + k: feats.get(k) for k in EXAM
        }})
    exam = pd.DataFrame(recs)
    exam["q_fail"] = exam["question_id"].map(q_fail)
    exam.to_csv(OUT / "mmlu_exam_features.csv", index=False)

    print("NEW EXAM FEATURES vs question-level fail rate")
    print(f"{'feature':<32} {'rho':>7} {'fail if true':>14} {'fail if false':>14} {'n_true':>7}")
    X, _, _ = prepare_matrix(exam)
    screen = univariate_spearman(X, exam["q_fail"].to_numpy())
    for _, r in screen.iterrows():
        col = r.feature
        s = exam[col] if col in exam.columns else None
        if s is not None and set(map(str, s.dropna().unique())) <= {"True", "False", "True", "False"}:
            t = exam.loc[s.astype(str) == "True", "q_fail"]
            f = exam.loc[s.astype(str) == "False", "q_fail"]
            print(f"{col:<32} {r.spearman:>7.3f} {t.mean():>14.3f} {f.mean():>14.3f} {len(t):>7}")
        else:
            print(f"{col:<32} {r.spearman:>7.3f}")

    screen.to_csv(OUT / "mmlu_exam_univariate.csv", index=False)
    print("\nwrote", OUT / "mmlu_exam_features.csv")


if __name__ == "__main__":
    main()
