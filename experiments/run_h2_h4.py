"""H2, reduced H3, and H4 on the 840 ten-trial cells.

H2  prompt features only vs all features (prompt + model columns).
H3  reduced stand-in: prompt -> prompt + model id -> prompt + model id + cutoff
    and recency gap. Context window, temperature, top-p, and output pressure do
    not vary on this grid, so the planned H3 cannot run.
H4  every learned arm vs the global, model, subject, and model x subject rates.

Splits:
  question-out  5 question-grouped folds, same as H1 (primary)
  subject-out   leave one MMLU subject out (14 folds)
  model-out     leave one model out, crossed with the 5 question folds (15 folds):
                train on the other two models' training questions, test the
                held-out model on unseen questions. Holding the model out alone
                lets the tree memorise each question from the other two models.
                Only the prompt-only arm can score an unseen model, and
                model x subject falls back to the subject rate.

Every arm is scored with the locked Phase 6 metrics (experiments/h_metrics.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.h_metrics import (  # noqa: E402
    OUT, baseline_predictions, brier_gap_ci, group_folds, load_cells, make_x, pack,
    question_folds, risk80_gap_ci, trial_brier,
)
from experiments.run_h1 import HEADS, fit_heads  # noqa: E402

ARMS = {
    "prompt only": dict(drop_model=True),
    "prompt + model id": dict(keep_model=["llm_model"]),
    "all features": dict(),
}
BASELINES = ["global mean", "model mean", "subject mean", "model x subject"]


def model_question_folds(df):
    """Held-out model x held-out questions; every cell is tested exactly once."""
    folds = []
    for _, not_model, is_model in group_folds(df, "llm_model"):
        for q_tr, q_te in question_folds(df):
            folds.append((not_model & q_tr, is_model & q_te))
    return folds


def run_split(df, folds, arms):
    """Out-of-fold predictions for the baselines and every (arm, head)."""
    matrices = {a: make_x(df, **ARMS[a])[0] for a in arms}
    pred = {n: np.full(len(df), np.nan) for n in BASELINES}
    for a in arms:
        for head in HEADS:
            pred[f"{a} / {head}"] = np.full(len(df), np.nan)
    for tr, te in folds:
        for name, p in baseline_predictions(df[tr], df[te]).items():
            pred[name][te] = p
        for a in arms:
            for head, p in fit_heads(matrices[a], df, tr, te).items():
                pred[f"{a} / {head}"][te] = p
    assert all(not np.isnan(p).any() for p in pred.values())
    return pred


def score_split(df, pred):
    nf = df["n_fail"].to_numpy(float)
    nc = df["n_correct"].to_numpy(float)
    base_b = trial_brier(nf, nc, pred["global mean"])
    rows = [pack(n, df, p, base_b) for n, p in pred.items()]
    learned = [r for r in rows if r["model"] not in BASELINES]
    best = min(learned, key=lambda r: r["brier"])["model"]
    return rows, best


def main():
    df = load_cells()
    report = {"n_rows": int(len(df)), "n_questions": int(df.question_id.nunique()),
              "arms": {a: int(make_x(df, **ARMS[a])[0].shape[1]) for a in ARMS},
              "splits": {}}

    splits = {
        "question-out": (list(question_folds(df)), list(ARMS)),
        "subject-out": ([(tr, te) for _, tr, te in group_folds(df, "question_category")],
                        list(ARMS)),
        "model-out": (model_question_folds(df), ["prompt only"]),
    }
    for split, (folds, arms) in splits.items():
        print(f"== {split}: {len(folds)} folds, arms {arms}")
        pred = run_split(df, folds, arms)
        rows, best = score_split(df, pred)
        for r in rows:
            print(f"  {r['model']:<30} Brier {r['brier']:.4f}  BSS {r['bss_vs_oof_base']:+.3f}  "
                  f"err@80 {r['risk_at_coverage_80']:.3f}  AUROC {r['auroc']:.3f}")
        entry = {
            "n_folds": len(folds),
            "arms": arms,
            "results": rows,
            "best_learned": best,
            "best_vs_model_x_subject": {
                "brier": brier_gap_ci(df, pred[best], pred["model x subject"]),
                "risk80": risk80_gap_ci(df, pred[best], pred["model x subject"]),
            },
        }
        if split != "model-out":
            entry["h2_prompt_minus_all"] = {
                head: {
                    "brier": brier_gap_ci(df, pred[f"prompt only / {head}"],
                                          pred[f"all features / {head}"]),
                    "risk80": risk80_gap_ci(df, pred[f"prompt only / {head}"],
                                            pred[f"all features / {head}"]),
                }
                for head in HEADS
            }
        report["splits"][split] = entry
        print(f"  best learned: {best}")

    report["note"] = (
        "Best learned arm is the lowest out-of-fold Brier among learned arms in that "
        "split. Gaps are question-bootstrap 95% intervals; negative means the first "
        "predictor is better. Subject-out: model x subject falls back to the model "
        "rate. Model-out: model mean falls back to the training prior and model x "
        "subject to the subject rate. Holding out only the model (no question split) "
        "gave prompt-only Brier 0.130, because the other two models' answers to the "
        "same question were in training; that number is leakage and is not reported."
    )
    (OUT / "h2_h4_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", OUT / "h2_h4_metrics.json")


if __name__ == "__main__":
    main()
