"""Do exam-trap flags add anything after model and subject are known?"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from evaluate import (
    OUT,
    Report,
    impute_train_test,
    make_ridge_logistic,
    metrics_bundle,
    prepare_matrix,
)
from run_phase3 import (
    N_REPEATS,
    N_SPLITS,
    apply_lookup,
    mean_lookup,
    model_category_shrunk,
    model_dummies,
)

EXAM_KEEP = [
    "f_is_except_ask",
    "f_is_best_answer_judgment",
    "f_is_hypothetical_scenario",
    "f_is_definition_ask",
    "f_is_formula_setup",
    "f_stem_word_count",
    "f_is_long_scenario",
    "f_mc_option_count",
    "f_option_mean_chars",
    "f_option_length_spread",
    "f_has_escape_option",
]


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def fit_log(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame):
    xtr, xte = impute_train_test(Xtr.to_numpy(dtype=float), Xte.to_numpy(dtype=float))
    est = make_ridge_logistic()
    est.fit(xtr, ytr)
    return est.predict_proba(xte)[:, 1]


def cat_dummies(train_cats: pd.Series, test_cats: pd.Series):
    levels = sorted(train_cats.unique())
    keep = levels[:-1] if len(levels) > 1 else levels
    tr = pd.DataFrame({f"cat={c}": (train_cats == c).astype(float) for c in keep})
    te = pd.DataFrame({f"cat={c}": (test_cats == c).astype(float) for c in keep})
    return tr, te


def main() -> None:
    rows = pd.read_csv(OUT / "mmlu_pilot_rows.csv", low_memory=False)
    exam = pd.read_csv(OUT / "mmlu_exam_features.csv")
    keep = ["question_id", *[c for c in EXAM_KEEP if c in exam.columns]]
    rows = rows.merge(exam[keep], on="question_id", how="left")

    y = rows["y_fail"].to_numpy(dtype=float)
    groups = rows["question_id"].to_numpy()
    present = [c for c in EXAM_KEEP if c in rows.columns]
    X_exam, _, _ = prepare_matrix(rows[present])
    exam_cols = list(X_exam.columns)

    methods = [
        "global",
        "model",
        "category",
        "model_x_category",
        "exam_only",
        "category_plus_exam",
        "model_x_category_plus_exam",
    ]
    bags = {m: [] for m in methods}

    rep = Report("exam_plus")
    rep.say("Exam flags on top of model and subject")
    rep.say("Question-grouped CV. n=1 Bernoulli. Lower Brier is better.")
    rep.say(f"exam columns: {', '.join(exam_cols)}")
    rep.say()

    for seed in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, te in cv.split(X_exam, y, groups):
            train, test = rows.iloc[tr], rows.iloc[te]
            ytr, yte = y[tr], y[te]
            g = float(ytr.mean())
            model_means = mean_lookup(train["llm_model"], ytr, g)
            pred_mx = model_category_shrunk(train, ytr, test, model_means, g)

            pred = {
                "global": np.full(len(te), g),
                "model": apply_lookup(test["llm_model"], model_means),
                "category": apply_lookup(
                    test["question_category"],
                    mean_lookup(train["question_category"], ytr, g),
                ),
                "model_x_category": pred_mx,
                "exam_only": fit_log(X_exam.iloc[tr], ytr, X_exam.iloc[te]),
            }

            ctr, cte = cat_dummies(train["question_category"], test["question_category"])
            plus_cat_tr = pd.concat(
                [X_exam.iloc[tr].reset_index(drop=True), ctr.reset_index(drop=True)], axis=1
            )
            plus_cat_te = pd.concat(
                [X_exam.iloc[te].reset_index(drop=True), cte.reset_index(drop=True)], axis=1
            )
            pred["category_plus_exam"] = fit_log(plus_cat_tr, ytr, plus_cat_te)

            mx_tr = model_category_shrunk(train, ytr, train, model_means, g)
            stacked_tr = X_exam.iloc[tr].reset_index(drop=True).copy()
            stacked_te = X_exam.iloc[te].reset_index(drop=True).copy()
            stacked_tr["mx"] = mx_tr
            stacked_te["mx"] = pred_mx
            pred["model_x_category_plus_exam"] = fit_log(stacked_tr, ytr, stacked_te)

            for name, p in pred.items():
                bags[name].append(metrics_bundle(yte, _clip(p)))

    def summ(name):
        df = pd.DataFrame(bags[name])
        return {
            "brier": [float(df.brier.mean()), float(df.brier.std())],
            "auc": [float(df.auc.mean()), float(df.auc.std())],
        }

    summary = {m: summ(m) for m in methods}
    rep.say(f"{'method':<32} {'Brier':>14} {'AUROC':>14}")
    for m in methods:
        s = summary[m]
        rep.say(
            f"{m:<32} {s['brier'][0]:>7.4f}+-{s['brier'][1]:<5.3f} "
            f"{s['auc'][0]:>7.3f}+-{s['auc'][1]:<5.3f}"
        )

    def beats(a, b):
        return summary[a]["brier"][0] < summary[b]["brier"][0]

    rep.say()
    rep.say(
        "exam_only beats category? "
        + ("YES" if beats("exam_only", "category") else "NO")
    )
    rep.say(
        "category+exam beats category? "
        + ("YES" if beats("category_plus_exam", "category") else "NO")
    )
    rep.say(
        "model x category + exam beats model x category? "
        + ("YES" if beats("model_x_category_plus_exam", "model_x_category") else "NO")
    )

    (OUT / "mmlu_exam_plus_results.json").write_text(json.dumps(summary, indent=2))
    (OUT / "mmlu_exam_plus_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")


if __name__ == "__main__":
    main()
