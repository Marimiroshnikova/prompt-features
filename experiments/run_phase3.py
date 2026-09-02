"""Phase 3 baselines on the 280 x 14 MMLU-Pro grid.

All means and feature selections are fit on training questions only.
A question_id never appears on both sides of a fold.

Label is y_fail from one generation (n=1 Bernoulli, not incorrect/10).
Primary metric: Brier. Log loss and AUROC are secondary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from evaluate import (
    LENGTH_FEATURES,
    OUT,
    Report,
    impute_train_test,
    make_ridge_logistic,
    metrics_bundle,
    prepare_matrix,
    select_features_fold,
)

N_SPLITS = 5
N_REPEATS = 3
SHRINK_K = 8.0
MAX_FEATURES = 15


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def mean_lookup(keys: pd.Series, values: np.ndarray, default: float) -> dict:
    s = pd.Series(values, index=keys.index).groupby(keys).mean()
    return {k: float(v) for k, v in s.items()} | {"__default__": float(default)}


def apply_lookup(keys: pd.Series, table: dict, default: float | None = None) -> np.ndarray:
    fill = table["__default__"] if default is None else default
    return keys.map(table).fillna(fill).to_numpy(dtype=float)


def model_category_shrunk(
    train: pd.DataFrame,
    ytr: np.ndarray,
    test: pd.DataFrame,
    model_means: dict,
    global_mean: float,
    k: float = SHRINK_K,
) -> np.ndarray:
    """Cell mean shrunk toward the model mean. Raw 20-question cells overfit."""
    tmp = train.assign(_y=ytr)
    stats = (
        tmp.groupby(["llm_model", "question_category"])["_y"]
        .agg(mean="mean", n="size")
        .reset_index()
    )
    preds = []
    cell = {
        (r.llm_model, r.question_category): (float(r.mean), int(r.n))
        for r in stats.itertuples(index=False)
    }
    for rec in test.itertuples(index=False):
        model_m = model_means.get(rec.llm_model, global_mean)
        pair = cell.get((rec.llm_model, rec.question_category))
        if pair is None:
            preds.append(model_m)
            continue
        m, n = pair
        w = n / (n + k)
        preds.append(w * m + (1.0 - w) * model_m)
    return np.asarray(preds, dtype=float)


def fit_predict_logistic(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    xtr, xte = impute_train_test(Xtr.to_numpy(dtype=float), Xte.to_numpy(dtype=float))
    est = make_ridge_logistic()
    est.fit(xtr, ytr)
    return est.predict_proba(xte)[:, 1]


def model_dummies(train_models: pd.Series, test_models: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = sorted(train_models.unique())
    # drop last level to avoid dummy trap with intercept
    keep = levels[:-1] if len(levels) > 1 else levels
    tr = pd.DataFrame({f"model={m}": (train_models == m).astype(float) for m in keep})
    te = pd.DataFrame({f"model={m}": (test_models == m).astype(float) for m in keep})
    return tr, te


def main() -> None:
    rows = pd.read_csv(OUT / "mmlu_pilot_rows.csv", low_memory=False)
    y = rows["y_fail"].to_numpy(dtype=float)
    groups = rows["question_id"].to_numpy()
    X_all, _, _ = prepare_matrix(rows)
    # Prompt-only matrix: drop model-dependent interaction columns.
    prompt_cols = [c for c in X_all.columns if c not in ("f_context_pressure", "f_recency_gap")]
    X_prompt = X_all[prompt_cols]
    length_cols = [c for c in LENGTH_FEATURES if c in X_prompt.columns]
    if "f_n_options" in X_prompt.columns and "f_n_options" not in length_cols:
        pass  # length-only stays length; n_options is a prompt feature for the short set

    # Question-level table for nested selection (one row per question).
    q_fail = rows.groupby("question_id")["y_fail"].mean()
    one = rows.drop_duplicates("question_id")
    X_q = X_prompt.loc[one.index].copy()
    X_q.index = one["question_id"].to_numpy()

    methods = [
        "global",
        "model",
        "category",
        "model_x_category",
        "length_only",
        "prompt_short",
        "prompt_short_plus_model",
    ]
    fold_metrics: dict[str, list[dict]] = {m: [] for m in methods}
    fold_features: list[list[str]] = []

    rep = Report("mmlu_phase3")
    rep.say("=" * 78)
    rep.say("Phase 3 baselines  —  280 questions × 14 models")
    rep.say("=" * 78)
    n_trials = int(rows["n_trials"].iloc[0]) if "n_trials" in rows.columns else 1
    rep.say(
        f"Label: y_fail = 1 - accuracy with n_trials={n_trials} "
        f"(plan wanted incorrect/10)."
    )
    rep.say("Splits: StratifiedGroupKFold on question_id; a question never crosses.")
    rep.say("Feature selection for prompt_short is nested inside each training fold.")
    rep.say(f"rows {len(rows)}   questions {rows.question_id.nunique()}   "
            f"fail rate {y.mean():.4f}")
    rep.say()

    for rep_i in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=rep_i
        )
        for fold, (tr, te) in enumerate(cv.split(X_prompt, y, groups), start=1):
            train, test = rows.iloc[tr], rows.iloc[te]
            ytr, yte = y[tr], y[te]
            g_mean = float(ytr.mean())

            pred = {"global": np.full(len(te), g_mean)}
            model_means = mean_lookup(train["llm_model"], ytr, g_mean)
            pred["model"] = apply_lookup(test["llm_model"], model_means)
            cat_means = mean_lookup(train["question_category"], ytr, g_mean)
            pred["category"] = apply_lookup(test["question_category"], cat_means)
            pred["model_x_category"] = model_category_shrunk(
                train, ytr, test, model_means, g_mean
            )

            pred["length_only"] = fit_predict_logistic(
                X_prompt.iloc[tr][length_cols], ytr, X_prompt.iloc[te][length_cols]
            )

            train_qids = pd.unique(train["question_id"])
            chosen = select_features_fold(
                X_q.loc[train_qids],
                q_fail.loc[train_qids].to_numpy(dtype=float),
                max_features=MAX_FEATURES,
            )
            if not chosen:
                chosen = length_cols[:1]
            if rep_i == 0:
                fold_features.append(chosen)
            pred["prompt_short"] = fit_predict_logistic(
                X_prompt.iloc[tr][chosen], ytr, X_prompt.iloc[te][chosen]
            )

            mtr, mte = model_dummies(train["llm_model"], test["llm_model"])
            plus_tr = pd.concat(
                [X_prompt.iloc[tr][chosen].reset_index(drop=True), mtr.reset_index(drop=True)],
                axis=1,
            )
            plus_te = pd.concat(
                [X_prompt.iloc[te][chosen].reset_index(drop=True), mte.reset_index(drop=True)],
                axis=1,
            )
            pred["prompt_short_plus_model"] = fit_predict_logistic(plus_tr, ytr, plus_te)

            for name, p in pred.items():
                fold_metrics[name].append(metrics_bundle(yte, _clip(p)))

    def summarise(name: str) -> dict:
        dfm = pd.DataFrame(fold_metrics[name])
        return {
            "brier": [float(dfm.brier.mean()), float(dfm.brier.std())],
            "log_loss": [float(dfm.log_loss.mean()), float(dfm.log_loss.std())],
            "auc": [float(dfm.auc.mean()), float(dfm.auc.std())],
            "n_folds": int(len(dfm)),
        }

    summary = {name: summarise(name) for name in methods}

    rep.say(f"{'method':<28} {'Brier':>14} {'log loss':>14} {'AUROC':>14}")
    for name in methods:
        s = summary[name]
        rep.say(
            f"{name:<28} {s['brier'][0]:>7.4f}+-{s['brier'][1]:<5.3f} "
            f"{s['log_loss'][0]:>7.4f}+-{s['log_loss'][1]:<5.3f} "
            f"{s['auc'][0]:>7.3f}+-{s['auc'][1]:<5.3f}"
        )

    def beats(a: str, b: str) -> str:
        return "YES" if summary[a]["brier"][0] < summary[b]["brier"][0] else "NO"

    rep.say()
    rep.say("Does prompt_short beat the Model baseline on Brier? "
            f"{beats('prompt_short', 'model')}")
    rep.say("Does prompt_short beat Model × category on Brier? "
            f"{beats('prompt_short', 'model_x_category')}")
    rep.say("Does prompt_short + model beat Model on Brier? "
            f"{beats('prompt_short_plus_model', 'model')}")
    rep.say("Does prompt_short + model beat Model × category on Brier? "
            f"{beats('prompt_short_plus_model', 'model_x_category')}")
    rep.say()
    rep.say("Lower Brier is better. If prompt_short does not beat Model,")
    rep.say("prompt features are not adding risk signal on this grid.")
    rep.say()

    if fold_features:
        from collections import Counter
        c = Counter(f for lst in fold_features for f in lst)
        rep.say("Features chosen inside repeat-0 training folds "
                f"(nested, {len(fold_features)} folds):")
        for feat, n in c.most_common():
            rep.say(f"  {feat:<42} {n}/{len(fold_features)}")

    payload = {
        "n_rows": int(len(rows)),
        "n_questions": int(rows.question_id.nunique()),
        "fail_rate": float(y.mean()),
        "n_trials": n_trials,
        "label": f"y_fail = 1-accuracy over {n_trials} trials; not a 10-sample probability",
        "metrics": summary,
        "nested_features_repeat0": fold_features,
        "comparisons": {
            "prompt_short_beats_model": beats("prompt_short", "model") == "YES",
            "prompt_short_beats_model_x_category": beats("prompt_short", "model_x_category") == "YES",
            "prompt_short_plus_model_beats_model": beats("prompt_short_plus_model", "model") == "YES",
            "prompt_short_plus_model_beats_model_x_category": (
                beats("prompt_short_plus_model", "model_x_category") == "YES"
            ),
        },
    }
    (OUT / "mmlu_phase3_results.json").write_text(json.dumps(payload, indent=2))
    (OUT / "mmlu_phase3_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")
    print(f"\nwrote {OUT / 'mmlu_phase3_report.txt'}")


if __name__ == "__main__":
    main()
