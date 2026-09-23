"""H1: binary high-risk label vs soft fail-rate target.

Question-grouped 5-fold CV on the 840 cells (280 MMLU-Pro questions x 3 Gemini
models x 10 trials). Every model sees the same prompt features and model specs.
question_id is not a feature. MMLU subject is not a feature.

Two model families, as Phase 4 asks:
  XGBoost    binary classifier on fail_rate > 0.5 vs regressor on fail_rate
  elastic-net logistic   on the binary label vs on the 10 trials themselves
                         (each cell expanded to fail / correct rows, weighted
                         by count), which is the soft target for a logistic

Scores are read as P(this answer is wrong). Brier and log loss are trial-weighted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.h_metrics import (  # noqa: E402
    N_BOOT, N_FOLDS, OUT, SEED, brier_gap_ci, load_cells, make_x, pack,
    question_bootstrap, question_folds, risk80_gap_ci, risk_at_coverage, trial_brier,
)

XGB_KW = dict(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=5,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    enable_categorical=True,
)


HEADS = ["xgb binary", "xgb soft", "logit binary", "logit soft"]
PREDICTOR = {
    "xgb binary": "binary classifier",
    "xgb soft": "soft regressor",
    "logit binary": "binary logistic",
    "logit soft": "soft logistic",
}


def elastic_net(X):
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    lr = LogisticRegression(solver="saga", l1_ratio=0.5, C=0.05, tol=1e-3,
                            max_iter=3000, random_state=SEED)
    return make_pipeline(prep, lr)


def fit_heads(X, df, tr, te):
    """Out-of-fold P(wrong) for every head in HEADS."""
    y_rate = df["fail_rate"].to_numpy(float)
    y_bin = (y_rate > 0.5).astype(int)
    Xtr = X.loc[tr].astype({c: object for c in X.columns
                            if isinstance(X[c].dtype, pd.CategoricalDtype)})
    Xte = X.loc[te].astype({c: object for c in X.columns
                            if isinstance(X[c].dtype, pd.CategoricalDtype)})
    out = {}

    clf = XGBClassifier(**XGB_KW, objective="binary:logistic", eval_metric="logloss")
    clf.fit(X.loc[tr], y_bin[tr], verbose=False)
    out["xgb binary"] = clf.predict_proba(X.loc[te])[:, 1]

    reg = XGBRegressor(**XGB_KW, objective="reg:squarederror", eval_metric="rmse")
    reg.fit(X.loc[tr], y_rate[tr], verbose=False)
    out["xgb soft"] = np.clip(reg.predict(X.loc[te]), 0.0, 1.0)

    lb = elastic_net(X)
    lb.fit(Xtr, y_bin[tr])
    out["logit binary"] = lb.predict_proba(Xte)[:, 1]

    nf = df["n_fail"].to_numpy(float)[tr]
    nc = df["n_correct"].to_numpy(float)[tr]
    Xe = pd.concat([Xtr, Xtr], ignore_index=True)
    ye = np.r_[np.ones(len(nf)), np.zeros(len(nc))]
    # divided by trials so each cell weighs 1, as in the binary head; same penalty strength
    we = np.r_[nf, nc] / np.r_[nf + nc, nf + nc]
    keep = we > 0
    ls = elastic_net(X)
    ls.fit(Xe[keep], ye[keep], logisticregression__sample_weight=we[keep])
    out["logit soft"] = ls.predict_proba(Xte)[:, 1]
    return out


def main():
    df = load_cells()
    nf = df["n_fail"].to_numpy(float)
    nc = df["n_correct"].to_numpy(float)
    X, constant, cat = make_x(df)

    pred = {name: np.zeros(len(df))
            for name in ["base rate", *PREDICTOR.values()]}
    for fold, (tr, te) in enumerate(question_folds(df), start=1):
        assert set(df.question_id[tr]).isdisjoint(df.question_id[te])
        prior = float(nf[tr].sum() / (nf[tr] + nc[tr]).sum())
        pred["base rate"][te] = prior
        for head, p in fit_heads(X, df, tr, te).items():
            pred[PREDICTOR[head]][te] = p
        print(f"fold {fold} train questions {df.question_id[tr].nunique()} "
              f"test {df.question_id[te].nunique()} prior {prior:.3f}")

    base_b = trial_brier(nf, nc, pred["base rate"])
    results = [pack(n, df, p, base_b) for n, p in pred.items()]
    for r in results:
        print(r["model"], {k: round(r[k], 4) for k in
                           ("brier", "bss_vs_oof_base", "aurc", "ece_15_quantile")})

    def risk80_drop(p):
        return lambda i: (risk_at_coverage(nf[i], nc[i], p[i], 0.80)
                          - nf[i].sum() / (nf[i] + nc[i]).sum())

    boot = {
        m: {
            "brier_minus_base": brier_gap_ci(df, pred[m], pred["base rate"]),
            "risk80_minus_base_rate": question_bootstrap(df, risk80_drop(pred[m])),
        }
        for m in PREDICTOR.values()
    }
    gap = {
        "brier": brier_gap_ci(df, pred["soft regressor"], pred["binary classifier"]),
        "risk80": risk80_gap_ci(df, pred["soft regressor"], pred["binary classifier"]),
    }
    gap_logit = {
        "brier": brier_gap_ci(df, pred["soft logistic"], pred["binary logistic"]),
        "risk80": risk80_gap_ci(df, pred["soft logistic"], pred["binary logistic"]),
    }

    by_model, by_subject = [], []
    for col, store, key in (("llm_model", by_model, "llm_model"),
                            ("question_category", by_subject, "subject")):
        for level in sorted(df[col].unique()):
            mask = (df[col] == level).to_numpy()
            for name, p in pred.items():
                store.append({
                    key: level,
                    "predictor": name,
                    "brier": trial_brier(nf[mask], nc[mask], p[mask]),
                    "risk_at_coverage_80": risk_at_coverage(nf[mask], nc[mask], p[mask], 0.80),
                    "fail_rate": float(nf[mask].sum() / (nf[mask] + nc[mask]).sum()),
                    "n_questions": int(df.loc[mask, "question_id"].nunique()),
                })

    payload = {
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "n_models": int(df["llm_model"].nunique()),
        "fail_rate": float(nf.sum() / (nf + nc).sum()),
        "binary_positive_rate": float((df["fail_rate"] > 0.5).mean()),
        "share_cells_all_wrong": float((df["fail_rate"] == 1.0).mean()),
        "share_cells_between": float(((df["fail_rate"] > 0) & (df["fail_rate"] < 1)).mean()),
        "binary_rule": "1 if fail_rate > 0.5 else 0",
        "n_features": int(X.shape[1]),
        "dropped_constant": constant,
        "categoricals": cat,
        "folds": N_FOLDS,
        "results": results,
        "bootstrap": {"n": N_BOOT, "unit": "question_id",
                      "delta_vs_oof_base": boot, "regressor_minus_classifier": gap,
                      "soft_logistic_minus_binary_logistic": gap_logit},
        "by_model": by_model,
        "by_subject": by_subject,
        "note": ("Test label is the same 10 trials. Split-half reliability of this "
                 "label is 0.975, so a fresh 20-30 sample was not collected. "
                 "No sealed holdout: 280 questions, no feature selection."),
    }
    (OUT / "h1_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT / "h1_metrics.json")


if __name__ == "__main__":
    main()
