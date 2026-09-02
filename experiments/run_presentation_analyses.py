"""Extra analyses for the Phase 2–3 talk.

Writes JSON + a text report. Does not invent 10-sample risk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from evaluate import (
    OUT,
    Report,
    brier,
    impute_train_test,
    log_loss_safe,
    make_ridge_logistic,
    metrics_bundle,
    prepare_matrix,
    select_features_fold,
)

SHRINK_K = 8.0
MAX_FEATURES = 15


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def shrunk_cell_pred(train, ytr, test, k=SHRINK_K):
    g = float(ytr.mean())
    model_m = pd.Series(ytr, index=train.index).groupby(train["llm_model"]).mean()
    tmp = train.assign(_y=ytr)
    stats = tmp.groupby(["llm_model", "question_category"])["_y"].agg(["mean", "size"])
    preds = []
    for rec in test.itertuples(index=False):
        mm = float(model_m.get(rec.llm_model, g))
        key = (rec.llm_model, rec.question_category)
        if key not in stats.index:
            preds.append(mm)
            continue
        m, n = float(stats.loc[key, "mean"]), int(stats.loc[key, "size"])
        w = n / (n + k)
        preds.append(w * m + (1.0 - w) * mm)
    return np.asarray(preds, dtype=float)


def model_pred(train, ytr, test):
    g = float(ytr.mean())
    mm = pd.Series(ytr, index=train.index).groupby(train["llm_model"]).mean()
    return test["llm_model"].map(mm).fillna(g).to_numpy(dtype=float)


def category_pred(train, ytr, test):
    g = float(ytr.mean())
    cm = pd.Series(ytr, index=train.index).groupby(train["question_category"]).mean()
    return test["question_category"].map(cm).fillna(g).to_numpy(dtype=float)


def coverage_risk(y, p, fracs=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Abstain on the highest-risk fraction; report remaining error rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    y_sorted = y[order]
    n = len(y)
    out = [{"coverage": 1.0, "error": float(y.mean()), "n_answered": int(n)}]
    for f in fracs:
        k = int(round(n * (1.0 - f)))
        if k <= 0:
            continue
        kept = y_sorted[-k:]  # lowest predicted risk after dropping top-f
        # After sorting by descending risk, the safest k are at the end.
        out.append(
            {
                "coverage": round(k / n, 4),
                "abstain": f,
                "error": float(kept.mean()),
                "n_answered": k,
            }
        )
    return out


def main() -> None:
    rows = pd.read_csv(OUT / "mmlu_pilot_rows.csv", low_memory=False)
    y = rows["y_fail"].to_numpy(dtype=float)
    groups = rows["question_id"].to_numpy()
    X_all, _, _ = prepare_matrix(rows)
    prompt_cols = [c for c in X_all.columns if c not in ("f_context_pressure", "f_recency_gap")]
    X_prompt = X_all[prompt_cols]
    q_fail = rows.groupby("question_id")["y_fail"].mean()
    one = rows.drop_duplicates("question_id")
    X_q = X_prompt.loc[one.index].copy()
    X_q.index = one["question_id"].to_numpy()

    rep = Report("mmlu_talk")
    payload: dict = {}

    # --- descriptive ---
    by_model = (
        rows.groupby("llm_model")
        .agg(n=("y_fail", "size"), fail=("y_fail", "mean"), acc=("y_fail", lambda s: 1 - s.mean()))
        .sort_values("fail", ascending=False)
    )
    by_cat = (
        rows.groupby("question_category")
        .agg(n=("y_fail", "size"), fail=("y_fail", "mean"))
        .sort_values("fail", ascending=False)
    )
    qdist = q_fail.value_counts().sort_index()
    n_never = int((q_fail == 0).sum())
    n_always = int((q_fail == 1).sum())
    n_split = int(((q_fail > 0) & (q_fail < 1)).sum())

    payload["descriptive"] = {
        "n_rows": int(len(rows)),
        "n_questions": int(rows.question_id.nunique()),
        "n_models": int(rows.llm_model.nunique()),
        "fail_rate": float(y.mean()),
        "by_model": by_model.reset_index().to_dict(orient="records"),
        "by_category": by_cat.reset_index().to_dict(orient="records"),
        "q_fail_never": n_never,
        "q_fail_always": n_always,
        "q_fail_split": n_split,
        "q_fail_mean": float(q_fail.mean()),
        "q_fail_std": float(q_fail.std()),
    }
    rep.say("DESCRIPTIVE")
    rep.say(f"fail rate {y.mean():.4f}  questions never missed {n_never}  "
            f"always missed {n_always}  mixed {n_split}")
    rep.say("models (fail rate):")
    for r in by_model.itertuples():
        rep.say(f"  {r.Index:<40} {r.fail:.3f}  acc={r.acc:.3f}")
    rep.say("categories (fail rate):")
    for r in by_cat.itertuples():
        rep.say(f"  {r.Index:<20} {r.fail:.3f}")

    # --- residuals after model x category (fit on all data for description only) ---
    mx = shrunk_cell_pred(rows, y, rows)
    resid = y - mx
    # correlate prompt features with residual at question level
    q_resid = pd.Series(resid, index=rows.index).groupby(rows["question_id"]).mean()
    from evaluate import univariate_spearman

    resid_screen = univariate_spearman(X_q, q_resid.loc[X_q.index].to_numpy())
    payload["residual"] = {
        "brier_in_sample_mx": brier(y, mx),
        "top": resid_screen.head(8).to_dict(orient="records") if not resid_screen.empty else [],
    }
    rep.say()
    rep.say("RESIDUAL after in-sample model×category (descriptive, leaky):")
    rep.say(f"  in-sample Brier {brier(y, mx):.4f}")
    if not resid_screen.empty:
        for _, r in resid_screen.head(8).iterrows():
            rep.say(f"  {r.feature:<42} rho={r.spearman:+.3f} q={r.q:.2e}")

    # --- question-out (already have phase3) load it ---
    p3 = json.loads((OUT / "mmlu_phase3_results.json").read_text())
    payload["question_out"] = p3["metrics"]

    # --- category-out ---
    cats = rows["question_category"].to_numpy()
    uniq_c = np.unique(cats)
    cat_out = {"model": [], "prompt_short": [], "model_x_category": []}
    for held in uniq_c:
        te = np.flatnonzero(cats == held)
        tr = np.flatnonzero(cats != held)
        ytr, yte = y[tr], y[te]
        train, test = rows.iloc[tr], rows.iloc[te]
        pred_m = model_pred(train, ytr, test)
        pred_mx = shrunk_cell_pred(train, ytr, test)  # unseen cat → falls back to model
        train_qids = pd.unique(train["question_id"])
        chosen = select_features_fold(
            X_q.loc[train_qids], q_fail.loc[train_qids].to_numpy(), max_features=MAX_FEATURES
        )
        if not chosen:
            chosen = [c for c in ("f_question_length_words",) if c in X_prompt.columns]
        xtr, xte = impute_train_test(
            X_prompt.iloc[tr][chosen].to_numpy(float),
            X_prompt.iloc[te][chosen].to_numpy(float),
        )
        est = make_ridge_logistic()
        est.fit(xtr, ytr)
        pred_p = est.predict_proba(xte)[:, 1]
        cat_out["model"].append(metrics_bundle(yte, _clip(pred_m)))
        cat_out["model_x_category"].append(metrics_bundle(yte, _clip(pred_mx)))
        cat_out["prompt_short"].append(metrics_bundle(yte, _clip(pred_p)))

    def avg(lst):
        df = pd.DataFrame(lst)
        return {c: [float(df[c].mean()), float(df[c].std())] for c in df.columns}

    payload["category_out"] = {k: avg(v) for k, v in cat_out.items()}
    rep.say()
    rep.say("CATEGORY-OUT (train 13 subjects, test the held-out subject)")
    for k, s in payload["category_out"].items():
        rep.say(f"  {k:<22} Brier {s['brier'][0]:.4f}+-{s['brier'][1]:.3f}  "
                f"AUC {s['auc'][0]:.3f}")

    # --- model-out ---
    mods = rows["llm_model"].to_numpy()
    uniq_m = np.unique(mods)
    model_out = {"category": [], "prompt_short": [], "global": []}
    for held in uniq_m:
        te = np.flatnonzero(mods == held)
        tr = np.flatnonzero(mods != held)
        ytr, yte = y[tr], y[te]
        train, test = rows.iloc[tr], rows.iloc[te]
        g = float(ytr.mean())
        pred_g = np.full(len(te), g)
        pred_c = category_pred(train, ytr, test)
        train_qids = pd.unique(train["question_id"])
        chosen = select_features_fold(
            X_q.loc[train_qids], q_fail.loc[train_qids].to_numpy(), max_features=MAX_FEATURES
        )
        if not chosen:
            chosen = [c for c in ("f_question_length_words",) if c in X_prompt.columns]
        xtr, xte = impute_train_test(
            X_prompt.iloc[tr][chosen].to_numpy(float),
            X_prompt.iloc[te][chosen].to_numpy(float),
        )
        est = make_ridge_logistic()
        est.fit(xtr, ytr)
        pred_p = est.predict_proba(xte)[:, 1]
        model_out["global"].append(metrics_bundle(yte, _clip(pred_g)))
        model_out["category"].append(metrics_bundle(yte, _clip(pred_c)))
        model_out["prompt_short"].append(metrics_bundle(yte, _clip(pred_p)))

    payload["model_out"] = {k: avg(v) for k, v in model_out.items()}
    rep.say()
    rep.say("MODEL-OUT (train 13 models, test the held-out model)")
    for k, s in payload["model_out"].items():
        rep.say(f"  {k:<22} Brier {s['brier'][0]:.4f}+-{s['brier'][1]:.3f}  "
                f"AUC {s['auc'][0]:.3f}")

    # --- coverage-risk on question-out OOF model×category ---
    oof_mx = np.zeros(len(rows))
    oof_prompt = np.zeros(len(rows))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(X_prompt, y, groups):
        train, test = rows.iloc[tr], rows.iloc[te]
        ytr = y[tr]
        oof_mx[te] = shrunk_cell_pred(train, ytr, test)
        train_qids = pd.unique(train["question_id"])
        chosen = select_features_fold(
            X_q.loc[train_qids], q_fail.loc[train_qids].to_numpy(), max_features=MAX_FEATURES
        )
        if not chosen:
            chosen = [c for c in ("f_question_length_words",) if c in X_prompt.columns]
        xtr, xte = impute_train_test(
            X_prompt.iloc[tr][chosen].to_numpy(float),
            X_prompt.iloc[te][chosen].to_numpy(float),
        )
        est = make_ridge_logistic()
        est.fit(xtr, ytr)
        oof_prompt[te] = est.predict_proba(xte)[:, 1]

    payload["coverage_risk"] = {
        "model_x_category": coverage_risk(y, oof_mx),
        "prompt_short": coverage_risk(y, oof_prompt),
        "base_error": float(y.mean()),
    }
    rep.say()
    rep.say("COVERAGE-RISK (question-out OOF)")
    for name, curve in payload["coverage_risk"].items():
        if name == "base_error":
            continue
        rep.say(f"  {name}:")
        for pt in curve:
            if "abstain" in pt:
                rep.say(f"    abstain {pt['abstain']:.0%} -> remaining error {pt['error']:.3f} "
                        f"(answered {pt['n_answered']})")
            else:
                rep.say(f"    answer all -> error {pt['error']:.3f}")

    (OUT / "mmlu_talk_results.json").write_text(json.dumps(payload, indent=2))
    (OUT / "mmlu_talk_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")
    print("wrote", OUT / "mmlu_talk_report.txt")


if __name__ == "__main__":
    main()
