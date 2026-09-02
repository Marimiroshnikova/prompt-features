"""ReaLMistake, with the confounds progressively removed.

The pooled result is contaminated two ways:
  * three task templates with very different surface shape,
  * an easy/difficult prompt-template variant that some features decode exactly.

So the error target is re-evaluated inside ever tighter strata:
  level 1  pooled                      (task and variant both free to leak)
  level 2  within task                 (task fixed)
  level 3  within task x variant       (task and variant both fixed - strictest)

If prompt features still beat length-only and the permutation null at level 3,
the signal is about the individual prompt, not about which template it came from.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import (
    LENGTH_FEATURES,
    OUT,
    Report,
    cv_auc,
    permutation_null,
    prepare_matrix,
    univariate_screen,
)

rep = Report("realmistake_strict")


def stratum_cv(X, y, groups, strata, kind="logreg", n_repeats=5):
    """CV within each stratum, pooled by sample-size-weighted mean AUC."""
    aucs, weights = [], []
    for s in np.unique(strata):
        m = strata == s
        if len(np.unique(y[m])) < 2 or m.sum() < 60:
            continue
        Xs = X[m] if isinstance(X, pd.DataFrame) else X[m]
        mean, _, scores = cv_auc(
            Xs.reset_index(drop=True), y[m], groups[m], kind=kind, n_repeats=n_repeats
        )
        aucs.append(mean)
        weights.append(m.sum())
    if not aucs:
        return np.nan, np.nan
    w = np.array(weights, dtype=float)
    return float(np.average(aucs, weights=w)), float(np.std(aucs))


def main() -> None:
    df = pd.read_csv(OUT / "realmistake_features.csv", low_memory=False).copy()
    X, kinds, coverage = prepare_matrix(df)
    y = df["y"].to_numpy()
    groups = pd.factorize(df["prompt"])[0]
    task = df["task"].to_numpy()
    variant = (df["task"] + "|" + df["difficulty"]).to_numpy()

    Xlen = X[[c for c in LENGTH_FEATURES if c in X]]

    rep.say("=" * 78)
    rep.say("ReaLMistake, error target, confounds removed step by step")
    rep.say("=" * 78)
    rep.say()
    rep.say("Why: ReaLMistake ships an easy/difficult prompt-template variant that")
    rep.say("some features decode exactly. Inside two of the three tasks,")
    rep.say("f_instruction_line_count separates easy from difficult at AUC 1.000, so")
    rep.say("the pooled numbers partly measure template recognition.")
    rep.say()
    rep.say("stratum counts:")
    for v in sorted(set(variant)):
        m = variant == v
        rep.say(f"  {v:<48} n={m.sum():<5} error rate {y[m].mean():.3f}")
    rep.say()

    levels = {
        "level 1  pooled": np.zeros(len(df), dtype=int),
        "level 2  within task": pd.factorize(task)[0],
        "level 3  within task x easy/difficult variant": pd.factorize(variant)[0],
    }

    out = {}
    rep.say(f"{'':<46} {'prompt feats':>14} {'length only':>14} {'null p95':>10}")
    for label, strata in levels.items():
        if label.startswith("level 1"):
            full, _, _ = cv_auc(X, y, groups, kind="logreg")
            lo, _, _ = cv_auc(Xlen, y, groups, kind="logreg")
            null = permutation_null(X, y, groups, n_perm=30)["p95"]
        else:
            full, _ = stratum_cv(X, y, groups, strata)
            lo, _ = stratum_cv(Xlen, y, groups, strata)
            # null within strata: shuffle labels inside each stratum
            rng = np.random.default_rng(7)
            nulls = []
            for _ in range(15):
                yp = y.copy()
                for s in np.unique(strata):
                    m = strata == s
                    yp[m] = rng.permutation(y[m])
                v, _ = stratum_cv(X, yp, groups, strata, n_repeats=1)
                nulls.append(v)
            null = float(np.nanpercentile(nulls, 95))
        rep.say(f"{label:<46} {full:>14.3f} {lo:>14.3f} {null:>10.3f}")
        out[label] = {"prompt_features": full, "length_only": lo, "null_p95": null}

    rep.say()
    rep.say("-" * 78)
    rep.say("Per-task detail (error target, prompt features only, grouped CV)")
    rep.say("-" * 78)
    per_task = {}
    for t in sorted(set(task)):
        m = task == t
        full, _, _ = cv_auc(X[m].reset_index(drop=True), y[m], groups[m], kind="logreg")
        lo, _, _ = cv_auc(Xlen[m].reset_index(drop=True), y[m], groups[m], kind="logreg")
        rep.say(f"  {t:<34} n={m.sum():<5} prompt feats {full:.3f}   length only {lo:.3f}")
        per_task[t] = {"prompt_features": full, "length_only": lo, "n": int(m.sum())}
    out["per_task"] = per_task

    rep.say()
    rep.say("-" * 78)
    rep.say("Univariate AUC for the error target, computed inside task x variant")
    rep.say("strata and pooled by weighted mean (so template recognition cannot")
    rep.say("contribute). Features that stay away from 0.500 here are real.")
    rep.say("-" * 78)

    strata = pd.factorize(variant)[0]
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        vals, wts = [], []
        for s in np.unique(strata):
            m = (strata == s) & ~np.isnan(x)
            if m.sum() < 60 or len(np.unique(y[m])) < 2 or len(np.unique(x[m])) < 2:
                continue
            vals.append(roc_auc_score(y[m], x[m]))
            wts.append(m.sum())
        if len(vals) < 2:
            continue
        pooled = float(np.average(vals, weights=np.array(wts, float)))
        rows.append({
            "feature": col,
            "within_stratum_auc": pooled,
            "abs": abs(pooled - 0.5),
            "n_strata": len(vals),
        })
    within = pd.DataFrame(rows).sort_values("abs", ascending=False)
    within.to_csv(OUT / "realmistake_within_stratum_univariate.csv", index=False)
    rep.say(f"{'feature':<42} {'within-stratum AUC':>19} {'strata':>7}")
    for _, r in within.head(15).iterrows():
        rep.say(f"{r.feature:<42} {r.within_stratum_auc:>19.3f} {int(r.n_strata):>7}")

    (OUT / "realmistake_strict_results.json").write_text(json.dumps(out, indent=2))
    (OUT / "realmistake_strict_report.txt").write_text(
        "\n".join(rep.lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
