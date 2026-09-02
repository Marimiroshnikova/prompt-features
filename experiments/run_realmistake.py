"""ReaLMistake: do prompt-only features predict that the response was wrong?

Two targets:
  error      - the responder model's answer was judged wrong by an expert. This is
               partly a property of the responder, so prompt-only features can
               only ever explain part of it.
  difficult  - ReaLMistake's own per-prompt difficulty annotation. Verified to be
               identical across responder models for every one of the 480 prompts,
               so it is a purely prompt-intrinsic target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (
    LENGTH_FEATURES,
    OUT,
    Report,
    cv_auc,
    permutation_null,
    prepare_matrix,
    univariate_screen,
)

rep = Report("realmistake")


def main() -> None:
    df = pd.read_csv(OUT / "realmistake_features.csv", low_memory=False)
    X, kinds, coverage = prepare_matrix(df)
    groups = pd.factorize(df["prompt"])[0]

    rep.say("=" * 78)
    rep.say("ReaLMistake  -  900 expert-labeled responses over 480 unique prompts")
    rep.say("=" * 78)
    rep.say(f"rows {len(df)}   unique prompts {df.prompt.nunique()}   "
            f"model columns {X.shape[1]} (from 138 features)")
    rep.say(f"error rate {df.y.mean():.3f}   "
            f"difficult rate {(df.difficulty == 'difficult').mean():.3f}")
    rep.say(f"features with <50% coverage on this data: "
            f"{int((coverage < 0.5).sum())} of {len(coverage)}")
    rep.say()

    is_llama = (df["model"] != "gpt-4-0613").astype(float).to_frame("responder_model")

    targets = {
        "error": df["y"].to_numpy(),
        "difficult": (df["difficulty"] == "difficult").astype(int).to_numpy(),
    }

    results: dict = {}

    for tname, y in targets.items():
        rep.say("-" * 78)
        rep.say(f"TARGET: {tname}   (positive rate {y.mean():.3f})")
        rep.say("-" * 78)

        sets = {
            "length only (2 features)": X[[c for c in LENGTH_FEATURES if c in X]],
            "all prompt features": X,
        }
        if tname == "error":
            sets["responder model only (1 feature)"] = is_llama
            sets["all prompt features + responder model"] = pd.concat(
                [X, is_llama], axis=1
            )

        rep.say(f"{'feature set':<44} {'logreg AUC':>16} {'gbm AUC':>16}")
        table = {}
        for label, mat in sets.items():
            lm, ls, _ = cv_auc(mat, y, groups, kind="logreg")
            gm, gs, _ = cv_auc(mat, y, groups, kind="gbm", n_repeats=3)
            table[label] = {"logreg": [lm, ls], "gbm": [gm, gs]}
            rep.say(f"{label:<44} {lm:>8.3f} +-{ls:<5.3f} {gm:>8.3f} +-{gs:<5.3f}")

        null = permutation_null(X, y, groups, kind="logreg", n_perm=30)
        rep.say()
        rep.say(f"permutation null for 'all prompt features' (labels shuffled "
                f"between prompts, {null['n_perm']} runs):")
        rep.say(f"    mean {null['mean']:.3f}   95th pct {null['p95']:.3f}   "
                f"max {null['max']:.3f}")
        real = table["all prompt features"]["logreg"][0]
        verdict = "ABOVE" if real > null["p95"] else "NOT above"
        rep.say(f"    observed {real:.3f} is {verdict} the 95th percentile of the null")
        rep.say()

        uni = univariate_screen(X, y, groups, coverage=coverage)
        uni.to_csv(OUT / f"realmistake_univariate_{tname}.csv", index=False)
        sig = uni[uni.significant]
        rep.say(f"univariate: {len(sig)} of {len(uni)} testable features significant "
                f"at BH q<0.05")
        rep.say(f"{'feature':<40} {'AUC':>6} {'95% CI (cluster bootstrap)':>28} {'q':>9}")
        for _, r in uni.head(12).iterrows():
            ci = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
            star = " *" if r.significant else "  "
            rep.say(f"{r.feature:<40} {r.auc:>6.3f} {ci:>28} {r.q:>9.2e}{star}")
        rep.say()

        results[tname] = {"cv": table, "permutation_null": null,
                          "n_significant": int(len(sig)),
                          "n_tested": int(len(uni))}

    (OUT / "realmistake_results.json").write_text(json.dumps(results, indent=2))
    (OUT / "realmistake_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")


if __name__ == "__main__":
    main()
