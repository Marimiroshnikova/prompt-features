"""Is the Mis-prompt result flaw detection, or corpus discrimination?

Mis-prompt's clean prompts are trivia questions from a different source than its
generated flawed prompts, so a classifier can score well without ever seeing a
flaw. Three probes separate the two explanations:

  1. style only     - a handful of surface-provenance features (entity density,
                      question category, casing). If these alone reach the full
                      model's score, the task is corpus discrimination.
  2. flaw only      - the features that are actually about prompt defects
                      (ambiguity, underspecification, contradiction, vagueness).
  3. style matched  - resample so both classes share the same word-count, entity-
                      count and question-category cells. Whatever survives here
                      is not explained by provenance style.

Also reports the corrected permutation nulls and per-flaw-subtype detectability.
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
    make_estimator,
    permutation_null,
    prepare_matrix,
)

rep = Report("misprompt_probe")

STYLE_KEYS = (
    "entity", "anchor", "proper_noun", "acronym", "numeral", "uppercase",
    "tokens_per_word", "question_category", "question_type", "named_entity_hint",
    "quoted_span", "id_like", "year", "currency", "unit", "percent",
)
FLAW_KEYS = (
    "ambigu", "underspecif", "vague", "hedge", "dangling", "unresolved",
    "missing_subject", "contradict", "negation", "exclusion", "conditional",
    "constraint", "multi_part", "sub_question",
)


def pick(X: pd.DataFrame, keys) -> list[str]:
    return [c for c in X.columns if any(k in c for k in keys)]


def matched_index(df: pd.DataFrame, cols_bins, seed=0) -> np.ndarray:
    """Exact 1:1 matching on a set of binned covariates."""
    rng = np.random.default_rng(seed)
    key = pd.Series([""] * len(df), index=df.index)
    for col, binner in cols_bins:
        key = key + "|" + binner(df[col]).astype(str)
    keep = []
    y = df["y"].to_numpy()
    for _, block in df.groupby(key).groups.items():
        block = np.asarray(block)
        pos = block[y[block] == 1]
        neg = block[y[block] == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def main() -> None:
    df = pd.read_csv(OUT / "misprompt_features.csv", low_memory=False)
    X, kinds, coverage = prepare_matrix(df)
    y = df["y"].to_numpy()

    style = pick(X, STYLE_KEYS)
    flaw = pick(X, FLAW_KEYS)
    rep.say("=" * 78)
    rep.say("Mis-prompt probe: flaw detection or corpus discrimination?")
    rep.say("=" * 78)
    rep.say(f"style-provenance subset: {len(style)} columns")
    rep.say(f"flaw-relevant subset:    {len(flaw)} columns")
    rep.say()

    rows = np.arange(len(df))
    naive_g = np.arange(len(df))

    rep.say("-" * 78)
    rep.say("Probe 1 and 2: which kind of feature carries the score?")
    rep.say("-" * 78)
    subsets = {
        "all 138 features": X,
        "style / provenance only": X[style],
        "flaw-relevant only": X[flaw],
        "length only": X[[c for c in LENGTH_FEATURES if c in X]],
    }
    probe = {}
    rep.say(f"{'feature subset':<30} {'AUC (gbm, random split)':>25}")
    for label, mat in subsets.items():
        m, s, _ = cv_auc(mat, y, naive_g, kind="gbm", n_repeats=2)
        rep.say(f"{label:<30} {m:>17.3f} +-{s:<5.3f}")
        probe[label] = m

    rep.say()
    rep.say("-" * 78)
    rep.say("Probe 3: match the two classes on provenance style, then retry")
    rep.say("-" * 78)

    def qbin(s, q=8):
        return pd.qcut(s.fillna(-1), q, duplicates="drop")

    schemes = {
        "word count matched": [
            ("f_question_length_words", lambda s: s.fillna(-1).astype(int))
        ],
        "word count + entity count matched": [
            ("f_question_length_words", lambda s: s.fillna(-1).astype(int)),
            ("f_entity_count", lambda s: s.fillna(-1).astype(int)),
        ],
        "word + entity + question category matched": [
            ("f_question_length_words", lambda s: s.fillna(-1).astype(int)),
            ("f_entity_count", lambda s: s.fillna(-1).astype(int)),
            ("f_question_category", lambda s: s.astype(str)),
        ],
    }
    matched = {}
    rep.say(f"{'matching':<44} {'n':>7} {'all feats':>11} {'flaw only':>11}")
    for label, spec in schemes.items():
        idx = matched_index(df, spec)
        if len(idx) < 400:
            rep.say(f"{label:<44} {len(idx):>7} too few rows after matching")
            continue
        Xs = X.iloc[idx].reset_index(drop=True)
        ys = y[idx]
        a, _, _ = cv_auc(Xs, ys, np.arange(len(idx)), kind="gbm", n_repeats=2)
        f, _, _ = cv_auc(Xs[flaw].reset_index(drop=True), ys,
                         np.arange(len(idx)), kind="gbm", n_repeats=2)
        rep.say(f"{label:<44} {len(idx):>7} {a:>11.3f} {f:>11.3f}")
        matched[label] = {"n": int(len(idx)), "all": a, "flaw_only": f}

    rep.say()
    rep.say("-" * 78)
    rep.say("Corrected permutation nulls (the earlier group-mode null was wrong")
    rep.say("for unequal group sizes)")
    rep.say("-" * 78)
    sub = df["secondary_category"].astype(str)
    g_sub = pd.factorize(np.where(y == 1, "flaw:" + sub, "clean:" + (
        np.arange(len(df)) % 40).astype(str)))[0]
    for mode in ("rows", "groups"):
        n = permutation_null(X, y, g_sub, kind="logreg", n_perm=10, mode=mode)
        rep.say(f"  subtype-grouped CV, {mode:<7} null: mean {n['mean']:.3f}  "
                f"p95 {n['p95']:.3f}  max {n['max']:.3f}  (n={n['n_perm']})")

    rep.say()
    rep.say("-" * 78)
    rep.say("Per-flaw-subtype detectability: train without the subtype, test on it")
    rep.say("(clean prompts split randomly; AUC over that subtype vs held-out clean)")
    rep.say("-" * 78)
    rng = np.random.default_rng(3)
    clean_idx = np.flatnonzero(y == 0)
    holdout_clean = rng.choice(clean_idx, size=3000, replace=False)
    train_clean = np.setdiff1d(clean_idx, holdout_clean)
    per_sub = {}
    Xv = X.to_numpy(dtype=float)
    med_all = np.nanmedian(Xv, axis=0)
    med_all = np.where(np.isnan(med_all), 0.0, med_all)
    Xf = np.where(np.isnan(Xv), med_all, Xv)
    for s in sorted(sub[y == 1].unique()):
        test_pos = np.flatnonzero((y == 1) & (sub == s).to_numpy())
        train_pos = np.flatnonzero((y == 1) & (sub != s).to_numpy())
        tr = np.concatenate([train_pos, train_clean])
        te = np.concatenate([test_pos, holdout_clean])
        est = make_estimator("gbm")
        est.fit(Xf[tr], y[tr])
        a = roc_auc_score(y[te], est.predict_proba(Xf[te])[:, 1])
        per_sub[s] = {"auc": float(a), "n": int(len(test_pos))}
        rep.say(f"  {s:<44} n={len(test_pos):<5} AUC {a:.3f}")

    out = {"probe_subsets": probe, "matched": matched, "per_subtype": per_sub}
    (OUT / "misprompt_probe_results.json").write_text(json.dumps(out, indent=2))
    (OUT / "misprompt_probe_report.txt").write_text(
        "\n".join(rep.lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
