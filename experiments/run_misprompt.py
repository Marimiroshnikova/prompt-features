"""Mis-prompt: do prompt-only features detect that the prompt itself is flawed?

The label here is a property of the prompt, which is exactly what promptfeat
claims to measure. But the two classes come from different corpora: the flawed
prompts are generated per flaw subtype (ids 1-14696) and the clean prompts are
trivia questions (ids 20001-34696). So the naive number measures "which corpus"
as much as "is flawed", and three controls are applied:

  naive                - random grouping, as a paper would report it
  unseen flaw subtype  - no flaw subtype appears in both train and test, so the
                         model cannot memorise a generator template
  length matched       - flawed and clean prompts resampled to the same word-count
                         distribution, killing the strongest corpus cue
  both                 - unseen subtype and length matched together
"""

from __future__ import annotations

import json

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

rep = Report("misprompt")


def length_matched_index(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """1:1 match flawed and clean prompts on exact word count."""
    rng = np.random.default_rng(seed)
    words = df["f_question_length_words"].fillna(-1).astype(int)
    keep = []
    for w, block in df.groupby(words).groups.items():
        block = np.asarray(block)
        pos = block[df.loc[block, "y"].to_numpy() == 1]
        neg = block[df.loc[block, "y"].to_numpy() == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    return np.sort(np.concatenate(keep))


def subtype_groups(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Flawed rows grouped by flaw subtype; clean rows scattered into own groups.

    This prevents a flaw subtype from appearing in both train and test, which is
    the leak that makes a generated dataset look easy.
    """
    rng = np.random.default_rng(seed)
    g = np.empty(len(df), dtype=object)
    sub = df["secondary_category"].astype(str).to_numpy()
    y = df["y"].to_numpy()
    g[y == 1] = ["flaw:" + s for s in sub[y == 1]]
    n_clean = int((y == 0).sum())
    g[y == 0] = ["clean:%d" % i for i in rng.integers(0, 40, n_clean)]
    return pd.factorize(g)[0]


def main() -> None:
    df = pd.read_csv(OUT / "misprompt_features.csv", low_memory=False)
    X, kinds, coverage = prepare_matrix(df)
    y = df["y"].to_numpy()
    Xlen = X[[c for c in LENGTH_FEATURES if c in X]]

    rep.say("=" * 78)
    rep.say("Mis-prompt  -  29,392 prompts, half carrying a planted flaw")
    rep.say("=" * 78)
    rep.say(f"rows {len(df)}   model columns {X.shape[1]}   flaw subtypes "
            f"{df.secondary_category.nunique()}")
    rep.say(f"flawed id range {df.loc[y==1,'prompt_id'].min()}-"
            f"{df.loc[y==1,'prompt_id'].max()}   clean id range "
            f"{df.loc[y==0,'prompt_id'].min()}-{df.loc[y==0,'prompt_id'].max()}")
    wp = df.loc[y == 1, "f_question_length_words"]
    wn = df.loc[y == 0, "f_question_length_words"]
    rep.say(f"median words: flawed {wp.median():.0f}, clean {wn.median():.0f}  "
            f"<- the corpus cue the controls have to remove")
    rep.say()

    rows_all = np.arange(len(df))
    lm = length_matched_index(df)
    rep.say(f"length-matched subsample: {len(lm)} rows "
            f"({y[lm].mean():.3f} flawed), median words "
            f"{df.loc[lm, 'f_question_length_words'].median():.0f} in both classes")
    rep.say()

    naive_groups = np.arange(len(df))
    sub_groups = subtype_groups(df)

    settings = {
        "naive (random grouping)": (rows_all, naive_groups),
        "unseen flaw subtype": (rows_all, sub_groups),
        "length matched": (lm, naive_groups[lm]),
        "length matched + unseen subtype": (lm, sub_groups[lm]),
    }

    results = {}
    rep.say(f"{'setting':<34} {'prompt feats':>13} {'length only':>13} "
            f"{'null p95':>10}")
    for label, (rows, groups) in settings.items():
        Xs = X.iloc[rows].reset_index(drop=True)
        Xl = Xlen.iloc[rows].reset_index(drop=True)
        ys = y[rows]
        gm, gs, _ = cv_auc(Xs, ys, groups, kind="gbm", n_repeats=2)
        lm_auc, _, _ = cv_auc(Xl, ys, groups, kind="gbm", n_repeats=2)
        null = permutation_null(Xs, ys, groups, kind="logreg", n_perm=10)["p95"]
        rep.say(f"{label:<34} {gm:>13.3f} {lm_auc:>13.3f} {null:>10.3f}")
        results[label] = {"prompt_features": gm, "prompt_features_sd": gs,
                          "length_only": lm_auc, "null_p95": null,
                          "n": int(len(rows))}

    rep.say()
    rep.say("-" * 78)
    rep.say("Univariate AUC on the length-matched subsample (corpus length cue")
    rep.say("removed). These are the features that actually see the flaw.")
    rep.say("-" * 78)
    uni = univariate_screen(
        X.iloc[lm].reset_index(drop=True), y[lm], naive_groups[lm], bootstrap=False
    )
    uni.to_csv(OUT / "misprompt_univariate_lengthmatched.csv", index=False)
    sig = uni[uni.significant]
    rep.say(f"{len(sig)} of {len(uni)} testable features significant at BH q<0.05")
    rep.say(f"{'feature':<44} {'AUC':>7} {'q':>11}")
    for _, r in uni.head(15).iterrows():
        rep.say(f"{r.feature:<44} {r.auc:>7.3f} {r.q:>11.2e}")

    (OUT / "misprompt_results.json").write_text(json.dumps(results, indent=2))
    (OUT / "misprompt_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")


if __name__ == "__main__":
    main()
