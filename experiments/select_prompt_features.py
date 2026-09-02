"""Question-out feature selection against question-level fail rate.

280 questions cannot carry 138 correlated features. Selection uses
q_fail_rate = mean(y_fail) across the 14 models so we are not rediscovering
which model was used.

The short list reported here is descriptive (stability across 5 question
folds). Phase 3 re-selects inside each training fold so test questions do
not pick the features used to score them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from evaluate import (
    OUT,
    Report,
    prepare_matrix,
    select_features_fold,
    univariate_spearman,
)

N_SPLITS = 5
MIN_FOLDS = 3
MAX_FEATURES = 15


def question_table(rows: pd.DataFrame) -> pd.DataFrame:
    q = (
        rows.groupby("question_id", as_index=False)
        .agg(q_fail_rate=("y_fail", "mean"), category=("question_category", "first"))
    )
    feat_cols = [
        c
        for c in rows.columns
        if c.startswith("f_")
        and not c.endswith(("__status", "__reason"))
        and c
        not in (
            "f_context_pressure",
            "f_recency_gap",
        )
    ]
    one = rows.drop_duplicates("question_id")[["question_id", *feat_cols]]
    return q.merge(one, on="question_id", how="left")


def main() -> None:
    rows = pd.read_csv(OUT / "mmlu_pilot_rows.csv", low_memory=False)
    q = question_table(rows)
    X, kinds, coverage = prepare_matrix(q)
    y = q["q_fail_rate"].to_numpy(dtype=float)
    # Stratify on a coarse difficulty bin so each fold has easy and hard items.
    bins = pd.qcut(y, q=4, duplicates="drop", labels=False).astype(int)

    rep = Report("mmlu_select")
    rep.say("=" * 78)
    rep.say("Prompt-feature selection on 280 questions (question-level fail rate)")
    rep.say("=" * 78)
    rep.say(f"questions {len(q)}   candidate columns {X.shape[1]}   "
            f"mean q_fail_rate {y.mean():.3f}")
    rep.say("n=1 Bernoulli per model; q_fail_rate averages the 14 models.")
    rep.say()

    full = univariate_spearman(X, y)
    full.to_csv(OUT / "mmlu_univariate_qfail.csv", index=False)
    sig = full[full.significant]
    rep.say(f"pooled (leaky, for inspection only): {len(sig)} of {len(full)} "
            f"features BH-significant vs q_fail_rate")
    rep.say(f"{'feature':<42} {'rho':>7} {'q':>10}")
    for _, r in full.head(15).iterrows():
        star = " *" if r.significant else ""
        rep.say(f"{r.feature:<42} {r.spearman:>7.3f} {r.q:>10.2e}{star}")
    rep.say()

    fold_lists: list[list[str]] = []
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for i, (tr, _te) in enumerate(cv.split(X, bins), start=1):
        chosen = select_features_fold(X.iloc[tr], y[tr], max_features=MAX_FEATURES)
        fold_lists.append(chosen)
        rep.say(f"fold {i} ({len(tr)} train questions): {', '.join(chosen)}")

    counts = Counter(f for lst in fold_lists for f in lst)
    stable = [f for f, n in counts.most_common() if n >= MIN_FOLDS][:MAX_FEATURES]
    flicker = [f for f, n in counts.most_common() if 0 < n < MIN_FOLDS]

    rep.say()
    rep.say(f"stable (in >= {MIN_FOLDS} of {N_SPLITS} folds), cap {MAX_FEATURES}:")
    for f in stable:
        rho = float(full.set_index("feature").loc[f, "spearman"]) if f in set(full.feature) else float("nan")
        rep.say(f"  {f:<42} folds={counts[f]}  pooled_rho={rho:+.3f}")
    rep.say()
    if flicker:
        rep.say("flicker (appeared in 1-2 folds):")
        for f in flicker:
            rep.say(f"  {f:<42} folds={counts[f]}")

    payload = {
        "n_questions": int(len(q)),
        "n_candidates": int(X.shape[1]),
        "min_folds": MIN_FOLDS,
        "max_features": MAX_FEATURES,
        "stable": stable,
        "fold_counts": dict(counts),
        "fold_lists": fold_lists,
        "note": (
            "Stability list is descriptive. Phase 3 re-selects inside each "
            "training fold so held-out questions do not pick the feature set."
        ),
    }
    (OUT / "mmlu_selected_features.json").write_text(json.dumps(payload, indent=2))
    (OUT / "mmlu_select_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")
    print(f"\nwrote {OUT / 'mmlu_selected_features.json'}")
    print(f"wrote {OUT / 'mmlu_select_report.txt'}")


if __name__ == "__main__":
    main()
