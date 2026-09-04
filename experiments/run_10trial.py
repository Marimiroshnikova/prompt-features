"""Phase 1 redone with 10 answers per cell.

The earlier selection ran on 1 Bernoulli draw per (question, model). "Risk" was
a coin flip, so nothing could be calibrated and every text feature lost to
model x subject. This run uses 280 questions x 3 models x 10 trials = 8,400
answers, so each cell carries a real fail probability out of 10.

Two labels, per the plan:
  q_fail_rate  mean fail rate across the 3 models, n = 30 trials per question
  cell         (question, model) fail rate out of 10, n = 840 cells

Guards kept from evaluate.py: question-grouped splits, BH over 138+ tests,
train-fold-only feature selection, permutation null, and every model quoted
next to the model x subject baseline it has to beat.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    OUT,
    Report,
    prepare_matrix,
    select_features_fold,
    univariate_spearman,
    value_columns,
)

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "reduced_10trial.csv"
N_TRIALS = 10
N_SPLITS = 5
MIN_FOLDS = 3
MAX_FEATURES = 15
SMOOTH_K = 10.0          # pseudo-trials pulling a cell mean toward its parent
SEED = 0



# --------------------------------------------------------------------------- #
# faster redundancy pruning (same rule as evaluate.drop_correlated, vectorised)
# --------------------------------------------------------------------------- #

def _fast_select(X: pd.DataFrame, y: np.ndarray, max_features: int = MAX_FEATURES,
                 corr_threshold: float = 0.9, min_coverage: float = 0.5) -> list[str]:
    screen = univariate_spearman(X, y, min_coverage=min_coverage)
    if screen.empty:
        return []
    ranked = [c for c in screen["feature"].tolist() if c in X.columns]
    sub = X[ranked]
    filled = sub.apply(lambda s: s.fillna(s.median()))
    R = filled.rank().to_numpy(dtype=float)
    R = R - R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = 1.0
    R = R / sd
    C = np.abs(R.T @ R) / len(R)
    kept: list[int] = []
    for i in range(len(ranked)):
        if all(C[i, j] < corr_threshold for j in kept):
            kept.append(i)
        if len(kept) >= max_features:
            break
    return [ranked[i] for i in kept]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    res = pd.read_csv(RESULTS, low_memory=False)
    feat = pd.read_csv(OUT / "mmlu_prompt_features.csv", low_memory=False)
    exam = pd.read_csv(OUT / "mmlu_exam_features.csv", low_memory=False)
    exam = exam.drop(columns=[c for c in ("q_fail", "category") if c in exam.columns])

    res["n_fail"] = N_TRIALS - res["n_correct"]
    cells = res[[
        "question_id", "question_category", "llm_model",
        "n_correct", "n_fail", "n_blank", "fail_rate",
    ]].copy()

    fcols = [c for c in feat.columns if c.startswith("f_")]
    ecols = [c for c in exam.columns if c.startswith("f_")]
    qfeat = feat[["question_id", *fcols]].merge(
        exam[["question_id", *ecols]], on="question_id", how="left"
    )

    q = (
        cells.groupby("question_id", as_index=False)
        .agg(
            q_fail_rate=("fail_rate", "mean"),
            n_fail_total=("n_fail", "sum"),
            n_total=("n_correct", lambda s: 0),
            category=("question_category", "first"),
        )
    )
    q["n_total"] = cells.groupby("question_id")["n_correct"].sum().to_numpy() + q["n_fail_total"]
    q = q.merge(qfeat, on="question_id", how="left")
    return cells, q


# --------------------------------------------------------------------------- #
# weighted metrics on cells
# --------------------------------------------------------------------------- #

def weighted_brier(cells: pd.DataFrame, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    nf = cells["n_fail"].to_numpy(dtype=float)
    nc = cells["n_correct"].to_numpy(dtype=float)
    num = (nf * (p - 1.0) ** 2 + nc * p ** 2).sum()
    return float(num / (nf + nc).sum())


def weighted_logloss(cells: pd.DataFrame, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    nf = cells["n_fail"].to_numpy(dtype=float)
    nc = cells["n_correct"].to_numpy(dtype=float)
    num = -(nf * np.log(p) + nc * np.log(1.0 - p)).sum()
    return float(num / (nf + nc).sum())


def cell_auc(cells: pd.DataFrame, p: np.ndarray) -> float:
    """Trial-level AUC, computed from the binomial counts without expanding rows."""
    nf = cells["n_fail"].to_numpy(dtype=float)
    nc = cells["n_correct"].to_numpy(dtype=float)
    order = np.argsort(p)
    pos = nf[order]
    neg = nc[order]
    # rank-based: for each score, count negatives strictly below + half of ties
    cum_neg = np.concatenate([[0.0], np.cumsum(neg)])[:-1]
    s = np.asarray(p, dtype=float)[order]
    conc = 0.0
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        tie_pos = pos[i:j + 1].sum()
        tie_neg = neg[i:j + 1].sum()
        below = cum_neg[i]
        conc += tie_pos * (below + 0.5 * tie_neg)
        i = j + 1
    denom = nf.sum() * nc.sum()
    return float(conc / denom) if denom else float("nan")


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

def smoothed_rate(df: pd.DataFrame, keys: list[str], prior: float, k: float) -> pd.Series:
    g = df.groupby(keys)[["n_fail", "n_correct"]].sum()
    n = g["n_fail"] + g["n_correct"]
    return (g["n_fail"] + k * prior) / (n + k)


def predict_group(test: pd.DataFrame, table: pd.Series, keys: list[str], fallback: float) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else pd.Index(test[keys[0]])
    return table.reindex(idx).fillna(fallback).to_numpy(dtype=float)


def expand(cells: pd.DataFrame, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row becomes two weighted rows: fail (y=1) and correct (y=0)."""
    Xd = np.vstack([X, X])
    y = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w = np.concatenate([
        cells["n_fail"].to_numpy(dtype=float),
        cells["n_correct"].to_numpy(dtype=float),
    ])
    keep = w > 0
    return Xd[keep], y[keep], w[keep]


def design(cells: pd.DataFrame, qfeat: pd.DataFrame, cols: list[str],
           models: list[str], cats: list[str], use_ctx: bool) -> np.ndarray:
    blocks = []
    if use_ctx:
        blocks.append(np.column_stack([(cells["llm_model"] == m).to_numpy(float) for m in models]))
        blocks.append(np.column_stack([(cells["question_category"] == c).to_numpy(float) for c in cats]))
        inter = np.column_stack([
            ((cells["llm_model"] == m) & (cells["question_category"] == c)).to_numpy(float)
            for m in models for c in cats
        ])
        blocks.append(inter)
    if cols:
        f = cells[["question_id"]].merge(qfeat[["question_id", *cols]], on="question_id", how="left")
        blocks.append(f[cols].to_numpy(dtype=float))
    return np.column_stack(blocks) if blocks else np.zeros((len(cells), 1))


def fit_predict_logistic(Xtr, ytr, wtr, Xte, C: float = 0.3) -> np.ndarray:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=6000, C=C))
    est.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    return est.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    cells, q = load()
    rep = Report("mmlu_10trial")
    say = rep.say

    n_answers = int((cells["n_fail"] + cells["n_correct"]).sum())
    models = sorted(cells["llm_model"].unique())
    cats = sorted(cells["question_category"].unique())

    say("=" * 78)
    say("MMLU-Pro miss prediction, 10 trials per cell")
    say("=" * 78)
    say(f"questions {q.shape[0]}   models {len(models)}   cells {len(cells)}   "
        f"answers {n_answers}")
    say(f"unparsed replies {int(cells['n_blank'].sum())} "
        f"({cells['n_blank'].sum() / n_answers:.2%}) — counted as wrong")
    say(f"overall fail rate {cells['n_fail'].sum() / n_answers:.4f}")
    say()

    say("fail rate by model")
    for m in models:
        s = cells[cells.llm_model == m]
        say(f"  {m:<26} {s['n_fail'].sum() / (s['n_fail'].sum() + s['n_correct'].sum()):.3f}")
    say()
    say("fail rate by subject")
    bysub = cells.groupby("question_category").apply(
        lambda s: s["n_fail"].sum() / (s["n_fail"].sum() + s["n_correct"].sum()),
        include_groups=False,
    ).sort_values(ascending=False)
    for c, v in bysub.items():
        say(f"  {c:<20} {v:.3f}")
    say()

    # ---- what 10 trials bought us -------------------------------------- #
    fr = cells["fail_rate"].to_numpy()
    say("-" * 78)
    say("What the 10th trial bought")
    say("-" * 78)
    say(f"cells always right (0/10)   {np.mean(fr == 0.0):.3f}")
    say(f"cells always wrong (10/10)  {np.mean(fr == 1.0):.3f}")
    say(f"cells in between            {np.mean((fr > 0) & (fr < 1)):.3f}  "
        f"<- invisible at n=1")
    say(f"variance of the label: n=1 would give {0.205 * 0.795:.4f} per draw; "
        f"observed across cells {fr.var():.4f}")
    say()
    qf = q["q_fail_rate"].to_numpy()
    say(f"questions no model ever missed   {int((qf == 0).sum())} / {len(qf)}")
    say(f"questions every model always missed {int((qf == 1).sum())} / {len(qf)}")
    say(f"questions with intermediate risk  {int(((qf > 0) & (qf < 1)).sum())} / {len(qf)}")
    say()

    # ---- univariate ------------------------------------------------------ #
    X, kinds, coverage = prepare_matrix(q)
    say("-" * 78)
    say("Univariate screen against q_fail_rate (Spearman, BH q<0.05)")
    say("-" * 78)
    say(f"candidate columns {X.shape[1]}   mean q_fail_rate {qf.mean():.3f}   "
        f"sd {qf.std():.3f}")
    full = univariate_spearman(X, qf)
    full.to_csv(OUT / "tentrial_univariate_qfail.csv", index=False)
    sig = full[full.significant]
    say(f"{len(sig)} of {len(full)} features survive BH at q<0.05")
    say()
    say(f"{'feature':<44} {'rho':>7} {'q':>10}  {'cov':>5}")
    for _, r in full.head(25).iterrows():
        star = " *" if r.significant else ""
        say(f"{r.feature:<44} {r.spearman:>7.3f} {r.q:>10.2e}  {r.coverage:>5.2f}{star}")
    say()

    # per-model univariate: is the signal the same for a weak and a strong model?
    say("Same screen, per model (top 8 by |rho|)")
    permodel = {}
    for m in models:
        sub = cells[cells.llm_model == m][["question_id", "fail_rate"]]
        qm = q[["question_id"]].merge(sub, on="question_id", how="left")
        u = univariate_spearman(X, qm["fail_rate"].to_numpy(dtype=float))
        permodel[m] = u
        u.to_csv(OUT / f"tentrial_univariate_{m}.csv", index=False)
        say(f"  {m}  ({int(u.significant.sum())} BH-significant)")
        for _, r in u.head(8).iterrows():
            say(f"      {r.feature:<40} {r.spearman:>7.3f}  q={r.q:.2e}")
    say()

    # ---- stability selection --------------------------------------------- #
    say("-" * 78)
    say(f"Stability selection, {N_SPLITS} question folds, "
        f"selection inside train fold only")
    say("-" * 78)
    bins = pd.qcut(qf, q=4, duplicates="drop", labels=False).astype(int)
    counts: Counter = Counter()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for i, (tr, _) in enumerate(skf.split(X, bins), start=1):
        picked = _fast_select(X.iloc[tr], qf[tr], max_features=MAX_FEATURES)
        counts.update(picked)
        say(f"fold {i} ({len(tr)} train questions): {', '.join(picked)}")
    say()
    stable = [f for f, n in counts.most_common() if n >= MIN_FOLDS]
    pooled = full.set_index("feature")
    say(f"stable (in >= {MIN_FOLDS} of {N_SPLITS} folds):")
    for f in stable:
        row = pooled.loc[f] if f in pooled.index else None
        rho = f"{row.spearman:+.3f}" if row is not None else "  n/a"
        qv = f"{row.q:.2e}" if row is not None else "n/a"
        say(f"  {f:<44} folds={counts[f]}  rho={rho}  q={qv}")
    say()
    say("flicker (1-2 folds):")
    for f, n in counts.most_common():
        if n < MIN_FOLDS:
            say(f"  {f:<44} folds={n}")
    say()

    # ---- head to head against model x subject ---------------------------- #
    say("-" * 78)
    say("Head to head on cells, question-grouped 5-fold CV")
    say("-" * 78)
    say("Brier and log loss are weighted by the 10 trials behind each cell.")
    say("A feature set that does not beat model x subject has found nothing.")
    say()

    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = cells["question_id"].to_numpy()
    names = [
        "global mean", "model only", "subject only", "model x subject",
        "features only", "length only", "model x subject + features",
        "model x subject + stable features",
    ]
    preds = {n: np.zeros(len(cells)) for n in names}
    chosen_per_fold: list[list[str]] = []

    for tr_idx, te_idx in gkf.split(cells, groups=groups):
        tr, te = cells.iloc[tr_idx], cells.iloc[te_idx]
        prior = tr["n_fail"].sum() / (tr["n_fail"].sum() + tr["n_correct"].sum())

        m_tab = smoothed_rate(tr, ["llm_model"], prior, SMOOTH_K)
        c_tab = smoothed_rate(tr, ["question_category"], prior, SMOOTH_K)
        mc_tab = smoothed_rate(tr, ["llm_model", "question_category"], prior, SMOOTH_K)

        preds["global mean"][te_idx] = prior
        preds["model only"][te_idx] = predict_group(te, m_tab, ["llm_model"], prior)
        preds["subject only"][te_idx] = predict_group(te, c_tab, ["question_category"], prior)
        preds["model x subject"][te_idx] = predict_group(
            te, mc_tab, ["llm_model", "question_category"], prior
        )

        # feature selection on TRAIN questions only
        tr_q = q[q.question_id.isin(tr.question_id)]
        Xtr_q, _, _ = prepare_matrix(tr_q)
        picked = _fast_select(Xtr_q, tr_q["q_fail_rate"].to_numpy(float),
                                      max_features=MAX_FEATURES)
        chosen_per_fold.append(picked)
        qX = pd.concat([q[["question_id"]], X], axis=1)

        for name, cols, ctx in [
            ("features only", picked, False),
            ("length only", [c for c in ("f_question_length_words", "f_context_token_count")
                             if c in X.columns], False),
            ("model x subject + features", picked, True),
            ("model x subject + stable features", stable, True),
        ]:
            cols = [c for c in cols if c in qX.columns]
            Xtr_d = design(tr, qX, cols, models, cats, ctx)
            Xte_d = design(te, qX, cols, models, cats, ctx)
            Xe, ye, we = expand(tr, Xtr_d)
            preds[name][te_idx] = fit_predict_logistic(Xe, ye, we, Xte_d)

    say(f"{'model':<38} {'Brier':>8} {'logloss':>9} {'AUC':>7}")
    results = {}
    for n in names:
        b = weighted_brier(cells, preds[n])
        ll = weighted_logloss(cells, preds[n])
        a = cell_auc(cells, preds[n])
        results[n] = {"brier": b, "log_loss": ll, "auc": a}
        say(f"{n:<38} {b:>8.4f} {ll:>9.4f} {a:>7.3f}")
    say()
    base = results["model x subject"]["brier"]
    for n in ("model x subject + features", "model x subject + stable features", "features only"):
        d = results[n]["brier"] - base
        verdict = "beats" if d < 0 else "loses to"
        say(f"{n}: Brier {results[n]['brier']:.4f} vs {base:.4f} "
            f"({d:+.4f}) — {verdict} model x subject")
    say()

    # permutation null: shuffle q_fail_rate across questions, keep the grid
    say("Permutation check (question labels shuffled, 20 draws)")
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(20):
        perm = cells.copy()
        qids = q["question_id"].to_numpy()
        mapping = dict(zip(qids, rng.permutation(qids)))
        perm["question_id"] = perm["question_id"].map(mapping)
        p = np.zeros(len(perm))
        for tr_idx, te_idx in gkf.split(perm, groups=perm["question_id"].to_numpy()):
            tr, te = perm.iloc[tr_idx], perm.iloc[te_idx]
            prior = tr["n_fail"].sum() / (tr["n_fail"].sum() + tr["n_correct"].sum())
            qX = pd.concat([q[["question_id"]], X], axis=1)
            cols = [c for c in stable if c in qX.columns]
            Xtr_d = design(tr, qX, cols, models, cats, True)
            Xte_d = design(te, qX, cols, models, cats, True)
            Xe, ye, we = expand(tr, Xtr_d)
            p[te_idx] = fit_predict_logistic(Xe, ye, we, Xte_d)
        null.append(weighted_brier(perm, p))
    say("  (the null re-uses the stable feature set, so compare it with that row)")
    say(f"  null  mean {np.mean(null):.4f}  p5 {np.percentile(null, 5):.4f}")
    say(f"  real  {results['model x subject + stable features']['brier']:.4f}")
    if results["model x subject + stable features"]["brier"] > np.mean(null):
        say("  the real features do WORSE than shuffled ones: the gain is not")
        say("  small, it is negative. 15 features on 672 training cells overfit.")
    say()

    # ---- selective prediction ------------------------------------------- #
    say("-" * 78)
    say("Coverage-risk: abstain on the riskiest cells")
    say("-" * 78)
    nf = cells["n_fail"].to_numpy(float)
    nt = nf + cells["n_correct"].to_numpy(float)
    say(f"{'abstain':>8} {'model x subj':>14} {'+ features':>12}")
    for frac in (0.0, 0.1, 0.2, 0.3):
        line = [f"{frac:>7.0%}"]
        for n in ("model x subject", "model x subject + features"):
            p = preds[n]
            k = int(round(frac * len(p)))
            keep = np.argsort(p)[: len(p) - k] if k else np.arange(len(p))
            line.append(f"{nf[keep].sum() / nt[keep].sum():>13.3f}")
        say(" ".join(line))
    say()

    # ---- can features tell a coin flip from a certain miss? -------------- #
    say("-" * 78)
    say("New question 10 trials makes askable: unstable vs settled")
    say("-" * 78)
    q2 = q.copy()
    q2["unstable"] = ((q2.q_fail_rate > 0.1) & (q2.q_fail_rate < 0.9)).astype(float)
    u = univariate_spearman(X, q2["unstable"].to_numpy(float))
    u.to_csv(OUT / "tentrial_univariate_unstable.csv", index=False)
    say(f"{int(q2.unstable.sum())} of {len(q2)} questions sit between 10% and 90% risk.")
    say(f"{int(u.significant.sum())} of {len(u)} features BH-significant for instability.")
    for _, r in u.head(10).iterrows():
        star = " *" if r.significant else ""
        say(f"  {r.feature:<44} {r.spearman:>7.3f}  q={r.q:.2e}{star}")
    say()

    payload = {
        "n_questions": int(len(q)),
        "n_models": len(models),
        "n_cells": int(len(cells)),
        "n_answers": n_answers,
        "overall_fail_rate": float(cells["n_fail"].sum() / n_answers),
        "fail_by_model": {m: float(cells[cells.llm_model == m]["n_fail"].sum() / (N_TRIALS * 280)) for m in models},
        "fail_by_subject": {k: float(v) for k, v in bysub.items()},
        "cell_shape": {
            "always_right": float(np.mean(fr == 0.0)),
            "always_wrong": float(np.mean(fr == 1.0)),
            "intermediate": float(np.mean((fr > 0) & (fr < 1))),
        },
        "n_bh_significant": int(len(sig)),
        "univariate_top": full.head(25)[["feature", "spearman", "q", "significant"]].to_dict("records"),
        "stable_features": {f: int(counts[f]) for f in stable},
        "stability_counts": {f: int(n) for f, n in counts.most_common()},
        "fold_selections": chosen_per_fold,
        "cv": results,
        "permutation_null_brier": {"mean": float(np.mean(null)), "p5": float(np.percentile(null, 5))},
    }
    (OUT / "tentrial_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "tentrial_report.txt").write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    q.to_csv(OUT / "tentrial_question_table.csv", index=False)
    cells.to_csv(OUT / "tentrial_cells.csv", index=False)
    print(f"\nwrote {OUT / 'tentrial_report.txt'}")


if __name__ == "__main__":
    main()
