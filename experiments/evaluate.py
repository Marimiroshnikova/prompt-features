"""Honest evaluation of promptfeat features against real labeled outcomes.

Guards against the usual ways a result like this comes out fake:

  * groups        - the same prompt text appears under two responder models, and
                    Mis-prompt reuses flaw templates heavily, so every split is
                    grouped and never puts related prompts on both sides.
  * baselines     - every model is quoted next to a length-only baseline and, for
                    ReaLMistake, a responder-model-only baseline. A feature set
                    that does not beat those has found nothing.
  * permutation   - labels are shuffled within the grouping and the whole CV is
                    re-run, giving the AUC actually reachable by chance on this
                    sample size instead of assuming 0.5.
  * cluster bootstrap - univariate confidence intervals resample groups, not
                    rows, so repeated prompts do not shrink the interval.
  * FDR           - 138 features means 138 tests; p-values get Benjamini-Hochberg.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "out"
RNG = np.random.default_rng(0)

LENGTH_FEATURES = ["f_question_length_words", "f_question_length_chars"]


# --------------------------------------------------------------------------- #
# matrix preparation
# --------------------------------------------------------------------------- #

def value_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("f_") and not c.endswith(("__status", "__reason"))
    ]


def prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], pd.Series]:
    """Return a numeric matrix, the kind of each source feature, and coverage."""
    cols = value_columns(df)
    out: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    coverage: dict[str, float] = {}

    for col in cols:
        s = df[col]
        coverage[col] = float(s.notna().mean())
        levels = set(map(str, s.dropna().unique()))
        if levels and levels <= {"True", "False"}:
            kinds[col] = "bool"
            out[col] = s.map(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            ).astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            kinds[col] = "numeric"
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            kinds[col] = "categorical"
            # one-hot, but only levels with enough support to be estimable
            counts = s.value_counts()
            for level in counts[counts >= 20].index:
                out[f"{col}={level}"] = (s == level).astype(float)

    X = pd.DataFrame(out, index=df.index)
    # drop constants: they carry no information and break standardisation
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep], kinds, pd.Series(coverage)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def log_loss_safe(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def metrics_bundle(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"brier": brier(y, p), "log_loss": log_loss_safe(y, p), "auc": float("nan")}
    if len(np.unique(y)) > 1:
        out["auc"] = auc(y, p)
    return out


def impute_train_test(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def make_ridge_logistic(C: float = 0.5):
    """L2 logistic without class_weight — probabilities stay on the base rate (Brier)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=C, class_weight=None, solver="lbfgs"),
    )


# --------------------------------------------------------------------------- #
# univariate screen
# --------------------------------------------------------------------------- #

def cluster_bootstrap_ci(
    y: np.ndarray, x: np.ndarray, groups: np.ndarray, n: int = 400
) -> tuple[float, float]:
    """Resample whole groups so repeated prompts do not fake precision."""
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    aucs = []
    for _ in range(n):
        picked = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        yy, xx = y[idx], x[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(auc(yy, xx))
    if not aucs:
        return (np.nan, np.nan)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def univariate_screen(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    min_coverage: float = 0.5,
    coverage: pd.Series | None = None,
    bootstrap: bool = True,
) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        mask = ~np.isnan(x)
        cov = mask.mean()
        if cov < min_coverage or len(np.unique(x[mask])) < 2:
            continue
        yy, xx, gg = y[mask], x[mask], groups[mask]
        if len(np.unique(yy)) < 2:
            continue
        a = auc(yy, xx)
        u = stats.mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
        lo, hi = cluster_bootstrap_ci(yy, xx, gg) if bootstrap else (np.nan, np.nan)
        rows.append(
            {
                "feature": col,
                "auc": a,
                "auc_abs": abs(a - 0.5) + 0.5,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": float(u.pvalue),
                "coverage": float(cov),
                "n": int(mask.sum()),
            }
        )
    res = pd.DataFrame(rows).sort_values("auc_abs", ascending=False)
    if res.empty:
        return res
    # Benjamini-Hochberg
    p = res["p"].to_numpy()
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        j = m - rank
        prev = min(prev, p[i] * m / j)
        q[i] = prev
    res["q"] = q
    res["significant"] = res["q"] < 0.05
    return res.reset_index(drop=True)


def _bh_q(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        j = m - rank
        prev = min(prev, p[i] * m / j)
        q[i] = prev
    return q


def univariate_spearman(X: pd.DataFrame, y: np.ndarray, min_coverage: float = 0.5) -> pd.DataFrame:
    """Rank features against a continuous target (question-level fail rate)."""
    y = np.asarray(y, dtype=float)
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.mean() < min_coverage or len(np.unique(x[mask])) < 2:
            continue
        if np.std(y[mask]) == 0:
            continue
        rho, p = stats.spearmanr(x[mask], y[mask])
        if np.isnan(rho):
            continue
        rows.append(
            {
                "feature": col,
                "spearman": float(rho),
                "abs_spearman": float(abs(rho)),
                "p": float(p),
                "coverage": float(mask.mean()),
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["q"] = _bh_q(res["p"].to_numpy())
    res["significant"] = res["q"] < 0.05
    return res.sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def drop_correlated(
    X: pd.DataFrame,
    ranked: list[str],
    threshold: float = 0.9,
) -> list[str]:
    """Greedy keep: walk ranked features, skip any |corr| >= threshold with a keeper."""
    kept: list[str] = []
    for col in ranked:
        if col not in X.columns:
            continue
        x = X[col]
        redundant = False
        for k in kept:
            pair = pd.concat([x, X[k]], axis=1).dropna()
            if len(pair) < 8:
                continue
            if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                continue
            rho = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
            if pd.notna(rho) and abs(rho) >= threshold:
                redundant = True
                break
        if not redundant:
            kept.append(col)
    return kept


def select_features_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    max_features: int = 15,
    corr_threshold: float = 0.9,
    min_coverage: float = 0.5,
) -> list[str]:
    """Train-fold only: top |Spearman| after dropping near-duplicates, cap max_features."""
    screen = univariate_spearman(X, y, min_coverage=min_coverage)
    if screen.empty:
        return []
    ranked = screen["feature"].tolist()
    kept = drop_correlated(X, ranked, threshold=corr_threshold)
    return kept[:max_features]


# --------------------------------------------------------------------------- #
# cross-validated models
# --------------------------------------------------------------------------- #

def make_estimator(kind: str):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"),
        )
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )


def cv_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logreg",
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    Xv = X.to_numpy(dtype=float)
    if kind == "logreg":
        # median impute inside the loop would be cleaner; medians are computed on
        # train folds only below.
        pass
    scores: list[float] = []
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in cv.split(Xv, y, groups):
            Xtr, Xte = Xv[tr].copy(), Xv[te].copy()
            med = np.nanmedian(Xtr, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            Xtr = np.where(np.isnan(Xtr), med, Xtr)
            Xte = np.where(np.isnan(Xte), med, Xte)
            est = make_estimator(kind)
            est.fit(Xtr, y[tr])
            pred = est.predict_proba(Xte)[:, 1]
            if len(np.unique(y[te])) < 2:
                continue
            scores.append(auc(y[te], pred))
    return float(np.mean(scores)), float(np.std(scores)), scores


def permutation_null(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logreg",
    n_perm: int = 30,
    seed: int = 1234,
    mode: str = "rows",
) -> dict:
    """Re-run the whole CV with the feature-label association destroyed.

    mode="rows"    shuffle labels across all rows. Answers "could this AUC come
                   from no association at all", and keeps the class balance.
    mode="groups"  give each group a label drawn without replacement from the
                   observed group labels, then apply it to every row in the
                   group. Only valid when groups are class-pure; it additionally
                   prices in the small effective sample size when the number of
                   groups is low, which widens the null a lot.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    group_label = np.array([y[idx][0] for idx in idx_by_group])

    nulls = []
    for _ in range(n_perm):
        y_perm = y.copy()
        if mode == "groups":
            shuffled = rng.permutation(group_label)
            for idx, lab in zip(idx_by_group, shuffled):
                y_perm[idx] = lab
        else:
            y_perm = rng.permutation(y)
        if len(np.unique(y_perm)) < 2:
            continue
        try:
            m, _, _ = cv_auc(X, y_perm, groups, kind=kind, n_repeats=1)
        except ValueError:
            continue
        nulls.append(m)
    if not nulls:
        return {"mean": np.nan, "p95": np.nan, "max": np.nan, "n_perm": 0}
    return {
        "mean": float(np.mean(nulls)),
        "p95": float(np.percentile(nulls, 95)),
        "max": float(np.max(nulls)),
        "n_perm": len(nulls),
        "mode": mode,
    }


@dataclass
class Report:
    name: str
    lines: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)
