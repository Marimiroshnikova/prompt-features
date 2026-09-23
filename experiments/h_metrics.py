"""Shared data loading, folds, and locked Phase 6 metrics for H1-H4.

Scores are P(this answer is wrong). Brier and log loss are weighted by the
trials behind each cell, so a 10-trial cell counts as ten outcomes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "out"
DATA = ROOT / "data" / "gaia_training_dataset.csv"
CELLS = OUT / "tentrial_cells.csv"

SEED = 42
N_FOLDS = 5
N_BOOT = 1000
LOGLOSS_EPS = 1e-4
ECE_BINS = 15

LABEL_COLS = [
    "row_id", "question_id", "target",
    "question_category", "n_correct", "n_fail", "n_blank", "fail_rate",
]
MODEL_COLS = ["llm_model", "model_family", "knowledge_cutoff_year", "f_recency_gap"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_cells() -> pd.DataFrame:
    """Teammate feature table joined to the 10-trial counts, one row per cell."""
    df = pd.read_csv(DATA)
    cells = pd.read_csv(CELLS)
    merged = df.merge(cells, on=["question_id", "llm_model"], how="left",
                      validate="one_to_one")
    if merged["fail_rate"].isna().any():
        raise SystemExit("training rows did not match tentrial_cells")
    if not np.allclose(1.0 - merged["target"].to_numpy(float),
                       merged["fail_rate"].to_numpy(float)):
        raise SystemExit("CSV target is not 1 - fail_rate")
    return merged


def make_x(df: pd.DataFrame, drop_model: bool = False,
           keep_model: list[str] | None = None):
    """Feature matrix. Constant columns are dropped; strings become categories.

    drop_model removes every model column. keep_model, when given, keeps only
    those model columns.
    """
    X = df.drop(columns=[c for c in LABEL_COLS if c in df.columns])
    if drop_model:
        X = X.drop(columns=[c for c in MODEL_COLS if c in X.columns])
    elif keep_model is not None:
        X = X.drop(columns=[c for c in MODEL_COLS if c in X.columns and c not in keep_model])
    nunique = X.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=constant)
    cat = [c for c in X.columns
           if X[c].dtype == object or isinstance(X[c].dtype, pd.StringDtype)]
    for c in cat:
        X[c] = X[c].astype("category")
    return X, constant, cat


def question_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED):
    """Yield (train_mask, test_mask) over cells; a question is never on both sides.

    Stratified on the question's mean fail rate (quartiles).
    """
    q_rate = df.groupby("question_id")["fail_rate"].mean()
    q_ids = q_rate.index.to_numpy()
    bins = pd.qcut(q_rate, q=4, duplicates="drop", labels=False).to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    qid = df["question_id"].to_numpy()
    for tr_q, _ in skf.split(q_ids, bins):
        tr = np.isin(qid, q_ids[tr_q])
        yield tr, ~tr


def group_folds(df: pd.DataFrame, col: str):
    """Leave-one-level-out folds on a column (subject-out, model-out)."""
    values = df[col].to_numpy()
    for level in sorted(df[col].unique()):
        te = values == level
        yield level, ~te, te


# --------------------------------------------------------------------------- #
# rate baselines, fit on training cells only
# --------------------------------------------------------------------------- #

def smoothed_rates(tr: pd.DataFrame, keys: list[str], prior: float, k: float = 10.0):
    g = tr.groupby(keys)[["n_fail", "n_correct"]].sum()
    n = g["n_fail"] + g["n_correct"]
    return (g["n_fail"] + k * prior) / (n + k)


def predict_rates(te: pd.DataFrame, table: pd.Series, keys: list[str],
                  fallback: np.ndarray | float) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(te[keys]) if len(keys) > 1 else te[keys[0]]
    out = table.reindex(idx).to_numpy(dtype=float)
    fb = np.broadcast_to(np.asarray(fallback, dtype=float), out.shape)
    return np.where(np.isnan(out), fb, out)


def baseline_predictions(tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, np.ndarray]:
    """Global, model, subject, model x subject rates. Unseen levels fall back."""
    prior = float(tr["n_fail"].sum() / (tr["n_fail"] + tr["n_correct"]).sum())
    m = predict_rates(te, smoothed_rates(tr, ["llm_model"], prior), ["llm_model"], prior)
    s = predict_rates(te, smoothed_rates(tr, ["question_category"], prior),
                      ["question_category"], prior)
    ms_table = smoothed_rates(tr, ["llm_model", "question_category"], prior)
    ms = predict_rates(te, ms_table, ["llm_model", "question_category"], s)
    # model x subject falls back to the model rate when the subject is unseen
    unseen_subject = ~te["question_category"].isin(tr["question_category"]).to_numpy()
    ms = np.where(unseen_subject, m, ms)
    return {
        "global mean": np.full(len(te), prior),
        "model mean": m,
        "subject mean": s,
        "model x subject": ms,
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def trial_brier(nf, nc, p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return float((nf * (p - 1.0) ** 2 + nc * p ** 2).sum() / (nf + nc).sum())


def trial_logloss(nf, nc, p, eps=LOGLOSS_EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float((-(nf * np.log(p) + nc * np.log(1.0 - p))).sum() / (nf + nc).sum())


def ece_quantile(y_rate, p, n_bins=ECE_BINS):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_rate, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p, y = p[order], y[order]
    err = 0.0
    for idx in np.array_split(np.arange(len(p)), n_bins):
        if len(idx):
            err += abs(p[idx].mean() - y[idx].mean()) * len(idx)
    return float(err / len(p))


def reliability_bins(y_rate, p, n_bins=10):
    """Quantile bins: mean prediction, mean observed rate, and count."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_rate, dtype=float)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            rows.append({"pred": float(p[idx].mean()), "obs": float(y[idx].mean()),
                         "n": int(len(idx))})
    return rows


def _selective(nf, nc, p):
    order = np.argsort(np.asarray(p, dtype=float), kind="mergesort")  # safest first
    c_nf = np.cumsum(np.asarray(nf, dtype=float)[order])
    c_n = np.cumsum((np.asarray(nf, dtype=float) + np.asarray(nc, dtype=float))[order])
    return c_n / c_n[-1], c_nf / c_n


def coverage_curve(nf, nc, p, n_grid=21):
    """Error among answered items after skipping the riskiest ones."""
    cov, risk = _selective(nf, nc, p)
    grid = np.linspace(0.05, 1.0, n_grid)
    idx = np.minimum(np.searchsorted(cov, grid, side="left"), len(cov) - 1)
    curve = risk[idx]
    aurc = float(np.trapezoid(curve, grid) / (grid[-1] - grid[0]))
    return grid, curve, aurc


def risk_at_coverage(nf, nc, p, target_cov):
    cov, risk = _selective(nf, nc, p)
    k = min(int(np.searchsorted(cov, target_cov, side="left")), len(cov) - 1)
    return float(risk[k])


def coverage_at_risk(nf, nc, p, target_risk):
    cov, risk = _selective(nf, nc, p)
    ok = np.where(risk <= target_risk)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def ranking_metrics(y_bin, p):
    y = np.asarray(y_bin, dtype=int)
    if y.min() == y.max():
        return {"auroc": None, "auprc": None, "auprc_base_rate": float(y.mean())}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "auprc_base_rate": float(y.mean()),
    }


def pack(name, df, p, base_brier=None):
    """Every locked metric for one out-of-fold prediction vector."""
    nf = df["n_fail"].to_numpy(float)
    nc = df["n_correct"].to_numpy(float)
    y_rate = df["fail_rate"].to_numpy(float)
    y_bin = (y_rate > 0.5).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    grid, curve, aurc = coverage_curve(nf, nc, p)
    brier = trial_brier(nf, nc, p)
    const = trial_brier(nf, nc, np.full(len(p), nf.sum() / (nf + nc).sum()))
    return {
        "model": name,
        "brier": brier,
        "brier_base_rate_constant": const,
        "bss": float(1.0 - brier / const),
        "bss_vs_oof_base": (float(1.0 - brier / base_brier)
                            if base_brier is not None else 0.0),
        "log_loss": trial_logloss(nf, nc, p),
        "ece_15_quantile": ece_quantile(y_rate, p),
        "rmse_vs_soft": float(np.sqrt(np.mean((p - y_rate) ** 2))),
        "rmse_noise_floor": float(np.sqrt(np.mean(y_rate * (1.0 - y_rate) / 10.0))),
        "aurc": aurc,
        "risk_at_coverage_90": risk_at_coverage(nf, nc, p, 0.90),
        "risk_at_coverage_80": risk_at_coverage(nf, nc, p, 0.80),
        "risk_at_coverage_70": risk_at_coverage(nf, nc, p, 0.70),
        "coverage_at_risk_0.15": coverage_at_risk(nf, nc, p, 0.15),
        "coverage_at_risk_0.10": coverage_at_risk(nf, nc, p, 0.10),
        **ranking_metrics(y_bin, p),
        "coverage_grid": [round(float(x), 4) for x in grid],
        "selective_risk": [round(float(x), 4) for x in curve],
        "reliability": reliability_bins(y_rate, p),
        "pred_hist": np.histogram(p, bins=10, range=(0.0, 1.0))[0].tolist(),
    }


# --------------------------------------------------------------------------- #
# question bootstrap
# --------------------------------------------------------------------------- #

def question_bootstrap(df, stat, n_boot=N_BOOT, seed=SEED):
    """Resample questions with replacement; stat(idx) -> float. Returns mean and 95% CI."""
    qid = df["question_id"].to_numpy()
    unique_q = np.unique(qid)
    q_to_idx = {q: np.where(qid == q)[0] for q in unique_q}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(unique_q, size=len(unique_q), replace=True)
        vals.append(stat(np.concatenate([q_to_idx[q] for q in draw])))
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()),
            "lo": float(np.quantile(arr, 0.025)),
            "hi": float(np.quantile(arr, 0.975))}


def brier_gap_ci(df, p_a, p_b):
    """Brier(a) - Brier(b) with a question bootstrap. Negative means a is better."""
    nf = df["n_fail"].to_numpy(float)
    nc = df["n_correct"].to_numpy(float)
    return question_bootstrap(
        df, lambda i: trial_brier(nf[i], nc[i], p_a[i]) - trial_brier(nf[i], nc[i], p_b[i]))


def risk80_gap_ci(df, p_a, p_b):
    """Error at 80% coverage, a minus b. Negative means a leaves fewer errors."""
    nf = df["n_fail"].to_numpy(float)
    nc = df["n_correct"].to_numpy(float)
    return question_bootstrap(
        df, lambda i: risk_at_coverage(nf[i], nc[i], p_a[i], 0.80)
        - risk_at_coverage(nf[i], nc[i], p_b[i], 0.80))
