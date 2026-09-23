"""Step 2: how many questions before a real prompt-feature signal can show up.

No API calls. Uses the 280 questions we already have.

1. Threshold: the smallest |Spearman rho| that survives multiple testing across
   all prompt features, for each candidate dataset size.
2. Power: the questions needed to detect a given rho with 80% power.
3. Reality check on the 280: the strongest observed feature vs the strongest
   feature under shuffled labels, and how stable the top-10 list is when we
   bootstrap-resample questions.
4. Cost of each size for the same 3 models x 10 trials, plus a sealed 20% at
   25 trials.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.h_metrics import OUT, SEED, load_cells, make_x  # noqa: E402

ALPHA = 0.05
POWER = 0.80
SIZES = [280, 500, 1000, 1500, 2000, 3000]
EFFECTS = [0.10, 0.15, 0.165, 0.20, 0.25]
N_PERM = 500
N_SUB = 200
SUB_SIZES = [100, 150, 200, 280]
TOP_K = 10

# Published paid-tier USD per 1M tokens, September 2026 (ai.google.dev pricing).
PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-flash-latest": (0.75, 3.75),
}
TOKENS_IN, TOKENS_OUT = 230, 140  # per call, measured on the 1-trial run
BUFFER = 2.2                      # thinking tokens + retries, as in BUDGET_10_TRIALS.md
TRIALS, SEALED_SHARE, SEALED_TRIALS = 10, 0.20, 25


def r_threshold(n, m, alpha=ALPHA):
    """|r| needed for one feature to clear BH when it is the only hit (alpha / m)."""
    t = stats.t.ppf(1.0 - alpha / m / 2.0, df=n - 2)
    return float(t / np.sqrt(n - 2 + t * t))


def n_for_power(rho, m, alpha=ALPHA, power=POWER):
    """Fisher-z sample size for a two-sided test at alpha / m."""
    z_a = stats.norm.ppf(1.0 - alpha / m / 2.0)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(((z_a + z_b) / np.arctanh(rho)) ** 2 + 3))


def power_at(n, rho, m, alpha=ALPHA):
    z_a = stats.norm.ppf(1.0 - alpha / m / 2.0)
    return float(stats.norm.cdf(np.arctanh(rho) * np.sqrt(n - 3) - z_a))


def ranked(X):
    """Column-wise ranks, NaN filled with the column median first."""
    Xf = X.fillna(X.median(numeric_only=True))
    return Xf.rank().to_numpy(float)


def spearman_all(R, y):
    """|Spearman rho| of every ranked column against y."""
    yr = stats.rankdata(y)
    Rc = R - R.mean(axis=0)
    yc = yr - yr.mean()
    denom = np.sqrt((Rc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (Rc * yc[:, None]).sum(axis=0) / denom
    return np.nan_to_num(rho)


def cost_per_question(trials):
    per_call = sum(TOKENS_IN * pi / 1e6 + TOKENS_OUT * po / 1e6 for pi, po in PRICES.values())
    return per_call * trials * BUFFER


def main():
    df = load_cells()
    q = df.groupby("question_id", sort=True)
    y = q["fail_rate"].mean().to_numpy(float)
    X, _, _ = make_x(df, drop_model=True)
    X = X.copy().assign(question_id=df["question_id"].to_numpy())
    X = X.groupby("question_id", sort=True).first()
    X = X.select_dtypes(include=[np.number])
    X = X.loc[:, X.nunique() > 1]
    m = int(X.shape[1])
    names = X.columns.to_numpy()
    R = ranked(X)
    n_q = len(y)
    print(f"{n_q} questions, {m} numeric prompt features")

    rho = spearman_all(R, y)
    order = np.argsort(-np.abs(rho))
    top = [{"feature": str(names[i]), "rho": float(rho[i])} for i in order[:TOP_K]]
    observed_max = float(np.abs(rho).max())
    print("observed max |rho|", round(observed_max, 3), top[0])

    rng = np.random.default_rng(SEED)
    null_max = np.array([np.abs(spearman_all(R, rng.permutation(y))).max()
                         for _ in range(N_PERM)])
    null = {"median": float(np.median(null_max)),
            "p95": float(np.quantile(null_max, 0.95)),
            "share_at_or_above_observed": float((null_max >= observed_max).mean())}
    print("null max |rho|", null)

    full_top = set(order[:TOP_K])
    stability = []
    for n in SUB_SIZES:
        overlaps, maxes = [], []
        for _ in range(N_SUB):
            idx = rng.choice(n_q, size=n, replace=True)
            r = spearman_all(R[idx], y[idx])
            overlaps.append(len(full_top & set(np.argsort(-np.abs(r))[:TOP_K])) / TOP_K)
            maxes.append(float(np.abs(r).max()))
        stability.append({"n": n, "top10_overlap_mean": float(np.mean(overlaps)),
                          "max_rho_median": float(np.median(maxes)),
                          "threshold": r_threshold(n, m)})
        print(stability[-1])

    sizes = []
    for n in SIZES:
        sealed = int(round(n * SEALED_SHARE))
        cost = (n - sealed) * cost_per_question(TRIALS) + sealed * cost_per_question(SEALED_TRIALS)
        new_q = max(n - n_q, 0)
        new_cost = cost * new_q / n
        sizes.append({
            "n_questions": n,
            "rho_threshold": r_threshold(n, m),
            "power_at_observed_rho": power_at(n, observed_max, m),
            "cost_usd_total": float(cost),
            "cost_usd_new_questions_only": float(new_cost),
            "calls": int((n - sealed) * TRIALS * len(PRICES) + sealed * SEALED_TRIALS * len(PRICES)),
        })
        print(sizes[-1])

    needed = [{"rho": r, "n_questions": n_for_power(r, m)} for r in EFFECTS]
    print("needed", needed)

    payload = {
        "n_questions_now": n_q,
        "n_features_tested": m,
        "alpha": ALPHA,
        "power": POWER,
        "rule": "one true feature among m; BH then reduces to alpha / m",
        "observed_max_rho": observed_max,
        "observed_top10": top,
        "null_max_rho": null,
        "stability": stability,
        "sizes": sizes,
        "needed_for_power": needed,
        "cost_assumptions": {
            "prices_usd_per_1m": {k: {"input": v[0], "output": v[1]} for k, v in PRICES.items()},
            "tokens_in": TOKENS_IN, "tokens_out": TOKENS_OUT, "buffer": BUFFER,
            "trials": TRIALS, "sealed_share": SEALED_SHARE, "sealed_trials": SEALED_TRIALS,
            "cost_per_question_10_trials": cost_per_question(TRIALS),
            "source": "ai.google.dev/gemini-api/docs/pricing, September 2026; "
                      "gemini-flash-latest is an alias and can move",
        },
    }
    (OUT / "power_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT / "power_metrics.json")


if __name__ == "__main__":
    main()
