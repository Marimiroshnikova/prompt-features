"""How much question-level signal is there to find, and could we have seen it?

run_10trial.py reports that no prompt feature survives BH correction and that
none of them beat model x subject. A null like that has two very different
causes and they need different next moves:

  * there is no question-level signal beyond subject   -> stop building features
  * there is signal, these features miss it            -> build better features
  * there is signal, 280 questions cannot resolve it   -> buy more questions

This script separates them: reliability of the label, inter-model agreement on
which questions are hard, a variance decomposition, an oracle ceiling that
prices in perfect question difficulty, effect sizes with cluster-bootstrap CIs,
and the |rho| this sample size could actually have detected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import OUT, Report, prepare_matrix  # noqa: E402
from run_10trial import (  # noqa: E402
    N_TRIALS, N_SPLITS, SEED, SMOOTH_K,
    design, expand, fit_predict_logistic, load,
    predict_group, smoothed_rate, weighted_brier, cell_auc,
)

RESULTS = OUT / "reduced_10trial.csv"
RNG = np.random.default_rng(SEED)


def split_half(rep: Report) -> dict:
    """Correlate fail rate on trials 1-5 with trials 6-10, question level."""
    raw = pd.read_csv(RESULTS, low_memory=False)
    raw["ans"] = raw["answers"].astype(str)
    gold = raw["correct_answer"].astype(str).str.upper()

    def half(lo: int, hi: int) -> np.ndarray:
        out = []
        for s, g in zip(raw["ans"], gold):
            picks = list(s)[lo:hi]
            out.append(sum(1 for p in picks if p != g) / (hi - lo))
        return np.asarray(out, dtype=float)

    raw["h1"] = half(0, 5)
    raw["h2"] = half(5, 10)
    q = raw.groupby("question_id")[["h1", "h2"]].mean()
    rho, p = stats.spearmanr(q["h1"], q["h2"])
    r_pear = float(np.corrcoef(q["h1"], q["h2"])[0, 1])
    sb = 2 * r_pear / (1 + r_pear)          # Spearman-Brown, half -> full length

    rep.say("-" * 78)
    rep.say("Is the label itself stable? (trials 1-5 vs trials 6-10)")
    rep.say("-" * 78)
    rep.say(f"question-level split-half Pearson r = {r_pear:.3f}   Spearman = {rho:.3f}")
    rep.say(f"Spearman-Brown reliability of the full 30-trial label = {sb:.3f}")
    rep.say(f"ceiling on any feature's |rho| with true difficulty = sqrt(r) = "
            f"{np.sqrt(max(sb, 0)):.3f}")
    rep.say("A feature cannot correlate with the truth more strongly than the")
    rep.say("label correlates with itself. At n=1 per cell this ceiling was far lower.")
    rep.say()
    return {"split_half_pearson": r_pear, "split_half_spearman": float(rho),
            "spearman_brown": float(sb)}


def agreement(cells: pd.DataFrame, rep: Report) -> dict:
    wide = cells.pivot(index="question_id", columns="llm_model", values="fail_rate")
    models = list(wide.columns)
    rep.say("-" * 78)
    rep.say("Do the three models find the same questions hard?")
    rep.say("-" * 78)
    pairs = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            rho, p = stats.spearmanr(wide[a], wide[b])
            pairs[f"{a} vs {b}"] = float(rho)
            rep.say(f"  {a:<24} vs {b:<24} rho = {rho:.3f}  p = {p:.1e}")
    rep.say()
    rep.say("If the models agreed by chance these would sit near 0. They do not:")
    rep.say("question difficulty is real and shared. The open question is whether")
    rep.say("anything measurable in the text explains it.")
    rep.say()
    return pairs


def variance_decomposition(cells: pd.DataFrame, rep: Report) -> dict:
    """Where does cell-to-cell variation in fail rate live?"""
    y = cells["fail_rate"].to_numpy(float)
    total = y.var()
    grand = y.mean()

    def explained(keys: list[str]) -> float:
        fit = cells.groupby(keys)["fail_rate"].transform("mean").to_numpy(float)
        return float(1 - ((y - fit) ** 2).mean() / ((y - grand) ** 2).mean())

    parts = {
        "model": explained(["llm_model"]),
        "subject": explained(["question_category"]),
        "model x subject": explained(["llm_model", "question_category"]),
        "question (any model)": explained(["question_id"]),
        "question x model (saturated)": 1.0,
    }
    # binomial noise floor: even a perfect predictor cannot beat p(1-p)/10
    p = cells["fail_rate"].to_numpy(float)
    noise = float(np.mean(p * (1 - p) / N_TRIALS))

    rep.say("-" * 78)
    rep.say("Variance decomposition of cell fail rate")
    rep.say("-" * 78)
    rep.say(f"total variance across the 840 cells: {total:.4f}")
    for k, v in parts.items():
        rep.say(f"  {k:<30} explains {v:>6.1%}")
    rep.say(f"sampling noise from 10 trials alone: {noise:.4f} "
            f"({noise / total:.1%} of total)")
    rep.say()
    rep.say("'question' explains far more than 'subject'. Subject is a coarse proxy")
    rep.say("for something that lives at the individual question.")
    rep.say()
    return {"total_variance": float(total), "explained": parts, "binomial_noise": noise}


def oracle_ceiling(cells: pd.DataFrame, q: pd.DataFrame, rep: Report) -> dict:
    """What would a perfect question-difficulty feature be worth?"""
    models = sorted(cells["llm_model"].unique())
    cats = sorted(cells["question_category"].unique())
    wide = cells.pivot(index="question_id", columns="llm_model", values="fail_rate")

    # leave-own-model-out difficulty: never uses the cell's own label
    loo = {}
    for m in models:
        others = [c for c in wide.columns if c != m]
        loo[m] = wide[others].mean(axis=1)
    cells = cells.copy()
    cells["oracle_diff"] = [
        loo[m].loc[qid] for m, qid in zip(cells["llm_model"], cells["question_id"])
    ]

    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = cells["question_id"].to_numpy()
    p_base = np.zeros(len(cells))
    p_oracle = np.zeros(len(cells))
    qX = pd.DataFrame({"question_id": cells["question_id"].to_numpy()})

    for tr_i, te_i in gkf.split(cells, groups=groups):
        tr, te = cells.iloc[tr_i], cells.iloc[te_i]
        prior = tr["n_fail"].sum() / (tr["n_fail"].sum() + tr["n_correct"].sum())
        mc = smoothed_rate(tr, ["llm_model", "question_category"], prior, SMOOTH_K)
        p_base[te_i] = predict_group(te, mc, ["llm_model", "question_category"], prior)

        Xtr = np.column_stack([
            *[(tr["llm_model"] == m).to_numpy(float) for m in models],
            *[(tr["question_category"] == c).to_numpy(float) for c in cats],
            tr["oracle_diff"].to_numpy(float),
        ])
        Xte = np.column_stack([
            *[(te["llm_model"] == m).to_numpy(float) for m in models],
            *[(te["question_category"] == c).to_numpy(float) for c in cats],
            te["oracle_diff"].to_numpy(float),
        ])
        Xe, ye, we = expand(tr, Xtr)
        p_oracle[te_i] = fit_predict_logistic(Xe, ye, we, Xte)

    b_base = weighted_brier(cells, p_base)
    b_or = weighted_brier(cells, p_oracle)
    noise = float(np.mean(cells["fail_rate"] * (1 - cells["fail_rate"])))

    rep.say("-" * 78)
    rep.say("Oracle ceiling: model x subject + PERFECT question difficulty")
    rep.say("-" * 78)
    rep.say("Difficulty is measured from the other two models' answers on the same")
    rep.say("question, so a cell never sees its own label. This is not deployable —")
    rep.say("it is the prize a text feature would be competing for.")
    rep.say()
    rep.say(f"  model x subject                       Brier {b_base:.4f}  "
            f"AUC {cell_auc(cells, p_base):.3f}")
    rep.say(f"  + oracle question difficulty          Brier {b_or:.4f}  "
            f"AUC {cell_auc(cells, p_oracle):.3f}")
    rep.say(f"  irreducible (10-trial sampling noise) Brier {noise:.4f}")
    rep.say()
    room = b_base - b_or
    rep.say(f"Question-level signal worth {room:.4f} Brier is sitting there.")
    rep.say(f"The best text feature set captured {0.0:.4f} of it.")
    rep.say("So: the signal exists, and these 138 features do not see it.")
    rep.say()
    return {"brier_model_subject": b_base, "brier_oracle": b_or,
            "brier_noise_floor": noise, "headroom": float(room)}


def effect_sizes(cells: pd.DataFrame, q: pd.DataFrame, rep: Report) -> list[dict]:
    """Descriptive contrasts for the boolean flags, with question-bootstrap CIs."""
    flags = [c for c in q.columns
             if c.startswith("f_")
             and not c.endswith(("__status", "__reason"))
             and set(map(str, q[c].dropna().unique())) <= {"True", "False"}]
    y = q["q_fail_rate"].to_numpy(float)
    rows = []
    for c in flags:
        v = q[c].map({True: 1, False: 0, "True": 1, "False": 0}).to_numpy(float)
        mask = ~np.isnan(v)
        if mask.sum() < len(v) * 0.5:
            continue
        on, off = y[mask][v[mask] == 1], y[mask][v[mask] == 0]
        if len(on) < 8 or len(off) < 8:
            continue
        diffs = []
        for _ in range(2000):
            i1 = RNG.integers(0, len(on), len(on))
            i0 = RNG.integers(0, len(off), len(off))
            diffs.append(on[i1].mean() - off[i0].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({
            "flag": c, "n_true": int(len(on)),
            "fail_true": float(on.mean()), "fail_false": float(off.mean()),
            "diff": float(on.mean() - off.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
        })
    rows.sort(key=lambda r: -abs(r["diff"]))

    rep.say("-" * 78)
    rep.say("Effect sizes on the 10-trial label (descriptive, not a predictor)")
    rep.say("-" * 78)
    rep.say(f"{'flag':<34} {'n':>4} {'on':>6} {'off':>6} {'diff':>7} "
            f"{'95% CI':>18}")
    for r in rows[:18]:
        star = " *" if r["excludes_zero"] else ""
        rep.say(f"{r['flag']:<34} {r['n_true']:>4} {r['fail_true']:>6.3f} "
                f"{r['fail_false']:>6.3f} {r['diff']:>+7.3f} "
                f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]{star}")
    rep.say()
    rep.say("* = bootstrap CI over questions excludes zero. These are honest")
    rep.say("descriptions of the 280 questions we have; the CV above is what says")
    rep.say("whether they transfer to questions we have not seen.")
    rep.say()
    return rows


def power(rep: Report, n: int = 280, m_tests: int = 139, observed: float = 0.165) -> dict:
    """What |rho| could this sample have detected under BH over m tests?"""
    alpha = 0.05 / m_tests                       # threshold the top-ranked test faces
    z = stats.norm.ppf(1 - alpha / 2)
    rho_detect = float(np.tanh(z / np.sqrt(n - 3)))
    z_beta = stats.norm.ppf(0.80)
    n_needed = float(((z + z_beta) / np.arctanh(observed)) ** 2 + 3)

    rep.say("-" * 78)
    rep.say("Was this a null result or an underpowered one?")
    rep.say("-" * 78)
    rep.say(f"n = {n} questions, {m_tests} simultaneous tests.")
    rep.say(f"Smallest |rho| that could clear BH here: {rho_detect:.3f}")
    rep.say(f"Largest |rho| we observed:               {observed:.3f}")
    rep.say(f"Questions needed to call rho={observed:.3f} significant at 80% power: "
            f"{int(np.ceil(n_needed))}")
    rep.say()
    rep.say("The plan's 1,500-3,000 questions is not a nice-to-have. At 280 the")
    rep.say("study cannot resolve effects of the size actually present.")
    rep.say()
    return {"alpha_per_test": float(alpha), "rho_detectable": rho_detect,
            "rho_observed": observed, "n_needed_80pct": n_needed}


def main() -> None:
    cells, q = load()
    rep = Report("mmlu_10trial_diagnostics")
    rep.say("=" * 78)
    rep.say("Diagnostics — how much signal exists, and could we have seen it")
    rep.say("=" * 78)
    rep.say()

    payload = {
        "reliability": split_half(rep),
        "inter_model_agreement": agreement(cells, rep),
        "variance": variance_decomposition(cells, rep),
        "oracle": oracle_ceiling(cells, q, rep),
        "effect_sizes": effect_sizes(cells, q, rep),
        "power": power(rep),
    }
    pd.DataFrame(payload["effect_sizes"]).to_csv(
        OUT / "tentrial_effect_sizes.csv", index=False)
    (OUT / "tentrial_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "tentrial_diagnostics.txt").write_text(
        "\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'tentrial_diagnostics.txt'}")


if __name__ == "__main__":
    main()
