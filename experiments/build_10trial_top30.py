"""Re-derive the Top 30 from the 10-trial labels.

web/top30.json was built from n=1 answers per cell, where the label was a single
Bernoulli draw. This ranks the same candidates against the 30-trial question fail
rate, and does not rank on |Spearman| alone — one correlation on 280 questions is
too easy to get by luck when 139 are being tested.

A feature's score blends three independent pieces of evidence:

  0.45  |rho|        Spearman against q_fail_rate, all 280 questions
  0.35  stability    how many of 5 question folds picked it, selection inside the fold
  0.20  consistency  does it point the same way for all three models, and how hard

plus a small bonus when a boolean flag's bootstrap CI over questions excludes zero.

Then the same hygiene as select_top30.py: drop anything correlated |rho| >= 0.9
with a feature already kept, one dummy per categorical parent, cap 30.

None of these are BH-significant at n=280 (see TENTRIAL_FINDINGS.md section 5 —
the detection floor is |rho| = 0.211 and the ceiling actually present is 0.165).
This is a ranked list of candidates to carry into a larger run, not a shortlist
of established predictors.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from evaluate import OUT, prepare_matrix  # noqa: E402

W_RHO, W_FOLD, W_CONS = 0.45, 0.35, 0.20
CI_BONUS = 0.05
CORR_THRESHOLD = 0.9
CAP = 30


def parent(col: str) -> str:
    name = col[2:] if col.startswith("f_") else col
    return name.split("=", 1)[0]


def parse_features_md(path: Path) -> dict[str, dict[str, str]]:
    """Group heading and the 'What we see' line for every documented feature."""
    if not path.exists():
        return {}
    info: dict[str, dict[str, str]] = {}
    group = ""
    name = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            group = line[3:].strip()
        elif line.startswith("### `"):
            m = re.match(r"### `([^`]+)`", line)
            name = m.group(1) if m else ""
            if name:
                info[name] = {"group": group, "what": ""}
        elif name and line.startswith("- **What we see:**"):
            info[name]["what"] = line.split("**What we see:**", 1)[1].strip()
            name = ""
    return info



RETRIEVAL_HEADLINERS = [
    ("f_anchor_count", 1), ("f_anchor_density", 2), ("f_unresolved_pronoun_count", 4),
    ("f_has_dangling_reference", 5), ("f_vague_term_count", 6),
]


def write_doc(top: pd.DataFrame, table: str, pooled: pd.DataFrame) -> None:
    """The readable version of the ranking, with the caveats attached to it."""
    nl = "\n"
    groups = top["group"].value_counts()
    tier_a = top[(top.folds >= 3) & (top.ci_excludes_zero == True)]  # noqa: E712
    tier_b = top[(top.folds >= 3) & (top.ci_excludes_zero != True)]  # noqa: E712
    new = top[~top.in_n1_top30]

    rows = ["| rank | feature | direction | what it measures |", "| ---: | --- | --- | --- |"]
    for _, r in top.iterrows():
        rows.append(f"| {int(r['rank'])} | `{r['label']}` | {r['direction']} | "
                    f"{str(r['what_we_see'] or '').rstrip('.')} |")

    gone = []
    for col, old_rank in RETRIEVAL_HEADLINERS:
        rho = float(pooled.loc[col, "spearman"]) if col in pooled.index else float("nan")
        gone.append(f"`{col[2:]}` (retrieval rank {old_rank}, here rho={rho:+.3f})")

    doc = f"""# The best 30 features — re-derived on the 10-trial data

Ranked against **`q_fail_rate`**: the mean fail rate of 3 Gemini models over
**10 answers each**, so 30 trials behind every question. `web/top30.json` was ranked on
1 answer per cell, where the label was a single coin flip.

Rebuild with `python experiments/build_10trial_top30.py`.

## Read this before quoting the list

**None of these 30 are statistically significant.** Zero of 139 features clear
Benjamini-Hochberg at q < 0.05. At n = 280 questions with 139 simultaneous tests the
smallest detectable |rho| is **0.211**, and the largest effect actually present is
**0.165** — the study cannot resolve what is there. This is a ranked list of
**candidates to carry into a larger run**, not a list of established predictors, and
none of them beats the model x subject baseline (see `TENTRIAL_FINDINGS.md`).

## How the ranking is built

One correlation on 280 questions is too easy to get by luck when 139 are being tested,
so |rho| is not the whole score. Each feature is scored on three independent pieces of
evidence, then de-duplicated:

| weight | evidence | what it guards against |
| ---: | --- | --- |
| {W_RHO} | \\|Spearman\\| vs `q_fail_rate`, all 280 questions | — |
| {W_FOLD} | picked in N of 5 question folds, selection inside the fold | a correlation that lives in one slice of the data |
| {W_CONS} | same sign across all 3 models, weighted by strength | a fluke only one model produces |
| +{CI_BONUS} | boolean flag whose bootstrap CI over questions excludes zero | an effect indistinguishable from zero |

Then drop anything correlated |rho| >= {CORR_THRESHOLD} with a feature already kept, one
dummy per categorical parent, cap {CAP}. Same hygiene as `select_top30.py`.

## The 30

`*` on an effect means the bootstrap CI over questions excludes zero. `folds` is
stability; `models` is how many of the 3 point the same way.

{table}

## What each one measures

{nl.join(rows)}

## Reading the list

**Exam-item traps take 6 of the top 13** — `is_definition_ask`, `is_long_scenario`,
`is_best_answer_judgment`, `stem_word_count`, `has_escape_option`, `is_except_ask`. That
group was written *after* reading the misses; the rest of the set was inherited from the
retrieval design. It is the clearest signal in the table about where the next round of
feature work should go.

**The retrieval headliners competed and lost.** {", ".join(gone)} — ranks 1, 2, 4, 5 and 6
of the retrieval Top 30 in `FEATURES.md` — are all screened here and none reaches
|rho| = 0.06. `is_ambiguous` falls from rank 3 to {int(top.loc[top.label == "is_ambiguous", "rank"].iloc[0])} and
`context_token_count` from 8 to {int(top.loc[top.label == "context_token_count", "rank"].iloc[0])}.
An MMLU-Pro item prints its ten options: there is nothing to retrieve and nothing
dangling, so the features that measure retrievability have no work to do. Spec mismatch,
not a bug.

**Three you can quote out loud** — stable in >= 3 folds *and* CI excludes zero:

{nl.join(f"- `{r.label}` — {r['diff']:+.3f} fail rate on vs off, 95% CI [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], n={int(r.n_true)} questions" for _, r in tier_a.iterrows())}

**{len(tier_b)} that are stable but unresolvable at this sample size** — keep measuring,
do not ship: {", ".join(f"`{r.label}`" for _, r in tier_b.iterrows())}.

**Rank {int(top.loc[top.label == "is_except_ask", "rank"].iloc[0])} is the one to fund.**
`is_except_ask` carries the largest effect in the whole feature set (+0.231: except/NOT
items fail 42.7% against 19.6%) on **11 questions**, which is why its CI still touches
zero. Sampling 100+ except/NOT items would settle it. Data problem, not a modelling
problem.

**Churn against the n=1 ranking: {len(new)} of {CAP} entries are new**
({", ".join(f"`{r.label}`" for _, r in new.iterrows())}). Two rankings built from the same
280 questions disagree on a third of their entries — a direct measure of how unstable any
ranking is at this n.

**Group coverage:**

{nl.join(f"- {g} — {n}" for g, n in groups.items())}

## Files

| file | what |
| --- | --- |
| `tentrial_top30.json` | the ranked 30, `web/top30.json` schema plus score, folds, effects |
| `tentrial_top30.csv` | the same as a table, with group and description columns |
| `tentrial_top30.md` | just the ranked table |
| `tentrial_univariate_qfail.csv` | all 139 candidates, unranked |
"""
    (OUT / "TOP30_10TRIAL.md").write_text(doc, encoding="utf-8")


def main() -> None:
    res = json.loads((OUT / "tentrial_results.json").read_text())
    diag = json.loads((OUT / "tentrial_diagnostics.json").read_text())
    pooled = pd.read_csv(OUT / "tentrial_univariate_qfail.csv").set_index("feature")
    folds = res["stability_counts"]
    eff = pd.DataFrame(diag["effect_sizes"]).set_index("flag")
    models = sorted(res["fail_by_model"])
    per_model = {
        m: pd.read_csv(OUT / f"tentrial_univariate_{m}.csv").set_index("feature")
        for m in models
    }
    docs = parse_features_md(REPO / "FEATURES.md")
    try:
        old = {f["column"] for f in json.loads(
            (REPO / "web" / "top30.json").read_text())["features"]}
    except Exception:
        old = set()

    rows = []
    max_rho = float(pooled["abs_spearman"].max())
    for col, r in pooled.iterrows():
        signed = []
        for m in models:
            pm = per_model[m]
            if col in pm.index:
                signed.append(float(pm.loc[col, "spearman"]) * np.sign(r["spearman"]))
        cons_raw = float(np.mean(signed)) if signed else 0.0
        agree = int(sum(1 for s in signed if s > 0))
        e = eff.loc[col] if col in eff.index else None
        rows.append({
            "column": col,
            "name": parent(col),
            "rho": float(r["spearman"]),
            "abs_rho": float(r["abs_spearman"]),
            "q": float(r["q"]),
            "significant": bool(r["significant"]),
            "folds": int(folds.get(col, 0)),
            "models_agree": agree,
            "cons_raw": cons_raw,
            "n_true": int(e["n_true"]) if e is not None else None,
            "diff": float(e["diff"]) if e is not None else None,
            "ci_lo": float(e["ci_lo"]) if e is not None else None,
            "ci_hi": float(e["ci_hi"]) if e is not None else None,
            "ci_excludes_zero": bool(e["excludes_zero"]) if e is not None else False,
        })
    df = pd.DataFrame(rows)
    max_cons = max(float(df["cons_raw"].max()), 1e-9)
    df["score"] = (
        W_RHO * df["abs_rho"] / max_rho
        + W_FOLD * df["folds"] / 5.0
        + W_CONS * df["cons_raw"].clip(lower=0) / max_cons
        + CI_BONUS * df["ci_excludes_zero"].astype(float)
    ).clip(upper=1.0)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # redundancy pruning on the actual feature matrix
    q = pd.read_csv(OUT / "tentrial_question_table.csv", low_memory=False)
    X, _, _ = prepare_matrix(q)
    ranked = [c for c in df["column"] if c in X.columns]
    filled = X[ranked].apply(lambda s: s.fillna(s.median()))
    R = filled.rank().to_numpy(dtype=float)
    R = R - R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = 1.0
    C = np.abs((R / sd).T @ (R / sd)) / len(R)
    pos = {c: i for i, c in enumerate(ranked)}

    kept: list[str] = []
    seen_parent: set[str] = set()
    dropped: list[tuple[str, str]] = []
    for col in ranked:
        p = parent(col)
        if p in seen_parent:
            dropped.append((col, f"another dummy of {p} already kept"))
            continue
        clash = next((k for k in kept if C[pos[col], pos[k]] >= CORR_THRESHOLD), None)
        if clash:
            dropped.append((col, f"|rho| >= {CORR_THRESHOLD} with {clash}"))
            continue
        kept.append(col)
        seen_parent.add(p)
        if len(kept) == CAP:
            break

    top = df[df.column.isin(kept)].copy()
    top["rank"] = top["column"].map({c: i + 1 for i, c in enumerate(kept)})
    top = top.sort_values("rank")
    top["label"] = top["column"].str.replace("^f_", "", regex=True)
    dummy = top["column"].str.contains("=")
    top["direction"] = np.where(
        top["rho"] >= 0,
        np.where(dummy, "this level = more misses", "higher = more misses"),
        np.where(dummy, "this level = fewer misses", "higher = fewer misses"),
    )
    top["group"] = top["name"].map(lambda n: docs.get(n, {}).get("group", "—"))
    top["what_we_see"] = top["name"].map(lambda n: docs.get(n, {}).get("what", ""))
    top["in_n1_top30"] = top["column"].isin(old)
    top.to_csv(OUT / "tentrial_top30.csv", index=False)

    payload = {
        "source": "280 MMLU-Pro questions x 3 Gemini models x 10 trials = 8,400 answers",
        "label": "q_fail_rate, mean fail rate over 3 models, 30 trials per question",
        "method": (
            f"score = {W_RHO} |Spearman| + {W_FOLD} fold stability + {W_CONS} "
            f"cross-model consistency (+{CI_BONUS} if bootstrap CI excludes zero); "
            f"drop |corr| >= {CORR_THRESHOLD}; one dummy per categorical; cap {CAP}"
        ),
        "note": (
            "Zero of 139 features are BH-significant at n=280; the detection floor "
            "is |rho|=0.211 and the largest effect present is 0.165. Ranked "
            "candidates for a larger run, not established predictors. None of them "
            "beat the model x subject baseline."
        ),
        "features": [
            {
                "rank": int(r["rank"]), "name": r["name"], "label": r["label"],
                "column": r["column"], "direction": r["direction"],
                "group": r["group"], "score": round(float(r["score"]), 4),
                "spearman": round(float(r["rho"]), 4), "q": round(float(r["q"]), 4),
                "significant": bool(r["significant"]),
                "folds": int(r["folds"]), "models_agree": int(r["models_agree"]),
                "effect": (None if pd.isna(r["diff"]) else {
                    "n_true": int(r["n_true"]), "diff": round(float(r["diff"]), 4),
                    "ci": [round(float(r["ci_lo"]), 4), round(float(r["ci_hi"]), 4)],
                    "excludes_zero": bool(r["ci_excludes_zero"]),
                }),
                "in_n1_top30": bool(r["in_n1_top30"]),
            }
            for _, r in top.iterrows()
        ],
        "dropped_for_redundancy": [{"column": c, "reason": why} for c, why in dropped[:40]],
    }
    (OUT / "tentrial_top30.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "| rank | feature | group | score | rho | BH q | folds | models | effect (on vs off) |",
        "| ---: | --- | --- | ---: | ---: | ---: | :---: | :---: | --- |",
    ]
    for _, r in top.iterrows():
        eff_txt = "—"
        if not pd.isna(r["diff"]):
            star = " *" if r["ci_excludes_zero"] else ""
            eff_txt = (f"{r['diff']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                       f"n={int(r['n_true'])}{star}")
        lines.append(
            f"| {int(r['rank'])} | `{r['label']}` | {r['group']} | {r['score']:.3f} | "
            f"{r['rho']:+.3f} | {r['q']:.2f} | {int(r['folds'])}/5 | "
            f"{int(r['models_agree'])}/3 | {eff_txt} |"
        )
    table = "\n".join(lines)
    (OUT / "tentrial_top30.md").write_text(table + "\n", encoding="utf-8")
    write_doc(top, table, pooled)
    print(table)
    churn = [r["label"] for _, r in top.iterrows() if not r["in_n1_top30"]]
    print(f"\nnew vs the n=1 top 30: {len(churn)} of {len(top)} — {', '.join(churn)}")
    print(f"wrote {OUT / 'tentrial_top30.json'}")


if __name__ == "__main__":
    main()
