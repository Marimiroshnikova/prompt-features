"""Combine the three lines of evidence into one ranked candidate list.

A feature earns a place by surviving more than one test:
  stability   picked in >= 3 of 5 question folds, selection done inside the fold
  pooled rho  Spearman against q_fail_rate over all 280 questions (leaky, ranking only)
  effect      fail rate on vs off, with a bootstrap CI over questions

Tier A  stable AND the CI excludes zero        -> carry these forward
Tier B  stable, effect not resolvable at n=280 -> keep measuring, do not ship
Tier C  large effect, too few items to trust   -> needs more questions, not more math
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import OUT  # noqa: E402

MIN_FOLDS = 3


def main() -> None:
    res = json.loads((OUT / "tentrial_results.json").read_text())
    diag = json.loads((OUT / "tentrial_diagnostics.json").read_text())
    uni = pd.read_csv(OUT / "tentrial_univariate_qfail.csv").set_index("feature")
    eff = pd.DataFrame(diag["effect_sizes"]).set_index("flag")

    stable = res["stable_features"]
    fold_counts = res["stability_counts"]
    rows = []
    names = (set(stable) | set(eff.index[eff.excludes_zero])
             | set(eff.index[eff["diff"].abs() >= 0.15]))
    for f in names:
        folds = int(fold_counts.get(f, 0))
        u = uni.loc[f] if f in uni.index else None
        e = eff.loc[f] if f in eff.index else None
        excl = bool(e["excludes_zero"]) if e is not None else False
        if folds >= MIN_FOLDS and excl:
            tier = "A"
        elif folds >= MIN_FOLDS:
            tier = "B"
        elif excl:
            tier = "B"
        else:
            tier = "C"
        if e is not None and abs(e["diff"]) > 0.2 and not excl:
            tier = "C"
        rows.append({
            "feature": f, "tier": tier, "folds": folds,
            "pooled_rho": float(u["spearman"]) if u is not None else float("nan"),
            "bh_q": float(u["q"]) if u is not None else float("nan"),
            "n_true": int(e["n_true"]) if e is not None else -1,
            "fail_on": float(e["fail_true"]) if e is not None else float("nan"),
            "fail_off": float(e["fail_false"]) if e is not None else float("nan"),
            "diff": float(e["diff"]) if e is not None else float("nan"),
            "ci_lo": float(e["ci_lo"]) if e is not None else float("nan"),
            "ci_hi": float(e["ci_hi"]) if e is not None else float("nan"),
        })
    df = pd.DataFrame(rows)
    df["absrho"] = df.pooled_rho.abs()
    df = df.sort_values(["tier", "folds", "absrho"], ascending=[True, False, False])
    df = df.drop(columns="absrho")
    df.to_csv(OUT / "tentrial_shortlist.csv", index=False)

    def fmt(v, spec="{:+.3f}"):
        return "—" if pd.isna(v) else spec.format(v)

    lines = ["| tier | feature | folds | rho | BH q | n | fail on | fail off | diff | 95% CI |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        d = r["diff"]
        ci = "—" if pd.isna(r.ci_lo) else f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
        lines.append(
            f"| {r.tier} | `{r.feature}` | {r.folds}/5 | {fmt(r.pooled_rho)} | "
            f"{'—' if pd.isna(r.bh_q) else f'{r.bh_q:.2f}'} | "
            f"{'—' if r.n_true < 0 else int(r.n_true)} | {fmt(r.fail_on, '{:.3f}')} | "
            f"{fmt(r.fail_off, '{:.3f}')} | {fmt(d)} | {ci} |"
        )
    (OUT / "tentrial_shortlist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT / 'tentrial_shortlist.csv'}")


if __name__ == "__main__":
    main()
