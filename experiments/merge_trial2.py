"""Merge trial-1 results.csv with a second GAIA run into n=2 columns.

Does not overwrite results.csv. Writes:
  results_trial2.csv          copy of the new run
  results_n2.csv              joined it_1 + it_2
  experiments/out/mmlu_pilot_rows.csv  rebuilt with n_trials=2
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mmlu_pilot import RESULTS, attach_specs, add_interactions, load_unique_questions  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TRIAL2_SRC_DEFAULT = Path(
    r"C:\Users\user\Desktop\GAIA_eval_trial2\results\runs\trial2_20260902\results.csv"
)
TRIAL2_COPY = REPO / "results_trial2.csv"
MERGED = REPO / "results_n2.csv"
PILOT = OUT / "mmlu_pilot_rows_n2.csv"
FEATS = OUT / "mmlu_prompt_features.csv"


def _require_complete(df: pd.DataFrame, label: str) -> None:
    if len(df) != 280 * 14:
        raise ValueError(f"{label}: expected 3920 rows, got {len(df)}")
    if df.duplicated(["llm_model", "question_id"]).any():
        raise ValueError(f"{label}: duplicate (model, question_id)")
    missing_ans = df["it_1_ans"].isna() | (df["it_1_ans"].astype(str).str.strip() == "")
    if int(missing_ans.sum()) > 0:
        print(f"WARN {label}: {int(missing_ans.sum())} rows with empty it_1_ans")


def merge(trial2_src: Path) -> pd.DataFrame:
    t1 = pd.read_csv(RESULTS, low_memory=False)
    t2 = pd.read_csv(trial2_src, low_memory=False)
    _require_complete(t1, "trial1")
    _require_complete(t2, "trial2")

    if Path(trial2_src).resolve() != TRIAL2_COPY.resolve():
        shutil.copy2(trial2_src, TRIAL2_COPY)

    keep = [
        "llm_model",
        "question_category",
        "question_id",
        "options",
        "correct_answer",
        "prompt",
        "it_1_ans",
        "it_1_raw",
        "correct_answered_num",
        "accuracy",
    ]
    a = t1[keep].copy()
    b = t2[["llm_model", "question_id", "it_1_ans", "it_1_raw", "correct_answer"]].copy()
    b = b.rename(columns={"it_1_ans": "it_2_ans", "it_1_raw": "it_2_raw", "correct_answer": "t2_gold"})

    m = a.merge(b, on=["llm_model", "question_id"], how="inner", validate="one_to_one")
    if len(m) != 3920:
        raise ValueError(f"inner join lost rows: {len(m)}")
    gold_mismatch = m["correct_answer"].astype(str) != m["t2_gold"].astype(str)
    if gold_mismatch.any():
        raise ValueError(f"gold letter mismatch on {int(gold_mismatch.sum())} rows")
    m = m.drop(columns=["t2_gold"])

    letters = set("ABCDEFGHIJ")
    gold = m["correct_answer"].astype(str).str.strip().str.upper()
    a1 = m["it_1_ans"].astype(str).str.strip().str.upper()
    a2 = m["it_2_ans"].astype(str).str.strip().str.upper()
    p1 = a1.isin(letters)
    p2 = a2.isin(letters)
    m["it_1_parse_ok"] = p1.astype(int)
    m["it_2_parse_ok"] = p2.astype(int)
    c1 = (p1 & (a1 == gold)).astype(int)
    c2 = (p2 & (a2 == gold)).astype(int)
    m["correct_answered_num"] = c1 + c2
    m["accuracy"] = m["correct_answered_num"] / 2.0
    m["flip"] = (p1 & p2 & (a1 != a2)).astype(int)
    m["n_trials"] = 2

    cols = [
        "llm_model",
        "question_category",
        "question_id",
        "options",
        "correct_answer",
        "prompt",
        "it_1_ans",
        "it_1_raw",
        "it_2_ans",
        "it_2_raw",
        "it_1_parse_ok",
        "it_2_parse_ok",
        "correct_answered_num",
        "accuracy",
        "flip",
        "n_trials",
    ]
    m[cols].to_csv(MERGED, index=False)
    return m


def rebuild_pilot(merged: pd.DataFrame) -> None:
    if not FEATS.exists():
        print("skip pilot rebuild: mmlu_prompt_features.csv missing")
        return
    feats = pd.read_csv(FEATS, low_memory=False)
    out = pd.DataFrame(
        {
            "llm_model": merged["llm_model"],
            "question_id": merged["question_id"].astype(int),
            "question_category": merged["question_category"],
            "correct_answer": merged["correct_answer"],
            "it_1_ans": merged["it_1_ans"],
            "it_2_ans": merged["it_2_ans"],
            "n_trials": 2,
            "y_fail": 1.0 - merged["accuracy"].astype(float),
            "flip": merged["flip"].astype(int),
        }
    )
    out["risk_target"] = out["y_fail"]
    feat_cols = ["question_id"] + [c for c in feats.columns if c.startswith("f_")]
    rows = out.merge(feats[feat_cols], on="question_id", how="left")
    rows = attach_specs(rows)
    rows = add_interactions(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(PILOT, index=False)
    print(f"pilot rows {len(rows)} fail_rate {rows.y_fail.mean():.4f} flip {rows.flip.mean():.4f}")


def write_report(merged: pd.DataFrame) -> None:
    p1 = merged["it_1_parse_ok"].astype(bool)
    p2 = merged["it_2_parse_ok"].astype(bool)
    both = p1 & p2
    lines = [
        "Trial 1 + Trial 2 merge",
        "n_trials = 2. Not the plan's 10.",
        "",
        f"rows {len(merged)}  questions {merged.question_id.nunique()}  models {merged.llm_model.nunique()}",
        f"mean accuracy (correct/2) {merged.accuracy.mean():.4f}",
        f"mean fail risk (1-accuracy) {(1-merged.accuracy).mean():.4f}",
        f"trial2 parsed letter {int(p2.sum())}/{len(merged)}",
        f"trial2 missing letter {int((~p2).sum())} (empty reply or parser miss; counted wrong)",
        f"both letters present {int(both.sum())}",
        f"letter flip among parsed pairs {float(merged.loc[both, 'flip'].mean()) if both.any() else 0:.4f}",
        "",
        "Flip rate by model (only rows with both letters):",
    ]
    tmp = merged.loc[both].copy()
    g = tmp.groupby("llm_model").agg(
        n=("flip", "size"),
        flip=("flip", "mean"),
        acc=("accuracy", "mean"),
    ).sort_values("flip", ascending=False)
    lines.append(g.to_string())
    lines.append("")
    lines.append("Flip rate by subject (only rows with both letters):")
    g2 = tmp.groupby("question_category").agg(
        n=("flip", "size"),
        flip=("flip", "mean"),
        acc=("accuracy", "mean"),
    ).sort_values("flip", ascending=False)
    lines.append(g2.to_string())
    path = OUT / "n2_merge_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else TRIAL2_SRC_DEFAULT
    print(f"merging trial2 from {src}")
    merged = merge(src)
    print(
        f"wrote {MERGED}  flip_rate={merged.flip.mean():.4f}  "
        f"acc={merged.accuracy.mean():.4f}"
    )
    rebuild_pilot(merged)
    write_report(merged)


if __name__ == "__main__":
    main()
