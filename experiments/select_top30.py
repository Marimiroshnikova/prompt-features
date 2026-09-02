"""Pick a Top 30 from MMLU question fail rate, not retrieval judgement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from evaluate import drop_correlated, prepare_matrix, univariate_spearman  # noqa: E402

OUT = REPO / "experiments" / "out"


def parent(col: str) -> str:
    name = col[2:] if col.startswith("f_") else col
    return name.split("=", 1)[0]


def main() -> None:
    rows = pd.read_csv(OUT / "mmlu_pilot_rows.csv", low_memory=False)
    exam = pd.read_csv(OUT / "mmlu_exam_features.csv")
    one = rows.drop_duplicates("question_id").copy()
    exam_cols = [c for c in exam.columns if c.startswith("f_")]
    one = one.merge(exam[["question_id"] + exam_cols], on="question_id", how="left")
    y = one["question_id"].map(rows.groupby("question_id")["y_fail"].mean()).to_numpy()
    X, _, _ = prepare_matrix(one)
    X = X.drop(columns=["f_context_pressure", "f_recency_gap"], errors="ignore")
    screen = univariate_spearman(X, y)
    kept_cols = drop_correlated(X, screen["feature"].tolist(), threshold=0.9)

    seen: set[str] = set()
    features = []
    for col in kept_cols:
        p = parent(col)
        if p in seen:
            continue
        seen.add(p)
        row = screen.loc[screen.feature == col].iloc[0]
        q = float(row.q) if pd.notna(row.q) else None
        features.append(
            {
                "name": p,
                "rank": len(features) + 1,
                "column": col,
                "spearman": round(float(row.spearman), 4),
                "q": None if q is None else round(q, 4),
                "significant": bool(row.significant),
            }
        )
        if len(features) == 30:
            break

    payload = {
        "source": "280 MMLU-Pro questions, fail rate across 14 models, n=1",
        "method": (
            "largest |Spearman| vs question fail rate; drop |corr| >= 0.9; "
            "one dummy per categorical; cap 30"
        ),
        "note": (
            "Better for this exam grid than the retrieval ranking. Still weak: "
            "most are not BH-significant. Does not beat model x subject."
        ),
        "features": features,
    }
    dest = REPO / "web" / "top30.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {dest} n={len(features)}")
    for item in features:
        print(f"{item['rank']:2} {item['name']:<32} rho={item['spearman']:+.3f} q={item['q']}")


if __name__ == "__main__":
    main()
