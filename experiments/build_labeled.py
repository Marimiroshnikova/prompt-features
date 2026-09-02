"""Extract promptfeat features over the ReaLMistake and Mis-prompt labeled sets.

Writes one feature table per dataset into experiments/out/, keeping the metadata
columns needed for grouped evaluation (prompt identity, model, task, flaw type).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from promptfeat import extract_features  # noqa: E402

RF = Path(r"C:/Users/user/Desktop/Retrieval-Failure")
OUT = Path(__file__).resolve().parent / "out"


def load_realmistake() -> pd.DataFrame:
    """900 rows: 480 unique prompts x up to 2 responder models, expert-labeled."""
    rows = []
    for path in sorted(RF.glob("data/realmistake/*/*.jsonl")):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            meta = rec["metadata"]
            model = meta["llm_response_model"]
            # metadata id is "<task>_<n>_<model>"; strip the model to group the
            # same prompt shown to different responders.
            prompt_id = meta["id"]
            if prompt_id.endswith("_" + model.split("/")[-1]):
                prompt_id = prompt_id[: -(len(model.split("/")[-1]) + 1)]
            rows.append(
                {
                    "prompt": rec["input"],
                    "prompt_id": prompt_id,
                    "task": meta["task_name"],
                    "task_source": meta.get("task_source"),
                    "model": model,
                    "difficulty": meta.get("difficulty"),
                    "error_categories": "|".join(rec.get("error_categories") or []),
                    "y": int(rec["error_label"] == "error"),
                }
            )
    return pd.DataFrame(rows)


def load_misprompt() -> pd.DataFrame:
    """29,392 rows: flawed prompts vs the dataset's own 'correct' prompts."""
    df = pd.read_csv(RF / "data/misprompt_full.csv")
    return pd.DataFrame(
        {
            "prompt": df["question"].astype(str),
            "prompt_id": df["id"],
            "split": df["split"],
            "primary_category": df["primary_category"],
            "secondary_category": df["secondary_category"],
            "y": (df["error"] == "error").astype(int),
        }
    )


def add_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    t0 = time.time()
    records = []
    for i, text in enumerate(df["prompt"]):
        records.append(extract_features(text, with_status=True))
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {label}: {i + 1}/{len(df)}  ({rate:.0f}/s)", flush=True)
    feats = pd.DataFrame(records, index=df.index)
    feats.columns = ["f_" + c for c in feats.columns]
    print(f"  {label}: {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
    return pd.concat([df, feats], axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rm = load_realmistake()
    print(f"ReaLMistake: {len(rm)} rows, {rm.prompt_id.nunique()} unique prompts, "
          f"error rate {rm.y.mean():.3f}")
    add_features(rm, "realmistake").to_csv(
        OUT / "realmistake_features.csv", index=False
    )

    mp = load_misprompt()
    print(f"Mis-prompt: {len(mp)} rows, error rate {mp.y.mean():.3f}, "
          f"{mp.secondary_category.nunique()} flaw subtypes")
    add_features(mp, "misprompt").to_csv(OUT / "misprompt_features.csv", index=False)

    print("done")


if __name__ == "__main__":
    main()
