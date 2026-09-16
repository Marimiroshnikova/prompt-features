"""Full audit of the deliverable workbook. Every check must print PASS."""

import datetime
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PATH = REPO / "experiments" / "out" / "mmlu_10trial_dataset.xlsx"
SRC = REPO / "experiments" / "out" / "reduced_10trial.csv"
SAMPLE = REPO / "data" / "mmlu_pro_sample_20_per_category.csv"
RUN_COLS = [f"it_{i}" for i in range(1, 11)]
INTER = ["f_context_pressure", "f_output_pressure", "f_recency_gap",
         "f_complexity_x_capability"]

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


size_kb = round(os.path.getsize(PATH) / 1024)
mtime = datetime.datetime.fromtimestamp(os.path.getmtime(PATH))
print(f"{PATH}\n  {size_kb} KB, written {mtime:%Y-%m-%d %H:%M}\n")

book = pd.ExcelFile(PATH)
q = pd.read_excel(PATH, sheet_name="Questions")
fdict = pd.read_excel(PATH, sheet_name="Feature_Dictionary")
readme = pd.read_excel(PATH, sheet_name="README")
src = pd.read_csv(SRC, low_memory=False)
src["question_id"] = src.question_id.astype(int)
sample = pd.read_csv(SAMPLE)

print("STRUCTURE")
check("three sheets: README, Questions, Feature_Dictionary",
      book.sheet_names == ["README", "Questions", "Feature_Dictionary"],
      str(book.sheet_names))
check("840 rows = 280 questions x 3 models",
      len(q) == 840 and q.question_id.nunique() == 280 and q.llm_model.nunique() == 3,
      f"{len(q)} rows, {q.question_id.nunique()} questions, {q.llm_model.nunique()} models")
check("every question has exactly 3 rows, one per model",
      set(q.groupby("question_id").size()) == {3}
      and set(q.groupby("question_id").llm_model.nunique()) == {3})
check("no duplicate question x model pair",
      not q.duplicated(["question_id", "llm_model"]).any())
check("no duplicate column names", len(set(q.columns)) == len(q.columns))
check("rows sorted question then model",
      q.equals(q.sort_values(["question_id", "llm_model"]).reset_index(drop=True)))

print("\nQUESTION CONTENT")
gold = sample.set_index("question_id")
check("question text matches the MMLU-Pro sample",
      (q.question.astype(str).values
       == q.question_id.map(gold.question).astype(str).values).all())
check("gold letter matches the MMLU-Pro sample",
      (q.correct_answer.astype(str).values
       == q.question_id.map(gold.answer).astype(str).values).all())
check("options present on every row",
      q.options.notna().all() and q.options.astype(str).str.startswith("[").all())
check("14 subjects x 20 questions",
      q.drop_duplicates("question_id").question_category.value_counts().unique().tolist() == [20],
      f"{q.question_category.nunique()} subjects")

print("\nRUN RESULTS")
m = q.merge(src[["question_id", "llm_model", "answers", "n_correct", "n_blank"]],
            on=["question_id", "llm_model"], suffixes=("", "_src"))
letters = m[RUN_COLS].astype(str).agg("".join, axis=1)
recount = sum((m[c].astype(str) == m.correct_answer.astype(str)).astype(int) for c in RUN_COLS)
check("no empty run cell", int(q[RUN_COLS].isna().sum().sum()) == 0)
check("it_1..it_10 reproduce the source answer string",
      bool((letters == m.answers.astype(str)).all()))
check("n_correct equals the source", bool((m.n_correct == m.n_correct_src).all()))
check("n_correct equals a fresh recount against the gold letter",
      bool((recount == m.n_correct).all()))
check("n_fail = 10 - n_correct", bool((m.n_fail == 10 - m.n_correct).all()))
check("fail_rate = n_fail / 10", bool((m.fail_rate == m.n_fail / 10).all()))
check("n_blank equals the count of '-' letters",
      bool(((m[RUN_COLS].astype(str) == "-").sum(axis=1) == m.n_blank).all()))
check("every letter is A-J or '-'",
      set(pd.unique(q[RUN_COLS].astype(str).values.ravel())) <= set("ABCDEFGHIJ-"))

print("\nFEATURE GROUPS")
fcols = [c for c in q.columns if c.startswith("f_")]
qlevel = [c for c in fcols if c not in INTER]
model_cols = ["model_family", "is_preview", "is_open_source", "has_custom_tools",
              "context_window_tokens", "knowledge_cutoff_year",
              "max_tokens_requested", "output_token_limit", "temperature", "top_p"]
check("question features identical across a question's 3 rows",
      int((q.groupby("question_id")[qlevel].nunique(dropna=False) > 1).sum().sum()) == 0,
      f"{len(qlevel)} columns")
check("model features constant within a model",
      int((q.groupby("llm_model")[model_cols].nunique(dropna=False) > 1).sum().sum()) == 0)
check("no row is missing every feature", int(q[fcols].isna().all(axis=1).sum()) == 0)
check("booleans stored as 0/1, never TRUE/FALSE",
      not any(set(map(str, q[c].dropna().unique())) & {"True", "False"} for c in q.columns))

print("\nYEAR / TEMPORAL FIX")
one = q.drop_duplicates("question_id")
check("no year outside 1000-2099",
      bool(one.f_year_max.dropna().between(1000, 2099).all()),
      f"{int(one.f_year_max.notna().sum())} questions name a year")
check("recency gap only where both year and cutoff exist",
      int(q.f_recency_gap.notna().sum())
      == int((q.f_year_max.notna() & q.knowledge_cutoff_year.notna()).sum()),
      f"{int(q.f_recency_gap.notna().sum())} rows")
check("recency gap = f_year_max - knowledge_cutoff_year",
      bool((q.f_recency_gap.dropna()
            == (q.f_year_max - q.knowledge_cutoff_year).dropna()).all()))
check("year_span = year_max - year_min",
      bool((one.f_year_span.dropna() == (one.f_year_max - one.f_year_min).dropna()).all()))
check("the three quantity questions no longer name a year",
      bool((one.set_index("question_id").loc[[9582, 11315, 8068], "f_year_count"] == 0).all()))
check("the dated history passages kept their years",
      one.set_index("question_id").loc[4923, "f_year_max"] == 1346
      and one.set_index("question_id").loc[4994, "f_year_max"] == 1500)

print("\nDICTIONARY")
listed = set(fdict.column.astype(str))
described = {c for c in q.columns if c in listed}
missing = [c for c in q.columns if c not in listed and not c.startswith("it_")]
check("every column is described (it_1..it_10 as one entry)",
      not missing and "it_1 ... it_10" in listed, f"missing: {missing}" if missing else "")
check("no dictionary entry refers to a column that is gone",
      all(c in q.columns or c == "it_1 ... it_10" for c in listed))
check("every dictionary row has a group and a meaning",
      fdict.group.notna().all() and fdict["what it measures"].notna().all())
check("README is filled in", len(readme) > 0 and readme.iloc[:, 1].notna().all())

print("\nINTERACTION COLUMNS (documented as flat, not a defect)")
for col in INTER:
    print(f"  {col:28s} distinct values: {q[col].nunique(dropna=False)}")

print("\nSAMPLE ROW (question 507, all three models)")
show = ["question_id", "llm_model", "correct_answer", *RUN_COLS[:4], "n_correct",
        "fail_rate", "f_year_max", "f_recency_gap", "f_mean_word_zipf"]
print(q[q.question_id == 507][show].to_string(index=False))

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + str(failures)}")
