# Ten trials per cell — what the features actually did

**Data:** 280 MMLU-Pro questions × 3 Gemini models × 10 answers = **8,400 answers**
(`results copy.csv`). This is the run `BUDGET_10_TRIALS.md` asked for. Every earlier
report in this folder was written on **1** answer per cell, where "risk" could only be
0 or 1.

**Reproduce:**

```bash
python experiments/run_10trial.py               # selection + head-to-head
python experiments/run_10trial_diagnostics.py   # is there signal, could we see it
python experiments/build_10trial_shortlist.py   # the ranked candidate table
```

---

## 1. The one-line answer

The label is now trustworthy, and **question difficulty is real and large** — it explains
**70%** of the variation in fail rate, against 16% for model × subject. But **none of the
138 prompt features find it.** Zero survive BH correction, and adding them to the
model × subject baseline makes it *worse* than shuffling the labels.

The result is not "features are useless." It is "**there is a big prize, these features
are not the key, and 280 questions is too few to tell a real 0.16 correlation from noise.**"

---

## 2. What the 10th trial bought

| | |
|---|---|
| Overall fail rate | 20.5% (unchanged from the 1-trial run) |
| Cells always right (0/10) | 66.2% |
| Cells always wrong (10/10) | 11.7% |
| **Cells in between** | **22.1%** ← completely invisible at n=1 |
| Unparsed replies | 44 / 8,400 (0.52%), 43 of them `gemini-2.5-flash-lite` |

**Split-half reliability** (trials 1–5 vs 6–10, per question): Pearson **r = 0.975**,
Spearman-Brown for the full 30-trial label **0.988**.

That number is the whole point of the extra spend. At n=1 the label was a coin flip and
no feature could have correlated with it above chance no matter how good it was. Now the
label is essentially noise-free, so a null result means something.

Fail rate by model: `gemini-2.5-flash-lite` 0.280 · `gemini-3.1-flash-lite` 0.216 ·
`gemini-flash-latest` 0.118.
Hardest subjects: health 0.475, law 0.405, engineering 0.340. Easiest: economics 0.070,
math 0.073, biology 0.078.

---

## 3. There is question-level signal, and it is big

The three models **agree on which questions are hard**:

| pair | Spearman |
|---|---|
| 2.5-flash-lite vs 3.1-flash-lite | 0.579 |
| 2.5-flash-lite vs flash-latest | 0.505 |
| 3.1-flash-lite vs flash-latest | 0.640 |

Variance decomposition over the 840 cells (total variance 0.1270):

| grouping | variance explained |
|---|---|
| model | 3.5% |
| subject | 11.6% |
| model × subject | 16.4% |
| **question** | **70.3%** |
| 10-trial sampling noise | 2.8% |

**Oracle ceiling.** Give the predictor a perfect measurement of a question's difficulty —
taken from the *other two* models' answers, so no cell sees its own label:

| predictor | Brier | AUC |
|---|---|---|
| model × subject | 0.1534 | 0.677 |
| **+ oracle question difficulty** | **0.1039** | **0.878** |
| irreducible 10-trial noise floor | 0.0360 | — |

**0.0495 of Brier is sitting on the table.** That is the prize. A text feature that
worked would claim some of it.

---

## 4. What the features claimed: nothing

Question-grouped 5-fold CV on cells, weighted by the 10 trials behind each cell.
Feature selection happens **inside the training fold**.

| predictor | Brier | log loss | AUC |
|---|---|---|---|
| global mean | 0.1637 | 0.5093 | 0.442 |
| model only | 0.1593 | 0.4955 | 0.590 |
| subject only | 0.1562 | 0.4942 | 0.645 |
| **model × subject** | **0.1534** | **0.4847** | **0.677** |
| features only | 0.1775 | 0.5684 | 0.476 |
| length only | 0.1643 | 0.5114 | 0.456 |
| model × subject + fold-selected features | 0.1782 | 0.5718 | 0.601 |
| model × subject + stable features | 0.1650 | 0.5255 | 0.645 |

Every feature variant **loses** to model × subject. Worse, the permutation check
(question labels shuffled, 20 draws) gives a null Brier of **0.1610** for the same
architecture — the real features score **0.1650**, i.e. *below their own null*. Fifteen
features on 672 training cells is overfitting, not signal.

Abstention does not rescue them either: dropping the riskiest 20% of cells by
model × subject takes error 20.5% → 16.0%; by model × subject + features, only → 19.5%.

**Univariate screen:** 0 of 139 features clear BH at q < 0.05 against `q_fail_rate`.
The largest |ρ| is 0.165 (`f_max_word_len`, `f_question_type=what`).

---

## 5. Was that a null, or too few questions?

With n = 280 and 139 simultaneous tests, the smallest |ρ| that can clear BH is **0.211**.
The largest |ρ| actually present is **0.165**.

**The study could not have detected the effects that are there.** To call ρ = 0.165
significant at 80% power needs **≈ 705 questions**.

So the plan's 1,500–3,000 questions is not a stretch goal; it is the minimum for this
question to be answerable. Ten trials fixed the *label*. It did not fix the *sample*.

---

## 6. The best features, ranked

Three independent tests, and a feature has to survive more than one:
**stability** (picked in ≥3 of 5 question folds, selection inside the fold),
**pooled ρ** (Spearman vs `q_fail_rate`, leaky — ranking only), and
**effect size** (fail rate on vs off, bootstrap CI over questions).

| tier | feature | folds | ρ | BH q | n | fail on | fail off | diff | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | `f_is_long_scenario` | 4/5 | +0.134 | 0.39 | 43 | 0.298 | 0.188 | +0.109 | [+0.011, +0.219] |
| A | `f_aggregation_flag` | 3/5 | +0.136 | 0.39 | 92 | 0.267 | 0.175 | +0.092 | [+0.010, +0.168] |
| A | `f_is_best_answer_judgment` | 3/5 | +0.127 | 0.47 | 15 | 0.418 | 0.193 | +0.225 | [+0.017, +0.429] |
| B | `f_question_type=what` | 5/5 | −0.165 | 0.37 | — | — | — | — | — |
| B | `f_max_word_len` | 5/5 | +0.165 | 0.37 | — | — | — | — | — |
| B | `f_is_definition_ask` | 5/5 | −0.153 | 0.37 | 24 | 0.154 | 0.210 | −0.056 | [−0.179, +0.088] |
| B | `f_min_word_zipf` | 4/5 | −0.152 | 0.37 | — | — | — | — | — |
| B | `f_domain_hint=legal` | 4/5 | +0.147 | 0.38 | — | — | — | — | — |
| B | `f_stem_word_count` | 4/5 | +0.134 | 0.39 | — | — | — | — | — |
| B | `f_rare_word_count` | 3/5 | +0.134 | 0.39 | — | — | — | — | — |
| B | `f_oov_like_count` | 3/5 | +0.122 | 0.53 | — | — | — | — | — |
| B | `f_has_relative_recency` | 1/5 | −0.055 | 0.79 | 28 | 0.111 | 0.215 | −0.105 | [−0.175, −0.024] |
| C | `f_is_except_ask` | 2/5 | +0.100 | 0.71 | 11 | 0.427 | 0.196 | +0.231 | [−0.018, +0.489] |

- **Tier A** — stable across folds *and* the bootstrap CI excludes zero. Carry these into
  the larger run. These are the three to name in a talk.
- **Tier B** — stable, but at n = 280 the effect cannot be resolved. Keep measuring; do
  not ship.
- **Tier C** — largest effect in the whole set (`f_is_except_ask`, +0.231) sitting on
  **11 questions**. This needs more questions, not more modelling.

Note the shape of Tier A: they are all **exam-format** flags — a long scenario, a question
that asks you to aggregate, a "which best describes" judgment call. The retrieval-style
features that dominate the top-30 ranking in `FEATURES.md` (`anchor_count`,
`is_ambiguous`, `unresolved_pronoun_count`) are **absent**. Those were designed to predict
retrieval failure on open queries, and an MMLU-Pro item with ten printed options has
nothing to retrieve. That is a spec mismatch, not a bug.

The strongest per-model signal is on the **best** model: on `gemini-flash-latest`,
`f_is_best_answer_judgment` reaches ρ = 0.204 (q = 0.08) and `f_has_escape_option`
ρ = 0.185. The weak models fail on knowledge, so wording barely matters; the strong model
has the knowledge and gets caught by the *format*. That is a real, defensible sentence and
it is new with this data.

---

## 7. One thing 10 trials makes newly askable

91 of 280 questions sit between 10% and 90% risk — genuinely unstable, not merely hard.
Predicting *instability* would be independently useful (route the coin-flips to
self-consistency, skip the certain misses). **0 of 139 features are BH-significant for it.**
Best candidates, all short of significance: `f_max_word_len` (+0.170),
`f_is_definition_ask` (−0.158), `f_question_type=what` (−0.148).

---

## 8. What to do next, in order

1. **More questions, not more features.** 705 minimum to resolve what is already visible;
   1,500 to have any room for a trap-flag subgroup. Cost scales the same way
   `BUDGET_10_TRIALS.md` computed — 3 models × 10 trials × 1,500 questions ≈ 45,000 calls.
2. **Oversample the trap items.** `f_is_except_ask` has 11 instances and the largest
   effect in the table. Deliberately sample 100+ except/NOT and "best describes" items;
   that alone could promote Tier C to Tier A.
3. **Write features for the option block, not the stem.** 70% of the variance is
   question-level and the current features barely look at the ten options. Candidates:
   distractor–answer semantic distance, numeric spread across options, how many options
   are defensible, whether the correct answer is the longest.
4. **Do not train Phase 4 (trees / XGBoost) yet.** On 280 questions the feature model is
   already scoring below its own permutation null. A bigger model makes that worse.
5. **Add the other 11 models back.** This run used 3 of the 14. Model coverage costs
   nothing in questions and would sharpen the "strong models fail on format" finding.

---

## Files

| file | what |
|---|---|
| `tentrial_report.txt` | full selection + head-to-head log |
| `tentrial_diagnostics.txt` | reliability, agreement, variance, oracle, power |
| `tentrial_shortlist.csv` / `.md` | the ranked table in §6 |
| `tentrial_univariate_qfail.csv` | all 139 features vs `q_fail_rate` |
| `tentrial_univariate_<model>.csv` | the same screen per model |
| `tentrial_univariate_unstable.csv` | features vs instability |
| `tentrial_effect_sizes.csv` | on/off contrasts with bootstrap CIs |
| `tentrial_question_table.csv` | 280 questions, label + all features |
| `tentrial_cells.csv` | 840 cells, `n_correct` / `n_fail` |
| `tentrial_results.json`, `tentrial_diagnostics.json` | every number above, machine-readable |

**Caveats.** 3 models, not 14. The 44 unparsed replies are counted as wrong; dropping
them instead moves `q_fail_rate` by at most 0.133 on a single question and leaves the
question ranking at Spearman 0.986, so it changes nothing here. The oracle in §3 is a
ceiling, not a deployable predictor — it reads other models' answers to the same question.
