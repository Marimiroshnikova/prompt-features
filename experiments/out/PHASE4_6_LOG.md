# Phases 4 to 6 — what we did and what we found

**Slides:** open `experiments/out/phase4_6_deck.html` in a browser (arrow keys to move). Every number in the deck is read from the three JSON files below.

## The answer in four lines

- **H1 does not hold.** For both XGBoost and elastic-net logistic, predicting the exact fail rate is no better than a yes/no "fail rate > 0.5" classifier. None of the four beats always guessing the average on Brier.
- **H2 to H4:** adding the model columns to the prompt features helps a little. No learned model beats the simple "this model on this subject" fail rate, on any split.
- **Skipping risky questions helps.** Skipping the riskiest 20% with a binary score takes error from 20.5% to 18.6%. When skipping more, the soft logistic ranks best: 15% error while still answering 55% of questions.
- **Step 2:** with 280 questions the best prompt feature is indistinguishable from noise. About 700 to 1,000 questions would be enough to see a real signal of that size. For 3 models × 10 answers, that costs about $21.

## 1. Starting point

- Checked GitHub first. Local `main` matched `origin/main` at `15b85f0`, and there were no open PRs.
- Data: 280 MMLU-Pro questions × 3 Gemini models (`gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-flash-latest`) × 10 answers each.
  - That is 840 cells and 8,400 graded answers.
  - 20.5% of the answers were wrong.
  - Target per cell: fail rate = wrong answers ÷ 10.
- The teammate's feature table (from `gaia-ml-share`) is copied to `data/gaia_training_dataset.csv`. Its `target` column is `1 − fail_rate`. The scripts check this on load.

## 2. The teammate notebooks

`gaia-ml-share` had two XGBoost notebooks.

| notebook | split | reported result |
|---|---|---|
| binary, rate > 0.5 | grouped by question | AUC 0.44 on holdout, not usable |
| regression, exact rate | **by row** | RMSE 0.29 vs 0.36 for the mean, R² 0.32 |

We found two problems with the regression:

- Replaying its row split (seed 42) shows that **all 145 test questions were also in training**. Prompt features are the same for all three models answering a question. So a test row is a question the model has already seen, just with a different model name.
- `question_id` was left in as a feature in both notebooks.

From here on, every test holds out whole questions. Neither `question_id` nor the MMLU subject is ever a feature; the subject is only used as a baseline.

## 3. Metrics (Phase 6), locked before testing

These come from `metrics.docx`. Which number decides depends on how the product uses the score.

- **The product skips the riskiest X%.** Only the ranking matters. Decide on the coverage–risk curve: the error at 90%, 80% and 70% coverage, the coverage reached at a target error, and AURC.
- **The product uses a threshold, or shows the number to the user.** The probability itself must be right. Decide on Brier, Brier Skill Score (BSS = 1 − Brier ÷ Brier of always guessing the average), the reliability diagram, and ECE on 15 quantile bins.

Also reported every time:

- log loss, clipped to [1e−4, 1 − 1e−4];
- AUROC, and AUPRC shown next to its base rate;
- a question-level bootstrap interval (1,000 resamples);
- results by model and by subject.

RMSE is shown only next to its noise floor.

Brier and log loss are weighted by the trials behind each cell, so each of the 10 answers counts once.

## 4. H1: binary label vs soft fail rate

`experiments/run_h1.py` produces `h1_metrics.json`. Details are in `H1_RESULTS.md`.

The setup is question-grouped 5-fold cross-validation with the same 144 features for every model. The plan asks for two model families:

- XGBoost, kept small: 80 trees, depth 3;
- elastic-net logistic: L1 ratio 0.5, C 0.05, with standardised numeric features and one-hot categoricals. The soft logistic is fit on each cell's fail and correct trials, weighted so each cell counts once.

| predictor | Brier | BSS | ECE | AURC | error at 80% coverage |
|---|---:|---:|---:|---:|---:|
| always guess the average | 0.163 | 0 | 0.044 | 0.220 | — |
| XGBoost binary | 0.165 | −0.012 | 0.063 | 0.170 | 18.6% |
| XGBoost soft | 0.169 | −0.038 | 0.068 | 0.177 | 19.7% |
| logistic binary | 0.163 | +0.000 | 0.053 | 0.151 | 18.7% |
| logistic soft | 0.162 | +0.006 | 0.059 | 0.145 | 19.5% |

- **Soft minus binary, Brier.** XGBoost +0.004, CI [−0.001, +0.010]; logistic −0.001, CI [−0.004, +0.002]. Both are ties.
- **Best Brier.** The soft logistic, but its CI against the base rate is [−0.006, +0.005]. That is zero skill.
- **Error at 80% coverage.** Both binary scores lower it by about 1.8–1.9 points, and both CIs exclude 0: real but small. The soft scores' drops include 0.
- **Skipping more.** The soft logistic reaches 15% error at 55% coverage and 10% at 24%. XGBoost binary reaches them at 32% and 2%.
- **Calibration** (reliability diagram, slide 6). At least 96% of predictions are below 0.4, although 11.7% of cells fail all 10 answers. The models hedge toward the average.
- RMSE is 0.36 against a noise floor of 0.06. The 10-trial label is reliable (split-half r = 0.975), so this null is not caused by a noisy label.

## 5. Step 1: H2, reduced H3, H4

`experiments/run_h2_h4.py` produces `h2_h4_metrics.json`.

It compares three feature sets (called arms), each with all four H1 models (XGBoost and logistic, binary and soft):

- prompt only (140 features);
- prompt + model id (141);
- all features (144): adds the knowledge cutoff and the recency gap.

Each arm is compared with four rate baselines: global, per model, per subject, and model × subject.

On unseen questions (question-out split):

| predictor | XGBoost binary | logistic soft |
|---|---:|---:|
| global mean | 0.163 | |
| subject mean | 0.156 | |
| model × subject | **0.155** | |
| prompt only | 0.171 | 0.164 |
| prompt + model id | 0.168 | 0.162 |
| all features | 0.165 | 0.162 |

- **H2:** prompt-only minus all features, Brier: XGBoost binary +0.006, CI [+0.003, +0.009]; logistic soft +0.002, CI [+0.0003, +0.003]. Both exclude 0, so the model columns help, a little.
- **Reduced H3:** adding the model id helps. Adding knowledge cutoff and recency on top helps XGBoost a bit, but not the logistic. The real H3 cannot run: context window, temperature, top-p and output pressure are identical for all three models.
- **H4:** the best learned arm (lowest Brier among 12) against model × subject, per split:

| split | model × subject | best learned | best arm | gap, 95% CI |
|---|---:|---:|---|---|
| unseen questions (5 folds) | 0.155 | 0.162 | prompt + model id, logistic soft | [−0.001, +0.017] |
| unseen subject (14 folds) | 0.161 | 0.167 | prompt + model id, XGBoost binary | [−0.001, +0.012] |
| unseen model and questions (15 folds) | 0.162 | 0.171 | prompt only, logistic binary | [−0.000, +0.019] |

- Every gap is positive (the learned model is worse), and every interval barely touches 0. Nothing beats the rate table.
- On ranking (error at 80% coverage), model × subject beats the best learned arm by 3.7 points on unseen questions, CI [+1.6, +6.1].

On unseen subjects, model × subject falls back to the model rate. On an unseen model, it falls back to the subject rate.

**A leak we caught.** Holding out a model without also holding out its questions gave prompt-only a Brier of 0.130, which looks excellent. It is not real: the other two models' answers to the same questions were in training, and the models agree on which questions are hard. The model-out split is therefore crossed with the question folds.

## 6. Step 2: how many questions we need (no API spend)

`experiments/run_power.py` produces `power_metrics.json`.

- We tested 131 numeric prompt features against each question's fail rate.
  - The strongest is `f_max_word_len`, with Spearman ρ = 0.165.
  - With 280 questions and 131 tests, a feature needs |ρ| ≥ 0.211 to pass the multiple-testing bar.
- Shuffling the labels 500 times produces a feature at least that strong **34%** of the time. The 95th percentile of those shuffled maxima is 0.197.
- When we bootstrap-resample the questions, the top-10 feature list keeps only 41% of its members.
- If ρ = 0.165 is real, **700 questions** would find it 80% of the time, and 1,000 questions 96% of the time.

| questions | ρ needed | chance to detect 0.165 | API calls | cost of new questions |
|---:|---:|---:|---:|---:|
| 280 | 0.211 | 22% | 10,920 | $0 |
| 500 | 0.158 | 56% | 19,500 | $7 |
| 1,000 | 0.112 | 96% | 39,000 | $21 |
| 1,500 | 0.092 | 100% | 58,500 | $36 |
| 3,000 | 0.065 | 100% | 117,000 | $81 |

How the costs are computed:

- 3 models × 10 answers per question; the sealed 20% gets 25 answers.
- About 230 input and 140 output tokens per call, times 2.2 for thinking tokens and retries.
- Prices per 1M input/output tokens, from Google in September 2026: 2.5 Flash-Lite $0.10/$0.40, 3.1 Flash-Lite $0.25/$1.50, Flash latest $0.75/$3.75. The "latest" alias can change.

## 7. What next

1. Collect 1,000 questions (about 71 per subject) × the same 3 models × 10 answers. That is roughly 39,000 calls and about $21.
2. Seal 20% of them before any feature choice, with 25 answers each.
3. Rerun the four scripts without changes.
4. For H3, add a second temperature on part of the set.

A feature set is promoted only if its 95% interval beats model × subject on the sealed set, on the metric that matches how the product uses the score.

## Checklist against `plan (1).docx`

| plan item | status |
|---|---|
| Phase 4/5, H1: binary vs soft target, same features, question-grouped CV | done |
| Two model families (XGBoost and elastic-net logistic) | done |
| Baselines: global, model, subject, model × subject | done |
| Splits: question-out, subject-out, model-out | done; model-out is crossed with question folds to stop a leak |
| H2 (prompt only vs all features) | done |
| H3 (configuration features) | reduced version only: the three models share one configuration, so there is nothing to vary |
| H4 (learned vs model × subject) | done on all three splits |
| Configuration-out split | not possible for the same reason as H3 |
| 20–30 fresh generations on test questions | not collected: the 10-trial label already has split-half reliability 0.975; planned for the sealed set in the next round |
| Sealed 20% holdout | not done: 280 questions is too few to split again; planned for the 1,000-question round |
| Random forest | optional in the plan; skipped |
| Phase 6 metrics: coverage–risk, AURC, Brier, BSS, reliability diagram, ECE, log loss, AUROC/AUPRC, bootstrap CIs, by-model/by-subject | done |

## Files

| file | what |
|---|---|
| `data/gaia_training_dataset.csv` | teammate feature table (840 rows) |
| `experiments/h_metrics.py` | data loading, folds, baselines, locked metrics, bootstrap |
| `experiments/run_h1.py` | H1 → `out/h1_metrics.json` |
| `experiments/run_h2_h4.py` | H2, reduced H3, H4 → `out/h2_h4_metrics.json` |
| `experiments/run_power.py` | Step 2 → `out/power_metrics.json` |
| `experiments/build_deck.py` | slides → `out/phase4_6_deck.html` |
| `experiments/out/H1_RESULTS.md` | H1 detail |
| `experiments/out/H2_H4_PLAN.md` | metric rule and H2–H4 design, written before running |

Reproduce:

```bash
python experiments/run_h1.py
python experiments/run_h2_h4.py
python experiments/run_power.py
python experiments/build_deck.py
```
