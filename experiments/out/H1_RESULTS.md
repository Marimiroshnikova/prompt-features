# H1: binary label vs soft fail rate

**Setup.** Question-grouped 5-fold cross-validation on 840 cells: 280 MMLU-Pro questions × 3 Gemini models × 10 trials.

- **Features.** Every model gets the same 144 columns: prompt features plus the model specs that actually vary between models.
  - `question_id` and the MMLU subject are not features.
  - No question sits in both a train fold and a test fold.
- **Model families.** The plan's Phase 4 asks for two:
  - XGBoost: a binary classifier vs a regressor on the fail rate;
  - elastic-net logistic: on the binary label vs on the 10 trials themselves. For the soft version, each cell becomes a fail row and a correct row, weighted by count and normalised so each cell weighs 1.
- **Targets.**
  - Soft: `fail_rate = n_fail / 10`.
  - Binary: `1` if `fail_rate > 0.5`, else `0`. That is 18.6% of cells.
- Every score is read as P(this answer is wrong).

Reproduce: `python experiments/run_h1.py`.

## Result

**H1 does not hold for either model family.** A soft target does not beat a binary one on Brier.

| predictor | Brier | BSS | ECE (15 quantile bins) | log loss | error at 80% coverage | AURC | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| base rate | 0.1632 | 0 | 0.044 | 0.508 | 21.3%* | 0.220 | 0.46 |
| XGBoost binary | 0.1652 | −0.012 | 0.063 | 0.516 | 18.6% | 0.170 | 0.58 |
| XGBoost soft | 0.1694 | −0.038 | 0.068 | 0.527 | 19.7% | 0.177 | 0.55 |
| logistic binary | 0.1632 | +0.000 | 0.053 | 0.505 | 18.7% | 0.151 | 0.61 |
| logistic soft | 0.1622 | +0.006 | 0.059 | 0.501 | 19.5% | 0.145 | 0.62 |

\* The base rate is nearly constant, so its skip order is arbitrary. The coverage drops below are measured against answering everything (20.5% error).

Question bootstrap, 1000 resamples:

- **Soft minus binary, Brier.**
  - XGBoost: +0.004, CI [−0.001, +0.010].
  - Logistic: −0.001, CI [−0.004, +0.002].
  - Both are ties.
- **Soft logistic vs base rate, Brier:** CI [−0.006, +0.005]. The best skill score (+0.006) is not distinguishable from zero.
- **Drop in error at 80% coverage.**
  - XGBoost binary: −1.8 points, CI [−3.4, −0.1].
  - Logistic binary: −1.9 points, CI [−3.5, −0.4].
  - Both soft drops include 0.
- **Skipping harder.** The logistic models rank best. The soft logistic reaches 15% error while answering 55% of questions, and 10% error at 24%. XGBoost binary reaches the same errors at 32% and 2%.
- **Calibration.** At least 96% of every model's predictions are below 0.4, although 11.7% of cells fail all 10 answers. The models hedge toward the average.
- **Label noise.** RMSE against the 10-trial rate is about 0.36, while the average binomial noise floor is 0.06. The miss is not 10-trial sampling noise.

We did not collect fresh 20–30 generations for the test questions. The split-half reliability of the current label is 0.975 (`TENTRIAL_FINDINGS.md`), so this null does not depend on that extra sample.

## Why the earlier notebooks disagreed

- Those notebooks scored RMSE on a row split. Replaying that split (seed 42) puts all 145 test questions inside the training set as well.
- They also left `question_id` in the feature matrix.
- On a question-out split, scored as a probability of being wrong, the regressor's RMSE advantage disappears.
