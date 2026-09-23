# Metrics lock, then H2–H4

Locked before any further model choice. H1 results: `H1_RESULTS.md`. H2–H4 were then run as specified here; results are in `PHASE4_6_LOG.md` and `h2_h4_metrics.json`. One change: model-out is crossed with the question folds, because holding out the model alone leaked each question's difficulty from the other two models.

## Which number decides

Report all of these on every comparison. Which one decides depends on how the score is used.

Always:

- Brier score on the trial outcomes, lower better.
- Brier Skill Score against the training-fold base rate. BSS = 1 − Brier_model / Brier_base. Above 0 means a real improvement on the probability.
- Coverage–risk curve, summarized as numbers: error among the items still answered at 90%, 80%, and 70% coverage; coverage achieved at a target error of 15% and 10%; area under that curve (mean selective risk).

If the product abstains on the riskiest X% and does not show a probability, the coverage numbers are the decision. A score of 0.90 on hard items and 0.80 on easy items wins this curve and fails calibration. Ranking is the whole task.

If the product refuses at a threshold (“skip when risk > 30%”) or shows the user a risk number, Brier, BSS, a reliability diagram with quantile bins and a count histogram, and ECE on a fixed scheme of 15 quantile bins are the decision. The threshold is a probability, so a score that only ranks is not enough.

Secondary, every time: log loss with probabilities clipped to [1e−4, 1−1e−4]; AUROC; AUPRC with the positive-class base rate printed beside it (here, 18.6% of cells have fail rate > 0.5).

RMSE against the 10-trial rate only with the noise floor printed next to it (0.06 on this grid). It is not a decision metric.

Every comparison also reports the base-rate baseline, a question-level bootstrap CI, and the same metrics by model and by MMLU subject.

A feature set is promoted only when the CI on the decision metric excludes “no better than the strongest baseline.”

## H2 — prompt only vs all features

Same question-grouped 5-fold, same two heads as H1 (the null did not pick one), no feature selection.

- Prompt only. Drop `llm_model`, `model_family`, `knowledge_cutoff_year`, `f_recency_gap`.
- All features used in H1. MMLU subject stays out. It is the H4 baseline, not a prompt feature.

Decision: the locked rule above, against the base rate and against each other.

## H3 — model limits and interactions

Not identifiable on this grid. These columns do not vary:

- `context_window_tokens` is 1,048,576 for every model.
- `temperature`, `top_p` are missing.
- `f_output_pressure` is 1 everywhere.
- `f_context_pressure` and `f_complexity_x_capability` are identical across the three models, so they are prompt features, not interactions.

What does vary is model identity, plus `knowledge_cutoff_year` / `f_recency_gap` for two of the three models (`gemini-flash-latest` has no cutoff).

Reduced comparison that can be run now, as a stand-in, not as the planned H3:

- Prompt only.
- Prompt + `llm_model`.
- Prompt + `llm_model` + `f_recency_gap`.

The planned H3 (a temperature or a context window the model was not trained on) waits for a second generation configuration. Configuration-out is not runnable.

## H4 — learned model vs the rate baselines

After H2, take the best arm and compare it, on the same folds, to training-fold rates only:

- global mean
- per-model mean
- per-subject mean (MMLU subject)
- model × subject mean

The earlier 10-trial logistic already lost to model × subject (Brier 0.153 vs 0.163 for the global mean). H4 is the pre-registered check for the tree, not a search over features.

Splits, in this order:

- Question-out. Primary. Already the 5-fold.
- Subject-out. Rotate one held-out subject. The model × subject baseline has not seen that subject, so it must fall back to the model-only rate. Say so in the table.
- Model-out. Leave one model out. Only the prompt-only arm can score the held-out model. An arm that uses `llm_model` as a category gets the training prior for that unseen id. Model × subject falls back to subject-only.
- Configuration-out. Skip until a second configuration exists.

Do not seal a holdout inside these 280 questions. Seal one when the set grows to the planned 1,500–3,000, and seal it before H2 chooses a feature set.
