# Phase 2 feature dictionary

Computed before generation. `n=1` Bernoulli labels in `results.csv`;
`risk_target` is currently `y_fail` and can later hold `incorrect/10`.

## Prompt features

Extracted once per `question_id` from the question stem plus lettered
options. The shared instruction line (`multiple choice questions about
{category}`) is excluded so category cannot leak into prompt-only features.

- `139` promptfeat measurements plus `f_n_options`.
- Definitions for the 138 live in `FEATURES.md`.
- `f_n_options`: number of multiple-choice options on this question (3–10).
- `category` is metadata / a Phase 3 baseline, not a prompt feature.

## Model / configuration features

| field | meaning | missingness |
| --- | --- | --- |
| `model_family` | id prefix (`gemini-2.5`, `gemma-4`, `gemini-latest`, …) | none |
| `is_preview` | `preview` appears in the model id | none |
| `is_open_source` | Gemma yes, Gemini no | none |
| `max_tokens_requested` | 1024 from GAIA `inference.json` | none; **zero variance** |
| `context_window_tokens` | published input limit | none on this 14-model list |
| `knowledge_cutoff_year` | published cutoff year | null for Gemma and `*-latest` aliases |
| `temperature` | sampling temperature | **null for every row** (API default, not stored) |

`gemini-flash-latest` is kept distinct from `gemini-2.5-flash`:
they have different accuracy, so they are different snapshots.
Lookup and citations: `experiments/model_specs.py`.

## Interaction features

- `f_context_pressure` = `f_context_token_count` / `context_window_tokens`. nunique=382, min=2.67029e-05, max=0.00338672. On 1M-token Gemini windows this is ~0 for every MMLU-Pro prompt; constants are dropped at model time.
- `output_pressure` **dropped: zero variance**. Every row requested 1024 tokens.
- `f_recency_gap` = `f_year_max` - `knowledge_cutoff_year`. non-null=297 of 3920 rows (8%). Null when the prompt names no year or the model has no published cutoff.
- Category × model is a Phase 3 **baseline**, not a learned one-hot here.

## Labels

- `n_trials = 1` (not a 10-sample probability).
- `y_fail = 1 - accuracy` (1 = the single generation was wrong).
- `risk_target = y_fail` today; replace with `incorrect/10` when that file exists.

## Short list (question-out stability)

138 features cannot be estimated on 280 questions. Selection is against
`q_fail_rate` (mean fail across 14 models), inside question-out folds.
Features that appeared in at least 3 of 5 folds, cap 15:

- `f_rare_word_count`, `f_oov_like_count`, `f_min_word_zipf` (rarity)
- `f_context_token_count`, `f_content_word_count`, `f_max_word_len` (size)
- `f_aggregation_flag`, `f_question_type=what`, `f_temporal_type=range`
- `f_mtld`, `f_type_token_ratio`
- `f_domain_hint=finance`

None of these is BH-significant on the pooled 280 questions. The list is
the least-noisy prompt subset for this grid, not a claim of signal.

Phase 3 re-selects inside each training fold. Headline: prompt-short
does **not** beat the per-model or model×category Brier baselines.
See `mmlu_phase3_report.txt`.
