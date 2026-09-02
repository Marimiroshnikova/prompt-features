# Phase 2 feature dictionary

Features are attached to the **run results** in `pilot_results.csv`
(same file at repo root and `experiments/out/`).
Only information available before generation is used.

Run grid: 280 questions × 14 models × **2** answers (`results_n2.csv`).
`risk_target` = 1 - accuracy, with accuracy in {0, 0.5, 1}.
The plan asked for incorrect/10; we still have 2, not 10.

## Prompt features

From question stem + lettered options, not the shared
`multiple choice questions about {category}` wrapper.

- 150 numeric/flag columns (`f_*`), including exam-trap flags.
- Token count: `f_context_token_count`.
- Task type: `question_category` plus `f_question_type`.
- Option count: `f_n_options`.
- Negation / except: `f_contains_negation`, `f_is_except_ask`.
- Constraints: `f_constraint_count`.
- Few-shot: `f_has_few_shot_examples`.
- Calculation / long hypo / best-answer: `f_is_formula_setup`,
  `f_is_long_scenario`, `f_is_best_answer_judgment`.
- Full one-line definitions: `FEATURES.md`.

## Model / configuration features

| field | meaning | missingness |
| --- | --- | --- |
| `llm_model` | model id used in the run | none |
| `model_family` | id prefix | none |
| `is_preview` | `preview` in the id | none |
| `is_open_source` | Gemma yes, Gemini no | none |
| `has_custom_tools` | id contains `customtools` | none |
| `context_window_tokens` | published input limit | none on this list |
| `knowledge_cutoff_year` | published cutoff year | null for Gemma and `*-latest` |
| `max_tokens_requested` / `output_token_limit` | 1024 | none; no variance |
| `temperature` | sampling temperature | **null** (not stored in the run) |
| `top_p` | nucleus sampling | **null** (not stored in the run) |

Citations: `experiments/model_specs.py`.

## Interaction features

- `f_context_pressure` = prompt tokens / context window.
- `f_output_pressure` = 1024/1024 = 1 for every row
  (expected output length was not logged).
- `f_recency_gap` = year in prompt − model cutoff year.
- `model_x_category` = task type × model id (string key).
- `f_complexity_x_capability` = prompt tokens × (window / 1e6).
  Window is the published limit, not a quality score.

Wrote 3920 rows × 183 columns.