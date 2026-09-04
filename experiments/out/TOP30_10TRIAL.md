# The best 30 features — re-derived on the 10-trial data

Ranked against **`q_fail_rate`**: the mean fail rate of 3 Gemini models over
**10 answers each**, so 30 trials behind every question. `web/top30.json` was ranked on
1 answer per cell, where the label was a single coin flip.

Rebuild with `python experiments/build_10trial_top30.py`.

## Read this before quoting the list

**None of these 30 are statistically significant.** Zero of 139 features clear
Benjamini-Hochberg at q < 0.05. At n = 280 questions with 139 simultaneous tests the
smallest detectable |rho| is **0.211**, and the largest effect actually present is
**0.165** — the study cannot resolve what is there. This is a ranked list of
**candidates to carry into a larger run**, not a list of established predictors, and
none of them beats the model x subject baseline (see `TENTRIAL_FINDINGS.md`).

## How the ranking is built

One correlation on 280 questions is too easy to get by luck when 139 are being tested,
so |rho| is not the whole score. Each feature is scored on three independent pieces of
evidence, then de-duplicated:

| weight | evidence | what it guards against |
| ---: | --- | --- |
| 0.45 | \|Spearman\| vs `q_fail_rate`, all 280 questions | — |
| 0.35 | picked in N of 5 question folds, selection inside the fold | a correlation that lives in one slice of the data |
| 0.2 | same sign across all 3 models, weighted by strength | a fluke only one model produces |
| +0.05 | boolean flag whose bootstrap CI over questions excludes zero | an effect indistinguishable from zero |

Then drop anything correlated |rho| >= 0.9 with a feature already kept, one
dummy per categorical parent, cap 30. Same hygiene as `select_top30.py`.

## The 30

`*` on an effect means the bootstrap CI over questions excludes zero. `folds` is
stability; `models` is how many of the 3 point the same way.

| rank | feature | group | score | rho | BH q | folds | models | effect (on vs off) |
| ---: | --- | --- | ---: | ---: | ---: | :---: | :---: | --- |
| 1 | `question_type=what` | Question structure | 0.982 | -0.165 | 0.37 | 5/5 | 3/3 | — |
| 2 | `max_word_len` | Size and shape | 0.926 | +0.165 | 0.37 | 5/5 | 3/3 | — |
| 3 | `is_definition_ask` | Exam-item traps | 0.886 | -0.153 | 0.37 | 5/5 | 3/3 | -0.056 [-0.179,+0.088] n=24 |
| 4 | `domain_hint=legal` | Domain and register | 0.876 | +0.147 | 0.38 | 4/5 | 3/3 | — |
| 5 | `is_long_scenario` | Exam-item traps | 0.857 | +0.134 | 0.39 | 4/5 | 3/3 | +0.109 [+0.011,+0.219] n=43 * |
| 6 | `min_word_zipf` | Rarity and vocabulary | 0.840 | -0.152 | 0.37 | 4/5 | 3/3 | — |
| 7 | `aggregation_flag` | Reasoning and multi-hop | 0.820 | +0.136 | 0.39 | 3/5 | 3/3 | +0.092 [+0.010,+0.168] n=92 * |
| 8 | `is_best_answer_judgment` | Exam-item traps | 0.798 | +0.127 | 0.47 | 3/5 | 3/3 | +0.225 [+0.017,+0.429] n=15 * |
| 9 | `stem_word_count` | Exam-item traps | 0.754 | +0.134 | 0.39 | 4/5 | 3/3 | — |
| 10 | `rare_word_count` | Rarity and vocabulary | 0.707 | +0.134 | 0.39 | 3/5 | 3/3 | — |
| 11 | `oov_like_count` | Rarity and vocabulary | 0.666 | +0.122 | 0.53 | 3/5 | 3/3 | — |
| 12 | `has_escape_option` | Exam-item traps | 0.663 | +0.119 | 0.53 | 2/5 | 3/3 | — |
| 13 | `is_except_ask` | Exam-item traps | 0.590 | +0.100 | 0.71 | 2/5 | 3/3 | +0.231 [-0.018,+0.489] n=11 |
| 14 | `content_word_count` | Readability and lexical difficulty | 0.581 | +0.117 | 0.53 | 2/5 | 3/3 | — |
| 15 | `question_category=Math` | Domain and register | 0.523 | -0.094 | 0.71 | 2/5 | 3/3 | — |
| 16 | `difficult_word_count` | Readability and lexical difficulty | 0.520 | +0.099 | 0.71 | 2/5 | 3/3 | — |
| 17 | `year_count` | Temporal constraints | 0.495 | +0.101 | 0.71 | 2/5 | 2/3 | — |
| 18 | `is_ambiguous` | Ambiguity and underspecification | 0.486 | +0.104 | 0.71 | 1/5 | 3/3 | — |
| 19 | `core_question_ratio` | Prompt-engineering artifacts | 0.470 | +0.102 | 0.71 | 1/5 | 3/3 | — |
| 20 | `is_yes_no_question` | Question structure | 0.407 | +0.088 | 0.71 | 1/5 | 3/3 | — |
| 21 | `context_token_count` | Size and shape | 0.405 | +0.093 | 0.71 | 1/5 | 3/3 | — |
| 22 | `is_comparison` | Reasoning and multi-hop | 0.403 | +0.097 | 0.71 | 1/5 | 2/3 | +0.030 [-0.060,+0.119] n=49 |
| 23 | `causal_flag` | Reasoning and multi-hop | 0.402 | +0.091 | 0.71 | 1/5 | 3/3 | +0.054 [-0.032,+0.140] n=62 |
| 24 | `temporal_type=range` | Temporal constraints | 0.398 | +0.087 | 0.71 | 1/5 | 3/3 | — |
| 25 | `is_multi_part` | Question structure | 0.396 | -0.079 | 0.74 | 1/5 | 3/3 | -0.086 [-0.171,+0.014] n=34 |
| 26 | `contains_negation` | Reasoning and multi-hop | 0.392 | +0.085 | 0.73 | 1/5 | 3/3 | +0.046 [-0.028,+0.122] n=96 |
| 27 | `has_relative_recency` | Temporal constraints | 0.389 | -0.055 | 0.79 | 1/5 | 3/3 | -0.105 [-0.175,-0.024] n=28 * |
| 28 | `non_ascii_ratio` | Domain and register | 0.378 | -0.081 | 0.74 | 1/5 | 3/3 | — |
| 29 | `jargon_ratio` | Rarity and vocabulary | 0.357 | +0.076 | 0.74 | 1/5 | 3/3 | — |
| 30 | `type_token_ratio` | Readability and lexical difficulty | 0.347 | -0.073 | 0.74 | 1/5 | 3/3 | — |

## What each one measures

| rank | feature | direction | what it measures |
| ---: | --- | --- | --- |
| 1 | `question_type=what` | this level = fewer misses | The leading question word of the actual question |
| 2 | `max_word_len` | higher = more misses | Length of the longest word |
| 3 | `is_definition_ask` | higher = fewer misses | Whether the item is a short What is / What are definition |
| 4 | `domain_hint=legal` | this level = more misses | Subject area of the prompt, from keyword lexicons |
| 5 | `is_long_scenario` | higher = more misses | Whether the stem is a long hypo (80 or more words) |
| 6 | `min_word_zipf` | higher = fewer misses | Corpus frequency of the rarest word in the prompt |
| 7 | `aggregation_flag` | higher = more misses | Whether the prompt asks for a count, total, ranking or extreme |
| 8 | `is_best_answer_judgment` | higher = more misses | Whether the item asks for the best / most likely / most nearly answer |
| 9 | `stem_word_count` | higher = more misses | Word count of the question stem, ignoring lettered options |
| 10 | `rare_word_count` | higher = more misses | How many words have a Zipf frequency below 3.0 |
| 11 | `oov_like_count` | higher = more misses | How many tokens look like identifiers rather than English words |
| 12 | `has_escape_option` | higher = more misses | Whether an option is all/none of the above or cannot be determined |
| 13 | `is_except_ask` | higher = more misses | Whether the question is an except / NOT / least / incorrect item |
| 14 | `content_word_count` | higher = more misses | How many words carry retrievable content |
| 15 | `question_category=Math` | this level = fewer misses | Coarse type of the ask, chosen by weighted evidence rather than first-match-wins |
| 16 | `difficult_word_count` | higher = more misses | How many words are outside the Dale-Chall list of familiar words |
| 17 | `year_count` | higher = more misses | How many four-digit years the prompt names |
| 18 | `is_ambiguous` | higher = more misses | Whether the prompt is too underspecified to retrieve for |
| 19 | `core_question_ratio` | higher = more misses | Share of the prompt that is the actual question, after removing instructions, examples and pasted context |
| 20 | `is_yes_no_question` | higher = more misses | Whether the prompt is a yes/no question |
| 21 | `context_token_count` | higher = more misses | Length of the prompt in GPT-style BPE tokens |
| 22 | `is_comparison` | higher = more misses | Whether the prompt compares two or more things |
| 23 | `causal_flag` | higher = more misses | Whether the prompt asks about causes or effects |
| 24 | `temporal_type=range` | this level = more misses | What kind of time constraint the prompt uses |
| 25 | `is_multi_part` | higher = fewer misses | Whether the prompt asks more than one thing |
| 26 | `contains_negation` | higher = more misses | Whether the prompt uses a negation word |
| 27 | `has_relative_recency` | higher = fewer misses | Whether the prompt asks for whatever is newest rather than a fixed date |
| 28 | `non_ascii_ratio` | higher = fewer misses | Share of characters outside plain ASCII |
| 29 | `jargon_ratio` | higher = more misses | Share of content words that look like specialist terminology |
| 30 | `type_token_ratio` | higher = fewer misses | Distinct words divided by total words |

## Reading the list

**Exam-item traps take 6 of the top 13** — `is_definition_ask`, `is_long_scenario`,
`is_best_answer_judgment`, `stem_word_count`, `has_escape_option`, `is_except_ask`. That
group was written *after* reading the misses; the rest of the set was inherited from the
retrieval design. It is the clearest signal in the table about where the next round of
feature work should go.

**The retrieval headliners competed and lost.** `anchor_count` (retrieval rank 1, here rho=+0.047), `anchor_density` (retrieval rank 2, here rho=-0.025), `unresolved_pronoun_count` (retrieval rank 4, here rho=-0.055), `has_dangling_reference` (retrieval rank 5, here rho=-0.036), `vague_term_count` (retrieval rank 6, here rho=+0.044) — ranks 1, 2, 4, 5 and 6
of the retrieval Top 30 in `FEATURES.md` — are all screened here and none reaches
|rho| = 0.06. `is_ambiguous` falls from rank 3 to 18 and
`context_token_count` from 8 to 21.
An MMLU-Pro item prints its ten options: there is nothing to retrieve and nothing
dangling, so the features that measure retrievability have no work to do. Spec mismatch,
not a bug.

**Three you can quote out loud** — stable in >= 3 folds *and* CI excludes zero:

- `is_long_scenario` — +0.109 fail rate on vs off, 95% CI [+0.011, +0.219], n=43 questions
- `aggregation_flag` — +0.092 fail rate on vs off, 95% CI [+0.010, +0.168], n=92 questions
- `is_best_answer_judgment` — +0.225 fail rate on vs off, 95% CI [+0.017, +0.429], n=15 questions

**8 that are stable but unresolvable at this sample size** — keep measuring,
do not ship: `question_type=what`, `max_word_len`, `is_definition_ask`, `domain_hint=legal`, `min_word_zipf`, `stem_word_count`, `rare_word_count`, `oov_like_count`.

**Rank 13 is the one to fund.**
`is_except_ask` carries the largest effect in the whole feature set (+0.231: except/NOT
items fail 42.7% against 19.6%) on **11 questions**, which is why its CI still touches
zero. Sampling 100+ except/NOT items would settle it. Data problem, not a modelling
problem.

**Churn against the n=1 ranking: 11 of 30 entries are new**
(`is_definition_ask`, `domain_hint=legal`, `question_category=Math`, `year_count`, `is_yes_no_question`, `is_comparison`, `causal_flag`, `is_multi_part`, `contains_negation`, `has_relative_recency`, `non_ascii_ratio`). Two rankings built from the same
280 questions disagree on a third of their entries — a direct measure of how unstable any
ranking is at this n.

**Group coverage:**

- Exam-item traps — 6
- Rarity and vocabulary — 4
- Reasoning and multi-hop — 4
- Question structure — 3
- Domain and register — 3
- Readability and lexical difficulty — 3
- Temporal constraints — 3
- Size and shape — 2
- Ambiguity and underspecification — 1
- Prompt-engineering artifacts — 1

## Files

| file | what |
| --- | --- |
| `tentrial_top30.json` | the ranked 30, `web/top30.json` schema plus score, folds, effects |
| `tentrial_top30.csv` | the same as a table, with group and description columns |
| `tentrial_top30.md` | just the ranked table |
| `tentrial_univariate_qfail.csv` | all 139 candidates, unranked |
