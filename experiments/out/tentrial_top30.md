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
