# Budget ask — 10 answers per question

We already have 1 answer × 280 questions × 14 models (3,920 calls).
The written plan wants 10. That is 9 more rounds, or 35,280 extra calls.

Tokens per call, from the run we already have: ~230 input, ~140 output.
Prices below are published Gemini paid-tier rates (per 1M tokens), Sept 2026.
Gemma 4 is **not** in this table — ask whoever pays what that endpoint costs.

## What to put on the budget request

| Scope | Gemini only (12 models) | With thinking + retry buffer (×2.2) |
|---|---|---|
| 1 more trial (what we already spent, order of) | ~$3 | ~$6 |
| **9 more trials (to reach 10)** | **~$25** | **~$55** |
| 10 trials from scratch | ~$28 | ~$62 |

Pro models are most of the bill (3.1 Pro preview pair ~$11 of the $28).

## Cheaper options if $60 is too much

| Option | 10-trial Gemini cost (buffered) | What you lose |
|---|---|---|
| All 12 Gemini models | ~$60 | nothing on Gemini |
| Drop both 3.1 Pro previews | ~$35 | two expensive models |
| Only 2.5 lite + flash + pro | ~$15 | no Gemini 3.x |
| 3 trials, not 10, all Gemini | ~$18 | weaker probability |

## What to tell the person with the card

> We need about **$25–60** to finish the pilot the plan described: 10 answers per question on the 280-question set. Without that, “risk” is only right/wrong, not a probability, and we should not train Phase 4 models. Gemma is extra and unpriced here.

Confirm live prices on [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) before you send the ask. Aliases (`*-latest`) can move.
