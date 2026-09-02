# Full picture — MMLU-Pro miss prediction

## The project in one paragraph

We have 280 exam questions and 14 Gemini/Gemma models. Each model answered each question once. About 20% of those answers were wrong. The hope was: measure the question text and predict the miss *before* the model answers. If that worked, we could skip hard items or send them to a stronger model.

## What the data is

| | |
|---|---|
| Questions | 280 MMLU-Pro, 20 per 14 subjects |
| Models | 14 Gemini / Gemma |
| Rows | 3,920 (but only 280 independent questions) |
| Answers per cell | **1** (the written plan wanted **10**) |
| Label | wrong / right, not a probability |
| Easy pole | $500 monthly deposit — 0/14 wrong |
| Hard pole | long law hypo; “which is NOT … retired” — 14/14 wrong |

Health fails 45%. Economics fails 4%. Weakest model 36% fail. Strongest 11%.

## What we already did

1. **Locked the rules.** Features from the question + options only. Same question never in train and test. No invented model specs.

2. **Scored 138 old features** (built for search/retrieval, not exams). They lose to “this model × this subject.”

3. **Read the misses** and wrote **11 exam-trap flags**: except/NOT, “most likely / best describes”, long hypo, definition, formula, option shape.

4. **Checked those flags two ways.**
   - As a description: “best describes” items fail 44% vs 19%. Except/NOT fail 41% vs 20%. Long hypos fail 29% vs 19%.
   - As a predictor in cross-validation: they do **not** beat model × subject. Adding them makes Brier worse (0.168 vs 0.156). Too few examples (11–15 items) to generalize.

5. **Harder splits.** Unseen subject: prompt/exam lose. Unseen model: category almost wins; text features still do not. Abstain on the riskiest 20% using model × subject: error 20.5% → 16.9%.

## The finding you can defend

Misses on this grid are mostly **which model** and **which subject**.  
Wording traps (except/NOT, “best answer”, long hypo) are real when they appear, but there are not enough of them here to beat those two facts.

Say: “the features ran, and they lost to model × subject.”  
Do not say: “the experiment failed.”

## The original 8-phase plan vs reality

| Phase | Plan | Us |
|---|---|---|
| 0 Contract | MMLU-Pro, 10 answers, risk = incorrect/10 | Locked, but we have 1 answer |
| 1 Pilot data | 200–280 × models × 10 | 280 × 14 × **1** |
| 2 Features | prompt + model + interaction | Done, plus exam flags |
| 3 Baselines | global / model / category / model×category / prompt logistic | Done. Model × category wins |
| 4 Learned models | logistic / trees, H1–H4 | **Not started** — would lose for the same reason |
| 5 Splits | question / category / model out | Preview done |
| 6 Metrics | Brier, calibration, coverage–risk | Brier + coverage–risk on 0/1 labels |
| 7 Scale | 1,500–3,000 questions | **Not started** |
| 8 Writeup | error analysis + report | Started (examples + exam flags) |

## What is actually missing (in order)

**1. Ten answers per question, not one.**  
Without this, “risk” is a coin flip. Brier cannot mean “30% chance of fail.” This is the hole in Phase 1.

**2. More questions, especially more trap items.**  
15 “best describes” questions cannot train a flag. The plan’s 1,500–3,000 is how those flags could become predictors, not just descriptions.

**3. Only then Phase 4.**  
XGBoost on 280 × 1-shot will rediscover model and subject. Do not do this yet.

**4. A short human catalog of miss types** (optional, presentable).  
Law hypo / except-NOT / obscure health fact / 10 near-identical options. We started this; it can be a one-page table with 2 examples each.

## What not to do next

- Add more retrieval-style features (length, Zipf, anchors). We already know they lose.
- Train trees to “get a better number.”
- Present Brier tables you cannot explain. Present the two example questions and the model/subject gap.

## Recommended next move

Do **(1) ten trials** if you can rerun the GAIA eval, **or** **(2) the one-page miss catalog** if you cannot spend API budget this week.

(2) is done: `MISS_CATALOG.md`.  
Budget ask for (1): `BUDGET_10_TRIALS.md` — about **$25–60** for 9 more Gemini rounds.
