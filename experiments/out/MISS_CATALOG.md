# Miss catalog — what actually goes wrong

Two real questions per type. Counts are “how many of the 14 models got it wrong.”

## 1. Except / NOT / LEAST

The question flips the match. The model picks the thing that *is* true.

| id | subject | wrong | question |
|---|---|---|---|
| 6214 | health | 14/14 | Which of the following is NOT one of the more frequently named activities by those who are retired? |
| 10649 | computer science | 14/14 | Of the following potential benefits, which is LEAST likely to be provided by the upgraded system? |

On the full 280: 11 such items, fail **41%** vs **20%** for the rest.

## 2. “Best” / “most likely” (no unique fact)

Several options are plausible. The model has to pick the preferred one.

| id | subject | wrong | question |
|---|---|---|---|
| 936 | law | 14/14 | A defendant, an indigent, was arrested and charged with attempted murder… [long bar-exam story] |
| 967 | law | 13/14 | A city imposes a municipal excise tax of $200 per year on commercial artists' studios… |

On the full 280: 15 such items, fail **44%** vs **19%**.

## 3. Long hypo (80+ words)

A made-up story, then a question. Every model missed the long defendant item.

| id | subject | wrong | question |
|---|---|---|---|
| 936 | law | 14/14 | same defendant hypo as above |
| 967 | law | 13/14 | same city tax hypo as above |

On the full 280: 43 long stems, fail **29%** vs **19%**.

## 4. Obscure health fact (short, still everyone misses)

Not long. Not a trick of wording. The model simply does not know, or guesses the survey/medical fact.

| id | subject | wrong | question |
|---|---|---|---|
| 6641 | health | 14/14 | What is the biggest risk factor for infection with Ebola? |
| 6774 | health | 14/14 | What is the morphology of the Dane particle? |

This is why “What is X?” is not always easy: in health those still fail a lot.

## 5. Easy pole (so you can contrast)

| id | subject | wrong | question |
|---|---|---|---|
| 138 | business | 0/14 | If at the beginning of each month a deposit of $500 is made… 8% compounded monthly… after five years? |
| 7999 | math | 0/14 | Independent events, P(A)=0.7, P(B)=0.2. Compute P(A ∩ B). |
| 6857 | economics | 0/14 | What is Market Socialism? |

## What this catalog is for

You can present these five types without Brier scores.

You cannot yet turn types 1–3 into a predictor that beats “health + weak model.” There are too few of them, and type 4 is just the subject.

When budget allows 10 answers, keep this list: those are the items where a probability (3/10 vs 9/10) would mean something.
