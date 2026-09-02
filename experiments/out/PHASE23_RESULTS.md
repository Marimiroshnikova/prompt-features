# Progress

## Where we were

The plan asked for 10 answers per question, then “can we predict a miss from the text?”

What we actually had: **1 answer** × 280 questions × 14 models = **3,920**.

What we found on that first run:

- **20.5%** of answers were wrong.
- Weak models miss much more than strong ones (**36%** vs **11%**).
- Health is hard (**45%** wrong). Economics is easy (**4%**).
- We measured **138** things in the question text. They did **not** beat “this model, on this subject.”

So: the miss is the model and the subject, not the wording.

## Where we are now

We are collecting a **second answer** on the same questions.

- About **half** of that run is on disk (1,861 / 3,920).
- When both answers exist, about **15%** change letter, about **10%** flip right/wrong.
- Accuracy is still about the same as the first run when a letter actually comes back.
- Some models return **empty** replies. Do not count those as wrong yet. Do not join the two runs yet.

Two answers are better than one. They are **not** the 10 the plan asked for.

## Next

1. Finish the second run and retry empty replies.
2. Then decide if we can pay for more answers.
3. Do not train a bigger predictor until then — it would just rediscover model and subject.
