# What we found (short)

We gave 280 exam questions to 14 Gemini/Gemma models. Each model answered each question once.

**About 1 in 5 answers was wrong** (805 / 3,920).

The miss depends on **which model** and **which subject**, not on how the question is worded.

- Weakest model: ~36% wrong (`gemini-2.5-flash-lite`)
- Strongest model: ~11% wrong (`gemini-flash-latest`)
- Hardest subject: health (~45% wrong)
- Easiest subject: economics (~4% wrong)

We measured the question text (length, rare words, “except / NOT”, long stories). Those measurements **do not** predict a miss better than “this model, on this subject.”

Traps exist, but they are rare:

- Everyone missed: “Which is NOT a common retired activity?”
- Nobody missed: “$500 deposited each month for five years”

**In one line:** we can say *who* fails *where*. We cannot yet say *this wording will fail* on a new question.

We only have one answer per cell (a second run is still collecting). That is not a probability. Do not train a bigger model yet.
