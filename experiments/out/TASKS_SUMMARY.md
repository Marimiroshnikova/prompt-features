# Tasks 1 and 2 — what we were asked and what we did

Slides: `experiments/out/phase4_6_deck.html` (open in a browser, arrow keys to move).
Full record: `experiments/out/PHASE4_6_LOG.md`.

## English

### Task 1: Test hypothesis H1 (Phases 4–5 of the plan)

**What was asked:** compare a classifier trained on a binary label with a regressor trained on the soft risk (the exact fail rate out of 10 answers). The test had to use splits where the same question never appears in both training and test.

**What we did:**
- **Checked the teammate's notebooks first.** The regression notebook split by rows, so all 145 test questions were also in training. It also used `question_id` as a feature. Its good score (R² 0.32) was therefore optimistic.
- **Ran H1 properly.** We used 5-fold cross-validation grouped by question, on 280 questions × 3 Gemini models × 10 answers.
  - Two model families, as the plan asks: XGBoost and elastic-net logistic, each in a binary and a soft version.
  - All four models got the same 144 features.
- **Result: H1 does not hold.** Soft does not beat binary in either family.
  - XGBoost: Brier 0.169 vs 0.165.
  - Logistic: 0.162 vs 0.163.
  - Both differences include zero, and no model beats always guessing the average fail rate (Brier 0.163).
- **Some practical use.**
  - If the product skips the riskiest 20% of questions, a binary score cuts the error from 20.5% to 18.6%. That drop is small but real.
  - When skipping more, the soft logistic ranks best: 15% error while still answering 55% of questions.

### Task 2: Plan the H2, H3 and H4 experiments (Phases 4–5 of the plan)

**What was asked:** design the three comparisons:
- **H2:** prompt-only features vs all features;
- **H3:** prompt only vs prompt + model limits vs the full interaction model;
- **H4:** learned models vs the simple average-rate baselines.

**What we did (we planned them, then also ran them):**
- **H2:** adding the model columns helps a little (XGBoost Brier 0.171 to 0.165, logistic 0.164 to 0.162; both intervals exclude zero), but the result still loses to the baseline.
- **H3:** the planned test is blocked. Context window, temperature and top-p are identical for all three models. We ran a reduced version (prompt, then + model name, then + knowledge cutoff) and saw the same small gain.
- **H4:** on unseen questions, unseen subjects and an unseen model, no learned model beats the "this model on this subject" rate.
- **A leak we caught and fixed.** Holding out one model at a time first looked excellent. That was because the other two models' answers to the same questions were still in training.
- **Metrics (Phase 6) were locked before testing.**
  - If the product skips the riskiest X%, the coverage-risk curve and AURC decide.
  - If it uses a threshold or shows the risk number, Brier, Brier Skill Score, the reliability diagram and ECE decide.
- **How much data is needed.** A power analysis with no API spend shows that 280 questions is too few; the best feature looks like noise. About 1,000 questions (roughly $21) would settle it.

## ქართულად

### დავალება 1: H1 ჰიპოთეზის ტესტირება (plan-ის ფაზები 4–5)

**რა გვთხოვეს:** შეგვედარებინა ბინარულ ლეიბლზე ნასწავლი კლასიფიკატორი რბილ რისკზე (შეცდომის ზუსტ სიხშირეზე 10 პასუხიდან) ნასწავლ რეგრესორთან. ტესტში ერთი და იგივე კითხვა სატრენინგოშიც და სატესტოშიც არ უნდა მოხვედრილიყო.

**რა გავაკეთეთ:**
- **ჯერ გუნდის ნოუთბუქები შევამოწმეთ.** რეგრესიის ნოუთბუქი სტრიქონებად ყოფდა მონაცემებს, ამიტომ სატესტო 145-ვე კითხვა სატრენინგოშიც იყო. `question_id`-იც ფიჩერად ჰქონდა. ამიტომ მისი კარგი შედეგი (R² 0.32) გადაჭარბებული იყო.
- **H1 სწორად გავუშვით.** გამოვიყენეთ კითხვების მიხედვით დაჯგუფებული 5-fold კროს-ვალიდაცია: 280 კითხვა × 3 Gemini მოდელი × 10 პასუხი.
  - plan-ის მიხედვით ორი მოდელის ოჯახი ავიღეთ: XGBoost და elastic-net logistic, თითოეული ბინარული და რბილი ვერსიით.
  - ოთხივე მოდელმა ერთი და იგივე 144 ფიჩერი მიიღო.
- **შედეგი: H1 არ დადასტურდა.** არც ერთ ოჯახში რბილი ვერსია ბინარულს არ ჯობნის.
  - XGBoost: Brier 0.169 და 0.165.
  - ლოგისტიკური: 0.162 და 0.163.
  - ორივე სხვაობის ინტერვალი ნულს მოიცავს. არც ერთი მოდელი არ ჯობნის შეცდომის საშუალო სიხშირის მუდმივ პროგნოზს (Brier 0.163).
- **პრაქტიკული სარგებელი მცირეა.**
  - თუ პროდუქტი ყველაზე სარისკო 20%-ს გამოტოვებს, ბინარული ქულა შეცდომას 20.5%-დან 18.6%-მდე ამცირებს. კლება პატარაა, მაგრამ რეალური.
  - მეტის გამოტოვებისას რბილი ლოგისტიკური საუკეთესოდ ალაგებს: 15% შეცდომა, როცა კითხვების 55%-ს ჯერ კიდევ პასუხობს.

### დავალება 2: H2, H3, H4 ექსპერიმენტების დაგეგმვა (plan-ის ფაზები 4–5)

**რა გვთხოვეს:** სამი შედარების დაგეგმვა:
- **H2:** მხოლოდ პრომპტის ფიჩერები და ყველა ფიჩერი;
- **H3:** მხოლოდ პრომპტი, პრომპტი + მოდელის ლიმიტები და სრული ინტერაქციების მოდელი;
- **H4:** ნასწავლი მოდელები და მარტივი საშუალო სიხშირის baseline-ები.

**რა გავაკეთეთ (დავგეგმეთ და გავუშვით კიდეც):**
- **H2:** მოდელის სვეტების დამატება ცოტას შველის (XGBoost Brier 0.171-დან 0.165-მდე, ლოგისტიკური 0.164-დან 0.162-მდე; ორივე ინტერვალი ნულს არ მოიცავს), მაგრამ შედეგი baseline-ს მაინც ჩამორჩება.
- **H3:** დაგეგმილი ტესტი ვერ გაეშვა, რადგან context window, temperature და top-p სამივე მოდელზე ერთნაირია. შემცირებული ვერსია გავუშვით (პრომპტი, მერე + მოდელის სახელი, მერე + knowledge cutoff) და იგივე მცირე ზრდა დავინახეთ.
- **H4:** უცნობ კითხვებზე, უცნობ საგნებსა და უცნობ მოდელზე ვერც ერთი ნასწავლი მოდელი ვერ აჯობა „ეს მოდელი ამ საგანზე“ სიხშირეს.
- **გაჟონვა ვიპოვეთ და გავასწორეთ.** როცა თითო მოდელს ცალკე ვტოვებდით, შედეგი ძალიან კარგად გამოიყურებოდა. მიზეზი ის იყო, რომ დანარჩენი ორი მოდელის პასუხები იმავე კითხვებზე სატრენინგოში რჩებოდა.
- **მეტრიკები (ფაზა 6) ტესტირებამდე დავაფიქსირეთ.**
  - თუ პროდუქტი ყველაზე სარისკო X%-ს გამოტოვებს, გადამწყვეტია coverage-risk მრუდი და AURC.
  - თუ ზღვარს იყენებს ან რისკის რიცხვს აჩვენებს, გადამწყვეტია Brier, Brier Skill Score, reliability diagram და ECE.
- **რამდენი მონაცემია საჭირო.** power analysis-მა, API ხარჯის გარეშე, აჩვენა, რომ 280 კითხვა ცოტაა; საუკეთესო ფიჩერი ხმაურისგან არ განსხვავდება. დაახლოებით 1,000 კითხვა (დაახლოებით $21) საკითხს გადაწყვეტს.
