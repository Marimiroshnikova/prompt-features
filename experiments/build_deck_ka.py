"""Build the Georgian Phase 4-5 slide deck from the saved result files.

Each of the two tasks (H1 test; H2/H3/H4 plan) is told in four steps: what was
asked, what we had, what we did, what we found. Reads the same JSON files as
build_deck.py and writes out/phase4_5_deck_ka.html.

    python experiments/build_deck_ka.py
"""

from __future__ import annotations

from build_deck import (ACCENT, BAD, CSS, FAINT, GOOD, INK, JS, MUTED, OUT, WARN, big,
                        bar_chart, by_name, ci, esc, hist_svg, line_chart, load, pct,
                        reliability_svg, slide, table)

DECK = OUT / "phase4_5_deck_ka.html"

STEP = {1: "ნაბიჯი 1: რას გვთხოვდნენ", 2: "ნაბიჯი 2: რა ინფორმაცია გვქონდა",
        3: "ნაბიჯი 3: რა გავაკეთეთ", 4: "ნაბიჯი 4: რა შედეგი მივიღეთ"}


def kick(task, step, extra=""):
    head = f"დავალება {task} · {STEP[step]}"
    return f"{head} · {extra}" if extra else head


def card(title, body, tone=""):
    return f'<div class="card {tone}"><h3>{esc(title)}</h3>{body}</div>'


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def build():
    h1 = load("h1_metrics.json")
    h24 = load("h2_h4_metrics.json")
    pw = load("power_metrics.json")

    r1 = by_name(h1["results"])
    base, binr, soft = r1["base rate"], r1["binary classifier"], r1["soft regressor"]
    lbin, lsoft = r1["binary logistic"], r1["soft logistic"]
    boot = h1["bootstrap"]
    qo, so, mo = (h24["splits"][k] for k in ("question-out", "subject-out", "model-out"))
    q = by_name(qo["results"])
    fail = h1["fail_rate"]
    all_wrong, between = h1["share_cells_all_wrong"], h1["share_cells_between"]
    all_right = 1 - all_wrong - between
    arms = h24["arms"]
    sizes = pw["sizes"]
    n1000 = next(s for s in sizes if s["n_questions"] == 1000)
    n_subjects = len({row["subject"] for row in h1["by_subject"]})

    slides = []

    # ------------------------------------------------------------------ title
    slides.append(
        '<section class="slide title">'
        '<p class="kicker">MMLU-Pro · შეცდომის პროგნოზი · plan.docx-ის ფაზები 4–5</p>'
        '<h1>შეიძლება თუ არა კითხვის ტექსტით წინასწარ ვთქვათ, როდის შეცდება LLM?</h1>'
        '<p class="lede">ორი დავალება გვქონდა: <b>H1 ჰიპოთეზის ტესტირება</b> და '
        '<b>H2, H3, H4 ექსპერიმენტების დაგეგმვა</b>. თითოეულს ოთხ ნაბიჯად ვყვებით.</p>'
        '<div class="two">'
        + card("დავალება 1 · H1", '<ol><li>რას გვთხოვდნენ</li><li>რა ინფორმაცია გვქონდა</li>'
               '<li>რა გავაკეთეთ</li><li>რა შედეგი მივიღეთ</li></ol>')
        + card("დავალება 2 · H2, H3, H4", '<ol><li>რას გვთხოვდნენ</li><li>რა ინფორმაცია გვქონდა</li>'
               '<li>რა გავაკეთეთ</li><li>რა შედეგი მივიღეთ</li></ol>')
        + '</div>'
        '<p class="small">ბოლოს: რამდენი მონაცემია საჭირო, რა ღირს და რა არის შემდეგი ნაბიჯი.</p>'
        '</section>')

    # ------------------------------------------------------------------ summary
    slides.append(slide(
        "მოკლედ, სანამ დეტალებზე გადავალთ",
        "მთავარი პასუხი: ჯერჯერობით ვერა",
        '<div class="row">'
        + big("არა", "H1: ზუსტი სიხშირე „კი/არა“ ლეიბლს ვერ ჯობნის", "bad")
        + big("ცოტა", "H2: მოდელის მონაცემები ოდნავ შველის", "accent")
        + big("ვერ", "H3: სრულად ვერ შემოწმდა, შესადარებელი არაფერია", "bad")
        + big("არა", "H4: ვერც ერთი მოდელი ვერ ჯობნის მარტივ ცხრილს", "bad")
        + '</div>'
        + ul([
            f"280 კითხვაზე საუკეთესო ფიჩერი ხმაურისგან არ გამოირჩევა. დაახლოებით "
            f"{n1000['n_questions']:,} კითხვა საკმარისი იქნება, დაახლოებით "
            f"${n1000['cost_usd_new_questions_only']:.0f} ღირს.",
            "ერთი სასარგებლო შედეგი: სარისკო კითხვების გამოტოვებით შეცდომა მცირდება, ოღონდ ცოტათი.",
        ])))

    # ================================================================ TASK 1
    slides.append(
        '<section class="slide title"><p class="kicker">დავალება 1</p>'
        '<h1>H1 ჰიპოთეზის ტესტირება</h1>'
        '<p class="lede">„ბინარულ ლეიბლზე ნასწავლი კლასიფიკატორი“ vs „რბილ რისკზე ნასწავლი რეგრესორი“ '
        '(plan.docx, ჰიპოთეზების ცხრილი და ფაზები 4–5)</p></section>')

    # 1.1 asked
    slides.append(slide(
        kick(1, 1),
        "რომელი სამიზნე ჯობია: „კი/არა“ თუ ზუსტი სიხშირე?",
        '<div class="two">'
        + card("ბინარული ლეიბლი (კლასიფიკატორი)",
               "<p>1, თუ მოდელი კითხვაზე <b>ნახევარზე მეტჯერ</b> ცდება, თორემ 0.</p>"
               '<p class="em">მაგალითი: 10-დან 3 არასწორი → 0</p>')
        + card("რბილი რისკი (რეგრესორი)",
               "<p>შეცდომის <b>ზუსტი სიხშირე</b>: არასწორი პასუხები ÷ 10.</p>"
               '<p class="em">მაგალითი: 10-დან 3 არასწორი → 0.3</p>')
        + '</div>'
        '<h3 style="margin-top:22px">plan-ის მოთხოვნები</h3>'
        + ul([
            "<b>ფაზა 4:</b> ჯერ მარტივი მოდელები: ლოგისტიკური / elastic-net და XGBoost (random forest არასავალდებულოა).",
            "<b>ფაზა 4:</b> რბილი სამიზნე იმავე 10 პასუხზე არ შეფასდეს; სატესტოზე სასურველია 20–30 ახალი პასუხი.",
            "<b>ფაზა 5:</b> ერთი კითხვის პასუხები ერთდროულად სატრენინგოშიც და სატესტოშიც არასდროს.",
            "<b>ფაზა 6:</b> მთავარი მეტრიკა Brier; ასევე calibration, ECE, log loss, coverage–risk.",
        ]),
        "H1 ამბობს: ზუსტი სიხშირე მეტ ინფორმაციას ატარებს (0.3 და 0.0 სხვადასხვაა, ბინარულში ორივე 0), "
        "ამიტომ მასზე ნასწავლი მოდელი უკეთეს პროგნოზს უნდა იძლეოდეს."))

    # 1.2 had: data
    slides.append(slide(
        kick(1, 2, "მონაცემები"),
        "280 კითხვა × 3 Gemini მოდელი × 10 პასუხი",
        '<div class="row">'
        + big("280", f"MMLU-Pro კითხვა, {n_subjects} საგანი")
        + big("840", "სტრიქონი (კითხვა × მოდელი)")
        + big("8,400", "შეფასებული პასუხი")
        + big(pct(fail), "პასუხებიდან არასწორი")
        + '</div>'
        '<div class="two"><div><h3>მოდელები</h3>'
        + ul(["<code>gemini-2.5-flash-lite</code>", "<code>gemini-3.1-flash-lite</code>",
              "<code>gemini-flash-latest</code>"])
        + f'<h3>ფიჩერები: {h1["n_features"]}</h3>'
        + ul([f"{arms['prompt only']} კითხვის (პრომპტის) ფიჩერი: სიგრძე, სიტყვები, რიცხვები, უარყოფა და ა.შ.",
              f"{h1['n_features'] - arms['prompt only']} მოდელის სვეტი: სახელი, ოჯახი, knowledge cutoff, recency gap."])
        + '</div><div><h3>როგორ ნაწილდება სამიზნე</h3>'
        + table(["სტრიქონის ტიპი", "წილი"],
                [["ყოველთვის სწორი (0/10 შეცდომა)", pct(all_right)],
                 ["შუალედური (1–9 შეცდომა)", pct(between)],
                 ["ყოველთვის არასწორი (10/10)", pct(all_wrong)],
                 ["ბინარული = 1 (ნახევარზე მეტი შეცდომა)", pct(h1["binary_positive_rate"])]])
        + '</div></div>',
        "მონაცემები გუნდის ცხრილიდან (gaia-ml-share) დავაკოპირეთ data/gaia_training_dataset.csv-ში. "
        "სტრიქონების უმეტესობა ყოველთვის სწორია ან ყოველთვის არასწორი, ამიტომ ზუსტი სიხშირე ბინარულისგან "
        f"მხოლოდ {pct(between, 0)} სტრიქონზე განსხვავდება."))

    # 1.2 had: teammate notebooks
    slides.append(slide(
        kick(1, 2, "გუნდის ნოუთბუქები"),
        "გუნდს უკვე ჰქონდა ორი XGBoost ნოუთბუქი, მაგრამ შედეგი სანდო არ იყო",
        table(["ნოუთბუქი", "როგორ გაყვეს", "რა შედეგი აჩვენა"],
              [["ბინარული (სიხშირე > 0.5)", "კითხვების მიხედვით", "AUC 0.44: შემთხვევითზე უარესი"],
               ["რეგრესია (ზუსტი სიხშირე)", "სტრიქონების მიხედვით", "RMSE 0.29 (საშუალო 0.36), R² 0.32"]],
              align=["left", "left", "left"])
        + '<div class="two" style="margin-top:22px">'
        + card("პრობლემა 1: გაჟონვა",
               "<p>რეგრესიამ სტრიქონები შემთხვევით გაყო. ერთ კითხვას სამი მოდელი პასუხობს, ამიტომ "
               "<b>სატესტო 145-ვე კითხვა სატრენინგოშიც იყო</b>, უბრალოდ სხვა მოდელის სახელით.</p>", "bad")
        + card("პრობლემა 2: question_id ფიჩერად",
               "<p>ორივე ნოუთბუქში <code>question_id</code> ფიჩერებში დარჩა. მოდელს შეეძლო "
               "კონკრეტული კითხვა „დაემახსოვრებინა“.</p>", "bad")
        + '</div>',
        "დასკვნა: R² 0.32 გადაჭარბებული იყო, მოდელი ტესტის კითხვებს უკვე იცნობდა. ამიტომ H1 თავიდან, "
        "სწორი გაყოფით გავუშვით."))

    # 1.3 did: pipeline
    slides.append(slide(
        kick(1, 3, "ტესტის აგება"),
        "ექვსი ნაბიჯი, რომ შედარება სამართლიანი ყოფილიყო",
        '<ol class="steps">'
        "<li><b>სწორი გაყოფა.</b> 5-fold კროს-ვალიდაცია, დაჯგუფებული კითხვის მიხედვით: კითხვა მთლიანად "
        "ან სატრენინგოშია, ან სატესტოში. fold-ები სირთულით დაბალანსებულია.</li>"
        "<li><b>გაჟონვის წყაროები ამოვიღეთ.</b> <code>question_id</code> და MMLU საგანი ფიჩერი არ არის; "
        "საგანი მხოლოდ baseline-ში გამოიყენება.</li>"
        "<li><b>ორი მოდელის ოჯახი, თითო ორ ვერსიად = 4 მოდელი.</b> XGBoost (ბინარული / რბილი) და "
        "elastic-net ლოგისტიკური (ბინარული / რბილი, 10 პასუხი, თითო ცალკე „ცდად“).</li>"
        f"<li><b>ერთნაირი პირობები.</b> ოთხივე მოდელს ერთი და იგივე {h1['n_features']} ფიჩერი და ერთი და იგივე fold-ები. "
        "XGBoost პატარაა (80 ხე, სიღრმე 3), რადგან 280 კითხვა ცოტაა.</li>"
        "<li><b>საზომი ჯოხი (baseline).</b> „ყოველთვის საშუალოს გამოცნობა“ სატრენინგო ნაწილიდან.</li>"
        "<li><b>სანდოობა.</b> 1,000-ჯერ ხელახლა ავიღეთ კითხვები (bootstrap) და ავაგეთ 95%-იანი ინტერვალი. "
        "სხვაობა მხოლოდ მაშინ ითვლება, თუ ინტერვალი ნულს არ მოიცავს.</li>"
        "</ol>",
        "ყველა მოდელი პროგნოზს ერთ ფორმატში იძლევა: „ალბათობა, რომ ეს პასუხი არასწორი იქნება“. "
        "ასე ბინარული და რბილი მოდელები ერთსა და იმავე სკალაზე დგანან. კოდი: experiments/run_h1.py."))

    # 1.3 did: metrics
    slides.append(slide(
        kick(1, 3, "მეტრიკები (ფაზა 6)"),
        "მეტრიკები შედეგების ნახვამდე დავაფიქსირეთ",
        '<div class="two">'
        + card("თუ პროდუქტი სარისკო X%-ს გამოტოვებს",
               "<p>მხოლოდ <b>რიგი</b> აქვს მნიშვნელობა: რომელი კითხვაა უფრო სარისკო.</p>"
               '<p class="em">გადამწყვეტი: coverage–risk მრუდი, შეცდომა 90 / 80 / 70% დაფარვაზე, AURC.</p>')
        + card("თუ ზღვარს იყენებს ან რისკს აჩვენებს",
               "<p>„გამოტოვე, თუ რისკი > 30%“ მხოლოდ მაშინ მუშაობს, თუ 30% მართლა 30%-ს ნიშნავს.</p>"
               '<p class="em">გადამწყვეტი: Brier, Brier Skill Score, reliability diagram, ECE.</p>')
        + '</div>'
        + table(["მეტრიკა", "რას ზომავს მარტივად", "უკეთესია"],
                [["Brier", "პროგნოზისა და რეალობის საშუალო კვადრატული სხვაობა", "დაბალი"],
                 ["BSS (Brier Skill Score)", "რამდენად ჯობია საშუალოს გამოცნობას; 0 = არაფრით", "მაღალი, > 0"],
                 ["ECE", "რამდენად ემთხვევა „30%“ რეალურ 30%-ს", "დაბალი"],
                 ["coverage–risk / AURC", "შეცდომა, როცა სარისკოებს ვტოვებთ; მრუდის ქვეშა ფართობი", "დაბალი"],
                 ["log loss", "სჯის თავდაჯერებულ შეცდომას", "დაბალი"],
                 ["AUROC", "რამდენად კარგად ალაგებს; 0.5 = შემთხვევითი", "მაღალი"]],
                align=["left", "left", "left"]),
        "metrics.docx-ის მიხედვით. დამატებით: AUPRC საბაზო სიხშირესთან ერთად, შედეგები მოდელისა და საგნის "
        "მიხედვით, RMSE მხოლოდ ხმაურის ზღვართან (0.06) ერთად."))

    # 1.4 result: table
    def h1_row(label, r):
        return [label, f"{r['brier']:.3f}", f"{r['bss_vs_oof_base']:+.3f}",
                f"{r['ece_15_quantile']:.3f}", pct(r['risk_at_coverage_80']),
                f"{r['aurc']:.3f}", f"{r['auroc']:.2f}"]

    gx = boot["regressor_minus_classifier"]["brier"]
    gl = boot["soft_logistic_minus_binary_logistic"]["brier"]
    lsoft_base = boot["delta_vs_oof_base"]["soft logistic"]["brier_minus_base"]
    slides.append(slide(
        kick(1, 4, "მთავარი ცხრილი"),
        "H1 არ დადასტურდა: ზუსტი სიხშირე „კი/არა“-ს ვერ ჯობნის",
        table(["პროგნოზი", "Brier", "BSS", "ECE", "შეცდომა, 20% გამოტოვებით", "AURC", "AUROC"],
              [["ყოველთვის საშუალო", f"{base['brier']:.3f}", "0", f"{base['ece_15_quantile']:.3f}",
                pct(fail) + " (ყველას პასუხობს)", f"{base['aurc']:.3f}", f"{base['auroc']:.2f}"],
               h1_row("XGBoost · ბინარული", binr),
               h1_row("XGBoost · რბილი", soft),
               h1_row("ლოგისტიკური · ბინარული", lbin),
               h1_row("ლოგისტიკური · რბილი", lsoft)],
              highlight={4: "hl"})
        + '<div class="row">'
        + big(f"{gx['mean']:+.3f}", f"XGBoost: რბილი − ბინარული, Brier · 95% CI {ci(gx)}")
        + big(f"{gl['mean']:+.3f}", f"ლოგისტიკური: რბილი − ბინარული · 95% CI {ci(gl)}")
        + big(f"{lsoft['bss_vs_oof_base']:+.3f}", f"საუკეთესო BSS · CI საშუალოსთან {ci(lsoft_base)}")
        + '</div>',
        "ორივე ოჯახში სხვაობის ინტერვალი ნულს მოიცავს: ფრეა. საუკეთესო (რბილი ლოგისტიკური) საშუალოს "
        "გამოცნობასაც ვერ ჯობნის სანდოდ. Brier, ECE, AURC: დაბალი უკეთესია; AUROC: მაღალი."))

    # 1.4 result: coverage
    grid = [100 * g for g in base["coverage_grid"]]
    d80 = boot["delta_vs_oof_base"]["binary classifier"]["risk80_minus_base_rate"]
    slides.append(slide(
        kick(1, 4, "გამოტოვების ეფექტი"),
        "სარისკო კითხვების გამოტოვება შველის, ოღონდ ცოტათი",
        line_chart(
            grid,
            [("XGBoost ბინარული", ACCENT, [100 * v for v in binr["selective_risk"]], False),
             ("XGBoost რბილი", WARN, [100 * v for v in soft["selective_risk"]], False),
             ("ლოგისტიკური რბილი", GOOD, [100 * v for v in lsoft["selective_risk"]], True)],
            0, 25, "კითხვების წილი, რომელსაც ჯერ კიდევ ვპასუხობთ (%)", "შეცდომა ნაპასუხებში (%)",
            refs=[(100 * fail, f"ყველას პასუხი {pct(fail)}", MUTED)], y_step=5),
        f"ყველაზე სარისკო 20%-ის გამოტოვება (ბინარული ქულით) შეცდომას {pct(fail)}-დან "
        f"{pct(binr['risk_at_coverage_80'])}-მდე ამცირებს (კლება {-100 * d80['mean']:.1f} პუნქტი, "
        f"95% CI {ci(d80, 100, 1)}): პატარა, მაგრამ რეალური. მეტის გამოტოვებისას რბილი ლოგისტიკური "
        f"საუკეთესოა: 15% შეცდომა, როცა კითხვების {pct(lsoft['coverage_at_risk_0.15'], 0)}-ს ჯერ კიდევ "
        f"პასუხობს (XGBoost: {pct(binr['coverage_at_risk_0.15'], 0)})."))

    # 1.4 result: calibration
    def share_below_04(r):
        return sum(r["pred_hist"][:4]) / sum(r["pred_hist"])

    rel_blocks = []
    for label, r, color in (("XGBoost ბინარული", binr, ACCENT), ("ლოგისტიკური რბილი", lsoft, GOOD)):
        rel_blocks.append(
            f'<div class="rel"><h3>{esc(label)} · ECE {r["ece_15_quantile"]:.3f}</h3>'
            + reliability_svg(r["reliability"], color, x_label="პროგნოზირებული რისკი",
                              y_label="რეალური შეცდომა")
            + '<p class="small tight">სტრიქონების რაოდენობა თითო 0.1-იან ინტერვალში</p>'
            + hist_svg(r["pred_hist"], color, rest_label="> 0.6") + '</div>')
    slides.append(slide(
        kick(1, 4, "კალიბრაცია"),
        "მოდელები ფრთხილობენ: „მაღალ რისკს“ თითქმის არასდროს ამბობენ",
        '<div class="two">' + "".join(rel_blocks) + '</div>',
        "წყვეტილი ხაზი = იდეალური კალიბრაცია (პროგნოზი 30% → რეალურად 30%). "
        f"პროგნოზების {pct(share_below_04(binr), 0)} (XGBoost) და {pct(share_below_04(lsoft), 0)} "
        f"(ლოგისტიკური) 0.4-ზე დაბალია, თუმცა სტრიქონების {pct(all_wrong)} 10-ვე პასუხში ცდება. "
        "მოდელები საშუალოსკენ იხრებიან, ამიტომაა მათი Brier საშუალოს Brier-თან ასე ახლოს."))

    # 1.4 conclusion
    slides.append(slide(
        kick(1, 4, "დასკვნა და გადაწყვეტილებები"),
        "დავალება 1: რა ვისწავლეთ და რა გადავწყვიტეთ",
        '<div class="two">'
        + card("რა ვისწავლეთ", ul([
            "H1 არ დადასტურდა: ორივე ოჯახში რბილი და ბინარული ფრეა.",
            "ვერც ერთი მოდელი სანდოდ ვერ ჯობნის საშუალოს გამოცნობას.",
            "გამოტოვებისთვის ბინარული ქულა ცოტას შველის (−1.8 პუნქტი 80%-ზე).",
            "გუნდის R² 0.32 გაჟონვის შედეგი იყო.",
        ]))
        + card("რა გადავწყვიტეთ", ul([
            "ტესტი ყოველთვის უცნობ კითხვებზე; <code>question_id</code> და საგანი არასდროს ფიჩერად.",
            "ორივე სამიზნე ვინარჩუნებთ, რადგან ერთმა მეორეს ვერ აჯობა.",
            "20–30 ახალი პასუხი <b>ჯერ არ</b> შევაგროვეთ: არსებული 10 პასუხი ძალიან სანდოა "
            "(split-half 0.975), შედეგი ხმაურზე არ არის დამოკიდებული.",
            "random forest გამოვტოვეთ (plan-ში არასავალდებულოა).",
        ]))
        + '</div>',
        "დეტალები: experiments/out/H1_RESULTS.md · რიცხვები: experiments/out/h1_metrics.json"))

    # ================================================================ TASK 2
    slides.append(
        '<section class="slide title"><p class="kicker">დავალება 2</p>'
        '<h1>H2, H3, H4 ექსპერიმენტების დაგეგმვა</h1>'
        '<p class="lede">გეგმა გაშვებამდე დავწერეთ (H2_H4_PLAN.md), მერე ექსპერიმენტები გავუშვით კიდეც.</p>'
        '</section>')

    # 2.1 asked
    slides.append(slide(
        kick(2, 1),
        "სამი კითხვა plan-ის ჰიპოთეზების ცხრილიდან",
        '<div class="three">'
        + card("H2", "<p>მხოლოდ <b>პრომპტის</b> ფიჩერები vs <b>ყველა</b> ფიჩერი.</p>"
               '<p class="small">საჭიროა თუ არა მოდელის შესახებ ინფორმაცია?</p>')
        + card("H3", "<p>მხოლოდ პრომპტი vs პრომპტი + <b>მოდელის ლიმიტები</b> vs სრული <b>ინტერაქციების</b> მოდელი.</p>"
               '<p class="small">context window, temperature, top-p, output pressure.</p>')
        + card("H4", "<p>ყველა ნასწავლი მოდელი vs მარტივი <b>baseline-ები</b>: საერთო, მოდელის, საგნის, "
               "მოდელი × საგნის საშუალო.</p>"
               '<p class="small">ჯობნის თუ არა ML უბრალო ცხრილს?</p>')
        + '</div>'
        '<h3 style="margin-top:22px">ფაზა 5: ოთხი გაყოფა, გაჟონვის გარეშე</h3>'
        + table(["გაყოფა", "რას ამოწმებს"],
                [["უცნობი კითხვები (question-out)", "ახალი კითხვები, იგივე საგნები და მოდელები"],
                 ["უცნობი საგანი (category-out)", "მთელი საგანი სატესტოშია"],
                 ["უცნობი მოდელი (model-out)", "მოდელი, რომელიც სწავლისას არ გვინახავს; ყველაზე ძლიერი ტესტი"],
                 ["უცნობი კონფიგურაცია", "ახალი temperature ან პარამეტრი, თუ საკმარისი კონფიგურაცია არსებობს"]],
                align=["left", "left"]),
        "plan ასევე ითხოვს: ბოლო სატესტო ნაწილი „დალუქული“ დარჩეს, სანამ ყველა გადაწყვეტილება მიღებული არ არის."))

    # 2.2 had
    slides.append(slide(
        kick(2, 2),
        "რა იცვლება სამ მოდელს შორის და რა არა",
        '<div class="two"><div>'
        + table(["სვეტი", "მნიშვნელობა", "იცვლება?"],
                [["context_window_tokens", "1,048,576 სამივეზე", "არა"],
                 ["temperature, top_p", "ცარიელია", "არა"],
                 ["f_output_pressure", "1 ყველგან", "არა"],
                 ["f_context_pressure", "სამივეზე ერთნაირი", "არა"],
                 ["llm_model, model_family", "3 მოდელი", "კი"],
                 ["knowledge_cutoff_year, f_recency_gap", "flash-latest-ს არ აქვს", "კი"]],
                align=["left", "left", "left"], highlight={4: "hl", 5: "hl"})
        + '</div><div>'
        + card("რას ნიშნავს ეს", ul([
            "<b>H3 სრულად ვერ შემოწმდება:</b> „ლიმიტები“ სამივე მოდელზე ერთნაირია, შესადარებელი არაფერია.",
            "<b>კონფიგურაციის გაყოფა</b> ვერ გაეშვება: მხოლოდ ერთი კონფიგურაცია გვაქვს.",
            "<b>H2 და H4 შესაძლებელია:</b> მოდელის სახელი და cutoff იცვლება, "
            f"{n_subjects} საგანი baseline-ისთვის საკმარისია.",
            "დალუქული ნაწილისთვის 280 კითხვა მეტისმეტად ცოტაა.",
        ]))
        + '</div></div>',
        "იგივე 840 სტრიქონი, რაც H1-ში. H1-ის შედეგი უკვე ვიცოდით: ფრე, ამიტომ H2–H4-ში ოთხივე მოდელს ვინარჩუნებთ."))

    # 2.3 did: plan
    slides.append(slide(
        kick(2, 3, "გეგმა"),
        "გეგმა გაშვებამდე დავწერეთ და დავაფიქსირეთ",
        '<div class="two"><div><h3>სამი ფიჩერების ნაკრები (arm)</h3>'
        + table(["ნაკრები", "ფიჩერი"],
                [["მხოლოდ პრომპტი", str(arms["prompt only"])],
                 ["პრომპტი + მოდელის სახელი", str(arms["prompt + model id"])],
                 ["ყველა (+ cutoff, recency gap)", str(arms["all features"])]])
        + '<p class="small">თითოეულზე ოთხივე H1 მოდელი → 12 ნასწავლი მოდელი.</p>'
        + '<h3>ოთხი baseline (მხოლოდ სატრენინგოდან)</h3>'
        + ul(["საერთო საშუალო", "მოდელის საშუალო", "საგნის საშუალო", "მოდელი × საგნის საშუალო"])
        + '</div><div><h3>სამი გაყოფა</h3>'
        + table(["გაყოფა", "fold"],
                [["უცნობი კითხვები", str(qo["n_folds"])],
                 ["უცნობი საგანი", str(so["n_folds"])],
                 ["უცნობი მოდელი + მისი კითხვები", str(mo["n_folds"])]])
        + card("გადაწყვეტილების წესი",
               "<p>ფიჩერების ნაკრები მხოლოდ მაშინ „იმარჯვებს“, თუ მისი 95% ინტერვალი "
               "<b>ყველაზე ძლიერ baseline-ს</b> სანდოდ ჯობნის.</p>")
        + '</div></div>',
        "H3-ის ნაცვლად შემცირებული ვერსია: პრომპტი → + მოდელის სახელი → + cutoff და recency gap. "
        "კოდი: experiments/run_h2_h4.py."))

    # 2.3 did: leak
    slides.append(slide(
        kick(2, 3, "გაჟონვა ვიპოვეთ"),
        "„უცნობი მოდელის“ ტესტი თავიდან საეჭვოდ კარგი გამოვიდა",
        '<div class="row">'
        + big("0.130", "Brier, როცა მხოლოდ მოდელს ვტოვებდით: საეჭვოდ კარგი", "bad")
        + big(f"{by_name(mo['results'])['prompt only / logit binary']['brier']:.3f}",
              "Brier გასწორების შემდეგ: მოდელიც და მისი კითხვებიც სატესტოშია", "accent")
        + '</div>'
        + '<div class="two">'
        + card("რატომ მოხდა",
               "<p>ერთ კითხვას სამივე მოდელი პასუხობს. როცა ერთ მოდელს ვტოვებდით, დანარჩენი ორის "
               "პასუხები <b>იმავე კითხვებზე</b> სატრენინგოში რჩებოდა. მოდელები ეთანხმებიან ერთმანეთს, "
               "რომელი კითხვაა რთული, ამიტომ პასუხი ფაქტობრივად „გაჟონა“.</p>", "bad")
        + card("როგორ გავასწორეთ",
               f"<p>„უცნობი მოდელის“ გაყოფა კითხვების fold-ებთან გადავკვეთეთ: 3 მოდელი × 5 fold = "
               f"{mo['n_folds']} fold. ტესტში მოდელიც უცნობია და კითხვებიც.</p>", "good")
        + '</div>',
        "0.130-იანი შედეგი არსად არ არის მოყვანილი, როგორც რეალური შედეგი."))

    # 2.4 H2 bars
    arm_ka = {"prompt only": "პრომპტი", "prompt + model id": "+ მოდელი",
              "all features": "ყველა"}
    head_style = (("xgb binary", "XGB ბინ.", ACCENT), ("xgb soft", "XGB რბ.", WARN),
                  ("logit binary", "ლოგ. ბინ.", INK), ("logit soft", "ლოგ. რბ.", GOOD))
    arm_rows = [(f"{arm_ka[arm]} · {label}", q[f"{arm} / {head}"]["brier"], color)
                for arm in arm_ka for head, label, color in head_style]
    h2x = qo["h2_prompt_minus_all"]["xgb binary"]["brier"]
    h2l = qo["h2_prompt_minus_all"]["logit soft"]["brier"]
    slides.append(slide(
        kick(2, 4, "H2 და შემცირებული H3"),
        "მოდელის მონაცემები ცოტას შველის, baseline-მდე მაინც ვერ აღწევს",
        bar_chart(arm_rows, 0.150, 0.175,
                  "Brier უცნობ კითხვებზე (დაბალი უკეთესია; ღერძი 0.150-დან იწყება)",
                  ticks=(0.150, 0.155, 0.160, 0.165, 0.170, 0.175), bar_h=15, gap=6,
                  refs=[(q["global mean"]["brier"], "საშუალო", MUTED),
                        (q["model x subject"]["brier"], "მოდელი × საგანი", GOOD)]),
        f"H2 (პრომპტი − ყველა): XGB ბინ. {h2x['mean']:+.4f} {ci(h2x, d=4)}, ლოგ. რბ. "
        f"{h2l['mean']:+.4f} {ci(h2l, d=4)}: ნულს არ მოიცავს, რეალური, მაგრამ პატარა. "
        "შემცირებული H3: მოდელის სახელი შველის, cutoff მხოლოდ XGBoost-ს."))

    # 2.4 H4 table
    split_ka = (("უცნობი კითხვები", qo), ("უცნობი საგანი", so), ("უცნობი მოდელი + კითხვები", mo))
    best_ka = {"prompt only": "პრომპტი", "prompt + model id": "პრომპტი + მოდელი", "all features": "ყველა"}
    head_ka = {"xgb binary": "XGB ბინარული", "xgb soft": "XGB რბილი",
               "logit binary": "ლოგ. ბინარული", "logit soft": "ლოგ. რბილი"}

    def split_row(label, e):
        r = by_name(e["results"])
        arm, head = e["best_learned"].split(" / ")
        g = e["best_vs_model_x_subject"]["brier"]
        return [label, f"{r['global mean']['brier']:.3f}", f"{r['model x subject']['brier']:.3f}",
                f"{r[e['best_learned']]['brier']:.3f}", f"{best_ka[arm]} · {head_ka[head]}", ci(g, d=4)]

    r80 = qo["best_vs_model_x_subject"]["risk80"]
    slides.append(slide(
        kick(2, 4, "H4"),
        "სამივე გაყოფაზე ML ვერ ჯობნის მარტივ ცხრილს „მოდელი × საგანი“",
        table(["გაყოფა", "საშუალო", "მოდელი × საგანი", "საუკეთესო ML", "საუკეთესო ML მოდელი",
               "ML − ცხრილი, 95% CI"],
              [split_row(lbl, e) for lbl, e in split_ka],
              align=["left", "right", "right", "right", "left", "right"])
        + '<div class="row">'
        + big("0 / 3", "გაყოფა, სადაც ML ცხრილს ჯობნის", "bad")
        + big(f"+{100 * r80['mean']:.1f} პუნქტი", "ML-ის მეტი შეცდომა 80% დაფარვაზე, უცნობ კითხვებზე · "
              f"CI {ci(r80, 100, 1)}", "bad")
        + '</div>',
        "Brier, დაბალი უკეთესია. დადებითი სხვაობა = ML უარესია. Brier-ის ინტერვალები ნულს ძლივს ეხება: "
        "ცხრილი საშუალოდ უკეთესია, ML-ის მოგება არსად ჩანს; დალაგებაში კი ცხრილი სანდოდ ჯობნის. "
        "უცნობ საგანზე ცხრილი მხოლოდ მოდელის საშუალოს იყენებს, უცნობ მოდელზე მხოლოდ საგნისას."))

    # 2.4 by model
    bm = {}
    for row in h1["by_model"]:
        bm.setdefault(row["llm_model"], {})[row["predictor"]] = row
    model_rows = [[m, pct(v["base rate"]["fail_rate"]), f"{v['base rate']['brier']:.3f}",
                   f"{v['binary classifier']['brier']:.3f}", f"{v['soft logistic']['brier']:.3f}"]
                  for m, v in sorted(bm.items(), key=lambda kv: -kv[1]["base rate"]["fail_rate"])]
    bs = {}
    for row in h1["by_subject"]:
        bs.setdefault(row["subject"], {})[row["predictor"]] = row
    hardest = max(bs, key=lambda k: bs[k]["base rate"]["fail_rate"])
    easiest = min(bs, key=lambda k: bs[k]["base rate"]["fail_rate"])
    slides.append(slide(
        kick(2, 4, "რატომ არის ცხრილი ასე ძლიერი"),
        "მოდელებსა და საგნებს შორის სხვაობა ფიჩერების ეფექტზე დიდია",
        table(["მოდელი", "შეცდომის სიხშირე", "Brier · საშუალო", "Brier · XGB ბინ.", "Brier · ლოგ. რბილი"],
              model_rows)
        + '<div class="row">'
        + big(f"{pct(bs[hardest]['base rate']['fail_rate'], 0)} vs {pct(bs[easiest]['base rate']['fail_rate'], 0)}",
              f"შეცდომის სიხშირე: {hardest} vs {easiest}", "accent")
        + '</div>',
        "„მოდელი × საგანი“ ორივე სხვაობას პირდაპირ იჭერს. კითხვის ტექსტის ფიჩერებს ამაზე მეტის თქმა "
        "280 კითხვაზე ჯერ არ შეუძლიათ."))

    # 2.4 conclusion
    slides.append(slide(
        kick(2, 4, "დასკვნა და გადაწყვეტილებები"),
        "დავალება 2: რა ვისწავლეთ და რა გადავწყვიტეთ",
        '<div class="two">'
        + card("რა ვისწავლეთ", ul([
            "<b>H2:</b> მოდელის სვეტები ცოტას, მაგრამ სანდოდ შველის.",
            "<b>H3:</b> მხოლოდ შემცირებული ვერსია; მოდელის სახელი შველის.",
            "<b>H4:</b> ვერც ერთი ნასწავლი მოდელი ვერ ჯობნის „მოდელი × საგანი“ ცხრილს, სამივე გაყოფაზე.",
            "„უცნობი მოდელის“ ტესტში გაჟონვა ვიპოვეთ და გავასწორეთ.",
        ]))
        + card("რა გადავწყვიტეთ", ul([
            "საზომი ჯოხი ამიერიდან „მოდელი × საგანია“, არა საერთო საშუალო.",
            "„უცნობი მოდელი“ ყოველთვის კითხვების fold-ებთან ერთად.",
            "სრული H3 და კონფიგურაციის გაყოფა მეორე კონფიგურაციამდე (მაგ. მეორე temperature) გადავდეთ.",
            "დალუქულ ნაწილს 1,000+ კითხვის ეტაპზე გამოვყოფთ, ფიჩერების არჩევამდე.",
        ]))
        + '</div>',
        "გეგმა: experiments/out/H2_H4_PLAN.md · რიცხვები: experiments/out/h2_h4_metrics.json"))

    # ================================================================ next
    null = pw["null_max_rho"]
    xs = [s["n_questions"] for s in sizes]
    n80 = next(r for r in pw["needed_for_power"] if abs(r["rho"] - 0.165) < 1e-9)
    slides.append(slide(
        "დამატებით · რამდენი მონაცემია საჭირო (API ხარჯის გარეშე)",
        "280 კითხვაზე საუკეთესო ფიჩერი ხმაურისგან არ გამოირჩევა",
        '<div class="split"><div>'
        + line_chart(
            xs, [("საჭირო ზღვარი", INK, [s["rho_threshold"] for s in sizes], False)],
            0.0, 0.25, "კითხვების რაოდენობა", "კავშირის სიძლიერე (|Spearman ρ|)",
            refs=[(pw["observed_max_rho"], f"ჩვენი საუკეთესო {pw['observed_max_rho']:.3f}", ACCENT),
                  (null["p95"], f"შემთხვევითი 95% {null['p95']:.3f}", BAD)],
            w=640, h=360, y_fmt="{:.2f}", y_step=0.05)
        + '</div><div class="stack">'
        + big(pct(null["share_at_or_above_observed"], 0), "შემთხვევით არეულ ლეიბლებზეც ამდენჯერ ჩნდება ასეთივე „ძლიერი“ ფიჩერი", "bad")
        + big(pct(pw["stability"][-1]["top10_overlap_mean"], 0), "top-10 ფიჩერიდან რამდენი რჩება ხელახალ შერჩევაზე", "bad")
        + big(f"{n80['n_questions']:,}", "კითხვა, რომ ეს სიგნალი 80%-ით ვიპოვოთ", "accent")
        + '</div></div>',
        f"შევამოწმეთ {pw['n_features_tested']} რიცხვითი ფიჩერი. საუკეთესოა "
        f"<code>{esc(pw['observed_top10'][0]['feature'])}</code> (ρ {pw['observed_top10'][0]['rho']:+.3f}), "
        f"მაგრამ 280 კითხვაზე მრავალჯერადი ტესტირების ზღვარი {sizes[0]['rho_threshold']:.3f}-ია."))

    cost_rows = [[f"{s['n_questions']:,}", f"{s['rho_threshold']:.3f}", pct(s["power_at_observed_rho"], 0),
                  f"{s['calls']:,}", f"${s['cost_usd_new_questions_only']:.0f}"] for s in sizes]
    slides.append(slide(
        "დამატებით · რა ღირს მეტი მონაცემი",
        f"გამოსავალი იაფია: {n1000['n_questions']:,} კითხვა დაახლოებით ${n1000['cost_usd_new_questions_only']:.0f}",
        table(["კითხვები", "საჭირო |ρ|", "ρ = 0.165-ის პოვნის შანსი", "API გამოძახება", "ახალი კითხვების ფასი"],
              cost_rows, highlight={xs.index(1000): "hl"}),
        "3 მოდელი × 10 პასუხი თითო კითხვაზე; დალუქულ 20%-ს 25 პასუხი. ფასები Google-ის (2026 სექტემბერი): "
        "2.5 Flash-Lite $0.10/$0.40, 3.1 Flash-Lite $0.25/$1.50, Flash latest $0.75/$3.75 1M ტოკენზე."))

    slides.append(slide(
        "შემდეგი ნაბიჯები",
        "მონაცემები გავზარდოთ და იგივე სკრიპტები ხელახლა გავუშვათ",
        '<ol class="steps">'
        f"<li><b>{n1000['n_questions']:,} კითხვის შეგროვება</b> (≈{n1000['n_questions'] // n_subjects} თითო საგანში) × "
        f"იგივე 3 მოდელი × 10 პასუხი: ≈{n1000['calls']:,} გამოძახება, ≈${n1000['cost_usd_new_questions_only']:.0f}.</li>"
        "<li><b>20%-ის დალუქვა თავიდანვე</b>, 25 პასუხით. მხოლოდ ბოლოს, ერთხელ ვეხებით.</li>"
        "<li><b>იგივე სკრიპტების გაშვება ცვლილების გარეშე:</b> <code>run_h1.py</code>, <code>run_h2_h4.py</code>, "
        "<code>run_power.py</code>. იგივე მეტრიკები, იგივე გაყოფები, იგივე წესი.</li>"
        "<li><b>H3-ისთვის მეორე temperature</b> მონაცემების ნაწილზე. მის გარეშე მოდელის ლიმიტები ვერ შემოწმდება.</li>"
        "</ol>",
        "ფიჩერების ნაკრები მხოლოდ მაშინ მიიღება, თუ მისი 95% ინტერვალი დალუქულ ნაწილზე „მოდელი × საგანს“ ჯობნის."))

    return slides


KA_CSS = CSS.replace(
    'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif',
    'font-family:"Segoe UI","Noto Sans Georgian","BPG Nino Mtavruli",Sylfaen,Arial,sans-serif'
) + f"""
.kicker{{text-transform:none;letter-spacing:.02em}}
.three{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-top:10px}}
.card.bad{{border-color:{BAD}55;background:#fff8f8}} .card.good{{border-color:{GOOD}55;background:#f6fff8}}
.card ul,.card ol{{margin:6px 0 0;padding-left:20px}} .card p{{margin:6px 0}}
.slide.title .two{{margin-top:24px}}
table{{font-size:16px}} p,li{{font-size:17px}}
.slide{{padding:34px 56px}} h2{{font-size:28px;margin:0 0 16px}} td{{padding:10px 12px}}
.note{{margin-top:10px;padding-top:10px}}
"""

KA_JS = JS.replace("arrow keys or click", "ისრები ან დაწკაპუნება")


def main():
    slides = build()
    doc = ("<!doctype html><html lang=\"ka\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           "<title>ფაზები 4–5 · H1 და H2–H4</title>"
           f"<style>{KA_CSS}</style></head><body>{''.join(slides)}"
           f"<div id=\"bar\"></div><script>{KA_JS}</script></body></html>")
    DECK.write_text(doc, encoding="utf-8")
    print("wrote", DECK, f"({len(slides)} slides)")


if __name__ == "__main__":
    main()
