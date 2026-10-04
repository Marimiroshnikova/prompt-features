"""Build the short Georgian Phase 4-5 deck from the saved result files.

One slide per hypothesis (claim, what we did, result), plus data, the split fix,
a summary and the next step. Reads the same JSON files as build_deck.py and
writes out/phase4_5_deck_ka.html.

    python experiments/build_deck_ka.py
"""

from __future__ import annotations

from build_deck import (ACCENT, BAD, CSS, FAINT, GOOD, INK, JS, MUTED, OUT, WARN,
                        bar_chart, by_name, esc, load, pct, table)

DECK = OUT / "phase4_5_deck_ka.html"
HEADS = ("xgb binary", "xgb soft", "logit binary", "logit soft")


def page(kicker, title, body):
    return (f'<section class="slide"><p class="kicker">{esc(kicker)}</p>'
            f'<h2>{esc(title)}</h2><div class="body">{body}</div></section>')


def box(title, body, tone=""):
    return f'<div class="box {tone}"><h3>{title}</h3>{body}</div>'


def num(value, label, tone=""):
    return f'<div class="num {tone}"><b>{esc(value)}</b><span>{label}</span></div>'


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def before_after(left_title, left, right_title, right, left_tone="bad", right_tone="good"):
    return ('<div class="ps">' + box(left_title, left, left_tone) + '<div class="arrow">→</div>'
            + box(right_title, right, right_tone) + '</div>')


def hyp(claim, did, result, verdict, tone, right):
    """Hypothesis slide body: three boxes on the left, a chart or table on the right."""
    return ('<div class="hyp"><div class="stack">'
            + box("ჰიპოთეზა", f"<p>{claim}</p>")
            + box("რა გავაკეთეთ", f"<p>{did}</p>")
            + box(f"შედეგი: {verdict}", f"<p>{result}</p>", tone)
            + f'</div><div>{right}</div></div>')


def build():
    h1 = load("h1_metrics.json")
    h24 = load("h2_h4_metrics.json")
    pw = load("power_metrics.json")

    r1 = by_name(h1["results"])
    base, xb, xs = r1["base rate"], r1["binary classifier"], r1["soft regressor"]
    lb, ls = r1["binary logistic"], r1["soft logistic"]
    splits = h24["splits"]
    q = by_name(splits["question-out"]["results"])
    mo = by_name(splits["model-out"]["results"])
    fail = h1["fail_rate"]
    n1000 = next(s for s in pw["sizes"] if s["n_questions"] == 1000)
    n_subjects = len({row["subject"] for row in h1["by_subject"]})

    def best_of(arm):
        return min(q[f"{arm} / {h}"]["brier"] for h in HEADS)

    def split_pair(key):
        r = by_name(splits[key]["results"])
        return r[splits[key]["best_learned"]]["brier"], r["model x subject"]["brier"]

    s = []

    # 1 title
    s.append(
        '<section class="slide title">'
        '<p class="kicker">Phase 4–5 · MMLU-Pro</p>'
        '<h1>შეგვიძლია თუ არა კითხვის ტექსტით წინასწარ ვთქვათ, შეცდება თუ არა LLM?</h1>'
        '<div class="two">'
        + box("დავალება 1", "<p>H1 ჰიპოთეზის ტესტირება</p>")
        + box("დავალება 2", "<p>H2, H3, H4 ექსპერიმენტების დაგეგმვა და გაშვება</p>")
        + '</div>'
        '<p class="answer">მოკლე პასუხი: <b>ჯერ ვერა.</b> 280 კითხვა ცოტაა. '
        f'საჭიროა დაახლოებით {n1000["n_questions"]:,} კითხვა (≈${n1000["cost_usd_new_questions_only"]:.0f}).</p>'
        '</section>')

    # 2 data + how we measure
    s.append(page(
        "რა გვქონდა",
        "მონაცემები და როგორ ვზომავთ",
        '<div class="nums">'
        + num("280", f"კითხვა<br>MMLU-Pro, {n_subjects} საგანი")
        + num("3", "Gemini მოდელი")
        + num("10", "პასუხი თითო<br>კითხვაზე")
        + num(pct(fail), "პასუხი არასწორი", "bad")
        + '</div>'
        '<div class="two">'
        + box("Brier score",
              "<p>მოდელი ამბობს: „ამ პასუხის შეცდომის ალბათობა X%“. Brier ზომავს, რამდენად ასცდა.</p>"
              "<p><b>0 = იდეალური. რაც დაბალია, მით უკეთესი.</b></p>")
        + box(f"Baseline = {base['brier']:.3f}",
              f"<p>ყოველ კითხვაზე უბრალოდ „{pct(fail)}“-ის თქმა, კითხვის წაკითხვის გარეშე.</p>"
              "<p><b>მოდელი ამაზე დაბალს უნდა იღებდეს.</b></p>")
        + '</div>'))

    # 3 split before -> after
    s.append(page(
        "რა გავაკეთეთ პირველ რიგში",
        "როგორ ვყოფთ მონაცემებს train-ად და test-ად",
        before_after(
            "ადრე (გუნდის notebook)",
            ul(["840 სტრიქონი შემთხვევით 80/20-ზე",
                "ერთი კითხვა = 3 სტრიქონი (3 მოდელი) → <b>train-შიც და test-შიც</b> ხვდებოდა",
                "<code>question_id</code> feature-ად იყო",
                "<b>R² 0.32</b>: მოდელმა test-ის კითხვები უკვე იცოდა"]),
            "ახლა (ჩვენი კოდი)",
            ul(["ჯერ <b>კითხვებს</b> ვყოფთ, მერე სტრიქონებს",
                "კითხვა სამივე მოდელით ერთად: <b>ან train-შია, ან test-ში</b>",
                "<code>question_id</code> ამოღებულია",
                "test-ში მხოლოდ <b>ახალი</b> კითხვებია → R² 0.32 გაქრა"]))))

    # 4 H1
    rows = [("baseline", base["brier"], MUTED),
            ("XGBoost · binary", xb["brier"], ACCENT),
            ("XGBoost · soft", xs["brier"], WARN),
            ("Logistic · binary", lb["brier"], INK),
            ("Logistic · soft", ls["brier"], GOOD)]
    s.append(page(
        "დავალება 1 · H1",
        "H1: soft target ჯობია binary-ს?",
        hyp("მოდელი უკეთესია, თუ ვასწავლით <b>soft target</b>-ზე (3/10 = 0.3), "
            "და არა <b>binary</b>-ზე (0 ან 1).",
            "4 მოდელი: XGBoost და Logistic, თითო ორივე target-ზე. ერთნაირი 144 feature, test-ში ახალი კითხვები.",
            "soft ≈ binary, სხვაობა შემთხვევითია. baseline-ს ვერცერთი ჯობნის. "
            f"<br><b>+</b> 20% სარისკოს გამოტოვებით შეცდომა {pct(fail)} → {pct(xb['risk_at_coverage_80'])}.",
            "✗ არ დადასტურდა", "bad",
            bar_chart(rows, 0.155, 0.172, "Brier (რაც დაბალია, მით უკეთესი)",
                      ticks=(0.155, 0.160, 0.165, 0.170), bar_h=34, gap=14, w=520,
                      refs=[(base["brier"], "baseline", MUTED)]))))

    # 5 H2
    rows2 = [("მხოლოდ კითხვის features", best_of("prompt only"), ACCENT),
             ("+ მოდელის სახელი", best_of("prompt + model id"), ACCENT),
             ("ყველა feature", best_of("all features"), ACCENT),
             ("baseline", q["global mean"]["brier"], MUTED)]
    s.append(page(
        "დავალება 2 · H2",
        "H2: მოდელის features შველის?",
        hyp("კითხვის features-ს თუ <b>მოდელის features</b>-ს დავუმატებთ (სახელი, ოჯახი, knowledge cutoff), "
            "პროგნოზი გაუმჯობესდება.",
            "3 feature-ნაკრები, თითოეული 4 მოდელით. ვადარებთ თითოეულის საუკეთესოს.",
            f"Brier {best_of('prompt only'):.3f} → {best_of('prompt + model id'):.3f}. "
            "სხვაობა რეალურია, მაგრამ პატარა. baseline-ს სანდოდ ვერ ჯობნის.",
            "✓ ცოტათი", "good",
            bar_chart(rows2, 0.155, 0.168, "Brier (რაც დაბალია, მით უკეთესი)",
                      ticks=(0.155, 0.160, 0.165), bar_h=34, gap=14, w=520,
                      refs=[(q["global mean"]["brier"], "baseline", MUTED)]))))

    # 6 H3
    s.append(page(
        "დავალება 2 · H3",
        "H3: მოდელის პარამეტრები შველის?",
        hyp("მოდელის პარამეტრები (context window, temperature, top-p) და მათი კავშირი კითხვასთან "
            "პროგნოზს კიდევ უფრო აუმჯობესებს.",
            "შევამოწმეთ, იცვლება თუ არა ეს პარამეტრები. არ იცვლება, ამიტომ გავუშვით "
            "<b>reduced H3</b>: + მოდელის სახელი, + knowledge cutoff.",
            "სრულად ვერ შემოწმდა. მოდელის სახელი შველის, cutoff თითქმის არა. "
            "საჭიროა მონაცემები მეორე temperature-ით.",
            "◐ ნაწილობრივ", "",
            '<h3>რა იცვლება 3 მოდელს შორის</h3>'
            + table(["პარამეტრი", "მნიშვნელობა", "იცვლება?"],
                    [["context window", "1,048,576 სამივეზე", "✗"],
                     ["temperature, top-p", "არ არის ჩაწერილი", "✗"],
                     ["მოდელის სახელი", "3 სხვადასხვა", "✓"],
                     ["knowledge cutoff", "2 მოდელს აქვს", "✓"]],
                    align=["left", "left", "center"], highlight={2: "hl", 3: "hl"}))))

    # 7 H4
    pairs = [("ახალი კითხვები", "question-out"), ("ახალი საგანი", "subject-out"),
             ("ახალი მოდელი", "model-out")]
    rows4 = []
    for label, key in pairs:
        ml, tab = split_pair(key)
        rows4 += [(f"{label} · ML", ml, ACCENT), (f"{label} · ცხრილი", tab, GOOD)]
    s.append(page(
        "დავალება 2 · H4",
        "H4: ML ჯობნის მარტივ ცხრილს?",
        hyp("ML მოდელი ჯობნის მარტივ ცხრილს: <b>„მოდელი × საგანი“</b> შეცდომის % "
            "(მაგ. „gemini-2.5 health-ში X%-ს ცდება“).",
            "3 ტესტი: ახალი კითხვები, ახალი საგანი, ახალი მოდელი. "
            f"leakage ვიპოვეთ და გავასწორეთ (0.130 → {mo['prompt only / logit binary']['brier']:.3f}).",
            "ცხრილი სამივე ტესტზე ჯობნის ან ტოლია. ML ცხრილს ვერსად აჯობა.",
            "✗ არ დადასტურდა", "bad",
            bar_chart(rows4, 0.150, 0.175, "Brier (რაც დაბალია, მით უკეთესი)",
                      ticks=(0.150, 0.160, 0.170), bar_h=26, gap=10, w=520))))

    # 8 summary
    s.append(page(
        "შეჯამება",
        "ოთხი ჰიპოთეზა: რა გამოვიდა",
        table(["", "ჰიპოთეზა", "შედეგი"],
              [["H1", "soft target ჯობია binary-ს", "✗ ფრეა"],
               ["H2", "მოდელის features შველის", "✓ ცოტათი"],
               ["H3", "მოდელის პარამეტრები შველის", "◐ ვერ შემოწმდა, მონაცემები არ იყო"],
               ["H4", "ML ჯობნის მარტივ ცხრილს", "✗ ცხრილი ჯობნის"]],
              align=["left", "left", "left"])
        + '<p class="answer">280 კითხვაზე კითხვის ტექსტი შეცდომის პროგნოზს თითქმის არაფერს მატებს. '
        'მთავარია, <b>რომელი მოდელია</b> და <b>რომელი საგანი</b>.</p>'))

    # 9 next
    null = pw["null_max_rho"]
    s.append(page(
        "შემდეგი ნაბიჯი",
        "მთავარი პრობლემა: მონაცემები ცოტაა",
        before_after(
            "პრობლემა",
            ul(["280 კითხვაზე საუკეთესო feature <b>ხმაურისგან არ განსხვავდება</b>",
                f"შემთხვევით არეულ მონაცემებზეც {pct(null['share_at_or_above_observed'], 0)} "
                "შემთხვევაში ჩნდება ასეთივე „ძლიერი“ feature"]),
            "გადაწყვეტა",
            ul([f"<b>{n1000['n_questions']:,} კითხვა</b> × 3 მოდელი × 10 პასუხი",
                f"≈{n1000['calls']:,} API call, <b>≈${n1000['cost_usd_new_questions_only']:.0f}</b>",
                "20% test თავიდანვე „დალუქული“",
                "იგივე კოდი, იგივე მეტრიკები"]))))

    return s


KA_CSS = CSS.replace(
    'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif',
    'font-family:"Segoe UI","Noto Sans Georgian",Sylfaen,Arial,sans-serif'
) + f"""
.kicker{{text-transform:none;letter-spacing:.02em;font-size:15px}}
.slide{{padding:40px 60px}}
h1{{font-size:38px}} h2{{font-size:32px;margin:0 0 22px}}
h3{{font-size:19px;margin:0 0 6px}}
p,li{{font-size:19px;line-height:1.45}}
.box{{border:1px solid {FAINT};border-radius:12px;padding:14px 20px;background:#fff}}
.box p{{margin:2px 0}} .box ul{{margin:4px 0 0;padding-left:22px}} .box li{{margin:4px 0}}
.box.bad{{border-color:{BAD}66;background:#fff6f6}} .box.bad h3{{color:{BAD}}}
.box.good{{border-color:{GOOD}66;background:#f3fbf5}} .box.good h3{{color:{GOOD}}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:18px}}
.hyp{{display:grid;grid-template-columns:.85fr 1.15fr;gap:24px;align-items:center}}
.hyp .box p{{font-size:18px}}
.stack{{display:flex;flex-direction:column;gap:12px}}
.ps{{display:grid;grid-template-columns:1fr 50px 1fr;gap:10px;align-items:stretch}}
.arrow{{display:flex;align-items:center;justify-content:center;font-size:40px;color:{MUTED}}}
.nums{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:8px}}
.num{{border:1px solid {FAINT};border-radius:12px;padding:18px;text-align:center}}
.num b{{display:block;font-size:48px;line-height:1.1}} .num span{{color:{MUTED};font-size:17px}}
.num.bad b{{color:{BAD}}}
.answer{{margin-top:22px;padding:14px 20px;background:#f6f8fa;border-left:4px solid {ACCENT};border-radius:6px}}
.slide.title h1{{max-width:1000px}}
table{{font-size:18px}} td{{padding:12px}}
.chart .lab{{font-size:16px}} .chart .val{{font-size:16px}} .chart .axis{{font-size:15px}}
"""

KA_JS = JS.replace("arrow keys or click", "ისრები ან დაწკაპუნება")


def main():
    slides = build()
    doc = ("<!doctype html><html lang=\"ka\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           "<title>Phase 4–5 · H1 და H2–H4</title>"
           f"<style>{KA_CSS}</style></head><body>{''.join(slides)}"
           f"<div id=\"bar\"></div><script>{KA_JS}</script></body></html>")
    DECK.write_text(doc, encoding="utf-8")
    print("wrote", DECK, f"({len(slides)} slides)")


if __name__ == "__main__":
    main()
