"""Build the short Georgian Phase 4-5 deck (10 slides) from the saved result files.

One idea per slide, problem -> solution where there was one. Reads the same
JSON files as build_deck.py and writes out/phase4_5_deck_ka.html.

    python experiments/build_deck_ka.py
"""

from __future__ import annotations

from build_deck import (ACCENT, BAD, CSS, FAINT, GOOD, INK, JS, MUTED, OUT, WARN,
                        bar_chart, by_name, esc, load, pct)

DECK = OUT / "phase4_5_deck_ka.html"


def page(kicker, title, body):
    return (f'<section class="slide"><p class="kicker">{esc(kicker)}</p>'
            f'<h2>{esc(title)}</h2><div class="body">{body}</div></section>')


def box(title, body, tone=""):
    return f'<div class="box {tone}"><h3>{title}</h3>{body}</div>'


def num(value, label, tone=""):
    return f'<div class="num {tone}"><b>{esc(value)}</b><span>{label}</span></div>'


def ps(problem, solution):
    return ('<div class="ps">'
            + box("პრობლემა", problem, "bad")
            + '<div class="arrow">→</div>'
            + box("გადაწყვეტა", solution, "good")
            + '</div>')


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def build():
    h1 = load("h1_metrics.json")
    h24 = load("h2_h4_metrics.json")
    pw = load("power_metrics.json")

    r1 = by_name(h1["results"])
    base, xb, xs = r1["base rate"], r1["binary classifier"], r1["soft regressor"]
    lb, ls = r1["binary logistic"], r1["soft logistic"]
    boot = h1["bootstrap"]
    qo = h24["splits"]["question-out"]
    q = by_name(qo["results"])
    mo = by_name(h24["splits"]["model-out"]["results"])
    fail = h1["fail_rate"]
    n1000 = next(s for s in pw["sizes"] if s["n_questions"] == 1000)
    n_subjects = len({row["subject"] for row in h1["by_subject"]})

    def best_of(arm):
        return min(q[f"{arm} / {h}"]["brier"] for h in ("xgb binary", "xgb soft", "logit binary", "logit soft"))

    s = []

    # 1 title
    s.append(
        '<section class="slide title">'
        '<p class="kicker">Phase 4–5 · MMLU-Pro</p>'
        '<h1>შეგვიძლია თუ არა კითხვის ტექსტით წინასწარ ვთქვათ, შეცდება თუ არა LLM?</h1>'
        '<div class="two">'
        + box("დავალება 1", "<p>H1 ჰიპოთეზის ტესტირება</p>")
        + box("დავალება 2", "<p>H2, H3, H4 ექსპერიმენტების დაგეგმვა</p>")
        + '</div>'
        '<p class="answer">მოკლე პასუხი: <b>ჯერ ვერა.</b> 280 კითხვა ცოტაა. '
        f'საჭიროა დაახლოებით {n1000["n_questions"]:,} კითხვა (≈${n1000["cost_usd_new_questions_only"]:.0f}).</p>'
        '</section>')

    # 2 task 1 asked
    dots = "".join(f'<span class="dot {"x" if i < 3 else "ok"}">{"✗" if i < 3 else "✓"}</span>'
                   for i in range(10))
    s.append(page(
        "დავალება 1 · რას გვთხოვდნენ",
        "H1: რომელი target ჯობია, binary თუ soft?",
        '<p class="center">ერთი კითხვა, ერთი მოდელი, 10 პასუხი:</p>'
        f'<div class="dots">{dots}</div>'
        '<div class="two">'
        + box("Binary target", "<p>შეცდა ნახევარზე მეტჯერ?</p><p class=\"bignum\">0</p>"
              "<p class=\"muted\">მხოლოდ „კი“ ან „არა“</p>")
        + box("Soft target", "<p>რამდენჯერ შეცდა?</p><p class=\"bignum\">0.3</p>"
              "<p class=\"muted\">ზუსტი სიხშირე: 3 / 10</p>")
        + '</div>'
        '<p class="answer">H1 ამბობს: soft target-ზე ნასწავლი მოდელი უფრო ზუსტ პროგნოზს მოგვცემს.</p>'))

    # 3 task 1 had
    s.append(page(
        "დავალება 1 · რა ინფორმაცია გვქონდა",
        "მონაცემები: 840 სტრიქონი",
        '<div class="nums">'
        + num("280", f"კითხვა<br>MMLU-Pro, {n_subjects} საგანი")
        + num("3", "Gemini მოდელი")
        + num("10", "პასუხი თითო<br>კითხვაზე")
        + num(pct(fail), "პასუხი არასწორი", "bad")
        + '</div>'
        '<div class="two">'
        + box("Features: 144", ul(["140 კითხვის ტექსტიდან (სიგრძე, სიტყვები, რიცხვები…)",
                                   "4 მოდელზე (სახელი, ოჯახი, knowledge cutoff)"]))
        + box("გუნდის ნამუშევარი", ul(["2 XGBoost notebook: binary და regression",
                                       "regression-ის შედეგი: R² 0.32, კარგად გამოიყურებოდა"]))
        + '</div>'))

    # 4 problem: leak in notebooks
    s.append(page(
        "დავალება 1 · რა გავაკეთეთ",
        "ჯერ გუნდის შედეგი შევამოწმეთ",
        ps(ul(["მონაცემები <b>rows-ით</b> გაიყო, არა კითხვებით",
               "test-ის 145-ვე კითხვა <b>train-შიც</b> იყო (სხვა მოდელის სახელით)",
               "<code>question_id</code> feature-ად იყო ჩართული",
               "→ R² 0.32 <b>მოჩვენებითია</b>: მოდელმა კითხვები უკვე „იცოდა“"]),
           ul(["split <b>კითხვების მიხედვით</b> (question-grouped 5-fold)",
               "ერთი კითხვა ან train-შია, ან test-ში, ორივეში არასდროს",
               "<code>question_id</code> და საგანი feature-ებიდან ამოვიღეთ",
               "→ ტესტი ახლა <b>ახალ</b> კითხვებზეა"]))))

    # 5 what we did
    s.append(page(
        "დავალება 1 · რა გავაკეთეთ",
        "4 მოდელი, ერთნაირ პირობებში",
        '<div class="split"><div class="grid2">'
        + box("XGBoost · binary", "")
        + box("XGBoost · soft", "")
        + box("Logistic · binary", "")
        + box("Logistic · soft", "")
        + '</div><div>'
        + ul(["<b>ერთი და იგივე</b> 144 feature და ერთი და იგივე split",
              "<b>Baseline:</b> ყოველთვის საშუალოს თქმა (20.5%)",
              "<b>მთავარი მეტრიკა:</b> Brier (რაც დაბალია, მით უკეთესი)",
              "<b>სანდოობა:</b> 95% CI (bootstrap, 1,000-ჯერ)"])
        + '</div></div>'))

    # 6 H1 result
    rows = [("საშუალო (baseline)", base["brier"], MUTED),
            ("XGBoost · binary", xb["brier"], ACCENT),
            ("XGBoost · soft", xs["brier"], WARN),
            ("Logistic · binary", lb["brier"], INK),
            ("Logistic · soft", ls["brier"], GOOD)]
    gx = boot["regressor_minus_classifier"]["brier"]
    gl = boot["soft_logistic_minus_binary_logistic"]["brier"]
    s.append(page(
        "დავალება 1 · რა შედეგი მივიღეთ",
        "H1 არ დადასტურდა: soft ≈ binary",
        '<div class="split"><div>'
        + bar_chart(rows, 0.155, 0.172, "Brier (რაც დაბალია, მით უკეთესი)",
                    ticks=(0.155, 0.160, 0.165, 0.170), bar_h=34, gap=14, w=560,
                    refs=[(base["brier"], "საშუალო", MUTED)])
        + '</div><div class="stack">'
        + box("soft vs binary", f"<p>სხვაობა: XGBoost {gx['mean']:+.3f}, Logistic {gl['mean']:+.3f}</p>"
              "<p class=\"muted\">CI ორივეგან მოიცავს 0-ს → ფრე</p>", "bad")
        + box("baseline-ს ვერც ერთი ჯობნის", f"<p>საუკეთესო {ls['brier']:.3f} ≈ საშუალო {base['brier']:.3f}: სხვაობა ნულის ფარგლებშია</p>", "bad")
        + box("ერთი სარგებელი", f"<p>20% ყველაზე სარისკო კითხვის გამოტოვებით შეცდომა "
              f"<b>{pct(fail)} → {pct(xb['risk_at_coverage_80'])}</b></p>", "good")
        + '</div></div>'))

    # 7 task 2 asked + had
    s.append(page(
        "დავალება 2 · რას გვთხოვდნენ და რა გვქონდა",
        "H2, H3, H4: სამი შედარება",
        '<div class="three">'
        + box("H2", "<p>მხოლოდ კითხვის features<br><b>vs</b><br>+ მოდელის features</p>")
        + box("H3", "<p>+ მოდელის პარამეტრები<br>(temperature, context window)</p>")
        + box("H4", "<p>ML მოდელი<br><b>vs</b><br>მარტივი ცხრილი:<br>„მოდელი × საგანი“ შეცდომის %</p>")
        + '</div>'
        '<div class="two">'
        + box("Test split-ები (plan, Phase 5)", ul(["ახალი კითხვები", "ახალი საგანი", "ახალი მოდელი",
                                                   "ახალი კონფიგურაცია"]))
        + box("რა გვქონდა", ul(["იგივე 840 სტრიქონი", "H1-ის 4 მოდელი",
                                "გეგმა ჯერ დავწერეთ, მერე გავუშვით"]))
        + '</div>'))

    # 8 problems -> solutions
    s.append(page(
        "დავალება 2 · რა გავაკეთეთ",
        "ორი პრობლემა და მათი გადაწყვეტა",
        ps("<p><b>H3:</b> სამივე მოდელს <b>ერთნაირი</b> temperature და context window აქვს → "
           "შესადარებელი არაფერია.</p>",
           "<p><b>Reduced H3:</b> ვამოწმებთ, რაც იცვლება: მოდელის სახელი და knowledge cutoff. "
           "სრულ H3-ს გავუშვებთ, როცა მეორე temperature გვექნება.</p>")
        + ps("<p><b>„ახალი მოდელის“ ტესტი</b> საეჭვოდ კარგი იყო: Brier <b>0.130</b>. "
             "იგივე კითხვები სხვა მოდელებით train-ში რჩებოდა (leakage).</p>",
             "<p>test-ში ახლა <b>მოდელიც და მისი კითხვებიც</b> ახალია → Brier "
             f"<b>{mo['prompt only / logit binary']['brier']:.3f}</b> (რეალური შედეგი).</p>")))

    # 9 H2-H4 result
    rows2 = [("მხოლოდ კითხვის features", best_of("prompt only"), ACCENT),
             ("+ მოდელის სახელი", best_of("prompt + model id"), ACCENT),
             ("ყველა feature", best_of("all features"), ACCENT),
             ("საშუალო (baseline)", q["global mean"]["brier"], MUTED),
             ("ცხრილი: მოდელი × საგანი", q["model x subject"]["brier"], GOOD)]
    s.append(page(
        "დავალება 2 · რა შედეგი მივიღეთ",
        "მარტივი ცხრილი ჯობნის ML მოდელს",
        '<div class="split"><div>'
        + bar_chart(rows2, 0.150, 0.168, "Brier (რაც დაბალია, მით უკეთესი)",
                    ticks=(0.150, 0.155, 0.160, 0.165), bar_h=34, gap=14, w=560)
        + '</div><div class="stack">'
        + box("H2 ✓ ცოტათი", "<p>მოდელის features ოდნავ შველის</p>", "good")
        + box("H3 ◐ ნაწილობრივ", "<p>მხოლოდ reduced ვერსია</p>")
        + box("H4 ✗", "<p>ML ვერ ჯობნის ცხრილს „მოდელი × საგანი“ სამივე split-ზე</p>", "bad")
        + '</div></div>'))

    # 10 main problem -> next step
    null = pw["null_max_rho"]
    s.append(page(
        "დასკვნა · შემდეგი ნაბიჯი",
        "მთავარი პრობლემა: მონაცემები ცოტაა",
        ps(ul(["280 კითხვაზე საუკეთესო feature <b>ხმაურისგან არ განსხვავდება</b>",
               f"შემთხვევით არეულ მონაცემებზეც {pct(null['share_at_or_above_observed'], 0)} "
               "შემთხვევაში ჩნდება ასეთივე „ძლიერი“ feature"]),
           ul([f"<b>{n1000['n_questions']:,} კითხვა</b> × 3 მოდელი × 10 პასუხი",
               f"≈{n1000['calls']:,} API call, <b>≈${n1000['cost_usd_new_questions_only']:.0f}</b>",
               "20% test თავიდანვე „დალუქული“",
               "იგივე კოდი, იგივე მეტრიკები"]))
        + '<p class="answer">ახლა მუშაობს მხოლოდ მარტივი ცხრილი „მოდელი × საგანი“. '
        'მეტი მონაცემით გავიგებთ, კითხვის ტექსტი თუ ამატებს რამეს.</p>'))

    return s


KA_CSS = CSS.replace(
    'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif',
    'font-family:"Segoe UI","Noto Sans Georgian",Sylfaen,Arial,sans-serif'
) + f"""
.kicker{{text-transform:none;letter-spacing:.02em;font-size:15px}}
.slide{{padding:40px 60px}}
h1{{font-size:38px}} h2{{font-size:32px;margin:0 0 24px}}
h3{{font-size:20px;margin:0 0 8px}}
p,li{{font-size:20px;line-height:1.45}}
.box{{border:1px solid {FAINT};border-radius:12px;padding:18px 22px;background:#fff}}
.box p{{margin:4px 0}} .box ul{{margin:4px 0 0;padding-left:22px}} .box li{{margin:4px 0}}
.box.bad{{border-color:{BAD}66;background:#fff6f6}} .box.bad h3{{color:{BAD}}}
.box.good{{border-color:{GOOD}66;background:#f3fbf5}} .box.good h3{{color:{GOOD}}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:18px}}
.three{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;text-align:center}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.grid2 .box{{padding:26px 16px;text-align:center}} .grid2 h3{{margin:0;font-size:21px}}
.split{{display:grid;grid-template-columns:1.25fr 1fr;gap:28px;align-items:center}}
.stack{{display:flex;flex-direction:column;gap:12px}} .stack p{{font-size:18px}}
.ps{{display:grid;grid-template-columns:1fr 50px 1fr;gap:10px;align-items:stretch;margin-bottom:16px}}
.arrow{{display:flex;align-items:center;justify-content:center;font-size:40px;color:{MUTED}}}
.nums{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:8px}}
.num{{border:1px solid {FAINT};border-radius:12px;padding:18px;text-align:center}}
.num b{{display:block;font-size:48px;line-height:1.1}} .num span{{color:{MUTED};font-size:17px}}
.num.bad b{{color:{BAD}}}
.bignum{{font-size:44px;font-weight:700;margin:6px 0;color:{ACCENT}}}
.muted{{color:{MUTED};font-size:17px}}
.center{{text-align:center;margin:0}}
.dots{{display:flex;gap:10px;justify-content:center;margin:12px 0 4px}}
.dot{{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:22px;font-weight:700;color:#fff}}
.dot.x{{background:{BAD}}} .dot.ok{{background:{GOOD}}}
.answer{{margin-top:22px;padding:14px 20px;background:#f6f8fa;border-left:4px solid {ACCENT};border-radius:6px}}
.slide.title h1{{max-width:1000px}}
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
