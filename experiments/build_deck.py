"""Build the Phase 4-6 slide deck from the saved result files.

Reads h1_metrics.json, h2_h4_metrics.json, power_metrics.json and writes one
self-contained HTML file (no CDN, charts are inline SVG). Arrow keys or
click to move between slides.

    python experiments/build_deck.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
DECK = OUT / "phase4_6_deck.html"

INK = "#1f2328"
MUTED = "#656d76"
FAINT = "#d0d7de"
ACCENT = "#0969da"
WARN = "#bc4c00"
GOOD = "#1a7f37"
BAD = "#cf222e"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def esc(s) -> str:
    return html.escape(str(s))


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def ci(c, scale=1.0, d=3, unit="") -> str:
    return f"[{c['lo'] * scale:+.{d}f}{unit}, {c['hi'] * scale:+.{d}f}{unit}]"


def by_name(rows):
    return {r["model"]: r for r in rows}


# --------------------------------------------------------------------------- #
# SVG charts
# --------------------------------------------------------------------------- #

def line_chart(x, series, y_min, y_max, x_label, y_label, refs=(), w=860, h=380,
               y_fmt="{:.0f}", y_step=None, x_fmt="{:.0f}"):
    """series: [(name, color, values, dashed)]; refs: [(value, label, color)]."""
    left, right, top, bottom = 70, 180, 44, 60
    pw, ph = w - left - right, h - top - bottom
    x0, x1 = min(x), max(x)

    def sx(v):
        return left + (v - x0) / (x1 - x0) * pw

    def sy(v):
        return top + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    step = y_step or (y_max - y_min) / 4
    v = y_min
    while v <= y_max + 1e-9:
        y = sy(v)
        parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{FAINT}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" '
                     f'class="tick">{esc(y_fmt.format(v))}</text>')
        v += step
    for xv in x:
        parts.append(f'<text x="{sx(xv):.1f}" y="{top + ph + 22}" text-anchor="middle" '
                     f'class="tick">{esc(x_fmt.format(xv))}</text>')
    for value, label, color in refs:
        y = sy(value)
        parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{color}" stroke-width="1.5" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{left + pw + 10}" y="{y + 5:.1f}" class="lab" '
                     f'fill="{color}">{esc(label)}</text>')
    lx = left
    for name, color, values, dashed in series:
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, values))
        dash = ' stroke-dasharray="3 4"' if dashed else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="3"{dash}/>')
        parts.append(f'<line x1="{lx}" x2="{lx + 22}" y1="14" y2="14" stroke="{color}" '
                     f'stroke-width="3"{dash}/>')
        parts.append(f'<text x="{lx + 30}" y="19" class="lab">{esc(name)}</text>')
        lx += 44 + 9 * len(name)
    parts.append(f'<text x="{left + pw / 2}" y="{h - 12}" text-anchor="middle" '
                 f'class="axis">{esc(x_label)}</text>')
    parts.append(f'<text transform="translate(18 {top + ph / 2}) rotate(-90)" '
                 f'text-anchor="middle" class="axis">{esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(rows, x_min, x_max, x_label, refs=(), w=860, fmt="{:.3f}", ticks=(),
              bar_h=30, gap=12):
    """Horizontal bars. rows: [(label, value, color)]; refs: [(value, label, color)]."""
    left, right, top = 250, 90, 34
    h = top + len(rows) * (bar_h + gap) + 50
    pw = w - left - right

    def sx(v):
        return left + (v - x_min) / (x_max - x_min) * pw

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for i, (label, value, color) in enumerate(rows):
        y = top + i * (bar_h + gap)
        parts.append(f'<text x="{left - 12}" y="{y + bar_h * 0.68:.1f}" text-anchor="end" '
                     f'class="lab">{esc(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{max(sx(value) - left, 1):.1f}" '
                     f'height="{bar_h}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{sx(value) + 8:.1f}" y="{y + bar_h * 0.68:.1f}" '
                     f'class="val">{esc(fmt.format(value))}</text>')
    bottom = top + len(rows) * (bar_h + gap)
    for t in ticks:
        parts.append(f'<text x="{sx(t):.1f}" y="{bottom + 16}" text-anchor="middle" '
                     f'class="tick">{esc(fmt.format(t))}</text>')
    for value, label, color in refs:
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 10}" y2="{bottom}" '
                     f'stroke="{color}" stroke-width="1.5" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 16}" text-anchor="middle" class="tick" '
                     f'fill="{color}">{esc(label)}</text>')
    parts.append(f'<text x="{left + pw / 2}" y="{h - 10}" text-anchor="middle" '
                 f'class="axis">{esc(x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def reliability_svg(rows, color, w=330, h=250, x_label="Predicted risk",
                    y_label="Observed fail rate"):
    left, top, size = 56, 12, 190
    mx = 0.6

    def sx(v):
        return left + min(v, mx) / mx * size

    def sy(v):
        return top + (1 - min(v, mx) / mx) * size

    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in (0.0, 0.2, 0.4, 0.6):
        p.append(f'<line x1="{left}" x2="{left + size}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" '
                 f'stroke="{FAINT}"/>')
        p.append(f'<text x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end" class="tick">'
                 f'{t:.1f}</text>')
        p.append(f'<text x="{sx(t):.1f}" y="{top + size + 18}" text-anchor="middle" class="tick">'
                 f'{t:.1f}</text>')
    p.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(mx)}" y2="{sy(mx)}" stroke="{MUTED}" '
             f'stroke-dasharray="5 5"/>')
    pts = " ".join(f"{sx(r['pred']):.1f},{sy(r['obs']):.1f}" for r in rows)
    p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    for r in rows:
        p.append(f'<circle cx="{sx(r["pred"]):.1f}" cy="{sy(r["obs"]):.1f}" r="5" fill="{color}"/>')
    p.append(f'<text x="{left + size / 2}" y="{top + size + 40}" text-anchor="middle" '
             f'class="axis">{esc(x_label)}</text>')
    p.append(f'<text transform="translate(14 {top + size / 2}) rotate(-90)" text-anchor="middle" '
             f'class="axis">{esc(y_label)}</text>')
    p.append("</svg>")
    return "".join(p)


def hist_svg(counts, color, w=330, h=80, rest_label="above 0.6"):
    left, size = 56, 190
    top = 10
    peak = max(counts) or 1
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    shown = counts[:6]  # 0 to 0.6, matching the reliability axis
    bw = size / len(shown)
    for k, c in enumerate(shown):
        bh = c / peak * 50
        p.append(f'<rect x="{left + k * bw + 2:.1f}" y="{top + 50 - bh:.1f}" width="{bw - 4:.1f}" '
                 f'height="{bh:.1f}" fill="{color}" opacity="0.55"/>')
        p.append(f'<text x="{left + (k + 0.5) * bw:.1f}" y="{top + 66}" text-anchor="middle" '
                 f'class="tick">{c}</text>')
    rest = sum(counts[6:])
    p.append(f'<text x="{left + size + 8}" y="{top + 30}" class="tick">'
             f'{esc(rest_label) + ": " + str(rest) if rest else ""}</text>')
    p.append("</svg>")
    return "".join(p)


def table(headers, rows, align=None, highlight=None):
    align = align or ["left"] + ["right"] * (len(headers) - 1)
    highlight = highlight or {}
    th = "".join(f'<th style="text-align:{a}">{esc(h)}</th>' for h, a in zip(headers, align))
    body = []
    for i, r in enumerate(rows):
        cls = f' class="{highlight[i]}"' if i in highlight else ""
        tds = "".join(f'<td style="text-align:{a}">{esc(c)}</td>' for c, a in zip(r, align))
        body.append(f"<tr{cls}>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #

def slide(kicker, title, body, note=""):
    foot = f'<p class="note">{note}</p>' if note else ""
    return (f'<section class="slide"><p class="kicker">{esc(kicker)}</p>'
            f'<h2>{esc(title)}</h2><div class="body">{body}</div>{foot}</section>')


def big(value, label, tone=""):
    return (f'<div class="big {tone}"><div class="num">{esc(value)}</div>'
            f'<div class="cap">{esc(label)}</div></div>')


def build():
    h1 = load("h1_metrics.json")
    h24 = load("h2_h4_metrics.json")
    pw = load("power_metrics.json")

    r1 = by_name(h1["results"])
    base, binr, soft = r1["base rate"], r1["binary classifier"], r1["soft regressor"]
    lbin, lsoft = r1["binary logistic"], r1["soft logistic"]
    boot = h1["bootstrap"]
    qo = h24["splits"]["question-out"]
    so = h24["splits"]["subject-out"]
    mo = h24["splits"]["model-out"]
    q = by_name(qo["results"])
    fail = h1["fail_rate"]

    slides = []

    # 1 title
    slides.append(
        '<section class="slide title">'
        '<p class="kicker">MMLU-Pro miss prediction · Phases 4 to 6</p>'
        '<h1>Can prompt features predict when an LLM will be wrong?</h1>'
        '<p class="lede">Not yet. On 280 questions, no learned model beats the simple '
        '“this model on this subject” fail rate. The data is too small to tell a weak '
        'real signal from noise. About 1,000 questions would settle it, for roughly '
        f'${pw["sizes"][2]["cost_usd_new_questions_only"]:.0f} in API calls.</p>'
        '<div class="row">'
        + big("No", "H1: soft fail rate beats a binary label", "bad")
        + big("No", "H2 to H4: features beat model × subject", "bad")
        + big("~1,000", "questions needed (we have 280)", "accent")
        + '</div></section>')

    # 2 data and leakage
    slides.append(slide(
        "What we started from",
        "280 questions × 3 Gemini models × 10 answers each",
        '<div class="row">'
        + big("840", "cells (question × model)")
        + big("8,400", "graded answers")
        + big(pct(fail), "of answers were wrong")
        + '</div>'
        '<div class="two"><div><h3>Target</h3><p>For each cell: fail rate = wrong answers ÷ 10. '
        f'Only {pct(h1["share_cells_between"], 0)} of cells sit between 0 and 1; most are always '
        'right or always wrong.</p></div>'
        '<div><h3>What was wrong in the first notebooks</h3><ul>'
        '<li>The regression split rows, not questions. All 145 test questions were also in training.</li>'
        '<li><code>question_id</code> was left in as a feature.</li>'
        '<li>So its R² of 0.32 was optimistic: the model had seen every test question.</li>'
        '</ul></div></div>',
        "From here on, every test uses questions the model never saw. Question id and MMLU subject are "
        "never features; subject is only used as a baseline."))

    # 3 metric rule
    slides.append(slide(
        "Phase 6 · metrics locked before testing",
        "Which number decides depends on how the score is used",
        '<div class="two">'
        '<div class="card"><h3>Product skips the riskiest X%</h3>'
        '<p>Only the order matters. A score that says 90% on hard questions and 80% on easy ones works perfectly.</p>'
        '<p class="em">Decide on: coverage–risk curve, error at 90 / 80 / 70% coverage, AURC.</p></div>'
        '<div class="card"><h3>Product uses a threshold or shows the number</h3>'
        '<p>“Skip if risk &gt; 30%” only works if 30% really means 30%.</p>'
        '<p class="em">Decide on: Brier, Brier Skill Score, reliability diagram, ECE (15 quantile bins).</p></div>'
        '</div>'
        '<p class="small">Always also reported: log loss (clipped to [1e−4, 1−1e−4]), AUROC, AUPRC next to its base rate, '
        'a question-level bootstrap interval, and results by model and by subject. '
        'RMSE only next to its noise floor. Brier Skill Score = 1 − Brier ÷ Brier of always guessing the average.</p>'))

    # 4 H1 table
    def h1_row(label, r):
        return [label, f"{r['brier']:.3f}", f"{r['bss_vs_oof_base']:+.3f}",
                f"{r['ece_15_quantile']:.3f}", f"{r['log_loss']:.3f}",
                pct(r['risk_at_coverage_80']), f"{r['aurc']:.3f}", f"{r['auroc']:.2f}"]

    gx = boot["regressor_minus_classifier"]["brier"]
    gl = boot["soft_logistic_minus_binary_logistic"]["brier"]
    slides.append(slide(
        "Phase 4 · H1",
        "Predicting the exact fail rate does not beat a yes/no label",
        table(
            ["Predictor", "Brier", "Skill", "ECE", "Log loss", "Error, skip 20%", "AURC", "AUROC"],
            [
                ["Always guess the average", f"{base['brier']:.3f}", "0", f"{base['ece_15_quantile']:.3f}",
                 f"{base['log_loss']:.3f}", pct(fail), f"{base['aurc']:.3f}", f"{base['auroc']:.2f}"],
                h1_row("XGBoost · binary (fail rate > 0.5)", binr),
                h1_row("XGBoost · soft (exact fail rate)", soft),
                h1_row("Elastic-net logistic · binary", lbin),
                h1_row("Elastic-net logistic · soft (10 trials)", lsoft),
            ],
            highlight={4: "hl"}),
        f"Lower is better except AUROC; skill below 0 means worse than guessing the average. "
        f"Soft minus binary on Brier: XGBoost {gx['mean']:+.3f} {ci(gx)}, logistic {gl['mean']:+.3f} "
        f"{ci(gl)}. Both intervals include 0: a tie. The soft logistic ranks best (lowest AURC) but its "
        "skill interval also includes 0. Question-grouped 5-fold, same 144 features for every model."))

    # 5 H1 coverage
    grid = [100 * g for g in base["coverage_grid"]]
    slides.append(slide(
        "Phase 4 · H1 · coverage–risk",
        "Skipping risky questions helps; the logistic ranks best when skipping a lot",
        line_chart(
            grid,
            [("XGBoost binary", ACCENT, [100 * v for v in binr["selective_risk"]], False),
             ("XGBoost soft", WARN, [100 * v for v in soft["selective_risk"]], False),
             ("Logistic soft", GOOD, [100 * v for v in lsoft["selective_risk"]], True)],
            0, 25, "Share of questions still answered (%)", "Error among answered (%)",
            refs=[(100 * fail, f"Answer all {pct(fail)}", MUTED)], y_step=5),
        f"At 80% coverage the XGBoost binary score leaves {pct(binr['risk_at_coverage_80'])} errors, a drop of "
        f"{-100 * boot['delta_vs_oof_base']['binary classifier']['risk80_minus_base_rate']['mean']:.1f} points "
        f"(95% CI {ci(boot['delta_vs_oof_base']['binary classifier']['risk80_minus_base_rate'], 100, 1, ' pt')}). "
        f"The binary logistic is the same ({pct(lbin['risk_at_coverage_80'])}); both soft scores' drops include 0. "
        f"Skipping harder, the logistic ranks better: 15% error while still answering "
        f"{pct(lsoft['coverage_at_risk_0.15'], 0)} of questions (XGBoost binary: {pct(binr['coverage_at_risk_0.15'], 0)}), "
        f"and 10% error at {pct(lsoft['coverage_at_risk_0.10'], 0)} (XGBoost: {pct(binr['coverage_at_risk_0.10'], 0)})."))

    # 6 reliability
    rel_blocks = []
    for label, r, color in (("XGBoost binary", binr, ACCENT), ("Logistic soft", lsoft, GOOD)):
        rel_blocks.append(
            f'<div class="rel"><h3>{esc(label)} · ECE {r["ece_15_quantile"]:.3f}</h3>'
            + reliability_svg(r["reliability"], color)
            + '<p class="small tight">Cells per predicted-risk bin (0.1 wide)</p>'
            + hist_svg(r["pred_hist"], color) + '</div>')
    def share_below_04(r):
        return sum(r["pred_hist"][:4]) / sum(r["pred_hist"])

    slides.append(slide(
        "Phase 6 · calibration",
        "Predicted risk stays low and narrow; no model says “high risk”",
        '<div class="two">' + "".join(rel_blocks) + '</div>',
        "Reliability: 10 equal-count bins of the out-of-fold prediction; the dashed line is perfect "
        "calibration. The histogram under each chart counts cells per fixed 0.1-wide bin, so thin bins "
        f"stay visible. {pct(share_below_04(binr), 0)} (XGBoost binary) and {pct(share_below_04(lsoft), 0)} "
        f"(logistic soft) of predictions are below 0.4, although {pct(h1['share_cells_all_wrong'])} "
        "of cells fail all 10 answers. "
        "The models hedge toward the average, which is why their Brier is close to it."))

    # 6 H2 + reduced H3
    arm_rows = []
    head_style = (("xgb binary", "XGB binary", ACCENT), ("xgb soft", "XGB soft", WARN),
                  ("logit binary", "logit binary", INK), ("logit soft", "logit soft", GOOD))
    for arm in ("prompt only", "prompt + model id", "all features"):
        for head, label, color in head_style:
            arm_rows.append((f"{arm} · {label}", q[f"{arm} / {head}"]["brier"], color))
    h2x = qo["h2_prompt_minus_all"]["xgb binary"]["brier"]
    h2l = qo["h2_prompt_minus_all"]["logit soft"]["brier"]
    slides.append(slide(
        "Phase 4 · H2 and reduced H3",
        "Adding the model columns helps a bit; none reach the baseline",
        bar_chart(arm_rows, 0.150, 0.175,
                  "Brier on unseen questions (lower is better; axis starts at 0.150)",
                  ticks=(0.150, 0.155, 0.160, 0.165, 0.170, 0.175), bar_h=18, gap=7,
                  refs=[(q["global mean"]["brier"], "average", MUTED),
                        (q["model x subject"]["brier"], "model × subject", GOOD)]),
        f"Prompt-only minus all features: XGB binary {h2x['mean']:+.4f} {ci(h2x, d=4)}, "
        f"logit soft {h2l['mean']:+.4f} {ci(h2l, d=4)}. Both intervals exclude 0: real, but small. "
        "The planned H3 cannot run: context window, temperature, top-p and output pressure are the "
        "same for all three models. Stand-in: prompt, then + model id, then + knowledge cutoff and recency gap."))

    # 7 H4 across splits
    def split_row(label, e):
        r = by_name(e["results"])
        best = r[e["best_learned"]]
        g = e["best_vs_model_x_subject"]["brier"]
        return [label, f"{r['global mean']['brier']:.3f}", f"{r['model x subject']['brier']:.3f}",
                f"{best['brier']:.3f}", e["best_learned"], ci(g, d=4)]

    slides.append(slide(
        "Phase 4 and 5 · H4",
        "On every split, no learned model beats “model × subject”",
        table(["Split", "Average", "Model × subject", "Best learned", "Best learned arm",
               "Best minus model × subject, 95% CI"],
              [split_row("Unseen questions (5 folds)", qo),
               split_row("Unseen subject (14 folds)", so),
               split_row("Unseen model and questions (15 folds)", mo)],
              align=["left", "right", "right", "right", "left", "right"]),
        "Brier, lower is better. A positive gap means the learned model is worse; an interval that touches 0 means "
        "a tie, not a win. On unseen subjects, model × subject "
        "can only use the model rate; on an unseen model, it can only use the subject rate. Holding out the model "
        "alone looked great (Brier 0.130) because the other two models had answered the same questions: that was "
        "leakage, and it is not reported."))

    # subgroups
    bm = {}
    for row in h1["by_model"]:
        bm.setdefault(row["llm_model"], {})[row["predictor"]] = row
    model_rows = []
    for m in sorted(bm, key=lambda k: -bm[k]["base rate"]["fail_rate"]):
        v = bm[m]
        model_rows.append([m, pct(v["base rate"]["fail_rate"]), f"{v['base rate']['brier']:.3f}",
                           f"{v['binary classifier']['brier']:.3f}", f"{v['soft logistic']['brier']:.3f}",
                           pct(v["binary classifier"]["risk_at_coverage_80"])])
    bs = {}
    for row in h1["by_subject"]:
        bs.setdefault(row["subject"], {})[row["predictor"]] = row
    wins = {name: sum(1 for v in bs.values() if v[name]["brier"] < v["base rate"]["brier"])
            for name in ("binary classifier", "soft logistic")}
    hardest = max(bs, key=lambda k: bs[k]["base rate"]["fail_rate"])
    easiest = min(bs, key=lambda k: bs[k]["base rate"]["fail_rate"])
    slides.append(slide(
        "Phase 6 · results by subgroup",
        "The gap between models is bigger than anything the features add",
        table(["Model", "Fail rate", "Brier · average", "Brier · XGB binary", "Brier · logit soft",
               "Error, skip 20% (XGB binary)"], model_rows)
        + '<div class="row">'
        + big(f"{wins['binary classifier']} of {len(bs)}", "subjects where XGB binary beats the average")
        + big(f"{wins['soft logistic']} of {len(bs)}", "subjects where logit soft beats the average")
        + big(f"{pct(bs[hardest]['base rate']['fail_rate'], 0)} vs {pct(bs[easiest]['base rate']['fail_rate'], 0)}",
              f"fail rate, {hardest} vs {easiest}")
        + '</div>',
        "H1 out-of-fold scores (question-grouped 5-fold), sliced after scoring. The average is the training-fold "
        "fail rate over all models, so within one model it is off by that model's own gap; this is why "
        "model × subject is the baseline to beat."))

    # 8 power: threshold vs n
    sizes = pw["sizes"]
    xs = [s["n_questions"] for s in sizes]
    null = pw["null_max_rho"]
    slides.append(slide(
        "Step 2 · power analysis, no API spend",
        "With 280 questions, our best feature looks like noise",
        '<div class="split"><div>'
        + line_chart(
            xs, [("Needed to pass", INK, [s["rho_threshold"] for s in sizes], False)],
            0.0, 0.25, "Number of questions", "Correlation with fail rate (|Spearman ρ|)",
            refs=[(pw["observed_max_rho"], f"our best {pw['observed_max_rho']:.3f}", ACCENT),
                  (null["p95"], f"shuffled 95% {null['p95']:.3f}", BAD)],
            w=640, h=360, y_fmt="{:.2f}", y_step=0.05)
        + '</div><div class="stack">'
        + big(pct(null["share_at_or_above_observed"], 0), "of label shuffles give a feature at least this strong", "bad")
        + big(f"{pw['needed_for_power'][2]['n_questions']:,}", "questions to find a 0.165 signal 80% of the time", "accent")
        + '</div></div>',
        f"Tested {pw['n_features_tested']} numeric prompt features; the best is "
        f"<code>{esc(pw['observed_top10'][0]['feature'])}</code> (ρ {pw['observed_top10'][0]['rho']:+.3f}). "
        f"To pass the multiple-testing bar with 280 questions it would need {sizes[0]['rho_threshold']:.3f}. "
        f"Under bootstrap resampling, the top-10 feature list keeps only "
        f"{pct(pw['stability'][-1]['top10_overlap_mean'], 0)} of its members."))

    # 9 cost
    ca = pw["cost_assumptions"]
    cost_rows = []
    for s in sizes:
        cost_rows.append([f"{s['n_questions']:,}", f"{s['rho_threshold']:.3f}",
                          pct(s["power_at_observed_rho"], 0), f"{s['calls']:,}",
                          f"${s['cost_usd_new_questions_only']:.0f}"])
    slides.append(slide(
        "Step 2 · what more data costs",
        f"The fix is cheap: about ${sizes[2]['cost_usd_new_questions_only']:.0f} for "
        f"{sizes[2]['n_questions']:,} questions",
        table(["Questions", "|ρ| needed", "Chance to detect ρ = 0.165", "API calls", "Cost of new questions"],
              cost_rows, highlight={2: "hl"}),
        f"3 models × 10 answers per question; the sealed 20% gets {ca['sealed_trials']} answers. "
        f"~{ca['tokens_in']} input and ~{ca['tokens_out']} output tokens per call, ×{ca['buffer']} for thinking "
        "tokens and retries. Prices per 1M tokens: 2.5 Flash-Lite $0.10/$0.40, 3.1 Flash-Lite $0.25/$1.50, "
        "Flash latest $0.75/$3.75 (Google, September 2026; the “latest” alias can move)."))

    # 10 next
    slides.append(slide(
        "What next",
        "Grow the data, then rerun the same scripts",
        '<ol class="steps">'
        f'<li><b>Collect {sizes[2]["n_questions"]:,} questions</b> (about '
        f'{sizes[2]["n_questions"] // 14} per subject) × the same 3 models × 10 answers. '
        f'About {sizes[2]["calls"]:,} calls and roughly '
        f'${sizes[2]["cost_usd_new_questions_only"]:.0f}.</li>'
        '<li><b>Seal 20% of the questions first</b>, with 25 answers each. Touch them only once, at the end.</li>'
        '<li><b>Rerun unchanged:</b> <code>run_h1.py</code>, <code>run_h2_h4.py</code>, <code>run_power.py</code>, '
        '<code>build_deck.py</code>. Same metrics, same splits, same decision rule.</li>'
        '<li><b>For H3, add a second temperature</b> on part of the set. Without it, model limits cannot be tested.</li>'
        '</ol>',
        "Promotion rule: a feature set counts only if its 95% interval beats model × subject on the sealed set, "
        "on the metric that matches how the product uses the score."))

    return slides


CSS = f"""
*{{box-sizing:border-box}}
html,body{{margin:0;height:100%;background:#f6f8fa;color:{INK};
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.slide{{display:none;position:absolute;inset:0;margin:auto;width:min(1200px,96vw);height:min(720px,94vh);
  background:#fff;border:1px solid {FAINT};border-radius:12px;padding:44px 60px;overflow:auto}}
.slide.on{{display:flex;flex-direction:column}}
.kicker{{margin:0 0 6px;color:{MUTED};font-size:14px;letter-spacing:.06em;text-transform:uppercase}}
h1{{font-size:40px;line-height:1.15;margin:8px 0 18px;max-width:900px}}
h2{{font-size:30px;line-height:1.2;margin:0 0 22px}}
h3{{font-size:18px;margin:0 0 8px}}
p,li{{font-size:18px;line-height:1.5}}
.lede{{font-size:21px;color:{INK};max-width:900px}}
.body{{flex:1}}
.note{{color:{MUTED};font-size:14px;line-height:1.5;margin:14px 0 0;border-top:1px solid {FAINT};padding-top:12px}}
.small{{color:{MUTED};font-size:15px}}
.row{{display:flex;gap:18px;margin:18px 0}}
.big{{flex:1;border:1px solid {FAINT};border-radius:10px;padding:18px 20px}}
.big .num{{font-size:40px;font-weight:650;line-height:1.1}}
.big .cap{{color:{MUTED};font-size:15px;margin-top:6px}}
.big.bad .num{{color:{BAD}}} .big.accent .num{{color:{ACCENT}}} .big.good .num{{color:{GOOD}}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:10px}}
.card{{border:1px solid {FAINT};border-radius:10px;padding:20px 22px}}
.em{{font-weight:600}}
.split{{display:grid;grid-template-columns:1.7fr 1fr;gap:24px;align-items:center}}
.stack{{display:flex;flex-direction:column;gap:14px}}
table{{border-collapse:collapse;width:100%;font-size:17px}}
th{{color:{MUTED};font-weight:600;font-size:14px;border-bottom:2px solid {FAINT};padding:10px 12px}}
td{{border-bottom:1px solid {FAINT};padding:12px}}
tr.hl td{{background:#ddf4ff}}
code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;background:#f6f8fa;
  padding:1px 5px;border-radius:4px}}
.chart{{width:100%;height:auto}}
.chart .tick{{font-size:13px;fill:{MUTED}}} .chart .axis{{font-size:14px;fill:{MUTED}}}
.chart .lab{{font-size:14px;fill:{INK}}} .chart .val{{font-size:14px;fill:{INK};font-weight:600}}
.steps li{{margin-bottom:12px}}
.rel .chart{{max-width:380px}} .tight{{margin:2px 0}}
#bar{{position:fixed;bottom:10px;left:0;right:0;text-align:center;color:{MUTED};font-size:13px}}
@media print{{html,body{{background:#fff}}.slide{{display:flex!important;position:relative;
  page-break-after:always;border:none;height:auto;min-height:100vh}}#bar{{display:none}}}}
"""

JS = """
const s=[...document.querySelectorAll('.slide')];let i=0;const bar=document.getElementById('bar');
function show(n){i=Math.max(0,Math.min(s.length-1,n));s.forEach((e,k)=>e.classList.toggle('on',k===i));
bar.textContent=(i+1)+' / '+s.length+'  ·  arrow keys or click';location.hash=i+1;}
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key))show(i+1);
if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key==='Home')show(0);if(e.key==='End')show(s.length-1);});
document.addEventListener('click',e=>{if(e.target.closest('a'))return;show(e.clientX>innerWidth/3?i+1:i-1);});
show((parseInt(location.hash.slice(1))||1)-1);
"""


def main():
    slides = build()
    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           "<title>MMLU-Pro miss prediction · Phases 4 to 6</title>"
           f"<style>{CSS}</style></head><body>{''.join(slides)}"
           f"<div id=\"bar\"></div><script>{JS}</script></body></html>")
    DECK.write_text(doc, encoding="utf-8")
    print("wrote", DECK, f"({len(slides)} slides)")


if __name__ == "__main__":
    main()
