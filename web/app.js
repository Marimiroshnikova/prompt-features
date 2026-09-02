"use strict";

const state = {
  schema: null,
  report: null,
  view: "top",
  filter: "",
  onlyActive: false,
  model: "",
  category: "",
  open: new Set(),
};

const el = (id) => document.getElementById(id);

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* ---------- boot ---------- */

async function boot() {
  const res = await fetch("/api/schema");
  state.schema = await res.json();
  el("feature-count").textContent = `${state.schema.feature_count} features`;
  renderBackends();
  renderSamples();
  renderModels();
  renderCategories();
  wire();
  const params = new URLSearchParams(location.search);
  const shared = params.get("prompt");
  // A link can name a feature to open, so a calculation can be shared as it is seen.
  const open = params.get("open");
  if (open) open.split(",").forEach((name) => state.open.add(name.trim()));
  const view = params.get("view");
  if (view && document.querySelector(`.tab[data-view="${view}"]`)) {
    state.view = view;
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.view === view);
    });
  }
  const model = params.get("model") || sessionStorage.getItem("model") || "";
  if (model && state.schema.models[model]) {
    state.model = model;
    el("model").value = model;
  }
  const cats = (state.schema.baseline && state.schema.baseline.by_category) || {};
  const category = params.get("category") || sessionStorage.getItem("category") || "";
  if (category && cats[category]) {
    state.category = category;
    el("category").value = category;
  }
  const initial = shared !== null ? shared : sessionStorage.getItem("prompt");
  if (initial) {
    el("prompt").value = initial;
    run();
  }
}

function renderBackends() {
  el("backends").innerHTML = Object.entries(state.schema.backends)
    .map(([name, info]) => {
      const cls = info.available ? "on" : "off";
      const title = info.available
        ? `${name} ${info.version}`
        : `${name} missing. ${info.install}`;
      return `<span class="backend ${cls}" title="${escapeHtml(title)}">${escapeHtml(name)}</span>`;
    })
    .join("");
}

function renderSamples() {
  el("samples").innerHTML = state.schema.samples
    .map((text, i) => `<button class="chip" data-i="${i}">${escapeHtml(text.replace(/\n/g, " \u23ce "))}</button>`)
    .join("");
  el("samples").querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      el("prompt").value = state.schema.samples[Number(chip.dataset.i)];
      run();
    });
  });
}

function renderModels() {
  const select = el("model");
  const ids = Object.keys(state.schema.models || {});
  ids.forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    select.appendChild(opt);
  });
}

function renderCategories() {
  const select = el("category");
  const cats = Object.keys((state.schema.baseline && state.schema.baseline.by_category) || {});
  cats.sort().forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });
}

function wire() {
  el("run").addEventListener("click", run);
  el("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      run();
    }
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.view = tab.dataset.view;
      renderList();
    });
  });
  el("filter").addEventListener("input", (e) => {
    state.filter = e.target.value.toLowerCase().trim();
    renderList();
  });
  el("only-active").addEventListener("change", (e) => {
    state.onlyActive = e.target.checked;
    renderList();
  });
  el("model").addEventListener("change", (e) => {
    state.model = e.target.value;
    sessionStorage.setItem("model", state.model);
    renderList();
  });
  el("category").addEventListener("change", (e) => {
    state.category = e.target.value;
    sessionStorage.setItem("category", state.category);
    renderList();
  });
  el("expand-all").addEventListener("click", () => {
    const rows = visibleFeatures();
    const allOpen = rows.every((f) => state.open.has(f.name));
    rows.forEach((f) => (allOpen ? state.open.delete(f.name) : state.open.add(f.name)));
    renderList();
  });
  el("copy-json").addEventListener("click", needsReport(copyJson));
  el("download-csv").addEventListener("click", needsReport(downloadCsv));
  el("download-report").addEventListener("click", needsReport(downloadReport));
}

function needsReport(fn) {
  return () => {
    if (!state.report) {
      showError("Extract features first, then the values can be exported.");
      return;
    }
    fn();
  };
}

/* ---------- run ---------- */

function showError(message) {
  const box = el("error");
  box.textContent = message;
  box.hidden = !message;
}

async function postJson(path, prompt) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const payload = await res.json();
      if (payload.error) detail = payload.error;
    } catch (ignored) {
      /* the body was not JSON, so the status line is all we have */
    }
    throw new Error(detail);
  }
  return res;
}

async function run() {
  const prompt = el("prompt").value;
  if (!prompt.trim()) {
    showError("Type a prompt first: every feature is measured from the text itself.");
    el("prompt").focus();
    return;
  }
  sessionStorage.setItem("prompt", prompt);
  const button = el("run");
  button.disabled = true;
  button.textContent = "Extracting\u2026";
  showError("");
  try {
    const res = await postJson("/api/explain", prompt);
    state.report = await res.json();
    el("empty").hidden = true;
    el("results").hidden = false;
    renderSummary();
    renderList();
  } catch (err) {
    showError(`Extraction failed: ${err.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Extract features";
  }
}

/* ---------- summary ---------- */

function renderSummary() {
  const s = state.report.summary;
  const pct = Math.round((Number(s.headline) || 0) * 100);
  const statuses = Object.entries(s.statuses || {})
    .filter(([k]) => k !== "ok")
    .map(([k, v]) => `${v} ${state.schema.status_labels[k] || k}`)
    .join(", ");
  const core = state.report.core_question;
  const diluted = core && core !== state.report.prompt.trim();

  el("summary").innerHTML = `
    <div class="card headline band-${s.band || "low"}">
      <div class="k">Retrieval difficulty</div>
      <div class="v">${escapeHtml(s.headline ?? "-")}</div>
      <div class="sub">${escapeHtml(s.band || "")} risk</div>
      <div class="gauge"><span style="width:${pct}%"></span></div>
    </div>
    <div class="card">
      <div class="k">Category</div>
      <div class="v" style="font-size:17px">${escapeHtml(s.category ?? "-")}</div>
      <div class="sub">question type: ${escapeHtml(s.question_type ?? "-")}</div>
    </div>
    <div class="card">
      <div class="k">Size</div>
      <div class="v" style="font-size:17px">${escapeHtml(s.words ?? "-")} words</div>
      <div class="sub">${escapeHtml(s.tokens ?? "-")} BPE tokens</div>
    </div>
    <div class="card">
      <div class="k">Computed</div>
      <div class="v" style="font-size:17px">${s.computed} / ${s.feature_count}</div>
      <div class="sub">${escapeHtml(statuses || "all values ok")}</div>
    </div>
    ${diluted ? `
    <div class="card" style="grid-column:1/-1">
      <div class="k">Core question after removing instructions, examples and pasted context</div>
      <div class="sub" style="font-family:var(--mono);color:#cbd5e2;margin-top:2px">${escapeHtml(core)}</div>
    </div>` : ""}
  `;
}

/* ---------- list ---------- */

function isActive(f) {
  const v = f.value;
  if (v === null || v === undefined) return false;
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") return !["other", "none", "-", ""].includes(v.toLowerCase());
  return true;
}

function matchesFilter(f) {
  if (!state.filter) return true;
  return (
    f.name.toLowerCase().includes(state.filter) ||
    f.summary.toLowerCase().includes(state.filter) ||
    (f.group_title || "").toLowerCase().includes(state.filter) ||
    (f.group || "").toLowerCase().includes(state.filter)
  );
}

function visibleFeatures() {
  let list;
  if (state.view === "top") list = state.report.top;
  else if (state.view === "issues") list = state.report.features.filter((f) => f.status !== "ok");
  else if (state.view === "plan") list = planVisibleFeatures();
  else list = state.report.features;
  return list.filter((f) => matchesFilter(f) && (!state.onlyActive || isActive(f) || f.status !== "ok"));
}

function featureValue(name) {
  const f = (state.report.features || []).find((row) => row.name === name);
  return f ? f.value : null;
}

function asNumber(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function displayOf(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(3);
    return String(value);
  }
  return String(value);
}

function specRow(field, value, extra) {
  const status = extra.status || (value === null || value === undefined ? "not_applicable" : "ok");
  const reason = extra.reason || "";
  return {
    name: field.name,
    summary: field.summary,
    formula: field.formula,
    why: field.why,
    dtype: extra.dtype || "label",
    backend: extra.backend || "model spec",
    value,
    display_value: extra.display_value || displayOf(value),
    status,
    reason,
    status_rules: extra.status_rules || [],
    steps: extra.steps || [],
    spans: [],
    lexicon_hits: [],
    needs: extra.needs || [],
    group_title: extra.group_title,
    group: extra.group,
    tier: 2,
    rank: null,
    rank_reason: "",
    value_range: extra.value_range || "",
  };
}

function modelRows() {
  const group = (state.schema.plan_groups || []).find((g) => g.key === "model");
  if (!group) return [];
  const spec = state.schema.models[state.model];
  if (!spec) return [];
  return group.fields.map((field) => {
    const value = spec[field.name];
    const missing = value === null || value === undefined;
    return specRow(field, missing ? null : value, {
      group: "model",
      group_title: "Model / configuration",
      status: missing ? "unavailable" : "ok",
      reason: missing
        ? "this eval did not store a value (API default, or unpublished on the model card)"
        : "",
      steps: missing ? [] : [`${field.name} = ${displayOf(value)} for ${state.model}`],
    });
  });
}

function interactionRows() {
  const group = (state.schema.plan_groups || []).find((g) => g.key === "interaction");
  if (!group || !state.model) return [];
  const spec = state.schema.models[state.model];
  if (!spec) return [];
  const byName = Object.fromEntries(group.fields.map((f) => [f.name, f]));
  const tokens = asNumber(featureValue("context_token_count"));
  const window = asNumber(spec.context_window_tokens);
  const year = asNumber(featureValue("year_max"));
  const cutoff = asNumber(spec.knowledge_cutoff_year);
  const cap = asNumber(spec.max_tokens_requested);

  const pressureOk = Number.isFinite(tokens) && Number.isFinite(window) && window > 0;
  const pressure = pressureOk ? tokens / window : null;
  const recencyOk = Number.isFinite(year) && Number.isFinite(cutoff);
  const recency = recencyOk ? year - cutoff : null;
  const outOk = Number.isFinite(cap) && cap > 0;
  const output = outOk ? 1 : null;

  return [
    specRow(byName.context_pressure, pressure, {
      group: "interaction",
      group_title: "Interaction",
      dtype: "float",
      backend: "derived",
      status: pressureOk ? "ok" : "not_applicable",
      reason: pressureOk ? "" : "needs context_token_count and a published context window",
      steps: pressureOk
        ? [`context_token_count = ${tokens}`, `context_window_tokens = ${window}`, `pressure = ${tokens} / ${window} = ${displayOf(pressure)}`]
        : [],
      needs: ["context_token_count", "context_window_tokens"],
    }),
    specRow(byName.recency_gap, recency, {
      group: "interaction",
      group_title: "Interaction",
      dtype: "float",
      backend: "derived",
      status: recencyOk ? "ok" : "not_applicable",
      reason: recencyOk
        ? ""
        : "needs a year in the prompt (year_max) and a published knowledge cutoff",
      steps: recencyOk
        ? [`year_max = ${year}`, `knowledge_cutoff_year = ${cutoff}`, `gap = ${year} - ${cutoff} = ${recency}`]
        : [],
      needs: ["year_max", "knowledge_cutoff_year"],
    }),
    specRow(byName.output_pressure, output, {
      group: "interaction",
      group_title: "Interaction",
      dtype: "float",
      backend: "derived",
      status: outOk ? "unreliable" : "not_applicable",
      reason: outOk
        ? "every model on this list requested 1024 tokens, so this is always 1"
        : "no output cap on the spec",
      steps: outOk ? [`max_tokens_requested = ${cap}`, `output_pressure = ${cap} / ${cap} = 1`] : [],
      needs: ["max_tokens_requested"],
    }),
  ];
}

function planVisibleFeatures() {
  const prompt = state.report.features || [];
  return [...prompt, ...modelRows(), ...interactionRows()];
}

function pct(rate) {
  if (rate === null || rate === undefined || Number.isNaN(Number(rate))) return "—";
  return `${(Number(rate) * 100).toFixed(1)}%`;
}

function baselineCardHtml() {
  const b = state.schema.baseline;
  if (!b || b.overall_fail == null) return "";
  const modelRate = state.model ? (b.by_model || {})[state.model] : null;
  let cellRate = null;
  if (state.model && state.category) {
    cellRate = ((b.by_model_category || {})[state.model] || {})[state.category];
  }
  const catRate = state.category ? (b.by_category || {})[state.category] : null;
  const modelNote = state.model
    ? "Phase 3 model baseline: this model's miss rate on the 280-question grid."
    : "Pick a model to see the model baseline that beat the prompt features.";
  const cellNote = state.model && state.category
    ? "The winner: this model on this subject."
    : "Pick a subject to see model × subject.";
  return `<section class="group plan-group">
    <h2>Baseline</h2>
    <p class="blurb">Not a prompt feature. Miss rate from one answer × 280 questions × 14 models. Prompt text lost to these lookups.</p>
    <div class="summary" style="margin-bottom:12px">
      <div class="card">
        <div class="k">Overall</div>
        <div class="v">${pct(b.overall_fail)}</div>
        <div class="sub">guessing, any model</div>
      </div>
      <div class="card">
        <div class="k">This model</div>
        <div class="v">${state.model ? pct(modelRate) : "—"}</div>
        <div class="sub">${escapeHtml(modelNote)}</div>
      </div>
      <div class="card">
        <div class="k">This subject</div>
        <div class="v">${state.category ? pct(catRate) : "—"}</div>
        <div class="sub">${state.category ? escapeHtml(state.category) : "pick a subject"}</div>
      </div>
      <div class="card headline">
        <div class="k">Model × subject</div>
        <div class="v">${cellRate == null ? "—" : pct(cellRate)}</div>
        <div class="sub">${escapeHtml(cellNote)}</div>
      </div>
    </div>
  </section>`;
}

function renderPlanGroups() {
  const wanted = new Set(visibleFeatures().map((f) => f.name));
  const plans = state.schema.plan_groups || [];
  return plans
    .map((plan) => {
      if (plan.key === "prompt") {
        const inner = (state.report.groups || [])
          .map((group) => {
            const members = group.features.filter((f) => wanted.has(f.name));
            if (!members.length) return "";
            return `<h3>${escapeHtml(group.title)} <span style="color:var(--faint)">(${members.length})</span></h3>
              <p class="blurb">${escapeHtml(group.blurb)}</p>
              <div class="rows">${members.map(rowHtml).join("")}</div>`;
          })
          .join("");
        if (!inner) return "";
        return `<section class="group plan-group">
            <h2>1. ${escapeHtml(plan.title)}</h2>
            <p class="blurb">${escapeHtml(plan.blurb)}</p>
            ${inner}
          </section>`;
      }

      const rows = plan.key === "model" ? modelRows() : interactionRows();
      const shown = rows.filter((f) => wanted.has(f.name));
      const needModel = !state.model;
      const empty = needModel
        ? `<div class="rows"><div class="nothing">Pick a model on the left to fill ${escapeHtml(plan.title)}.</div></div>`
        : shown.length
          ? `<div class="rows">${shown.map(rowHtml).join("")}</div>`
          : `<div class="rows"><div class="nothing">No ${escapeHtml(plan.title).toLowerCase()} rows match this filter.</div></div>`;
      const n = plan.key === "model" ? "2" : "3";
      return `<section class="group plan-group">
          <h2>${n}. ${escapeHtml(plan.title)}${shown.length ? ` <span style="color:var(--faint)">(${shown.length})</span>` : ""}</h2>
          <p class="blurb">${escapeHtml(plan.blurb)}</p>
          ${empty}
        </section>`;
    })
    .join("");
}

function renderList() {
  if (!state.report) return;
  const features = visibleFeatures();
  const host = el("feature-list");

  if (state.view === "plan") {
    host.innerHTML = baselineCardHtml() + renderPlanGroups();
  } else if (!features.length) {
    host.innerHTML = `<div class="rows"><div class="nothing">No features match this filter.</div></div>`;
  } else if (state.view === "all") {
    const wanted = new Set(features.map((f) => f.name));
    host.innerHTML = state.report.groups
      .map((group) => {
        const members = group.features.filter((f) => wanted.has(f.name));
        if (!members.length) return "";
        return `<section class="group">
            <h2>${escapeHtml(group.title)} <span style="color:var(--faint)">(${members.length})</span></h2>
            <p class="blurb">${escapeHtml(group.blurb)}</p>
            <div class="rows">${members.map(rowHtml).join("")}</div>
          </section>`;
      })
      .join("");
  } else {
    const heading =
      state.view === "top"
        ? `<h2>The 30 most important features, in rank order</h2><p class="blurb">Ranked by expected power to predict retrieval failure. Click any row to see the calculation.</p>`
        : `<h2>Features that could not be computed normally (${features.length})</h2><p class="blurb">Each one says why, so a downstream model is never fed a fake zero.</p>`;
    host.innerHTML = `<section class="group">${heading}<div class="rows">${features.map(rowHtml).join("")}</div></section>`;
  }

  host.querySelectorAll(".head").forEach((head) => {
    head.addEventListener("click", () => {
      const name = head.parentElement.dataset.name;
      if (state.open.has(name)) state.open.delete(name);
      else state.open.add(name);
      renderList();
    });
  });
}

function valueClass(f) {
  if (f.value === null || f.value === undefined) return "none";
  if (typeof f.value === "boolean") return f.value ? "true" : "false";
  if (typeof f.value === "number" && f.value === 0) return "zero";
  return "";
}

function badgeClass(status) {
  return {
    ok: "ok",
    not_applicable: "na",
    undefined: "undefined",
    unreliable: "unreliable",
    unavailable: "unavailable",
  }[status] || "na";
}

function rowHtml(f) {
  const open = state.open.has(f.name);
  const rank = f.tier === 1 && f.rank ? `<span class="rank tier1">${f.rank}</span>` : `<span class="rank"></span>`;
  const label = state.schema.status_labels[f.status] || f.status;
  return `<div class="row ${open ? "open" : ""}" data-name="${escapeHtml(f.name)}">
      <div class="head">
        ${rank}
        <span class="name">${escapeHtml(f.name)}<span class="meaning">${escapeHtml(f.summary)}</span></span>
        <span class="value ${valueClass(f)}">${escapeHtml(f.display_value)}</span>
        <span class="badge ${badgeClass(f.status)}">${escapeHtml(label)}</span>
      </div>
      ${open ? detailHtml(f) : ""}
    </div>`;
}

function highlight(text, spans) {
  if (!spans || !spans.length) return "";
  const ranges = spans
    .filter((s) => s.end > s.start)
    .map((s) => [s.start, s.end])
    .sort((a, b) => a[0] - b[0]);
  const merged = [];
  ranges.forEach(([start, end]) => {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  });
  let out = "";
  let cursor = 0;
  merged.forEach(([start, end]) => {
    out += escapeHtml(text.slice(cursor, start));
    out += `<mark>${escapeHtml(text.slice(start, end))}</mark>`;
    cursor = end;
  });
  out += escapeHtml(text.slice(cursor));
  return out;
}

function detailHtml(f) {
  const parts = [];

  if (f.tier === 1 && f.rank_reason) {
    parts.push(`<div class="rank-note">Rank ${f.rank} of 30: ${escapeHtml(f.rank_reason)}</div>`);
  }

  if (f.status !== "ok") {
    const cls = badgeClass(f.status);
    parts.push(`<div class="status-note ${cls}"><strong>${escapeHtml(state.schema.status_labels[f.status] || f.status)}:</strong> ${escapeHtml(f.reason)}</div>`);
  }

  parts.push(`<h4>What we see</h4><p>${escapeHtml(f.summary)}</p>`);
  parts.push(`<h4>How it is calculated</h4><div class="formula">${escapeHtml(f.formula)}</div>`);

  if (f.steps && f.steps.length) {
    parts.push(`<h4>Calculation for this prompt</h4><ol class="steps">${f.steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol>`);
  }

  if (f.spans && f.spans.length) {
    parts.push(`<h4>Matched text in the prompt</h4><div class="matched">${highlight(state.report.normalized, f.spans)}</div>`);
  }

  if (f.status_rules && f.status_rules.length) {
    parts.push(`<h4>When it cannot be computed</h4><ol class="steps">${f.status_rules.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ol>`);
  }

  parts.push(`<h4>Why it matters for retrieval</h4><p class="why">${escapeHtml(f.why)}</p>`);

  const meta = [`type: ${f.dtype}`, `computed by: ${f.backend}`, `group: ${f.group_title}`];
  if (f.value_range) meta.push(`range: ${f.value_range}`);
  if (f.lexicon_hits && f.lexicon_hits.length) meta.push(`lexicons: ${f.lexicon_hits.join(", ")}`);
  if (f.needs && f.needs.length) meta.push(`built from: ${f.needs.join(", ")}`);
  parts.push(`<div class="meta">${meta.map((m) => `<span>${escapeHtml(m)}</span>`).join("")}</div>`);

  return `<div class="detail">${parts.join("")}</div>`;
}

/* ---------- exports ---------- */

function valuesObject() {
  const out = {};
  state.report.features.forEach((f) => {
    out[f.name] = f.value;
  });
  return out;
}

async function copyJson() {
  const text = JSON.stringify(valuesObject(), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    flash(el("copy-json"), "Copied");
  } catch {
    flash(el("copy-json"), "Copy failed");
  }
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadCsv() {
  const features = state.report.features;
  const header = ["prompt", ...features.map((f) => f.name), ...features.map((f) => `${f.name}__status`)];
  const row = [
    state.report.prompt,
    ...features.map((f) => f.value),
    ...features.map((f) => f.status),
  ];
  const csv = `${header.map(csvCell).join(",")}\n${row.map(csvCell).join(",")}\n`;
  save(csv, "text/csv", "prompt-features.csv");
}

async function downloadReport() {
  try {
    const res = await postJson("/api/report", state.report.prompt);
    save(await res.text(), "text/markdown", "prompt-features-report.md");
  } catch (err) {
    showError(`Report failed: ${err.message}`);
  }
}

function save(text, type, filename) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function flash(button, message) {
  const original = button.textContent;
  button.textContent = message;
  setTimeout(() => (button.textContent = original), 1200);
}

boot();
