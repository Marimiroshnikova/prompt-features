"""End-to-end smoke test against a running app.py.

Usage:  python tests/smoke_api.py [base_url]
Checks every route the web UI depends on and prints one line per check.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"

PROMPTS = [
    "Who wrote The Hobbit?",
    "What did he mean by that?",
    "\u00bfQui\u00e9n escribi\u00f3 El Hobbit y en qu\u00e9 a\u00f1o?",
    "Example 1: a\nExample 2: b\nExample 3: c\nWhat is next?",
    "",
]

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


def _body(response) -> dict | str:
    text = response.read().decode("utf-8")
    if "json" in response.headers.get("Content-Type", ""):
        return json.loads(text)
    return text


def get(path: str) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as response:
            return response.status, _body(response)
    except urllib.error.HTTPError as error:
        return error.code, _body(error)


def post(path: str, payload: dict) -> tuple[int, dict | str]:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, _body(response)
    except urllib.error.HTTPError as error:
        return error.code, _body(error)


print("static files")
for path, needle in (("/", "<title"), ("/styles.css", "body"), ("/app.js", "fetch")):
    status, body = get(path)
    check(f"GET {path}", status == 200 and needle in body)

print("schema")
status, schema = get("/api/schema")
check("GET /api/schema", status == 200)
check("schema lists features", len(schema.get("features", [])) > 100, f"{len(schema.get('features', []))} features")
check("schema lists 30 ranked", len(schema.get("top30", [])) == 30)
check("schema lists groups", len(schema.get("groups", [])) >= 8)
check("schema lists 3 plan groups", len(schema.get("plan_groups", [])) == 3)
check(
    "plan groups are prompt / model / interaction",
    [g.get("key") for g in schema.get("plan_groups", [])] == ["prompt", "model", "interaction"],
)
check("schema lists models", len(schema.get("models", [])) >= 1)
check("schema has model baseline", "overall_fail" in (schema.get("baseline") or {}))
first = schema["features"][0]
for field in ("name", "group", "dtype", "summary", "formula", "why"):
    check(f"schema field {field}", bool(first.get(field)))

print("health")
status, health = get("/api/health")
check("GET /api/health", status == 200)
check("health reports backends", bool(health.get("backends")))
for backend, info in sorted(health.get("backends", {}).items()):
    if info.get("available"):
        print(f"        {backend}: ready {info.get('version', '')}")
    else:
        print(f"        {backend}: MISSING - {info.get('install')}")
check(
    "every NLP backend is installed",
    all(info.get("available") for info in health.get("backends", {}).values()),
)

print("explain")
for prompt in PROMPTS:
    status, report = post("/api/explain", {"prompt": prompt})
    label = repr(prompt[:40])
    if prompt.strip() == "":
        check(f"empty prompt rejected {label}", status == 400 and "error" in report)
        continue
    check(f"POST /api/explain {label}", status == 200)
    check(
        f"  full feature set {label}",
        len(report.get("features", [])) == len(schema["features"]),
        f"{len(report.get('features', []))}",
    )
    check(f"  30 ranked returned {label}", len(report.get("top", [])) == 30)
    check(f"  summary present {label}", "headline" in report.get("summary", {}))
    traced = [f for f in report["features"] if f["steps"]]
    check(f"  traces present {label}", len(traced) > 100, f"{len(traced)} with steps")
    unexplained = [f["name"] for f in report["features"] if f["status"] != "ok" and not f["reason"]]
    check(f"  every gap explains itself {label}", not unexplained, str(unexplained[:3]))

print("report")
status, markdown = post("/api/report", {"prompt": "Who wrote The Hobbit?"})
check("POST /api/report", status == 200)
check("report returns markdown text", isinstance(markdown, str) and markdown.startswith("# Prompt"))
check("report contains a trace", "Calculation for this prompt" in markdown)

print("bad input")
status, payload = post("/api/explain", {"nope": 1})
check("missing prompt key rejected", status == 400, str(payload)[:70])
status, payload = post("/api/explain", {"prompt": "x" * 3_000_000})
check("oversized prompt rejected", status == 400, str(payload)[:70])
status, body = get("/api/nope")
check("unknown GET route is 404", status == 404)
status, payload = post("/api/nope", {"prompt": "hi"})
check("unknown POST route is 404", status == 404)

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for item in failures:
        print(f"  - {item}")
    sys.exit(1)
print("all checks passed")
