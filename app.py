"""Web app for exploring prompt features.

    python app.py            # then open http://127.0.0.1:8765
    python app.py --port 9000 --no-browser

Runs on the Python standard library only. Endpoints:

    GET  /                -> the single-page UI in web/
    GET  /api/schema      -> every feature declaration, no prompt needed
    GET  /api/health      -> which NLP backends loaded
    POST /api/explain     -> {"prompt": "..."} -> values plus calculation traces
    POST /api/report      -> {"prompt": "..."} -> the Markdown report as text

Environment variables, for hosted deployments:

    PORT               port to bind (default 8765; most hosts set this)
    HOST               interface to bind (default 127.0.0.1; use 0.0.0.0 hosted)
    MAX_PROMPT_CHARS   longest prompt accepted (default 20000)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import explain as explain_cli
from promptfeat import REGISTRY, TOP30_FEATURES, explain_prompt, feature_declaration, nlp
from promptfeat.registry import GROUP_BLURBS, GROUP_TITLES, STATUS_LABELS, features_by_group

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"


def _load_model_specs() -> dict:
    try:
        from experiments.model_specs import SPECS

        return SPECS
    except ImportError:
        pass
    path = ROOT / "experiments" / "model_specs.py"
    if not path.is_file():
        return {}
    import importlib.util

    spec = importlib.util.spec_from_file_location("model_specs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "SPECS", {})


MODEL_SPECS = _load_model_specs()


def _load_json(name: str) -> dict:
    path = WEB_DIR / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# Analysis cost grows with prompt length (a 20k-char paste takes about two
# seconds), so a public instance needs a bound to stay responsive.
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "20000"))
MAX_BODY = MAX_PROMPT_CHARS * 4 + 1024  # UTF-8 worst case, plus the JSON wrapper

SAMPLES = [
    "Which of the following is NOT a noble gas?\n\nA. Helium\nB. Neon\nC. Nitrogen\nD. Argon",
    "What is market socialism?\n\nA. State ownership with some market prices\nB. A sales tax\nC. A trade union\nD. A central bank",
    "Which of the following best describes the role of the Federal Reserve?\n\nA. Set the federal budget\nB. Conduct monetary policy\nC. Collect tariffs\nD. Run elections",
    "If a deposit of $500 is made at 8% compounded monthly, what is the amount after five years?\n\nA. $670\nB. $745\nC. $910\nD. $1,020",
    "A defendant was arrested in 2024 and charged with attempted murder. Who must prove insanity?\n\nA. The prosecution\nB. The defense\nC. The judge\nD. The jury",
]


# The written plan's three groups. Prompt is everything this page extracts.
# Model and interaction need a model id; they are not in the prompt string.
PLAN_GROUPS = [
    {
        "key": "prompt",
        "title": "Prompt",
        "blurb": "Measured from the prompt text alone. The 12 headings inside this group are the same features as All features.",
    },
    {
        "key": "model",
        "title": "Model / configuration",
        "blurb": "Published specs for the selected model. Not in the prompt. Temperature is missing on this list.",
        "fields": [
            {
                "name": "model_family",
                "summary": "Id prefix (gemini-2.5, gemma-4, gemini-latest, ...).",
                "formula": "prefix of the model id",
                "why": "Families differ in fail rate more than any prompt wording we measured.",
            },
            {
                "name": "is_preview",
                "summary": "Whether the model id contains preview.",
                "formula": "'preview' in model id",
                "why": "Preview snapshots can behave unlike the stable id with a similar name.",
            },
            {
                "name": "is_open_source",
                "summary": "Gemma yes, Gemini no.",
                "formula": "true for Gemma ids",
                "why": "Open weights and closed APIs are different systems, even on the same questions.",
            },
            {
                "name": "max_tokens_requested",
                "summary": "Output cap used for this eval (1024).",
                "formula": "1024 from the GAIA inference config",
                "why": "A short cap can truncate the letter. On this list every model has the same cap, so the column does not vary.",
            },
            {
                "name": "context_window_tokens",
                "summary": "Published input limit.",
                "formula": "model card input window",
                "why": "Needed to compute context pressure: prompt tokens divided by this window.",
            },
            {
                "name": "knowledge_cutoff_year",
                "summary": "Published cutoff year, or missing.",
                "formula": "model card cutoff, else null",
                "why": "Needed to compute recency gap against years named in the prompt.",
            },
            {
                "name": "temperature",
                "summary": "Sampling temperature.",
                "formula": "API default; this eval did not store it",
                "why": "Temperature changes how often the same question is missed. It is unknown here.",
            },
        ],
    },
    {
        "key": "interaction",
        "title": "Interaction",
        "blurb": "Needs both the prompt and a model. Context pressure is prompt tokens over the window. Recency gap is a year in the prompt minus the model's cutoff.",
        "fields": [
            {
                "name": "context_pressure",
                "summary": "How much of the model's window the prompt uses.",
                "formula": "context_token_count / context_window_tokens",
                "why": "A prompt that fills the window is a different risk than the same prompt on a 1M window.",
            },
            {
                "name": "recency_gap",
                "summary": "Years between a date in the prompt and the model's cutoff.",
                "formula": "year_max - knowledge_cutoff_year",
                "why": "A 2026 fact on a 2025 cutoff cannot be in the weights.",
            },
            {
                "name": "output_pressure",
                "summary": "Requested output tokens over the cap.",
                "formula": "max_tokens_requested / max_tokens_requested",
                "why": "Would catch a too-short cap. Every model here requested 1024 of 1024, so this is always 1.",
            },
        ],
    },
]


def schema_payload() -> dict:
    groups = []
    for key, members in features_by_group():
        groups.append(
            {
                "key": key,
                "title": GROUP_TITLES[key],
                "blurb": GROUP_BLURBS[key],
                "features": [f.name for f in members],
            }
        )
    models = {
        model_id: {
            "model_id": model_id,
            **{k: v for k, v in spec.items() if k != "spec_source"},
            "spec_source": spec.get("spec_source"),
        }
        for model_id, spec in MODEL_SPECS.items()
    }
    return {
        "feature_count": len(REGISTRY),
        "top30": [item["name"] for item in _load_json("top30.json").get("features", [])]
        or TOP30_FEATURES,
        "top30_detail": _load_json("top30.json"),
        "features": [feature_declaration(f) for f in REGISTRY.values()],
        "groups": groups,
        "plan_groups": PLAN_GROUPS,
        "models": models,
        "baseline": _load_json("baseline.json"),
        "status_labels": STATUS_LABELS,
        "samples": SAMPLES,
        "backends": nlp.backend_report(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "promptfeat/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in str(args[0]):
            super().log_message(fmt, *args)

    # --- helpers ---------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(
            code,
            json.dumps(payload, default=str).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _discard_body(self) -> None:
        """Read and throw away an oversized body.

        Replying before the client has finished sending aborts the connection on
        Windows, so the caller never sees the 400 explaining what went wrong.
        """
        remaining = int(self.headers.get("Content-Length") or 0)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        raw = self.rfile.read(length)
        charset = self.headers.get_content_charset() or "utf-8"
        # Some CLI clients (PowerShell, curl on Windows) send single-byte
        # encodings without saying so, which would break non-English prompts.
        for encoding in (charset, "utf-8", "cp1252", "latin-1"):
            try:
                return json.loads(raw.decode(encoding))
            except (UnicodeDecodeError, LookupError):
                continue
            except ValueError:
                return {}
        return {}

    # --- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/schema":
            self._json(schema_payload())
            return
        if path == "/api/health":
            self._json({"backends": nlp.backend_report(), "features": len(REGISTRY)})
            return
        if path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html")
            return
        target = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR.resolve() in target.parents and target.is_file():
            self._serve_file(target)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        too_long = f"prompt too long; the limit is {MAX_PROMPT_CHARS} characters"
        if int(self.headers.get("Content-Length") or 0) > MAX_BODY:
            self._discard_body()
            self._json({"error": too_long}, 400)
            return
        payload = self._read_json()
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            self._json({"error": "expected a JSON body with a 'prompt' string"}, 400)
            return
        if not prompt.strip():
            self._json({"error": "the prompt is empty, so there is nothing to measure"}, 400)
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            self._json({"error": too_long}, 400)
            return
        if path == "/api/explain":
            self._json(explain_prompt(prompt))
            return
        if path == "/api/report":
            text = explain_cli.render(explain_prompt(prompt))
            self._send(200, text.encode("utf-8"), "text/markdown; charset=utf-8")
            return
        self._json({"error": f"unknown endpoint {path}"}, 404)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if guessed.startswith("text/") or guessed in ("application/javascript",):
            guessed += "; charset=utf-8"
        self._send(200, path.read_bytes(), guessed)


def main() -> None:
    # Hosts capture stdout through a pipe, which Python buffers by default, so
    # the startup lines would not reach the log until the buffer filled.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    # A hosted process has no display to open a browser on.
    hosted = "PORT" in os.environ or args.host == "0.0.0.0"

    report = nlp.backend_report()
    missing = [name for name, info in report.items() if not info["available"]]
    print(f"promptfeat: {len(REGISTRY)} features ready")
    for name, info in report.items():
        mark = "ok" if info["available"] else "MISSING"
        version = f" {info['version']}" if info["version"] else ""
        print(f"  {name:<11} {mark}{version}")
    if missing:
        print(
            "Features needing a missing backend will report status 'unavailable' "
            "with an install hint rather than failing."
        )

    if hosted:
        # Loading spaCy takes a few seconds. Do it now so the first visitor
        # does not pay for it.
        start = time.perf_counter()
        explain_prompt("Who wrote The Hobbit?")
        print(f"warmed up in {time.perf_counter() - start:.1f}s")

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nServing {url}  (Ctrl+C to stop)")
    print(f"Accepting prompts up to {MAX_PROMPT_CHARS} characters")
    if not args.no_browser and not hosted:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
