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

# Analysis cost grows with prompt length (a 20k-char paste takes about two
# seconds), so a public instance needs a bound to stay responsive.
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "20000"))
MAX_BODY = MAX_PROMPT_CHARS * 4 + 1024  # UTF-8 worst case, plus the JSON wrapper

SAMPLES = [
    "Who wrote The Hobbit?",
    "What about it?",
    "Instructions:\nUse the docs only.\nCompare ibuprofen and aspirin for fever in children after 2020. What dose is safe? What should be avoided?",
    "What are the latest treatments for type 2 diabetes?",
    "Can you tell me more about that thing we discussed?",
    'Find the paper titled "Attention Is All You Need" and summarize it in JSON.',
    "List three EU countries except France that joined after 2004",
    "Who directed the film that won the Oscar for best picture in 1994?",
    "Example 1: Paris -> France\nExample 2: Rome -> Italy\nExample 3: Tokyo ->",
    "\u00bfQui\u00e9n escribi\u00f3 El Hobbit y en qu\u00e9 a\u00f1o?",
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
    return {
        "feature_count": len(REGISTRY),
        "top30": TOP30_FEATURES,
        "features": [feature_declaration(f) for f in REGISTRY.values()],
        "groups": groups,
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
