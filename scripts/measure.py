#!/usr/bin/env python3
"""NPU usability measurement CLI.

Subcommands:
  extract  Scan benchmark operator trees and emit source metrics JSON
  score    Score x.y.z items from extract + hardware spec + benchmark perf
  serve    Serve report.html and /api/extract for the in-page 度量 button
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from extract_source import extract_all, load_operators  # noqa: E402
from score import merge_into_measure_data, score_all  # noqa: E402


def _read_json(path: str, default=None):
    if not path:
        return default if default is not None else {}
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def cmd_extract(args):
    roots = {}
    if args.ascendc:
        roots["AscendC"] = args.ascendc
    if args.superscalar:
        roots["SuperScalar"] = args.superscalar
    if args.simt:
        roots["SIMT"] = args.simt
    if not roots:
        raise SystemExit("Need at least one of --ascendc / --superscalar / --simt")
    data = extract_all(roots, args.operators)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for arch, blob in data["arches"].items():
        found = sum(1 for o in blob["operators"].values() if not o["metrics"].get("missing"))
        print(f"  {arch}: {found}/{len(blob['operators'])} operators with source")


def cmd_score(args):
    extract = _read_json(args.extract)
    hw = _read_json(args.hw)
    perf = _read_json(args.perf)
    measure = _read_json(args.data) if args.data else {}
    patterns = measure.get("patterns") if measure else None
    computed = score_all(extract, hw, perf, patterns)
    Path(args.out).write_text(json.dumps(computed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.merge_data:
        merged = merge_into_measure_data(measure, computed)
        Path(args.merge_data).write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"merged scores into {args.merge_data}")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + (fmt % args) + "\n")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/operators":
            self._json(200, load_operators())
            return
        if path in ("/", "/index.html"):
            self.path = "/report.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if path == "/api/extract":
            roots = payload.get("roots") or {}
            missing = [k for k, v in roots.items() if v and not Path(v).exists()]
            if missing:
                self._json(400, {"error": "path not found", "missing": missing})
                return
            try:
                data = extract_all({k: v for k, v in roots.items() if v})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": str(e)})
                return
            self._json(200, data)
            return
        if path == "/api/score":
            extract = payload.get("extract") or {}
            hw = payload.get("hw") or {}
            perf = payload.get("perf") or {}
            measure = _read_json(str(ROOT / "measure_data.json"))
            computed = score_all(extract, hw, perf, measure.get("patterns"))
            self._json(200, computed)
            return
        self._json(404, {"error": "not found"})


def cmd_serve(args):
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving {ROOT} on http://127.0.0.1:{args.port}/report.html")
    print("POST /api/extract  {roots:{AscendC,SuperScalar,SIMT}}")
    print("POST /api/score    {extract,hw,perf}")
    httpd.serve_forever()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Extract source metrics from benchmark trees")
    e.add_argument("--ascendc", default="", help="ops-nn / ops-math / ops-transformer checkout")
    e.add_argument("--superscalar", default="", help="SuperNpuBench kernels/solution (or kernels/) root")
    e.add_argument("--simt", default="", help="CUDA kernel tree")
    e.add_argument("--operators", default=str(SCRIPTS / "operators.json"))
    e.add_argument("--out", default="extract_result.json")
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("score", help="Score x.y.z items")
    s.add_argument("--extract", required=True)
    s.add_argument("--hw", default="")
    s.add_argument("--perf", default="")
    s.add_argument("--data", default=str(ROOT / "measure_data.json"))
    s.add_argument("--out", default="computed_scores.json")
    s.add_argument("--merge-data", default="", help="Optional path to write merged measure_data.json")
    s.set_defaults(func=cmd_score)

    v = sub.add_parser("serve", help="HTTP server for report + 度量 API")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
