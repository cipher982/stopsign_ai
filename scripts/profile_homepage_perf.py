#!/usr/bin/env python3
"""Profile the homepage using a consistent public/direct/browser setup."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urljoin

DEFAULT_PUBLIC_BASE = "https://crestwoodstopsign.com"
DEFAULT_DIRECT_BASE = "http://100.125.140.78:8002"
DEFAULT_SETTLE_SECONDS = 8
DEFAULT_RUNS = 3
DEFAULT_IMAGE_COUNT = 5

DEFAULT_STATIC_PATHS = [
    "/static/base.css",
    "/static/js/home.js",
    "/static/js/video-player.js",
]

CORE_PATHS = [
    "/",
    "/load-video",
    "/api/recent-vehicle-passes",
    "/api/live-stats",
    "/stream/stream.m3u8",
]

BROWSER_EXPR = r"""(() => {
  const nav = performance.getEntriesByType("navigation")[0];
  const navSummary = nav ? {
    ttfb: Math.round(nav.responseStart - nav.requestStart),
    dcl: Math.round(nav.domContentLoadedEventEnd - nav.startTime),
    load: Math.round(nav.loadEventEnd - nav.startTime),
    total: Math.round(nav.duration)
  } : null;
  const allResources = performance.getEntriesByType("resource")
    .map(r => ({
      name: r.name,
      initiatorType: r.initiatorType,
      start: Math.round(r.startTime),
      ttfb: Math.round(r.responseStart - r.requestStart),
      download: Math.round(r.responseEnd - r.responseStart),
      total: Math.round(r.duration),
      transferSize: r.transferSize || 0,
      proto: r.nextHopProtocol
    }));
  const startupWindowMs = navSummary ? navSummary.load + 1000 : 3000;
  const startupResources = allResources.filter(r => r.start <= startupWindowMs);
  const countKinds = resources => ({
    img: resources.filter(r => r.initiatorType === "img").length,
    xhr: resources.filter(r => r.initiatorType === "xmlhttprequest").length,
    script: resources.filter(r => r.initiatorType === "script").length,
    link: resources.filter(r => r.initiatorType === "link").length
  });
  return JSON.stringify({
  nav: navSummary,
  startup_window_ms: startupWindowMs,
  startup_counts: countKinds(startupResources),
  all_counts: countKinds(allResources),
  startup_top: startupResources
    .sort((a, b) => b.total - a.total)
    .slice(0, 12),
  all_top: allResources
    .sort((a, b) => b.total - a.total)
    .slice(0, 12),
  streaming: {
    manifest_requests: allResources.filter(r => r.name.endsWith("/stream/stream.m3u8")).length,
    segment_requests: allResources.filter(r => /\/stream\/.*\.ts(?:$|\?)/.test(r.name)).length
  },
  dom: {
    recentPassCards: document.querySelectorAll("#recentPasses a").length,
    recentPassImages: document.querySelectorAll("#recentPasses img").length,
    readyState: document.readyState
  }
});
})()"""


def run_command(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    return proc.stdout


def measure_url(url: str, timeout: int = 15) -> dict[str, Any]:
    cmd = [
        "curl",
        "--max-time",
        str(timeout),
        "-L",
        "-o",
        "/dev/null",
        "-s",
        "-w",
        "%{http_code} %{time_starttransfer} %{time_total} %{size_download}",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "").strip()
    if proc.returncode == 0 and output:
        code, start, total, size = output.split()
        return {
            "ok": True,
            "code": int(code),
            "starttransfer_s": round(float(start), 3),
            "total_s": round(float(total), 3),
            "size_bytes": int(size),
        }
    return {
        "ok": False,
        "error_code": proc.returncode,
        "stdout": output,
        "stderr": (proc.stderr or "").strip(),
    }


def fetch_recent_image_paths(public_base: str, limit: int) -> list[str]:
    html = run_command(["curl", "-s", urljoin(public_base, "/api/recent-vehicle-passes")])
    found: list[str] = []
    seen: set[str] = set()
    for path in re.findall(r'src="([^"]+)"', html):
        if not path.startswith("/"):
            continue
        if path in seen:
            continue
        seen.add(path)
        found.append(path)
        if len(found) >= limit:
            break
    return found


def discover_static_asset_paths(public_base: str) -> list[str]:
    html = run_command(["curl", "-s", urljoin(public_base, "/")])
    found: list[str] = []
    seen: set[str] = set()
    for path in re.findall(r'(?:src|href)="([^"]+)"', html):
        if not path.startswith("/static/"):
            continue
        if not re.search(r"\.(?:css|js)(?:\?|$)", path):
            continue
        if path in seen:
            continue
        seen.add(path)
        found.append(path)
    return found or DEFAULT_STATIC_PATHS


def parse_browser_eval_output(output: str) -> dict[str, Any]:
    text = output.strip()
    if not text:
        raise ValueError("Empty browser-use eval output")

    if "result:" in text:
        text = text.split("result:", 1)[1].strip()

    parsed: Any
    if text.startswith("{") and text.endswith("}"):
        parsed = json.loads(text)
    else:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(f"Unexpected parsed browser payload: {type(parsed)!r}")
    return parsed


def profile_browser(public_base: str, runs: int, settle_seconds: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sessions: list[str] = []
    try:
        for run_idx in range(1, runs + 1):
            session = f"homepage-perf-{os.getpid()}-{run_idx}"
            sessions.append(session)
            url = f"{public_base}/?profile_run={int(time.time())}_{run_idx}"
            run_command(["browser-use", "--session", session, "open", url], check=True)
            time.sleep(settle_seconds)
            raw = run_command(["browser-use", "--session", session, "eval", BROWSER_EXPR], check=True)
            parsed = parse_browser_eval_output(raw)
            parsed["run"] = run_idx
            parsed["url"] = url
            results.append(parsed)
            run_command(["browser-use", "--session", session, "close"], check=False)
    finally:
        for session in sessions:
            run_command(["browser-use", "--session", session, "close"], check=False)
        run_command(["browser-use", "close", "--all"], check=False)
    return results


def build_matrix(public_base: str, direct_base: str, image_count: int) -> dict[str, Any]:
    static_paths = discover_static_asset_paths(public_base)
    image_paths = fetch_recent_image_paths(public_base, image_count)
    paths = [CORE_PATHS[0], *static_paths, *CORE_PATHS[1:], *image_paths]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "path": path,
                "public": measure_url(urljoin(public_base, path)),
                "direct": measure_url(urljoin(direct_base, path)),
            }
        )
    return {
        "paths": paths,
        "static_paths": static_paths,
        "image_paths": image_paths,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-base", default=DEFAULT_PUBLIC_BASE)
    parser.add_argument("--direct-base", default=DEFAULT_DIRECT_BASE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--settle-seconds", type=int, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--image-count", type=int, default=DEFAULT_IMAGE_COUNT)
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--output", help="Optional path to write JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "public_base": args.public_base,
        "direct_base": args.direct_base,
        "runs": args.runs,
        "settle_seconds": args.settle_seconds,
        "image_count": args.image_count,
    }
    if not args.skip_matrix:
        data["matrix"] = build_matrix(args.public_base, args.direct_base, args.image_count)
    if not args.skip_browser:
        data["browser_runs"] = profile_browser(args.public_base, args.runs, args.settle_seconds)

    output = json.dumps(data, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
            fh.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
