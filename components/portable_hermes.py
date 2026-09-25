#!/usr/bin/env python3
"""Prepare and launch the bundled HermesAgent-20 native runner locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


SOURCE_SHA256 = "01548df005a9c3feec092767a49a4499ddcca0df6cdc81b3201299f9ceb0e8ca"
OLD_ROOT = "const ROOT = '/opt/contested-quality-suite';"
OLD_RUN = "const runDir = path.join(ROOT, `hermes20-${profile}-c${workers}${smoke ? '-smoke' : ''}`);"
NEW_RUN = ("const runDir = process.env.HA20_RUN_DIR;\n"
           "if (!runDir || !path.isAbsolute(runDir)) throw new Error('HA20_RUN_DIR must be absolute');")
OLD_SELECTED = "const selected = smoke ? SCENARIOS.filter(s => s.id === smokeScenario) : SCENARIOS.filter(s => !doneIds.has(s.id));"
OLD_STATUS = "status:results.length===SCENARIOS.length?'completed':smoke?'smoke':'incomplete'"


def served_model(endpoint: str) -> str:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=10) as response:
        payload = json.load(response)
    data = payload.get("data") or []
    if not data or not isinstance(data[0].get("id"), str):
        raise ValueError("model endpoint returned no first model ID")
    return data[0]["id"]


def prepare(args: argparse.Namespace) -> None:
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("bundled Hermes runner source differs from pinned native version")
    if served_model(args.endpoint) != args.model:
        raise ValueError("Hermes endpoint first model ID differs from requested model")
    benchmark = args.benchmark.resolve()
    if not (benchmark / "dist" / "lib" / "benchmark.js").is_file():
        raise FileNotFoundError("Hermes benchmark not built; run ./benchmark setup")
    image = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
                           capture_output=True, text=True, check=True).stdout.strip()
    text = source.decode()
    if any(text.count(site) != 1 for site in (OLD_ROOT, OLD_RUN, OLD_SELECTED, OLD_STATUS)):
        raise ValueError("Hermes source patch sites changed")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Hermes native output already exists: {output}")
    output.mkdir(parents=True)
    patched = text.replace(OLD_ROOT, "const ROOT = " + json.dumps(str(benchmark.parent)) + ";")
    patched = patched.replace(OLD_RUN, NEW_RUN)
    selected_ids = args.ids.split(",")
    if len(selected_ids) != len(set(selected_ids)) or not selected_ids:
        raise ValueError("Hermes selected scenario IDs are empty or duplicate")
    patched = patched.replace(OLD_SELECTED,
                              "const HOT_IDS = new Set(" + json.dumps(selected_ids) + ");\n"
                              "const selected = smoke ? SCENARIOS.filter(s => HOT_IDS.has(s.id) && s.id === smokeScenario) : SCENARIOS.filter(s => HOT_IDS.has(s.id) && !doneIds.has(s.id));")
    patched = patched.replace(OLD_STATUS,
                              "status:results.length===HOT_IDS.size?'completed':smoke?'smoke':'incomplete'")
    (output / "run_hermes20.mjs").write_text(patched)
    (output / "portable-protocol.json").write_text(json.dumps({
        "source_sha256": SOURCE_SHA256, "benchmark_root": str(benchmark),
        "endpoint": args.endpoint, "model": args.model, "image": args.image,
        "selected_ids": selected_ids,
        "image_id": image}, indent=2) + "\n")


def launch(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    runner = output / "run_hermes20.mjs"
    if not runner.is_file():
        raise FileNotFoundError(f"Hermes prepared runner missing: {runner}")
    env = os.environ.copy()
    env.update({"HA20_RUN_DIR": str(output), "HA20_MODEL_BASE_URL": args.endpoint,
                "HA20_VERIFIER_IMAGE": args.image, "HA20_SMOKE": "0",
                "HA20_RESUME": "1" if (output / "progress.json").exists() else "0"})
    os.execvpe(args.node, [args.node, str(runner), args.profile, str(args.workers)], env)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for flag in ("source", "benchmark", "output"):
        prep.add_argument("--" + flag, type=Path, required=True)
    for flag in ("endpoint", "model", "image"):
        prep.add_argument("--" + flag, required=True)
    prep.add_argument("--ids", required=True)
    run = sub.add_parser("launch")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--image", required=True)
    run.add_argument("--profile", required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--node", default="node")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        launch(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"Hermes portable runner: {exc}", file=sys.stderr)
        raise SystemExit(2)
