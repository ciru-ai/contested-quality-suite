#!/usr/bin/env python3
"""One-command launcher for the contested quality suite.

The default runs 54 lightweight selected cases. --all runs all 95 and prepares
the larger native agent runtimes. The first served OpenAI-compatible model ID
is discovered from /v1/models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import run_suite
from components.slim_humaneval_grade import EXPECTED_SHA256
from components.terminal_storage_preflight import check_space


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "bundle" / "vendor"
VENVS = ROOT / ".venvs"
PROTOCOL_BYTES = (ROOT / "protocol.json").read_bytes()
PROTOCOL = json.loads(PROTOCOL_BYTES)
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_BYTES).hexdigest()
SOURCE_TAG = hashlib.sha256((ROOT / "bundle" / "vendor-lock.json").read_bytes()).hexdigest()[:12]
AIDER_IMAGE = f"contested-aider:{SOURCE_TAG}"
HERMES_IMAGE = f"contested-hermes-verifier:{SOURCE_TAG}"
DEFAULT_COMPONENTS = ("arc_challenge_1172", "ifeval_541_strict",
                      "humaneval_plus_164", "tool_eval_hard15_v2_1_0")


def verify_vendor() -> None:
    lock = json.loads((ROOT / "bundle" / "vendor-lock.json").read_text())
    files = lock["files"]
    if lock.get("file_count") != len(files):
        raise ValueError("bundled source lock has an invalid file count")
    for relative, expected in files.items():
        path = VENDOR / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"bundled source changed or missing: {relative}")
    reference = ROOT / "selectors" / "humaneval-plus-expected.json"
    if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("HumanEval+ selected reference outputs changed or missing")


def command(argv: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required on this host")
    return path


def image_id(tag: str) -> str | None:
    if not shutil.which("docker"):
        return None
    result = subprocess.run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_setup(names: list[str]) -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")
    needs_python = set(names) & {"ifeval_541_strict", "humaneval_plus_164",
                                 "tool_eval_hard15_v2_1_0"}
    uv = require_executable("uv") if needs_python else None
    needs_docker = set(names) & {"aider_polyglot_225", "hermesagent_20",
                                 "terminal_core19_pass2"}
    if needs_docker:
        docker = require_executable("docker")
        if subprocess.run([docker, "info"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError("Docker is installed but this user cannot access the Docker daemon")
        if "terminal_core19_pass2" in names:
            check_space(PROTOCOL["defaults"]["terminal_min_docker_free_gib"])
    else:
        docker = None
    if "humaneval_plus_164" in names:
        require_executable("bwrap")
    if "hermesagent_20" in names:
        require_executable("node")
    VENVS.mkdir(parents=True, exist_ok=True)
    environments = {
        "ifeval": ["absl-py==2.5.0", "langdetect==1.0.9", "nltk==3.10.3", "immutabledict==4.3.1"],
        "he-slim": ["tree-sitter==0.26.0", "tree-sitter-python==0.25.0"],
        "tool-eval": ["-e", str(VENDOR / "tool-eval")],
    }
    needed = {"ifeval": "ifeval_541_strict", "he-slim": "humaneval_plus_164",
              "tool-eval": "tool_eval_hard15_v2_1_0"}
    for env_name, packages in environments.items():
        if needed[env_name] not in names:
            continue
        python = VENVS / env_name / "bin" / "python"
        marker = VENVS / env_name / "contested-ready.json"
        if marker.is_file() and python.is_file():
            continue
        if not python.is_file():
            command([uv, "venv", str(VENVS / env_name)])
        command([uv, "pip", "install", "--python", str(python), *packages])
        marker.write_text(json.dumps({"packages": packages}) + "\n")
    if "ifeval_541_strict" in names and not (VENDOR / "nltk_data" / "tokenizers" / "punkt_tab" / "english").is_dir():
        raise FileNotFoundError("bundled English NLTK sentence data is missing")
    hermes = VENDOR / "hermesagent20-benchmark"
    if "hermesagent_20" in names and not (hermes / "dist" / "lib" / "benchmark.js").is_file():
        raise FileNotFoundError("bundled Hermes runtime is missing")
    if "aider_polyglot_225" in names and not image_id(AIDER_IMAGE):
        command([docker, "build", "--network", "host", "-t", AIDER_IMAGE, "-f",
                 str(VENDOR / "aider" / "benchmark" / "Dockerfile.contested"),
                 str(VENDOR / "aider")])
    if "hermesagent_20" in names and not image_id(HERMES_IMAGE):
        command([docker, "build", "--network", "host", "-t", HERMES_IMAGE, "-f",
                 str(hermes / "verification" / "Dockerfile"),
                 str(hermes / "verification")])


def model_at(endpoint: str) -> dict:
    if not endpoint.startswith(("http://", "https://")) or not endpoint.rstrip("/").endswith("/v1"):
        raise ValueError("endpoint must be an OpenAI-compatible /v1 URL")
    request = urllib.request.Request(endpoint.rstrip("/") + "/models")
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        request.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"model endpoint is unavailable at {endpoint}: {exc}") from exc
    data = payload.get("data") or []
    if not isinstance(data, list) or not data or not isinstance(data[0].get("id"), str):
        raise ValueError(f"no first model ID at {endpoint}/models")
    return data[0]


def portable_config(endpoint: str, served: dict, args: argparse.Namespace) -> dict:
    model = args.model or served["id"]
    if model != served["id"]:
        raise ValueError(f"requested model {model!r} differs from endpoint's first ID {served['id']!r}")
    label = args.label or model
    profile = re.sub(r"[^A-Za-z0-9_.-]", "-", label)[:60].strip("-.") or "model"
    owner = str(served.get("owned_by", "openai-compatible")).lower()
    tool_backend = args.tool_backend or ("llamacpp" if "llama" in owner else "vllm")
    cfg = {
        "run": {"output_root": str(ROOT / "runs")},
        "protocol": {"id": PROTOCOL["id"], "version": PROTOCOL["version"],
                     "sha256": PROTOCOL_SHA256},
        "model": {"label": label, "model": model, "api_base_url": endpoint},
        "components": {
            "arc_challenge_1172": {"enabled": True,
                                   "native_harness": False},
            "ifeval_541_strict": {"enabled": True,
                                  "ifeval_python": str(VENVS / "ifeval" / "bin" / "python"),
                                  "ifeval_eval_dir": str(VENDOR / "ifeval")},
            "humaneval_plus_164": {"enabled": True,
                                   "native_harness": False,
                                   "slim_python": str(VENVS / "he-slim" / "bin" / "python"),
                                   "max_output_tokens": PROTOCOL["response_caps"]["humaneval_plus_164"]},
            "aider_polyglot_225": {"enabled": True, "mode": "docker", "host": "local",
                                   "portable": True, "auto_model_settings": True,
                                   "aider_repo": str(VENDOR / "aider"),
                                   "dataset_dir": str(VENDOR / "aider" / "tmp.benchmarks" / "polyglot-benchmark"),
                                   "overlay_path": str(VENDOR / "benchmark-bounded-feedback.py"),
                                   "endpoint": endpoint, "model": "openai/" + model,
                                   "image": AIDER_IMAGE, "image_id": image_id(AIDER_IMAGE) or "",
                                   "threads": args.aider_threads},
            "hermesagent_20": {"enabled": True, "base_url": endpoint,
                               "served_model_id": model, "profile": profile,
                               "workers": args.hermes_workers,
                               "verifier_image": HERMES_IMAGE},
            "terminal_core19_pass2": {"enabled": True, "mode": "run",
                                      "workdir": str(VENDOR / "terminal-bench-mini"),
                                      "endpoint": endpoint, "model": model,
                                      "platform": args.platform or "portable-linux",
                                      "model_name": label, "engine": args.engine or owner,
                                      "backend": args.backend or "unspecified",
                                      "context_length": args.terminal_context_length,
                                      "max_output_tokens": PROTOCOL["response_caps"]["terminal_core19_pass2"]},
            "tool_eval_hard15_v2_1_0": {"enabled": True, "base_url": endpoint,
                                        "model": model, "backend": tool_backend,
                                        "runner": str(VENVS / "tool-eval" / "bin" / "tool-eval-bench"),
                                        "backend_kwargs": {}, "max_output_tokens": PROTOCOL["response_caps"]["tool_eval_hard15_v2_1_0"]},
        },
    }
    return cfg


def selection(args: argparse.Namespace) -> list[str]:
    if args.all and args.components:
        raise ValueError("choose --all or --components, not both")
    requested = args.components or (None if args.all else ",".join(DEFAULT_COMPONENTS))
    available = {name: {"enabled": True} for name in run_suite.MANIFEST["components"]}
    return run_suite.chosen_components({"components": available}, requested)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", choices=("run", "setup", "check", "plan", "doctor", "collect"), default="run")
    p.add_argument("--endpoint", default=os.environ.get("BENCH_API_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--model", help="Optional exact first ID from /v1/models; discovered by default")
    p.add_argument("--label", help="Checkpoint/quantization label for the result")
    p.add_argument("--run-id", help="Unique output name; generated if omitted")
    p.add_argument("--run-dir", type=Path, help="Existing run directory for collect")
    p.add_argument("--components", help="Comma-separated component IDs")
    p.add_argument("--all", action="store_true", help="Run all 95 cases, including heavy agent components")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--platform")
    p.add_argument("--engine")
    p.add_argument("--backend")
    p.add_argument("--tool-backend", choices=("llamacpp", "vllm", "litellm"))
    p.add_argument("--aider-threads", type=int, default=PROTOCOL["defaults"]["aider_threads"],
                   help="Concurrent Aider tasks; default 1 for one-session endpoints")
    p.add_argument("--terminal-context-length", type=int,
                   help="Explicit Terminal model context when /v1/models omits it")
    p.add_argument("--hermes-workers", type=int, default=1)
    args = p.parse_args()
    if args.terminal_context_length is not None and args.terminal_context_length < 1:
        raise ValueError("--terminal-context-length must be positive")
    run_suite.verify_package()
    verify_vendor()
    if args.action == "check":
        print("Package manifest and selectors verified; bundled sources are present.")
        return 0
    if args.action == "setup":
        ensure_setup(selection(args))
        print("Selected component dependencies are ready.")
        return 0
    if args.action != "collect":
        served = model_at(args.endpoint)
        names = selection(args)
        ensure_setup(names)
    else:
        if not args.run_dir:
            raise ValueError("collect requires --run-dir")
        provenance = json.loads((args.run_dir / "run-metadata.json").read_text())
        if provenance.get("protocol", {}).get("sha256") != PROTOCOL_SHA256:
            raise ValueError("collect requires results generated under this protocol version")
        args.endpoint = provenance["model"]["api_base_url"]
        served = {"id": provenance["model"]["model"],
                  "owned_by": "openai-compatible"}
        args.label = provenance["model"]["label"]
        if not args.components:
            args.components = ",".join(provenance["components"])
        names = selection(args)
    config = portable_config(args.endpoint, served, args)
    run_suite.ADAPTER_MODULES = {name: "components.portable_adapter" for name in run_suite.ADAPTER_MODULES}
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir.resolve() if args.run_dir else ROOT / "runs" / run_id
    if args.action in ("plan", "doctor"):
        plan = run_suite.make_plan(config, names, run_dir)
        issues = run_suite.doctor(plan) + run_suite.launch_config_issues(config, names)
        print(json.dumps({"run_dir": str(run_dir), "steps": run_suite.preview_plan(plan),
                          "issues": issues}, indent=2))
        return 2 if args.action == "doctor" and issues else 0
    if args.action == "collect":
        result = run_suite.collect_all(config, names, run_dir)
    else:
        result = run_suite.run(config, names, run_dir, resume=args.resume,
                               continue_on_error=False)
    print(json.dumps(result, indent=2))
    return 0 if result.get("complete") else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        raise SystemExit(2)
