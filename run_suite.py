#!/usr/bin/env python3
"""Plan, run, resume, and score the seven-component contested quality suite.

Each component keeps its original task environment and grader. Adapters launch
the native harness and normalize its result records; score_suite.py aggregates
only the selected cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from score_suite import score


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "suite.json").read_text())
ADAPTER_MODULES = {
    "arc_challenge_1172": "components.text_coding_adapter",
    "ifeval_541_strict": "components.text_coding_adapter",
    "humaneval_plus_164": "components.text_coding_adapter",
    "aider_polyglot_225": "components.aider_terminal_adapter",
    "terminal_core19_pass2": "components.aider_terminal_adapter",
    "hermesagent_20": "components.agent_tool_adapter",
    "tool_eval_hard15_v2_1_0": "components.agent_tool_adapter",
}
SENSITIVE = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
LOCKED_INPUT_SHA256 = {
    "protocol.json": "fb7c60a51f3dd82a9357dcad3dd7756d0199cfd5e6854756013899bebab9d2ae",
    "selectors/aider-tasks.txt": "04111c1d039977f244bf90d8ed79b14269b086b825638d3f2b0f2b32bf754d61",
    "selectors/arc-challenge.jsonl": "977f6210e66686ea363a203a49819ff6edf0c40fcb9059d42487b19313d2d452",
    "selectors/hermesagent-scenarios.txt": "c759090d2604b4d19f51c43a08fa8ef672cbb257bda1de187008c1b88a0da3fe",
    "selectors/humaneval-plus-native.jsonl": "eb78ce3e4392a99e1f182cd15b460a4881af0f2fdbbe42d9a8805d654799bc0f",
    "selectors/humaneval-plus.jsonl": "094108c65d24e7d238809a613ac3b4eeee9e4afab2791008fde84c9bfc5758ba",
    "selectors/ifeval-input-data.jsonl": "43ee03d362f54f4fcc376d93229eaf9416f8689e7c5ff096088e8dbf99728f2d",
    "selectors/tool-eval-hard15-v2.1.0.txt": "b8eac80e9b7d9281a902c9a993a32b6b5b7cbda40be05514e0c4d77d7ba803e7",
    "terminal-core19-hotlist.json": "5a92ebc70cd08b8f2c0a73af574337f5f5838526f5790faac905a9e4eb60aece",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def verify_package() -> None:
    cases = MANIFEST["cases"]
    identities = [{key: case[key] for key in ("suite", "case_id", "tier")} for case in cases]
    if digest(identities) != MANIFEST["selection_sha256"]:
        raise ValueError("suite.json selection SHA-256 mismatch")
    count_fields = {"case_count", "contested_count", "frontier_count"}
    component_specs = {name: {key: value for key, value in spec.items() if key not in count_fields}
                       for name, spec in MANIFEST["components"].items()}
    content = {"cases": [{key: case[key] for key in ("suite", "case_id", "tier", "task")}
                         for case in cases], "component_specs": component_specs}
    if digest(content) != MANIFEST["content_sha256"]:
        raise ValueError("suite.json content SHA-256 mismatch")
    for relative, expected in LOCKED_INPUT_SHA256.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"locked benchmark input changed or missing: {relative}")


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if not isinstance(config.get("components"), dict):
        raise ValueError("config must define [components.<suite_id>] sections")
    if not isinstance(config.get("model"), dict):
        raise ValueError("config must define a [model] section")
    return config


def chosen_components(config: dict, requested: str | None) -> list[str]:
    available = list(MANIFEST["components"])
    if requested:
        names = [part.strip() for part in requested.split(",") if part.strip()]
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValueError(f"unknown component IDs: {', '.join(unknown)}")
        if len(names) != len(set(names)):
            raise ValueError("duplicate component selection")
        return names
    return [name for name in available
            if config["components"].get(name, {}).get("enabled", False)]


def component_config(config: dict, name: str) -> dict:
    raw = config["components"].get(name)
    if not isinstance(raw, dict):
        raise ValueError(f"missing [components.{name}] section")
    merged = {**config.get("model", {}), **raw}
    merged.pop("enabled", None)
    return merged


def launch_config_issues(config: dict, names: list[str]) -> list[str]:
    """Reject template values and inconsistent checkpoint labels before inference."""
    issues = []
    label = config.get("model", {}).get("label")
    if not isinstance(label, str) or not label.strip() or label.startswith(("REPLACE_WITH_", "YOUR_")):
        issues.append("[model].label must identify the actual checkpoint and quantization")
    for name in names:
        cfg = component_config(config, name)
        if cfg.get("mode", "run") in ("import", "collect"):
            continue
        if cfg.get("label") != label:
            issues.append(f"{name}: component label must equal the common [model].label")
        for key, value in cfg.items():
            if key == "label" or (name == "hermesagent_20" and key == "model"):
                continue
            if isinstance(value, str) and ("REPLACE_WITH_" in value or "YOUR_" in value):
                issues.append(f"{name}: configure {key} (template placeholder remains)")
    return issues


def component_provenance(config: dict, names: list[str]) -> dict:
    provenance = {}
    for name in names:
        cfg = component_config(config, name)
        provenance[name] = {
            "checkpoint_label": cfg.get("label"),
            "served_model_id": cfg.get("served_model_id", cfg.get("model")),
            "endpoint": next((cfg[key] for key in ("endpoint", "base_url", "api_base_url")
                              if key in cfg), None),
            "host": cfg.get("host", "local"),
            "mode": cfg.get("mode", "run"),
            "aider_threads": cfg.get("threads") if name == "aider_polyglot_225" else None,
            "context_length_override": cfg.get("context_length") if name == "terminal_core19_pass2" else None,
            "max_output_tokens": cfg.get("max_output_tokens"),
        }
    return provenance


def adapter(name: str):
    return importlib.import_module(ADAPTER_MODULES[name])


def validate_step(name: str, step: dict, number: int) -> dict:
    if not isinstance(step, dict):
        raise ValueError(f"{name} step {number} must be an object")
    argv = step.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
        raise ValueError(f"{name} step {number} requires a nonempty argv string list")
    env = step.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError(f"{name} step {number} env must map strings to strings")
    cwd = step.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError(f"{name} step {number} cwd must be a string")
    label = step.get("name") or f"step-{number}"
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"{name} step {number} has an invalid name")
    timeout = step.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError(f"{name} step {number} has an invalid timeout_seconds")
    return {"name": label, "argv": argv, "cwd": cwd, "env": env,
            "timeout_seconds": timeout}


def make_plan(config: dict, names: list[str], run_dir: Path) -> dict:
    verify_package()
    planned = {}
    for name in names:
        cfg = component_config(config, name)
        raw_steps = adapter(name).build_steps(name, cfg, ROOT, run_dir)
        if not isinstance(raw_steps, list):
            raise ValueError(f"{name} adapter build_steps must return a list")
        planned[name] = [validate_step(name, step, i) for i, step in enumerate(raw_steps, 1)]
        if not planned[name] and cfg.get("mode") not in ("import", "collect"):
            raise ValueError(f"{name} has no launch steps; use mode='import' with native results")
    return planned


def redact_step(step: dict) -> dict:
    argv = []
    redact_next = False
    for token in step["argv"]:
        if redact_next:
            argv.append("<redacted>")
            redact_next = False
        elif any(token.lower() == flag for flag in ("--api-key", "--token", "--password", "--secret")):
            argv.append(token)
            redact_next = True
        elif any(token.lower().startswith(flag) for flag in ("--api-key=", "--token=", "--password=", "--secret=")):
            argv.append(token.split("=", 1)[0] + "=<redacted>")
        else:
            argv.append(token)
    return {"name": step["name"], "argv": argv, "cwd": step["cwd"],
            "env_keys": sorted(step["env"]), "timeout_seconds": step["timeout_seconds"]}


def public_plan(plan: dict) -> dict:
    return {name: [redact_step(step) for step in steps] for name, steps in plan.items()}


def preview_plan(plan: dict) -> dict:
    preview = {}
    for component, steps in plan.items():
        preview[component] = []
        for step in steps:
            public = redact_step(step)
            first = public["argv"][:3]
            argv_preview = [value if len(value) <= 100 else value[:97] + "..." for value in first]
            preview[component].append({
                "name": step["name"], "argv_preview": argv_preview,
                "cwd": step["cwd"], "command_sha256": digest(public),
            })
    return preview


def doctor(plan: dict) -> list[str]:
    issues = []
    for name, steps in plan.items():
        for step in steps:
            command = step["argv"][0]
            if "/" in command and not Path(command).exists():
                issues.append(f"{name}/{step['name']}: executable not found: {command}")
            elif "/" not in command and shutil.which(command) is None:
                issues.append(f"{name}/{step['name']}: command not in PATH: {command}")
            if step["cwd"] and not Path(step["cwd"]).is_dir():
                issues.append(f"{name}/{step['name']}: cwd not found: {step['cwd']}")
            if (len(step["argv"]) > 1 and "python" in Path(command).name.lower()
                    and step["argv"][1].endswith(".py")
                    and not Path(step["argv"][1]).is_file()):
                issues.append(f"{name}/{step['name']}: Python script not found: {step['argv'][1]}")
    return issues


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_event(run_dir: Path, event: dict) -> None:
    with (run_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at_utc": utc_now(), **event}, ensure_ascii=False) + "\n")


def execute_step(name: str, index: int, step: dict, run_dir: Path) -> dict:
    label = f"{name}-{index:02d}-{step['name']}"
    out_path = run_dir / "logs" / f"{label}.stdout.log"
    err_path = run_dir / "logs" / f"{label}.stderr.log"
    out_path.parent.mkdir(exist_ok=True)
    append_event(run_dir, {"event": "step_start", "component": name, "step": step["name"],
                           "index": index})
    started = time.monotonic()
    print(f"[{name}] {step['name']} ...", flush=True)
    try:
        with out_path.open("w") as stdout, err_path.open("w") as stderr:
            result = subprocess.run(step["argv"], cwd=step["cwd"],
                                    env={**os.environ, **step["env"]},
                                    stdout=stdout, stderr=stderr,
                                    timeout=step["timeout_seconds"], check=False)
        return_code = result.returncode
        detail = None
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        detail = f"timeout after {exc.timeout} seconds"
    except OSError as exc:
        return_code = 127
        detail = str(exc)
    record = {"return_code": return_code, "duration_seconds": round(time.monotonic() - started, 3),
              "stdout_log": str(out_path), "stderr_log": str(err_path), "detail": detail}
    append_event(run_dir, {"event": "step_end", "component": name, "step": step["name"],
                           "index": index, **record})
    print(f"[{name}] {step['name']}: {'ok' if return_code == 0 else f'failed ({return_code})'}", flush=True)
    return record


def normalize_component(name: str, raw_rows: list[dict]) -> tuple[list[dict], int]:
    if not isinstance(raw_rows, list):
        raise ValueError(f"{name} collector must return a list")
    wanted = {case["case_id"] for case in MANIFEST["cases"] if case["suite"] == name}
    seen = set()
    output = []
    ignored = 0
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("suite") != name or not isinstance(row.get("case_id"), str):
            raise ValueError(f"{name} collector returned a malformed outcome")
        case_id = row["case_id"]
        if case_id not in wanted:
            ignored += 1
            continue
        if case_id in seen:
            raise ValueError(f"{name} collector returned duplicate case {case_id}")
        if type(row.get("passed")) is not bool:
            raise ValueError(f"{name}/{case_id} collector passed must be boolean")
        seen.add(case_id)
        output.append({"suite": name, "case_id": case_id, "passed": row["passed"]})
    missing = wanted - seen
    if missing:
        raise ValueError(f"{name} collector missing {len(missing)} selected cases: "
                         + ", ".join(sorted(missing)[:5]))
    order = {case["case_id"]: i for i, case in enumerate(MANIFEST["cases"]) if case["suite"] == name}
    output.sort(key=lambda row: order[row["case_id"]])
    return output, ignored


def collect_all(config: dict, names: list[str], run_dir: Path) -> dict:
    verify_package()
    all_rows = []
    errors = {}
    counts = {}
    for name in names:
        try:
            rows, ignored = normalize_component(name, adapter(name).collect(
                name, component_config(config, name), ROOT, run_dir))
            all_rows.extend(rows)
            counts[name] = {"selected": len(rows), "ignored_parent_rows": ignored}
            component_path = run_dir / "outcomes" / f"{name}.jsonl"
            component_path.parent.mkdir(exist_ok=True)
            component_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            print(f"[{name}] collected {len(rows)} selected outcomes", flush=True)
        except Exception as exc:  # Keep completed components when another native run is unavailable.
            errors[name] = str(exc)
            stale_component = run_dir / "outcomes" / f"{name}.jsonl"
            stale_component.unlink(missing_ok=True)
            print(f"[{name}] collection failed: {exc}", file=sys.stderr, flush=True)
    output_path = run_dir / "outcomes.jsonl"
    output_path.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    report = score(output_path, allow_partial=True)
    if config.get("protocol"):
        report["protocol"] = config["protocol"]
    suite_complete = report["completed_cases"] == report["total_cases"]
    requested_complete = len(counts) == len(names) and not errors
    score_path = run_dir / ("score.json" if suite_complete else "score.partial.json")
    (run_dir / ("score.partial.json" if suite_complete else "score.json")).unlink(missing_ok=True)
    write_json(score_path, report)
    summary = {"complete": requested_complete, "suite_complete": suite_complete,
               "score_path": str(score_path), "outcomes_path": str(output_path),
               "component_counts": counts, "errors": errors,
               "completed_cases": report["completed_cases"], "total_cases": report["total_cases"]}
    write_json(run_dir / "collection.json", summary)
    return summary


def run(config: dict, names: list[str], run_dir: Path, *, resume: bool,
        continue_on_error: bool) -> dict:
    if config.get("protocol") and config["protocol"].get("sha256") != LOCKED_INPUT_SHA256["protocol.json"]:
        raise ValueError("run protocol hash differs from the packaged protocol")
    config_problems = launch_config_issues(config, names)
    if config_problems:
        raise ValueError("run configuration needs setup:\n" + "\n".join(config_problems))
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {run_dir}; use --resume")
    os.umask(0o077)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = make_plan(config, names, run_dir)
    problems = doctor(plan)
    if problems:
        raise RuntimeError("preflight failed:\n" + "\n".join(problems))
    plan_hash = digest(plan)
    config_hash = digest(config)
    state_path = run_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not resume or state.get("plan_sha256") != plan_hash or state.get("config_sha256") != config_hash:
            raise ValueError("resume requires the same config, component list, and command plan")
        if state.get("suite_content_sha256") != MANIFEST["content_sha256"]:
            raise ValueError("suite content changed since this run began")
        if state.get("protocol_sha256") != config.get("protocol", {}).get("sha256"):
            raise ValueError("run protocol changed since this run began")
    else:
        state = {"suite_id": MANIFEST["id"], "suite_content_sha256": MANIFEST["content_sha256"],
                 "protocol_sha256": config.get("protocol", {}).get("sha256"),
                 "plan_sha256": plan_hash, "config_sha256": config_hash, "created_at_utc": utc_now(),
                 "steps": {}}
        write_json(state_path, state)
    write_json(run_dir / "plan.json", public_plan(plan))
    write_json(run_dir / "run-metadata.json", {
        "suite_id": MANIFEST["id"], "selection_sha256": MANIFEST["selection_sha256"],
        "content_sha256": MANIFEST["content_sha256"], "run_dir": str(run_dir),
        "protocol": config.get("protocol"),
        "components": names, "model": {k: v for k, v in config.get("model", {}).items()
                                      if not any(secret in k.upper() for secret in SENSITIVE)},
        "component_model_provenance": component_provenance(config, names),
        "created_at_utc": state["created_at_utc"],
    })
    failed = {}
    finished_components = []
    for name, steps in plan.items():
        for index, step in enumerate(steps, 1):
            key = f"{name}:{index}:{step['name']}"
            if state["steps"].get(key, {}).get("return_code") == 0:
                print(f"[{name}] {step['name']}: already complete", flush=True)
                continue
            outcome = execute_step(name, index, step, run_dir)
            state["steps"][key] = outcome
            write_json(state_path, state)
            if outcome["return_code"] != 0:
                failed[name] = f"step {step['name']} returned {outcome['return_code']}"
                if not continue_on_error:
                    partial = collect_all(config, finished_components, run_dir) if finished_components else None
                    print(f"Stopped after {name}; inspect the native log and adapter resume instructions.", file=sys.stderr)
                    return {"complete": False, "execution_errors": failed,
                            "run_dir": str(run_dir), "partial_collection": partial}
                break
        if name not in failed:
            finished_components.append(name)
    summary = collect_all(config, names, run_dir)
    summary["execution_errors"] = failed
    if failed:
        summary["complete"] = False
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "doctor", "run", "collect"))
    parser.add_argument("--config", type=Path, required=True, help="TOML native harness and model configuration")
    parser.add_argument("--components", help="Comma-separated component IDs; defaults to enabled config sections")
    parser.add_argument("--run-id", help="Safe name under [run].output_root for plan or run")
    parser.add_argument("--run-dir", type=Path, help="Existing run directory for collect")
    parser.add_argument("--resume", action="store_true", help="Resume a run with the same plan and config")
    parser.add_argument("--continue-on-error", action="store_true", help="Attempt other components after a failed step")
    parser.add_argument("--verbose", action="store_true", help="Show full redacted argv in plan or doctor output")
    args = parser.parse_args()
    config = load_config(args.config)
    names = chosen_components(config, args.components)
    if not names:
        raise ValueError("no components selected; enable sections or use --components")
    output_root = Path(config.get("run", {}).get("output_root", ROOT / "runs")).expanduser()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        run_id = args.run_id or ("preview" if args.action in ("plan", "doctor") else
                                 datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
            raise ValueError("--run-id must use letters, digits, dot, underscore, or hyphen")
        run_dir = output_root / run_id
    if args.action == "collect":
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        result = collect_all(config, names, run_dir)
    elif args.action in ("plan", "doctor"):
        plan = make_plan(config, names, run_dir)
        config_issues = launch_config_issues(config, names)
        result = {"suite_id": MANIFEST["id"], "run_dir": str(run_dir),
                  "selection_sha256": MANIFEST["selection_sha256"],
                  "content_sha256": MANIFEST["content_sha256"],
                  "steps": public_plan(plan) if args.verbose else preview_plan(plan),
                  "preflight_issues": doctor(plan) + (config_issues if args.action == "doctor" else []),
                  "component_model_provenance": component_provenance(config, names)}
    else:
        result = run(config, names, run_dir, resume=args.resume,
                     continue_on_error=args.continue_on_error)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.action == "doctor" and result["preflight_issues"]:
        return 2
    if args.action in ("run", "collect") and not result.get("complete", False):
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
