"""Native Aider Polyglot and Terminal-Bench adapters for the contested suite.

The public contract is ``build_steps(component, cfg, root, run_dir)`` and
``collect(component, cfg, root, run_dir)``. Steps are ordinary subprocess argv
arrays; the caller owns process supervision and logs. Collection returns only
native-grader decisions, never model-generated self assessments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any


AIDER = "aider_polyglot_225"
TERMINAL = "terminal_core19_pass2"
SERVER_SH = Path.home() / ".codex/skills/work-on-home-servers/scripts/server.sh"
DEFAULT_TERMINAL_WORKDIR = Path(__file__).resolve().parents[1] / "bundle/vendor/terminal-bench-mini"
AIDER_REPO_REV = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
AIDER_DATA_REV = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
AIDER_OVERLAY_SHA = "2497a20ad688e6422c838318f408f0ae3c67793504d550a137d6d1a58f04c99e"
AIDER_PORTABLE_OVERLAY_SHA = "874d9c837a5f2c5c68aa794cd48a213a2536ae151ef0a91d01cd29301b0efdbb"
AIDER_IMAGE = "ciru/aider-benchmark-jest29:20260922"
AIDER_IMAGE_ID = "sha256:d57bb69aa426865008da92a439830d8d8cadd0014d57cc1c242902c99ed1b7b8"
AIDER_REMOTE_ROOT = Path("/opt/benchmark-host/runs")
TERMINAL_SUITE_ID = "qwen38-contested-terminal-core19-v1"
TERMINAL_MANIFEST_HASH = "fbd5c22b2891c2eab2c3d27b1b918e6f5951a748f5616d3d26efefbe0075a63a"


class AdapterError(ValueError):
    """Invalid configuration, native output, or benchmark provenance."""


def _required(cfg: dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{key} must be a non-empty string")
    return value.strip()


def _selected(root: Path, component: str) -> list[str]:
    manifest = json.loads((root / "suite.json").read_text())
    cases = [row["case_id"] for row in manifest["cases"] if row["suite"] == component]
    if not cases or len(cases) != len(set(cases)):
        raise AdapterError(f"{component}: selection is empty or contains duplicate IDs")
    return cases


def _step(name: str, argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    return {"name": name, "argv": argv, "cwd": str(cwd) if cwd else None, "env": None}


def _host(cfg: dict[str, Any]) -> str | None:
    host = cfg.get("host")
    if host in (None, "", "local"):
        return None
    if host not in ("benchmark-host", "benchmark-host"):
        raise AdapterError("Aider host must be 'local' or 'benchmark-host'")
    if not SERVER_SH.is_file():
        raise AdapterError(f"server wrapper does not exist: {SERVER_SH}")
    return "benchmark-host"


def _on_host(argv: list[str], host: str | None) -> list[str]:
    return [str(SERVER_SH), host, "run", shlex.join(argv)] if host else argv


def _remote_output_dir(cfg: dict[str, Any], run_dir: Path) -> str:
    run_id = run_dir.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in (".", ".."):
        raise AdapterError(f"run directory basename is not a safe remote run ID: {run_id!r}")
    explicit = cfg.get("remote_output_dir")
    if explicit:
        if not isinstance(explicit, str):
            raise AdapterError("remote_output_dir must be a string")
        value = explicit.replace("{run_id}", run_id)
    else:
        base = Path(cfg.get("remote_root") or AIDER_REMOTE_ROOT)
        if not base.is_absolute():
            raise AdapterError("remote_root must be absolute on the benchmark host")
        value = str(base / run_id)
    path = Path(value)
    if not path.is_absolute() or "{" in value or "}" in value:
        raise AdapterError("remote_output_dir must be an absolute path; only {run_id} is supported")
    return str(path)


# This script executes on the host that owns the Aider checkout. It validates
# the source and selector before creating a unique output directory. The native
# benchmark itself runs in Docker in a later step.
_AIDER_PREPARE = r'''import hashlib, json, pathlib, shutil, subprocess, sys
d = json.loads(sys.argv[1])
p = lambda key: pathlib.Path(d[key]).expanduser().resolve()
repo, dataset, overlay, output = (p(x) for x in ('aider_repo','dataset_dir','overlay_path','output_dir'))
settings = p('model_settings') if d.get('model_settings') else None
for name, path in [('aider_repo',repo),('dataset_dir',dataset),('overlay_path',overlay)]:
    if not path.exists(): raise SystemExit(f'{name} does not exist: {path}')
if settings and not settings.is_file(): raise SystemExit(f'model_settings does not exist: {settings}')
if not settings and not d.get('model_settings_text'): raise SystemExit('Aider model settings are missing')
def rev(path): return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()
if not d.get('portable'):
    if rev(repo) != d['aider_repo_rev']: raise SystemExit('Aider checkout revision differs from pinned protocol')
    if rev(dataset) != d['dataset_rev']: raise SystemExit('Aider dataset revision differs from pinned protocol')
if hashlib.sha256(overlay.read_bytes()).hexdigest() != d['overlay_sha256']: raise SystemExit('Aider patched harness SHA-256 differs from pinned protocol')
tasks = sorted(x.relative_to(dataset).as_posix() for x in dataset.glob('*/exercises/practice/*') if x.is_dir())
selected = d['task_ids']
matches = [task for task in tasks for key in selected if key in task]
expected_count = len(selected) if d.get('portable') else 225
if len(tasks) != expected_count or len(matches) != len(selected) or sorted(matches) != sorted(selected):
    raise SystemExit(f'Aider source/keyword selector mismatch: dataset={len(tasks)}, matches={len(matches)}, expected={len(selected)}')
if not shutil.which('docker'): raise SystemExit('Docker CLI is unavailable on Aider host')
image_id = subprocess.check_output(['docker','image','inspect',d['image'],'--format','{{.Id}}'], text=True).strip()
if image_id != d['image_id']: raise SystemExit(f'Aider Docker image ID differs from pinned setup: {image_id}')
if d.get('api_env_file') and not pathlib.Path(d['api_env_file']).is_file(): raise SystemExit('Aider API env file is missing')
if output.exists(): raise SystemExit(f'Aider output already exists; refusing to reuse: {output}')
if not d.get('prepare'):
    print(f'Aider doctor passed: {len(selected)} tasks; new output={output}')
    raise SystemExit(0)
output.mkdir(parents=True, exist_ok=False)
settings_bytes = settings.read_bytes() if settings else d['model_settings_text'].encode()
(output/'model-settings.yml').write_bytes(settings_bytes)
(output/'hotlist-protocol.json').write_text(json.dumps({**d, 'model_settings_sha256': hashlib.sha256(settings_bytes).hexdigest()}, indent=2)+'\n')
print(f'Aider prepared: {len(selected)} tasks; output={output}')
'''


_AIDER_RUN_GUARD = r'''import os, pathlib, sys
output = pathlib.Path(sys.argv[1])
if not output.is_dir(): raise SystemExit(f'Aider output was not prepared: {output}')
if any(output.rglob('.aider.results.json')):
    raise SystemExit(f'Aider output already contains native results; use a fresh run ID rather than overwriting: {output}')
os.execvp(sys.argv[2], sys.argv[2:])
'''


_TERMINAL_RUN_GUARD = r'''import os, pathlib, sys
job = pathlib.Path(sys.argv[1])
if job.exists():
    raise SystemExit(f'Terminal native job already exists: {job}; use terminal_bench.py resume on this job, then run_suite.py collect')
os.execv(sys.argv[2], sys.argv[2:])
'''


def _aider_settings(cfg: dict[str, Any], root: Path, run_dir: Path) -> tuple[dict[str, Any], str | None]:
    host = _host(cfg)
    if host:
        output_dir = _remote_output_dir(cfg, run_dir)
    else:
        output_dir = str(Path(cfg.get("output_dir") or run_dir / AIDER).expanduser().resolve())
    settings = {
        "aider_repo": _required(cfg, "aider_repo"),
        "dataset_dir": _required(cfg, "dataset_dir"),
        "overlay_path": _required(cfg, "overlay_path"),
        "model_settings": None if cfg.get("auto_model_settings") else _required(cfg, "model_settings"),
        "output_dir": output_dir,
        "aider_repo_rev": AIDER_REPO_REV,
        "dataset_rev": AIDER_DATA_REV,
        "overlay_sha256": AIDER_PORTABLE_OVERLAY_SHA if cfg.get("portable") else AIDER_OVERLAY_SHA,
        "task_ids": _selected(root, AIDER),
        "model": _required(cfg, "model"),
        "endpoint": _required(cfg, "endpoint"),
        "image": str(cfg.get("image") or AIDER_IMAGE),
        "tries": 2,
        "edit_format": "whole",
        "portable": bool(cfg.get("portable", False)),
    }
    settings["image_id"] = str(cfg.get("image_id") or (AIDER_IMAGE_ID if settings["image"] == AIDER_IMAGE else ""))
    if not settings["image_id"].startswith("sha256:"):
        raise AdapterError("A custom Aider image requires its exact image_id sha256 digest")
    if cfg.get("auto_model_settings"):
        # The benchmark protocol's model-specific settings with only the
        # served model name substituted; no external model-settings file needed.
        settings["model_settings_text"] = (
            "- name: " + json.dumps(settings["model"]) + "\n"
            "  edit_format: whole\n"
            "  weak_model_name: null\n"
            "  use_repo_map: false\n"
            "  use_temperature: 0\n"
            "  streaming: false\n"
            "  extra_params:\n"
            "    stream: false\n"
            "    max_tokens: 16384\n"
            "    temperature: 0\n"
            "    extra_body:\n"
            "      chat_template_kwargs:\n"
            "        enable_thinking: false\n"
        )
    if cfg.get("api_env_file"):
        settings["api_env_file"] = _required(cfg, "api_env_file")
    return settings, host


def _build_aider_steps(cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    if cfg.get("mode", "import") == "import":
        _required(cfg, "results_path")
        return []
    if cfg.get("mode") != "docker":
        raise AdapterError("Aider mode must be 'docker' or 'import'")
    d, host = _aider_settings(cfg, root, run_dir)
    model = d["model"]
    if not model.startswith("openai/"):
        raise AdapterError("Aider model must be the native openai/<served-id> value")
    if not d["endpoint"].startswith(("http://", "https://")):
        raise AdapterError("Aider endpoint must be an HTTP(S) OpenAI-compatible base URL")
    threads = cfg.get("threads", 8)
    if type(threads) is not int or threads < 1:
        raise AdapterError("Aider threads must be a positive integer")
    if cfg.get("authenticated"):
        if host and not d.get("api_env_file"):
            raise AdapterError("remote authenticated Aider run requires api_env_file on the benchmark host")
        if not host and not os.getenv("OPENAI_API_KEY"):
            raise AdapterError("local authenticated Aider run requires OPENAI_API_KEY in the environment")
    output_dir = d["output_dir"]
    doctor_payload = json.dumps({**d, "prepare": False}, separators=(",", ":"))
    prepare_payload = json.dumps({**d, "prepare": True}, separators=(",", ":"))
    doctor = _on_host(["python3", "-c", _AIDER_PREPARE, doctor_payload], host)
    prepare = _on_host(["python3", "-c", _AIDER_PREPARE, prepare_payload], host)
    inner = [
        "./benchmark/benchmark.py", "/benchmarks/full",
        "--model", model,
        "--edit-format", "whole",
        "--threads", str(threads),
        "--tries", "2",
        "--read-model-settings", "/benchmarks/model-settings.yml",
        "--exercises-dir", "polyglot-benchmark",
        "--keywords", ",".join(d["task_ids"]),
    ]
    mounts = [
        (d["aider_repo"], "/aider", "readonly"),
        (d["overlay_path"], "/aider/benchmark/benchmark.py", "readonly"),
        (output_dir, "/benchmarks", ""),
        (d["dataset_dir"], "/benchmarks/polyglot-benchmark", "readonly"),
    ]
    docker = ["docker", "run", "--rm", "--network", "host", "--memory=48g", "--memory-swap=48g", "--cpus=24"]
    for source, target, readonly in mounts:
        mount = f"type=bind,src={source},dst={target}"
        if readonly:
            mount += ",readonly"
        docker.extend(["--mount", mount])
    if cfg.get("authenticated") and host:
        docker.extend(["--env-file", d["api_env_file"]])
    docker.extend([
        "-e", "OPENAI_API_BASE=" + d["endpoint"],
        *(["-e", "OPENAI_API_KEY"] if cfg.get("authenticated") and not host else []),
        *(["-e", "OPENAI_API_KEY=local"] if not cfg.get("authenticated") else []),
        "-e", "AIDER_DOCKER=1", "-e", "AIDER_BENCHMARK_DIR=/benchmarks",
        *(["-e", "AIDER_PORTABLE_SOURCE_REV=" + d["aider_repo_rev"]] if d["portable"] else []),
        "-e", "PYTHONHASHSEED=0", d["image"], "bash", "-lc", shlex.join(inner),
    ])
    return [
        _step("aider-doctor", doctor),
        _step("aider-prepare", prepare),
        _step("aider-run", _on_host(["python3", "-c", _AIDER_RUN_GUARD, output_dir, *docker], host)),
    ]


def _build_terminal_steps(cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    if cfg.get("mode", "run") == "import":
        _required(cfg, "results_path")
        return []
    if cfg.get("mode", "run") != "run":
        raise AdapterError("Terminal mode must be 'run' or 'import'")
    workdir = Path(cfg.get("workdir") or DEFAULT_TERMINAL_WORKDIR).expanduser().resolve()
    runner = workdir / "terminal_bench.py"
    if not runner.is_file():
        raise AdapterError(f"Terminal-Bench runner missing: {runner}")
    suite = (root / "terminal-core19-hotlist.json").resolve()
    if not suite.is_file():
        raise AdapterError(f"Terminal-Bench suite manifest missing: {suite}")
    endpoint = _required(cfg, "endpoint")
    model = _required(cfg, "model")
    common = ["--suite", str(suite), "--tier", "full", "--endpoint", endpoint, "--model", model]
    doctor = [str(runner), "doctor", *common]
    output = Path(cfg.get("output_dir") or run_dir / TERMINAL / "results").expanduser().resolve()
    run_id = run_dir.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise AdapterError(f"Terminal run ID is not a safe native job name: {run_id!r}")
    job_name = f"contested-{run_id}-terminal"
    if len(job_name) > 220:
        raise AdapterError("Terminal native job name exceeds 220 characters")
    command = [
        str(runner), "run", *common,
        "--platform", _required(cfg, "platform"),
        "--model-name", _required(cfg, "model_name"),
        "--engine", _required(cfg, "engine"),
        "--backend", _required(cfg, "backend"),
        "--attempts", "2", "--concurrency", "1",
        "--results-dir", str(output),
        "--job-name", job_name,
    ]
    for key, flag in (
        ("platform_name", "--platform-name"),
        ("engine_version", "--engine-version"),
        ("backend_version", "--backend-version"),
        ("quant", "--quant"),
        ("inference_profile", "--inference-profile"),
        ("tag", "--tag"),
    ):
        if cfg.get(key):
            command.extend([flag, str(cfg[key])])
    guarded = [sys.executable, "-c", _TERMINAL_RUN_GUARD,
               str(workdir / "jobs" / job_name), *command]
    return [_step("terminal-doctor", doctor, workdir),
            _step("terminal-run", guarded, workdir)]


def build_steps(component: str, cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Return native runner steps without starting model inference."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if component == AIDER:
        return _build_aider_steps(cfg, root, run_dir)
    if component == TERMINAL:
        return _build_terminal_steps(cfg, root, run_dir)
    raise AdapterError(f"Unsupported component: {component}")


def _parse_aider_dir(base: Path, selected: list[str], *, expected_model: str | None = None) -> list[dict[str, Any]]:
    base = base / "full" if (base / "full").is_dir() else base
    if not base.is_dir():
        raise AdapterError(f"Aider results directory missing: {base}")
    rows = []
    for case_id in selected:
        path = base / case_id / ".aider.results.json"
        if not path.is_file():
            raise AdapterError(f"Aider result missing: {path}")
        raw = json.loads(path.read_text())
        if raw.get("testdir") != "/benchmarks/full/" + case_id:
            raise AdapterError(f"Aider testdir mismatch in {path}")
        if raw.get("testcase") != case_id.rsplit("/", 1)[-1]:
            raise AdapterError(f"Aider testcase mismatch in {path}")
        if expected_model and raw.get("model") != expected_model:
            raise AdapterError(f"Aider model mismatch in {path}")
        if raw.get("edit_format") != "whole":
            raise AdapterError(f"Aider edit format mismatch in {path}")
        outcomes = raw.get("tests_outcomes")
        if not isinstance(outcomes, list) or not outcomes or len(outcomes) > 2 or any(type(x) is not bool for x in outcomes):
            raise AdapterError(f"Aider attempts invalid in {path}: {outcomes!r}")
        if not any(outcomes) and len(outcomes) != 2:
            raise AdapterError(f"Aider failed case has no second attempt: {path}")
        if any(type(raw.get(key, 0)) is not int or raw.get(key, 0) != 0 for key in ("test_timeouts", "num_error_outputs", "num_exhausted_context_windows")):
            raise AdapterError(f"Aider native model/harness error in {path}; inspect before scoring")
        rows.append({"suite": AIDER, "case_id": case_id, "passed": any(outcomes)})
    return rows


def _copy_remote_aider(cfg: dict[str, Any], run_dir: Path, selected: list[str]) -> Path:
    host = _host(cfg)
    if not host:
        raise AdapterError("Remote Aider copy requested without host")
    remote_output = _remote_output_dir(cfg, run_dir)
    local = run_dir / AIDER / "native-results" / "full"
    for case_id in selected:
        destination = local / case_id / ".aider.results.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = f"{remote_output}/full/{case_id}/.aider.results.json"
        result = subprocess.run(
            [str(SERVER_SH), host, "copy-from", source, str(destination)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise AdapterError(f"Cannot copy remote Aider result {source}: {result.stderr.strip()}")
    return local.parent


def _collect_aider(cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    selected = _selected(root, AIDER)
    mode = cfg.get("mode", "import")
    if mode == "docker":
        if _host(cfg):
            base = _copy_remote_aider(cfg, run_dir, selected)
        else:
            base = Path(cfg.get("output_dir") or run_dir / AIDER).expanduser().resolve()
        model = _required(cfg, "model")
    elif mode == "import":
        base = Path(_required(cfg, "results_path")).expanduser().resolve()
        model = cfg.get("model")
    else:
        raise AdapterError("Aider mode must be 'docker' or 'import'")
    return _parse_aider_dir(base, selected, expected_model=model)


def _terminal_model_dirs(base: Path) -> list[Path]:
    if (base / "summary.json").is_file():
        return [base]
    return sorted(path.parent for path in base.rglob("summary.json") if path.is_file())


def _validate_terminal_task(raw: dict[str, Any], case_id: str, content_sha256: str, path: Path) -> bool:
    """Validate one native task export and return its pass@2 grade."""
    if raw.get("task") != case_id:
        raise AdapterError(f"Terminal task ID mismatch: {path}")
    provenance = raw.get("task_provenance") or {}
    if provenance.get("content_sha256") != content_sha256:
        raise AdapterError(f"Terminal task-content digest mismatch: {path}")
    attempts = raw.get("attempts")
    if not isinstance(attempts, list) or not (1 <= len(attempts) <= 2):
        raise AdapterError(f"Terminal pass@2 attempts missing or excessive: {path}")
    exception_types = [(item.get("exception") or {}).get("exception_type") for item in attempts]
    if any(kind not in (None, "AgentTimeoutError") for kind in exception_types):
        raise AdapterError(f"Terminal task has a non-scorable agent or harness exception: {path}")
    passed = raw.get("passed")
    if type(passed) is not bool or passed != any(item.get("reward") == 1 for item in attempts):
        raise AdapterError(f"Terminal pass@2 reward mismatch: {path}")
    if not passed and len(attempts) < 2:
        raise AdapterError(f"Terminal failed task has no second attempt: {path}")
    # Harbor marks an agent timeout as completed=False. Once both allowed
    # attempts exist, that is a genuine benchmark failure, not a missing run.
    if not raw.get("completed") and not any(kind == "AgentTimeoutError" for kind in exception_types):
        raise AdapterError(f"Terminal task is incomplete without a scored timeout: {path}")
    return passed


def _parse_terminal_dir(base: Path, selected: list[str], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dirs = _terminal_model_dirs(base)
    if len(dirs) != 1:
        raise AdapterError(f"Expected one Terminal result set below {base}; found {len(dirs)}")
    model_dir = dirs[0]
    summary = json.loads((model_dir / "summary.json").read_text())
    identity = summary.get("suite") or {}
    if identity.get("id") != TERMINAL_SUITE_ID or identity.get("manifest_hash") != TERMINAL_MANIFEST_HASH:
        raise AdapterError(f"Terminal suite identity mismatch in {model_dir / 'summary.json'}")
    if summary.get("total_tasks") != len(selected):
        raise AdapterError("Terminal result summary does not cover exactly four selected tasks")
    files = summary.get("results")
    expected_files = {f"results-{case_id}.json" for case_id in selected}
    if not isinstance(files, list) or set(files) != expected_files or len(files) != len(expected_files):
        raise AdapterError("Terminal summary result file set differs from selected hotlist")
    rows = []
    for case_id in selected:
        path = model_dir / f"results-{case_id}.json"
        if not path.is_file():
            raise AdapterError(f"Terminal task result missing: {path}")
        raw = json.loads(path.read_text())
        suite = raw.get("suite") or {}
        if suite.get("manifest_hash") != TERMINAL_MANIFEST_HASH or suite.get("id") != TERMINAL_SUITE_ID:
            raise AdapterError(f"Terminal task suite identity mismatch: {path}")
        passed = _validate_terminal_task(
            raw, case_id, manifest["tasks"][case_id]["content_sha256"], path
        )
        rows.append({"suite": TERMINAL, "case_id": case_id, "passed": passed})
    if summary.get("passed_tasks") != sum(row["passed"] for row in rows):
        raise AdapterError("Terminal summary pass count differs from per-task results")
    return rows


def _collect_terminal(cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / "terminal-core19-hotlist.json").read_text())
    selected = _selected(root, TERMINAL)
    if sorted(manifest["tiers"]["full"]) != sorted(selected):
        raise AdapterError("Terminal native suite and unified manifest select different tasks")
    mode = cfg.get("mode", "run")
    if mode == "import":
        base = Path(_required(cfg, "results_path")).expanduser().resolve()
    elif mode == "run":
        base = Path(cfg.get("output_dir") or run_dir / TERMINAL / "results").expanduser().resolve()
    else:
        raise AdapterError("Terminal mode must be 'run' or 'import'")
    return _parse_terminal_dir(base, selected, manifest)


def collect(component: str, cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Collect all selected native grades for one component, or raise."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if component == AIDER:
        return _collect_aider(cfg, root, run_dir)
    if component == TERMINAL:
        return _collect_terminal(cfg, root, run_dir)
    raise AdapterError(f"Unsupported component: {component}")
