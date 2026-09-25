"""Native runners and outcome collectors for the two agent/tool components.

The launcher executes on the controller host through the documented server.sh SSH wrapper.
Inference is never performed by this module during import, planning, or collect.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


HERMES = "hermesagent_20"
TOOL_EVAL = "tool_eval_hard15_v2_1_0"
SUPPORTED = {HERMES, TOOL_EVAL}
SERVER_TOOL = "/opt/benchmark-host/server.sh"
HERMES_NATIVE_ROOT = "/opt/benchmark-host/hermes"
HERMES_SOURCE_SHA256 = "01548df005a9c3feec092767a49a4499ddcca0df6cdc81b3201299f9ceb0e8ca"
HERMES_BENCHMARK_COMMIT = "57d7766bf3db8c40696e3ed937d43c8c85f4cd6c"
HERMES_DEFAULT_IMAGE = "hermesagent20-verifier:official-20260924"
HERMES_DEFAULT_IMAGE_ID = "sha256:a8b529864f981b795cebcfeef2881128b392353cbf16265ac802bf1fcec851b8"
TOOL_VERSION = "2.1.0"
TOOL_SOURCE_ROOT = "/opt/benchmark-host/tool-eval/evals"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_HERMES_LINE = "const runDir = path.join(ROOT, `hermes20-${profile}-c${workers}${smoke ? '-smoke' : ''}`);"
_HERMES_REPLACEMENT = (
    "const runDir = process.env.HA20_RUN_DIR;\n"
    "if (!runDir || !path.isAbsolute(runDir)) throw new Error('HA20_RUN_DIR must be absolute');"
)

# Executed on a remote host only when the orchestrator launches the prepare step.
# It patches output placement, not scenario definitions, agent calls, or grading.
_PREPARE_HERMES = r"""
import hashlib
import pathlib
import subprocess
import sys
import json
import urllib.request
source, target_dir, expected_sha, old, new, image, expected_image_id, base_url, expected_served_id = sys.argv[1:10]
original = pathlib.Path(source).read_bytes()
actual_sha = hashlib.sha256(original).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(f'Hermes native runner SHA-256 mismatch: {actual_sha}')
text = original.decode('utf-8')
if text.count(old) != 1:
    raise SystemExit('Hermes run-directory assignment changed')
if expected_image_id:
    actual_image_id = subprocess.check_output(
        ['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True
    ).strip()
    if actual_image_id != expected_image_id:
        raise SystemExit(f'Hermes verifier image ID mismatch: {actual_image_id}')
if expected_served_id:
    with urllib.request.urlopen(base_url.rstrip('/') + '/models', timeout=15) as response:
        models = json.load(response)
    served_id = ((models.get('data') or [{}])[0]).get('id')
    if served_id != expected_served_id:
        raise SystemExit(f'Hermes served model mismatch: expected {expected_served_id!r}, found {served_id!r}')
target = pathlib.Path(target_dir)
if not target.is_absolute() or target.exists():
    raise SystemExit('Hermes run directory must be new and absolute: ' + str(target))
target.parent.mkdir(parents=True, exist_ok=True)
target.mkdir()
(target / 'agent_tool_hermes_runner.mjs').write_text(text.replace(old, new), encoding='utf-8')
""".strip()

# Produces only compact metadata and score rows, never raw tool transcripts.
_READ_HERMES = r"""
import json
import pathlib
import sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
print(json.dumps({'summary': summary, 'manifest': manifest}, separators=(',', ':')))
""".strip()

_PREPARE_TOOL = r"""
import pathlib
import subprocess
import hashlib
import sys
runner, target_dir, source_root = sys.argv[1:]
version = subprocess.check_output([runner, '--version'], text=True).strip()
if version != 'tool-eval-bench 2.1.0':
    raise SystemExit(f'Tool-Eval must be v2.1.0; found {version!r}')
for name, expected in (
    ('scenarios_hardmode.py', '61b9c84a091e4edf34566aded5618966b84825f279da0d33256910b055548e48'),
    ('scenarios_hardmode_expanded.py', '2909271fbe3d927c6715dda59450098d1ee604bb53e4379060f58e59e1b74b45'),
):
    actual = hashlib.sha256((pathlib.Path(source_root) / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Tool-Eval {name} fixture SHA-256 mismatch: {actual}')
target = pathlib.Path(target_dir)
if not target.is_absolute() or target.exists():
    raise SystemExit('Tool-Eval run directory must be new and absolute: ' + str(target))
target.parent.mkdir(parents=True, exist_ok=True)
target.mkdir()
""".strip()

_MARK_TOOL_STARTED = r"""
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
with path.open('x') as stream:
    stream.write('Native Tool-Eval attempt started; use a new suite run ID after interruption.\n')
""".strip()

_READ_TOOL = r"""
import json
import pathlib
import sys
result = json.loads(pathlib.Path(sys.argv[1]).read_text())
compact = {k: result.get(k) for k in ('status', 'tool_eval_bench_version', 'total_scenarios', 'config')}
compact['scenario_results'] = [
    {k: row.get(k) for k in ('scenario_id', 'status', 'points')}
    for row in result['scores']['scenario_results']
]
print(json.dumps(compact, separators=(',', ':')))
""".strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _component(component: str) -> None:
    _require(component in SUPPORTED, f"Unsupported agent/tool component: {component}")


def _safe_run_name(run_dir: Path) -> str:
    name = run_dir.name
    _require(bool(_SAFE_NAME.fullmatch(name)), "run_dir name must be a simple unique slug")
    return name


def _remote_dir(component: str, cfg: dict[str, Any], run_dir: Path) -> str:
    default_root = (
        "/opt/benchmark-host/runs/hermes"
        if component == HERMES
        else "/opt/benchmark-host/runs/tool-eval"
    )
    remote_root = PurePosixPath(str(cfg.get("remote_root", default_root)))
    _require(remote_root.is_absolute(), "remote_root must be an absolute POSIX path")
    _require(".." not in remote_root.parts, "remote_root cannot contain '..'")
    _require(not remote_root.is_relative_to(PurePosixPath(HERMES_NATIVE_ROOT)),
             "remote_root cannot be inside the historical Hermes source root")
    return str(remote_root / _safe_run_name(run_dir) / ("hermes" if component == HERMES else "tool_eval"))


def _selected(root: Path, component: str) -> list[str]:
    filename = (
        "hermesagent-scenarios.txt" if component == HERMES else "tool-eval-hard15-v2.1.0.txt"
    )
    path = root / "selectors" / filename
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pattern = r"HA-\d{2}" if component == HERMES else r"TC-\d{2}"
    _require(bool(ids) and len(ids) == len(set(ids)), f"Invalid or duplicate IDs in {path}")
    _require(all(re.fullmatch(pattern, value) for value in ids), f"Unexpected IDs in {path}")
    expected_count = 11 if component == HERMES else 4
    _require(len(ids) == expected_count, f"Expected {expected_count} selected IDs in {path}")
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    selected = {case["case_id"] for case in suite["cases"] if case["suite"] == component}
    _require(selected == set(ids), f"{path} differs from suite.json")
    return ids


def _remote_step(name: str, server_tool: str, host: str, command: str) -> dict[str, Any]:
    return {"name": name, "argv": [server_tool, host, "run", command], "cwd": None, "env": None}


def _nonempty(cfg: dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    _require(isinstance(value, str) and bool(value.strip()), f"Missing nonempty {key}")
    return value


def _mode(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("mode", "run"))
    _require(mode in {"run", "import", "collect"}, "mode must be run, import, or collect")
    return mode


def _endpoint(cfg: dict[str, Any], *, require_v1: bool) -> str:
    url = _nonempty(cfg, "base_url")
    parsed = urlsplit(url)
    _require(parsed.scheme in {"http", "https"} and bool(parsed.hostname),
             "base_url must be an HTTP(S) endpoint")
    _require(parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment,
             "base_url must not contain credentials, query, or fragment")
    if require_v1:
        _require(parsed.path.rstrip("/").endswith("/v1"),
                 "Hermes base_url must be an OpenAI-compatible /v1 URL")
    return url.rstrip("/")


def build_steps(component: str, cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Return shell-free local argv steps; remote shell arguments are quoted.

    The prepare step insists on a fresh run directory, protecting archived results.
    The caller executes these steps serially, then calls :func:`collect`.
    """
    _component(component)
    _selected(root, component)
    if component == HERMES:
        _nonempty(cfg, "served_model_id")
    if _mode(cfg) in {"import", "collect"}:
        return []
    remote_dir = _remote_dir(component, cfg, run_dir)
    host = str(cfg.get("host", "benchmark-host-a" if component == HERMES else "benchmark-host-b"))
    server_tool = str(cfg.get("server_tool", SERVER_TOOL))
    _require(host in {"benchmark-host-a", "benchmark-host-b", "ciru", "benchmark-host-c"}, f"Unsupported server.sh host: {host}")

    if component == HERMES:
        base_url = _endpoint(cfg, require_v1=True)
        served_model_id = _nonempty(cfg, "served_model_id")
        profile = str(cfg.get("profile", "contested"))
        _require(bool(_SAFE_NAME.fullmatch(profile)), "Hermes profile must be a simple slug")
        workers = int(cfg.get("workers", 1))
        _require(workers in {1, 2, 4, 8}, "Hermes workers must be 1, 2, 4, or 8")
        native_root = str(cfg.get("native_root", HERMES_NATIVE_ROOT))
        _require(native_root == HERMES_NATIVE_ROOT,
                 "Pinned Hermes runner requires the original native_root path")
        source = str(PurePosixPath(native_root) / "run_hermes20.mjs")
        node = str(cfg.get("node", "node"))
        python = str(cfg.get("python", "python3"))
        image = str(cfg.get("verifier_image", HERMES_DEFAULT_IMAGE))
        image_id = str(cfg.get("verifier_image_id", HERMES_DEFAULT_IMAGE_ID if image == HERMES_DEFAULT_IMAGE else ""))
        _require(bool(image_id), "A custom Hermes verifier_image requires verifier_image_id")
        prepare = shlex.join([python, "-c", _PREPARE_HERMES, source, remote_dir,
                              HERMES_SOURCE_SHA256, _HERMES_LINE, _HERMES_REPLACEMENT,
                              image, image_id, base_url, served_model_id])
        run_args = shlex.join([
            f"HA20_RUN_DIR={remote_dir}", f"HA20_MODEL_BASE_URL={base_url}",
            f"HA20_VERIFIER_IMAGE={image}", "HA20_SMOKE=0",
            node, str(PurePosixPath(remote_dir) / "agent_tool_hermes_runner.mjs"), profile, str(workers),
        ])
        progress_path = str(PurePosixPath(remote_dir) / "progress.json")
        run = (
            f"if test -f {shlex.quote(progress_path)}; then ha20_resume=1; "
            "else ha20_resume=0; fi; "
            f'env HA20_RESUME="$ha20_resume" {run_args}'
        )
        return [
            _remote_step("hermes_prepare", server_tool, host, prepare),
            _remote_step("hermes_run20", server_tool, host, run),
        ]

    base_url = _endpoint(cfg, require_v1=False)
    model = _nonempty(cfg, "model")
    backend = _nonempty(cfg, "backend")
    runner = str(cfg.get("runner", "/opt/benchmark-host/bin/tool-eval-bench"))
    python = str(cfg.get("python", str(PurePosixPath(runner).parent / "python")))
    source_root = str(cfg.get("source_root", TOOL_SOURCE_ROOT))
    ids = _selected(root, component)
    _require(backend in {"vllm", "litellm", "llamacpp"}, "Tool-Eval backend must be vllm, litellm, or llamacpp")
    backend_kwargs = cfg.get("backend_kwargs", {"reasoning_effort": "xhigh"})
    _require(isinstance(backend_kwargs, dict), "backend_kwargs must be a JSON object")
    _require(not any(word in str(key).upper() for key in backend_kwargs
                     for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")),
             "backend_kwargs must not contain credentials in the stored command plan")
    result_path = str(PurePosixPath(remote_dir) / "result.json")
    reports_path = str(PurePosixPath(remote_dir) / "reports")
    args = [
        runner, "--base-url", base_url, "--backend", backend, "--model", model,
        "--temperature", str(cfg.get("temperature", 1.0)),
        "--top-p", str(cfg.get("top_p", 0.95)),
        "--backend-kwargs", json.dumps(backend_kwargs, separators=(",", ":"), sort_keys=True),
        "--seed", str(cfg.get("seed", 1)),
        "--parallel", str(cfg.get("parallel", 1)),
        "--max-turns", str(cfg.get("max_turns", 8)),
        "--timeout", str(cfg.get("timeout", 3600)),
        "--no-warmup", "--no-probe-engine", "--no-live", "--hardmode-only",
        "--scenarios", *ids,
        "--json-file", result_path, "--output-dir", reports_path,
    ]
    # This mirrors the completed xhigh hard-15 protocol while running four tasks.
    prepare = shlex.join([python, "-c", _PREPARE_TOOL, runner, remote_dir, source_root])
    marker = str(PurePosixPath(remote_dir) / "attempt-started.txt")
    mark = shlex.join([python, "-c", _MARK_TOOL_STARTED, marker])
    return [
        _remote_step("tool_eval_prepare", server_tool, host, prepare),
        _remote_step("tool_eval_run4", server_tool, host, mark + " && " + shlex.join(args)),
    ]


def parse_hermes(summary: dict[str, Any], manifest: dict[str, Any], selected: list[str],
                 cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Validate a complete native 20-case run and retain selected strict grades."""
    cfg = cfg or {}
    _require(summary.get("status") == "completed", "Hermes run is not completed")
    _require(manifest.get("benchmarkCommit") == HERMES_BENCHMARK_COMMIT, "Hermes benchmark commit differs")
    _require(manifest.get("smoke") is False, "Hermes smoke run is not a full benchmark")
    all_ids = {f"HA-{i:02d}" for i in range(1, 21)}
    expected_ids = set(selected) if cfg.get("selected_only") else all_ids
    _require(set(manifest.get("scenarioIds", [])) == expected_ids,
             "Hermes manifest scenario IDs differ from the requested set")
    rows = summary.get("scenarios")
    _require(isinstance(rows, list) and len(rows) == len(expected_ids),
             "Hermes summary needs " + str(len(expected_ids)) + " scenarios")
    indexed = {row.get("scenarioId"): row for row in rows}
    _require(len(indexed) == len(expected_ids) and set(indexed) == expected_ids,
             "Hermes scenario IDs are missing or duplicate")
    if "base_url" in cfg:
        _require(manifest.get("model", {}).get("inferenceBaseUrl") == cfg["base_url"].rstrip("/"),
                 "Hermes endpoint differs from requested base_url")
    if "served_model_id" in cfg:
        _require(manifest.get("model", {}).get("providerModel") == cfg["served_model_id"] and
                 manifest.get("model", {}).get("exposedModel") == cfg["served_model_id"],
                 "Hermes manifest served model differs from configured served_model_id")
    if "verifier_image" in cfg:
        _require(manifest.get("image") == cfg["verifier_image"], "Hermes verifier image differs")
    if "profile" in cfg:
        _require(manifest.get("profile") == cfg["profile"], "Hermes profile differs")
    if "workers" in cfg:
        _require(manifest.get("workers") == int(cfg["workers"]), "Hermes workers differ")
    outcomes = []
    for case_id in selected:
        row = indexed[case_id]
        status, score = row.get("status"), row.get("score")
        _require(status in {"pass", "partial", "fail"}, f"Hermes {case_id} has invalid status")
        _require(isinstance(score, (int, float)) and not isinstance(score, bool),
                 f"Hermes {case_id} has invalid score")
        _require((status == "pass") == (score == 100), f"Hermes {case_id} pass/score disagree")
        outcomes.append({"suite": HERMES, "case_id": case_id, "passed": status == "pass"})
    return outcomes


def parse_tool_eval(result: dict[str, Any], selected: list[str],
                    cfg: dict[str, Any] | None = None,
                    allow_full: bool = False) -> list[dict[str, Any]]:
    """Validate a native v2.1.0 hard-mode result and retain strict passes."""
    cfg = cfg or {}
    _require(result.get("status") == "completed", "Tool-Eval run is not completed")
    _require(result.get("tool_eval_bench_version") == TOOL_VERSION, "Tool-Eval must be v2.1.0")
    rows = result.get("scenario_results")
    _require(isinstance(rows, list), "Tool-Eval scenario_results is missing")
    indexed = {row.get("scenario_id"): row for row in rows}
    permitted = [set(selected)]
    if allow_full:
        permitted.append({f"TC-{i:02d}" for i in range(70, 85)})
    _require(len(rows) == len(indexed) and set(indexed) in permitted,
             "Tool-Eval result must contain selected four cases or the complete hard-15")
    _require(result.get("total_scenarios") == len(rows), "Tool-Eval total_scenarios differs")
    native_cfg = result.get("config") or {}
    _require(set(native_cfg.get("scenario_ids", [])) == set(indexed), "Tool-Eval config scenario IDs differ")
    if "model" in cfg:
        _require(native_cfg.get("model") == cfg["model"], "Tool-Eval model differs")
    if "backend" in cfg:
        _require(native_cfg.get("backend") == cfg["backend"], "Tool-Eval backend differs")
    outcomes = []
    expected_points = {"pass": 2, "partial": 1, "fail": 0}
    for case_id in selected:
        row = indexed[case_id]
        status = row.get("status")
        _require(status in expected_points, f"Tool-Eval {case_id} has invalid status")
        _require(row.get("points") == expected_points[status],
                 f"Tool-Eval {case_id} native status and points disagree")
        outcomes.append({"suite": TOOL_EVAL, "case_id": case_id, "passed": status == "pass"})
    return outcomes


def _read_remote(server_tool: str, host: str, command: str) -> dict[str, Any]:
    completed = subprocess.run(
        [server_tool, host, "run", command], capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Could not read {host} native result: {completed.stderr.strip()[:1000]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{host} native result is not valid JSON") from exc


def collect(component: str, cfg: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Read native grader output from the benchmark host without modifying it."""
    _component(component)
    selected = _selected(root, component)
    if component == HERMES:
        _nonempty(cfg, "served_model_id")
    mode = _mode(cfg)
    remote_dir = _remote_dir(component, cfg, run_dir) if mode == "run" else None
    host = str(cfg.get("host", "benchmark-host-a" if component == HERMES else "benchmark-host-b"))
    server_tool = str(cfg.get("server_tool", SERVER_TOOL))
    if component == HERMES:
        summary_path = str(PurePosixPath(remote_dir) / "summary.json") if mode == "run" else _nonempty(cfg, "summary_path")
        manifest_path = str(PurePosixPath(remote_dir) / "manifest.json") if mode == "run" else _nonempty(cfg, "manifest_path")
        if host == "local":
            _require(mode != "run", "local host supports import/collect mode only")
            payload = {"summary": json.loads(Path(summary_path).read_text(encoding="utf-8")),
                       "manifest": json.loads(Path(manifest_path).read_text(encoding="utf-8"))}
            return parse_hermes(payload["summary"], payload["manifest"], selected, cfg)
        python = str(cfg.get("python", "python3"))
        payload = _read_remote(server_tool, host,
                               shlex.join([python, "-c", _READ_HERMES, summary_path, manifest_path]))
        return parse_hermes(payload["summary"], payload["manifest"], selected, cfg)
    runner = str(cfg.get("runner", "/opt/benchmark-host/bin/tool-eval-bench"))
    python = str(cfg.get("python", str(PurePosixPath(runner).parent / "python")))
    result_path = str(PurePosixPath(remote_dir) / "result.json") if mode == "run" else _nonempty(cfg, "result_path")
    if host == "local":
        _require(mode != "run", "local host supports import/collect mode only")
        source = json.loads(Path(result_path).read_text(encoding="utf-8"))
        payload = {k: source.get(k) for k in ("status", "tool_eval_bench_version", "total_scenarios", "config")}
        payload["scenario_results"] = [
            {k: row.get(k) for k in ("scenario_id", "status", "points")}
            for row in source["scores"]["scenario_results"]
        ]
        return parse_tool_eval(payload, selected, cfg, allow_full=True)
    payload = _read_remote(server_tool, host, shlex.join([python, "-c", _READ_TOOL, result_path]))
    return parse_tool_eval(payload, selected, cfg, allow_full=mode in {"import", "collect"})
