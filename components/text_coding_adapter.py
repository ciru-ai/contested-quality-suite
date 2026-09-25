"""Runnable text/coding adapters for the contested-quality suite.

The public entry points are build_steps(component, cfg, root, run_dir) and
collect(component, cfg, root, run_dir).  Native scorers make every pass/fail
decision; this module only validates identities and normalizes their output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ARC = "arc_challenge_1172"
IFEVAL = "ifeval_541_strict"
HUMANEVAL = "humaneval_plus_164"
SUPPORTED = frozenset({ARC, IFEVAL, HUMANEVAL})

DEFAULT_IFEVAL_DIR = Path("/opt/benchmark-data/ifeval")
DEFAULT_IFEVAL_PYTHON = Path("/opt/benchmark-data/ifeval-venv/bin/python")
DEFAULT_EVALSCOPE_PYTHON = Path("/opt/benchmark-data/evalscope-venv/bin/python")
DEFAULT_EVALPLUS_PYTHON = Path(
    "/opt/benchmark-data/"
    "scoring-venv/bin/python"
)
DEFAULT_HE_SOURCE = Path(
    "/opt/benchmark-data/"
    "sources/HumanEvalPlus-v0.1.10.jsonl"
)
HE_SOURCE_MD5 = "fe585eb4df8c88d844eeb463ea4d0302"


def _component(component: str) -> None:
    if component not in SUPPORTED:
        raise ValueError(f"unsupported text/coding component: {component}")


def _model(cfg: dict[str, Any]) -> str:
    value = cfg.get("model")
    if not isinstance(value, str) or not value:
        raise ValueError("config must provide nonempty 'model'")
    return value


def _url(cfg: dict[str, Any]) -> str:
    value = cfg.get("api_base_url")
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise ValueError("config must provide 'api_base_url' beginning http:// or https://")
    return value.rstrip("/")


def _selector(root: Path, filename: str) -> Path:
    path = root / "selectors" / filename
    if not path.is_file():
        raise FileNotFoundError(f"required selector missing: {path}")
    return path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(item)
    return rows


def _expected(root: Path, component: str) -> set[str]:
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    return {case["case_id"] for case in suite["cases"] if case["suite"] == component}


def _complete(rows: list[dict[str, Any]], expected: set[str], component: str) -> list[dict[str, Any]]:
    observed = [row["case_id"] for row in rows]
    duplicates = sorted({key for key in observed if observed.count(key) > 1})
    if duplicates:
        raise ValueError(f"{component}: duplicate graded case IDs: {duplicates}")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"{component}: incomplete/mismatched grades; missing={missing}, extra={extra}")
    if any(type(row["passed"]) is not bool for row in rows):
        raise ValueError(f"{component}: native grade must be a boolean")
    return sorted(rows, key=lambda row: row["case_id"])


def _step(name: str, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict:
    return {"name": name, "argv": argv, "cwd": str(cwd) if cwd else None, "env": env}


def _python(cfg: dict, field: str, default: Path) -> str:
    value = str(cfg.get(field, default))
    if not Path(value).is_file():
        raise FileNotFoundError(f"{field} executable missing: {value}")
    return value


def _source_path(cfg: dict, key: str, default: Path, run_dir: Path) -> Path:
    value = Path(cfg.get(key, default))
    return (value if value.is_absolute() else run_dir / value).resolve()


def build_steps(component: str, cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    """Return sequential native run steps without executing inference.

    `cfg` has common `model` and `api_base_url` plus component-specific fields
    documented in text_coding_SETUP.md.  A caller executes the returned argv
    with the listed cwd and inherited environment; env, if non-None, adds to it.
    """
    _component(component)
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    mode = cfg.get("mode", "run")
    if mode == "import":
        return []
    if mode != "run":
        raise ValueError(f"{component}: mode must be 'run' or 'import'")
    model, api_base = _model(cfg), _url(cfg)
    key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(key_env, str) or not key_env:
        raise ValueError("api_key_env must be a nonempty environment-variable name")

    if component == ARC:
        py = _python(cfg, "evalscope_python", DEFAULT_EVALSCOPE_PYTHON)
        generation = cfg.get("arc_generation", {
            "temperature": 0.0,
            "max_tokens": 256,
            "seed": 42,
            "top_p": 1.0,
            "stream": False,
            "retries": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        })
        if not isinstance(generation, dict):
            raise ValueError("arc_generation must be a JSON object")
        return [_step("arc-native-evalscope", [
            py, str(root / "components" / "text_coding_arc_run.py"),
            "--selector", str(_selector(root, "arc-challenge.jsonl")),
            "--work-dir", str(run_dir / "arc"),
            "--api-url", api_base,
            "--model", model,
            "--generation-json", json.dumps(generation, separators=(",", ":")),
            "--api-key-env", key_env,
        ])]

    if component == IFEVAL:
        py = _python(cfg, "ifeval_python", DEFAULT_IFEVAL_PYTHON)
        eval_dir = Path(cfg.get("ifeval_eval_dir", DEFAULT_IFEVAL_DIR)).resolve()
        generator = root / "components" / "text_coding_ifeval_generate.py"
        grader = eval_dir / "instruction_following_eval" / "evaluation_main.py"
        if not generator.is_file() or not grader.is_file():
            raise FileNotFoundError(f"IFEval generation driver or official strict grader missing: {generator}, {grader}")
        if cfg.get("ifeval_workers", 1) != 1:
            raise ValueError("IFEval hotlist uses one first response per prompt; ifeval_workers must be 1")
        inp = _selector(root, "ifeval-input-data.jsonl")
        out = run_dir / "ifeval"
        generation = [
            py, str(generator), "--input", str(inp), "--output", str(out / "responses.jsonl"),
            "--url", api_base + "/chat/completions", "--model", model,
            "--max-tokens", str(cfg.get("ifeval_max_tokens", 2048)),
            "--temperature", str(cfg.get("ifeval_temperature", 0.0)),
            "--timeout", str(cfg.get("ifeval_timeout", 300)),
            "--api-key-env", key_env,
        ]
        for key in ("top_p", "top_k", "min_p"):
            field = "ifeval_" + key
            if field in cfg:
                generation.extend(["--" + key.replace("_", "-"), str(cfg[field])])
        if cfg.get("ifeval_thinking", False):
            generation.append("--thinking")
        return [
            _step("ifeval-generate", generation, cwd=eval_dir),
            _step("ifeval-strict-native-grade", [
                py, "-m", "instruction_following_eval.evaluation_main", f"--input_data={inp}",
                f"--input_response_data={out / 'responses.jsonl'}",
                f"--output_dir={out}",
            ], cwd=eval_dir),
        ]

    py = _python(cfg, "evalplus_python", DEFAULT_EVALPLUS_PYTHON)
    source = Path(cfg.get("evalplus_dataset", DEFAULT_HE_SOURCE)).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"HumanEval+ source dataset missing: {source}")
    actual_md5 = hashlib.md5(source.read_bytes()).hexdigest()
    if actual_md5 != HE_SOURCE_MD5:
        raise ValueError(f"HumanEval+ source MD5 mismatch: {actual_md5}")
    selector = _selector(root, "humaneval-plus-native.jsonl")
    output = run_dir / "humaneval"
    sample_file = output / "samples.jsonl"
    generation = cfg.get("humaneval_generation", {
        "temperature": 0.0, "max_tokens": 1024, "n": 1, "seed": 0,
        "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False, "cache_prompt": False,
    })
    if not isinstance(generation, dict):
        raise ValueError("humaneval_generation must be a JSON object")
    generator_step = _step("humaneval-generate", [
        py, str(root / "components" / "text_coding_humaneval_generate.py"),
        "--selector", str(selector), "--dataset", str(source),
        "--samples", str(sample_file), "--api-url", api_base + "/chat/completions",
        "--model", model, "--generation-json", json.dumps(generation, separators=(",", ":")),
        "--api-key-env", key_env,
    ])
    grader_argv = _evalplus_sandbox_argv(cfg, root, output, py, source, selector)
    return [generator_step, _step("humaneval-plus-native-grade", grader_argv)]


def _evalplus_sandbox_argv(cfg: dict, root: Path, output: Path, py: str, source: Path, selector: Path) -> list[str]:
    """Run EvalPlus untrusted code inside a closed Bubblewrap namespace."""
    bwrap = str(cfg.get("bubblewrap", "/usr/bin/bwrap"))
    if not Path(bwrap).is_file():
        raise FileNotFoundError(f"Bubblewrap executable missing: {bwrap}")
    runtime_roots = cfg.get("evalplus_runtime_roots")
    if runtime_roots is None:
        real_py = Path(py).resolve()
        runtime_roots = [str(real_py.parents[2])]
        if Path("/nix/store").is_dir():
            runtime_roots.append("/nix/store")
    if not isinstance(runtime_roots, list) or not all(isinstance(x, str) for x in runtime_roots):
        raise ValueError("evalplus_runtime_roots must be a list of absolute paths")
    ro_roots = [Path("/usr"), Path("/lib"), Path("/lib64"), Path(py).parent.parent]
    ro_roots.extend(Path(x).resolve() for x in runtime_roots)
    argv = [bwrap, "--unshare-all", "--die-with-parent", "--new-session"]
    seen: set[str] = set()
    for path in ro_roots:
        if path.exists() and str(path) not in seen:
            argv.extend(["--ro-bind", str(path), str(path)])
            seen.add(str(path))
    argv.extend([
        "--dir", "/work", "--bind", str(output), "/work",
        "--dir", "/app",
        "--ro-bind", str(root / "components" / "text_coding_humaneval_grade.py"), "/app/grade.py",
        "--ro-bind", str(selector), "/app/selected.jsonl",
        "--tmpfs", "/tmp", "--dir", "/tmp/cache", "--dir", "/tmp/cache/evalplus",
        "--ro-bind", str(source), "/tmp/cache/evalplus/HumanEvalPlus-v0.1.10.jsonl",
        "--proc", "/proc", "--dev", "/dev", "--chdir", "/work", "--clearenv",
        "--setenv", "HOME", "/tmp", "--setenv", "XDG_CACHE_HOME", "/tmp/cache",
        "--setenv", "PATH", str(Path(py).parent),
        "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
        py, "/app/grade.py", "--samples", "/work/samples.jsonl",
        "--selector", "/app/selected.jsonl",
        "--output", "/work/native_scores.json",
    ])
    return argv


def _arc_selector(root: Path) -> dict[str, dict]:
    rows = _jsonl(_selector(root, "arc-challenge.jsonl"))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ARC selector has duplicate IDs")
    return {row["id"]: row for row in rows}


def _arc_prompt(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("ARC native review lacks messages")
    user = [msg.get("content") for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]
    if len(user) != 1 or not isinstance(user[0], str):
        raise ValueError("ARC native review must contain exactly one user prompt")
    return user[0]


def _collect_arc(cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    selector = _arc_selector(root)
    if "arc_review_path" in cfg:
        files = [_source_path(cfg, "arc_review_path", Path(""), run_dir)]
    else:
        files = list((run_dir / "arc" / "reviews").rglob("arc_ARC-Challenge.jsonl"))
    if len(files) != 1:
        raise ValueError(f"ARC expected one native review JSONL, found {len(files)}: {files}")
    rows = []
    for native in _jsonl(files[0]):
        metadata = native.get("sample_score", {}).get("sample_metadata", {})
        raw_id = metadata.get("id")
        if raw_id not in selector:
            continue  # import may provide all 1,172 native ARC reviews
        selected = selector[raw_id]
        target = native.get("target")
        if isinstance(target, list) and len(target) == 1:
            target = target[0]
        if _arc_prompt(native) != selected["prompt"] or target != selected["target"]:
            raise ValueError(f"ARC prompt/target identity mismatch for {raw_id}")
        grade = native["sample_score"]["score"]
        accuracy = grade.get("value", {}).get("accuracy")
        if grade.get("status") != "success" or type(accuracy) not in (int, float) or accuracy not in (0, 1):
            raise ValueError(f"ARC invalid native score for {raw_id}: {grade}")
        rows.append({"suite": ARC, "case_id": f"arc:ARC-Challenge:{raw_id}", "passed": accuracy == 1})
    return _complete(rows, _expected(root, ARC), ARC)


def _collect_ifeval(cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    selector = _jsonl(_selector(root, "ifeval-input-data.jsonl"))
    by_prompt = {row["prompt"]: row for row in selector}
    if len(by_prompt) != len(selector):
        raise ValueError("IFEval selector has duplicate prompts")
    base = run_dir / "ifeval"
    responses_path = _source_path(cfg, "ifeval_responses_path", base / "responses.jsonl", run_dir)
    strict_path = _source_path(cfg, "ifeval_strict_path", base / "eval_results_strict.jsonl", run_dir)
    responses = [row for row in _jsonl(responses_path) if row.get("prompt") in by_prompt]
    if len(responses) != len(selector) or {row.get("prompt") for row in responses} != set(by_prompt):
        raise ValueError("IFEval incomplete or mismatched response set")
    if any(not isinstance(row.get("response"), str) or not row["response"].strip() for row in responses):
        raise ValueError("IFEval empty response: native generator may have exhausted retries")
    for response in responses:
        if response.get("key") != by_prompt[response["prompt"]]["key"]:
            raise ValueError("IFEval response key/prompt mismatch")
    graded = _jsonl(strict_path)
    response_by_prompt = {row["prompt"]: row["response"] for row in responses}
    rows = []
    for native in graded:
        prompt = native.get("prompt")
        if prompt not in by_prompt:
            continue  # import may provide all 541 official graded rows
        selected = by_prompt[prompt]
        flags = native.get("follow_instruction_list")
        passed = native.get("follow_all_instructions")
        if native.get("instruction_id_list") != selected["instruction_id_list"]:
            raise ValueError(f"IFEval instruction identity mismatch for {selected['key']}")
        if not isinstance(flags, list) or len(flags) != len(selected["instruction_id_list"]):
            raise ValueError(f"IFEval invalid per-instruction score for {selected['key']}")
        if not all(type(flag) is bool for flag in flags) or type(passed) is not bool or passed != all(flags):
            raise ValueError(f"IFEval inconsistent strict score for {selected['key']}")
        if native.get("response") != response_by_prompt[prompt]:
            raise ValueError(f"IFEval grade/response mismatch for {selected['key']}")
        rows.append({"suite": IFEVAL, "case_id": str(selected["key"]), "passed": passed})
    return _complete(rows, _expected(root, IFEVAL), IFEVAL)


def _collect_humaneval(cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    selected = {row["task_id"]: row for row in _jsonl(_selector(root, "humaneval-plus-native.jsonl"))}
    base = run_dir / "humaneval"
    if cfg.get("mode", "run") != "import":
        sample_rows = _jsonl(base / "samples.jsonl")
        if len(sample_rows) != len(selected) or {row.get("task_id") for row in sample_rows} != set(selected):
            raise ValueError("HumanEval+ generated sample set is incomplete")
    source = Path(cfg.get("evalplus_dataset", DEFAULT_HE_SOURCE))
    if not source.is_file() or hashlib.md5(source.read_bytes()).hexdigest() != HE_SOURCE_MD5:
        raise ValueError("HumanEval+ pinned v0.1.10 source dataset missing or MD5 mismatch")
    native_path = _source_path(cfg, "humaneval_results_path", base / "native_scores.json", run_dir)
    if not native_path.is_file():
        raise FileNotFoundError(native_path)
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if isinstance(native, dict) and "results" in native:
        if native.get("dataset_md5") != HE_SOURCE_MD5:
            raise ValueError("HumanEval+ native grader used different source dataset")
        results = native["results"]
    elif isinstance(native, dict) and "eval" in native:
        if native.get("hash") not in (None, HE_SOURCE_MD5):
            raise ValueError("HumanEval+ EvalPlus result source hash mismatch")
        results = []
        for task_id, entries in native["eval"].items():
            if task_id not in selected:
                continue
            if not isinstance(entries, list) or len(entries) != 1:
                raise ValueError(f"HumanEval+ needs exactly one first sample for {task_id}")
            results.append(entries[0])
    elif isinstance(native, list):  # IU4 native check_correctness score array
        results = [row for row in native if row.get("task_id") in selected]
    else:
        raise ValueError("HumanEval+ unrecognized native result format")
    if not isinstance(results, list):
        raise ValueError("HumanEval+ native scores missing results list")
    rows = []
    for result in results:
        task_id = result.get("task_id")
        if task_id not in selected:
            raise ValueError(f"HumanEval+ native grade has unknown ID: {task_id}")
        base_status = result.get("base_status")
        plus_status = result.get("plus_status")
        if base_status is None and isinstance(result.get("base"), list):
            base_status = result["base"][0]
        if plus_status is None and isinstance(result.get("plus"), list):
            plus_status = result["plus"][0]
        if base_status not in ("pass", "fail", "timeout") or plus_status not in ("pass", "fail", "timeout"):
            raise ValueError(f"HumanEval+ native grader status invalid for {task_id}")
        rows.append({"suite": HUMANEVAL, "case_id": task_id,
                     "passed": base_status == plus_status == "pass"})
    return _complete(rows, _expected(root, HUMANEVAL), HUMANEVAL)


def collect(component: str, cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    """Validate a complete native result and return selected standard rows."""
    _component(component)
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if component == ARC:
        return _collect_arc(cfg, root, run_dir)
    if component == IFEVAL:
        return _collect_ifeval(cfg, root, run_dir)
    return _collect_humaneval(cfg, root, run_dir)
