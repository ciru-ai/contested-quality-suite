"""Local, distributable runner for all seven hotlist components.

Every task asset is resolved below the extracted suite directory. The two
containerized agent benchmarks use images built from bundled source during
setup. Only an OpenAI-compatible model endpoint is external.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from . import aider_terminal_adapter as at
from . import agent_tool_adapter as ag
from . import text_coding_adapter as tc


TEXT = {tc.ARC, tc.IFEVAL, tc.HUMANEVAL}
TERMINAL = at.TERMINAL
AIDER = at.AIDER
HERMES = ag.HERMES
TOOL = ag.TOOL_EVAL


def _step(name: str, argv: list[str], *, cwd: Path | None = None,
          env: dict[str, str] | None = None) -> dict:
    return {"name": name, "argv": argv, "cwd": str(cwd) if cwd else None, "env": env or {}}


def _hermes_dir(run_dir: Path) -> Path:
    return run_dir / "hermesagent_20" / "native"


def _hermes_steps(cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    if cfg.get("mode") == "import":
        return []
    vendor = root / "bundle" / "vendor"
    native = _hermes_dir(run_dir)
    python = str(cfg.get("python", sys.executable))
    image = str(cfg.get("verifier_image", "contested-hermes-verifier:source-v1"))
    endpoint = str(cfg["base_url"])
    model = str(cfg["served_model_id"])
    prepare = [python, str(root / "components" / "portable_hermes.py"),
               "prepare", "--source", str(vendor / "run_hermes20.mjs"),
               "--benchmark", str(vendor / "hermesagent20-benchmark"),
               "--output", str(native), "--endpoint", endpoint,
               "--model", model, "--image", image,
               "--ids", ",".join(ag._selected(root, HERMES))]
    run = [python, str(root / "components" / "portable_hermes.py"),
           "launch", "--output", str(native), "--endpoint", endpoint,
           "--image", image, "--profile", str(cfg.get("profile", "contested")),
           "--workers", str(cfg.get("workers", 1)),
           "--node", str(cfg.get("node", "node"))]
    return [_step("hermes-prepare", prepare),
            _step("hermes-run20", run, cwd=vendor / "hermesagent20-benchmark")]


def _tool_dir(run_dir: Path) -> Path:
    return run_dir / "tool_eval_hard15_v2_1_0"


def _tool_steps(cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    if cfg.get("mode") == "import":
        return []
    runner = str(cfg.get("runner", root / ".venvs" / "tool-eval" / "bin" / "tool-eval-bench"))
    output = _tool_dir(run_dir)
    ids = ag._selected(root, TOOL)
    prepare = [sys.executable, str(root / "components" / "portable_tool.py"),
               "prepare", "--runner", runner, "--output", str(output)]
    backend_kwargs = {**cfg.get("backend_kwargs", {}),
                      "max_tokens": int(cfg.get("max_output_tokens", 8192))}
    if backend_kwargs["max_tokens"] < 1:
        raise ValueError("Tool-Eval max_output_tokens must be positive")
    args = [runner, "--base-url", str(cfg["base_url"]),
            "--backend", str(cfg.get("backend", "vllm")),
            "--model", str(cfg["model"]),
            "--temperature", str(cfg.get("temperature", 1.0)),
            "--top-p", str(cfg.get("top_p", 0.95)),
            "--backend-kwargs", json.dumps(backend_kwargs, sort_keys=True),
            "--seed", str(cfg.get("seed", 1)),
            "--parallel", str(cfg.get("parallel", 1)),
            "--max-turns", str(cfg.get("max_turns", 8)),
            "--timeout", str(cfg.get("timeout", 3600)),
            "--no-warmup", "--no-probe-engine", "--no-live", "--hardmode-only",
            "--scenarios", *ids,
            "--json-file", str(output / "result.json"),
            "--output-dir", str(output / "reports")]
    guarded = [sys.executable, str(root / "components" / "portable_tool.py"),
               "launch", "--output", str(output), "--", *args]
    return [_step("tool-prepare", prepare), _step("tool-run4", guarded)]


def build_steps(component: str, cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    root, run_dir = root.resolve(), run_dir.resolve()
    if component == tc.ARC and not cfg.get("native_harness"):
        generation = cfg.get("arc_generation", {
            "temperature": 0.0, "max_tokens": 256, "seed": 42, "top_p": 1.0,
            "stream": False, "retries": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        })
        return [_step("arc-selected", [sys.executable, str(root / "components" / "slim_arc.py"),
                                       "--selector", str(root / "selectors" / "arc-challenge.jsonl"),
                                       "--work-dir", str(run_dir / "arc"),
                                       "--api-url", str(cfg["api_base_url"]),
                                       "--model", str(cfg["model"]),
                                       "--generation-json", json.dumps(generation, separators=(",", ":"))])]
    if component == tc.HUMANEVAL and not cfg.get("native_harness"):
        generation = cfg.get("humaneval_generation", {
            "temperature": 0.0, "max_tokens": int(cfg.get("max_output_tokens", 8192)), "n": 1, "seed": 0,
            "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False, "cache_prompt": False,
        })
        base = run_dir / "humaneval"
        selector = root / "selectors" / "humaneval-plus-native.jsonl"
        samples = base / "samples.jsonl"
        generate = [str(cfg["slim_python"]), str(root / "components" / "text_coding_humaneval_generate.py"),
                    "--slim", "--selector", str(selector), "--samples", str(samples),
                    "--api-url", str(cfg["api_base_url"]).rstrip("/") + "/chat/completions",
                    "--model", str(cfg["model"]),
                    "--generation-json", json.dumps(generation, separators=(",", ":"))]
        grade = [sys.executable, str(root / "components" / "slim_humaneval_grade.py"),
                 "--selector", str(selector), "--expected",
                 str(root / "selectors" / "humaneval-plus-expected.json"),
                 "--samples", str(samples), "--output", str(base / "native_scores.json")]
        return [_step("humaneval-generate-selected", generate),
                _step("humaneval-grade-selected", grade)]
    if component in TEXT:
        steps = tc.build_steps(component, cfg, root, run_dir)
        if component == tc.IFEVAL:
            for step in steps:
                step["env"] = {**(step.get("env") or {}),
                               "NLTK_DATA": str(root / "bundle" / "vendor" / "nltk_data")}
        return steps
    if component in (AIDER, TERMINAL):
        return at.build_steps(component, cfg, root, run_dir)
    if component == HERMES:
        return _hermes_steps(cfg, root, run_dir)
    if component == TOOL:
        return _tool_steps(cfg, root, run_dir)
    raise ValueError(f"unknown component: {component}")


def collect(component: str, cfg: dict, root: Path, run_dir: Path) -> list[dict]:
    root, run_dir = root.resolve(), run_dir.resolve()
    if component == tc.ARC and not cfg.get("native_harness"):
        selected = {row["id"]: row for row in tc._jsonl(root / "selectors" / "arc-challenge.jsonl")}
        graded = tc._jsonl(run_dir / "arc" / "grades.jsonl")
        rows = []
        for row in graded:
            source = selected.get(row.get("id"))
            if source is None or row.get("prompt_sha256") != hashlib.sha256(source["prompt"].encode()).hexdigest():
                raise ValueError("ARC selected case identity mismatch")
            if row.get("target") != source["target"] or type(row.get("passed")) is not bool:
                raise ValueError("ARC selected grade identity mismatch")
            if row["passed"] != (row.get("answer") == source["target"]):
                raise ValueError("ARC selected grade and answer disagree")
            rows.append({"suite": tc.ARC, "case_id": f"arc:ARC-Challenge:{row['id']}",
                         "passed": row["passed"]})
        return tc._complete(rows, tc._expected(root, tc.ARC), tc.ARC)
    if component == tc.HUMANEVAL and not cfg.get("native_harness"):
        base = run_dir / "humaneval"
        selected = {row["task_id"]: row for row in tc._jsonl(root / "selectors" / "humaneval-plus-native.jsonl")}
        samples = tc._jsonl(base / "samples.jsonl")
        if len(samples) != len(selected) or {row["task_id"] for row in samples} != set(selected):
            raise ValueError("HumanEval+ generated sample set incomplete")
        truncated = [row["task_id"] for row in samples if row.get("truncated")]
        if truncated:
            raise ValueError("HumanEval+ output truncated; results are non-scorable: " + ", ".join(truncated))
        if any("finish_reason" not in row or "completion_tokens" not in row for row in samples):
            raise ValueError("HumanEval+ generation metadata missing; do not mix older protocol outputs")
        fixture = json.loads((root / "selectors" / "humaneval-plus-expected.json").read_text())
        expected_sha = hashlib.sha256((root / "selectors" / "humaneval-plus-native.jsonl").read_bytes()).hexdigest()
        native = json.loads((base / "native_scores.json").read_text())
        if fixture.get("selector_sha256") != expected_sha or native.get("selector_sha256") != expected_sha:
            raise ValueError("HumanEval+ fixture or grade selector hash mismatch")
        if native.get("dataset_md5") != tc.HE_SOURCE_MD5:
            raise ValueError("HumanEval+ source identity mismatch")
        rows = []
        for row in native["results"]:
            if row.get("task_id") not in selected or row.get("base_status") not in {"pass", "fail", "timeout"} or row.get("plus_status") not in {"pass", "fail", "timeout"}:
                raise ValueError("HumanEval+ selected grade invalid")
            rows.append({"suite": tc.HUMANEVAL, "case_id": row["task_id"],
                         "passed": row["base_status"] == row["plus_status"] == "pass"})
        return tc._complete(rows, tc._expected(root, tc.HUMANEVAL), tc.HUMANEVAL)
    if component in TEXT:
        return tc.collect(component, cfg, root, run_dir)
    if component in (AIDER, TERMINAL):
        return at.collect(component, cfg, root, run_dir)
    if component == HERMES:
        native = _hermes_dir(run_dir)
        summary = json.loads((native / "summary.json").read_text())
        manifest = json.loads((native / "manifest.json").read_text())
        return ag.parse_hermes(summary, manifest, ag._selected(root, HERMES),
                               {**cfg, "selected_only": True})
    if component == TOOL:
        path = _tool_dir(run_dir) / "result.json"
        result = json.loads(path.read_text())
        cap = int(cfg.get("max_output_tokens", 8192))
        truncated = [row.get("scenario_id") for row in result["scores"]["scenario_results"]
                     if re.search(r"^response_finish_reason_\d+=(?:length|max_tokens)\b",
                                  str(row.get("raw_log", "")), re.MULTILINE)
                     or re.search(rf"^response_finish_reason_\d+=\S+ completion_tokens={cap}\b",
                                  str(row.get("raw_log", "")), re.MULTILINE)]
        if truncated:
            raise ValueError("Tool-Eval output truncated; results are non-scorable: "
                             + ", ".join(str(x) for x in truncated))
        compact = {key: result.get(key) for key in
                   ("status", "tool_eval_bench_version", "total_scenarios", "config")}
        compact["scenario_results"] = [
            {key: row.get(key) for key in ("scenario_id", "status", "points")}
            for row in result["scores"]["scenario_results"]
        ]
        return ag.parse_tool_eval(compact, ag._selected(root, TOOL), cfg)
    raise ValueError(f"unknown component: {component}")
