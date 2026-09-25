"""Protocol setup checks that require no model inference."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bundle/vendor/terminal-bench-mini"))
sys.path.insert(0, str(ROOT / "bundle/vendor/tool-eval/src"))

import benchmark
from components import aider_terminal_adapter as at
from components import portable_adapter as pa
from components.slim_humaneval_grade import sandbox_command
from components.terminal_storage_preflight import check_space
from suite_manifest import SuiteManifest
import terminal_bench
import results as terminal_results


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        args = SimpleNamespace(model=None, label="test", tool_backend="vllm",
                               aider_threads=1, hermes_workers=1,
                               terminal_concurrency=1, tool_parallel=1,
                               terminal_context_length=262144, platform=None,
                               engine=None, backend=None)
        self.config = benchmark.portable_config(
            "http://127.0.0.1:8080/v1", {"id": "test", "owned_by": "local"}, args
        )

    def test_selected_tasks_and_protocol_are_locked(self) -> None:
        suite = SuiteManifest.load(ROOT / "terminal-core19-hotlist.json")
        self.assertEqual(len(suite.tasks), 4)
        self.assertEqual(self.config["protocol"]["sha256"], benchmark.PROTOCOL_SHA256)
        self.assertEqual(self.config["components"][at.AIDER]["threads"], 1)

    def test_terminal_context_and_cap_reach_native_runner(self) -> None:
        cfg = self.config["components"][at.TERMINAL]
        steps = at.build_steps(at.TERMINAL, cfg, ROOT, Path("/tmp/protocol-test-run"))
        self.assertEqual([s["name"] for s in steps],
                         ["terminal-storage-preflight", "terminal-doctor", "terminal-run"])
        self.assertEqual(steps[1]["argv"][-2:], ["--context-length", "262144"])
        run = steps[2]["argv"]
        self.assertEqual(run[run.index("--max-output-tokens") + 1], "8192")
        native = terminal_bench.build_config(
            job_name="test", tasks_dir=Path("/tmp"), tasks=["x"], model="test",
            endpoint="http://127.0.0.1:8080/v1", api_key="local", concurrency=1,
            context_length=262144, max_output_tokens=8192,
            agent_timeout_seconds=60, keep_containers=False,
        )
        kwargs = native["agents"][0]["kwargs"]
        self.assertEqual(kwargs["llm_kwargs"]["max_tokens"], 8192)
        self.assertEqual(kwargs["model_info"]["max_output_tokens"], 8192)

    def test_four_agent_workers_reach_each_native_runner(self) -> None:
        for name, key in ((at.AIDER, "threads"), (pa.HERMES, "workers"),
                          (at.TERMINAL, "concurrency"), (pa.TOOL, "parallel")):
            self.config["components"][name][key] = 4
        terminal = at.build_steps(at.TERMINAL, self.config["components"][at.TERMINAL],
                                  ROOT, Path("/tmp/protocol-test-run"))[2]["argv"]
        self.assertEqual(terminal[terminal.index("--concurrency") + 1], "4")
        tool = pa._tool_steps(self.config["components"][pa.TOOL], ROOT,
                              Path("/tmp/protocol-test-run"))[1]["argv"]
        self.assertEqual(tool[tool.index("--parallel") + 1], "4")

    def test_tool_and_humaneval_caps_are_explicit(self) -> None:
        tool = pa._tool_steps(self.config["components"][pa.TOOL], ROOT,
                              Path("/tmp/protocol-test-run"))[1]["argv"]
        self.assertEqual(json.loads(tool[tool.index("--backend-kwargs") + 1])["max_tokens"], 8192)
        human_cfg = {**self.config["model"],
                     **self.config["components"][pa.tc.HUMANEVAL]}
        human = pa.build_steps(pa.tc.HUMANEVAL, human_cfg, ROOT,
                               Path("/tmp/protocol-test-run"))[0]["argv"]
        self.assertEqual(json.loads(human[-1])["max_tokens"], 8192)

    def test_humaneval_sandbox_executes_resolved_python(self) -> None:
        if not shutil.which("bwrap"):
            self.skipTest("bwrap unavailable")
        probe = {"solution": "def probe(): return 1", "entry_point": "probe",
                 "inputs": [[]], "expected": [1], "times": [0.01]}
        cmd = sandbox_command(ROOT / "components/slim_humaneval_grade.py")
        self.assertIn(str(Path(sys.executable).resolve()), cmd)
        result = subprocess.run(cmd, input=json.dumps(probe), text=True,
                                capture_output=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "pass")

    def test_tool_stream_records_length_stop(self) -> None:
        try:
            import httpx
            from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
        except ImportError:
            self.skipTest("Tool-Eval dependencies unavailable")
        payload_seen = {}

        def handler(request):
            payload_seen.update(json.loads(request.content))
            body = "data: " + json.dumps({"choices": [{"delta": {"content": "x"},
                                                        "finish_reason": None}]}) + "\n\n"
            body += "data: " + json.dumps({"choices": [{"delta": {},
                                                         "finish_reason": "length"}],
                                          "usage": {"completion_tokens": 8192}}) + "\n\n"
            body += "data: [DONE]\n\n"
            return httpx.Response(200, text=body,
                                  headers={"content-type": "text/event-stream"})

        async def check():
            adapter = OpenAICompatibleAdapter()
            adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            result = await adapter.chat_completion(
                model="test", messages=[{"role": "user", "content": "hi"}],
                base_url="http://localhost:8080/v1",
                extra_params={"max_tokens": 8192}, stream=True,
            )
            await adapter.aclose()
            return result

        result = asyncio.run(check())
        self.assertEqual(payload_seen["max_tokens"], 8192)
        self.assertEqual(result.raw_response["finish_reason"], "length")
        self.assertEqual(result.completion_tokens, 8192)

    def test_low_docker_space_is_infrastructure_error(self) -> None:
        with patch("components.terminal_storage_preflight.shutil.which", return_value="/usr/bin/docker"), \
             patch("components.terminal_storage_preflight.subprocess.check_output", return_value="/tmp\n"), \
             patch("components.terminal_storage_preflight.shutil.disk_usage",
                   return_value=SimpleNamespace(free=2 * 1024**3)):
            with self.assertRaisesRegex(RuntimeError, "not a model failure"):
                check_space(12)

    def test_terminal_output_limit_hit_is_not_scored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / "trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "agent/trajectory.json").write_text(json.dumps({
                "steps": [{"source": "agent", "metrics": {"completion_tokens": 8192}}]
            }))
            model_dir = root / "model"
            model_dir.mkdir()
            attempt = terminal_results.normalized_attempt(
                trial, {"trial_name": "trial", "agent_result": {},
                        "verifier_result": {"rewards": {"reward": 0}}},
                attempt=1, model_dir=model_dir, repo_root=root,
                transcript_name="transcript.json", max_output_tokens=8192,
            )
            self.assertEqual(attempt["generation"]["output_limit_hits"], 1)
            raw = {"task": "cobol-modernization",
                   "task_provenance": {"content_sha256": "sha256:" + "0" * 64},
                   "evaluation_profile": {"agent": {"max_output_tokens": 8192}},
                   "attempts": [attempt], "passed": False, "completed": True}
            with self.assertRaisesRegex(at.AdapterError, "non-scorable"):
                at._validate_terminal_task(
                    raw, "cobol-modernization", "sha256:" + "0" * 64,
                    root / "result.json", expected_output_cap=8192,
                )


if __name__ == "__main__":
    unittest.main()
