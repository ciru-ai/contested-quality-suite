#!/usr/bin/env python3
"""Grade the five selected HumanEval+ tasks with locked EvalPlus outputs.

Generated code runs in a Bubblewrap process without host files or network.
For these five tasks EvalPlus uses exact equality: their native expected outputs
contain no floats or special-case oracles.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path


DATASET_MD5 = "fe585eb4df8c88d844eeb463ea4d0302"
EXPECTED_SHA256 = "db86db131ee6fc14c08c833531b1d0b3afd38e62a24b6ad57c3d42997291d08d"


class TestTimeout(Exception):
    pass


def _alarm(_signum: int, _frame: object) -> None:
    raise TestTimeout()


def worker() -> None:
    payload = json.load(sys.stdin)
    maximum_memory = 4 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory, maximum_memory))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    source: dict = {}
    signal.signal(signal.SIGALRM, _alarm)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            exec(payload["solution"], source)
            fn = source[payload["entry_point"]]
        except BaseException:
            print(json.dumps({"status": "fail"}), file=sys.__stdout__)
            return
        for arguments, expected, reference_time in zip(payload["inputs"], payload["expected"], payload["times"]):
            try:
                signal.setitimer(signal.ITIMER_REAL, max(1.0, float(reference_time) * 4.0))
                actual = fn(*arguments)
                signal.setitimer(signal.ITIMER_REAL, 0)
                if actual != expected:
                    print(json.dumps({"status": "fail"}), file=sys.__stdout__)
                    return
            except TestTimeout:
                print(json.dumps({"status": "timeout"}), file=sys.__stdout__)
                return
            except BaseException:
                print(json.dumps({"status": "fail"}), file=sys.__stdout__)
                return
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": "pass"}), file=sys.__stdout__)


def sandbox_command(script: Path) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("Bubblewrap (bwrap) is required to grade model-generated Python")
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session"]
    for root in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(root).exists():
            command += ["--ro-bind", root, root]
    if Path("/nix/store").exists():
        command += ["--ro-bind", "/nix/store", "/nix/store"]
    command += ["--dir", "/app", "--ro-bind", str(script), "/app/grade.py",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "/usr/bin/python3", "/app/grade.py", "--worker"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    if not all((args.selector, args.expected, args.samples, args.output)):
        parser.error("--selector, --expected, --samples, and --output are required")
    selected = [json.loads(line) for line in args.selector.read_text().splitlines() if line.strip()]
    fixture_bytes = args.expected.read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != EXPECTED_SHA256:
        raise ValueError("HumanEval+ reference output fixture hash mismatch")
    fixture = json.loads(fixture_bytes)
    if fixture.get("dataset_md5") != DATASET_MD5 or fixture.get("selector_sha256") != hashlib.sha256(args.selector.read_bytes()).hexdigest():
        raise ValueError("HumanEval+ expected outputs differ from the locked selector")
    if len(selected) != 5 or set(fixture["expected"]) != {row["task_id"] for row in selected}:
        raise ValueError("HumanEval+ fixture needs the five selected tasks")
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    by_id = {row["task_id"]: row for row in samples}
    if len(samples) != 5 or set(by_id) != set(fixture["expected"]):
        raise ValueError("HumanEval+ generated sample set is incomplete or duplicate")
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    command = sandbox_command(Path(__file__).resolve())
    probe = {"solution": "def probe(): return 1", "entry_point": "probe",
             "inputs": [[]], "expected": [1], "times": [0.01]}
    checked = subprocess.run(command, input=json.dumps(probe), text=True,
                             capture_output=True, timeout=10)
    if checked.returncode or json.loads(checked.stdout).get("status") != "pass":
        raise RuntimeError(f"HumanEval+ sandbox preflight failed: {checked.stderr.strip()}")
    for row in selected:
        task_id = row["task_id"]
        solution = by_id[task_id].get("solution")
        if not isinstance(solution, str):
            raise ValueError(f"{task_id}: missing sanitized solution")
        graded = {}
        for tier in ("base", "plus"):
            inputs = row[tier + "_input"]
            expected = fixture["expected"][task_id][tier]
            times = fixture["expected"][task_id][tier + "_time"]
            if len(inputs) != len(expected) or len(inputs) != len(times):
                raise ValueError(f"{task_id}: {tier} input/output lengths differ")
            request = {"solution": solution, "entry_point": row["entry_point"],
                       "inputs": inputs, "expected": expected, "times": times}
            try:
                process = subprocess.run(command, input=json.dumps(request), text=True,
                                         capture_output=True, timeout=max(120, 3 * len(inputs)))
                result = json.loads(process.stdout) if process.returncode == 0 else {"status": "fail"}
                status = result.get("status") if result.get("status") in {"pass", "fail", "timeout"} else "fail"
            except subprocess.TimeoutExpired:
                status = "timeout"
            graded[tier + "_status"] = status
        results.append({"task_id": task_id, **graded})
        print(f"{task_id}: base={graded['base_status']} plus={graded['plus_status']}", flush=True)
    args.output.write_text(json.dumps({"dataset_md5": DATASET_MD5,
                                       "evalplus": "0.3.1-selected-exact",
                                       "selector_sha256": fixture["selector_sha256"],
                                       "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
