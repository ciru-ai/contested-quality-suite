#!/usr/bin/env python3
"""Grade selected HumanEval+ samples with EvalPlus native tests in bwrap.

This process executes untrusted generated Python and must run only in the
closed Bubblewrap invocation assembled by text_coding_adapter.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evalplus.data import get_human_eval_plus
from evalplus.evaluate import check_correctness, get_groundtruth


DATASET_MD5 = "fe585eb4df8c88d844eeb463ea4d0302"
SOURCE = Path("/tmp/cache/evalplus/HumanEvalPlus-v0.1.10.jsonl")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--selector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source_bytes = SOURCE.read_bytes()
    if hashlib.md5(source_bytes).hexdigest() != DATASET_MD5:
        raise ValueError("HumanEval+ grader dataset MD5 mismatch")
    full = get_human_eval_plus(version="v0.1.10")
    if len(full) != 164:
        raise ValueError("HumanEval+ native loader did not return the full 164-task source")
    selected = [json.loads(line) for line in args.selector.read_text().splitlines() if line.strip()]
    ids = [row["task_id"] for row in selected]
    if len(ids) != len(set(ids)) or any(row != full.get(row["task_id"]) for row in selected):
        raise ValueError("HumanEval+ selector differs from native dataset")
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    by_id = {row["task_id"]: row for row in samples}
    if len(samples) != len(ids) or set(by_id) != set(ids):
        raise ValueError("HumanEval+ samples incomplete or duplicate")
    problems = {task_id: full[task_id] for task_id in ids}
    ground_key = "hotlist-" + hashlib.sha256(source_bytes + "\n".join(ids).encode()).hexdigest()
    ground = get_groundtruth(problems, ground_key, [])
    results = []
    for task_id in ids:
        solution = by_id[task_id].get("solution")
        if not isinstance(solution, str):
            raise ValueError(f"{task_id}: missing sanitized solution")
        native = check_correctness("humaneval", 0, problems[task_id], solution, ground[task_id],
                                   fast_check=False, min_time_limit=1.0, gt_time_limit_factor=4.0)
        base_status, plus_status = native["base"][0], native["plus"][0]
        if base_status not in {"pass", "fail", "timeout"} or plus_status not in {"pass", "fail", "timeout"}:
            raise ValueError(f"{task_id}: unexpected EvalPlus score {base_status}/{plus_status}")
        results.append({"task_id": task_id, "base_status": base_status,
                        "plus_status": plus_status})
        print(f"{task_id}: base={base_status} plus={plus_status}", flush=True)
    args.output.write_text(json.dumps({"dataset_md5": DATASET_MD5, "evalplus": "0.3.1",
                                       "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
