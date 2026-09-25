#!/usr/bin/env python3
"""Run the selected ARC questions with EvalScope's native ARC adapter/grader.

The only override is the dataset-loading seam: selected, hash-checked original
ARC records are supplied in memory. EvalScope still renders its ARC prompt,
queries the model, extracts the answer, and calculates accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


CHOICE_RE = re.compile(r"^([A-Z])\) (.*)$")


def load_selector(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("ARC selector empty or has duplicate IDs")
    for row in rows:
        prompt = row["prompt"]
        segments = prompt.split("\n\n", 2)
        if len(segments) != 3:
            raise ValueError(f"{row['id']}: malformed ARC prompt")
        choices = []
        for index, line in enumerate(segments[2].splitlines()):
            match = CHOICE_RE.fullmatch(line)
            if match is None or match.group(1) != chr(ord("A") + index):
                raise ValueError(f"{row['id']}: malformed ARC option {index + 1}")
            choices.append(match.group(2))
        if not 2 <= len(choices) <= 5 or row["target"] not in [chr(ord("A") + n) for n in range(len(choices))]:
            raise ValueError(f"{row['id']}: invalid ARC choices or target")
        row["question"] = segments[1]
        row["choices"] = choices
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selector", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--api-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--generation-json", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    rows = load_selector(args.selector)
    generation = json.loads(args.generation_json)
    if not isinstance(generation, dict):
        raise ValueError("generation JSON must be an object")

    from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

    for row in rows:
        rendered = prompt(row["question"], row["choices"], MultipleChoiceTemplate.SINGLE_ANSWER)
        if rendered != row["prompt"]:
            raise ValueError(f"{row['id']}: EvalScope ARC prompt differs from locked selector")
    if args.check_only:
        print(f"ARC native prompt identity verified for {len(rows)} selected questions")
        return

    from evalscope import run_task
    from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
    from evalscope.benchmarks.arc.arc_adapter import ARCAdapter
    from evalscope.config import TaskConfig

    def selected_load(self):
        samples = [
            Sample(id=i, group_id=i, input=row["question"], choices=row["choices"],
                   target=row["target"], metadata={"id": row["id"]})
            for i, row in enumerate(rows)
        ]
        return DatasetDict({"ARC-Challenge": MemoryDataset(samples, name="ARC-Challenge")}), None

    ARCAdapter.load = selected_load
    args.work_dir.mkdir(parents=True, exist_ok=False)
    cfg = TaskConfig(
        model=args.model,
        model_id=args.model,
        datasets=["arc"],
        dataset_args={"arc": {"subset_list": ["ARC-Challenge"], "few_shot_num": 0}},
        eval_type="openai_api",
        eval_backend="Native",
        eval_batch_size=1,
        api_url=args.api_url,
        api_key=os.environ.get(args.api_key_env, "local"),
        generation_config=generation,
        repeats=1,
        seed=int(generation.get("seed", 42)),
        work_dir=str(args.work_dir),
        no_timestamp=True,
        collect_perf=False,
        ignore_errors=False,
    )
    run_task(cfg)


if __name__ == "__main__":
    main()
