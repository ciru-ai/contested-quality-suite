#!/usr/bin/env python3
"""Generate one first sample for each selected HumanEval+ task.

The chat prompt and EvalPlus sanitization follow the archived IU4 protocol.
No model-produced Python is executed in this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

DATASET_MD5 = "fe585eb4df8c88d844eeb463ea4d0302"


def read_selected(selector: Path, dataset: Path | None) -> list[dict]:
    if dataset is not None:
        if hashlib.md5(dataset.read_bytes()).hexdigest() != DATASET_MD5:
            raise ValueError("HumanEvalPlus-v0.1.10 source MD5 mismatch")
        full_rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(full_rows) != 164:
            raise ValueError("HumanEval+ source dataset must contain 164 tasks")
        full = {row["task_id"]: row for row in full_rows}
    else:
        full = None
    rows = [json.loads(x) for x in selector.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or len(rows) != len({row["task_id"] for row in rows}):
        raise ValueError("HumanEval+ selector empty or has duplicate task IDs")
    for row in rows:
        if full is not None and row != full.get(row["task_id"]):
            raise ValueError(f"HumanEval+ selector differs from pinned dataset: {row['task_id']}")
    return rows


def payload(row: dict, model: str, generation: dict) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant good at coding."},
            {"role": "user", "content":
             "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n"
             "```python\n" + row["prompt"].strip() + "\n```"},
        ],
        **generation,
    }
    if body.get("n", 1) != 1 or body.get("stream", False):
        raise ValueError("HumanEval+ requires one nonstream first sample per task")
    return body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selector", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--slim", action="store_true")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--api-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--generation-json", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if not args.slim and args.dataset is None:
        raise ValueError("--dataset is required for native mode")
    if args.slim and args.dataset is not None:
        raise ValueError("--slim uses the locked five-case selector without the full source dataset")
    if args.slim:
        from slim_sanitize import sanitize
    else:
        from evalplus.sanitize import sanitize
    rows = read_selected(args.selector, args.dataset)
    generation = json.loads(args.generation_json)
    if not isinstance(generation, dict):
        raise ValueError("generation JSON must be an object")
    bodies = [payload(row, args.model, generation) for row in rows]
    if args.check_only:
        print(f"HumanEval+ locked task identity and prompt setup verified for {len(rows)} cases")
        return
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    if args.samples.exists():
        raise FileExistsError(f"fresh first-sample run required; samples already exist: {args.samples}")
    raw_dir = args.samples.parent / "raw"
    raw_dir.mkdir(exist_ok=False)
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    samples = []
    for row, body in zip(rows, bodies):
        task_id = row["task_id"]
        request = urllib.request.Request(args.api_url, data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            reply = json.load(response)
        choices = reply.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"{task_id}: expected one OpenAI chat choice")
        message = choices[0].get("message", {})
        raw = message.get("content") or ""
        if not isinstance(raw, str):
            raise ValueError(f"{task_id}: completion content is not text")
        code = sanitize(raw, entrypoint=row["entry_point"])
        (raw_dir / f"{task_id.replace('/', '-')}.json").write_text(
            json.dumps({"task_id": task_id, "request": body, "response": reply,
                        "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest()},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finish_reason = choices[0].get("finish_reason")
        usage = reply.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None and (type(completion_tokens) is not int or completion_tokens < 0):
            raise ValueError(f"{task_id}: invalid completion token count")
        cap = generation.get("max_tokens")
        truncated = finish_reason in ("length", "max_tokens") or (
            type(cap) is int and type(completion_tokens) is int and completion_tokens >= cap
        )
        samples.append({"task_id": task_id, "solution": code,
                        "finish_reason": finish_reason,
                        "completion_tokens": completion_tokens,
                        "truncated": truncated})
        print(f"{task_id}: generated first sample ({finish_reason}, {completion_tokens} completion tokens)", flush=True)
    args.samples.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
                            encoding="utf-8")
    summary = {"max_tokens": generation.get("max_tokens"),
               "cases": [{key: row[key] for key in ("task_id", "finish_reason", "completion_tokens", "truncated")}
                         for row in samples]}
    (args.samples.parent / "generation-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    truncated_ids = [row["task_id"] for row in samples if row["truncated"]]
    if truncated_ids:
        raise ValueError("HumanEval+ responses reached the output cap; non-scorable: "
                         + ", ".join(truncated_ids))


if __name__ == "__main__":
    main()
