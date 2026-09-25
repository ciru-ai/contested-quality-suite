#!/usr/bin/env python3
"""Generate first responses for the locked IFEval subset.

Uses the archived one-user-message request shape. The official strict IFEval
grader remains the sole authority for instruction-level pass/fail decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-p", type=float)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = p.parse_args()
    inputs = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not inputs or len(inputs) != len({row["prompt"] for row in inputs}):
        raise ValueError("IFEval selected input empty or has duplicate prompts")
    if args.output.exists():
        raise FileExistsError(f"fresh first-sample IFEval run required: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extras = {"chat_template_kwargs": {"enable_thinking": bool(args.thinking)}}
    if args.top_k is not None:
        extras["top_k"] = args.top_k
    if args.min_p is not None:
        extras["min_p"] = args.min_p
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    outputs = []
    request_log = []
    for item in inputs:
        prompt = item["prompt"]
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens, "stream": False, "temperature": args.temperature,
                **extras}
        if args.top_p is not None:
            body["top_p"] = args.top_p
        request = urllib.request.Request(args.url, data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            reply = json.load(response)
        choices = reply.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"IFEval key {item['key']}: expected one OpenAI chat choice")
        text = choices[0].get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"IFEval key {item['key']}: empty response or transport failure")
        outputs.append({"key": item["key"], "prompt": prompt, "response": text.strip()})
        request_log.append({"key": item["key"], "request": body,
                            "finish_reason": choices[0].get("finish_reason"),
                            "usage": reply.get("usage")})
        print(f"IFEval {len(outputs)}/{len(inputs)} key={item['key']}", flush=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
                           encoding="utf-8")
    (args.output.parent / "requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in request_log),
        encoding="utf-8")


if __name__ == "__main__":
    main()
