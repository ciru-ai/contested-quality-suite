#!/usr/bin/env python3
"""Run the selected ARC questions with EvalScope's locked prompt and answer rule.

The single-answer parser below is the relevant EvalScope 1.12.0 logic, reduced
to the selected English single-choice form. The archive preserves the original
native adapter as a compatibility reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path


MARKER = re.compile(r"(?i)ANSWER\s*:\s*\**\s*")
BRACKETED = re.compile(r"[\(\[（【]\s*([A-Za-z\d](?:\s*[,，/、]\s*[A-Za-z\d])*)\s*[\)\]）】]")
PLAIN = re.compile(r"([A-Za-z\d][A-Za-z\d ,/、]*)")
TOKEN = re.compile(r"[A-Za-z\d]+")
OPTIONS = re.compile(r"^([A-Z])\) (.*)$")


def parse_answer(completion: str, n_choices: int) -> str | None:
    allowed = {chr(ord("A") + i) for i in range(n_choices)}
    for marker in reversed(list(MARKER.finditer(completion))):
        tail = completion[marker.end():]
        for pattern in (BRACKETED, PLAIN):
            match = pattern.match(tail)
            if match is None:
                continue
            capture = match.group(1)
            tokens = list(TOKEN.finditer(capture))
            end = 0
            for i, token in enumerate(tokens):
                word = token.group(0)
                if word in allowed or set(word).issubset(allowed):
                    end = token.end()
                    continue
                follows = end > 0
                precedes = i + 1 < len(tokens) and (
                    tokens[i + 1].group(0) in allowed or
                    set(tokens[i + 1].group(0)).issubset(allowed))
                if follows and precedes and word.lower() in {"and", "or"}:
                    continue
                break
            if end:
                answer = capture[:end].strip().rstrip(".")
                return answer if answer in allowed else None
    for letter in reversed(completion):
        if letter.isupper():
            return letter if letter in allowed else None
    return None


def checked_rows(selector: Path) -> list[dict]:
    rows = [json.loads(line) for line in selector.read_text().splitlines() if line.strip()]
    if len(rows) != 18 or len({row["id"] for row in rows}) != 18:
        raise ValueError("ARC selector must contain 18 unique cases")
    for row in rows:
        pieces = row["prompt"].split("\n\n", 2)
        if len(pieces) != 3:
            raise ValueError(f"invalid ARC prompt: {row['id']}")
        choices = pieces[2].splitlines()
        if not 2 <= len(choices) <= 5:
            raise ValueError(f"invalid ARC choices: {row['id']}")
        for i, text in enumerate(choices):
            match = OPTIONS.fullmatch(text)
            if not match or match.group(1) != chr(ord("A") + i):
                raise ValueError(f"invalid ARC choice order: {row['id']}")
        if row["target"] not in {chr(ord("A") + i) for i in range(len(choices))}:
            raise ValueError(f"invalid ARC target: {row['id']}")
        row["choice_count"] = len(choices)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-json", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    rows = checked_rows(args.selector)
    if args.work_dir.exists():
        raise FileExistsError(f"ARC output already exists: {args.work_dir}")
    args.work_dir.mkdir(parents=True)
    generation = json.loads(args.generation_json)
    generation.pop("retries", None)  # EvalScope transport setting, not an API parameter.
    headers = {"Content-Type": "application/json"}
    key = os.getenv(args.api_key_env)
    if key:
        headers["Authorization"] = "Bearer " + key
    with (args.work_dir / "grades.jsonl").open("w") as stream:
        for row in rows:
            body = {"model": args.model, "messages": [{"role": "user", "content": row["prompt"]}],
                    **generation}
            request = urllib.request.Request(args.api_url.rstrip("/") + "/chat/completions",
                                             data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = json.load(response)
            raw = reply["choices"][0]["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError(f"ARC non-text response: {row['id']}")
            answer = parse_answer(raw, row["choice_count"])
            result = {"id": row["id"], "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                      "target": row["target"], "response": raw, "answer": answer,
                      "passed": answer == row["target"]}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{row['id']}: {answer or '-'} {'pass' if result['passed'] else 'fail'}", flush=True)


if __name__ == "__main__":
    main()
