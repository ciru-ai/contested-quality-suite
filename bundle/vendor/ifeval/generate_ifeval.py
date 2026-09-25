#!/usr/bin/env python3
"""Generate IFEval responses from an OpenAI-compatible server. Resume-safe, optional concurrency."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def chat(url: str, model: str, prompt: str, max_tokens: int, extra: dict, timeout: int) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        **extra,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    msg = data["choices"][0]["message"]
    return (msg.get("content") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["prompt"]] = row
        print(f"resume: {len(done)}/{len(prompts)} already done", file=sys.stderr)

    extra: dict = {"temperature": args.temperature}
    if args.top_p is not None:
        extra["top_p"] = args.top_p
    extra_body: dict = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    extra_body["chat_template_kwargs"] = {"enable_thinking": bool(args.thinking)}
    extra["extra_body"] = extra_body
    # some servers want kwargs at top level
    extra["chat_template_kwargs"] = extra_body["chat_template_kwargs"]
    if args.top_k is not None:
        extra["top_k"] = args.top_k

    pending = [item for item in prompts if item["prompt"] not in done]
    t0 = time.time()
    n_new = 0
    lock = threading.Lock()
    fh = out_path.open("a")

    def one(item: dict) -> dict:
        prompt = item["prompt"]
        response = ""
        for attempt in range(3):
            try:
                response = chat(args.url, args.model, prompt, args.max_tokens, extra, args.timeout)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception) as e:
                print(f"retry {attempt+1} key={item.get('key')}: {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        return {"key": item.get("key"), "prompt": prompt, "response": response}

    try:
        if args.workers <= 1:
            iterator = ((i, one(item)) for i, item in enumerate(pending, 1))
            for i, row in iterator:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_new += 1
                if n_new % 10 == 0 or n_new == len(pending):
                    elapsed = time.time() - t0
                    print(f"{len(done)+n_new}/{len(prompts)} new={n_new} {n_new/elapsed:.2f} req/s", file=sys.stderr)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(one, item): item for item in pending}
                for fut in as_completed(futs):
                    row = fut.result()
                    with lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        n_new += 1
                        if n_new % 10 == 0 or n_new == len(pending):
                            elapsed = time.time() - t0
                            print(
                                f"{len(done)+n_new}/{len(prompts)} new={n_new} "
                                f"{n_new/elapsed:.2f} req/s workers={args.workers}",
                                file=sys.stderr,
                            )
    finally:
        fh.close()
    print(f"wrote {out_path} ({len(done)+n_new} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
