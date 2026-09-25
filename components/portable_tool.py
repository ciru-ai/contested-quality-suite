#!/usr/bin/env python3
"""Fresh-output guard for the bundled Tool-Eval v2.1.0 source runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--runner", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    launch = actions.add_parser("launch")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        version = subprocess.run([args.runner, "--version"], capture_output=True,
                                 text=True, check=True).stdout.strip()
        if version != "tool-eval-bench 2.1.0":
            raise ValueError(f"Tool-Eval v2.1.0 required; found {version}")
        output.mkdir(parents=True, exist_ok=False)
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not output.is_dir() or not command:
        raise ValueError("Tool-Eval output or launch command missing")
    with (output / "attempt-started.txt").open("x") as stream:
        stream.write("Use a fresh suite run ID after an interrupted Tool-Eval attempt.\n")
    os.execvp(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"Tool-Eval portable runner: {exc}", file=sys.stderr)
        raise SystemExit(2)
