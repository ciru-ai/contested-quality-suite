#!/usr/bin/env python3
"""Reject Terminal runs when Docker has too little room for native task setup."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def check_space(min_free_gib: int = 12) -> tuple[Path, int]:
    if min_free_gib < 1:
        raise ValueError("Terminal minimum free space must be positive")
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker is required for Terminal-Bench")
    root = subprocess.check_output(
        [docker, "info", "--format", "{{.DockerRootDir}}"], text=True
    ).strip()
    path = Path(root)
    if not path.is_dir():
        raise RuntimeError(f"Docker storage directory is unavailable: {path}")
    free = shutil.disk_usage(path).free
    if free < min_free_gib * 1024**3:
        raise RuntimeError(
            f"Terminal infrastructure preflight: Docker storage at {path} has "
            f"{free / 1024**3:.1f} GiB free; at least {min_free_gib} GiB is required "
            "to build and prepare the selected task images. Free Docker storage "
            "before starting a new run. This is not a model failure."
        )
    return path, free


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-gib", type=int, default=12)
    args = parser.parse_args()
    path, free = check_space(args.min_free_gib)
    print(f"Terminal Docker storage: {free / 1024**3:.1f} GiB free at {path}")
