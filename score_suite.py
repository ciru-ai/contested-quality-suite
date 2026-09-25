#!/usr/bin/env python3
"""Score native-grader outcomes for the Qwen3.8 contested hotlist.

Input is JSONL with at least {"suite": "...", "case_id": "...", "passed": true}.
The runner of each original benchmark supplies the pass/fail decision.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def score(result_path: Path, *, allow_partial: bool) -> dict:
    manifest = json.loads((ROOT / "suite.json").read_text())
    selected = {(case["suite"], case["case_id"]): case for case in manifest["cases"]}
    results = {}
    ignored = 0
    with result_path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("suite"), str(row.get("case_id")))
            if key not in selected:
                ignored += 1
                continue
            if key in results:
                raise ValueError(f"duplicate result on line {number}: {key}")
            if type(row.get("passed")) is not bool:
                raise ValueError(f"passed must be boolean on line {number}: {key}")
            results[key] = row["passed"]
    missing = sorted(set(selected) - set(results))
    if missing and not allow_partial:
        preview = ", ".join(f"{suite}/{case_id}" for suite, case_id in missing[:5])
        raise ValueError(f"missing {len(missing)} selected outcomes: {preview}")
    per_suite: dict[str, dict] = defaultdict(lambda: {"passed": 0, "completed": 0, "total": 0})
    frontier = {"passed": 0, "completed": 0, "total": 0}
    for key, case in selected.items():
        bucket = frontier if case["tier"] == "frontier" else per_suite[case["suite"]]
        bucket["total"] += 1
        if key in results:
            bucket["completed"] += 1
            bucket["passed"] += int(results[key])
    for bucket in per_suite.values():
        bucket["pass_rate"] = (bucket["passed"] / bucket["completed"]
                               if bucket["completed"] else None)
    frontier["pass_rate"] = (frontier["passed"] / frontier["completed"]
                             if frontier["completed"] else None)
    complete_primary = all(v["completed"] == v["total"] for v in per_suite.values())
    macro = (sum(v["pass_rate"] for v in per_suite.values()) / len(per_suite)
             if complete_primary else None)
    covered = {name: value for name, value in per_suite.items()
               if value["completed"] == value["total"]}
    covered_macro = (sum(value["pass_rate"] for value in covered.values()) / len(covered)
                     if covered else None)
    return {
        "suite_id": manifest["id"],
        "selection_sha256": manifest["selection_sha256"],
        "content_sha256": manifest["content_sha256"],
        "primary_macro_pass_rate": macro,
        "primary_macro_percent": round(macro * 100, 2) if macro is not None else None,
        "completed_component_macro_percent": (round(covered_macro * 100, 2)
                                               if covered_macro is not None else None),
        "completed_components": sorted(covered),
        "primary_components": dict(sorted(per_suite.items())),
        "frontier": frontier,
        "completed_cases": len(results),
        "total_cases": len(selected),
        "missing_cases": [{"suite": suite, "case_id": case_id} for suite, case_id in missing],
        "ignored_unselected_rows": ignored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL of native-grader outcomes")
    parser.add_argument("--allow-partial", action="store_true", help="Report coverage before all components finish")
    parser.add_argument("--output", type=Path, help="Write the score JSON to this path")
    args = parser.parse_args()
    report = score(args.results, allow_partial=args.allow_partial)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
