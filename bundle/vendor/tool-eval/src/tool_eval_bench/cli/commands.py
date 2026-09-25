"""Scenario resolution helpers for the CLI.

Extracted from the monolithic ``cli/bench.py`` so that the scenario-selection
rules can be unit-tested and reused without importing the full CLI dispatch.
"""

from __future__ import annotations

import argparse

from tool_eval_bench.domain.scenarios import ScenarioDefinition


def resolve_scenarios(args: argparse.Namespace) -> list[ScenarioDefinition]:
    """Resolve scenarios from --short, --scenarios, --categories, and --hardmode flags.

    Priority: --scenarios (individual IDs) > --categories > --short > all.
    --hardmode-only runs Category P scenarios exclusively.
    --hardmode adds Category P scenarios to whichever base set is selected.
    """
    from tool_eval_bench.evals.scenarios import (
        ALL_SCENARIOS,
        ALL_SCENARIOS_WITH_HARDMODE,
        SCENARIOS,
    )
    from tool_eval_bench.evals.scenarios_hardmode import HARDMODE_SCENARIOS

    # Determine the base scenario pool
    if getattr(args, "hardmode_only", False):
        base = list(HARDMODE_SCENARIOS)
    elif args.short:
        base = list(SCENARIOS)
        if getattr(args, "hardmode", False):
            base.extend(HARDMODE_SCENARIOS)
    elif getattr(args, "hardmode", False):
        base = list(ALL_SCENARIOS_WITH_HARDMODE)
    else:
        base = list(ALL_SCENARIOS)

    if args.scenarios:
        requested = set(args.scenarios)
        return [s for s in base if s.id in requested]

    if args.categories:
        cats = {c.upper() for c in args.categories}
        return [s for s in base if s.category.value in cats]

    return base


def resolve_all_scenarios_for_ids(
    scenario_ids: list[str],
) -> list[ScenarioDefinition]:
    """Resolve ScenarioDefinitions by ID from ALL known scenarios.

    Used when reconstructing merged summaries from service-returned dicts
    (e.g. after resume merge) where we need the full definitions for scoring.
    """
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

    by_id = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}
    return [by_id[sid] for sid in scenario_ids if sid in by_id]
