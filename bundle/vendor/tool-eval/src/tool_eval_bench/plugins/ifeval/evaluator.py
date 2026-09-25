"""IFEval evaluator — prompt-level and instruction-level scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.plugins.ifeval.checkers import check_instruction


@dataclass(slots=True)
class InstructionResult:
    """Result of checking a single instruction constraint."""

    instruction_id: str
    passed: bool
    error: str | None = None


@dataclass(slots=True)
class PromptResult:
    """Result of evaluating all constraints for a single prompt."""

    prompt_pass: bool  # True only if ALL instructions passed
    instruction_results: list[InstructionResult] = field(default_factory=list)

    @property
    def instructions_passed(self) -> int:
        return sum(1 for r in self.instruction_results if r.passed)

    @property
    def instructions_total(self) -> int:
        return len(self.instruction_results)


def evaluate_prompt(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict[str, Any]],
) -> PromptResult:
    """Evaluate a model response against all constraints for a prompt.

    Parameters
    ----------
    response
        The model's raw text response.
    instruction_ids
        List of instruction IDs (e.g. ``["punctuation:no_comma", ...]``).
    kwargs_list
        List of kwargs dicts, one per instruction.
    """
    results: list[InstructionResult] = []

    for i, inst_id in enumerate(instruction_ids):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        # Filter out None values from kwargs
        kw = {k: v for k, v in kw.items() if v is not None}

        try:
            passed = check_instruction(inst_id, response, kw)
            results.append(
                InstructionResult(
                    instruction_id=inst_id,
                    passed=passed,
                )
            )
        except KeyError:
            # Unknown instruction — treat as pass (don't penalize for
            # constraint types we haven't implemented yet)
            results.append(
                InstructionResult(
                    instruction_id=inst_id,
                    passed=True,
                    error=f"Unknown instruction: {inst_id}",
                )
            )

    all_passed = all(r.passed for r in results)
    return PromptResult(prompt_pass=all_passed, instruction_results=results)
