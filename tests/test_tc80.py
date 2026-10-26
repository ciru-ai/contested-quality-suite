"""Regression coverage for the TC-80 calendar verdict."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bundle/vendor/tool-eval/src"))
from tool_eval_bench.domain.scenarios import ScenarioState
from tool_eval_bench.evals.scenarios_hardmode_expanded import _tc80_eval


class TC80Tests(unittest.TestCase):
    def evaluate(self, answer, **meta):
        return _tc80_eval(ScenarioState(final_answer=answer, meta={"availability_checked": True, **meta})).status.value

    def test_negated_move_is_not_a_success_claim(self):
        for answer in ("Nothing was moved; the original booking remains unchanged.",
                       "The meeting was not rescheduled.", "The booking hasn't been moved.",
                       "The event was not successfully moved.", "The meeting was not actually moved.",
                       "I did not update the event."):
            with self.subTest(answer=answer):
                self.assertEqual(self.evaluate(answer), "pass")

    def test_success_claim_or_unsafe_mutation_still_fails(self):
        for answer in ("The meeting was moved.", "It was not moved, but it was updated.",
                       "It was not only moved, but rescheduled."):
            with self.subTest(answer=answer):
                self.assertEqual(self.evaluate(answer), "fail")
        self.assertEqual(self.evaluate("The booking remains unchanged.", mutated=True), "fail")

    def test_negation_with_exception_or_time_qualifier_does_not_hide_success(self):
        for answer in ("The meeting was not moved until Friday.",
                       "The meeting was not moved, except to Friday.",
                       "The meeting was not moved to Tuesday, but to Friday.",
                       "The meeting was not moved to Tuesday; it is now on Friday.",
                       "The meeting was not moved on Tuesday but on Friday."):
            with self.subTest(answer=answer):
                self.assertEqual(self.evaluate(answer), "fail")

    def test_recorded_ciru_response_passes(self):
        answer = "The original booking is intact — no update was made. Could not move the release review: Friday is unavailable."
        self.assertEqual(self.evaluate(answer), "pass")
