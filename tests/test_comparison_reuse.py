# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Resuming the comparison study must not mix in records from other code or other settings,
and the analysis must notice a study whose runs used different diagnostics settings."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_per_degree_study as analysis  # noqa: E402
import run_comparison_study as study  # noqa: E402

CLEAN = {"commit": "a" * 40, "commit_source": "local_git"}
SPEC = {
    "label": "x",
    "algorithm": "polynomial_cl",
    "learning_rate": [0.1, 0.2],
    "sampling": "s",
}


def record(**changes):
    base = {
        "status": "completed",
        "spec": dict(SPEC),
        "protocol_fingerprint": "fp",
        "benchmark": dict(CLEAN),
    }
    base.update(changes)
    return base


class ReuseTests(unittest.TestCase):
    def reusable(self, rec, spec=SPEC, fingerprint="fp", code=CLEAN):
        return study.reusable(rec, spec, fingerprint, code)

    def test_a_matching_completed_record_is_reused(self):
        self.assertTrue(self.reusable(record()))

    def test_a_record_from_another_benchmark_commit_is_not_reused(self):
        other = {"commit": "b" * 40, "commit_source": "local_git"}
        self.assertFalse(self.reusable(record(benchmark=other)))
        self.assertFalse(
            self.reusable(record(benchmark=None))
        )  # written before records named code

    def test_a_failed_run_is_retried(self):
        self.assertFalse(self.reusable(record(status="failed")))

    def test_a_changed_spec_or_protocol_is_not_reused(self):
        self.assertFalse(
            self.reusable(record(), spec={**SPEC, "learning_rate": [0.1, 0.3]})
        )
        self.assertFalse(self.reusable(record(), fingerprint="other"))

    def test_a_stored_list_equals_the_specs_tuple(self):
        self.assertTrue(
            self.reusable(record(), spec={**SPEC, "learning_rate": (0.1, 0.2)})
        )

    def test_code_identity_is_not_enforced_for_a_dirty_or_unknown_checkout(self):
        for source in ("local_git_dirty", "unknown"):
            code = {"commit": "c" * 40, "commit_source": source}
            self.assertTrue(self.reusable(record(benchmark=None), code=code), source)
        self.assertTrue(
            self.reusable(
                record(benchmark=None),
                code={"commit": None, "commit_source": "unknown"},
            )
        )


class DiagnosticsConsistencyTests(unittest.TestCase):
    def check(self, settings):
        with tempfile.TemporaryDirectory() as tmp:
            for index, value in enumerate(settings):
                protocol = {} if value == "absent" else {"diagnostics": value}
                (Path(tmp) / f"cell_{index}.json").write_text(
                    json.dumps({"protocol": protocol})
                )
            return analysis.study_diagnostics(Path(tmp))

    def test_a_consistent_study_reports_its_setting(self):
        self.assertEqual(self.check([True, True]), (True, None))
        self.assertEqual(self.check([False, False]), (False, None))

    def test_a_core_that_does_not_record_it_gives_none(self):
        self.assertEqual(self.check(["absent", "absent"]), (None, None))

    def test_mixed_settings_are_a_problem(self):
        setting, problem = self.check([True, False])
        self.assertIsNone(setting)
        self.assertIn("diagnostics", problem)
        self.assertIsNotNone(self.check([True, "absent"])[1])


if __name__ == "__main__":
    unittest.main()
