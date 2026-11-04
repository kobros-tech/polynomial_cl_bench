# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Official runs require a verified pinned core and a clean benchmark; development runs record why not."""

import unittest
import warnings
from unittest import mock

from helpers import small_protocol, synthetic_dataset

from polynomial_cl_bench import provenance as prov
from polynomial_cl_bench.algorithms import polynomial as module
from polynomial_cl_bench.algorithms.polynomial import AdapterError, PolynomialAdapter
from polynomial_cl_bench.provenance import ProvenanceError, enforce_run_mode

PIN = "9" * 40


def provenance(**override):
    base = {
        "version": "0.1.0",
        "commit": PIN,
        "commit_source": "pip_vcs_install",
        "remote_url": "https://github.com/kobros-tech/polynomial_cl_demo.git",
        "remote_verified": True,
        "remote_verification": "pip fetched the commit",
        "pinned_commit": PIN,
        "matches_pin": True,
        "pin_remote_status": "confirmed",
        "official_problems": [],
    }
    base.update(override)
    return base


def bench(**override):
    base = {
        "version": "0.1.0",
        "commit": "b" * 40,
        "commit_source": "local_git",
        "remote_url": None,
        "official_problems": [],
    }
    base.update(override)
    return base


BAD = {
    "mismatch": provenance(
        commit="1" * 40,
        matches_pin=False,
        official_problems=[
            f"core commit {'1' * 40} differs from the pinned commit {PIN}"
        ],
    ),
    "dirty": provenance(
        commit_source="local_git_dirty",
        official_problems=["core checkout has uncommitted changes to tracked files"],
    ),
    "unknown": provenance(
        commit=None,
        commit_source="unknown",
        matches_pin=None,
        official_problems=["core commit could not be determined"],
    ),
    "unverified_remote": provenance(
        remote_verified=False,
        pin_remote_status="unverified",
        official_problems=["core remote identity not verified: no matching remote"],
    ),
}
BAD_BENCH = bench(
    commit_source="local_git_dirty",
    official_problems=["benchmark checkout has uncommitted or untracked changes"],
)


class EnforcementRule(unittest.TestCase):
    def test_official_rejects_mismatched_dirty_unknown_and_unverified_cores(self):
        for name, info in BAD.items():
            with self.subTest(name), self.assertRaises(ProvenanceError):
                enforce_run_mode("official", info, bench())

    def test_official_rejects_a_dirty_benchmark(self):
        with self.assertRaises(ProvenanceError):
            enforce_run_mode("official", provenance(), BAD_BENCH)

    def test_development_never_rejects(self):
        for info in BAD.values():
            enforce_run_mode("development", info, BAD_BENCH)

    def test_official_on_the_pin_passes_without_any_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            enforce_run_mode("official", provenance(), bench())

    def test_real_provenance_reports_problems_consistently(self):
        info = prov.core_provenance()
        qualifies = (
            info["matches_pin"] is True
            and info["remote_verified"]
            and info["commit_source"] in ("local_git", "pip_vcs_install")
        )
        self.assertEqual(bool(info["official_problems"]), not qualifies)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            enforce_run_mode("beta", provenance(), bench())
        with self.assertRaises(AdapterError):
            PolynomialAdapter(mode="beta")


class AdapterModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol()

    def run_with(self, mode, info, bench_info=None):
        calls = []
        real = module.run_condition

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        adapter = PolynomialAdapter(mode=mode)
        with (
            mock.patch.object(module, "core_provenance", lambda: info),
            mock.patch.object(
                module, "benchmark_provenance", lambda: bench_info or bench()
            ),
            mock.patch.object(module, "run_condition", spy),
        ):
            try:
                result = adapter.run_sequence(self.dataset, self.protocol, 0)
            except ProvenanceError:
                return None, calls
        return result, calls

    def test_official_refusal_happens_before_any_training(self):
        for name, info in BAD.items():
            with self.subTest(name):
                result, calls = self.run_with("official", info)
                self.assertIsNone(result)
                self.assertEqual(calls, [])

    def test_official_refusal_for_a_dirty_benchmark_happens_before_any_training(self):
        result, calls = self.run_with("official", provenance(), BAD_BENCH)
        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_official_run_records_mode_and_both_commits(self):
        result, calls = self.run_with("official", provenance())
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.run_config["run_mode"], "official")
        self.assertEqual(result.algorithm["run_mode"], "official")
        core = result.algorithm["core"]
        self.assertTrue(core["remote_verified"])
        self.assertEqual(core["commit"], PIN)
        self.assertEqual(result.algorithm["benchmark"]["commit"], "b" * 40)

    def test_development_run_proceeds_and_records_why_it_is_not_official(self):
        result, calls = self.run_with("development", BAD["mismatch"], BAD_BENCH)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.run_config["run_mode"], "development")
        self.assertTrue(result.algorithm["core"]["official_problems"])
        self.assertTrue(result.algorithm["benchmark"]["official_problems"])
        self.assertFalse(result.algorithm["core"]["matches_pin"])

    def test_default_mode_is_development(self):
        self.assertEqual(PolynomialAdapter().mode, "development")


if __name__ == "__main__":
    unittest.main()
