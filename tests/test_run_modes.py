"""Official runs require the pinned, clean core; development runs record why they would not."""

import unittest
import warnings
from unittest import mock

from helpers import small_protocol, synthetic_dataset
from polynomial_cl_bench import provenance as prov
from polynomial_cl_bench.algorithms import polynomial as module
from polynomial_cl_bench.algorithms.polynomial import AdapterError, PolynomialAdapter
from polynomial_cl_bench.provenance import (
    ProvenanceError,
    RemoteUnverifiedWarning,
    enforce_run_mode,
)

PIN = "9" * 40


def provenance(**override):
    base = {
        "version": "0.1.0", "commit": PIN, "commit_source": "local_git", "remote_url": None,
        "remote_verified": False, "pinned_commit": PIN, "matches_pin": True,
        "pin_remote_status": "unverified", "official_problems": [],
    }
    base.update(override)
    return base


BAD = {
    "mismatch": provenance(
        commit="1" * 40, matches_pin=False,
        official_problems=[f"core commit {'1' * 40} differs from the pinned commit {PIN}"],
    ),
    "dirty": provenance(
        commit_source="local_git_dirty",
        official_problems=["core checkout has uncommitted changes to tracked files"],
    ),
    "unknown": provenance(
        commit=None, commit_source="unknown", matches_pin=None,
        official_problems=["core commit could not be determined"],
    ),
}


class EnforcementRule(unittest.TestCase):
    def test_official_rejects_mismatched_dirty_and_unknown_cores(self):
        for name, info in BAD.items():
            with self.subTest(name), self.assertRaises(ProvenanceError):
                enforce_run_mode("official", info)

    def test_development_never_rejects(self):
        for info in BAD.values():
            enforce_run_mode("development", info)

    def test_official_on_the_pin_warns_while_remote_is_unverified(self):
        with self.assertWarns(RemoteUnverifiedWarning):
            enforce_run_mode("official", provenance())

    def test_no_warning_once_the_remote_is_confirmed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            enforce_run_mode("official", provenance(pin_remote_status="confirmed"))

    def test_project_status_is_unverified_until_someone_changes_it(self):
        self.assertEqual(prov.PIN_REMOTE_STATUS, "unverified")
        self.assertEqual(prov.core_provenance()["pin_remote_status"], "unverified")

    def test_real_provenance_reports_problems_consistently(self):
        info = prov.core_provenance()
        self.assertEqual(
            bool(info["official_problems"]),
            info["matches_pin"] is not True or info["commit_source"] in ("local_git_dirty", "unknown"),
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            enforce_run_mode("beta", provenance())
        with self.assertRaises(AdapterError):
            PolynomialAdapter(mode="beta")


class AdapterModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol()

    def run_with(self, mode, info):
        calls = []
        real = module.run_condition

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        adapter = PolynomialAdapter(mode=mode)
        with mock.patch.object(module, "core_provenance", lambda: info), mock.patch.object(
            module, "run_condition", spy
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

    def test_official_run_on_the_pin_records_mode_and_unverified_remote(self):
        with self.assertWarns(RemoteUnverifiedWarning):
            result, calls = self.run_with("official", provenance())
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.run_config["run_mode"], "official")
        self.assertEqual(result.algorithm["run_mode"], "official")
        core = result.algorithm["core"]
        self.assertEqual(core["pin_remote_status"], "unverified")
        self.assertFalse(core["remote_verified"])

    def test_development_run_proceeds_and_records_why_it_is_not_official(self):
        result, calls = self.run_with("development", BAD["mismatch"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.run_config["run_mode"], "development")
        self.assertTrue(result.algorithm["core"]["official_problems"])
        self.assertFalse(result.algorithm["core"]["matches_pin"])

    def test_default_mode_is_development(self):
        self.assertEqual(PolynomialAdapter().mode, "development")


if __name__ == "__main__":
    unittest.main()
