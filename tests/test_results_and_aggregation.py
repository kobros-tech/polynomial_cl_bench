# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Record validation, resource measurement, statistics, and experiment-level validation."""

import copy
import json
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from helpers import small_protocol
from test_runner import Base, go, spec

from polynomial_cl_bench import aggregate, resources, results
from polynomial_cl_bench.summarize import summarize_directory


def completed_record(tmp, **kw):
    out = Path(tmp) / "exp"
    go(spec(**kw.pop("spec_kw", {})), out)
    return json.loads((out / "runs" / "a__seed0.json").read_text()), out


class RecordValidation(Base):
    def setUp(self):
        super().setUp()
        self.rec, _ = completed_record(self.tmp.name)

    def problems(self, mutate):
        rec = copy.deepcopy(self.rec)
        mutate(rec)
        return results.validate_record(rec)

    def test_clean_record_has_no_problems(self):
        self.assertEqual(results.validate_record(self.rec), [])

    def test_tampered_metric_is_detected(self):
        self.assertTrue(
            self.problems(lambda r: r["metrics"].__setitem__("final_accuracy", 0.99))
        )
        self.assertTrue(
            self.problems(lambda r: r["metrics"].__setitem__("mean_forgetting", 0.0))
        )

    def test_tampered_matrix_is_detected(self):
        def tamper(r):
            r["metrics"]["accuracy_matrix"][4][0] = 0.0

        self.assertTrue(any("does not match" in p for p in self.problems(tamper)))

    def test_matrix_must_have_nulls_above_the_diagonal(self):
        def tamper(r):
            r["metrics"]["accuracy_matrix"][0][1] = 0.5

        self.assertTrue(self.problems(tamper))

    def test_run_key_binds_seed_config_and_protocol(self):
        self.assertTrue(self.problems(lambda r: r.__setitem__("seed", 7)))
        self.assertTrue(
            self.problems(lambda r: r["algorithm"]["config"].__setitem__("skill", 0.1))
        )
        self.assertTrue(
            self.problems(lambda r: r["protocol"].__setitem__("fingerprint", "0" * 64))
        )

    def test_required_fields_and_schema_version(self):
        self.assertTrue(self.problems(lambda r: r.pop("resources")))
        self.assertTrue(
            any(
                "schema_version" in p
                for p in self.problems(lambda r: r.__setitem__("schema_version", "9.9"))
            )
        )

    def test_resources_reject_negative_or_non_numeric_values(self):
        self.assertTrue(
            self.problems(lambda r: r["resources"].__setitem__("wall_time_seconds", -1))
        )
        self.assertTrue(
            self.problems(
                lambda r: r["resources"].__setitem__("wall_time_seconds", "fast")
            )
        )
        self.assertTrue(self.problems(lambda r: r["resources"].pop("peak_memory_mb")))

    def test_failed_record_rules(self):
        def f(r):
            r["status"] = "failed"

        self.assertTrue(self.problems(f))  # failed but with metrics and no error

    def test_nan_cannot_be_serialised(self):
        rec = copy.deepcopy(self.rec)
        rec["resources"]["wall_time_seconds"] = float("nan")
        self.assertTrue(
            any("not strict JSON" in p for p in results.validate_record(rec))
        )
        with self.assertRaises(ValueError):
            json.dumps(rec, allow_nan=False)

    def test_written_files_are_strict_json_with_null_for_nan(self):
        text = (Path(self.tmp.name) / "exp" / "runs" / "a__seed0.json").read_text()
        self.assertNotIn("NaN", text)
        self.assertIsNone(json.loads(text)["metrics"]["accuracy_matrix"][0][1])

    def test_write_is_atomic(self):
        path = Path(self.tmp.name) / "x.json"
        with (
            mock.patch.object(os, "replace", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            results.write_record(path, self.rec)
        self.assertFalse(path.exists())
        self.assertEqual(list(Path(self.tmp.name).glob("x.json*")), [])

    def test_invalid_record_is_refused_at_write_time(self):
        rec = copy.deepcopy(self.rec)
        rec["metrics"]["final_accuracy"] = 0.0
        with self.assertRaises(results.RecordError):
            results.write_record(Path(self.tmp.name) / "y.json", rec)

    def test_memory_budget_formula(self):
        self.assertEqual(
            results.memory_budget_by_experience(small_protocol(memory_per_class=200)),
            [0, 400, 800, 1200, 1600],
        )


class Resources(unittest.TestCase):
    def test_meter_reports_time_and_does_not_swallow_errors(self):
        meter = resources.RunMeter()
        with self.assertRaises(KeyError), meter:
            _ = np.ones(10).sum()
            raise KeyError("x")
        report = meter.report()
        self.assertGreaterEqual(report["wall_time_seconds"], 0)
        self.assertGreaterEqual(report["cpu_time_seconds"], 0)

    def test_peak_memory_is_null_when_the_counter_cannot_be_reset(self):
        with (
            mock.patch.object(resources, "reset_peak_rss", return_value=False),
            resources.RunMeter() as meter,
        ):
            pass
        self.assertIsNone(meter.report()["peak_memory_mb"])
        self.assertIsNone(meter.report()["peak_memory_scope"])

    @unittest.skipUnless(
        resources.reset_peak_rss(), "peak RSS counter cannot be reset here"
    )
    def test_peak_memory_sees_a_large_allocation(self):
        with resources.RunMeter() as meter:
            block = np.ones(20_000_000)  # 160 MB
            block.sum()
        del block
        self.assertGreater(
            meter.report()["peak_memory_mb"], meter.report()["rss_before_run_mb"] + 100
        )

    def test_environment_has_the_thread_settings(self):
        env = resources.environment_info()
        self.assertIn("OMP_NUM_THREADS", env["thread_env"])
        self.assertIn("python_version", env)


class Statistics(unittest.TestCase):
    def test_summary_values(self):
        s = aggregate.summarize([0.6, 0.7, 0.8])
        self.assertAlmostEqual(s["mean"], 0.7)
        self.assertAlmostEqual(s["std"], 0.1)  # sample std, ddof=1
        half = 4.303 * 0.1 / math.sqrt(3)
        self.assertAlmostEqual(s["ci95"][0], 0.7 - half, places=6)
        self.assertEqual(s["n"], 3)

    def test_single_value_has_no_std_or_interval(self):
        s = aggregate.summarize([0.5])
        self.assertEqual((s["n"], s["mean"], s["std"], s["ci95"]), (1, 0.5, None, None))

    def test_none_values_are_not_counted(self):
        self.assertEqual(aggregate.summarize([None, None])["n"], 0)
        self.assertEqual(aggregate.summarize([1.0, None, 3.0])["n"], 2)

    def test_t_table_is_never_below_the_true_critical_value(self):
        try:
            from scipy import stats
        except ImportError:
            self.skipTest("scipy not installed")
        for df in range(1, 600):
            true = stats.t.ppf(0.975, df)
            self.assertGreaterEqual(aggregate.t_critical_95(df), true - 6e-4, df)
            self.assertLess(aggregate.t_critical_95(df) - true, 0.03, df)


def fake_rec(label, seed, final, forgetting, status="completed"):
    return {
        "seed": seed,
        "status": status,
        "algorithm": {"label": label, "name": label},
        "error": None if status == "completed" else {"message": "x"},
        "metrics": {"final_accuracy": final, "mean_forgetting": forgetting},
    }


class PairedComparison(unittest.TestCase):
    def test_differences_counts_and_exclusions(self):
        a = [
            fake_rec("a", s, v, 0.1)
            for s, v in [(0, 0.8), (1, 0.7), (2, 0.9), (3, 0.5)]
        ]
        b = [
            fake_rec("b", 0, 0.7, 0.2),
            fake_rec("b", 1, 0.8, 0.2),
            fake_rec("b", 2, 0.9, 0.2),
            fake_rec("b", 3, 0.0, 0.0, "failed"),
        ]
        c = aggregate.paired_comparison(a, b, "final_accuracy", ("a", "b"))
        self.assertEqual(c["paired_seeds"], [0, 1, 2])
        self.assertEqual(c["excluded_seeds"], [3])
        for got, want in zip(c["per_seed_differences"], [0.1, -0.1, 0.0]):
            self.assertAlmostEqual(got, want)
        self.assertEqual((c["a_higher"], c["b_higher"], c["ties"]), (1, 1, 1))
        self.assertAlmostEqual(c["mean_difference"], 0.0)
        self.assertIn("tentative", c["qualification"])
        self.assertNotIn(
            "significant", c["qualification"].replace("no significance", "")
        )

    def test_one_pair_has_no_interval(self):
        c = aggregate.paired_comparison(
            [fake_rec("a", 0, 0.8, 0)], [fake_rec("b", 0, 0.7, 0)]
        )
        self.assertIsNone(c["ci95_mean_difference"])
        self.assertIn("no interval", c["qualification"])

    def test_no_common_seed(self):
        c = aggregate.paired_comparison(
            [fake_rec("a", 0, 0.8, 0)], [fake_rec("b", 1, 0.7, 0)]
        )
        self.assertEqual(c["n_pairs"], 0)
        self.assertIsNone(c["mean_difference"])


class AggregateCounts(Base):
    def test_missing_and_failed_runs_are_counted_separately(self):
        s = spec(
            [
                ("good", {"algorithm": "g"}),
                ("bad", {"algorithm": "x", "behavior": "raise_seed1"}),
            ]
        )
        go(s, self.out)
        (self.out / "runs" / "good__seed2.json").unlink()
        summary = summarize_directory(self.out)
        good = summary["aggregate"]["algorithms"]["good"]
        self.assertEqual(
            (good["n_completed"], good["n_failed"], good["n_missing"]), (2, 0, 1)
        )
        self.assertEqual(good["missing_seeds"], [2])
        self.assertFalse(good["complete"])
        self.assertTrue(
            any("good seed 2 has no record" in p for p in summary["problems"])
        )

    def test_per_experience_statistics_use_the_sample_std(self):
        go(spec([("a", {"algorithm": "fake_a", "skill": 0.8})]), self.out)
        recs = [
            json.loads(p.read_text())
            for p in sorted((self.out / "runs").glob("*.json"))
        ]
        per_exp = np.array([r["metrics"]["accuracy_per_experience"] for r in recs])
        got = summarize_directory(self.out)["aggregate"]["algorithms"]["a"]["metrics"][
            "accuracy_per_experience"
        ]
        np.testing.assert_allclose(got["mean"], per_exp.mean(axis=0))
        np.testing.assert_allclose(got["std"], per_exp.std(axis=0, ddof=1))

    def test_aggregates_match_a_direct_computation(self):
        go(spec([("a", {"algorithm": "fake_a", "skill": 0.8})]), self.out)
        recs = [
            json.loads(p.read_text())
            for p in sorted((self.out / "runs").glob("*.json"))
        ]
        finals = [r["metrics"]["final_accuracy"] for r in recs]
        got = summarize_directory(self.out)["aggregate"]["algorithms"]["a"]["metrics"][
            "final_accuracy"
        ]
        self.assertAlmostEqual(got["mean"], float(np.mean(finals)))
        self.assertAlmostEqual(got["std"], float(np.std(finals, ddof=1)))


class ExperimentValidation(Base):
    def clean(self):
        go(spec(), self.out)
        return summarize_directory(self.out)

    def test_clean_experiment_has_no_integrity_problems_but_is_not_publishable(self):
        summary = self.clean()
        self.assertEqual(summary["problems"], [])
        v = summary["validation"]
        self.assertTrue(v["valid"] and not v["publishable"])
        text = " ".join(v["publication_blockers"])
        self.assertIn("not 'official'", text)
        self.assertIn("not selected by validation tuning", text)
        self.assertIn("reproducibility check was not run", text)

    def test_summary_files_are_written(self):
        self.clean()
        self.assertTrue((self.out / "summary.json").is_file())
        md = (self.out / "summary.md").read_text()
        self.assertIn("Publishable comparison: **no**", md)
        self.assertNotIn("NaN", (self.out / "summary.json").read_text())

    def test_different_plans_for_one_seed_are_an_integrity_problem(self):
        self.clean()
        p = self.out / "runs" / "b__seed1.json"
        rec = json.loads(p.read_text())
        rec["plan_fingerprint"] = "tampered"
        p.write_text(json.dumps(rec))
        problems = summarize_directory(self.out)["problems"]
        self.assertTrue(
            any(
                "seed 1" in x and "different experience/replay plans" in x
                for x in problems
            )
        )

    def test_different_dataset_is_an_integrity_problem(self):
        self.clean()
        p = self.out / "runs" / "b__seed0.json"
        rec = json.loads(p.read_text())
        rec["dataset"]["fingerprint"] = "other"
        p.write_text(json.dumps(rec))
        self.assertTrue(
            any(
                "dataset array fingerprints" in x
                for x in summarize_directory(self.out)["problems"]
            )
        )

    def test_mixed_commits_are_an_integrity_problem(self):
        self.clean()
        p = self.out / "runs" / "b__seed0.json"
        rec = json.loads(p.read_text())
        rec["environment"]["benchmark_commit"] = "e" * 40
        p.write_text(json.dumps(rec))
        self.assertTrue(
            any(
                "different core/benchmark commits" in x
                for x in summarize_directory(self.out)["problems"]
            )
        )

    def test_hand_edited_record_is_reported_not_skipped(self):
        self.clean()
        p = self.out / "runs" / "a__seed2.json"
        rec = json.loads(p.read_text())
        rec["metrics"]["final_accuracy"] = 1.0
        p.write_text(json.dumps(rec))
        summary = summarize_directory(self.out)
        self.assertTrue(any("a__seed2.json" in x for x in summary["problems"]))
        a = summary["aggregate"]["algorithms"]["a"]
        self.assertEqual(a["n_completed"], 2)  # not used...
        self.assertEqual(
            a["n_missing"], 1
        )  # ...and not silently dropped: counted as missing

    def test_core_cross_check_disagreement_is_an_integrity_problem(self):
        self.clean()
        p = self.out / "runs" / "a__seed0.json"
        rec = json.loads(p.read_text())
        rec["checks"] = {
            "core_reported_cross_check": {
                "final_accuracy_abs_diff": 0.01,
                "mean_forgetting_abs_diff": 0.0,
                "tolerance": 1e-9,
                "agrees": False,
            }
        }
        p.write_text(json.dumps(rec))
        self.assertTrue(
            any(
                "disagree with the algorithm's own report" in x
                for x in summarize_directory(self.out)["problems"]
            )
        )

    def test_publishable_when_every_condition_holds(self):
        """Everything satisfied: official mode, tuned configs, test split, reproducibility."""
        tuned = {
            "selection": "validation_tuning",
            "tuning_official_eligible": True,
            "criterion": "rule",
            "selection_margin": 0.005,
            "selection_seeds": [100, 101],
            "budget_runs": 30,
            "validation_fingerprint": "v" * 64,
        }
        s = spec(
            [
                ("a", {"algorithm": "fa", "hyperparameter_provenance": tuned}),
                ("b", {"algorithm": "fb", "hyperparameter_provenance": tuned}),
            ],
            seeds=(0, 1, 2, 3, 4),
        )
        s = type(s)(
            **{
                **s.__dict__,
                "protocol": small_protocol(seeds=(0, 1, 2, 3, 4)),
                "mode": "development",
            }
        )
        go(s, self.out, reproduce_check=True)
        summary = summarize_directory(self.out)
        # development mode and the missing core/pin provenance still block, nothing else does
        remaining = summary["validation"]["publication_blockers"]
        self.assertTrue(
            all(
                ("not 'official'" in b) or b.startswith("provenance problems")
                for b in remaining
            ),
            remaining,
        )
        self.assertEqual(summary["problems"], [])

    def _blockers_for(self, prov_a, prov_b):
        from polynomial_cl_bench.validation import _selection_procedure_blockers

        configs = {
            "a": {"hyperparameter_provenance": prov_a},
            "b": {"hyperparameter_provenance": prov_b},
        }
        return _selection_procedure_blockers(["a", "b"], configs)

    def _prov(self, **override):
        base = {
            "selection": "validation_tuning",
            "criterion": "rule",
            "selection_margin": 0.005,
            "selection_seeds": [100, 101],
            "budget_runs": 30,
            "validation_fingerprint": "v" * 64,
        }
        return {**base, **override}

    def test_same_selection_procedure_raises_no_blocker(self):
        self.assertEqual(self._blockers_for(self._prov(), self._prov()), [])

    def test_different_selection_procedures_block_a_comparison(self):
        for key, other in [
            ("criterion", "another rule"),
            ("selection_margin", 0.01),
            ("selection_seeds", [100, 102]),
            ("budget_runs", 15),
            ("validation_fingerprint", "w" * 64),
        ]:
            with self.subTest(key=key):
                blockers = self._blockers_for(self._prov(), self._prov(**{key: other}))
                self.assertEqual(len(blockers), 1, blockers)
                self.assertIn(key, blockers[0])

    def test_unrecorded_selection_procedure_blocks_rather_than_passes(self):
        legacy = self._prov()
        del legacy["budget_runs"]
        blockers = self._blockers_for(self._prov(), legacy)
        self.assertTrue(
            any("not recorded" in b and "budget_runs" in b for b in blockers)
        )


if __name__ == "__main__":
    unittest.main()
