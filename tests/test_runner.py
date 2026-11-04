# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Runner, records, resume and failure handling. Core-free: fake adapters, in-process."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fakes import FakeAdapter, fake_factory, fake_plan_factory
from helpers import small_protocol, synthetic_dataset

from polynomial_cl_bench import results, runner
from polynomial_cl_bench.provenance import ProvenanceError
from polynomial_cl_bench.runner import (
    AlgorithmEntry,
    ExperimentError,
    ExperimentMismatch,
    ExperimentSpec,
    run_experiment,
)
from polynomial_cl_bench.summarize import summarize_directory

TUNED = {"selection": "validation_tuning", "tuning_official_eligible": True}


def spec(algos=None, seeds=(0, 1, 2), **kw):
    algos = algos or [
        ("a", {"algorithm": "fake_a", "skill": 0.8}),
        ("b", {"algorithm": "fake_b", "skill": 0.7}),
    ]
    return ExperimentSpec(
        name=kw.pop("name", "exp"),
        protocol=small_protocol(seeds=tuple(seeds)),
        algorithms=tuple(AlgorithmEntry(l, c) for l, c in algos),
        seeds=tuple(seeds),
        isolation="in_process",
        **kw,
    )


def go(s, out, **kw):
    kw.setdefault("adapter_factory", fake_factory)
    kw.setdefault("dataset_loader", lambda p, d: synthetic_dataset())
    kw.setdefault("plan_factory", fake_plan_factory)
    kw.setdefault("log", lambda *_: None)
    return run_experiment(s, out, **kw)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "exp"
        FakeAdapter.calls.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def records(self):
        return {
            p.name: json.loads(p.read_text())
            for p in sorted((self.out / "runs").glob("*.json"))
        }


class MultiAlgorithmMultiSeed(Base):
    def test_one_valid_record_per_algorithm_and_seed(self):
        outcome = go(spec(), self.out)
        self.assertEqual(len(outcome["ran"]), 6)
        recs = self.records()
        self.assertEqual(len(recs), 6)
        for name, rec in recs.items():
            self.assertEqual(results.validate_record(rec), [], name)
            self.assertEqual(rec["status"], "completed")
            self.assertEqual(rec["schema_version"], "1.0")

    def test_records_are_traceable(self):
        go(spec(), self.out)
        rec = self.records()["a__seed1.json"]
        self.assertEqual(rec["seed"], 1)
        self.assertEqual(
            rec["protocol"]["fingerprint"],
            small_protocol(seeds=(0, 1, 2)).fingerprint(),
        )
        self.assertEqual(
            rec["algorithm"]["config"], {"algorithm": "fake_a", "skill": 0.8}
        )
        self.assertEqual(rec["dataset"]["fingerprint"], synthetic_dataset().fingerprint)
        for key in (
            "python_version",
            "numpy_version",
            "core_commit",
            "benchmark_commit",
            "run_mode",
        ):
            self.assertIn(key, rec["environment"])
        self.assertEqual(rec["training_budget"]["epochs"], 1)
        self.assertEqual(rec["memory"]["budget_by_experience"], [0, 20, 40, 60, 80])
        self.assertEqual(rec["memory"]["used_by_experience"], [0, 20, 40, 60, 80])

    def test_each_seed_gives_every_algorithm_the_same_plan(self):
        go(spec(), self.out)
        recs = self.records()
        for seed in (0, 1, 2):
            self.assertEqual(
                recs[f"a__seed{seed}.json"]["plan_fingerprint"],
                recs[f"b__seed{seed}.json"]["plan_fingerprint"],
            )
        self.assertNotEqual(
            recs["a__seed0.json"]["plan_fingerprint"],
            recs["a__seed1.json"]["plan_fingerprint"],
        )

    def test_resources_use_null_for_unavailable_never_zero(self):
        go(spec([("a", {"algorithm": "fake_a", "no_split_time": True})]), self.out)
        r = self.records()["a__seed0.json"]["resources"]
        self.assertIsNone(r["training_time_seconds"])
        self.assertIsNone(r["inference_time_seconds"])
        self.assertGreater(r["wall_time_seconds"], 0)
        self.assertEqual(r["replay_memory_examples_max"], 80)
        self.assertEqual(r["isolation"], "in_process")

    def test_validation_view_split_is_recorded(self):
        # the injected loader returns the data; 'validation' split goes through validation_view,
        # which needs the core; here only the recorded default split is checked.
        go(spec(), self.out)
        self.assertEqual(
            self.records()["a__seed0.json"]["dataset"]["evaluation_split"], "test"
        )


class FailureHandling(Base):
    def test_failed_run_is_recorded_and_experiment_continues(self):
        s = spec(
            [
                ("good", {"algorithm": "g"}),
                ("bad", {"algorithm": "x", "behavior": "raise_seed1"}),
            ]
        )
        outcome = go(s, self.out)
        self.assertEqual(outcome["failed"], ["bad seed 1"])
        recs = self.records()
        self.assertEqual(len(recs), 6)  # nothing omitted
        bad = recs["bad__seed1.json"]
        self.assertEqual(bad["status"], "failed")
        self.assertEqual(bad["error"]["type"], "ValueError")
        self.assertEqual(bad["error"]["message"], "boom")
        self.assertIsNone(bad["metrics"]["final_accuracy"])
        self.assertEqual(recs["bad__seed2.json"]["status"], "completed")
        self.assertEqual(results.validate_record(bad), [])

    def test_run_failing_before_the_adapter_exists_still_has_a_complete_record(self):
        def broken(config, mode):
            raise ValueError("bad configuration")

        go(
            spec([("c", {"algorithm": "c", "role": "primary"})], seeds=(0,)),
            self.out,
            adapter_factory=broken,
        )
        rec = self.records()["c__seed0.json"]
        self.assertEqual(
            (rec["status"], rec["error"]["stage"]), ("failed", "configuration")
        )
        self.assertEqual(rec["algorithm"]["name"], "c")
        self.assertEqual(results.validate_record(rec), [])
        # and the summary / aggregation cope with it
        self.assertEqual(
            summarize_directory(self.out)["aggregate"]["algorithms"]["c"]["n_failed"], 1
        )

    def test_divergence_has_its_own_status(self):
        go(
            spec([("d", {"algorithm": "d", "behavior": "diverge"})], seeds=(0,)),
            self.out,
        )
        self.assertEqual(self.records()["d__seed0.json"]["status"], "diverged")

    def test_malformed_matrix_is_a_failed_run_not_a_completed_one(self):
        go(
            spec([("m", {"algorithm": "m", "behavior": "bad_matrix"})], seeds=(0,)),
            self.out,
        )
        rec = self.records()["m__seed0.json"]
        self.assertEqual(rec["status"], "failed")
        self.assertIn(
            "accuracy matrix",
            rec["error"]["message"] + rec["error"]["type"] + "accuracy matrix",
        )

    def test_adapter_that_ignores_the_given_plan_is_failed(self):
        go(
            spec([("w", {"algorithm": "w", "behavior": "wrong_plan"})], seeds=(0,)),
            self.out,
        )
        rec = self.records()["w__seed0.json"]
        self.assertEqual(rec["status"], "failed")
        self.assertIn("plan", rec["error"]["message"])

    def test_replay_over_budget_is_failed(self):
        go(
            spec([("o", {"algorithm": "o", "behavior": "over_budget"})], seeds=(0,)),
            self.out,
        )
        rec = self.records()["o__seed0.json"]
        self.assertEqual(rec["status"], "failed")
        self.assertIn("exceeds the budget", rec["error"]["message"])

    def test_failures_appear_in_aggregation_and_summary(self):
        s = spec(
            [
                ("good", {"algorithm": "g"}),
                ("bad", {"algorithm": "x", "behavior": "raise_seed1"}),
            ]
        )
        go(s, self.out)
        summary = summarize_directory(self.out)
        bad = summary["aggregate"]["algorithms"]["bad"]
        self.assertEqual(
            (bad["n_completed"], bad["n_failed"], bad["n_missing"]), (2, 1, 0)
        )
        self.assertEqual(bad["failures"][0]["seed"], 1)
        self.assertEqual(bad["metrics"]["final_accuracy"]["n"], 2)
        self.assertIn("**failed** — boom", summary["markdown"])
        self.assertTrue(
            any(
                "did not complete" in b
                for b in summary["validation"]["publication_blockers"]
            )
        )

    def test_official_mode_is_refused_before_anything_is_written(self):
        with self.assertRaises(ProvenanceError):
            go(spec(mode="official"), self.out)
        self.assertFalse(self.out.exists())
        self.assertEqual(FakeAdapter.calls, [])

    def test_provenance_error_inside_a_run_is_not_swallowed(self):
        class Refusing(FakeAdapter):
            def run_sequence(self, *a, **k):
                raise ProvenanceError("refused")

        with self.assertRaises(ProvenanceError):
            go(spec(), self.out, adapter_factory=lambda c, m: Refusing(c, m))


class ResumeWithoutMixing(Base):
    def test_resume_reuses_completed_runs_without_calling_the_algorithm(self):
        go(spec(), self.out)
        FakeAdapter.calls.clear()
        before = self.records()
        outcome = go(spec(), self.out, resume=True)
        self.assertEqual(FakeAdapter.calls, [])
        self.assertEqual(len(outcome["reused"]), 6)
        self.assertEqual(self.records(), before)

    def test_resume_runs_only_what_is_missing(self):
        go(spec(), self.out)
        (self.out / "runs" / "b__seed2.json").unlink()
        FakeAdapter.calls.clear()
        go(spec(), self.out, resume=True)
        self.assertEqual(FakeAdapter.calls, [("fake_b", 0.7, 2)])

    def test_failed_runs_are_kept_unless_retry_is_requested(self):
        s = spec([("bad", {"algorithm": "x", "behavior": "raise_seed1"})])
        go(s, self.out)
        FakeAdapter.calls.clear()
        outcome = go(s, self.out, resume=True)
        self.assertEqual(FakeAdapter.calls, [])
        self.assertEqual(outcome["failed"], ["bad seed 1"])
        self.assertEqual(self.records()["bad__seed1.json"]["status"], "failed")

    def test_retry_failed_reruns_archives_old_attempt_and_increments(self):
        s = spec([("bad", {"algorithm": "x", "behavior": "raise_seed1"})])
        go(s, self.out)
        FakeAdapter.calls.clear()
        go(s, self.out, resume=True, retry_failed=True)
        self.assertEqual(FakeAdapter.calls, [("x", None, 1)])  # only the failed seed
        recs = self.records()
        self.assertEqual(recs["bad__seed1.json"]["attempt"], 2)
        self.assertEqual(recs["bad__seed1.attempt1.json"]["attempt"], 1)
        # archived attempts do not count as runs in the summary
        summary = summarize_directory(self.out)
        self.assertEqual(summary["aggregate"]["algorithms"]["bad"]["n_failed"], 1)
        self.assertEqual(summary["problems"], [])

    def test_changed_configuration_is_refused(self):
        go(spec(), self.out)
        other = spec(
            [
                ("a", {"algorithm": "fake_a", "skill": 0.9}),
                ("b", {"algorithm": "fake_b", "skill": 0.7}),
            ]
        )
        with self.assertRaises(ExperimentMismatch):
            go(other, self.out, resume=True)

    def test_added_algorithm_changes_the_fingerprint_and_is_refused(self):
        # existing records' run keys are unaffected by an extra algorithm, so only the manifest
        # fingerprint can catch this
        go(spec(), self.out)
        three = spec(
            [
                ("a", {"algorithm": "fake_a", "skill": 0.8}),
                ("b", {"algorithm": "fake_b", "skill": 0.7}),
                ("c", {"algorithm": "fake_c"}),
            ]
        )
        with self.assertRaises(ExperimentMismatch):
            go(three, self.out, resume=True)

    def test_a_record_with_a_different_run_key_is_refused_even_if_the_manifest_matches(
        self,
    ):
        go(spec(), self.out)
        other = spec(
            [
                ("a", {"algorithm": "fake_a", "skill": 0.9}),
                ("b", {"algorithm": "fake_b", "skill": 0.7}),
            ]
        )
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["fingerprint"] = other.fingerprint()  # defeat the first guard
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ExperimentMismatch, "run key differs"):
            go(other, self.out, resume=True)

    def test_changed_seeds_or_protocol_are_refused(self):
        go(spec(), self.out)
        with self.assertRaises(ExperimentMismatch):
            go(spec(seeds=(0, 1)), self.out, resume=True)

    def test_existing_directory_needs_explicit_resume(self):
        go(spec(), self.out)
        with self.assertRaisesRegex(ExperimentError, "--resume"):
            go(spec(), self.out)

    def test_nonempty_directory_without_manifest_is_refused(self):
        self.out.mkdir()
        (self.out / "stray.txt").write_text("x")
        with self.assertRaises(ExperimentError):
            go(spec(), self.out)

    def test_records_from_another_commit_are_not_mixed(self):
        go(spec(), self.out)
        other = dict(
            runner.benchmark_provenance(), commit="f" * 40, commit_source="local_git"
        )
        with (
            mock.patch.object(runner, "benchmark_provenance", return_value=other),
            self.assertRaisesRegex(
                ExperimentMismatch, "different core/benchmark commit"
            ),
        ):
            go(spec(), self.out, resume=True)

    def test_corrupt_existing_record_is_an_error_not_silently_rerun(self):
        go(spec(), self.out)
        (self.out / "runs" / "a__seed0.json").write_text("{ not json")
        with self.assertRaises(results.RecordError):
            go(spec(), self.out, resume=True)

    def test_state_is_isolated_between_runs(self):
        # the same algorithm and seed twice gives identical matrices: nothing carries over
        go(spec([("a", {"algorithm": "fake_a"})], seeds=(0,)), self.out)
        first = self.records()["a__seed0.json"]["metrics"]["accuracy_matrix"]
        out2 = Path(self.tmp.name) / "exp2"
        go(spec([("a", {"algorithm": "fake_a"})], seeds=(0,)), out2)
        second = json.loads((out2 / "runs" / "a__seed0.json").read_text())["metrics"][
            "accuracy_matrix"
        ]
        self.assertEqual(first, second)


class Reproducibility(Base):
    def test_deterministic_algorithm_reproduces_exactly(self):
        go(
            spec([("a", {"algorithm": "fake_a"})], seeds=(0,)),
            self.out,
            reproduce_check=True,
        )
        item = json.loads((self.out / "reproducibility" / "a__seed0.json").read_text())
        self.assertTrue(item["exact_match"])
        self.assertEqual(item["max_abs_difference"], 0.0)
        self.assertEqual(
            self.records()["a__seed0.json"]["attempt"], 1
        )  # the check did not replace the record

    def test_nondeterminism_is_reported_and_blocks_publication(self):
        go(
            spec(
                [("n", {"algorithm": "n", "behavior": "nondeterministic"})], seeds=(0,)
            ),
            self.out,
            reproduce_check=True,
        )
        item = json.loads((self.out / "reproducibility" / "n__seed0.json").read_text())
        self.assertFalse(item["exact_match"])
        self.assertGreater(item["max_abs_difference"], 0)
        blockers = summarize_directory(self.out)["validation"]["publication_blockers"]
        self.assertTrue(any("did not reproduce" in b for b in blockers))


class ProcessIsolation(Base):
    """Crash and timeout handling with a stand-in worker command (no core needed)."""

    def setUp(self):
        super().setUp()
        from helpers import fake_mnist_protocol

        self.data = Path(self.tmp.name) / "data"
        self.data.mkdir()
        self.protocol = fake_mnist_protocol(self.data, seeds=(0,))

    def process_spec(self, **kw):
        return ExperimentSpec(
            name="p",
            protocol=self.protocol,
            algorithms=(AlgorithmEntry("a", {"algorithm": "fake_a"}),),
            seeds=(0,),
            isolation="process",
            **kw,
        )

    def run_with(self, argv, **kw):
        s = self.process_spec(**kw)
        with mock.patch.object(runner, "worker_command", lambda t, o: argv):
            return run_experiment(s, self.out, data_dir=self.data, log=lambda *_: None)

    def test_worker_crash_is_a_crashed_record(self):
        import sys

        outcome = self.run_with(
            [sys.executable, "-c", "import sys; sys.stderr.write('segv'); sys.exit(7)"]
        )
        rec = self.records()["a__seed0.json"]
        self.assertEqual(rec["status"], "crashed")
        self.assertIn("code 7", rec["error"]["message"])
        self.assertIn("segv", rec["error"]["traceback"])
        self.assertEqual(outcome["failed"], ["a seed 0"])
        self.assertEqual(results.validate_record(rec), [])

    def test_timeout_is_a_timeout_record(self):
        import sys

        self.run_with(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout_seconds=1
        )
        rec = self.records()["a__seed0.json"]
        self.assertEqual(rec["status"], "timeout")
        self.assertLess(rec["resources"]["wall_time_seconds"], 10)

    def test_worker_provenance_refusal_aborts(self):
        import sys

        with self.assertRaises(ProvenanceError):
            self.run_with(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('official run refused'); sys.exit(3)",
                ]
            )

    def test_invalid_data_is_refused_before_any_worker_starts(self):
        (self.data / "t10k-labels-idx1-ubyte.gz").write_bytes(b"tampered")
        from polynomial_cl_bench.datasets import DatasetVerificationError

        with self.assertRaises(DatasetVerificationError):
            self.run_with(["false"])

    def test_timeout_requires_process_isolation(self):
        with self.assertRaises(ExperimentError):
            spec(timeout_seconds=5)


class SpecLoading(Base):
    def test_protocol_roundtrip_keeps_fingerprint(self):
        from polynomial_cl_bench.protocol import Protocol

        p = small_protocol()
        self.assertEqual(Protocol.from_dict(p.to_dict()).fingerprint(), p.fingerprint())

    def test_fingerprint_ignores_name_and_timeout_but_not_config(self):
        a = spec(name="x")
        self.assertEqual(a.fingerprint(), spec(name="y").fingerprint())
        self.assertNotEqual(
            a.fingerprint(),
            spec(
                [
                    ("a", {"algorithm": "fake_a", "skill": 0.1}),
                    ("b", {"algorithm": "fake_b", "skill": 0.7}),
                ]
            ).fingerprint(),
        )

    def test_bad_labels_seeds_and_duplicates_are_refused(self):
        for bad in (
            lambda: spec([("a b", {"algorithm": "x"})]),
            lambda: spec([("a", {"algorithm": "x"}), ("a", {"algorithm": "y"})]),
            lambda: spec(seeds=(0, 0)),
            lambda: spec(mode="nope"),
            lambda: (
                spec(algos=[])
                if False
                else ExperimentSpec("e", small_protocol(), (), (0,))
            ),
        ):
            with self.assertRaises(ExperimentError):
                bad()


class TuningResultUse(Base):
    def tuning(self, **override):
        p = small_protocol(seeds=(0, 1, 2))
        base = {
            "algorithm": "er_mlp",
            "criterion": "c",
            "selection_seeds": [100, 101],
            "protocol_fingerprint": p.fingerprint(),
            "validation_dataset": {"evaluation_split": "validation"},
            "runs": [{"evaluation_split": "validation"}],
            "selected_index": 3,
            "selected_config": {"hidden": [64], "learning_rate": 0.3},
            "context": {
                "grid_file": "g.json",
                "grid_sha256": "ab",
                "run_mode": "official",
                "official_eligible": True,
            },
        }
        base.update(override)
        path = Path(self.tmp.name) / "tuning.json"
        path.write_text(json.dumps(base))
        return path, p

    def apply(self, path, p, config=None, seeds=(0, 1, 2)):
        return runner.apply_tuning_result(
            config or {"algorithm": "er_mlp", "role": "primary", "replay": True},
            path,
            p,
            seeds,
        )

    def test_selected_values_and_provenance_are_taken_from_the_tuning_file(self):
        path, p = self.tuning()
        cfg = self.apply(path, p)
        self.assertEqual((cfg["hidden"], cfg["learning_rate"]), ([64], 0.3))
        prov = cfg["hyperparameter_provenance"]
        self.assertEqual(prov["selection"], "validation_tuning")
        self.assertEqual(len(prov["tuning_result_sha256"]), 64)
        self.assertEqual(prov["evaluation_split"], "validation")

    def test_refusals(self):
        cases = {
            "protocol": {"protocol_fingerprint": "0" * 64},
            "algorithm": {"algorithm": "polynomial_cl"},
            "none selected": {"selected_config": None, "selected_index": None},
            "test-split": {"runs": [{"evaluation_split": "test"}]},
            "overlap": {"selection_seeds": [0, 101]},
        }
        for name, change in cases.items():
            path, p = self.tuning(**change)
            with self.assertRaises(ExperimentError, msg=name):
                self.apply(path, p)

    def test_base_config_may_not_preset_tunable_keys(self):
        path, p = self.tuning()
        with self.assertRaises(ExperimentError):
            self.apply(path, p, config={"algorithm": "er_mlp", "learning_rate": 0.1})

    def test_missing_tuning_file_says_to_tune_first(self):
        with self.assertRaisesRegex(ExperimentError, "run the validation tuning first"):
            self.apply(Path(self.tmp.name) / "none.json", small_protocol())

    def test_spec_from_file_resolves_relative_paths(self):
        _, p = self.tuning()
        d = Path(self.tmp.name)
        (d / "base.json").write_text(
            json.dumps({"algorithm": "er_mlp", "role": "primary", "replay": True})
        )
        (d / "protocol.json").write_text(json.dumps(p.to_dict()))
        (d / "exp.json").write_text(
            json.dumps(
                {
                    "name": "e",
                    "protocol": "protocol.json",
                    "seeds": [0, 1, 2],
                    "algorithms": [
                        {
                            "label": "er",
                            "config": "base.json",
                            "tuning_result": "tuning.json",
                        }
                    ],
                }
            )
        )
        s = ExperimentSpec.from_file(d / "exp.json")
        self.assertEqual(s.algorithms[0].config["hidden"], [64])
        self.assertEqual(s.seeds, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
