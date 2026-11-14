# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Real-core integration on synthetic data: the benchmark driving the actual ``polynomial_cl``.

What this file establishes, and what it does not:

* It runs the benchmark's runner, the real worker subprocess, the polynomial adapter and the
  core's own ``run_condition`` and MNIST loader (``download=False``, sha256 verified) on tiny
  synthetic IDX archives. That is *real-core integration*.
* It does **not** show that the core was installed from the pin declared in ``pyproject.toml``
  (a local editable checkout passes). That is a separate property, checked by
  ``CoreInstallation.test_installed_core_is_the_declared_pin``, which CI enables with
  ``BENCH_REQUIRE_PINNED_CORE=1`` after a clean ``pip install``.
* It does not touch real MNIST (``test_full_split_mnist``, opt-in) and says nothing about
  independent reproducibility of any published number.

Other integration tests skip when the core is missing. In CI set ``BENCH_REQUIRE_CORE=1`` so a
missing or incompatible core fails here instead of turning those tests into silent skips.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from helpers import PROTOCOL_PATH, fake_mnist_protocol

from polynomial_cl_bench import provenance, results
from polynomial_cl_bench.runner import AlgorithmEntry, ExperimentSpec, run_experiment
from polynomial_cl_bench.summarize import summarize_directory
from polynomial_cl_bench.algorithms.polynomial import CORE_REPORTS_TIMINGS

try:
    import polynomial_cl.continual as core_continual
    import polynomial_cl.data as core_data

    HAVE_CORE = True
except ImportError:
    HAVE_CORE = False

REQUIRE_CORE = os.environ.get("BENCH_REQUIRE_CORE") == "1"
REQUIRE_PINNED_CORE = os.environ.get("BENCH_REQUIRE_PINNED_CORE") == "1"

# Names the benchmark imports from the core (adapter, plan, datasets, tools).
CORE_CONTINUAL_NAMES = (
    "TASK_ORDER",
    "holdout_split",
    "make_feature_data",
    "make_task_plan",
    "run_condition",
)

ER = {
    "algorithm": "er_mlp",
    "hidden": [8],
    "learning_rate": 0.1,
    "replay": True,
    "role": "primary",
}


def polynomial_reference_config() -> dict:
    path = Path(PROTOCOL_PATH).parent / "polynomial_expanding_ordinary.json"
    return json.loads(path.read_text())


class CoreInstallation(unittest.TestCase):
    def test_core_is_importable_when_required(self):
        if not HAVE_CORE:
            if REQUIRE_CORE:
                self.fail(
                    "BENCH_REQUIRE_CORE=1 but polynomial_cl.continual is not importable"
                )
            self.skipTest("polynomial_cl is not installed")

    @unittest.skipUnless(HAVE_CORE, "polynomial_cl is not installed")
    def test_core_exposes_the_api_the_benchmark_uses(self):
        missing = [n for n in CORE_CONTINUAL_NAMES if not hasattr(core_continual, n)]
        self.assertEqual(missing, [], "core API drift: polynomial_cl.continual")
        self.assertTrue(hasattr(core_data, "load_projected_mnist"))
        from polynomial_cl.continual.sampling import SAMPLING_POLICIES

        self.assertEqual(set(SAMPLING_POLICIES), {"shuffled", "class_balanced"})

    @unittest.skipUnless(HAVE_CORE, "polynomial_cl is not installed")
    @unittest.skipUnless(
        REQUIRE_PINNED_CORE, "set BENCH_REQUIRE_PINNED_CORE=1 after a clean install"
    )
    def test_installed_core_is_the_declared_pin(self):
        core = provenance.core_provenance()
        pin = provenance.pinned_core_commit()
        self.assertIsNotNone(
            pin, "pyproject.toml declares no pinned polynomial-cl commit"
        )
        self.assertEqual(len(pin), 40, "the pin must be a full immutable commit SHA")
        self.assertEqual(core["commit"], pin)
        self.assertEqual(core["commit_source"], "pip_vcs_install")
        self.assertTrue(core["remote_verified"])
        self.assertTrue(core["matches_pin"])


@unittest.skipUnless(HAVE_CORE, "polynomial_cl is not installed")
class RealCoreThroughTheRunner(unittest.TestCase):
    """Polynomial reference + ER-MLP, fresh processes, shared plan, reproducibility repeat."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "data").mkdir()
        spec = ExperimentSpec(
            name="real_core_synthetic",
            protocol=fake_mnist_protocol(root / "data", seeds=(0, 1)),
            algorithms=(
                AlgorithmEntry("poly", polynomial_reference_config()),
                AlgorithmEntry("er", ER),
            ),
            seeds=(0, 1),
            isolation="process",
        )
        cls.out = root / "out"
        run_experiment(
            spec,
            cls.out,
            data_dir=root / "data",
            reproduce_check=True,
            log=lambda *_: None,
        )
        cls.records = {
            p.stem: json.loads(p.read_text())
            for p in sorted((cls.out / "runs").glob("*.json"))
        }
        cls.summary = summarize_directory(cls.out, write=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_run_completed_and_records_validate(self):
        self.assertEqual(
            sorted(self.records),
            ["er__seed0", "er__seed1", "poly__seed0", "poly__seed1"],
        )
        for name, record in self.records.items():
            self.assertEqual(record["status"], "completed", (name, record["error"]))
            self.assertEqual(results.validate_record(record), [], name)
            self.assertEqual(record["resources"]["isolation"], "process")

    def test_polynomial_results_come_from_the_installed_core(self):
        core = provenance.core_provenance()
        for seed in (0, 1):
            record = self.records[f"poly__seed{seed}"]
            recorded = record["algorithm"]["core"]
            self.assertEqual(recorded["commit"], core["commit"])
            self.assertEqual(recorded["version"], core["version"])
            self.assertTrue(record["checks"]["core_reported_cross_check"]["agrees"])
            # Null (never zero) when the installed core cannot separate training time.
            training = record["resources"]["training_time_seconds"]
            if CORE_REPORTS_TIMINGS:
                self.assertGreater(training, 0)
            else:
                self.assertIsNone(training)
            self.assertEqual(record["dataset"]["evaluation_split"], "test")

    def test_algorithms_share_the_plan_and_dataset_at_each_seed(self):
        for seed in (0, 1):
            poly, er = (
                self.records[f"poly__seed{seed}"],
                self.records[f"er__seed{seed}"],
            )
            self.assertEqual(poly["plan_fingerprint"], er["plan_fingerprint"])
            self.assertEqual(
                poly["dataset"]["fingerprint"], er["dataset"]["fingerprint"]
            )
            self.assertEqual(
                poly["protocol"]["fingerprint"], er["protocol"]["fingerprint"]
            )

    def test_reproducibility_repeat_is_exact_for_both_algorithms(self):
        repeats = self.summary["reproducibility"]
        self.assertEqual({r["label"] for r in repeats}, {"poly", "er"})
        for repeat in repeats:
            self.assertEqual(repeat["rerun_status"], "completed")
            self.assertTrue(repeat["exact_match"], repeat)

    def test_experiment_validation_is_clean_but_synthetic_data_is_never_publishable(
        self,
    ):
        validation = self.summary["validation"]
        self.assertTrue(validation["valid"], validation["problems"])
        self.assertEqual(validation["problems"], [])
        self.assertFalse(validation["publishable"])
        self.assertTrue(
            any("official" in blocker for blocker in validation["publication_blockers"])
        )


if __name__ == "__main__":
    unittest.main()
