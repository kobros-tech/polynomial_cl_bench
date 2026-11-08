# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Runner against the real adapters and the real worker process. Needs the pinned core.

Skipped when ``polynomial_cl`` is not importable. Uses tiny fake MNIST archives and a tiny
budget, so it is fast; full Split MNIST stays in ``test_full_split_mnist``.
"""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import polynomial_cl.continual  # noqa: F401

    HAVE_CORE = True
except ImportError:
    HAVE_CORE = False

from helpers import fake_mnist_protocol, synthetic_dataset

from polynomial_cl_bench import results
from polynomial_cl_bench.runner import AlgorithmEntry, ExperimentSpec, run_experiment
from polynomial_cl_bench.summarize import summarize_directory
from polynomial_cl_bench.algorithms.polynomial import CORE_REPORTS_TIMINGS

ER = {
    "algorithm": "er_mlp",
    "hidden": [8],
    "learning_rate": 0.1,
    "replay": True,
    "role": "primary",
}
ER2 = {**ER, "hidden": [16]}


@unittest.skipUnless(HAVE_CORE, "polynomial_cl is not installed")
class ProcessIsolatedRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        self.protocol = fake_mnist_protocol(self.root / "data", seeds=(0, 1))

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, isolation):
        return ExperimentSpec(
            name="itg",
            protocol=self.protocol,
            algorithms=(
                AlgorithmEntry("er8", ER),
                AlgorithmEntry("er16", ER2),
            ),
            seeds=(0, 1),
            isolation=isolation,
        )

    def execute(self, isolation, out):
        run_experiment(
            self.spec(isolation),
            self.root / out,
            data_dir=self.root / "data",
            log=lambda *_: None,
        )
        return {
            p.name: json.loads(p.read_text())
            for p in sorted((self.root / out / "runs").glob("*.json"))
        }

    def test_two_algorithms_two_seeds_in_fresh_processes(self):
        recs = self.execute("process", "proc")
        self.assertEqual(len(recs), 4)
        for name, r in recs.items():
            self.assertEqual(r["status"], "completed", (name, r["error"]))
            self.assertEqual(results.validate_record(r), [])
            self.assertEqual(r["resources"]["isolation"], "process")
            self.assertIsNotNone(r["resources"]["training_time_seconds"])
            self.assertEqual(
                r["resources"]["n_parameters"],
                16 * (8 if "er8" in name else 16)
                + (8 if "er8" in name else 16) * 10
                + (8 if "er8" in name else 16)
                + 10,
            )
        for seed in (0, 1):
            self.assertEqual(
                recs[f"er8__seed{seed}.json"]["plan_fingerprint"],
                recs[f"er16__seed{seed}.json"]["plan_fingerprint"],
            )
        self.assertEqual(summarize_directory(self.root / "proc")["problems"], [])

    def test_process_isolation_does_not_change_the_numbers(self):
        proc = self.execute("process", "proc")
        spec = self.spec("in_process")
        run_experiment(
            spec, self.root / "inproc", data_dir=self.root / "data", log=lambda *_: None
        )
        for name, rec in proc.items():
            other = json.loads((self.root / "inproc" / "runs" / name).read_text())
            self.assertEqual(
                rec["metrics"]["accuracy_matrix"],
                other["metrics"]["accuracy_matrix"],
                name,
            )
            self.assertEqual(rec["plan_fingerprint"], other["plan_fingerprint"])
            self.assertEqual(
                rec["dataset"]["fingerprint"], other["dataset"]["fingerprint"]
            )

    @unittest.skipUnless(Path("/proc/self/clear_refs").exists(), "Linux only")
    def test_peak_memory_is_measured_per_run_in_process_mode(self):
        recs = self.execute("process", "mem")
        for r in recs.values():
            self.assertIsNotNone(r["resources"]["peak_memory_mb"])
            self.assertGreater(r["resources"]["peak_memory_mb"], 0)

    def test_invalid_config_becomes_a_record_and_the_other_runs_continue(self):
        spec = ExperimentSpec(
            name="itg",
            protocol=self.protocol,
            algorithms=(
                AlgorithmEntry("ok", ER),
                AlgorithmEntry("broken", {**ER, "learning_rate": -1.0}),
            ),
            seeds=(0,),
            isolation="process",
        )
        out = self.root / "fail"
        outcome = run_experiment(
            spec, out, data_dir=self.root / "data", log=lambda *_: None
        )
        self.assertEqual(outcome["failed"], ["broken seed 0"])
        bad = json.loads((out / "runs" / "broken__seed0.json").read_text())
        self.assertEqual(bad["status"], "failed")
        self.assertEqual(bad["error"]["stage"], "configuration")
        self.assertIn("learning_rate", bad["error"]["message"])
        summary = summarize_directory(out)
        self.assertEqual(summary["aggregate"]["algorithms"]["broken"]["n_failed"], 1)

    def test_er_run_reproduces_exactly(self):
        out = self.root / "rep"
        run_experiment(
            self.spec("process"),
            out,
            data_dir=self.root / "data",
            reproduce_check=True,
            log=lambda *_: None,
        )
        items = [
            json.loads(p.read_text()) for p in (out / "reproducibility").glob("*.json")
        ]
        self.assertEqual(len(items), 2)
        self.assertTrue(all(i["exact_match"] for i in items))


@unittest.skipUnless(HAVE_CORE, "polynomial_cl is not installed")
class PolynomialThroughTheRunner(unittest.TestCase):
    """Needs the real core's ``run_condition``; skipped against anything lacking it."""

    def test_polynomial_reference_completes_and_agrees_with_the_core_report(self):
        try:
            from polynomial_cl.continual import run_condition  # noqa: F401
        except ImportError:
            self.skipTest("core has no run_condition")
        from helpers import PROTOCOL_PATH, small_protocol

        config = json.loads(
            (
                Path(PROTOCOL_PATH).parent / "polynomial_expanding_ordinary.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            spec = ExperimentSpec(
                name="poly",
                protocol=small_protocol(seeds=(0,)),
                algorithms=(AlgorithmEntry("poly", config),),
                seeds=(0,),
                isolation="in_process",
            )
            run_experiment(
                spec,
                Path(tmp) / "o",
                dataset_loader=lambda p, d: synthetic_dataset(),
                log=lambda *_: None,
            )
            rec = json.loads(
                (Path(tmp) / "o" / "runs" / "poly__seed0.json").read_text()
            )
            self.assertEqual(rec["status"], "completed", rec["error"])
            self.assertTrue(rec["checks"]["core_reported_cross_check"]["agrees"])
            training = rec["resources"]["training_time_seconds"]
            if (
                CORE_REPORTS_TIMINGS
            ):  # an older core cannot separate it: null, never zero
                self.assertGreater(training, 0)
            else:
                self.assertIsNone(training)
            self.assertIsNotNone(rec["resources"]["wall_time_seconds"])


if __name__ == "__main__":
    unittest.main()
