# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Every shipped configuration validates, and the tuned experiment fails loudly without tuning."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from polynomial_cl_bench.protocol import Protocol
from polynomial_cl_bench.runner import ExperimentError, ExperimentSpec
from polynomial_cl_bench.tune import load_grid

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class ShippedConfigs(unittest.TestCase):
    def test_protocol_validates(self):
        protocol = Protocol.from_file(CONFIGS / "split_mnist_cil_v1.json")
        self.assertEqual(len(protocol.fingerprint()), 64)

    def test_grids_load_and_are_approved(self):
        for name in ("er_mlp_grid.json", "polynomial_grid.json"):
            grid = load_grid(CONFIGS / name)
            self.assertEqual(grid.approval_status, "approved", name)

    def test_untuned_reference_experiment_loads(self):
        spec = ExperimentSpec.from_file(
            CONFIGS / "experiment_polynomial_reference.json"
        )
        self.assertEqual(spec.labels, ["polynomial_reference"])

    def test_tuned_experiment_refuses_without_tuning_results_and_says_what_to_do(self):
        if (ROOT / "results" / "tuning_polynomial.json").exists():
            self.skipTest("real tuning results are present")
        with self.assertRaises(ExperimentError) as caught:
            ExperimentSpec.from_file(CONFIGS / "experiment_tuned_comparison.json")
        self.assertIn("run the validation tuning first", str(caught.exception))

    def test_every_algorithm_config_referenced_by_an_experiment_exists(self):
        for path in CONFIGS.glob("experiment_*.json"):
            for entry in json.loads(path.read_text())["algorithms"]:
                self.assertTrue(
                    (CONFIGS / entry["config"]).is_file(), (path.name, entry)
                )

    def test_experiment_fingerprint_changes_with_a_config_value(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in CONFIGS.glob("*.json"):
                shutil.copy(f, d / f.name)
            a = ExperimentSpec.from_file(d / "experiment_polynomial_reference.json")
            cfg = json.loads((d / "polynomial_expanding_ordinary.json").read_text())
            cfg["learning_rate"] = cfg["learning_rate"] * 2
            (d / "polynomial_expanding_ordinary.json").write_text(json.dumps(cfg))
            b = ExperimentSpec.from_file(d / "experiment_polynomial_reference.json")
            self.assertNotEqual(a.fingerprint(), b.fingerprint())


if __name__ == "__main__":
    unittest.main()
