# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Full Split MNIST runs against the real data. About 30-60 s: opt in with BENCH_SLOW=1.

The MNIST directory defaults to ../data/mnist; override with BENCH_MNIST_DIR.
The reference numbers below are polynomial_cl's own committed records (core commit 9e15732);
the tolerance allows BLAS differences between machines.
"""

import os
import unittest
from pathlib import Path

import numpy as np
from helpers import (
    PER_DEGREE_CONFIG_PATH,
    POLYNOMIAL_CONFIG_PATH,
    ROOT,
    shipped_protocol,
)

from polynomial_cl_bench import metrics
from polynomial_cl_bench.algorithms.polynomial import PolynomialAdapter
from polynomial_cl_bench.datasets import load_split_mnist

DATA_DIR = Path(os.environ.get("BENCH_MNIST_DIR", ROOT / "data" / "mnist"))


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


@unittest.skipUnless(os.environ.get("BENCH_SLOW") == "1", "set BENCH_SLOW=1 to run")
class FullSplitMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = shipped_protocol()
        cls.dataset = load_split_mnist(
            cls.protocol, DATA_DIR
        )  # verifies the real file hashes

    def check_against_core(self, result):
        core, m = result.extras["core_reported"], result.accuracy_matrix
        self.assertAlmostEqual(
            metrics.final_mean_accuracy(m), core["final_mean_accuracy"], places=12
        )
        self.assertAlmostEqual(
            metrics.mean_forgetting(m), core["mean_forgetting"], places=12
        )

    def test_loaded_data_shapes(self):
        self.assertEqual(self.dataset.train_x.shape, (60000, 16))
        self.assertEqual(self.dataset.test_x.shape, (10000, 16))
        self.assertEqual(self.dataset.source_hashes, self.protocol.dataset["files"])

    def test_primary_reference_is_expanding_ordinary_class_balanced(self):
        adapter = PolynomialAdapter.from_file(POLYNOMIAL_CONFIG_PATH)
        # The configuration under test, asserted explicitly rather than assumed.
        info = adapter.describe()
        self.assertEqual(info["role"], "primary_reference")
        self.assertEqual(
            (info["condition"], info["strategy"]), ("expanding", "ordinary")
        )
        self.assertEqual(info["configuration"]["learning_rate"], 0.1)
        self.assertEqual(
            info["configuration"]["feature_power"], 0.0
        )  # was never asserted before
        self.assertEqual(self.protocol.sampling, "class_balanced")
        self.assertEqual(self.protocol.projection_seed, 0)

        first = adapter.run_sequence(self.dataset, self.protocol, seed=0)
        second = adapter.run_sequence(self.dataset, self.protocol, seed=0)
        self.assertTrue(same(first.accuracy_matrix, second.accuracy_matrix))
        self.check_against_core(first)

        c = first.run_config
        self.assertEqual(
            (
                c["condition"],
                c["strategy"],
                c["sampling"],
                c["learning_rate"],
                c["feature_power"],
            ),
            ("expanding", "ordinary", "class_balanced", 0.1, 0.0),
        )
        self.assertEqual(
            (c["epochs"], c["batch_size"], c["memory_per_class"]), (3, 128, 200)
        )
        self.assertEqual(c["dataset"]["projection_seed"], 0)
        self.assertEqual(first.replay_examples_used_by_task, [0, 400, 800, 1200, 1600])

        # core: results/per_degree/sampling_e3_lr0.1.json, class_balanced, seed 0
        m = first.accuracy_matrix
        self.assertAlmostEqual(
            metrics.final_mean_accuracy(m), 0.6770214422462445, delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mean_forgetting(m), 0.0708057983859818, delta=1e-6
        )
        print(
            "\nprimary_reference seed 0: acc",
            metrics.final_mean_accuracy(m),
            "forgetting",
            metrics.mean_forgetting(m),
            "wall_s",
            first.resources["wall_time_seconds_including_evaluation"],
            "core",
            first.algorithm["core"],
        )

    def test_experimental_per_degree_vector_matches_core_record(self):
        adapter = PolynomialAdapter.from_file(PER_DEGREE_CONFIG_PATH)
        self.assertEqual(adapter.describe()["role"], "experimental")
        result = adapter.run_sequence(self.dataset, self.protocol, seed=3)
        self.check_against_core(result)
        self.assertEqual(result.run_config["learning_rate"], [0.75, 1.5, 1.5, 6.0])
        m = result.accuracy_matrix
        # core: results/per_degree/cell_ordinary_class_balanced.json, 'combined', seed 3
        self.assertAlmostEqual(
            metrics.final_mean_accuracy(m), 0.7734281931972611, delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mean_forgetting(m), 0.09086581818480258, delta=1e-6
        )
        print(
            "\nexperimental per-degree seed 3: acc",
            metrics.final_mean_accuracy(m),
            "forgetting",
            metrics.mean_forgetting(m),
        )


if __name__ == "__main__":
    unittest.main()
