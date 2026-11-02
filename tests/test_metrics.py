import unittest

import numpy as np

from polynomial_cl import metrics as core_metrics
from polynomial_cl_bench import metrics
from polynomial_cl_bench.metrics import MatrixError

N = np.nan
R3 = [[0.9, N, N], [0.7, 0.8, N], [0.6, 0.5, 0.9]]


class MetricTests(unittest.TestCase):
    def test_hand_computed_values(self):
        self.assertAlmostEqual(metrics.final_mean_accuracy(R3), (0.6 + 0.5 + 0.9) / 3)
        self.assertEqual(metrics.accuracy_per_experience(R3), [0.6, 0.5, 0.9])
        self.assertEqual(metrics.current_accuracy(R3), [0.9, 0.8, 0.9])
        np.testing.assert_allclose(
            metrics.mean_accuracy_by_step(R3), [0.9, 0.75, (0.6 + 0.5 + 0.9) / 3]
        )
        np.testing.assert_allclose(metrics.forgetting_by_experience(R3), [0.3, 0.3, 0.0])
        self.assertAlmostEqual(metrics.mean_forgetting(R3), 0.3)

    def test_forgetting_uses_best_ever_accuracy_not_diagonal(self):
        r = [[0.9, N, N], [0.95, 0.8, N], [0.6, 0.5, 0.9]]
        self.assertAlmostEqual(metrics.forgetting_by_experience(r)[0], 0.35)

    def test_last_experience_excluded_from_mean_forgetting(self):
        r = [[0.9, N], [0.9, 0.1]]
        self.assertEqual(metrics.mean_forgetting(r), 0.0)  # only experience 0 counts
        self.assertEqual(metrics.forgetting_by_experience(r), [0.0, 0.0])

    def test_matches_core_definitions_on_random_matrices(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            t = int(rng.integers(2, 8))
            r = np.full((t, t), np.nan)
            for i in range(t):
                r[i, : i + 1] = rng.random(i + 1)
            core = core_metrics.forgetting(r, t - 1)
            np.testing.assert_allclose(metrics.forgetting_by_experience(r), core, atol=1e-15)
            self.assertAlmostEqual(
                metrics.mean_forgetting(r), float(np.mean(core[:-1])), places=15
            )

    def test_malformed_matrices_rejected(self):
        for bad in (
            [[0.5, 0.5], [0.5, 0.5]],  # upper triangle not NaN
            [[0.5, N], [N, 0.5]],  # lower triangle missing
            [[0.5, N, N], [0.5, 0.5, N]],  # not square
            [[1.2, N], [0.5, 0.5]],  # out of range
            [0.5],  # not 2-D
        ):
            with self.assertRaises(MatrixError):
                metrics.final_mean_accuracy(bad)
        with self.assertRaises(MatrixError):
            metrics.mean_forgetting([[0.5]])


if __name__ == "__main__":
    unittest.main()
