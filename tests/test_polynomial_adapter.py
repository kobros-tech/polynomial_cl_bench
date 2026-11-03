import json
import unittest
from unittest import mock

import numpy as np

from helpers import POLYNOMIAL_CONFIG_PATH, small_protocol, synthetic_dataset
from polynomial_cl import metrics as core_metrics
from polynomial_cl.class_incremental import TASK_ORDER
from polynomial_cl_bench import metrics
from polynomial_cl_bench.algorithms import polynomial as module
from polynomial_cl_bench.algorithms.polynomial import AdapterError, PolynomialAdapter
from polynomial_cl_bench.provenance import commit_from_direct_url, core_provenance
from polynomial_cl_bench.datasets import array_fingerprint


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


class SpyRunCondition:
    """Records what the adapter hands to the core, then calls the real function."""

    def __init__(self):
        self.calls = []
        self.real = module.run_condition

    def __call__(self, condition, data, tasks, memory, **kwargs):
        self.calls.append((condition, data, tasks, memory, kwargs))
        return self.real(condition, data, tasks, memory, **kwargs)


class PolynomialAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol()

    def run_adapter(self, adapter=None, seed=0, protocol=None):
        adapter = adapter or PolynomialAdapter()
        return adapter.run_sequence(self.dataset, protocol or self.protocol, seed)

    # --- what is passed to the core ------------------------------------------------------
    def test_core_receives_protocol_values_and_explicit_class_balanced_sampling(self):
        spy = SpyRunCondition()
        with mock.patch.object(module, "run_condition", spy):
            result = self.run_adapter(seed=3)
        (condition, _, _, _, kwargs), = spy.calls
        self.assertEqual(condition, "expanding")
        self.assertEqual(kwargs["strategy"], "ordinary")
        self.assertEqual(kwargs["sampling"], "class_balanced")
        self.assertEqual(kwargs["seed"], 3)
        self.assertEqual(kwargs["epochs"], self.protocol.epochs)
        self.assertEqual(kwargs["batch_size"], self.protocol.batch_size)
        self.assertEqual(kwargs["learning_rate"], 0.1)
        self.assertEqual(result.sampling, "class_balanced")

    def test_shuffled_ablation_is_passed_through_and_recorded(self):
        spy = SpyRunCondition()
        with mock.patch.object(module, "run_condition", spy):
            result = self.run_adapter(protocol=small_protocol(sampling="shuffled"))
        self.assertEqual(spy.calls[0][4]["sampling"], "shuffled")
        self.assertEqual(result.sampling, "shuffled")

    def test_training_indices_only_come_from_the_current_experience(self):
        spy = SpyRunCondition()
        with mock.patch.object(module, "run_condition", spy):
            self.run_adapter()
        _, data, tasks, memory, _ = spy.calls[0]
        self.assertEqual(len(tasks), len(TASK_ORDER))
        for classes, indices in zip(TASK_ORDER, tasks):
            self.assertEqual(set(data.train_y[indices]), set(classes))
        for class_id, indices in memory.items():
            self.assertTrue(np.all(data.train_y[indices] == class_id))
            self.assertLessEqual(len(indices), self.protocol.memory_per_class)

    def test_replay_budget_is_reported(self):
        result = self.run_adapter()
        self.assertEqual(result.replay_examples_used_by_task, [0, 20, 40, 60, 80])
        no_replay = self.run_adapter(PolynomialAdapter(strategy="ordinary_no_replay"))
        self.assertEqual(no_replay.replay_examples_used_by_task, [0] * 5)

    # --- output ----------------------------------------------------------------------------
    def test_matrix_shape_and_nan_pattern(self):
        result = self.run_adapter()
        self.assertEqual(result.accuracy_matrix.shape, (5, 5))
        metrics.validate_matrix(result.accuracy_matrix)  # raises if malformed

    def test_benchmark_metrics_equal_core_reported_values(self):
        result = self.run_adapter()
        core = result.extras["core_reported"]
        m = result.accuracy_matrix
        self.assertAlmostEqual(metrics.final_mean_accuracy(m), core["final_mean_accuracy"], places=12)
        self.assertAlmostEqual(metrics.mean_forgetting(m), core["mean_forgetting"], places=12)
        np.testing.assert_allclose(
            metrics.forgetting_by_experience(m), core["forgetting_by_task"], atol=1e-12
        )
        np.testing.assert_allclose(
            metrics.current_accuracy(m), core["current_accuracy_by_task"], atol=1e-12
        )
        np.testing.assert_allclose(
            metrics.mean_accuracy_by_step(m), core["mean_accuracy_by_task"], atol=1e-12
        )
        np.testing.assert_allclose(
            metrics.forgetting_by_experience(m), core_metrics.forgetting(m, 4), atol=1e-12
        )

    def test_unavailable_resources_are_none_not_zero(self):
        r = self.run_adapter().resources
        for key in ("training_time_seconds", "inference_time_seconds", "peak_memory_mb"):
            self.assertIsNone(r[key])
        self.assertGreater(r["wall_time_seconds_including_evaluation"], 0)
        self.assertEqual(r["n_parameters"], 969 * 10)

    def test_algorithm_record_is_json_serialisable_and_complete(self):
        info = self.run_adapter().algorithm
        json.dumps(info)
        self.assertEqual((info["condition"], info["strategy"]), ("expanding", "ordinary"))
        self.assertEqual(info["configuration"]["learning_rate"], 0.1)
        self.assertIn("commit", info["core"])
        self.assertEqual(
            info["hyperparameter_provenance"], {}
        )  # default constructor claims nothing

    def test_shipped_config_declares_unselected_defaults(self):
        adapter = PolynomialAdapter.from_file(POLYNOMIAL_CONFIG_PATH)
        info = adapter.describe()
        self.assertEqual((info["condition"], info["strategy"]), ("expanding", "ordinary"))
        self.assertEqual(
            info["hyperparameter_provenance"]["selection"],
            "core_default_not_validation_selected",
        )

    # --- determinism and isolation -------------------------------------------------------
    def test_fixed_seed_is_deterministic(self):
        a = self.run_adapter(seed=0)
        b = self.run_adapter(seed=0)
        self.assertTrue(same(a.accuracy_matrix, b.accuracy_matrix))

    def test_different_seeds_differ(self):
        a = self.run_adapter(seed=0)
        b = self.run_adapter(seed=1)
        self.assertFalse(same(a.accuracy_matrix, b.accuracy_matrix))

    def test_state_is_isolated_between_runs_and_instances(self):
        shared = PolynomialAdapter()
        first = self.run_adapter(shared, seed=0)
        self.run_adapter(shared, seed=1)  # an intervening run must not leak into seed 0
        again = self.run_adapter(shared, seed=0)
        fresh = self.run_adapter(PolynomialAdapter(), seed=0)
        self.assertTrue(same(first.accuracy_matrix, again.accuracy_matrix))
        self.assertTrue(same(first.accuracy_matrix, fresh.accuracy_matrix))

    def test_shared_data_is_not_modified(self):
        adapter = PolynomialAdapter()
        self.run_adapter(adapter)
        data = adapter._features[1]
        before = array_fingerprint(data.train_x, data.train_y, data.test_x, data.test_y)
        self.run_adapter(adapter, seed=1)
        self.assertEqual(
            before, array_fingerprint(data.train_x, data.train_y, data.test_x, data.test_y)
        )
        self.assertEqual(
            self.dataset.fingerprint,
            array_fingerprint(
                self.dataset.train_x, self.dataset.train_y,
                self.dataset.test_x, self.dataset.test_y,
            ),
        )

    # --- refusals ---------------------------------------------------------------------------
    def test_unsupported_task_order_is_refused(self):
        order = ((1, 0), (2, 3), (4, 5), (6, 7), (8, 9))
        with self.assertRaisesRegex(AdapterError, "task order"):
            self.run_adapter(protocol=small_protocol(task_order=order))

    def test_unknown_condition_or_strategy_is_refused(self):
        with self.assertRaises(AdapterError):
            PolynomialAdapter(condition="nonsense")
        with self.assertRaises(AdapterError):
            PolynomialAdapter(strategy="nonsense")

    # --- provenance ---------------------------------------------------------------------------
    def test_commit_parsed_from_pip_direct_url(self):
        url = {"url": "https://example.invalid/r.git", "vcs_info": {"vcs": "git", "commit_id": "ab" * 20}}
        self.assertEqual(commit_from_direct_url(url), "ab" * 20)
        self.assertIsNone(commit_from_direct_url({"url": "file:///x", "dir_info": {}}))
        self.assertIsNone(commit_from_direct_url(None))

    def test_core_provenance_reports_a_commit_in_this_environment(self):
        info = core_provenance()
        self.assertEqual(info["version"], "0.1.0")
        self.assertRegex(info["commit"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
