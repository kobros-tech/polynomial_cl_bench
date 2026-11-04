# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""ER-MLP: gradients, masking, replay exposure, evaluation, determinism, refusals, divergence."""

import copy
import json
import unittest
from unittest import mock

import numpy as np
from helpers import small_protocol, synthetic_dataset
from polynomial_cl.class_incremental import TASK_ORDER
from polynomial_cl.class_incremental.experiment import accuracy_by_task

from polynomial_cl_bench import metrics
from polynomial_cl_bench.algorithms import er_mlp as module
from polynomial_cl_bench.algorithms.base import AdapterError
from polynomial_cl_bench.algorithms.er_mlp import (
    ERMLPAdapter,
    TrainingDiverged,
    init_params,
    loss_and_gradients,
    masked_accuracy_row,
    train_experience,
)
from polynomial_cl_bench.datasets import array_fingerprint
from polynomial_cl_bench.plan import PlanError, make_plan


def adapter(**kw):
    kw.setdefault("hidden", (12,))
    kw.setdefault("learning_rate", 0.1)
    return ERMLPAdapter(**kw)


class Gradients(unittest.TestCase):
    def numeric(self, params, x, y, eligible, eps=1e-6):
        def loss():
            return loss_and_gradients(params, x, y, eligible)[0]

        grads = []
        for array in params.arrays():
            g = np.zeros_like(array)
            for index in np.ndindex(*array.shape):
                original = array[index]
                array[index] = original + eps
                up = loss()
                array[index] = original - eps
                down = loss()
                array[index] = original
                g[index] = (up - down) / (2 * eps)
            grads.append(g)
        return grads

    def check(self, hidden, eligible):
        rng = np.random.default_rng(0)
        params = init_params(hidden, seed=1, dtype=np.float64, n_inputs=5, n_outputs=7)
        for b in params.biases:  # non-zero biases so their gradients are exercised
            b += rng.normal(0, 0.3, b.shape)
        x = rng.normal(size=(9, 5))
        eligible = np.asarray(eligible)
        y = rng.integers(0, len(eligible), size=9)
        _, grad_w, grad_b = loss_and_gradients(params, x, y, eligible)
        numeric = self.numeric(params, x, y, eligible)
        analytic = [*grad_w, *grad_b]
        for a, n in zip(analytic, numeric):
            np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-8)
        ineligible = np.setdiff1d(np.arange(7), eligible)
        self.assertTrue(np.all(grad_w[-1][:, ineligible] == 0))
        self.assertTrue(np.all(grad_b[-1][ineligible] == 0))

    def test_gradients_match_finite_differences(self):
        for hidden in ((), (6,), (6, 4)):
            with self.subTest(hidden=hidden):
                self.check(hidden, [0, 1, 2, 3])

    def test_masked_columns_have_exactly_zero_gradient_and_the_rest_are_exact(self):
        self.check((5,), [1, 4, 6])

    def test_loss_is_the_mean_cross_entropy_over_eligible_columns_only(self):
        params = init_params((), seed=0, dtype=np.float64, n_inputs=3, n_outputs=4)
        params.weights[0][:, 3] = (
            50.0  # a future class with huge logits must not matter
        )
        x = np.array([[1.0, 2.0, -1.0], [0.5, -0.2, 0.3]])
        y = np.array([0, 2])
        loss = loss_and_gradients(params, x, y, np.array([0, 1, 2]))[0]
        logits = x @ params.weights[0][:, :3]
        expected = -np.mean(
            np.log(np.exp(logits)[np.arange(2), y] / np.exp(logits).sum(axis=1))
        )
        self.assertAlmostEqual(loss, expected, places=12)


class Masking(unittest.TestCase):
    def test_future_output_columns_are_never_read_or_updated(self):
        dataset = synthetic_dataset()
        params = init_params((8,), seed=3)
        before = copy.deepcopy(params)
        seen = np.array([0, 1, 2, 3])
        x, y = np.asarray(dataset.train_x), np.asarray(dataset.train_y).astype(np.int64)
        rows = np.flatnonzero(np.isin(y, seen))
        train_experience(
            params,
            x[rows],
            y[rows],
            x[:0],
            y[:0],
            epochs=2,
            learning_rate=0.5,
            batch_size=16,
            seed=0,
            eligible_classes=seen,
            sampling="class_balanced",
        )
        np.testing.assert_array_equal(
            params.weights[-1][:, 4:], before.weights[-1][:, 4:]
        )
        np.testing.assert_array_equal(params.biases[-1][4:], before.biases[-1][4:])
        self.assertFalse(
            np.array_equal(params.weights[-1][:, :4], before.weights[-1][:, :4])
        )

    def test_labels_outside_the_eligible_classes_are_refused(self):
        params = init_params((), seed=0)
        x, y = np.zeros((4, 16), np.float32), np.array([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            train_experience(
                params,
                x,
                y,
                x[:0],
                y[:0],
                epochs=1,
                learning_rate=0.1,
                batch_size=2,
                seed=0,
                eligible_classes=np.array([0, 1]),
                sampling="shuffled",
            )

    def test_prediction_argmax_ignores_future_classes_even_when_they_dominate(self):
        rng = np.random.default_rng(0)
        test_y = np.repeat(np.arange(10), 12)
        logits = rng.normal(size=(120, 10))
        logits[:, 4:] += 50.0  # an unmasked argmax would always pick a future class
        step = 1  # only classes 0-3 are known
        row = masked_accuracy_row(logits, test_y, TASK_ORDER, step)
        eligible = [0, 1, 2, 3]
        predicted = np.asarray(eligible)[np.argmax(logits[:, eligible], axis=1)]
        for j in range(2):
            mask = np.isin(test_y, TASK_ORDER[j])
            self.assertAlmostEqual(row[j], np.mean(predicted[mask] == test_y[mask]))
        self.assertTrue(np.all(np.isnan(row[2:])))
        self.assertTrue(
            np.all(np.argmax(logits, axis=1) >= 4)
        )  # the unmasked rule would score 0

    def test_masked_rule_equals_the_cores_accuracy_by_task_for_a_linear_head(self):
        rng = np.random.default_rng(1)
        weights = rng.normal(size=(16, 10))
        x, test_y = rng.normal(size=(200, 16)), rng.integers(0, 10, 200)
        for step in range(5):
            np.testing.assert_allclose(
                masked_accuracy_row(x @ weights, test_y, TASK_ORDER, step),
                accuracy_by_task(weights, x, test_y, step + 1),
                atol=1e-12,
                equal_nan=True,
            )


class SpiedRun(unittest.TestCase):
    """One run with every training call and parameter snapshot recorded."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset(train_per_class=40)
        cls.protocol = small_protocol(epochs=2, batch_size=16, memory_per_class=7)
        cls.plan = make_plan(cls.dataset.train_y, 4, cls.protocol.memory_per_class)
        cls.calls, cls.snapshots = [], []
        real = module.train_experience

        def spy(params, features, labels, replay_features, replay_labels, **kw):
            cls.calls.append(
                (
                    np.array(features),
                    np.array(labels),
                    np.array(replay_features),
                    np.array(replay_labels),
                    kw,
                )
            )
            out = real(params, features, labels, replay_features, replay_labels, **kw)
            cls.snapshots.append(copy.deepcopy(params))
            return out

        with mock.patch.object(module, "train_experience", spy):
            cls.result = adapter().run_sequence(cls.dataset, cls.protocol, 4, cls.plan)

    # --- replay exposure and leakage ---------------------------------------------------------
    def test_training_pool_is_current_experience_plus_old_class_replay_only(self):
        d, plan = self.dataset, self.plan
        self.assertEqual(len(self.calls), 5)
        for task, (x, y, rx, ry, kw) in enumerate(self.calls):
            current = set(TASK_ORDER[task])
            old = {c for t in TASK_ORDER[:task] for c in t}
            self.assertEqual(set(y.tolist()), current)
            self.assertTrue(set(ry.tolist()) <= old)
            self.assertEqual(
                sorted(kw["eligible_classes"].tolist()), sorted(current | old)
            )
            np.testing.assert_array_equal(
                x, d.train_x[plan.tasks[task]]
            )  # training rows
            np.testing.assert_array_equal(rx, d.train_x[plan.replay_indices(task)])
            self.assertEqual(kw["seed"], 4 + task)
            self.assertEqual(kw["sampling"], "class_balanced")

    def test_replay_exemplars_are_exactly_the_shared_plans_memory(self):
        for task, (_, _, _, ry, _) in enumerate(self.calls):
            expected = self.dataset.train_y[self.plan.replay_indices(task)]
            np.testing.assert_array_equal(ry, expected)
            if task:
                counts = np.bincount(ry, minlength=10)
                self.assertTrue(
                    np.all(counts[: 2 * task] == self.protocol.memory_per_class)
                )
                self.assertTrue(
                    np.all(counts[2 * task :] == 0)
                )  # no future-class exemplar

    def test_replay_budget_and_old_classes_are_actually_drawn(self):
        self.assertEqual(self.result.replay_examples_used_by_task, [0, 14, 28, 42, 56])
        for task, stats in enumerate(self.result.extras["training_stats"]):
            drawn = stats["samples_drawn_by_class"]
            self.assertTrue(
                all(d == 0 for d in drawn[2 * (task + 1) :])
            )  # nothing from the future
            if task:  # class-balanced: replayed classes are seen at (near) the same rate as new
                self.assertTrue(all(d > 0 for d in drawn[: 2 * task]))
                self.assertLessEqual(
                    max(drawn[: 2 * (task + 1)]) - min(drawn[: 2 * (task + 1)]),
                    stats["n_updates"] * 2,
                )

    # --- evaluation ----------------------------------------------------------------------------
    @staticmethod
    def independent_forward(params, x):
        a = x.astype(np.float64)
        for w, b in zip(params.weights[:-1], params.biases[:-1]):
            a = np.maximum(a @ w.astype(np.float64) + b, 0)
        return a @ params.weights[-1].astype(np.float64) + params.biases[-1]

    def test_every_matrix_row_equals_an_independent_evaluation_of_the_weights(self):
        order = self.protocol.task_order
        test_x, test_y = (
            np.asarray(self.dataset.test_x),
            np.asarray(self.dataset.test_y),
        )
        for step, params in enumerate(self.snapshots):
            logits = self.independent_forward(params, test_x)
            seen = [c for t in order[: step + 1] for c in t]
            predicted = np.asarray(seen)[np.argmax(logits[:, seen], axis=1)]
            for j in range(step + 1):
                mask = np.isin(test_y, order[j])
                self.assertAlmostEqual(
                    self.result.accuracy_matrix[step, j],
                    float(np.mean(predicted[mask] == test_y[mask])),
                    places=6,
                )
            self.assertTrue(
                np.all(np.isnan(self.result.accuracy_matrix[step, step + 1 :]))
            )

    def test_matrix_is_well_formed_and_metrics_apply(self):
        m = metrics.validate_matrix(self.result.accuracy_matrix)
        self.assertEqual(m.shape, (5, 5))
        self.assertGreater(
            metrics.final_mean_accuracy(m), 0.1
        )  # learns something on this data
        metrics.mean_forgetting(m)

    # --- record --------------------------------------------------------------------------------
    def test_record_is_complete_serialisable_and_marks_unavailable_values_none(self):
        c = self.result.run_config
        self.assertEqual(
            (c["seed"], c["hidden"], c["learning_rate"], c["replay"]),
            (4, [12], 0.1, True),
        )
        self.assertEqual(
            (c["epochs"], c["batch_size"], c["memory_per_class"]), (2, 16, 7)
        )
        self.assertEqual(c["dataset"]["evaluation_split"], "test")
        self.assertEqual(c["plan_fingerprint"], self.plan.fingerprint())
        self.assertEqual(c["role"], "primary")
        json.dumps(c)
        json.dumps(self.result.algorithm)
        r = self.result.resources
        self.assertIsNone(r["peak_memory_mb"])
        self.assertGreater(r["training_time_seconds"], 0)
        self.assertGreater(r["inference_time_seconds"], 0)
        self.assertEqual(r["n_parameters"], 16 * 12 + 12 + 12 * 10 + 10)
        core = self.result.algorithm["core"]
        self.assertEqual(core["commit"], module.core_provenance()["commit"])
        self.assertIn("official_problems", core)
        self.assertIn("commit", self.result.algorithm["benchmark"])


class ZeroReplayAblation(unittest.TestCase):
    def test_ablation_is_a_separate_role_and_really_uses_no_replay(self):
        dataset, protocol = synthetic_dataset(), small_protocol()
        calls = []
        real = module.train_experience

        def spy(params, f, y, rf, ry, **kw):
            calls.append(len(ry))
            return real(params, f, y, rf, ry, **kw)

        with mock.patch.object(module, "train_experience", spy):
            ablation = adapter(replay=False, role="ablation_no_replay").run_sequence(
                dataset, protocol, 0
            )
        self.assertEqual(calls, [0] * 5)
        self.assertEqual(ablation.replay_examples_used_by_task, [0] * 5)
        self.assertEqual(ablation.run_config["role"], "ablation_no_replay")
        self.assertFalse(ablation.run_config["replay"])
        primary = adapter().run_sequence(dataset, protocol, 0)
        self.assertEqual(primary.replay_examples_used_by_task, [0, 20, 40, 60, 80])
        self.assertFalse(
            np.array_equal(
                primary.accuracy_matrix, ablation.accuracy_matrix, equal_nan=True
            )
        )

    def test_role_and_replay_flag_must_agree(self):
        with self.assertRaises(AdapterError):
            adapter(replay=False)  # primary role cannot drop replay
        with self.assertRaises(AdapterError):
            adapter(replay=True, role="ablation_no_replay")


class Determinism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.protocol = synthetic_dataset(), small_protocol()

    def run_seed(self, seed, a=None):
        return (a or adapter()).run_sequence(self.dataset, self.protocol, seed)

    def test_fixed_seed_is_deterministic_and_seeds_differ(self):
        a, b, c = self.run_seed(0), self.run_seed(0), self.run_seed(1)
        np.testing.assert_array_equal(a.accuracy_matrix, b.accuracy_matrix)
        self.assertFalse(
            np.array_equal(a.accuracy_matrix, c.accuracy_matrix, equal_nan=True)
        )

    def test_state_is_isolated_between_runs_and_instances(self):
        shared = adapter()
        first = self.run_seed(0, shared)
        self.run_seed(1, shared)
        again = self.run_seed(0, shared)
        fresh = self.run_seed(0)
        np.testing.assert_array_equal(first.accuracy_matrix, again.accuracy_matrix)
        np.testing.assert_array_equal(first.accuracy_matrix, fresh.accuracy_matrix)

    def test_initialisation_is_seeded_and_independent_of_batching(self):
        a, b, c = (init_params((8,), s) for s in (0, 0, 1))
        np.testing.assert_array_equal(a.weights[0], b.weights[0])
        self.assertFalse(np.array_equal(a.weights[0], c.weights[0]))

    def test_shared_data_and_plan_are_not_modified(self):
        d = self.dataset
        before = array_fingerprint(d.train_x, d.train_y, d.test_x, d.test_y)
        plan = make_plan(d.train_y, 0, self.protocol.memory_per_class)
        fingerprint = plan.fingerprint()
        adapter().run_sequence(d, self.protocol, 0, plan)
        self.assertEqual(
            before, array_fingerprint(d.train_x, d.train_y, d.test_x, d.test_y)
        )
        self.assertEqual(fingerprint, plan.fingerprint())
        with self.assertRaises(ValueError):
            plan.tasks[0][0] = 1  # arrays are read-only

    def test_giving_the_identical_plan_explicitly_changes_nothing(self):
        plan = make_plan(self.dataset.train_y, 2, self.protocol.memory_per_class)
        with_plan = adapter().run_sequence(self.dataset, self.protocol, 2, plan)
        without = adapter().run_sequence(self.dataset, self.protocol, 2)
        np.testing.assert_array_equal(
            with_plan.accuracy_matrix, without.accuracy_matrix
        )


class RefusalsAndDivergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.protocol = synthetic_dataset(), small_protocol()

    def test_invalid_configuration_is_refused(self):
        for kw in (
            {"hidden": (0,)},
            {"hidden": (4.5,)},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
            {"role": "best"},
            {"mode": "beta"},
        ):
            with self.subTest(kw), self.assertRaises(AdapterError):
                adapter(**kw)

    def test_unsupported_task_order_and_foreign_datasets_are_refused(self):
        from dataclasses import replace

        order = ((1, 0), (2, 3), (4, 5), (6, 7), (8, 9))
        with self.assertRaisesRegex(AdapterError, "task order"):
            adapter().run_sequence(self.dataset, small_protocol(task_order=order), 0)
        with self.assertRaisesRegex(AdapterError, "projected with seed"):
            adapter().run_sequence(
                replace(self.dataset, projection_seed=1), self.protocol, 0
            )
        with self.assertRaisesRegex(AdapterError, "verified files"):
            adapter().run_sequence(
                replace(self.dataset, source_hashes={}), self.protocol, 0
            )

    def test_a_plan_from_another_seed_or_dataset_is_refused(self):
        plan = make_plan(self.dataset.train_y, 0, self.protocol.memory_per_class)
        with self.assertRaisesRegex(PlanError, "seed"):
            adapter().run_sequence(self.dataset, self.protocol, 1, plan)
        other = synthetic_dataset(train_per_class=31)
        with self.assertRaisesRegex(PlanError, "training labels"):
            adapter().run_sequence(other, self.protocol, 0, plan)
        wrong_memory = make_plan(self.dataset.train_y, 0, 3)
        with self.assertRaisesRegex(PlanError, "memory_per_class"):
            adapter().run_sequence(self.dataset, self.protocol, 0, wrong_memory)

    def test_divergence_is_reported_not_returned_as_a_result(self):
        with self.assertRaises(TrainingDiverged) as caught:
            adapter(learning_rate=1e30).run_sequence(self.dataset, self.protocol, 0)
        self.assertIn("experience 0", str(caught.exception))

    def test_official_mode_refuses_before_any_training(self):
        from polynomial_cl_bench.provenance import ProvenanceError

        bad = {"official_problems": ["x"]}
        calls = []
        with (
            mock.patch.object(module, "core_provenance", lambda: bad),
            mock.patch.object(
                module, "benchmark_provenance", lambda: {"official_problems": []}
            ),
            mock.patch.object(
                module, "train_experience", lambda *a, **k: calls.append(1)
            ),
            self.assertRaises(ProvenanceError),
        ):
            adapter(mode="official").run_sequence(self.dataset, self.protocol, 0)
        self.assertEqual(calls, [])

    def test_config_file_round_trip(self):
        raw = {
            "algorithm": "er_mlp",
            "hidden": [8, 8],
            "learning_rate": 0.05,
            "role": "validation_candidate",
        }
        a = ERMLPAdapter.from_dict(raw)
        self.assertEqual((a.hidden, a.learning_rate, a.replay), ((8, 8), 0.05, True))
        with self.assertRaises(AdapterError):
            ERMLPAdapter.from_dict(
                {"algorithm": "other", "hidden": [], "learning_rate": 1}
            )


if __name__ == "__main__":
    unittest.main()
