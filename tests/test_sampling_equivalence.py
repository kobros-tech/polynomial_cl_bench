# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The benchmark's batching must reproduce the core's *actual* behaviour, not just class
proportions: same batches, same leftover-slot choices, same cycles, same update counts.

Four independent angles, so a failure points at the cause:

1. draw-for-draw equality with the core's private batcher (private import is confined to tests);
2. the same batches captured from inside a real, public ``fit_task`` call;
3. an end-to-end check: a bias-free linear model trained by the benchmark's trainer must
   reproduce ``fit_task``'s weights, for both policies and across experiences;
4. a full ER-MLP run and a full core run on one shared plan report identical batch statistics.
"""

import unittest
from unittest import mock

import numpy as np
import polynomial_cl.class_incremental.training as core_training
from helpers import small_protocol, synthetic_dataset
from polynomial_cl.class_incremental import make_feature_data, run_condition

from polynomial_cl_bench import sampling
from polynomial_cl_bench.algorithms.er_mlp import (
    ERMLPAdapter,
    init_params,
    train_experience,
)
from polynomial_cl_bench.plan import make_plan


def labels_with_counts(counts):
    return np.repeat(np.arange(len(counts)), counts)


# (name, class counts, batch_size, n_updates): leftovers, short classes, missing classes...
CASES = [
    ("ten classes, leftover 8 (128 % 10)", [20] * 8 + [300, 310], 128, 25),
    ("no leftover (120 % 10 == 0)", [50] * 10, 120, 12),
    ("four classes, leftover 0", [30, 40, 50, 60], 128, 15),
    ("tiny class repeats across many cycles", [3, 200, 200], 128, 40),
    ("batch smaller than class count", [10] * 10, 7, 30),
    ("single class", [37], 16, 9),
    ("class counts below their slots", [2, 2, 2, 2, 2], 128, 10),
    ("odd batch, two classes", [25, 31], 33, 20),
]


class AgainstCorePrivateBatcher(unittest.TestCase):
    def test_every_batch_and_the_generator_state_equal_the_cores(self):
        for name, counts, batch_size, n_updates in CASES:
            for seed in (0, 1, 123):
                with self.subTest(name, seed=seed):
                    labels = labels_with_counts(counts)
                    rng_a, rng_b = (
                        np.random.default_rng(seed),
                        np.random.default_rng(seed),
                    )
                    mine = list(
                        sampling.class_balanced_batches(
                            labels, n_updates, batch_size, rng_a
                        )
                    )
                    core = list(
                        core_training._balanced_batches(
                            labels, n_updates, batch_size, rng_b
                        )
                    )
                    self.assertEqual(len(mine), len(core))
                    for a, b in zip(mine, core):
                        np.testing.assert_array_equal(a, b)
                    self.assertEqual(
                        rng_a.bit_generator.state, rng_b.bit_generator.state
                    )


class BatchSemantics(unittest.TestCase):
    """The documented behaviour, checked directly on the benchmark's batches."""

    def batches(self, counts, batch_size, n_updates, seed=0):
        labels = labels_with_counts(counts)
        out = list(
            sampling.class_balanced_batches(
                labels, n_updates, batch_size, np.random.default_rng(seed)
            )
        )
        return labels, out

    def test_class_allocation_leftover_slots_and_full_batches(self):
        counts, batch_size, n_updates = [500] * 10, 128, 400
        labels, batches = self.batches(counts, batch_size, n_updates)
        per_batch = np.array([np.bincount(labels[b], minlength=10) for b in batches])
        self.assertTrue(
            np.all(per_batch.sum(axis=1) == batch_size)
        )  # every batch is full
        self.assertEqual(
            set(np.unique(per_batch)), {12, 13}
        )  # 128 // 10 base, +1 on leftovers
        self.assertTrue(
            np.all((per_batch == 13).sum(axis=1) == 8)
        )  # exactly 128 % 10 bonuses
        share = (per_batch == 13).mean(
            axis=0
        )  # leftovers rotate over classes at random
        self.assertTrue(np.all(np.abs(share - 0.8) < 0.1), share)

    def test_classes_without_examples_do_not_take_part(self):
        labels = labels_with_counts([40, 40, 0, 0, 40])  # classes 2 and 3 absent
        batches = list(
            sampling.class_balanced_batches(labels, 10, 31, np.random.default_rng(0))
        )
        for batch in batches:
            counts = np.bincount(labels[batch], minlength=5)
            self.assertEqual(counts[2] + counts[3], 0)
            self.assertEqual(sorted(counts[[0, 1, 4]]), [10, 10, 11])  # 31 = 3 * 10 + 1

    def test_each_class_is_read_through_shuffled_cycles_before_repeating(self):
        counts = [7, 50, 50]
        labels, batches = self.batches(counts, 30, 60)
        for cls, n in enumerate(counts):
            stream = np.concatenate([b[labels[b] == cls] for b in batches])
            for start in range(0, len(stream) - n + 1, n):
                self.assertEqual(
                    sorted(stream[start : start + n]),
                    list(np.flatnonzero(labels == cls)),
                    f"class {cls}: cycle at {start} is not a permutation",
                )
        stream = np.concatenate([b[labels[b] == 0] for b in batches])
        self.assertFalse(
            np.array_equal(stream[:7], stream[7:14])
        )  # reshuffled, not replayed

    def test_a_short_class_is_repeated_never_padded_from_another_class(self):
        labels, batches = self.batches([2, 100], 64, 5)
        for batch in batches:
            self.assertEqual(np.bincount(labels[batch], minlength=2).tolist(), [32, 32])

    def test_incomplete_batches_and_update_counts(self):
        n_rows, batch_size, epochs = 250, 64, 3
        label_columns = np.arange(n_rows) % 5
        for policy in sampling.SAMPLING_POLICIES:
            batches = list(
                sampling.training_batches(
                    label_columns,
                    epochs=epochs,
                    batch_size=batch_size,
                    sampling=policy,
                    seed=4,
                )
            )
            per_epoch = -(-n_rows // batch_size)  # 4
            self.assertEqual(len(batches), epochs * per_epoch)
            sizes = [len(b) for b in batches[:per_epoch]]
            if policy == "shuffled":
                self.assertEqual(sizes, [64, 64, 64, 58])  # last batch is short
                for e in range(epochs):  # each epoch is one permutation of all rows
                    epoch = np.concatenate(batches[e * per_epoch : (e + 1) * per_epoch])
                    self.assertEqual(sorted(epoch), list(range(n_rows)))
            else:
                self.assertEqual(
                    sizes, [64] * 4
                )  # all batches full: 256 rows drawn, not 250


class CapturedFromRealFitTask(unittest.TestCase):
    """Batches the core really consumes inside the public ``fit_task``."""

    def run_core(self, sampling_policy):
        dataset = synthetic_dataset(train_per_class=40)
        plan = make_plan(dataset.train_y, seed=3, memory_per_class=12)
        task = 2
        seen = np.arange(6)
        current, replay = plan.tasks[task], plan.replay_indices(task)
        x = np.asarray(dataset.train_x, dtype=np.float64)
        y = np.asarray(dataset.train_y).astype(np.int64)
        recorded, stats = [], {}
        real = core_training._balanced_batches

        def spy(*args, **kwargs):
            for batch in real(*args, **kwargs):
                recorded.append(batch.copy())
                yield batch

        with mock.patch.object(core_training, "_balanced_batches", spy):
            core_training.fit_task(
                np.zeros((16, 10)),
                x[current],
                y[current],
                x[replay],
                y[replay],
                epochs=3,
                learning_rate=0.05,
                batch_size=32,
                seed=3 + task,
                eligible_classes=seen,
                sampling=sampling_policy,
                stats=stats,
            )
        labels = np.concatenate([y[current], y[replay]])
        return labels, recorded, stats

    def test_class_balanced_batches_equal_those_inside_fit_task(self):
        labels, recorded, stats = self.run_core("class_balanced")
        mine = list(
            sampling.training_batches(
                np.searchsorted(np.arange(6), labels),
                epochs=3,
                batch_size=32,
                sampling="class_balanced",
                seed=5,
            )
        )
        self.assertEqual(len(mine), len(recorded))
        for a, b in zip(mine, recorded):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(mine), stats["n_updates"])


class LinearModelReproducesFitTask(unittest.TestCase):
    """Same batches + same update rule => same weights as the core's softmax head."""

    def test_weights_match_the_core_for_both_policies_across_experiences(self):
        dataset = synthetic_dataset(train_per_class=40)
        x = np.asarray(dataset.train_x, dtype=np.float64)
        y = np.asarray(dataset.train_y).astype(np.int64)
        plan = make_plan(dataset.train_y, seed=2, memory_per_class=9)
        for policy in sampling.SAMPLING_POLICIES:
            with self.subTest(policy):
                core_w = np.zeros((16, 10))
                mine = init_params(
                    (), seed=0, dtype=np.float64, use_bias=False, zero=True
                )
                for task in range(5):
                    cur, rep = plan.tasks[task], plan.replay_indices(task)
                    seen = np.arange(2 * (task + 1))
                    kwargs = {
                        "epochs": 2,
                        "batch_size": 24,
                        "seed": 2 + task,
                        "eligible_classes": seen,
                        "sampling": policy,
                    }
                    core_stats, my_stats = {}, {}
                    core_training.fit_task(
                        core_w,
                        x[cur],
                        y[cur],
                        x[rep],
                        y[rep],
                        learning_rate=0.1,
                        stats=core_stats,
                        **kwargs,
                    )
                    train_experience(
                        mine,
                        x[cur],
                        y[cur],
                        x[rep],
                        y[rep],
                        learning_rate=0.1,
                        stats=my_stats,
                        **kwargs,
                    )
                    np.testing.assert_allclose(
                        mine.weights[0], core_w, atol=1e-9, rtol=0
                    )
                    for key in (
                        "n_updates",
                        "samples_drawn_by_class",
                        "examples_processed",
                        "examples_available",
                        "mean_batch_composition",
                        "pool_sizes_by_class",
                    ):
                        self.assertEqual(my_stats[key], core_stats[key], (task, key))


class FullRunsAgreeOnBatchStatistics(unittest.TestCase):
    """ER-MLP and the polynomial core, one shared plan: identical sampling record."""

    def test_per_experience_statistics_equal_the_cores_record(self):
        dataset = synthetic_dataset(train_per_class=40)
        for policy in sampling.SAMPLING_POLICIES:
            with self.subTest(policy):
                protocol = small_protocol(
                    epochs=2, batch_size=24, memory_per_class=9, sampling=policy
                )
                plan = make_plan(dataset.train_y, 1, protocol.memory_per_class)
                data = make_feature_data(
                    dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y
                )
                record = run_condition(
                    "expanding",
                    data,
                    list(plan.tasks),
                    dict(plan.memory),
                    seed=1,
                    epochs=protocol.epochs,
                    learning_rate=0.1,
                    batch_size=protocol.batch_size,
                    sampling=policy,
                )
                result = ERMLPAdapter(hidden=(8,), learning_rate=0.1).run_sequence(
                    dataset, protocol, 1, plan
                )
                for task, core_stats in enumerate(record["sampling_by_task"]):
                    mine = result.extras["training_stats"][task]
                    for key in (
                        "pool_sizes_by_class",
                        "samples_drawn_by_class",
                        "n_updates",
                        "examples_processed",
                        "examples_available",
                        "mean_batch_composition",
                    ):
                        self.assertEqual(mine[key], core_stats[key], (task, key))
                    expected_updates = protocol.epochs * -(
                        -(len(plan.tasks[task]) + len(plan.replay_indices(task)))
                        // protocol.batch_size
                    )
                    self.assertEqual(mine["n_updates"], expected_updates)


if __name__ == "__main__":
    unittest.main()
