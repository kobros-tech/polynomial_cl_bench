# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Phase 1 review: the adapter must preserve the core's evaluation semantics, record every
configuration value, and keep local provenance distinct from remote provenance."""

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from helpers import (
    PER_DEGREE_CONFIG_PATH,
    POLYNOMIAL_CONFIG_PATH,
    small_protocol,
    synthetic_dataset,
)
from polynomial_cl.class_incremental import TASK_ORDER, experiment
from polynomial_cl.class_incremental.features import scaled_polynomial_features

from polynomial_cl_bench import provenance as prov
from polynomial_cl_bench.algorithms import polynomial as module
from polynomial_cl_bench.algorithms.polynomial import AdapterError, PolynomialAdapter


def independent_accuracy(weights, test_x, test_y, task_order, step):
    """Benchmark-side evaluation: experience j's test examples, argmax over classes seen
    through experience ``step`` only. Uses the protocol's class lists, not the core's."""
    seen = [c for task in task_order[: step + 1] for c in task]
    logits = test_x @ weights
    predicted = np.asarray(seen)[np.argmax(logits[:, seen], axis=1)]
    row = np.full(len(task_order), np.nan)
    for j in range(step + 1):
        mask = np.isin(test_y, task_order[j])
        row[j] = np.mean(predicted[mask] == test_y[mask])
    return row


class EvaluationSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol()
        cls.eval_calls, cls.fit_calls = [], []

        real_eval, real_fit = experiment.accuracy_by_task, experiment.fit_task

        def eval_spy(weights, test_x, test_y, n_seen):
            cls.eval_calls.append(
                (weights.copy(), np.array(test_x), np.array(test_y), n_seen)
            )
            return real_eval(weights, test_x, test_y, n_seen)

        def fit_spy(weights, features, labels, replay_features, replay_labels, **kw):
            cls.fit_calls.append((np.array(labels), np.array(replay_labels), kw))
            return real_fit(
                weights, features, labels, replay_features, replay_labels, **kw
            )

        with (
            mock.patch.object(experiment, "accuracy_by_task", eval_spy),
            mock.patch.object(experiment, "fit_task", fit_spy),
        ):
            cls.result = PolynomialAdapter().run_sequence(cls.dataset, cls.protocol, 0)

    def test_matrix_rows_equal_an_independent_evaluation_of_the_core_weights(self):
        self.assertEqual(len(self.eval_calls), 5)
        for step, (weights, _, test_y, _) in enumerate(self.eval_calls):
            count = weights.shape[0]
            test_features = scaled_polynomial_features(self.dataset.test_x, 3)[
                :, :count
            ]
            expected = independent_accuracy(
                weights,
                test_features,
                np.asarray(self.dataset.test_y),
                self.protocol.task_order,
                step,
            )
            np.testing.assert_allclose(
                self.result.accuracy_matrix[step], expected, atol=1e-12, equal_nan=True
            )

    def test_evaluation_uses_the_independent_test_set(self):
        for weights, test_x, test_y, _ in self.eval_calls:
            expected = scaled_polynomial_features(self.dataset.test_x, 3)[
                :, : weights.shape[0]
            ]
            np.testing.assert_array_equal(test_y, self.dataset.test_y)
            np.testing.assert_allclose(test_x, expected, atol=1e-6)
            # None of the test labels/rows can be the training set (different draws).
            self.assertEqual(len(test_y), len(self.dataset.test_y))

    def test_future_class_output_weights_are_never_trained(self):
        for step, (weights, _, _, _) in enumerate(self.eval_calls):
            n_seen = sum(len(t) for t in self.protocol.task_order[: step + 1])
            self.assertTrue(
                np.all(weights[:, n_seen:] == 0), f"future columns moved at {step}"
            )

    def test_training_sees_only_current_experience_plus_old_class_replay(self):
        self.assertEqual(len(self.fit_calls), 5)
        for step, (labels, replay_labels, kw) in enumerate(self.fit_calls):
            current = set(TASK_ORDER[step])
            old = {c for task in TASK_ORDER[:step] for c in task}
            self.assertEqual(set(labels.tolist()), current)
            self.assertTrue(set(replay_labels.tolist()) <= old)
            self.assertEqual(
                sorted(kw["eligible_classes"].tolist()), sorted(current | old)
            )
            self.assertEqual(kw["sampling"], "class_balanced")


class CoreMaskingRule(unittest.TestCase):
    """Characterises the pinned core's evaluation rule where masking actually matters."""

    def test_future_classes_cannot_win_even_when_their_logits_are_largest(self):
        rng = np.random.default_rng(0)
        order = TASK_ORDER
        dataset = synthetic_dataset()
        features = scaled_polynomial_features(dataset.test_x, 3)
        weights = rng.normal(0, 0.1, size=(features.shape[1], 10)).astype(np.float32)
        weights[0, 4:] += (
            50.0  # constant feature: future classes dominate unless masked
        )
        step = 1  # only classes 0-3 have been introduced
        core = experiment.accuracy_by_task(
            weights, features, np.asarray(dataset.test_y), step + 1
        )
        masked = independent_accuracy(
            weights, features, np.asarray(dataset.test_y), order, step
        )
        np.testing.assert_allclose(core[: step + 1], masked[: step + 1], atol=1e-12)
        self.assertTrue(np.all(np.isnan(core[step + 1 :])))
        unmasked = np.argmax(features @ weights, axis=1)
        self.assertTrue(
            np.all(unmasked >= 4)
        )  # without masking, every prediction is wrong


class RunRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol(epochs=2, batch_size=16, memory_per_class=7)

    def test_every_relevant_value_is_recorded(self):
        adapter = PolynomialAdapter.from_file(POLYNOMIAL_CONFIG_PATH)
        result = adapter.run_sequence(self.dataset, self.protocol, 4)
        c = result.run_config
        self.assertEqual(c["seed"], 4)
        self.assertEqual(c["sampling"], "class_balanced")
        self.assertEqual(c["dataset"]["projection_seed"], 0)
        self.assertEqual(c["dataset"]["fingerprint"], self.dataset.fingerprint)
        self.assertEqual(c["dataset"]["source_hashes"], self.dataset.source_hashes)
        self.assertEqual((c["condition"], c["strategy"]), ("expanding", "ordinary"))
        self.assertEqual(c["learning_rate"], 0.1)
        self.assertEqual(c["feature_power"], 0.0)
        self.assertEqual(
            (c["epochs"], c["batch_size"], c["memory_per_class"]), (2, 16, 7)
        )
        self.assertEqual(c["task_order"], [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        self.assertEqual(c["protocol"]["fingerprint"], self.protocol.fingerprint())
        self.assertEqual(c["role"], "primary_reference")
        self.assertEqual(result.replay_examples_used_by_task, [0, 14, 28, 42, 56])
        self.assertEqual(
            result.algorithm["core"]["commit"], prov.core_provenance()["commit"]
        )
        json.dumps(c)  # serialisable as is

    def test_changing_a_protocol_value_changes_the_recorded_fingerprint(self):
        a = PolynomialAdapter().run_sequence(self.dataset, self.protocol, 0)
        b = PolynomialAdapter().run_sequence(
            self.dataset, replace(self.protocol, sampling="shuffled"), 0
        )
        self.assertNotEqual(
            a.run_config["protocol"]["fingerprint"],
            b.run_config["protocol"]["fingerprint"],
        )
        self.assertEqual(b.run_config["sampling"], "shuffled")

    def test_dataset_with_wrong_projection_or_unverified_source_is_refused(self):
        wrong_seed = replace(self.dataset, projection_seed=1)
        with self.assertRaisesRegex(AdapterError, "projected with seed"):
            PolynomialAdapter().run_sequence(wrong_seed, self.protocol, 0)
        unverified = replace(self.dataset, source_hashes={})
        with self.assertRaisesRegex(AdapterError, "verified files"):
            PolynomialAdapter().run_sequence(unverified, self.protocol, 0)

    def test_core_echo_mismatch_is_detected(self):
        real = module.run_condition

        def lying(*args, **kwargs):
            record = real(*args, **kwargs)
            record["learning_rate"] = 9.9
            return record

        with (
            mock.patch.object(module, "run_condition", lying),
            self.assertRaisesRegex(AdapterError, "learning_rate"),
        ):
            PolynomialAdapter().run_sequence(self.dataset, self.protocol, 0)


class TwoConfigurations(unittest.TestCase):
    def test_reference_and_experimental_configs_are_distinct_and_unclaimed(self):
        ref = PolynomialAdapter.from_file(POLYNOMIAL_CONFIG_PATH).describe()
        exp = PolynomialAdapter.from_file(PER_DEGREE_CONFIG_PATH).describe()
        self.assertEqual(ref["role"], "primary_reference")
        self.assertEqual(ref["configuration"]["learning_rate"], 0.1)
        self.assertEqual(exp["role"], "experimental")
        self.assertEqual(exp["configuration"]["learning_rate"], [0.75, 1.5, 1.5, 6.0])
        for info in (ref, exp):
            self.assertEqual(
                (info["condition"], info["strategy"]), ("expanding", "ordinary")
            )
            self.assertEqual(
                info["hyperparameter_provenance"]["optimality_claim"], "none"
            )
        self.assertEqual(
            exp["hyperparameter_provenance"]["selection"],
            "validation_selected_by_core_per_degree_study",
        )
        self.assertEqual(
            ref["hyperparameter_provenance"]["selection"],
            "core_default_not_validation_selected",
        )

    def test_per_degree_vector_reaches_the_core_and_runs(self):
        seen = {}
        real = module.run_condition

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        adapter = PolynomialAdapter.from_file(PER_DEGREE_CONFIG_PATH)
        with mock.patch.object(module, "run_condition", spy):
            result = adapter.run_sequence(synthetic_dataset(), small_protocol(), 0)
        self.assertEqual(list(seen["learning_rate"]), [0.75, 1.5, 1.5, 6.0])
        self.assertEqual(seen["sampling"], "class_balanced")
        self.assertEqual(result.run_config["learning_rate"], [0.75, 1.5, 1.5, 6.0])
        self.assertEqual(result.run_config["role"], "experimental")

    def test_invalid_role_is_refused(self):
        with self.assertRaises(AdapterError):
            PolynomialAdapter(role="best")


class FakeDistribution:
    version = "0.1.0"

    def __init__(self, direct_url):
        self._text = json.dumps(direct_url)

    def read_text(self, name):
        return self._text if name == "direct_url.json" else None


class Provenance(unittest.TestCase):
    def provenance_for(self, direct_url):
        real = prov.metadata.distribution

        def distribution(name):
            return (
                FakeDistribution(direct_url) if name == "polynomial-cl" else real(name)
            )

        with mock.patch.object(prov.metadata, "distribution", distribution):
            return prov.core_provenance()

    def test_pip_vcs_install_counts_as_remote_fetched_and_checks_the_pin(self):
        pin = prov.pinned_core_commit()
        self.assertRegex(pin, r"^[0-9a-f]{40}$")
        url = "https://github.com/kobros-tech/polynomial_cl_demo.git"
        ok = self.provenance_for(
            {"url": url, "vcs_info": {"vcs": "git", "commit_id": pin}}
        )
        self.assertEqual(
            (
                ok["commit_source"],
                ok["remote_verified"],
                ok["matches_pin"],
                ok["remote_url"],
            ),
            ("pip_vcs_install", True, True, url),
        )
        other = self.provenance_for(
            {"url": url, "vcs_info": {"vcs": "git", "commit_id": "0" * 40}}
        )
        self.assertFalse(other["matches_pin"])

    def test_local_checkout_is_never_reported_as_remote_verified(self):
        with tempfile.TemporaryDirectory() as tmp:

            def git(*args):
                subprocess.run(
                    [
                        "git",
                        "-C",
                        tmp,
                        "-c",
                        "user.name=t",
                        "-c",
                        "user.email=t@t",
                        *args,
                    ],
                    check=True,
                    capture_output=True,
                )

            git("init", "-q")
            (Path(tmp) / "f.txt").write_text("a")
            git("add", "f.txt")
            git("commit", "-q", "-m", "c")
            head = subprocess.run(
                ["git", "-C", tmp, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            url = {"url": f"file://{tmp}", "dir_info": {"editable": True}}
            clean = self.provenance_for(url)
            self.assertEqual(clean["commit"], head)
            self.assertEqual(clean["commit_source"], "local_git")
            self.assertFalse(clean["remote_verified"])
            self.assertIsNone(clean["remote_url"])
            self.assertFalse(clean["matches_pin"])  # not the pinned commit
            (Path(tmp) / "f.txt").write_text("b")
            self.assertEqual(
                self.provenance_for(url)["commit_source"], "local_git_dirty"
            )

    def test_unknown_source_has_no_commit_and_no_claim(self):
        info = self.provenance_for(
            {"url": "https://example.invalid/pkg.whl", "archive_info": {}}
        )
        self.assertIsNone(info["commit"])
        self.assertEqual(info["commit_source"], "unknown")
        self.assertFalse(info["remote_verified"])
        self.assertIsNone(info["matches_pin"])


if __name__ == "__main__":
    unittest.main()
