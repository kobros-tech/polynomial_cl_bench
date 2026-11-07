# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Shared experience/replay plan, validation separation, tuning selection and budget."""

import json
import pickle
import unittest
from dataclasses import replace
from typing import ClassVar
from unittest import mock

import numpy as np
from helpers import small_protocol, synthetic_dataset
from polynomial_cl.continual import holdout_split, make_task_plan

from polynomial_cl_bench import tuning
from polynomial_cl_bench.algorithms import er_mlp, polynomial
from polynomial_cl_bench.algorithms.base import SequenceResult
from polynomial_cl_bench.algorithms.er_mlp import ERMLPAdapter
from polynomial_cl_bench.algorithms.polynomial import PolynomialAdapter
from polynomial_cl_bench.datasets import DatasetVerificationError, validation_view
from polynomial_cl_bench.plan import ExperiencePlan, PlanError, make_plan
from polynomial_cl_bench.provenance import ProvenanceError


class SharedPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset(train_per_class=40)
        cls.protocol = small_protocol(epochs=1, batch_size=24, memory_per_class=9)

    def test_plan_is_exactly_the_cores_make_task_plan(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        tasks, memory = make_task_plan(np.asarray(self.dataset.train_y), 3, 9)
        for a, b in zip(plan.tasks, tasks):
            np.testing.assert_array_equal(a, b)
        for c in range(10):
            np.testing.assert_array_equal(plan.memory[c], memory[c])

    def test_replay_indices_follow_the_cores_concatenation_order(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        self.assertEqual(len(plan.replay_indices(0)), 0)
        for task in range(1, 5):
            expected = np.concatenate([plan.memory[c] for c in range(2 * task)])
            np.testing.assert_array_equal(plan.replay_indices(task), expected)

    def test_plan_arrays_are_read_only_and_fingerprint_is_stable_and_seed_sensitive(
        self,
    ):
        a, b, c = (make_plan(self.dataset.train_y, s, 9) for s in (3, 3, 4))
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), c.fingerprint())
        with self.assertRaises(ValueError):
            a.memory[0][0] = 0

    def test_fingerprint_changes_when_replay_arrays_are_exchanged_between_classes(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        swapped = dict(plan.memory)
        swapped[0], swapped[1] = plan.memory[1], plan.memory[0]
        other = replace(plan, memory=swapped)
        self.assertNotEqual(plan.fingerprint(), other.fingerprint())

    def test_fingerprint_depends_on_class_ids_not_just_array_order(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        # same arrays in the same sorted order, but filed under different class ids
        relabelled = {c + 100: plan.memory[c] for c in plan.memory}
        other = replace(plan, memory=relabelled)
        self.assertNotEqual(plan.fingerprint(), other.fingerprint())

    def test_fingerprint_depends_on_task_order_and_label_identity(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        reordered = replace(plan, task_order=tuple(reversed(plan.task_order)))
        self.assertNotEqual(plan.fingerprint(), reordered.fingerprint())
        relabelled = replace(plan, label_fingerprint="0" * 64)
        self.assertNotEqual(plan.fingerprint(), relabelled.fingerprint())

    def test_memory_mapping_is_immutable_and_isolated_from_its_source(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        before = plan.fingerprint()
        with self.assertRaises(TypeError):
            plan.memory[0] = plan.memory[1]
        with self.assertRaises(TypeError):
            del plan.memory[0]
        source = dict(plan.memory)
        rebuilt = ExperiencePlan(
            seed=plan.seed,
            memory_per_class=plan.memory_per_class,
            task_order=plan.task_order,
            train_size=plan.train_size,
            label_fingerprint=plan.label_fingerprint,
            tasks=plan.tasks,
            memory=source,
        )
        source[0] = plan.memory[1]  # mutating the caller's dict must not leak in
        self.assertEqual(rebuilt.fingerprint(), before)

    def test_plan_survives_pickling_with_the_same_fingerprint(self):
        plan = make_plan(self.dataset.train_y, 3, 9)
        clone = pickle.loads(pickle.dumps(plan))
        self.assertEqual(clone.fingerprint(), plan.fingerprint())

    def test_both_adapters_receive_identical_experiences_and_replay_rows(self):
        plan = make_plan(self.dataset.train_y, 1, self.protocol.memory_per_class)
        seen_core, seen_mlp = {}, []
        real_core, real_mlp = polynomial.run_condition, er_mlp.train_experience

        def core_spy(condition, data, tasks, memory, **kw):
            seen_core["tasks"], seen_core["memory"] = tasks, memory
            return real_core(condition, data, tasks, memory, **kw)

        def mlp_spy(params, f, y, rf, ry, **kw):
            seen_mlp.append((np.array(f), np.array(rf)))
            return real_mlp(params, f, y, rf, ry, **kw)

        with (
            mock.patch.object(polynomial, "run_condition", core_spy),
            mock.patch.object(er_mlp, "train_experience", mlp_spy),
        ):
            poly = PolynomialAdapter().run_sequence(
                self.dataset, self.protocol, 1, plan
            )
            mlp = ERMLPAdapter((6,), 0.1).run_sequence(
                self.dataset, self.protocol, 1, plan
            )
        for a, b in zip(seen_core["tasks"], plan.tasks):
            np.testing.assert_array_equal(a, b)  # the very plan, not a reconstruction
        self.assertTrue(
            all(seen_core["memory"][c] is plan.memory[c] for c in range(10))
        )
        for task, (_, replay_x) in enumerate(seen_mlp):
            np.testing.assert_array_equal(
                replay_x, self.dataset.train_x[plan.replay_indices(task)]
            )
        self.assertEqual(
            poly.run_config["plan_fingerprint"], mlp.run_config["plan_fingerprint"]
        )
        self.assertEqual(
            poly.replay_examples_used_by_task, mlp.replay_examples_used_by_task
        )

    def test_polynomial_result_is_unchanged_by_passing_the_identical_plan(self):
        plan = make_plan(self.dataset.train_y, 2, self.protocol.memory_per_class)
        with_plan = PolynomialAdapter().run_sequence(
            self.dataset, self.protocol, 2, plan
        )
        without = PolynomialAdapter().run_sequence(self.dataset, self.protocol, 2)
        np.testing.assert_array_equal(
            with_plan.accuracy_matrix, without.accuracy_matrix
        )

    def test_polynomial_refuses_a_foreign_plan(self):
        plan = make_plan(self.dataset.train_y, 0, self.protocol.memory_per_class)
        with self.assertRaises(PlanError):
            PolynomialAdapter().run_sequence(self.dataset, self.protocol, 1, plan)


class ValidationView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset(train_per_class=40)
        cls.protocol = small_protocol()

    def test_view_is_the_cores_holdout_split_and_places_validation_in_the_test_slot(
        self,
    ):
        view = validation_view(self.dataset, self.protocol)
        fx, fy, vx, vy = holdout_split(
            self.dataset.train_x,
            self.dataset.train_y,
            self.protocol.validation["holdout_fraction"],
            self.protocol.validation["holdout_seed"],
        )
        for got, want in (
            (view.train_x, fx),
            (view.train_y, fy),
            (view.test_x, vx),
            (view.test_y, vy),
        ):
            np.testing.assert_array_equal(got, want)
        self.assertEqual(view.evaluation_split, "validation")
        self.assertEqual(view.derived_from, self.dataset.fingerprint)
        self.assertNotEqual(view.fingerprint, self.dataset.fingerprint)

    def test_view_shares_no_row_with_its_training_part_and_none_with_the_real_test_set(
        self,
    ):
        view = validation_view(self.dataset, self.protocol)
        rows = lambda a: {tuple(r) for r in np.asarray(a).tolist()}
        self.assertFalse(rows(view.train_x) & rows(view.test_x))
        self.assertFalse(rows(self.dataset.test_x) & rows(view.test_x))
        self.assertEqual(
            len(view.train_y) + len(view.test_y), len(self.dataset.train_y)
        )

    def test_view_cannot_be_cut_from_a_view(self):
        with self.assertRaises(DatasetVerificationError):
            validation_view(validation_view(self.dataset, self.protocol), self.protocol)


# --- tuning ------------------------------------------------------------------------------------


class FakeAdapter:
    """Returns a prescribed accuracy matrix (or raises), recording what it was given."""

    log: ClassVar[list] = []

    def __init__(self, final, forgetting_drop=0.0, error=None):
        self.final, self.drop, self.error = final, forgetting_drop, error

    def run_sequence(self, dataset, protocol, seed, plan=None):
        FakeAdapter.log.append((dataset, seed, plan))
        if self.error:
            raise self.error
        m = np.full((5, 5), np.nan)
        for i in range(5):
            m[i, : i + 1] = self.final + (self.drop if i < 4 else 0.0)
        for i in range(4):
            m[i, 0] = min(
                1.0, self.final + self.drop
            )  # first experience decays by `drop`
        m[4, :] = self.final
        return SequenceResult(
            m,
            {},
            {},
            "class_balanced",
            {
                "dataset": {"evaluation_split": dataset.evaluation_split},
                "plan_fingerprint": plan.fingerprint(),
            },
            [0] * 5,
        )


class Tuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset(train_per_class=40)
        cls.protocol = small_protocol(epochs=1, batch_size=24, memory_per_class=8)

    def test_budget_seeds_and_grid_are_validated(self):
        f = lambda cfg: FakeAdapter(0.5)
        grid = [{"x": 1}, {"x": 2}, {"x": 3}]
        with self.assertRaisesRegex(tuning.TuningError, "budget is 6 runs"):
            tuning.tune("fake", f, grid, self.dataset, self.protocol, expected_runs=30)
        with self.assertRaisesRegex(tuning.TuningError, "overlap"):
            tuning.tune(
                "fake", f, grid, self.dataset, self.protocol, selection_seeds=(0, 100)
            )
        with self.assertRaises(tuning.TuningError):
            tuning.tune("fake", f, [], self.dataset, self.protocol)
        with self.assertRaises(tuning.TuningError):
            tuning.tune(
                "fake", f, grid, self.dataset, self.protocol, selection_seeds=(100, 100)
            )
        ok = tuning.tune("fake", f, grid, self.dataset, self.protocol, expected_runs=6)
        self.assertEqual(
            (ok.budget_runs, len(ok.runs), ok.selection_seeds), (6, 6, (100, 101))
        )

    def select(self, specs):
        grid = [{"i": i} for i in range(len(specs))]
        return tuning.tune(
            "fake", lambda cfg: specs[cfg["i"]], grid, self.dataset, self.protocol
        )

    def test_accuracy_first_then_lower_forgetting_then_grid_order(self):
        # equal accuracy for configs 1-3; 2 forgets least; 0 is less accurate despite no forgetting
        specs = [
            FakeAdapter(0.50, 0.0),
            FakeAdapter(0.80, 0.2),
            FakeAdapter(0.80, 0.1),
            FakeAdapter(0.80, 0.1),
        ]
        result = self.select(specs)
        self.assertEqual(
            result.ranking, [2, 3, 1, 0]
        )  # 2 before 3: equal everything, grid order
        self.assertEqual((result.selected_index, result.selected_config), (2, {"i": 2}))
        self.assertEqual([c["accuracy_rank"] for c in result.configs], [4, 3, 1, 2])

    def test_accuracy_gap_beyond_the_margin_beats_lower_forgetting(self):
        result = self.select(
            [FakeAdapter(0.60, 0.0), FakeAdapter(0.61, 0.3)]
        )  # gap 0.01 > 0.005
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.candidates_within_margin, [1])

    def test_within_the_margin_lower_forgetting_wins_even_at_lower_accuracy(self):
        result = self.select(
            [FakeAdapter(0.800, 0.2), FakeAdapter(0.796, 0.0)]
        )  # gap 0.004
        self.assertEqual(result.candidates_within_margin, [0, 1])
        self.assertEqual(result.selected_index, 1)
        self.assertAlmostEqual(result.max_mean_final_accuracy, 0.800)
        self.assertEqual(
            [c["within_margin_of_max"] for c in result.configs], [True, True]
        )
        self.assertEqual([c["selected"] for c in result.configs], [False, True])

    def test_margin_boundary_is_inclusive_and_just_outside_is_excluded(self):
        inside = self.select(
            [FakeAdapter(0.800, 0.2), FakeAdapter(0.795, 0.0)]
        )  # gap exactly 0.005
        self.assertEqual(inside.selected_index, 1)
        outside = self.select([FakeAdapter(0.800, 0.2), FakeAdapter(0.7949, 0.0)])
        self.assertEqual(outside.selected_index, 0)

    def test_margin_is_anchored_to_the_maximum_not_chained(self):
        # B is within 0.005 of A, C is within 0.005 of B but 0.008 from A: C is not a candidate
        result = self.select(
            [FakeAdapter(0.800, 0.2), FakeAdapter(0.796, 0.1), FakeAdapter(0.792, 0.0)]
        )
        self.assertEqual(result.candidates_within_margin, [0, 1])
        self.assertEqual(result.selected_index, 1)

    def test_an_ineligible_configuration_never_sets_the_maximum(self):
        result = self.select(
            [
                FakeAdapter(0.99, error=RuntimeError("x")),
                FakeAdapter(0.50, 0.3),
                FakeAdapter(0.498, 0.0),
            ]
        )
        self.assertAlmostEqual(result.max_mean_final_accuracy, 0.50)
        self.assertEqual(result.selected_index, 2)

    def test_context_is_recorded_verbatim(self):
        r = tuning.tune(
            "fake",
            lambda c: FakeAdapter(0.5),
            [{"x": 1}],
            self.dataset,
            self.protocol,
            context={"run_mode": "development"},
        )
        self.assertEqual(r.to_dict()["context"], {"run_mode": "development"})

    def test_failed_and_divergent_runs_are_preserved_and_ineligible(self):
        from polynomial_cl_bench.algorithms.er_mlp import TrainingDiverged

        result = self.select(
            [
                FakeAdapter(
                    0.9, error=TrainingDiverged("experience 2: non-finite loss")
                ),
                FakeAdapter(0.9, error=ValueError("bad config")),
                FakeAdapter(0.4),
            ]
        )
        statuses = [(r["config_index"], r["status"]) for r in result.runs]
        self.assertEqual(statuses[:2], [(0, "diverged")] * 2)
        self.assertEqual(statuses[2:4], [(1, "failed")] * 2)
        self.assertIn("non-finite loss", result.runs[0]["error"])
        self.assertEqual(len(result.runs), 6)  # nothing dropped
        self.assertEqual([c["eligible"] for c in result.configs], [False, False, True])
        self.assertEqual(
            result.selected_index, 2
        )  # the better-looking failures cannot win

    def test_one_failed_seed_makes_the_whole_configuration_ineligible(self):
        calls = {"n": 0}

        class Flaky(FakeAdapter):
            def run_sequence(self, dataset, protocol, seed, plan=None):
                calls["n"] += 1
                if seed == 101:
                    raise RuntimeError("boom")
                return super().run_sequence(dataset, protocol, seed, plan)

        result = self.select([Flaky(0.99), FakeAdapter(0.3)])
        self.assertEqual(result.configs[0]["statuses"], ["completed", "failed"])
        self.assertEqual(result.selected_index, 1)

    def test_no_eligible_configuration_selects_nothing_but_keeps_everything(self):
        result = self.select([FakeAdapter(0.5, error=RuntimeError("x"))])
        self.assertIsNone(result.selected_index)
        self.assertIsNone(result.selected_config)
        self.assertEqual(len(result.runs), 2)

    def test_result_serialises_strictly_without_nan(self):
        result = self.select(
            [FakeAdapter(0.7, 0.1), FakeAdapter(0.5, error=RuntimeError("x"))]
        )
        text = json.dumps(result.to_dict(), allow_nan=False)
        self.assertEqual(json.loads(text)["selected_index"], 0)

    def test_provenance_refusal_aborts_instead_of_being_recorded_as_a_failed_run(self):
        with self.assertRaises(ProvenanceError):
            self.select(
                [FakeAdapter(0.5, error=ProvenanceError("official run refused"))]
            )

    def test_every_run_is_on_the_validation_view_with_one_shared_plan_per_seed(self):
        FakeAdapter.log.clear()
        self.select([FakeAdapter(0.5), FakeAdapter(0.6), FakeAdapter(0.7)])
        view = validation_view(self.dataset, self.protocol)
        by_seed = {}
        for dataset, seed, plan in FakeAdapter.log:
            self.assertEqual(dataset.evaluation_split, "validation")
            self.assertEqual(dataset.fingerprint, view.fingerprint)
            by_seed.setdefault(seed, set()).add(id(plan))
        self.assertEqual(set(by_seed), {100, 101})
        self.assertTrue(
            all(len(ids) == 1 for ids in by_seed.values())
        )  # same object for all configs

    def test_selection_cannot_reach_the_real_test_arrays(self):
        poisoned = replace(
            self.dataset,
            test_x=np.full_like(self.dataset.test_x, np.nan),
            test_y=np.zeros_like(self.dataset.test_y),
        )
        grid = [{"hidden": [6], "learning_rate": lr} for lr in (0.02, 0.1)]
        build = lambda cfg: ERMLPAdapter(
            tuple(cfg["hidden"]), cfg["learning_rate"], role="validation_candidate"
        )
        clean = tuning.tune("er_mlp", build, grid, self.dataset, self.protocol)
        dirty = tuning.tune("er_mlp", build, grid, poisoned, self.protocol)
        # Everything except the measured times must be identical (times are never reproducible).
        timing = ("wall_time_seconds", "cpu_time_seconds")
        without_timing = lambda runs: [
            {k: v for k, v in r.items() if k not in timing} for r in runs
        ]
        self.assertEqual(
            without_timing(clean.to_dict()["runs"]),
            without_timing(dirty.to_dict()["runs"]),
        )
        self.assertTrue(all(r["status"] == "completed" for r in dirty.runs))
        self.assertEqual({r["evaluation_split"] for r in dirty.runs}, {"validation"})


class SameBudgetForBothAlgorithms(unittest.TestCase):
    """ER-MLP and the polynomial reference tuned with the same split, seeds, budget, criterion
    and (per seed) the very same plan."""

    def test_end_to_end_with_real_adapters(self):
        dataset = synthetic_dataset(train_per_class=40)
        protocol = small_protocol(epochs=1, batch_size=24, memory_per_class=8)
        plans_seen = {"er_mlp": [], "polynomial": []}

        def tracked(name, adapter):
            real = adapter.run_sequence

            def run(dataset, protocol, seed, plan=None):
                plans_seen[name].append((seed, plan.fingerprint()))
                return real(dataset, protocol, seed, plan)

            adapter.run_sequence = run
            return adapter

        mlp = tuning.tune(
            "er_mlp",
            lambda c: tracked(
                "er_mlp",
                ERMLPAdapter(
                    c["hidden"], c["learning_rate"], role="validation_candidate"
                ),
            ),
            [{"hidden": [8], "learning_rate": lr} for lr in (0.01, 0.1, 0.3)],
            dataset,
            protocol,
            expected_runs=6,
        )
        poly = tuning.tune(
            "polynomial_cl",
            lambda c: tracked(
                "polynomial",
                PolynomialAdapter(
                    learning_rate=c["learning_rate"], role="validation_candidate"
                ),
            ),
            [{"learning_rate": lr} for lr in (0.01, 0.1, 0.3)],
            dataset,
            protocol,
            expected_runs=6,
        )
        for result in (mlp, poly):
            self.assertEqual(
                (result.budget_runs, result.selection_seeds, result.criterion),
                (6, (100, 101), tuning.CRITERION),
            )
            self.assertIsNotNone(result.selected_index)
        self.assertEqual(mlp.validation_dataset, poly.validation_dataset)
        self.assertEqual(
            sorted(set(plans_seen["er_mlp"])), sorted(set(plans_seen["polynomial"]))
        )
        self.assertEqual(
            len(set(plans_seen["er_mlp"])), 2
        )  # one plan per selection seed


if __name__ == "__main__":
    unittest.main()
