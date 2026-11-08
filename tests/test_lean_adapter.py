# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The adapter runs the core's lean benchmark mode without changing what is learned.

Cost accounting only: the accuracy matrix of a lean run must equal that of a run with the
core's diagnostics switched on, resources must say how the run was executed, and an older core
without the lean mode or timings must still work, with the unavailable values reported as None.
"""

import json
import unittest
from unittest import mock

import numpy as np
from helpers import small_protocol, synthetic_dataset

from polynomial_cl_bench import results
from polynomial_cl_bench.algorithms import polynomial as module
from polynomial_cl_bench.algorithms.polynomial import (
    CORE_HAS_LEAN_MODE,
    CORE_REPORTS_TIMINGS,
    PolynomialAdapter,
)

needs_lean_core = unittest.skipUnless(
    CORE_HAS_LEAN_MODE, "installed core has no lean mode"
)
needs_timing_core = unittest.skipUnless(
    CORE_REPORTS_TIMINGS, "installed core cannot report timings"
)


class Spy:
    def __init__(self):
        self.kwargs = []
        self.real = module.run_condition

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        return self.real(*args, **kwargs)


class LeanAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol()

    def run_adapter(self, adapter=None, seed=0):
        adapter = adapter or PolynomialAdapter()
        return adapter.run_sequence(self.dataset, self.protocol, seed)

    @needs_lean_core
    def test_benchmark_runs_use_the_lean_mode_by_default(self):
        spy = Spy()
        with mock.patch.object(module, "run_condition", spy):
            result = self.run_adapter()
        self.assertIs(spy.kwargs[0]["diagnostics"], False)
        self.assertFalse(result.resources["execution"]["core_diagnostics"])

    @needs_lean_core
    def test_diagnostics_can_be_switched_on_for_research(self):
        spy = Spy()
        with mock.patch.object(module, "run_condition", spy):
            result = self.run_adapter(PolynomialAdapter(diagnostics=True))
        self.assertIs(spy.kwargs[0]["diagnostics"], True)
        self.assertTrue(result.resources["execution"]["core_diagnostics"])

    @needs_lean_core
    def test_lean_and_full_runs_have_identical_accuracy_matrices(self):
        for seed in (0, 1):
            lean = self.run_adapter(PolynomialAdapter(), seed)
            full = self.run_adapter(PolynomialAdapter(diagnostics=True), seed)
            self.assertTrue(
                np.array_equal(
                    lean.accuracy_matrix, full.accuracy_matrix, equal_nan=True
                )
            )
            self.assertEqual(lean.extras["core_reported"], full.extras["core_reported"])

    def test_diagnostics_are_not_part_of_the_algorithm_identity(self):
        lean, full = PolynomialAdapter(), PolynomialAdapter(diagnostics=True)
        self.assertEqual(lean.describe(), full.describe())

    @needs_timing_core
    def test_training_and_inference_time_come_from_the_core_split(self):
        r = self.run_adapter().resources
        parts = r["training_time_seconds"] + r["inference_time_seconds"]
        self.assertGreater(r["training_time_seconds"], 0)
        self.assertGreater(r["inference_time_seconds"], 0)
        self.assertLessEqual(parts, r["wall_time_seconds_including_evaluation"])
        timings = r["execution"]["core_timings_seconds"]
        self.assertEqual(timings["diagnostics_seconds"], 0.0)

    def test_feature_construction_is_reported_once_per_dataset(self):
        adapter = PolynomialAdapter()
        first = self.run_adapter(adapter, 0).resources
        again = self.run_adapter(adapter, 1).resources
        self.assertGreater(first["feature_construction_seconds"], 0)
        self.assertFalse(first["execution"]["features_from_cache"])
        self.assertEqual(again["feature_construction_seconds"], 0.0)
        self.assertTrue(again["execution"]["features_from_cache"])

    def test_features_prepared_elsewhere_count_as_no_construction(self):
        donor = PolynomialAdapter()
        donor._feature_data(self.dataset)
        adapter = PolynomialAdapter()
        adapter._features = donor._features  # what tools/run_comparison_study.py does
        r = self.run_adapter(adapter).resources
        self.assertEqual(r["feature_construction_seconds"], 0.0)

    def test_resources_are_strict_json(self):
        json.dumps(self.run_adapter().resources, allow_nan=False)

    def test_an_older_core_without_lean_mode_or_timings_still_works(self):
        spy = Spy()
        with (
            mock.patch.object(module, "CORE_HAS_LEAN_MODE", False),
            mock.patch.object(module, "CORE_REPORTS_TIMINGS", False),
            mock.patch.object(module, "run_condition", spy),
        ):
            r = self.run_adapter().resources
        self.assertNotIn("diagnostics", spy.kwargs[0])
        self.assertNotIn("timings", spy.kwargs[0])
        self.assertIsNone(r["training_time_seconds"])
        self.assertIsNone(r["inference_time_seconds"])
        self.assertIsNone(r["execution"]["core_timings_seconds"])
        self.assertFalse(r["execution"]["core_lean_mode_available"])
        self.assertTrue(r["execution"]["core_diagnostics"])  # what really ran


class ResourceSchemaTests(unittest.TestCase):
    def test_new_resource_fields_are_optional_and_validated_when_present(self):
        self.assertEqual(
            results._OPTIONAL_NUMERIC_RESOURCES, ("feature_construction_seconds",)
        )

    def test_runner_defaults_carry_the_new_fields_as_unavailable(self):
        from polynomial_cl_bench import runner

        self.assertIsNone(runner._RESOURCE_DEFAULTS["feature_construction_seconds"])
        self.assertIsNone(runner._RESOURCE_DEFAULTS["adapter_execution"])


if __name__ == "__main__":
    unittest.main()
