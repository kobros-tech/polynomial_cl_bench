# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The adapter's support for the core's degree control (per-degree factors on the rates).

The identity control must leave everything unchanged (same accuracy matrix, same record keys),
a real control must reach the core and be recorded, invalid or unsupported controls must be
refused before any training, and a core without the feature must still run every
configuration that does not use it.
"""

import unittest
from unittest import mock

import numpy as np
from helpers import small_protocol, synthetic_dataset

from polynomial_cl_bench.algorithms import polynomial as module
from polynomial_cl_bench.algorithms.polynomial import (
    CORE_HAS_DEGREE_CONTROL,
    AdapterError,
    PolynomialAdapter,
)

needs_control_core = unittest.skipUnless(
    CORE_HAS_DEGREE_CONTROL, "installed core has no degree control"
)
RATES = [0.75, 1.5, 1.5, 6.0]


class Capture:
    """Forwards to the real run_condition and keeps the keyword arguments."""

    def __init__(self):
        self.kwargs = []
        self.real = module.run_condition

    def __call__(self, *args, **kwargs):
        self.kwargs.append(dict(kwargs))
        return self.real(*args, **kwargs)


class ConfigurationTests(unittest.TestCase):
    def test_identity_is_stored_as_no_control(self):
        adapter = PolynomialAdapter(degree_control_factors=[1, 1, 1, 1])
        self.assertIsNone(adapter.degree_control_factors)
        self.assertNotIn("degree_control_factors", adapter.describe()["configuration"])

    def test_invalid_controls_are_refused(self):
        for bad in (
            [1, 1, 1],
            [1, 1, 1, 1, 1],
            [[1, 1], [1, 1]],
            [1, 1, 1, 1.5],
            [1, 1, 1, -0.1],
            [1, 1, 1, float("nan")],
            [1, 1, 1, float("inf")],
            [1, 1, 1, True],
            [1, 1, 1, "0.5"],
            [1, 1, 1, None],
            0.5,
        ):
            with self.subTest(bad=bad), self.assertRaises(AdapterError):
                PolynomialAdapter(degree_control_factors=bad)

    @needs_control_core
    def test_a_real_control_is_refused_with_feature_power(self):
        with self.assertRaises(AdapterError):
            PolynomialAdapter(feature_power=0.5, degree_control_factors=[1, 0.5, 1, 1])

    @needs_control_core
    def test_identity_is_accepted_with_feature_power(self):
        PolynomialAdapter(feature_power=0.5, degree_control_factors=[1, 1, 1, 1])

    def test_a_core_without_degree_control_refuses_a_real_control(self):
        with mock.patch.object(module, "CORE_HAS_DEGREE_CONTROL", False):
            with self.assertRaises(AdapterError):
                PolynomialAdapter(degree_control_factors=[1, 0.5, 1, 1])
            PolynomialAdapter(degree_control_factors=[1, 1, 1, 1])  # identity: fine
            PolynomialAdapter()  # nothing changes for existing configurations

    def test_from_dict_reads_the_key(self):
        raw = {
            "algorithm": "polynomial_cl",
            "condition": "expanding",
            "strategy": "ordinary",
            "learning_rate": RATES,
            "degree_control_factors": [1, 0.25, 0, 1],
        }
        if CORE_HAS_DEGREE_CONTROL:
            adapter = PolynomialAdapter.from_dict(raw)
            self.assertEqual(adapter.degree_control_factors, (1.0, 0.25, 0.0, 1.0))
        else:
            with self.assertRaises(AdapterError):
                PolynomialAdapter.from_dict(raw)


@needs_control_core
class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.protocol = small_protocol()

    def run_adapter(self, **kwargs):
        return PolynomialAdapter(learning_rate=RATES, **kwargs).run_sequence(
            self.dataset, self.protocol, seed=0
        )

    def test_identity_reproduces_the_uncontrolled_run(self):
        plain = self.run_adapter()
        identity = self.run_adapter(degree_control_factors=[1, 1, 1, 1])
        self.assertTrue(
            np.array_equal(
                plain.accuracy_matrix, identity.accuracy_matrix, equal_nan=True
            )
        )
        self.assertNotIn("degree_control_factors", identity.run_config)
        self.assertEqual(plain.run_config, identity.run_config)

    def test_a_control_reaches_the_core_and_is_recorded(self):
        spy = Capture()
        with mock.patch.object(module, "run_condition", spy):
            result = self.run_adapter(degree_control_factors=[1, 0.25, 0, 1])
        self.assertEqual(spy.kwargs[0]["degree_control_factors"], [1.0, 0.25, 0.0, 1.0])
        self.assertEqual(
            result.run_config["degree_control_factors"], [1.0, 0.25, 0.0, 1.0]
        )
        self.assertEqual(
            result.algorithm["configuration"]["degree_control_factors"],
            [1.0, 0.25, 0.0, 1.0],
        )

    def test_a_control_changes_what_is_learned(self):
        plain = self.run_adapter()
        controlled = self.run_adapter(degree_control_factors=[1, 0.05, 0, 1])
        self.assertFalse(
            np.array_equal(
                plain.accuracy_matrix, controlled.accuracy_matrix, equal_nan=True
            )
        )

    def test_a_scalar_rate_is_accepted_with_a_control(self):
        # The core then reports a per-degree vector; the adapter must not mistake that for
        # a different rate.
        result = PolynomialAdapter(
            learning_rate=1.0, degree_control_factors=[1, 0.5, 1, 1]
        ).run_sequence(self.dataset, self.protocol, seed=0)
        self.assertEqual(result.run_config["learning_rate"], 1.0)

    def test_a_core_that_ignores_the_control_is_caught(self):
        real = module.run_condition

        def ignoring(*args, **kwargs):
            kwargs.pop("degree_control_factors", None)
            return real(*args, **kwargs)

        with mock.patch.object(module, "run_condition", ignoring):
            with self.assertRaises(AdapterError):
                self.run_adapter(degree_control_factors=[1, 0.5, 1, 1])

    def test_a_core_reporting_other_factors_is_caught(self):
        real = module.run_condition

        def other(*args, **kwargs):
            record = real(*args, **kwargs)
            record["degree_control_factors"] = [1.0, 1.0, 1.0, 1.0]
            return record

        with mock.patch.object(module, "run_condition", other):
            with self.assertRaises(AdapterError):
                self.run_adapter(degree_control_factors=[1, 0.5, 1, 1])


if __name__ == "__main__":
    unittest.main()
