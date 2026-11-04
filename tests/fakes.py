# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Fake adapters and plans so runner behaviour can be tested without the core or real training."""

from typing import ClassVar

import numpy as np

from polynomial_cl_bench.algorithms.base import SequenceResult


class TrainingDiverged(
    RuntimeError
):  # name matters: the runner maps it to status 'diverged'
    pass


class FakePlan:
    def __init__(self, seed):
        self.seed = seed

    def fingerprint(self):
        return f"plan-seed-{self.seed}"


def fake_plan_factory(dataset, protocol, seed):
    return FakePlan(seed)


def lower_triangular(skill, seed, n=5):
    rng = np.random.default_rng(seed)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1):
            m[i, j] = float(np.clip(skill - 0.05 * (i - j) + rng.normal(0, 0.01), 0, 1))
    return m


class FakeAdapter:
    calls: ClassVar[
        list
    ] = []  # (label-ish, seed) per run_sequence call, for resume tests

    def __init__(self, config, mode):
        self.config, self.mode = config, mode
        self.behavior = config.get("behavior", "ok")

    def describe(self):
        return {
            "name": self.config["algorithm"],
            "role": "primary",
            "configuration": {
                k: v for k, v in self.config.items() if k not in ("algorithm", "role")
            },
            "hyperparameter_provenance": self.config.get(
                "hyperparameter_provenance", {}
            ),
            "core": {},
            "benchmark": {},
        }

    def run_sequence(self, dataset, protocol, seed, plan=None):
        FakeAdapter.calls.append(
            (self.config["algorithm"], self.config.get("skill"), seed)
        )
        b = self.behavior
        if b == "raise" or (b == "raise_seed1" and seed == 1):
            raise ValueError("boom")
        if b == "diverge":
            raise TrainingDiverged("experience 2: non-finite loss at update 3")
        matrix = lower_triangular(
            self.config.get("skill", 0.8), seed, len(protocol.task_order)
        )
        if b == "nondeterministic":
            matrix = matrix + np.where(
                np.isnan(matrix),
                0,
                np.random.default_rng().normal(0, 1e-6, matrix.shape),
            )
        if b == "bad_matrix":
            matrix[0, 0] = 1.5
        replay = [
            protocol.memory_per_class * 2 * t for t in range(len(protocol.task_order))
        ]
        if b == "over_budget":
            replay[-1] += 1
        plan_fp = "other-plan" if b == "wrong_plan" else plan.fingerprint()
        return SequenceResult(
            accuracy_matrix=matrix,
            algorithm=self.describe(),
            resources={
                "wall_time_seconds_including_evaluation": 0.5,
                "training_time_seconds": None
                if self.config.get("no_split_time")
                else 0.3,
                "inference_time_seconds": None
                if self.config.get("no_split_time")
                else 0.1,
                "peak_memory_mb": None,
                "n_parameters": 1000,
            },
            sampling=protocol.sampling,
            run_config={
                "seed": seed,
                "protocol": {"fingerprint": protocol.fingerprint()},
                "dataset": {
                    "fingerprint": dataset.fingerprint,
                    "evaluation_split": dataset.evaluation_split,
                },
                "plan_fingerprint": plan_fp,
            },
            replay_examples_used_by_task=replay,
        )


def fake_factory(config, mode):
    return FakeAdapter(config, mode)
