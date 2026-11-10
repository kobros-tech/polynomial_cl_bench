# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Adapter wrapping a complete ``polynomial_cl`` class-incremental run.

The core exposes ``run_condition``, which trains all experiences and evaluates after each in a
single call (no per-experience hook, no predict). This adapter therefore wraps one whole run. It
reimplements nothing: features, training, replay and update rules all come from the core.

Limits that follow from that (see README): the core evaluates internally, so the adapter cannot
prove from outside that test data is untouched during training; per-class accuracy and peak
memory are not available and are reported as None.

Cost accounting. The adapter runs the core in its lean benchmark mode (``diagnostics=False``):
the benchmark reads only the accuracy matrix and the core's accuracy/forgetting summary, so the
core's observation-only work (loss passes, per-degree statistics, old-class indicators) would
only add time that ER-MLP does not pay. That work never influences learning, so the accuracy
matrix is bit-for-bit the same either way (checked in ``tests/test_lean_adapter.py``).
Where the installed core can report it (``timings=``), ``training_time_seconds`` is the time in
the core's trainer and ``inference_time_seconds`` the time spent building the accuracy matrix;
an older core cannot, and both stay None. The one-off construction of the degree-3 design
matrix is reported separately as ``feature_construction_seconds`` (it is part of the runner's
wall time, and is only paid again by a new process or a new dataset).
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
from polynomial_cl.continual import (
    CONDITIONS,
    STRATEGIES,
    make_feature_data,
    run_condition,
)
from polynomial_cl.continual.sampling import SAMPLING_POLICIES

from ..datasets import SplitMNIST
from ..plan import check_inputs, resolve_plan
from ..protocol import Protocol
from ..provenance import (
    RUN_MODES,
    benchmark_provenance,
    core_provenance,
    enforce_run_mode,
)
from .base import AdapterError, SequenceResult  # AdapterError re-exported for importers

N_OUTPUTS = 10  # the core's single softmax head

# What the installed core offers, read once from the real function (tests patch the module's
# ``run_condition`` with spies that forward to it, so this must not look at the patched name).
_CORE_PARAMETERS = inspect.signature(run_condition).parameters
CORE_HAS_LEAN_MODE = "diagnostics" in _CORE_PARAMETERS
CORE_REPORTS_TIMINGS = "timings" in _CORE_PARAMETERS

ROLES = ("primary_reference", "experimental", "validation_candidate")


class PolynomialAdapter:
    name = "polynomial_cl"

    def __init__(
        self,
        condition: str = "expanding",
        strategy: str = "ordinary",
        learning_rate: float = 0.1,
        feature_power: float = 0.0,
        hyperparameter_provenance: dict | None = None,
        role: str = "experimental",
        mode: str = "development",
        diagnostics: bool = False,
    ):
        if condition not in CONDITIONS:
            raise AdapterError(
                f"unknown condition {condition!r}; core supports {CONDITIONS}"
            )
        if strategy not in STRATEGIES:
            raise AdapterError(
                f"unknown strategy {strategy!r}; core supports {STRATEGIES}"
            )
        if mode not in RUN_MODES:
            raise AdapterError(f"mode must be one of {RUN_MODES}")
        if role not in ROLES:
            raise AdapterError(f"role must be one of {ROLES}")
        # The core's class-incremental run_condition does not validate feature_power (a negative
        # value would silently invert the per-feature step sizes), so the adapter does.
        if (
            np.ndim(feature_power)
            or not np.isfinite(feature_power)
            or feature_power < 0
        ):
            raise AdapterError("feature_power must be a finite, non-negative scalar")
        self.condition = condition
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.feature_power = feature_power
        self.hyperparameter_provenance = hyperparameter_provenance or {}
        self.role = role
        self.mode = mode
        # Execution detail, not part of the configuration or of a run's identity: the core's
        # observation-only diagnostics never change what is learned. Off for benchmarking;
        # switch on only to inspect the polynomial model itself.
        self.diagnostics = bool(diagnostics)
        self._features = (
            None  # (dataset fingerprint, FeatureData); read-only, shared by runs
        )
        self._feature_seconds = 0.0  # time spent constructing features in the last call

    @classmethod
    def from_dict(cls, raw: dict, mode: str = "development") -> PolynomialAdapter:
        if raw.get("algorithm") != "polynomial_cl":
            raise AdapterError("not a polynomial_cl algorithm configuration")
        return cls(
            condition=raw["condition"],
            strategy=raw["strategy"],
            learning_rate=raw["learning_rate"],
            feature_power=raw.get("feature_power", 0.0),
            hyperparameter_provenance=raw.get("hyperparameter_provenance"),
            role=raw.get("role", "experimental"),
            mode=mode,
        )

    @classmethod
    def from_file(cls, path, mode: str = "development") -> PolynomialAdapter:
        return cls.from_dict(json.loads(Path(path).read_text()), mode=mode)

    def describe(self) -> dict:
        learning_rate = (
            [float(v) for v in np.ravel(self.learning_rate)]
            if np.ndim(self.learning_rate)
            else float(self.learning_rate)
        )
        return {
            "name": self.name,
            "role": self.role,
            "run_mode": self.mode,
            "condition": self.condition,
            "strategy": self.strategy,
            "configuration": {
                "learning_rate": learning_rate,
                "feature_power": float(self.feature_power),
            },
            "hyperparameter_provenance": self.hyperparameter_provenance,
            "core": core_provenance(),
            "benchmark": benchmark_provenance(),
        }

    def _feature_data(self, dataset: SplitMNIST):
        """Degree-3 features, computed once per dataset and reused read-only across seeds.

        ``self._feature_seconds`` is the construction time of this call: 0.0 when the
        features were already available (a repeated seed in the same process).
        """
        self._feature_seconds = 0.0
        if self._features is None or self._features[0] != dataset.fingerprint:
            started = time.perf_counter()
            data = make_feature_data(
                dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y
            )
            for array in (data.train_x, data.train_y, data.test_x, data.test_y):
                array.setflags(write=False)
            self._feature_seconds = time.perf_counter() - started
            self._features = (dataset.fingerprint, data)
        return self._features[1]

    def run_sequence(
        self, dataset: SplitMNIST, protocol: Protocol, seed: int, plan=None
    ) -> SequenceResult:
        # Official mode raises before any training unless core and benchmark are verified.
        enforce_run_mode(self.mode, core_provenance(), benchmark_provenance())
        check_inputs(dataset, protocol)
        if protocol.sampling not in SAMPLING_POLICIES:
            raise AdapterError(
                f"core does not support sampling {protocol.sampling!r}; "
                f"supported: {SAMPLING_POLICIES}"
            )
        data = self._feature_data(dataset)
        feature_seconds = self._feature_seconds
        plan = resolve_plan(
            dataset, protocol, seed, plan
        )  # shared, or the identical own plan
        tasks, memory = list(plan.tasks), dict(plan.memory)
        core_options, timings = {}, {}
        if CORE_HAS_LEAN_MODE:
            core_options["diagnostics"] = self.diagnostics
        if CORE_REPORTS_TIMINGS:
            core_options["timings"] = timings
        started = time.perf_counter()
        record = run_condition(
            self.condition,
            data,
            tasks,
            memory,
            seed=seed,
            epochs=protocol.epochs,
            learning_rate=self.learning_rate,
            batch_size=protocol.batch_size,
            strategy=self.strategy,
            feature_power=self.feature_power,
            sampling=protocol.sampling,  # always passed explicitly
            **core_options,
        )
        elapsed = time.perf_counter() - started
        if (
            CORE_HAS_LEAN_MODE
            and bool(record.get("diagnostics", True)) != self.diagnostics
        ):
            # Diagnostics roughly double a run's time and differ by update rule, so a run
            # that silently used the other setting would distort the cost comparison with
            # ER-MLP. The setting is passed explicitly, so this can only fire if a core
            # ignores or changes it.
            raise AdapterError(
                f"core ran with diagnostics={record.get('diagnostics', True)!r}, "
                f"configured {self.diagnostics!r}"
            )
        if record["sampling"] != protocol.sampling:
            raise AdapterError(
                f"core reports sampling {record['sampling']!r}, protocol requires "
                f"{protocol.sampling!r}"
            )
        expected_rate = self.describe()["configuration"]["learning_rate"]
        if record["learning_rate"] != expected_rate:
            raise AdapterError(
                f"core ran with learning_rate {record['learning_rate']!r}, "
                f"configured {expected_rate!r}"
            )
        if (record["condition"], record["strategy"]) != (self.condition, self.strategy):
            raise AdapterError(
                "core ran a different condition or strategy than configured"
            )
        matrix = np.array(
            [
                [np.nan if v is None else v for v in row]
                for row in record["accuracy_matrix"]
            ],
            dtype=float,
        )
        replays = self.strategy != "ordinary_no_replay"
        replay_used = [
            sum(len(memory[c]) for c in range(2 * t)) if replays else 0
            for t in range(len(tasks))
        ]
        return SequenceResult(
            accuracy_matrix=matrix,
            algorithm=self.describe(),
            resources={
                # Includes the core's in-loop evaluation (and its diagnostics when switched
                # on); not training alone. Feature construction is outside this figure.
                "wall_time_seconds_including_evaluation": elapsed,
                # From the core's own split of the run; None when the core cannot report it.
                "training_time_seconds": timings.get("training_seconds"),
                "inference_time_seconds": timings.get("evaluation_seconds"),
                "feature_construction_seconds": feature_seconds,
                "peak_memory_mb": None,
                "n_parameters": int(record["feature_count_by_task"][-1] * N_OUTPUTS),
                # How this run was executed (not what it learned); stored with the record.
                "execution": {
                    "core_lean_mode_available": CORE_HAS_LEAN_MODE,
                    "core_diagnostics": bool(record.get("diagnostics", True)),
                    "features_from_cache": feature_seconds == 0.0,
                    "core_timings_seconds": dict(timings) if timings else None,
                },
            },
            sampling=record["sampling"],
            run_config={
                "seed": seed,
                "protocol": {
                    "name": protocol.name,
                    "version": protocol.version,
                    "fingerprint": protocol.fingerprint(),
                },
                "dataset": {
                    "name": protocol.dataset["name"],
                    "fingerprint": dataset.fingerprint,
                    "source_hashes": dict(dataset.source_hashes),
                    "projection_seed": dataset.projection_seed,
                    "evaluation_split": dataset.evaluation_split,
                },
                "plan_fingerprint": plan.fingerprint(),
                "task_order": [list(t) for t in protocol.task_order],
                "condition": self.condition,
                "strategy": self.strategy,
                "learning_rate": expected_rate,
                "feature_power": float(self.feature_power),
                "epochs": protocol.epochs,
                "batch_size": protocol.batch_size,
                "sampling": record["sampling"],
                "memory_per_class": protocol.memory_per_class,
                "role": self.role,
                "run_mode": self.mode,
            },
            replay_examples_used_by_task=replay_used,
            extras={
                "core_reported": {
                    k: record[k]
                    for k in (
                        "final_mean_accuracy",
                        "mean_forgetting",
                        "forgetting_by_task",
                        "current_accuracy_by_task",
                        "mean_accuracy_by_task",
                        "feature_count_by_task",
                        "active_degree_by_task",
                    )
                },
            },
        )
