# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Hyper-parameter selection on a validation hold-out, identical for every algorithm.

* Data: ``validation_view`` (training part to fit, hold-out of the *training* data to
  evaluate). The real test arrays are not reachable from here.
* Seeds: ``selection_seeds`` (default 100, 101) must not overlap the protocol's evaluation
  seeds. One shared ``ExperiencePlan`` per seed is reused by every configuration.
* Budget: ``len(grid) * len(selection_seeds)`` runs, optionally asserted via ``expected_runs``
  so ER-MLP and the polynomial reference are tuned under the same budget.
* Criterion (accuracy first, approved): let ``best`` be the highest mean final mean accuracy
  over the selection seeds among eligible configurations. Every configuration with mean accuracy
  within ``SELECTION_MARGIN`` (0.005) of ``best`` is a candidate; the candidate with the lowest
  mean forgetting is selected. Exact ties on forgetting go to the higher accuracy, then to the
  earlier position in the grid. A configuration is eligible only if **every** selection-seed run
  completed.
* Nothing is dropped: failed and divergent runs stay in the result with their error.

``ProvenanceError`` is never caught: an official-mode refusal aborts the whole tuning.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics
from .datasets import SplitMNIST, validation_view
from .plan import make_plan
from .protocol import Protocol
from .provenance import ProvenanceError

DEFAULT_SELECTION_SEEDS = (100, 101)
SELECTION_MARGIN = 0.005
CRITERION = (
    "candidates_within_0.005_of_max_mean_final_accuracy"
    "__select_min_mean_forgetting__then_higher_accuracy__then_grid_order"
)


class TuningError(ValueError):
    """The tuning request itself is invalid (not a failed run)."""


@dataclass
class TuningResult:
    algorithm: str
    criterion: str
    selection_seeds: tuple
    budget_runs: int
    protocol_fingerprint: str
    validation_dataset: dict
    grid: list
    runs: list  # every run, in (config, seed) order, including failed ones
    configs: list  # per configuration aggregate + eligibility
    selected_index: int | None
    selected_config: dict | None
    ranking: list = field(
        default_factory=list
    )  # eligible config indices by accuracy (descriptive)
    selection_margin: float = SELECTION_MARGIN
    max_mean_final_accuracy: float | None = None
    candidates_within_margin: list = field(default_factory=list)
    context: dict = field(
        default_factory=dict
    )  # run mode, provenance, grid file: set by caller

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "criterion": self.criterion,
            "selection_seeds": list(self.selection_seeds),
            "budget_runs": self.budget_runs,
            "protocol_fingerprint": self.protocol_fingerprint,
            "validation_dataset": self.validation_dataset,
            "grid": self.grid,
            "runs": self.runs,
            "configs": self.configs,
            "selected_index": self.selected_index,
            "selected_config": self.selected_config,
            "ranking": self.ranking,
            "selection_margin": self.selection_margin,
            "max_mean_final_accuracy": self.max_mean_final_accuracy,
            "candidates_within_margin": self.candidates_within_margin,
            "context": self.context,
        }

    def write_json(self, path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )


def _status_for(error: Exception) -> str:
    return "diverged" if type(error).__name__ == "TrainingDiverged" else "failed"


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def tune(
    algorithm: str,
    factory,
    grid,
    dataset: SplitMNIST,
    protocol: Protocol,
    *,
    selection_seeds=DEFAULT_SELECTION_SEEDS,
    expected_runs: int | None = None,
    margin: float = SELECTION_MARGIN,
    context: dict | None = None,
) -> TuningResult:
    """Run ``factory(config)`` for every grid entry and selection seed on the validation view.

    ``factory`` maps a grid entry (a dict) to an adapter; the adapter's ``run_sequence`` gets
    the validation view and the shared plan of that seed.
    """
    grid = [dict(entry) for entry in grid]
    selection_seeds = tuple(selection_seeds)
    if not grid:
        raise TuningError("the grid is empty")
    if not selection_seeds or len(set(selection_seeds)) != len(selection_seeds):
        raise TuningError("selection seeds must be distinct and non-empty")
    overlap = set(selection_seeds) & set(protocol.seeds)
    if overlap:
        raise TuningError(
            f"selection seeds {sorted(overlap)} overlap the evaluation seeds"
        )
    budget = len(grid) * len(selection_seeds)
    if expected_runs is not None and budget != expected_runs:
        raise TuningError(f"tuning budget is {budget} runs, expected {expected_runs}")

    view = validation_view(dataset, protocol)
    plans = {
        s: make_plan(view.train_y, s, protocol.memory_per_class)
        for s in selection_seeds
    }
    runs = []
    for index, entry in enumerate(grid):
        for seed in selection_seeds:
            base = {"config_index": index, "seed": seed, "config": entry}
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            try:
                result = factory(entry).run_sequence(view, protocol, seed, plans[seed])
                # Same definition as the runner's figures: includes in-loop evaluation and,
                # for the polynomial adapter, feature construction; excludes plan building.
                wall = time.perf_counter() - wall_start
                cpu = time.process_time() - cpu_start
                matrix = metrics.validate_matrix(result.accuracy_matrix)
                runs.append(
                    base
                    | {
                        "status": "completed",
                        "error": None,
                        "wall_time_seconds": wall,
                        "cpu_time_seconds": cpu,
                        "final_mean_accuracy": metrics.final_mean_accuracy(matrix),
                        "mean_forgetting": metrics.mean_forgetting(matrix),
                        "accuracy_matrix": [
                            [_finite(v) for v in row] for row in matrix
                        ],
                        "plan_fingerprint": result.run_config.get("plan_fingerprint"),
                        "evaluation_split": result.run_config["dataset"][
                            "evaluation_split"
                        ],
                    }
                )
            except ProvenanceError:
                raise
            except Exception as error:  # noqa: BLE001 - recorded, never hidden
                runs.append(
                    base
                    | {
                        "status": _status_for(error),
                        "error": f"{type(error).__name__}: {error}",
                        "wall_time_seconds": None,
                        "cpu_time_seconds": None,
                        "final_mean_accuracy": None,
                        "mean_forgetting": None,
                        "accuracy_matrix": None,
                        "plan_fingerprint": plans[seed].fingerprint(),
                        "evaluation_split": view.evaluation_split,
                    }
                )

    configs = []
    for index, entry in enumerate(grid):
        mine = [r for r in runs if r["config_index"] == index]
        eligible = all(r["status"] == "completed" for r in mine)
        configs.append(
            {
                "config_index": index,
                "config": entry,
                "statuses": [r["status"] for r in mine],
                "eligible": eligible,
                "mean_final_accuracy": float(
                    np.mean([r["final_mean_accuracy"] for r in mine])
                )
                if eligible
                else None,
                "mean_forgetting": float(np.mean([r["mean_forgetting"] for r in mine]))
                if eligible
                else None,
            }
        )
    eligible_ids = [c["config_index"] for c in configs if c["eligible"]]
    ranking = sorted(  # descriptive order by accuracy; the selection rule is below
        eligible_ids,
        key=lambda i: (
            -configs[i]["mean_final_accuracy"],
            configs[i]["mean_forgetting"],
            i,
        ),
    )
    best = candidates = selected = None
    if eligible_ids:
        best = max(configs[i]["mean_final_accuracy"] for i in eligible_ids)
        candidates = [
            i
            for i in eligible_ids
            if configs[i]["mean_final_accuracy"] >= best - margin - 1e-12
        ]
        selected = min(
            candidates,
            key=lambda i: (
                configs[i]["mean_forgetting"],
                -configs[i]["mean_final_accuracy"],
                i,
            ),
        )
    for c in configs:
        c["accuracy_rank"] = (
            ranking.index(c["config_index"]) + 1 if c["eligible"] else None
        )
        c["within_margin_of_max"] = bool(candidates and c["config_index"] in candidates)
        c["selected"] = c["config_index"] == selected
    return TuningResult(
        algorithm=algorithm,
        criterion=CRITERION,
        selection_seeds=selection_seeds,
        budget_runs=budget,
        protocol_fingerprint=protocol.fingerprint(),
        validation_dataset={
            "fingerprint": view.fingerprint,
            "derived_from": view.derived_from,
            "evaluation_split": view.evaluation_split,
            "n_train": len(view.train_y),
            "n_validation": len(view.test_y),
            "holdout_fraction": protocol.validation["holdout_fraction"],
            "holdout_seed": protocol.validation["holdout_seed"],
        },
        grid=grid,
        runs=runs,
        configs=configs,
        selected_index=selected,
        selected_config=None if selected is None else grid[selected],
        ranking=ranking,
        selection_margin=margin,
        max_mean_final_accuracy=best,
        candidates_within_margin=candidates or [],
        context=dict(context or {}),
    )
