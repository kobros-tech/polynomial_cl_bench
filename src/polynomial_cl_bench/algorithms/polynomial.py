"""Adapter wrapping a complete ``polynomial_cl`` class-incremental run.

The core exposes ``run_condition``, which trains all experiences and evaluates after each in a
single call (no per-experience hook, no predict). This adapter therefore wraps one whole run. It
reimplements nothing: features, training, replay and update rules all come from the core.

Limits that follow from that (see README): the core evaluates internally, so the adapter cannot
prove from outside that test data is untouched during training; per-class accuracy, separate
inference time and peak memory are not available and are reported as None.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from polynomial_cl.class_incremental import (
    TASK_ORDER,
    make_feature_data,
    make_task_plan,
    run_condition,
)
from polynomial_cl.class_incremental.experiment import CONDITIONS, STRATEGIES
from polynomial_cl.class_incremental.training import SAMPLING_POLICIES

from ..datasets import SplitMNIST
from ..protocol import Protocol
from .base import SequenceResult

N_OUTPUTS = 10  # the core's single softmax head


class AdapterError(RuntimeError):
    """The adapter cannot honour the protocol."""


ROLES = ("primary_reference", "experimental")


def commit_from_direct_url(direct_url: dict | None) -> str | None:
    """Commit recorded by pip for a VCS install (PEP 610), else None."""
    if not direct_url:
        return None
    return (direct_url.get("vcs_info") or {}).get("commit_id")


def pinned_core_commit() -> str | None:
    """The commit this benchmark's own metadata pins the core to, if parseable."""
    for requirement in metadata.requires("polynomial-cl-bench") or []:
        match = re.match(r"polynomial-cl\s*@\s*git\+\S+@([0-9a-f]{40})\b", requirement)
        if match:
            return match.group(1)
    return None


def core_provenance() -> dict:
    """Which core was imported, and how much of that claim is verified.

    ``commit_source`` says where the commit came from:

    * ``pip_vcs_install``: pip installed it from ``remote_url`` and recorded the resolved commit
      (PEP 610). ``remote_verified`` is True: the commit was fetched from that remote.
    * ``local_git`` / ``local_git_dirty``: read with ``git rev-parse`` from a local checkout
      (editable or path install). The commit exists locally; **nothing here shows it exists on
      any remote**, so ``remote_verified`` is False. ``_dirty`` means tracked files are modified.
    * ``unknown``: no commit could be determined.

    ``matches_pin`` compares the commit with the pin in this package's metadata (None if either
    side is unknown); a mismatch means the run does not use the pinned core.
    """
    dist = metadata.distribution("polynomial-cl")
    text = dist.read_text("direct_url.json")
    direct_url = json.loads(text) if text else None
    commit = commit_from_direct_url(direct_url)
    source, remote_url = "unknown", None
    if commit is not None:
        source, remote_url = "pip_vcs_install", (direct_url or {}).get("url")
    else:
        url = (direct_url or {}).get("url", "")
        if url.startswith("file://"):
            directory = Path(url[len("file://"):])
            try:
                commit = subprocess.run(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True, timeout=30,
                ).stdout.strip()
                dirty = subprocess.run(
                    ["git", "-C", str(directory), "status", "--porcelain",
                     "--untracked-files=no"],
                    capture_output=True, text=True, check=True, timeout=30,
                ).stdout.strip()
                source = "local_git_dirty" if dirty else "local_git"
            except (OSError, subprocess.SubprocessError):
                commit = None
    pinned = pinned_core_commit()
    return {
        "version": dist.version,
        "commit": commit,
        "commit_source": source,
        "remote_url": remote_url,
        "remote_verified": source == "pip_vcs_install",
        "pinned_commit": pinned,
        "matches_pin": None if commit is None or pinned is None else commit == pinned,
    }


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
    ):
        if condition not in CONDITIONS:
            raise AdapterError(f"unknown condition {condition!r}; core supports {CONDITIONS}")
        if strategy not in STRATEGIES:
            raise AdapterError(f"unknown strategy {strategy!r}; core supports {STRATEGIES}")
        if role not in ROLES:
            raise AdapterError(f"role must be one of {ROLES}")
        self.condition = condition
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.feature_power = feature_power
        self.hyperparameter_provenance = hyperparameter_provenance or {}
        self.role = role
        self._features = None  # (dataset fingerprint, FeatureData); read-only, shared by runs

    @classmethod
    def from_file(cls, path) -> "PolynomialAdapter":
        raw = json.loads(Path(path).read_text())
        if raw.get("algorithm") != "polynomial_cl":
            raise AdapterError("not a polynomial_cl algorithm configuration")
        return cls(
            condition=raw["condition"],
            strategy=raw["strategy"],
            learning_rate=raw["learning_rate"],
            feature_power=raw.get("feature_power", 0.0),
            hyperparameter_provenance=raw.get("hyperparameter_provenance"),
            role=raw.get("role", "experimental"),
        )

    def describe(self) -> dict:
        learning_rate = (
            [float(v) for v in np.ravel(self.learning_rate)]
            if np.ndim(self.learning_rate)
            else float(self.learning_rate)
        )
        return {
            "name": self.name,
            "role": self.role,
            "condition": self.condition,
            "strategy": self.strategy,
            "configuration": {
                "learning_rate": learning_rate,
                "feature_power": float(self.feature_power),
            },
            "hyperparameter_provenance": self.hyperparameter_provenance,
            "core": core_provenance(),
        }

    def _feature_data(self, dataset: SplitMNIST):
        """Degree-3 features, computed once per dataset and reused read-only across seeds."""
        if self._features is None or self._features[0] != dataset.fingerprint:
            data = make_feature_data(
                dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y
            )
            for array in (data.train_x, data.train_y, data.test_x, data.test_y):
                array.setflags(write=False)
            self._features = (dataset.fingerprint, data)
        return self._features[1]

    def run_sequence(self, dataset: SplitMNIST, protocol: Protocol, seed: int) -> SequenceResult:
        if tuple(tuple(t) for t in protocol.task_order) != tuple(TASK_ORDER):
            raise AdapterError(
                "the core's task order is fixed to "
                f"{TASK_ORDER}; the protocol asks for {protocol.task_order}"
            )
        if protocol.sampling not in SAMPLING_POLICIES:
            raise AdapterError(
                f"core does not support sampling {protocol.sampling!r}; "
                f"supported: {SAMPLING_POLICIES}"
            )
        if dataset.projection_seed != protocol.projection_seed:
            raise AdapterError(
                f"dataset was projected with seed {dataset.projection_seed}; "
                f"the protocol requires {protocol.projection_seed}"
            )
        if dataset.source_hashes != protocol.dataset["files"]:
            raise AdapterError("dataset was not built from the protocol's verified files")
        data = self._feature_data(dataset)
        tasks, memory = make_task_plan(data.train_y, seed, protocol.memory_per_class)
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
        )
        elapsed = time.perf_counter() - started
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
            raise AdapterError("core ran a different condition or strategy than configured")
        matrix = np.array(
            [[np.nan if v is None else v for v in row] for row in record["accuracy_matrix"]],
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
                # Includes the core's in-loop evaluation and diagnostics; not training alone.
                "wall_time_seconds_including_evaluation": elapsed,
                "training_time_seconds": None,
                "inference_time_seconds": None,
                "peak_memory_mb": None,
                "n_parameters": int(record["feature_count_by_task"][-1] * N_OUTPUTS),
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
                },
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
