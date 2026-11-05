# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Experience Replay with a small MLP (ER-MLP): the first independent competitor.

Benchmark-owned NumPy implementation; it shares nothing with the polynomial model except the
protocol, the projected 16-dimensional inputs and the experience/replay plan.

Design (concatenated-replay form)
---------------------------------
* Input: the protocol's projected 16-d vectors, unchanged. Model: ``16 -> hidden... -> 10``
  with ReLU hidden layers and one 10-way head (``hidden=()`` is a linear softmax head).
* Per experience the current experience's examples and the replay exemplars of earlier
  classes are **concatenated** into one pool, exactly like the core's ``fit_task``; replay
  exemplars come from the shared ``ExperiencePlan``.
* Training and evaluation only let classes introduced so far compete (softmax over the
  eligible output columns). Output columns of future classes are never read or updated.
* Batches come from ``sampling.training_batches`` (identical to the core), seeded with
  ``seed + experience`` as in the core; plain SGD on the batch-mean cross-entropy.
* ``replay=False`` is the separate **zero-replay ablation** (role ``ablation_no_replay``), never
  part of the primary configuration.

Nothing here is tuned: ``hidden`` and ``learning_rate`` are required arguments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polynomial_cl.continual import TASK_ORDER

from ..datasets import SplitMNIST
from ..plan import check_inputs, resolve_plan
from ..protocol import Protocol
from ..provenance import (
    RUN_MODES,
    benchmark_provenance,
    core_provenance,
    enforce_run_mode,
)
from ..sampling import SAMPLING_POLICIES, training_batches, updates_per_epoch
from .base import AdapterError, SequenceResult

N_INPUTS = 16
N_OUTPUTS = 10
ROLES = ("primary", "ablation_no_replay", "validation_candidate")


class TrainingDiverged(RuntimeError):
    """The loss or the weights became non-finite; the run has no valid result."""


# --- model ---------------------------------------------------------------------------------


@dataclass
class MLPParams:
    weights: list  # weights[l]: (fan_in, fan_out); the last one is the 10-way head
    biases: list
    use_bias: bool = (
        True  # False only for equivalence tests against the bias-free core head
    )

    def arrays(self):
        return [*self.weights, *self.biases]

    @property
    def n_parameters(self) -> int:
        return int(
            sum(a.size for a in self.weights)
            + (sum(b.size for b in self.biases) if self.use_bias else 0)
        )


def init_params(
    hidden,
    seed: int,
    dtype=np.float32,
    use_bias: bool = True,
    n_inputs: int = N_INPUTS,
    n_outputs: int = N_OUTPUTS,
    zero: bool = False,
):
    """He-normal hidden layers, ``1/sqrt(fan_in)`` head, zero biases.

    The generator is separate from every batching generator (``SeedSequence([seed, 7919])``),
    so initialisation and sampling cannot interfere. ``zero=True`` gives an all-zero model.
    """
    sizes = [n_inputs, *hidden, n_outputs]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    weights, biases = [], []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        std = np.sqrt((1.0 if last else 2.0) / fan_in)
        w = (
            np.zeros((fan_in, fan_out))
            if zero
            else rng.normal(0.0, std, (fan_in, fan_out))
        )
        weights.append(w.astype(dtype))
        biases.append(np.zeros(fan_out, dtype=dtype))
    return MLPParams(weights, biases, use_bias)


def full_logits(params: MLPParams, x: np.ndarray) -> np.ndarray:
    """Logits of every output column (used only for evaluation, which then masks)."""
    a = x
    for w, b in zip(params.weights[:-1], params.biases[:-1]):
        a = np.maximum(a @ w + (b if params.use_bias else 0), 0)
    return a @ params.weights[-1] + (params.biases[-1] if params.use_bias else 0)


def loss_and_gradients(params: MLPParams, x, y_columns, eligible):
    """Mean cross-entropy over ``eligible`` output columns and its exact gradients.

    ``y_columns`` indexes into ``eligible``. The gradient of every non-eligible output column
    is exactly zero (they take no part in the softmax), so SGD cannot move them.
    """
    n_layers = len(params.weights)
    activations, pre = [x], []
    a = x
    for w, b in zip(params.weights[:-1], params.biases[:-1]):
        z = a @ w + (b if params.use_bias else 0)
        pre.append(z)
        a = np.maximum(z, 0)
        activations.append(a)
    w_out = params.weights[-1][:, eligible]
    logits = a @ w_out + (params.biases[-1][eligible] if params.use_bias else 0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    rows = np.arange(len(y_columns))
    loss = float(-np.mean(log_probs[rows, y_columns]))
    d_logits = np.exp(log_probs)
    d_logits[rows, y_columns] -= 1.0
    d_logits /= len(y_columns)

    grad_w = [np.zeros_like(w) for w in params.weights]
    grad_b = [np.zeros_like(b) for b in params.biases]
    grad_w[-1][:, eligible] = a.T @ d_logits
    if params.use_bias:
        grad_b[-1][eligible] = d_logits.sum(axis=0)
    d_a = d_logits @ w_out.T
    for layer in range(n_layers - 2, -1, -1):
        d_z = d_a * (pre[layer] > 0)
        grad_w[layer] = activations[layer].T @ d_z
        if params.use_bias:
            grad_b[layer] = d_z.sum(axis=0)
        d_a = d_z @ params.weights[layer].T
    return loss, grad_w, grad_b


def masked_accuracy_row(logits, test_y, task_order, step: int) -> np.ndarray:
    """Row ``step`` of the accuracy matrix: argmax over classes introduced through ``step``
    only; accuracy on each experience seen so far, NaN for later ones."""
    eligible = np.sort([c for task in task_order[: step + 1] for c in task])
    predicted = eligible[np.argmax(logits[:, eligible], axis=1)]
    row = np.full(len(task_order), np.nan)
    for j in range(step + 1):
        mask = np.isin(test_y, task_order[j])
        row[j] = np.mean(predicted[mask] == test_y[mask])
    return row


# --- training ------------------------------------------------------------------------------


def train_experience(
    params: MLPParams,
    features,
    labels,
    replay_features,
    replay_labels,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    eligible_classes,
    sampling: str,
    stats: dict | None = None,
):
    """One experience: concatenate current and replay rows, then SGD over the policy's batches.

    Mirrors the core's ``fit_task``: replay is appended after the current rows, labels must be
    eligible, ``ceil(rows / batch_size)`` updates per epoch, one Generator seeded with ``seed``.
    """
    if len(replay_labels):
        features = np.concatenate((features, replay_features), axis=0)
        labels = np.concatenate((labels, replay_labels), axis=0)
    if sampling not in SAMPLING_POLICIES:
        raise ValueError(f"sampling must be one of {SAMPLING_POLICIES}")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    labels = np.asarray(labels).astype(np.int64, copy=False)
    eligible = np.asarray(eligible_classes, dtype=np.int64)
    if np.any(np.diff(eligible) <= 0) or not np.isin(labels, eligible).all():
        raise ValueError("eligible_classes must be sorted/unique and cover every label")
    label_columns = np.searchsorted(eligible, labels)
    features = np.asarray(features, dtype=params.weights[0].dtype)
    n_outputs = params.weights[-1].shape[1]
    drawn = np.zeros(n_outputs, dtype=np.int64)
    n_updates = 0
    with np.errstate(
        over="ignore", invalid="ignore"
    ):  # overflow is reported as divergence
        for batch in training_batches(
            label_columns,
            epochs=epochs,
            batch_size=batch_size,
            sampling=sampling,
            seed=seed,
        ):
            yb = label_columns[batch]
            drawn += np.bincount(eligible[yb], minlength=n_outputs)
            loss, grad_w, grad_b = loss_and_gradients(
                params, features[batch], yb, eligible
            )
            if not np.isfinite(loss):
                raise TrainingDiverged(f"non-finite loss at update {n_updates}")
            for w, g in zip(params.weights, grad_w):
                w -= learning_rate * g
            if params.use_bias:
                for b, g in zip(params.biases, grad_b):
                    b -= learning_rate * g
            n_updates += 1
    if not all(np.all(np.isfinite(a)) for a in params.arrays()):
        raise TrainingDiverged("non-finite weights after training")
    if stats is not None:
        stats["samples_drawn_by_class"] = drawn.tolist()
        stats["n_updates"] = n_updates
        stats["examples_processed"] = int(drawn.sum())
        stats["examples_available"] = int(epochs * len(labels))
        stats["mean_batch_composition"] = (drawn / n_updates).tolist()
        stats["pool_sizes_by_class"] = np.bincount(
            eligible[label_columns], minlength=n_outputs
        ).tolist()
        stats["updates_per_epoch"] = updates_per_epoch(len(labels), batch_size)
    return params


# --- adapter -------------------------------------------------------------------------------


class ERMLPAdapter:
    name = "er_mlp"

    def __init__(
        self,
        hidden,
        learning_rate: float,
        replay: bool = True,
        role: str = "primary",
        hyperparameter_provenance: dict | None = None,
        mode: str = "development",
    ):
        hidden = tuple(hidden)
        if not all(
            isinstance(h, int) and not isinstance(h, bool) and h >= 1 for h in hidden
        ):
            raise AdapterError("hidden must be a sequence of positive integers")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise AdapterError("learning_rate must be finite and positive")
        if role not in ROLES:
            raise AdapterError(f"role must be one of {ROLES}")
        if mode not in RUN_MODES:
            raise AdapterError(f"mode must be one of {RUN_MODES}")
        if role == "primary" and not replay:
            raise AdapterError(
                "the primary configuration uses replay; use the ablation role"
            )
        if role == "ablation_no_replay" and replay:
            raise AdapterError("the zero-replay ablation must have replay=False")
        self.hidden, self.learning_rate, self.replay = (
            hidden,
            float(learning_rate),
            bool(replay),
        )
        self.role, self.mode = role, mode
        self.hyperparameter_provenance = hyperparameter_provenance or {}

    @classmethod
    def from_dict(cls, raw: dict, mode: str = "development") -> ERMLPAdapter:
        if raw.get("algorithm") != "er_mlp":
            raise AdapterError("not an er_mlp algorithm configuration")
        return cls(
            hidden=raw["hidden"],
            learning_rate=raw["learning_rate"],
            replay=raw.get("replay", True),
            role=raw.get("role", "primary"),
            hyperparameter_provenance=raw.get("hyperparameter_provenance"),
            mode=mode,
        )

    @classmethod
    def from_file(cls, path, mode: str = "development") -> ERMLPAdapter:
        return cls.from_dict(json.loads(Path(path).read_text()), mode=mode)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "run_mode": self.mode,
            "configuration": {
                "hidden": list(self.hidden),
                "learning_rate": self.learning_rate,
                "replay": self.replay,
                "form": "concatenated_replay",
                "optimizer": "sgd",
                "activation": "relu",
                "initialisation": "he_normal_hidden_inverse_sqrt_head_zero_bias",
                "float_dtype": "float32",
            },
            "hyperparameter_provenance": self.hyperparameter_provenance,
            "core": core_provenance(),
            "benchmark": benchmark_provenance(),
        }

    def run_sequence(
        self, dataset: SplitMNIST, protocol: Protocol, seed: int, plan=None
    ) -> SequenceResult:
        # The shared plan and task order come from the core, so the same pin rules apply.
        enforce_run_mode(self.mode, core_provenance(), benchmark_provenance())
        check_inputs(dataset, protocol)
        plan = resolve_plan(dataset, protocol, seed, plan)
        n_tasks = len(TASK_ORDER)
        params = init_params(self.hidden, seed)
        matrix = np.full((n_tasks, n_tasks), np.nan)
        replay_used, training_stats = [], []
        train_seconds = infer_seconds = 0.0
        started = time.perf_counter()
        for task in range(n_tasks):
            seen = np.sort([c for classes in TASK_ORDER[: task + 1] for c in classes])
            current = plan.tasks[task]
            replay = (
                plan.replay_indices(task)
                if self.replay
                else np.empty(0, dtype=np.int64)
            )
            replay_used.append(len(replay))
            stats = {}
            tick = time.perf_counter()
            try:
                train_experience(
                    params,
                    dataset.train_x[current],
                    dataset.train_y[current],
                    dataset.train_x[replay],
                    dataset.train_y[replay],
                    epochs=protocol.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=protocol.batch_size,
                    seed=seed + task,
                    eligible_classes=seen,
                    sampling=protocol.sampling,
                    stats=stats,
                )
            except TrainingDiverged as error:
                raise TrainingDiverged(f"experience {task}: {error}") from error
            train_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            logits = full_logits(params, np.asarray(dataset.test_x, dtype=np.float32))
            matrix[task] = masked_accuracy_row(
                logits, np.asarray(dataset.test_y), protocol.task_order, task
            )
            infer_seconds += time.perf_counter() - tick
            training_stats.append(stats)
        elapsed = time.perf_counter() - started
        return SequenceResult(
            accuracy_matrix=matrix,
            algorithm=self.describe(),
            resources={
                "wall_time_seconds_including_evaluation": elapsed,
                "training_time_seconds": train_seconds,
                "inference_time_seconds": infer_seconds,
                "peak_memory_mb": None,
                "n_parameters": params.n_parameters,
            },
            sampling=protocol.sampling,
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
                "hidden": list(self.hidden),
                "learning_rate": self.learning_rate,
                "replay": self.replay,
                "epochs": protocol.epochs,
                "batch_size": protocol.batch_size,
                "sampling": protocol.sampling,
                "memory_per_class": protocol.memory_per_class,
                "role": self.role,
                "run_mode": self.mode,
            },
            replay_examples_used_by_task=replay_used,
            extras={"training_stats": training_stats},
        )
