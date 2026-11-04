# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Versioned, machine-readable result records (one JSON file per algorithm x seed run).

A record is self-describing and traceable: it carries the algorithm and its exact
configuration, the dataset and protocol identity (fingerprints, not paths), the seed, the plan
fingerprint (which fixes the experiences and replay exemplars), the training budget, the memory
policy and actual replay-memory use, the metrics (all recomputed by this package from the
accuracy matrix), the resources, and the environment including the core and benchmark commits.

``null`` means "not measured / not available" (never zero). A record whose ``status`` is not
``completed`` has ``null`` metrics and an ``error`` object; it is kept, never dropped.

``validate_record`` re-derives everything derivable (run key, every metric, memory limits) so a
hand-edited or truncated file is detected. Experiment-level checks (same plan fingerprint for
every algorithm at a given seed, identical dataset, single core commit, ...) live in
``validation.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from . import metrics

SCHEMA_VERSION = "1.0"
STATUSES = ("completed", "failed", "diverged", "timeout", "crashed")
EVALUATION_SPLITS = ("test", "validation")
_TOL = 1e-9

_REQUIRED = (
    "schema_version",
    "record_type",
    "run_key",
    "status",
    "attempt",
    "error",
    "algorithm",
    "dataset",
    "protocol",
    "seed",
    "plan_fingerprint",
    "training_budget",
    "memory",
    "data_representation",
    "metrics",
    "resources",
    "checks",
    "environment",
    "timestamps",
)
_METRIC_KEYS = (
    "accuracy_matrix",
    "final_accuracy",
    "mean_forgetting",
    "accuracy_per_experience",
    "current_accuracy",
    "mean_accuracy_by_step",
    "forgetting_by_experience",
    "per_class_accuracy",
)
_NUMERIC_RESOURCES = (
    "wall_time_seconds",
    "cpu_time_seconds",
    "peak_memory_mb",
    "rss_before_run_mb",
    "process_peak_rss_mb",
    "training_time_seconds",
    "inference_time_seconds",
    "adapter_wall_time_seconds_including_evaluation",
    "n_parameters",
    "replay_memory_examples_max",
    "replay_memory_bytes_max",
)


class RecordError(ValueError):
    """A result record is malformed or inconsistent."""


# --- canonical JSON and identity ---------------------------------------------------------


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


_NON_RESULT_KEYS = ("hyperparameter_provenance", "role")


def algorithm_identity(algorithm: dict) -> dict:
    """What determines the numbers an algorithm produces: its name and resolved adapter
    configuration (``algorithm["config"]``), without descriptive provenance or role. Derived
    from the configuration rather than the adapter's description so that a run that fails
    before the adapter exists still has the same key as a successful run of that config."""
    config = algorithm["config"]
    return {
        "name": config.get("algorithm"),
        "config": {k: v for k, v in config.items() if k not in _NON_RESULT_KEYS},
    }


def dataset_identity(protocol, evaluation_split: str) -> dict:
    """What fixes the data without loading it: verified archive hashes + projection + split."""
    return {
        "name": protocol.dataset["name"],
        "source_hashes": dict(protocol.dataset["files"]),
        "projection_seed": protocol.projection_seed,
        "n_components": protocol.n_components,
        "evaluation_split": evaluation_split,
        "validation": dict(protocol.validation)
        if evaluation_split == "validation"
        else None,
    }


def run_key(
    algorithm: dict, protocol_fingerprint: str, dataset: dict, seed: int
) -> str:
    """Identity of one run: same key means same algorithm configuration, protocol, data, seed."""
    return sha256_of(
        {
            "algorithm": algorithm_identity(algorithm),
            "protocol": protocol_fingerprint,
            "dataset": dataset,
            "seed": seed,
        }
    )


# --- building ------------------------------------------------------------------------------


def _clean(value):
    """Make a value strict-JSON safe: numpy scalars to Python, NaN to ``None``."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def memory_budget_by_experience(protocol) -> list:
    """Replay-memory ceiling before each experience: examples_per_class x classes already seen."""
    seen, budget = 0, []
    for classes in protocol.task_order:
        budget.append(protocol.memory_per_class * seen)
        seen += len(classes)
    return budget


def metrics_block(matrix) -> dict:
    """Every metric, derived by this package from the accuracy matrix alone."""
    r = metrics.validate_matrix(matrix)
    return {
        "accuracy_matrix": _clean(r),
        "final_accuracy": metrics.final_mean_accuracy(r),
        "mean_forgetting": metrics.mean_forgetting(r),
        "accuracy_per_experience": metrics.accuracy_per_experience(r),
        "current_accuracy": metrics.current_accuracy(r),
        "mean_accuracy_by_step": metrics.mean_accuracy_by_step(r),
        "forgetting_by_experience": metrics.forgetting_by_experience(r),
        # Adapters return an accuracy matrix only, so per-class accuracy is not available.
        "per_class_accuracy": None,
    }


def empty_metrics() -> dict:
    return {key: None for key in _METRIC_KEYS}


def matrix_from_record(record: dict) -> np.ndarray:
    """The accuracy matrix of a completed record as an array (``null`` -> NaN)."""
    raw = record["metrics"]["accuracy_matrix"]
    if raw is None:
        raise RecordError(
            f"record has no accuracy matrix (status {record.get('status')!r})"
        )
    return np.array(
        [[np.nan if v is None else v for v in row] for row in raw], dtype=float
    )


def build_record(
    *,
    label: str,
    algorithm: dict,
    protocol,
    dataset_info: dict,
    seed: int,
    status: str,
    attempt: int,
    plan_fingerprint: str | None,
    resources: dict,
    environment: dict,
    timestamps: dict,
    matrix=None,
    error: dict | None = None,
    memory_used=None,
    checks: dict | None = None,
    experiment: dict | None = None,
) -> dict:
    """Assemble one record. ``matrix`` is required exactly when ``status == 'completed'``."""
    if status not in STATUSES:
        raise RecordError(f"status must be one of {STATUSES}, got {status!r}")
    if (status == "completed") != (matrix is not None):
        raise RecordError("a matrix is required for, and only for, a completed run")
    if (status == "completed") == (error is not None):
        raise RecordError("an error object is required for, and only for, a failed run")
    used = None if memory_used is None else [int(v) for v in memory_used]
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run",
        "run_key": run_key(
            algorithm, protocol.fingerprint(), dataset_info["identity"], seed
        ),
        "experiment": experiment,
        "status": status,
        "attempt": attempt,
        "error": error,
        "algorithm": {"label": label, **algorithm},
        "dataset": {
            "name": protocol.dataset["name"],
            "identity": dataset_info["identity"],
            "fingerprint": dataset_info.get("fingerprint"),
            "source_hashes": dict(protocol.dataset["files"]),
            "projection_seed": protocol.projection_seed,
            "evaluation_split": dataset_info["identity"]["evaluation_split"],
        },
        "protocol": {
            "name": protocol.name,
            "version": protocol.version,
            "fingerprint": protocol.fingerprint(),
            "task_order": [list(t) for t in protocol.task_order],
        },
        "seed": seed,
        "plan_fingerprint": plan_fingerprint,
        "training_budget": {
            "epochs": protocol.epochs,
            "batch_size": protocol.batch_size,
            "sampling": protocol.sampling,
            "note": "epochs and batch size are the shared budget; update counts and wall time "
            "are not interchangeable across methods, see resources",
        },
        "memory": {
            "policy": "fixed_per_class_random_subset",
            "examples_per_class": protocol.memory_per_class,
            "budget_by_experience": memory_budget_by_experience(protocol),
            "used_by_experience": used,
            "max_used": None if not used else max(used),
        },
        "data_representation": {
            "projection_seed": protocol.projection_seed,
            "n_components": protocol.n_components,
            "description": protocol.projection_description,
        },
        "metrics": metrics_block(matrix) if status == "completed" else empty_metrics(),
        "resources": resources,
        "checks": checks or {},
        "environment": environment,
        "timestamps": timestamps,
    }
    record = _clean(record)
    validate_or_raise(record)
    return record


# --- validation ----------------------------------------------------------------------------


def _number_or_none(value) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_record(record: dict) -> list:
    """Return every problem found (empty list: the record is internally consistent)."""
    problems = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    missing = [k for k in _REQUIRED if k not in record]
    if missing:
        return [f"missing fields: {missing}"]
    if not isinstance(record["algorithm"], dict) or "config" not in record["algorithm"]:
        return ["algorithm.config (the resolved adapter configuration) is missing"]
    if record["schema_version"] != SCHEMA_VERSION:
        problems.append(f"unsupported schema_version {record['schema_version']!r}")
    if record["record_type"] != "run":
        problems.append("record_type must be 'run'")
    status = record["status"]
    if status not in STATUSES:
        problems.append(f"unknown status {status!r}")
    if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
        problems.append("seed must be an integer")
    if not isinstance(record["attempt"], int) or record["attempt"] < 1:
        problems.append("attempt must be a positive integer")
    split = record["dataset"].get("evaluation_split")
    if split not in EVALUATION_SPLITS:
        problems.append(f"unknown evaluation_split {split!r}")
    for key in ("fingerprint", "task_order", "name", "version"):
        if key not in record["protocol"]:
            problems.append(f"protocol.{key} missing")
    try:
        text = canonical_json(record)  # strict: rejects NaN/Infinity
        json.loads(text)
    except (ValueError, TypeError) as error:
        return problems + [f"record is not strict JSON: {error}"]

    # run key is re-derived
    try:
        expected_key = run_key(
            record["algorithm"],
            record["protocol"]["fingerprint"],
            record["dataset"]["identity"],
            record["seed"],
        )
        if expected_key != record["run_key"]:
            problems.append(
                "run_key does not match the algorithm/protocol/dataset/seed fields"
            )
    except (KeyError, TypeError) as error:
        problems.append(f"run_key cannot be re-derived: {error!r}")

    resources = record["resources"]
    for key in _NUMERIC_RESOURCES:
        if key not in resources:
            problems.append(f"resources.{key} missing (use null for unavailable)")
        elif not _number_or_none(resources[key]):
            problems.append(f"resources.{key} must be null or a finite number >= 0")
    if resources.get("peak_memory_mb") is not None and not resources.get(
        "peak_memory_scope"
    ):
        problems.append("peak_memory_mb is set without peak_memory_scope")

    memory = record["memory"]
    used, budget = memory.get("used_by_experience"), memory.get("budget_by_experience")
    n = len(record["protocol"]["task_order"])
    if not isinstance(budget, list) or len(budget) != n:
        problems.append(
            "memory.budget_by_experience must list one value per experience"
        )
    elif used is not None:
        if len(used) != n:
            problems.append(
                "memory.used_by_experience must list one value per experience"
            )
        elif any(u > b for u, b in zip(used, budget)):
            problems.append(f"replay memory used {used} exceeds the budget {budget}")
        if used and memory.get("max_used") != max(used):
            problems.append("memory.max_used is not the maximum of used_by_experience")

    m = record["metrics"]
    if any(k not in m for k in _METRIC_KEYS):
        problems.append(f"metrics must contain {list(_METRIC_KEYS)}")
        return problems
    if status == "completed":
        if record["error"] is not None:
            problems.append("a completed run must have error == null")
        if not record["plan_fingerprint"]:
            problems.append("a completed run must record its plan_fingerprint")
        try:
            matrix = metrics.validate_matrix(matrix_from_record(record))
            if matrix.shape[0] != n:
                problems.append(
                    f"matrix is {matrix.shape[0]}x{matrix.shape[0]}, expected {n}x{n}"
                )
            else:
                recomputed = metrics_block(matrix)
                for key in _METRIC_KEYS:
                    if key == "accuracy_matrix":
                        continue
                    if not _close(m[key], recomputed[key]):
                        problems.append(
                            f"metrics.{key} does not match the accuracy matrix"
                        )
        except (metrics.MatrixError, RecordError, TypeError, ValueError) as error:
            problems.append(f"accuracy matrix invalid: {error}")
    else:
        if not isinstance(record["error"], dict) or not record["error"].get("message"):
            problems.append(
                "a non-completed run must carry an error object with a message"
            )
        if any(m[k] is not None for k in _METRIC_KEYS):
            problems.append("a non-completed run must have null metrics")
    return problems


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(_close(x, y) for x, y in zip(a, b))
        )
    return abs(float(a) - float(b)) <= _TOL


def validate_or_raise(record: dict) -> dict:
    problems = validate_record(record)
    if problems:
        raise RecordError("invalid result record: " + "; ".join(problems))
    return record


# --- files ---------------------------------------------------------------------------------


def write_record(path, record: dict) -> None:
    """Validate, then write atomically (a crash never leaves a half-written record)."""
    validate_or_raise(record)
    write_json_atomic(path, record)


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj), indent=2, allow_nan=False) + "\n"
    handle, temp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def read_record(path) -> dict:
    """Read and validate; raises ``RecordError`` naming the file for any problem."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise RecordError(f"{path}: unreadable: {error}") from error
    problems = validate_record(record)
    if problems:
        raise RecordError(f"{path}: " + "; ".join(problems))
    return record
