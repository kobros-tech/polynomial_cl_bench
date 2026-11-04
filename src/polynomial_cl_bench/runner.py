# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Experiment runner: many algorithms x many seeds, one result record per run.

    python -m polynomial_cl_bench.runner --spec configs/experiment_tuned_comparison.json \\
        --data-dir data/mnist --out results/tuned_comparison [--resume] [--retry-failed] \\
        [--reproduce-check]

Design rules
------------
* The runner knows nothing about algorithm internals: it builds an adapter from a resolved
  configuration, hands it the shared dataset and the shared per-seed plan, and records what
  comes back. Core-dependent imports happen lazily, so everything here is testable with
  fake adapters and without the core.
* **Failures are results.** Any exception in a run (divergence, bad matrix, adapter error,
  crash, timeout) becomes a record with a non-``completed`` status and its error. Nothing is
  omitted from aggregation. Only ``ProvenanceError`` (official-mode refusal) and
  ``KeyboardInterrupt`` abort the experiment.
* **Isolation.** ``isolation: process`` (default for the CLI) runs every algorithm x seed in a
  fresh Python process: no state can leak between runs, a crash or timeout cannot take the
  experiment down, and peak memory has a clean, comparable baseline. ``in_process`` is for
  tests and quick checks. Runs are executed sequentially on purpose: concurrent runs would
  contaminate each other's timings.
* **Resume without mixing.** The output directory carries a manifest with the fingerprint of
  the full resolved experiment (protocol, every algorithm configuration, seeds, mode, split,
  isolation). Resuming into a directory with a different fingerprint is refused; a completed
  record is reused only if its run key and the core/benchmark commits match the current ones;
  a failed record is re-run only with ``--retry-failed`` and the old record is kept as
  ``<name>.attempt<k>.json`` so failure history is never lost.
* **Tuned configurations.** An algorithm entry may name a ``tuning_result``: the selected
  hyper-parameters then come from that validation-only tuning file, after checks on its
  protocol fingerprint, split and seed disjointness, and its hash is recorded as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, results
from .protocol import Protocol
from .provenance import (
    RUN_MODES,
    ProvenanceError,
    benchmark_provenance,
    core_provenance,
    enforce_run_mode,
)
from .resources import RunMeter, environment_info

ISOLATIONS = ("process", "in_process")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TUNABLE_KEYS = {
    "er_mlp": ("hidden", "learning_rate"),
    "polynomial_cl": ("learning_rate", "feature_power"),
}
CORE_AGREEMENT_TOLERANCE = 1e-9
TRACE_LIMIT = 4000


class ExperimentError(RuntimeError):
    """The experiment request or its output directory is unusable."""


class ExperimentMismatch(ExperimentError):
    """Existing results belong to a different experiment, or to different code."""


# --- specification -------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgorithmEntry:
    label: str
    config: (
        dict  # resolved adapter configuration (tuned values and provenance included)
    )


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    protocol: Protocol
    algorithms: tuple
    seeds: tuple
    mode: str = "development"
    evaluation_split: str = "test"
    isolation: str = "process"
    timeout_seconds: float | None = None

    def __post_init__(self):
        if not LABEL_PATTERN.match(self.name):
            raise ExperimentError(
                f"experiment name {self.name!r} must match {LABEL_PATTERN.pattern}"
            )
        if self.mode not in RUN_MODES:
            raise ExperimentError(f"mode must be one of {RUN_MODES}")
        if self.evaluation_split not in results.EVALUATION_SPLITS:
            raise ExperimentError(
                f"evaluation_split must be one of {results.EVALUATION_SPLITS}"
            )
        if self.isolation not in ISOLATIONS:
            raise ExperimentError(f"isolation must be one of {ISOLATIONS}")
        if not self.algorithms:
            raise ExperimentError("an experiment needs at least one algorithm")
        labels = [a.label for a in self.algorithms]
        if len(set(labels)) != len(labels):
            raise ExperimentError(f"algorithm labels must be unique, got {labels}")
        for label in labels:
            if not LABEL_PATTERN.match(label):
                raise ExperimentError(
                    f"algorithm label {label!r} must match {LABEL_PATTERN.pattern}"
                )
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or not all(
                isinstance(s, int) and not isinstance(s, bool) for s in self.seeds
            )
        ):
            raise ExperimentError("seeds must be a non-empty list of distinct integers")
        if self.timeout_seconds is not None and not self.timeout_seconds > 0:
            raise ExperimentError("timeout_seconds must be positive")
        if self.timeout_seconds is not None and self.isolation != "process":
            raise ExperimentError("a timeout needs process isolation")

    @property
    def labels(self) -> list:
        return [a.label for a in self.algorithms]

    def to_dict(self) -> dict:
        """Resolved, JSON-able form: what the fingerprint and the manifest are built from."""
        return {
            "name": self.name,
            "protocol": self.protocol.to_dict(),
            "algorithms": [
                {"label": a.label, "config": a.config} for a in self.algorithms
            ],
            "seeds": list(self.seeds),
            "mode": self.mode,
            "evaluation_split": self.evaluation_split,
            "isolation": self.isolation,
        }

    def fingerprint(self) -> str:
        """Identity of the experiment. The timeout is excluded: it does not change results."""
        body = self.to_dict()
        body.pop("name")
        return results.sha256_of(body)

    @classmethod
    def from_file(cls, path) -> ExperimentSpec:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict, base_dir=".") -> ExperimentSpec:
        base = Path(base_dir)
        try:
            protocol = _load_protocol(raw["protocol"], base)
            seeds = tuple(raw["seeds"]) if raw.get("seeds") else tuple(protocol.seeds)
            entries = []
            for item in raw["algorithms"]:
                config = _load_json(item["config"], base)
                if item.get("tuning_result"):
                    config = apply_tuning_result(
                        config, base / item["tuning_result"], protocol, seeds
                    )
                entries.append(AlgorithmEntry(item["label"], config))
            return cls(
                name=raw["name"],
                protocol=protocol,
                algorithms=tuple(entries),
                seeds=seeds,
                mode=raw.get("mode", "development"),
                evaluation_split=raw.get("evaluation_split", "test"),
                isolation=raw.get("isolation", "process"),
                timeout_seconds=raw.get("timeout_seconds"),
            )
        except KeyError as error:
            raise ExperimentError(f"missing experiment field: {error}") from error


def _load_json(value, base: Path) -> dict:
    if isinstance(value, dict):
        return dict(value)
    return json.loads((base / value).read_text())


def _load_protocol(value, base: Path) -> Protocol:
    if isinstance(value, dict):
        return Protocol.from_dict(value)
    return Protocol.from_file(base / value)


def apply_tuning_result(config: dict, tuning_path, protocol: Protocol, seeds) -> dict:
    """Fill an algorithm's tunable hyper-parameters from a validation-only tuning result.

    Refuses unless: the tuning is for this algorithm and this protocol (fingerprint), selected
    something, ran every run on the validation split, and its selection seeds are disjoint
    from the experiment's seeds. The base configuration must not already set a tunable key.
    """
    algorithm = config.get("algorithm")
    if algorithm not in TUNABLE_KEYS:
        raise ExperimentError(f"no tunable keys known for algorithm {algorithm!r}")
    path = Path(tuning_path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise ExperimentError(
            f"tuning result {path} cannot be read ({error}); run the validation tuning first"
        ) from error
    tuning = json.loads(raw_bytes)
    problems = []
    if tuning.get("algorithm") != algorithm:
        problems.append(f"tuning is for {tuning.get('algorithm')!r}, not {algorithm!r}")
    if tuning.get("protocol_fingerprint") != protocol.fingerprint():
        problems.append(
            "tuning was run under a different protocol (fingerprint differs)"
        )
    if tuning.get("selected_config") is None:
        problems.append("tuning selected no configuration")
    if tuning.get("validation_dataset", {}).get(
        "evaluation_split"
    ) != "validation" or any(
        r.get("evaluation_split") != "validation" for r in tuning.get("runs", [])
    ):
        problems.append("tuning was not evaluated on the validation split")
    overlap = set(tuning.get("selection_seeds", [])) & set(seeds)
    if overlap:
        problems.append(
            f"selection seeds {sorted(overlap)} overlap the experiment seeds"
        )
    present = [k for k in TUNABLE_KEYS[algorithm] if k in config]
    if present:
        problems.append(f"base config already sets tunable keys {present}")
    if problems:
        raise ExperimentError(
            f"cannot use tuning result {path.name}: " + "; ".join(problems)
        )
    selected = tuning["selected_config"]
    if set(selected) != set(TUNABLE_KEYS[algorithm]):
        raise ExperimentError(f"selected_config keys {sorted(selected)} are unexpected")
    context = tuning.get("context", {})
    resolved = {**config, **selected}
    resolved["hyperparameter_provenance"] = {
        "selection": "validation_tuning",
        "tuning_result_file": path.name,
        "tuning_result_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selected_index": tuning["selected_index"],
        "selection_seeds": tuning["selection_seeds"],
        "criterion": tuning["criterion"],
        "grid_file": context.get("grid_file"),
        "grid_sha256": context.get("grid_sha256"),
        "tuning_run_mode": context.get("run_mode"),
        "tuning_official_eligible": context.get("official_eligible"),
        "evaluation_split": "validation",
    }
    return resolved


# --- adapters --------------------------------------------------------------------------------


def build_adapter(config: dict, mode: str):
    """Resolved configuration -> adapter. Adapter modules (and the core) load lazily here."""
    name = config.get("algorithm")
    if name == "er_mlp":
        from .algorithms.er_mlp import ERMLPAdapter

        return ERMLPAdapter.from_dict(config, mode=mode)
    if name == "polynomial_cl":
        from .algorithms.polynomial import PolynomialAdapter

        return PolynomialAdapter.from_dict(config, mode=mode)
    raise ExperimentError(f"unknown algorithm {name!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _environment(mode: str) -> dict:
    core, bench = core_provenance(), benchmark_provenance()
    return {
        **environment_info(),
        "package_version": __version__,
        "core_version": core["version"],
        "core_commit": core["commit"],
        "core_commit_source": core["commit_source"],
        "core_remote_verified": core["remote_verified"],
        "core_matches_pin": core["matches_pin"],
        "benchmark_commit": bench["commit"],
        "benchmark_commit_source": bench["commit_source"],
        "run_mode": mode,
        "official_problems": [
            *[f"core: {p}" for p in core["official_problems"]],
            *[f"benchmark: {p}" for p in bench["official_problems"]],
        ],
    }


def _provenance_signature(environment: dict) -> tuple:
    return (
        environment.get("core_commit"),
        environment.get("core_commit_source"),
        environment.get("benchmark_commit"),
        environment.get("benchmark_commit_source"),
    )


_RESOURCE_DEFAULTS = {
    "wall_time_seconds": None,
    "cpu_time_seconds": None,
    "peak_memory_mb": None,
    "peak_memory_scope": None,
    "rss_before_run_mb": None,
    "process_peak_rss_mb": None,
    "training_time_seconds": None,
    "inference_time_seconds": None,
    "adapter_wall_time_seconds_including_evaluation": None,
    "n_parameters": None,
    "replay_memory_examples_max": None,
    "replay_memory_bytes_max": None,
    "isolation": None,
}


def _base_algorithm(config: dict) -> dict:
    """What is known about an algorithm before (or without) constructing its adapter."""
    return {
        "config": config,
        "name": config.get("algorithm"),
        "role": config.get("role"),
        "hyperparameter_provenance": config.get("hyperparameter_provenance", {}),
    }


def _error_object(error: BaseException, stage: str) -> dict:
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "type": type(error).__name__,
        "message": str(error) or type(error).__name__,
        "stage": stage,
        "traceback": trace[-TRACE_LIMIT:],
    }


def _failure_status(error: BaseException) -> str:
    return "diverged" if type(error).__name__ == "TrainingDiverged" else "failed"


# --- one run ----------------------------------------------------------------------------------


def execute_run(
    *,
    label: str,
    config: dict,
    protocol: Protocol,
    dataset,
    seed: int,
    plan,
    mode: str,
    attempt: int = 1,
    isolation: str = "in_process",
    experiment: dict | None = None,
    adapter_factory=build_adapter,
) -> dict:
    """Run one algorithm for one seed and return its validated record. Never raises for an
    algorithm failure (it becomes a ``failed``/``diverged`` record); ``ProvenanceError`` and
    ``KeyboardInterrupt`` propagate."""
    environment = _environment(mode)
    started = _utc_now()
    identity = results.dataset_identity(protocol, dataset.evaluation_split)
    dataset_info = {"identity": identity, "fingerprint": dataset.fingerprint}
    plan_fingerprint = plan.fingerprint() if plan is not None else None
    algorithm = _base_algorithm(config)
    resources = dict(_RESOURCE_DEFAULTS, isolation=isolation)
    meter = RunMeter()
    stage = "configuration"
    try:
        adapter = adapter_factory(config, mode)
        algorithm = {**_base_algorithm(config), **_describe(adapter)}
        stage = "run"
        with meter:
            result = adapter.run_sequence(dataset, protocol, seed, plan)
        stage = "record"
        resources.update(_resources(meter, result, dataset))
        _check_run_config(result, protocol, dataset, seed, plan_fingerprint)
        record = results.build_record(
            label=label,
            algorithm=algorithm,
            protocol=protocol,
            dataset_info=dataset_info,
            seed=seed,
            status="completed",
            attempt=attempt,
            plan_fingerprint=plan_fingerprint,
            resources=resources,
            environment=environment,
            timestamps={"started_utc": started, "finished_utc": _utc_now()},
            matrix=result.accuracy_matrix,
            memory_used=result.replay_examples_used_by_task,
            checks=_core_cross_check(result),
            experiment=experiment,
        )
        return record
    except ProvenanceError:
        raise
    except Exception as error:  # noqa: BLE001 - recorded, never hidden
        if stage == "run" or meter.wall_time_seconds is not None:
            resources.update(
                {k: v for k, v in meter.report().items() if k in resources}
            )
        return results.build_record(
            label=label,
            algorithm=algorithm,
            protocol=protocol,
            dataset_info=dataset_info,
            seed=seed,
            status=_failure_status(error),
            attempt=attempt,
            plan_fingerprint=plan_fingerprint,
            resources=resources,
            environment=environment,
            timestamps={"started_utc": started, "finished_utc": _utc_now()},
            error=_error_object(error, stage),
            experiment=experiment,
        )


def _describe(adapter) -> dict:
    description = adapter.describe()
    keep = (
        "name",
        "role",
        "condition",
        "strategy",
        "configuration",
        "hyperparameter_provenance",
        "core",
        "benchmark",
    )
    return {k: description[k] for k in keep if k in description}


def _resources(meter: RunMeter, result, dataset) -> dict:
    reported = result.resources or {}
    used = result.replay_examples_used_by_task
    max_used = max(used) if used else None
    itemsize = dataset.train_x.dtype.itemsize * dataset.train_x.shape[1]
    values = {
        **meter.report(),
        "training_time_seconds": reported.get("training_time_seconds"),
        "inference_time_seconds": reported.get("inference_time_seconds"),
        "adapter_wall_time_seconds_including_evaluation": reported.get(
            "wall_time_seconds_including_evaluation"
        ),
        "n_parameters": reported.get("n_parameters"),
        "replay_memory_examples_max": max_used,
        "replay_memory_bytes_max": None
        if max_used is None
        else int(max_used * itemsize),
    }
    return values


def _check_run_config(result, protocol, dataset, seed, plan_fingerprint) -> None:
    """The adapter must report having run exactly what the runner asked for."""
    config = result.run_config or {}
    problems = []
    if config.get("seed") != seed:
        problems.append(f"adapter ran seed {config.get('seed')!r}, asked for {seed}")
    if config.get("protocol", {}).get("fingerprint") != protocol.fingerprint():
        problems.append("adapter reports a different protocol fingerprint")
    if config.get("dataset", {}).get("fingerprint") != dataset.fingerprint:
        problems.append("adapter reports a different dataset fingerprint")
    if config.get("dataset", {}).get("evaluation_split") != dataset.evaluation_split:
        problems.append("adapter reports a different evaluation split")
    if (
        plan_fingerprint is not None
        and config.get("plan_fingerprint") != plan_fingerprint
    ):
        problems.append("adapter did not use the plan it was given")
    if problems:
        raise ExperimentError("run_config mismatch: " + "; ".join(problems))


def _core_cross_check(result) -> dict:
    """Compare this package's metrics with values the algorithm itself reported (polynomial
    core), when it reports them. Disagreement is recorded, and experiment validation flags it."""
    reported = (result.extras or {}).get("core_reported")
    if not reported:
        return {}
    from . import metrics

    matrix = metrics.validate_matrix(result.accuracy_matrix)
    acc = abs(
        metrics.final_mean_accuracy(matrix) - float(reported["final_mean_accuracy"])
    )
    forget = abs(metrics.mean_forgetting(matrix) - float(reported["mean_forgetting"]))
    return {
        "core_reported_cross_check": {
            "final_accuracy_abs_diff": acc,
            "mean_forgetting_abs_diff": forget,
            "tolerance": CORE_AGREEMENT_TOLERANCE,
            "agrees": bool(max(acc, forget) <= CORE_AGREEMENT_TOLERANCE),
        }
    }


# --- process isolation ------------------------------------------------------------------------


def worker_command(task_path, output_path) -> list:
    return [
        sys.executable,
        "-m",
        "polynomial_cl_bench.worker",
        str(task_path),
        str(output_path),
    ]


def _unrun_record(spec, entry, seed, attempt, status, error, wall, experiment) -> dict:
    """A record for a run whose worker produced nothing usable (crash, timeout)."""
    identity = results.dataset_identity(spec.protocol, spec.evaluation_split)
    resources = dict(_RESOURCE_DEFAULTS, isolation="process", wall_time_seconds=wall)
    return results.build_record(
        label=entry.label,
        algorithm=_base_algorithm(entry.config),
        protocol=spec.protocol,
        dataset_info={"identity": identity, "fingerprint": None},
        seed=seed,
        status=status,
        attempt=attempt,
        plan_fingerprint=None,
        resources=resources,
        environment=_environment(spec.mode),
        timestamps={"started_utc": _utc_now(), "finished_utc": _utc_now()},
        error=error,
        experiment=experiment,
    )


def run_in_subprocess(spec, entry, seed, attempt, data_dir, experiment) -> dict:
    with tempfile.TemporaryDirectory(prefix="plcb_run_") as tmp:
        task_path = Path(tmp) / "task.json"
        output_path = Path(tmp) / "record.json"

        task_path.write_text(
            json.dumps(
                {
                    "label": entry.label,
                    "config": entry.config,
                    "protocol": spec.protocol.to_dict(),
                    "data_dir": str(data_dir),
                    "seed": seed,
                    "mode": spec.mode,
                    "evaluation_split": spec.evaluation_split,
                    "attempt": attempt,
                    "experiment": experiment,
                }
            )
        )

        started = time.perf_counter()
        try:
            done = subprocess.run(
                worker_command(task_path, output_path),
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            tail = error.stderr or b""
            tail = tail.decode(errors="replace") if isinstance(tail, bytes) else tail
            return _unrun_record(
                spec,
                entry,
                seed,
                attempt,
                "timeout",
                {
                    "type": "TimeoutExpired",
                    "message": (
                        f"run exceeded {spec.timeout_seconds} s and was killed"
                    ),
                    "stage": "worker",
                    "traceback": tail[-TRACE_LIMIT:],
                },
                time.perf_counter() - started,
                experiment,
            )

        wall = time.perf_counter() - started
        if done.returncode == 3:  # worker refused: official-mode provenance
            raise ProvenanceError(
                done.stderr.strip().splitlines()[-1] if done.stderr else "refused"
            )

        if done.returncode != 0 or not output_path.exists():
            return _unrun_record(
                spec,
                entry,
                seed,
                attempt,
                "crashed",
                {
                    "type": "WorkerExit",
                    "message": f"worker exited with code {done.returncode}",
                    "stage": "worker",
                    "traceback": (done.stderr or "")[-TRACE_LIMIT:],
                },
                wall,
                experiment,
            )

        record = json.loads(output_path.read_text())
        problems = results.validate_record(record)
        if problems:
            return _unrun_record(
                spec,
                entry,
                seed,
                attempt,
                "failed",
                {
                    "type": "InvalidRecord",
                    "message": (
                        "worker produced an invalid record: " + "; ".join(problems)
                    ),
                    "stage": "record",
                    "traceback": "",
                },
                wall,
                experiment,
            )

        return record


# --- experiment -------------------------------------------------------------------------------


def run_path(out_dir: Path, label: str, seed: int) -> Path:
    return Path(out_dir) / "runs" / f"{label}__seed{seed}.json"


def _manifest(spec: ExperimentSpec) -> dict:
    return {
        "schema_version": results.SCHEMA_VERSION,
        "record_type": "experiment_manifest",
        "name": spec.name,
        "fingerprint": spec.fingerprint(),
        "spec": spec.to_dict(),
        "expected_runs": [
            {"label": a.label, "seed": s} for s in spec.seeds for a in spec.algorithms
        ],
        "created_utc": _utc_now(),
        "environment": _environment(spec.mode),
    }


def _prepare_directory(spec, out_dir: Path, resume: bool) -> None:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ExperimentError(
                f"{out_dir} already holds an experiment; use --resume or another directory"
            )
        existing = json.loads(manifest_path.read_text())
        if existing.get("fingerprint") != spec.fingerprint():
            raise ExperimentMismatch(
                f"{out_dir} was created for a different experiment (fingerprint "
                f"{existing.get('fingerprint')!r} != {spec.fingerprint()!r}); refusing to mix "
                "configurations. Use a new output directory."
            )
        return
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ExperimentError(
            f"{out_dir} is not empty and has no manifest; refusing to use it"
        )
    results.write_json_atomic(manifest_path, _manifest(spec))
    (out_dir / "runs").mkdir(exist_ok=True)


def expected_key(spec, entry, seed) -> str:
    identity = results.dataset_identity(spec.protocol, spec.evaluation_split)
    return results.run_key(
        {"config": entry.config}, spec.protocol.fingerprint(), identity, seed
    )


def run_experiment(
    spec: ExperimentSpec,
    out_dir,
    *,
    data_dir=None,
    resume: bool = False,
    retry_failed: bool = False,
    reproduce_check: bool = False,
    dataset_loader=None,
    plan_factory=None,
    adapter_factory=build_adapter,
    log=print,
) -> dict:
    """Run (or resume) the experiment and write one record per algorithm x seed.

    Returns ``{"ran": [...], "reused": [...], "failed": [...], "out_dir": ...}``. Failed runs
    are listed, kept on disk and counted by aggregation; they do not stop the experiment.
    """
    out_dir = Path(out_dir)
    enforce_run_mode(
        spec.mode, core_provenance(), benchmark_provenance()
    )  # before any work
    if spec.isolation == "process" and adapter_factory is not build_adapter:
        raise ExperimentError("a custom adapter_factory needs in_process isolation")
    _prepare_directory(spec, out_dir, resume)
    experiment = {"name": spec.name, "fingerprint": spec.fingerprint()}
    here = _provenance_signature(_environment(spec.mode))
    state = {"dataset": None, "plans": {}}

    def in_process(entry, seed, attempt):
        if state["dataset"] is None:
            state["dataset"] = _load_dataset(spec, data_dir, dataset_loader)

        dataset = state["dataset"]
        if seed not in state["plans"]:
            state["plans"][seed] = (plan_factory or _default_plan)(
                dataset, spec.protocol, seed
            )

        return execute_run(
            label=entry.label,
            config=entry.config,
            protocol=spec.protocol,
            dataset=dataset,
            seed=seed,
            plan=state["plans"][seed],
            mode=spec.mode,
            attempt=attempt,
            isolation="in_process",
            experiment=experiment,
            adapter_factory=adapter_factory,
        )

    if spec.isolation == "process":
        if data_dir is None:
            raise ExperimentError("process isolation needs a data directory")
        from .datasets import verify_mnist_files

        verify_mnist_files(spec.protocol, data_dir)  # fail once, up front, not per run

    def execute(entry, seed, attempt):
        if spec.isolation == "process":
            return run_in_subprocess(spec, entry, seed, attempt, data_dir, experiment)
        return in_process(entry, seed, attempt)

    outcome = {"ran": [], "reused": [], "failed": [], "out_dir": str(out_dir)}
    for seed in spec.seeds:
        for entry in spec.algorithms:
            path, name = (
                run_path(out_dir, entry.label, seed),
                f"{entry.label} seed {seed}",
            )
            attempt = 1
            if path.exists():
                existing = results.read_record(path)
                if existing["run_key"] != expected_key(spec, entry, seed):
                    raise ExperimentMismatch(
                        f"{path.name}: run key differs from this experiment's"
                    )
                if existing["status"] == "completed":
                    if _provenance_signature(existing["environment"]) != here:
                        raise ExperimentMismatch(
                            f"{path.name} was produced by a different core/benchmark commit "
                            f"{_provenance_signature(existing['environment'])} than the current "
                            f"{here}; refusing to mix. Use a new output directory."
                        )
                    log(f"reuse   {name}")
                    outcome["reused"].append(name)
                    continue
                if not retry_failed:
                    log(
                        f"keep    {name}: {existing['status']} (use --retry-failed to rerun)"
                    )
                    outcome["failed"].append(name)
                    continue
                attempt = existing["attempt"] + 1
                path.rename(
                    path.with_name(f"{path.stem}.attempt{existing['attempt']}.json")
                )
            log(f"run     {name} (attempt {attempt})")
            record = execute(entry, seed, attempt)
            results.write_record(path, record)
            outcome["ran"].append(name)
            if record["status"] != "completed":
                outcome["failed"].append(name)
                log(f"FAILED  {name}: {record['status']}: {record['error']['message']}")
    if reproduce_check:
        _reproduce_check(spec, out_dir, execute, log)
    return outcome


def _load_dataset(spec, data_dir, loader):
    if loader is None:
        from .datasets import load_split_mnist

        def loader(protocol, directory):
            return load_split_mnist(protocol, directory)

    dataset = loader(spec.protocol, data_dir)
    if spec.evaluation_split == "validation":
        from .datasets import validation_view

        dataset = validation_view(dataset, spec.protocol)
    return dataset


def _default_plan(dataset, protocol, seed):
    from .plan import make_plan

    return make_plan(dataset.train_y, seed, protocol.memory_per_class)


def _reproduce_check(spec, out_dir: Path, execute, log) -> None:
    """Re-execute each algorithm's first completed seed and compare accuracy matrices exactly."""
    for entry in spec.algorithms:
        original = next(
            (
                r
                for s in spec.seeds
                if (p := run_path(out_dir, entry.label, s)).exists()
                and (r := results.read_record(p))["status"] == "completed"
            ),
            None,
        )
        if original is None:
            continue
        seed = original["seed"]
        log(f"repeat  {entry.label} seed {seed} (reproducibility check)")
        again = execute(entry, seed, original["attempt"])
        entry_out = {
            "label": entry.label,
            "seed": seed,
            "run_key": original["run_key"],
            "rerun_status": again["status"],
            "exact_match": False,
            "max_abs_difference": None,
            "thread_env": original["environment"]["thread_env"],
            "checked_utc": _utc_now(),
        }
        if again["status"] == "completed":
            a, b = (
                results.matrix_from_record(original),
                results.matrix_from_record(again),
            )
            entry_out["exact_match"] = bool(np.array_equal(a, b, equal_nan=True))
            entry_out["max_abs_difference"] = float(np.nanmax(np.abs(a - b)))
        else:
            entry_out["error"] = again["error"]["message"]
        results.write_json_atomic(
            Path(out_dir) / "reproducibility" / f"{entry.label}__seed{seed}.json",
            entry_out,
        )


# --- command line -----------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--resume", action="store_true", help="continue an existing output directory"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run failed runs (needs --resume)",
    )
    parser.add_argument("--reproduce-check", action="store_true")
    parser.add_argument(
        "--mode", choices=RUN_MODES, help="override the spec's run mode"
    )
    parser.add_argument(
        "--isolation", choices=ISOLATIONS, help="override the spec's isolation"
    )
    args = parser.parse_args(argv)
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed needs --resume")
    spec = ExperimentSpec.from_file(args.spec)
    if args.mode or args.isolation:
        spec = replace(
            spec,
            mode=args.mode or spec.mode,
            isolation=args.isolation or spec.isolation,
        )
    outcome = run_experiment(
        spec,
        args.out,
        data_dir=args.data_dir,
        resume=args.resume,
        retry_failed=args.retry_failed,
        reproduce_check=args.reproduce_check,
    )
    from .summarize import summarize_directory

    summary = summarize_directory(args.out)
    print(summary["markdown"])
    print(
        f"\nran {len(outcome['ran'])}, reused {len(outcome['reused'])}, not completed {len(outcome['failed'])}"
    )
    return 0 if not outcome["failed"] and not summary["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
