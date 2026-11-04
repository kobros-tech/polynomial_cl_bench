# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Aggregation across seeds, and paired comparisons between algorithms.

Input is validated result records only. Failed, diverged, timed-out, crashed and missing runs
are counted and listed, never silently dropped; metric statistics use completed runs only and
always state how many that is.

Statistics are descriptive. A 95% interval is a t-interval for the mean over seeds. Seeds vary
initialisation, sampling and replay selection but **not** the data split, so intervals do not
cover variation due to the choice of train/test data, and with a handful of seeds they are
wide and rest on a normality assumption. No p-values and no "better/worse" verdicts are
produced; paired comparisons report differences and how many pairs favour each side.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from . import results

# Two-sided 95% Student-t critical values, df = 1..30 (standard table).
_T95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)
_T95_TAIL = (
    (40, 2.042),
    (60, 2.021),
    (120, 2.000),
)  # df <= limit: value at the band's lower edge (conservative)
MIN_PAIRS_NOTE = 10


def t_critical_95(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if df <= 30:
        return _T95[df - 1]
    for limit, value in _T95_TAIL:
        if df <= limit:
            return value
    return 1.980  # df > 120: the df=120 value, conservative (limit is 1.960)


def summarize(values) -> dict:
    """n, mean, sample std (ddof=1), 95% t-interval, min and max; ``None`` where undefined."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    n = len(v)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95": None,
            "min": None,
            "max": None,
        }
    mean = float(v.mean())
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "std": None,
            "ci95": None,
            "min": mean,
            "max": mean,
        }
    std = float(v.std(ddof=1))
    half = t_critical_95(n - 1) * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _elementwise(arrays) -> dict | None:
    if not arrays:
        return None
    stack = np.stack(arrays)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN cells (experiment not yet seen)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0, ddof=1) if len(arrays) > 1 else None
    return {
        "n": len(arrays),
        "mean": mean.tolist(),
        "std": None if std is None else std.tolist(),
    }


def group_by_algorithm(records) -> dict:
    groups: dict = {}
    for record in records:
        groups.setdefault(record["algorithm"]["label"], []).append(record)
    return groups


def aggregate_algorithm(label: str, records, expected_seeds) -> dict:
    """Statistics for one algorithm over its runs. ``expected_seeds`` fixes what 'missing' means."""
    expected = list(expected_seeds)
    by_seed = {r["seed"]: r for r in records}
    completed = [
        by_seed[s]
        for s in expected
        if s in by_seed and by_seed[s]["status"] == "completed"
    ]
    failures = [
        {
            "seed": s,
            "status": by_seed[s]["status"],
            "error": (by_seed[s]["error"] or {}).get("message"),
        }
        for s in expected
        if s in by_seed and by_seed[s]["status"] != "completed"
    ]
    missing = [s for s in expected if s not in by_seed]
    metrics_block = {
        key: summarize([r["metrics"][key] for r in completed])
        for key in ("final_accuracy", "mean_forgetting")
    }
    for key in ("accuracy_per_experience", "current_accuracy", "mean_accuracy_by_step"):
        metrics_block[key] = _elementwise(
            [np.array(r["metrics"][key], dtype=float) for r in completed]
        )
    metrics_block["accuracy_matrix"] = _elementwise(
        [results.matrix_from_record(r) for r in completed]
    )
    resource_keys = (
        "wall_time_seconds",
        "cpu_time_seconds",
        "training_time_seconds",
        "inference_time_seconds",
        "peak_memory_mb",
        "rss_before_run_mb",
        "n_parameters",
        "replay_memory_examples_max",
        "replay_memory_bytes_max",
    )
    resources = {
        k: summarize([r["resources"][k] for r in completed]) for k in resource_keys
    }
    first = (completed or records or [None])[0]
    return {
        "label": label,
        "algorithm": None
        if first is None
        else first["algorithm"].get("name")
        or first["algorithm"]["config"].get("algorithm"),
        "role": None if first is None else first["algorithm"].get("role"),
        "hyperparameter_provenance": None
        if first is None
        else first["algorithm"].get("hyperparameter_provenance"),
        "n_expected": len(expected),
        "n_completed": len(completed),
        "n_failed": len(failures),
        "n_missing": len(missing),
        "completed_seeds": [r["seed"] for r in completed],
        "failures": failures,
        "missing_seeds": missing,
        "complete": len(completed) == len(expected),
        "metrics": metrics_block,
        "resources": resources,
        "note_resources": "n in each resource entry is the number of completed runs that "
        "reported that measurement; null statistics mean it was unavailable for every run",
    }


def paired_comparison(
    a_records, b_records, metric: str = "final_accuracy", labels=("a", "b")
) -> dict:
    """Per-seed differences ``a - b`` of ``metric`` over seeds both completed.

    Reports mean difference, its std and t-interval, and counts of pairs favouring each side
    (for forgetting, lower is better; the counts use ``a_higher``/``b_higher`` on the raw
    values, so no direction is hidden in the wording). Never a significance claim.
    """
    a_ok = {r["seed"]: r for r in a_records if r["status"] == "completed"}
    b_ok = {r["seed"]: r for r in b_records if r["status"] == "completed"}
    seeds = sorted(set(a_ok) & set(b_ok))
    excluded = sorted((set(a_ok) | set(b_ok)) - set(seeds))
    diffs = [a_ok[s]["metrics"][metric] - b_ok[s]["metrics"][metric] for s in seeds]
    summary = summarize(diffs)
    n = len(diffs)
    return {
        "metric": metric,
        "a": labels[0],
        "b": labels[1],
        "difference": "a - b",
        "paired_seeds": seeds,
        "excluded_seeds": excluded,
        "n_pairs": n,
        "per_seed_differences": diffs,
        "mean_difference": summary["mean"],
        "std_difference": summary["std"],
        "ci95_mean_difference": summary["ci95"],
        "a_higher": int(sum(d > 0 for d in diffs)),
        "b_higher": int(sum(d < 0 for d in diffs)),
        "ties": int(sum(d == 0 for d in diffs)),
        "qualification": (
            f"{n} paired seed(s). Descriptive only: seeds vary initialisation, sampling and "
            "replay selection but not the data split; the t-interval assumes roughly normal "
            "differences."
            + (
                f" Fewer than {MIN_PAIRS_NOTE} pairs: treat any apparent difference as tentative."
                if n < MIN_PAIRS_NOTE
                else ""
            )
            + (" Fewer than 2 pairs: no interval." if n < 2 else "")
        ),
    }


def aggregate_experiment(records, expected_labels, expected_seeds) -> dict:
    """Per-algorithm aggregates plus every pairwise paired comparison (final accuracy and
    mean forgetting) over the labels in experiment order."""
    groups = group_by_algorithm(records)
    algorithms = {
        label: aggregate_algorithm(label, groups.get(label, []), expected_seeds)
        for label in expected_labels
    }
    pairs = []
    for i, a in enumerate(expected_labels):
        for b in expected_labels[i + 1 :]:
            for metric in ("final_accuracy", "mean_forgetting"):
                pairs.append(
                    paired_comparison(
                        groups.get(a, []), groups.get(b, []), metric, (a, b)
                    )
                )
    return {"algorithms": algorithms, "paired_comparisons": pairs}
