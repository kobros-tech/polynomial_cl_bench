# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""The paper's research questions RQ2-RQ4 under equal tuning budgets, through the benchmark.

Every cell of the design below is tuned on the validation view with the *same* approved
15-configuration grid (``configs/polynomial_grid.json``: learning rate x feature power), the
*same* selection seeds, the benchmark's own selection rule (``tuning.tune``) and the same
budget (30 runs), then run once on the test split with the protocol's seeds. Cells therefore
differ only in the factor under study; none is given extra tuning.

    RQ2  Does gradual expansion help?      condition: fixed_linear | fixed_capacity | expanding
    RQ3  How strong is sampling?           sampling:  class_balanced | shuffled
    RQ4  Plasticity versus retention       strategy:  ordinary | frozen_old | strategy_b

``fixed_capacity`` has the final feature capacity of ``expanding`` from the first experience, so
RQ2 compares the schedule at equal capacity; ``fixed_linear`` is the degree-1 floor. Reported next
to accuracy and forgetting is the accuracy on arrival (diagonal of the accuracy matrix), because
retention bought with weaker learning is not better continual learning.

RQ1 (does expansion preserve existing predictions?) is a property of the model, not of a
benchmark run; it is tested by the core's own test suite and is not re-tested here.
RQ5 (degree control) is ``tools/run_degree_control_study.py``. RQ6 (against an independent
baseline) is the official ``tuned_comparison`` experiment.

Exploratory and development-mode: the cells are not an approved experiment spec. One dataset,
one projection, seeds vary shuffling and replay selection only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from polynomial_cl_bench import metrics
from polynomial_cl_bench.aggregate import paired_comparison, summarize
from polynomial_cl_bench.algorithms.polynomial import PolynomialAdapter
from polynomial_cl_bench.datasets import load_split_mnist
from polynomial_cl_bench.plan import make_plan
from polynomial_cl_bench.protocol import Protocol
from polynomial_cl_bench.provenance import benchmark_provenance, core_provenance
from polynomial_cl_bench.tune import load_grid
from polynomial_cl_bench.tuning import tune
from polynomial_cl_bench.verify_data import DEFAULT_PROTOCOL

REFERENCE = "expanding / ordinary / class_balanced"
SAMPLINGS = ("class_balanced", "shuffled")
# (condition, strategy) rows; each is run under both sampling policies.
ROWS = (
    ("fixed_linear", "ordinary"),
    ("fixed_capacity", "ordinary"),
    ("expanding", "ordinary"),
    ("expanding", "frozen_old"),
    ("expanding", "strategy_b"),
)
RQ_OF = {
    "fixed_linear": "RQ2",
    "fixed_capacity": "RQ2",
    "frozen_old": "RQ4",
    "strategy_b": "RQ4",
}


def label(condition, strategy, sampling):
    return f"{condition} / {strategy} / {sampling}"


def run_test(adapter, dataset, protocol):
    records = []
    for seed in protocol.seeds:
        plan = make_plan(dataset.train_y, seed, protocol.memory_per_class)
        try:
            result = adapter.run_sequence(dataset, protocol, seed, plan)
        except Exception as error:  # noqa: BLE001 - recorded as a failed run, never dropped
            records.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        m = result.accuracy_matrix
        arrival = metrics.current_accuracy(m)
        records.append(
            {
                "seed": seed,
                "status": "completed",
                "arrival_accuracy": arrival,
                "metrics": {
                    "final_accuracy": metrics.final_mean_accuracy(m),
                    "mean_forgetting": metrics.mean_forgetting(m),
                    "newest_accuracy": arrival[-1],
                    "mean_arrival_accuracy": float(np.mean(arrival)),
                },
            }
        )
    return records


def describe(records):
    ok = [r for r in records if r["status"] == "completed"]
    out = {"completed": len(ok), "failed": len(records) - len(ok)}
    for name in (
        "final_accuracy",
        "mean_forgetting",
        "newest_accuracy",
        "mean_arrival_accuracy",
    ):
        out[name] = summarize([r["metrics"][name] for r in ok])
    return out


def pair(a, b, names, metric):
    return paired_comparison(a, b, metric, names)


def fmt(s):
    return "—" if s["mean"] is None else f"{s['mean']:.4f} ± {s['std'] or 0:.4f}"


def report(result):
    lines = [
        "# Research questions RQ2-RQ4 at equal tuning budget (exploratory, development mode)",
        "",
        f"Every cell tuned on the validation view over the same 15 configurations "
        f"(learning rate x feature power), selection seeds {result['selection_seeds']}; "
        f"test seeds {result['test_seeds']}. Mean ± sample std.",
        "",
        "| cell | RQ | selected (lr, p) | final acc | forgetting | newest-exp acc | mean arrival |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, cell in result["cells"].items():
        sel = cell["selected_config"]
        d = cell["test"]
        lines.append(
            f"| {name} | {cell['question']} | "
            + (
                f"({sel['learning_rate']:g}, {sel['feature_power']:g})"
                if sel
                else "none"
            )
            + f" | {fmt(d['final_accuracy'])} | {fmt(d['mean_forgetting'])} | "
            f"{fmt(d['newest_accuracy'])} | {fmt(d['mean_arrival_accuracy'])} |"
        )
    lines += [
        "",
        f"## Paired differences against the reference `{REFERENCE}` and between sampling policies",
        "",
        "| a | b | metric | mean diff (a−b) | 95% CI | a higher / b higher |",
        "|---|---|---|---|---|---|",
    ]
    for comp in result["paired"]:
        for metric, p in comp["metrics"].items():
            ci = p["ci95_mean_difference"]
            lines.append(
                f"| {comp['a']} | {comp['b']} | {metric} | "
                + (
                    "—"
                    if p["mean_difference"] is None
                    else f"{p['mean_difference']:+.4f}"
                )
                + f" | {'—' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | "
                f"{p['a_higher']} / {p['b_higher']} |"
            )
    lines += ["", f"_{result['note']}_"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--grid", default="configs/polynomial_grid.json")
    parser.add_argument("--out", required=True, help="JSON path; never overwritten")
    parser.add_argument(
        "--cache-dir",
        help="directory for finished cells; a rerun skips them (resumable, e.g. after a timeout)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="stop starting new cells after this long and exit with code 3 (rerun to continue)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    base_protocol = Protocol.from_file(args.protocol)
    spec = load_grid(args.grid)
    if set(spec.seeds) & set(base_protocol.seeds):
        parser.error("selection seeds must not overlap the protocol's evaluation seeds")
    dataset = load_split_mnist(base_protocol, args.data_dir)
    started = time.perf_counter()

    cells = {}
    runs = {}
    for sampling in SAMPLINGS:
        protocol = dataclasses.replace(base_protocol, sampling=sampling)
        for condition, strategy in ROWS:
            name = label(condition, strategy, sampling)
            cache = (
                Path(args.cache_dir) / (name.replace(" / ", "__") + ".json")
                if args.cache_dir
                else None
            )
            if cache and cache.exists():
                cells[name] = json.loads(cache.read_text())
                runs[name] = cells[name]["test_records"]
                print(f"cached: {name}", flush=True)
                continue
            if args.max_seconds and time.perf_counter() - started > args.max_seconds:
                print("time budget reached; rerun to continue", flush=True)
                return 3
            print(f"tuning + testing: {name}", flush=True)

            def factory(entry, condition=condition, strategy=strategy):
                return PolynomialAdapter(
                    condition=condition,
                    strategy=strategy,
                    learning_rate=entry["learning_rate"],
                    feature_power=entry["feature_power"],
                    role="validation_candidate",
                )

            tuned = tune(
                "polynomial_cl",
                factory,
                spec.entries,
                dataset,
                protocol,
                selection_seeds=spec.seeds,
                expected_runs=spec.expected_runs,
                margin=spec.margin,
                context={"run_mode": "development", "exploratory": True, "cell": name},
            )
            chosen = tuned.selected_config
            records = []
            if chosen is not None:
                adapter = PolynomialAdapter(
                    condition=condition,
                    strategy=strategy,
                    learning_rate=chosen["learning_rate"],
                    feature_power=chosen["feature_power"],
                )
                records = run_test(adapter, dataset, protocol)
            runs[name] = records
            cells[name] = {
                "question": RQ_OF.get(
                    condition if condition != "expanding" else strategy,
                    "RQ2-RQ4 reference",
                )
                if (condition, strategy) != ("expanding", "ordinary")
                else "reference (RQ2/RQ3/RQ4)",
                "condition": condition,
                "strategy": strategy,
                "sampling": sampling,
                "selected_config": chosen,
                "validation_runs": tuned.budget_runs,
                "test": describe(records),
                "test_records": records,
            }
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(cells[name]))

    names = (
        "final_accuracy",
        "mean_forgetting",
        "newest_accuracy",
        "mean_arrival_accuracy",
    )
    comparisons = []
    for sampling in SAMPLINGS:
        ref = label("expanding", "ordinary", "class_balanced")
        for condition, strategy in ROWS:
            a = label(condition, strategy, sampling)
            if a == ref or not runs.get(a) or not runs.get(ref):
                continue
            comparisons.append(
                {
                    "a": a,
                    "b": ref,
                    "metrics": {
                        m: pair(runs[a], runs[ref], (a, ref), m) for m in names
                    },
                }
            )
    for condition, strategy in ROWS:  # RQ3: the same row under the two policies
        a = label(condition, strategy, "shuffled")
        b = label(condition, strategy, "class_balanced")
        if runs.get(a) and runs.get(b):
            comparisons.append(
                {
                    "a": a,
                    "b": b,
                    "metrics": {m: pair(runs[a], runs[b], (a, b), m) for m in names},
                }
            )
    result = {
        "exploratory": True,
        "run_mode": "development",
        "protocol_fingerprint": base_protocol.fingerprint(),
        "grid": Path(args.grid).name,
        "selection_seeds": list(spec.seeds),
        "test_seeds": list(base_protocol.seeds),
        "core": core_provenance(),
        "benchmark": benchmark_provenance(),
        "cells": cells,
        "paired": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "Exploratory: equal tuning budget (the approved 15-configuration shared-rate grid, "
            "selection seeds 100/101) for every cell; update rules are not given their own "
            "per-degree rates here. Seeds vary shuffling and replay selection, not the data "
            "split or projection; intervals are descriptive."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    text = report(result)
    out.with_suffix(".md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
