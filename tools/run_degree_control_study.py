# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""What does the core's degree control add? An independent evaluation under the benchmark protocol.

The core's degree-control experiment (``experiments/degree_control.py`` in the core) selects
per-degree factors on its own validation split and reports test results on its own seeds. This
tool asks the same question through the benchmark, so the answer does not rest on the core's
code path alone:

* every run goes through the benchmark's ``PolynomialAdapter`` (and ``ERMLPAdapter``), the shared
  experience plan, and the benchmark's own metrics recomputed from the accuracy matrix;
* candidates are generated and ranked with the core's pure helpers (``individual_candidates``,
  ``combination_candidates``, ``choose_feasible``: Pareto front of accuracy and forgetting, then
  accuracy minus forgetting, after an acquisition floor), unchanged;
* candidate selection uses ``validation_view`` and the benchmark's selection seeds (100, 101);
  the protocol's evaluation seeds are used once, for the final comparison, on the test split;
* the final comparison is paired by seed against each strategy's own all-ones baseline and, as
  reference, against the benchmark's tuned ER-MLP and tuned shared-rate polynomial.

Reported next to accuracy and forgetting is **accuracy on arrival** (the diagonal of the accuracy
matrix): lower forgetting obtained by learning each experience less well is not better continual
learning, and the core's own experiment shows that is what degree control does.

This is an exploratory, development-mode study. The candidate set is the core's, not a
maintainer-approved benchmark grid, so nothing here is an official benchmark result and it is not
fed to any experiment spec. The base per-degree rates of each strategy are the core's
(``BASELINES``), carried over unchanged and not re-tuned here.

    python tools/run_degree_control_study.py --data-dir data/mnist --out results/degree_control.json \\
        --er-tuning results/tuning_er_mlp.json --poly-tuning results/tuning_polynomial.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from polynomial_cl.continual.degree_control import (
        ACQUISITION_FLOOR,
        BASELINES,
        ONES,
        choose_feasible,
        combination_candidates,
        individual_candidates,
        is_feasible,
        name_of,
    )
    from polynomial_cl.continual.degree_study import tradeoff_score
except ImportError:  # a core without degree control
    sys.exit(
        "the installed core has no degree control (polynomial_cl.continual.degree_control); "
        "install a core that provides it"
    )

from polynomial_cl_bench import metrics
from polynomial_cl_bench.aggregate import paired_comparison, summarize
from polynomial_cl_bench.algorithms.er_mlp import ERMLPAdapter
from polynomial_cl_bench.algorithms.polynomial import PolynomialAdapter
from polynomial_cl_bench.datasets import load_split_mnist, validation_view
from polynomial_cl_bench.plan import make_plan
from polynomial_cl_bench.protocol import Protocol
from polynomial_cl_bench.provenance import benchmark_provenance, core_provenance
from polynomial_cl_bench.verify_data import DEFAULT_PROTOCOL


def run_seeds(adapter, dataset, protocol, seeds):
    """One record per seed with the benchmark's own metrics; failures are kept, never dropped."""
    records = []
    for seed in seeds:
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
                "plan_fingerprint": plan.fingerprint(),
                "arrival_accuracy": arrival,
                "final_accuracy_per_experience": metrics.accuracy_per_experience(m),
                "metrics": {
                    "final_accuracy": metrics.final_mean_accuracy(m),
                    "mean_forgetting": metrics.mean_forgetting(m),
                    "newest_accuracy": arrival[-1],
                    "mean_arrival_accuracy": float(np.mean(arrival)),
                    "min_arrival_accuracy": float(min(arrival)),
                },
            }
        )
    return records


def selection_metrics(records):
    """The quantities the core's selection helpers read, from benchmark records.

    ``None`` unless every selection seed completed (a candidate is eligible only then, the same
    rule as the benchmark's tuning).
    """
    if not records or any(r["status"] != "completed" for r in records):
        return None
    arrival_by_experience = np.mean([r["arrival_accuracy"] for r in records], axis=0)
    return {
        "final_mean_accuracy": float(
            np.mean([r["metrics"]["final_accuracy"] for r in records])
        ),
        "mean_forgetting": float(
            np.mean([r["metrics"]["mean_forgetting"] for r in records])
        ),
        "final_current_accuracy": float(
            np.mean([r["metrics"]["newest_accuracy"] for r in records])
        ),
        "min_mean_immediate_accuracy": float(arrival_by_experience.min()),
        "immediate_accuracy_by_experience": arrival_by_experience.tolist(),
    }


def select_for_strategy(key, view, protocol, seeds, floor, log):
    """Experiments A and B of the core's protocol on the validation view only."""
    base = BASELINES[key]
    measured, vectors, failed = {}, {}, []

    def measure(group, candidates):
        for name, factors in candidates.items():
            if name in measured or name in failed:
                continue
            adapter = PolynomialAdapter(
                strategy=base["strategy"],
                learning_rate=list(base["rates"]),
                degree_control_factors=list(factors),
            )
            m = selection_metrics(run_seeds(adapter, view, protocol, seeds))
            vectors[name] = list(factors)
            if m is None:
                failed.append(name)
                log(f"  [{key}/{group}] {name}: FAILED")
                continue
            measured[name] = m
            log(
                f"  [{key}/{group}] {name}: acc {m['final_mean_accuracy']:.3f} "
                f"forg {m['mean_forgetting']:.3f} min-arrival "
                f"{m['min_mean_immediate_accuracy']:.3f} "
                f"{'ok' if is_feasible(m, floor) else 'INFEASIBLE'}"
            )

    individual = individual_candidates()
    measure("A", individual)
    baseline_name = name_of(ONES)
    if baseline_name not in measured:
        raise SystemExit(f"{key}: the all-ones baseline failed on validation")
    scores = {n: tradeoff_score(measured[n]) for n in individual if n in measured}
    feasible_individual = {n for n in scores if is_feasible(measured[n], floor)}
    multiple = combination_candidates(scores, feasible=feasible_individual)
    measure("B", multiple)
    pools = {
        "best_individual": {n: measured[n] for n in individual if n in measured},
        "best_multiple": {n: measured[n] for n in multiple if n in measured},
        "best_overall": measured,
    }
    chosen = {
        label: (choose_feasible(pool, floor) if pool else None)
        for label, pool in pools.items()
    }
    return {
        "strategy": base["strategy"],
        "rates": list(base["rates"]),
        "acquisition_floor": floor,
        "n_candidates": len(measured) + len(failed),
        "failed_candidates": failed,
        "vectors": vectors,
        "validation_metrics": measured,
        "infeasible": [n for n, m in measured.items() if not is_feasible(m, floor)],
        "baseline_feasible": is_feasible(measured[baseline_name], floor),
        "selected": chosen,
        "no_feasible_candidate": [k for k, v in chosen.items() if v is None],
    }


def describe_runs(records):
    ok = [r for r in records if r["status"] == "completed"]
    out = {
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "arrival_accuracy_by_experience": (
            np.mean([r["arrival_accuracy"] for r in ok], axis=0).tolist()
            if ok
            else None
        ),
    }
    for name in (
        "final_accuracy",
        "mean_forgetting",
        "newest_accuracy",
        "mean_arrival_accuracy",
        "min_arrival_accuracy",
    ):
        out[name] = summarize([r["metrics"][name] for r in ok])
    return out


def paired(a, b, labels):
    return {
        metric: paired_comparison(a, b, metric, labels)
        for metric in (
            "final_accuracy",
            "mean_forgetting",
            "newest_accuracy",
            "mean_arrival_accuracy",
        )
    }


def selected_hyperparameters(path, algorithm):
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if data.get("algorithm") != algorithm or data.get("selected_config") is None:
        sys.exit(f"{path}: not a finished tuning result for {algorithm}")
    return data["selected_config"]


def fmt(s, digits=4):
    return (
        "—"
        if s["mean"] is None
        else f"{s['mean']:.{digits}f} ± {s['std'] or 0:.{digits}f}"
    )


def report(result):
    lines = [
        "# Degree control through the benchmark (exploratory, development mode)",
        "",
        f"Protocol {result['protocol']['name']} v{result['protocol']['version']}; selection on the "
        f"validation view with seeds {result['selection_seeds']}, final comparison on the test "
        f"split with seeds {result['test_seeds']}; acquisition floor {result['floor']:g}.",
        "",
        "Mean ± sample std over test seeds. **Arrival** = accuracy on an experience right after it "
        "was learned (the diagonal); low forgetting bought with low arrival accuracy is not "
        "better continual learning.",
        "",
        "| method | final acc | forgetting | newest-exp acc | min arrival | arrival by experience |",
        "|---|---|---|---|---|---|",
    ]
    for label, d in result["test"].items():
        arrival = d["arrival_accuracy_by_experience"]
        lines.append(
            f"| {label} | {fmt(d['final_accuracy'])} | {fmt(d['mean_forgetting'])} | "
            f"{fmt(d['newest_accuracy'])} | {fmt(d['min_arrival_accuracy'])} | "
            + (" / ".join(f"{v:.3f}" for v in arrival) if arrival else "—")
            + " |"
        )
    lines += [
        "",
        "## Paired differences (a − b, same seed)",
        "",
        "| a | b | metric | mean diff | 95% CI | a higher / b higher |",
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
    lines += ["", "## Selection on validation data", ""]
    for key, sel in result["selection"].items():
        lines.append(
            f"- **{key}**: {sel['n_candidates']} candidates, "
            f"{len(sel['infeasible'])} infeasible under the floor; baseline feasible: "
            f"{sel['baseline_feasible']}; selected (best overall): "
            f"`{sel['selected']['best_overall']}`"
            + (
                f"; no feasible candidate for {sel['no_feasible_candidate']}"
                if sel["no_feasible_candidate"]
                else ""
            )
        )
    lines += [
        "",
        f"_{result['note']}_",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--out", required=True, help="JSON path; never overwritten")
    parser.add_argument("--strategies", nargs="+", default=list(BASELINES))
    parser.add_argument("--floor", type=float, default=ACQUISITION_FLOOR)
    parser.add_argument("--select-seeds", nargs="+", type=int, default=[100, 101])
    parser.add_argument(
        "--er-tuning", help="tuning_er_mlp.json, for the ER-MLP reference"
    )
    parser.add_argument(
        "--poly-tuning", help="tuning_polynomial.json, shared-rate reference"
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    for s in args.strategies:
        if s not in BASELINES:
            parser.error(f"unknown strategy {s!r}; choose from {list(BASELINES)}")
    if not 0.0 <= args.floor <= 1.0:
        parser.error("--floor must be in [0, 1]")
    protocol = Protocol.from_file(args.protocol)
    if set(args.select_seeds) & set(protocol.seeds):
        parser.error("selection seeds must not overlap the protocol's evaluation seeds")

    def log(message):
        print(message, flush=True)

    dataset = load_split_mnist(protocol, args.data_dir)  # verifies the archive hashes
    view = validation_view(dataset, protocol)
    started = time.perf_counter()

    selection = {}
    for key in args.strategies:
        log(f"selecting for {key} on validation data (floor {args.floor:g})")
        selection[key] = select_for_strategy(
            key, view, protocol, args.select_seeds, args.floor, log
        )

    test_runs, vectors = {}, {}

    def run_test(label, adapter):
        test_runs[label] = run_seeds(adapter, dataset, protocol, protocol.seeds)

    for key in args.strategies:
        base = BASELINES[key]
        run_test(
            f"{key}: baseline",
            PolynomialAdapter(
                strategy=base["strategy"],
                learning_rate=list(base["rates"]),
                degree_control_factors=list(ONES),
            ),
        )
        chosen = selection[key]["selected"]["best_overall"]
        if chosen is not None:
            vectors[f"{key}: degree control"] = selection[key]["vectors"][chosen]
            run_test(
                f"{key}: degree control",
                PolynomialAdapter(
                    strategy=base["strategy"],
                    learning_rate=list(base["rates"]),
                    degree_control_factors=selection[key]["vectors"][chosen],
                ),
            )
    reference = {}
    cfg = selected_hyperparameters(args.poly_tuning, "polynomial_cl")
    if cfg:
        reference["polynomial: tuned shared rate"] = cfg
        run_test(
            "polynomial: tuned shared rate",
            PolynomialAdapter(
                learning_rate=cfg["learning_rate"], feature_power=cfg["feature_power"]
            ),
        )
    cfg = selected_hyperparameters(args.er_tuning, "er_mlp")
    if cfg:
        reference["er_mlp: tuned"] = cfg
        run_test(
            "er_mlp: tuned",
            ERMLPAdapter(hidden=cfg["hidden"], learning_rate=cfg["learning_rate"]),
        )

    comparisons = []
    for key in args.strategies:
        a, b = f"{key}: degree control", f"{key}: baseline"
        if a in test_runs:
            comparisons.append(
                {"a": a, "b": b, "metrics": paired(test_runs[a], test_runs[b], (a, b))}
            )
        for ref in ("er_mlp: tuned",):
            for label in (f"{key}: baseline", f"{key}: degree control"):
                if label in test_runs and ref in test_runs:
                    comparisons.append(
                        {
                            "a": label,
                            "b": ref,
                            "metrics": paired(
                                test_runs[label], test_runs[ref], (label, ref)
                            ),
                        }
                    )
    result = {
        "exploratory": True,
        "run_mode": "development",
        "protocol": {"name": protocol.name, "version": protocol.version},
        "protocol_fingerprint": protocol.fingerprint(),
        "floor": args.floor,
        "selection_seeds": list(args.select_seeds),
        "test_seeds": list(protocol.seeds),
        "core": core_provenance(),
        "benchmark": benchmark_provenance(),
        "selection": selection,
        "selected_vectors": vectors,
        "reference_configs": reference,
        "test": {label: describe_runs(r) for label, r in test_runs.items()},
        "test_runs": test_runs,
        "paired": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "Exploratory: the candidate set is the core's, not a maintainer-approved benchmark "
            "grid; base rates per strategy are the core's, not re-tuned here; one dataset, one "
            "projection, seeds vary shuffling and replay selection only."
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
