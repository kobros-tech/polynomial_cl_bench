# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Analyse the core's per-degree study (``experiments/run_per_degree_study.sh``) -> JSON + tables.

Reads, from ``--results-dir`` (the core's ``results/per_degree``):

* ``cell_<strategy>_<sampling>.json``   one cell per (strategy, sampling): 18 learning-rate
  vectors, each with validation metrics (selection seeds) and test runs (test seeds), 3 epochs;
* ``epochs10_<strategy>.json``          the shared and the selected vector re-run at 10 epochs
  (test seeds only, no validation);
* ``logs/*.log``                        the only place wall-clock time per run is recorded.

Four selection criteria are applied, each reporting the OTHER two metrics and the parameters:

1. highest accuracy      (ties: lower forgetting)
2. lowest forgetting     (ties: higher accuracy)
3. smallest time         (ties: higher accuracy)
4. best trade-off        distance to the ideal point after min-max normalisation of the
                         three objectives (accuracy up, forgetting down, time down). This is a
                         CONVENTION, not a measurement: ``tradeoff_sensitivity`` lists the
                         winner under other weightings so the dependence is visible.

Two ranking bases are produced and must not be confused:

* ``validation``  candidates are ranked on the validation split (selection seeds); their TEST
                  results are then reported. This is an honest estimate of what a practitioner
                  who selects without peeking at the test set obtains.
* ``test``        candidates are ranked on the test seeds themselves, and the 10-epoch
                  re-runs take part. This is descriptive (the best cell in hindsight) and
                  optimistically biased as an estimate of generalisation.

Low forgetting is not a virtue in itself: a model that learns little forgets little. Every row
therefore carries newest-experience accuracy so such degenerate winners are visible.

Standard library only. Run:

    python tools/analyze_per_degree_study.py --results-dir ../polynomial_cl_demo/results/per_degree \\
        --out-json results/analysis/per_degree_analysis.json \\
        --out-md   results/analysis/per_degree_analysis.md \\
        --out-csv  results/analysis/per_degree_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"
CRITERIA = ("max_accuracy", "min_forgetting", "min_time", "best_tradeoff")
CRITERION_TITLES = {
    "max_accuracy": "Highest accuracy",
    "min_forgetting": "Lowest forgetting",
    "min_time": "Smallest time",
    "best_tradeoff": "Best trade-off (accuracy, forgetting, time)",
}
# Wall time per run (seconds) recorded by the core's scripts, e.g. "... seed 3: acc 0.770 forg 0.092 (3.6s)"
SEED_LINE = re.compile(
    r"^\s+(?P<label>.+?) seed (?P<seed>\d+): acc (?P<acc>[\d.]+) forg (?P<forg>[\d.]+) \((?P<sec>[\d.]+)s\)\s*$"
)
TEST_HEADER = re.compile(
    r"^\s*\[(?P<strategy>\w+)/(?P<sampling>\w+)\] test \d+/\d+: (?P<label>\S+) "
)
VALIDATION_HEADER = re.compile(
    r"^\s*\[(?P<strategy>\w+)/(?P<sampling>\w+)\] validation: (?P<label>.+?)\s*$"
)
SHARED_LR = re.compile(r"^shared lr=(?P<lr>[\d.eE+-]+)$")


# --------------------------------------------------------------------------- log parsing


def parse_logs(log_dir: Path):
    """Return ({(strategy, sampling, label, seed): (acc, forg, sec)} for test runs,
    the same for validation runs keyed by the vector label or ('shared', lr)).

    Keys carry the epoch count (10 for ``epochs10_*.log``, else 3) because the 10-epoch re-runs
    reuse the strategy, sampling, label and seed of the 3-epoch runs. A key that still appears
    twice (a re-run appended to a log) keeps the LAST value and is counted in ``duplicates``.
    """
    test, validation, duplicates = {}, {}, 0
    for path in sorted(log_dir.glob("*.log")):
        epochs = 10 if path.name.startswith("epochs10_") else 3
        mode = None  # ("test"|"validation", strategy, sampling, label)
        for line in path.read_text(errors="replace").splitlines():
            m = TEST_HEADER.match(line)
            if m:
                mode = ("test", m["strategy"], m["sampling"], m["label"])
                continue
            m = VALIDATION_HEADER.match(line)
            if m:
                mode = (
                    "validation",
                    m["strategy"],
                    m["sampling"],
                    m["label"].split(" [")[0],
                )
                continue
            m = SEED_LINE.match(line)
            if not m or mode is None:
                continue
            kind, strategy, sampling, label = mode
            key = (strategy, sampling, label, epochs, int(m["seed"]))
            record = (float(m["acc"]), float(m["forg"]), float(m["sec"]))
            target = test if kind == "test" else validation
            if key in target:
                duplicates += 1
            target[key] = record
    return test, validation, duplicates


# --------------------------------------------------------------------------- statistics


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def std(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def summarise_runs(runs):
    return {
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "accuracy_mean": mean([r["final_mean_accuracy"] for r in runs]),
        "accuracy_std": std([r["final_mean_accuracy"] for r in runs]),
        "forgetting_mean": mean([r["mean_forgetting"] for r in runs]),
        "forgetting_std": std([r["mean_forgetting"] for r in runs]),
        "current_accuracy_mean": mean([r["final_current_accuracy"] for r in runs]),
        "current_accuracy_std": std([r["final_current_accuracy"] for r in runs]),
    }


def feature_count(run):
    counts = run.get("feature_count_by_task")
    return counts[-1] if counts else None


# --------------------------------------------------------------------------- candidates


def load_candidates(results_dir: Path, problems: list):
    test_log, val_log, duplicates = parse_logs(results_dir / "logs")
    if duplicates:
        problems.append(
            f"{duplicates} log lines were duplicated; the last occurrence was used"
        )
    candidates = []

    def timing(log, strategy, sampling, label, epochs, runs, problems_prefix):
        """Mean wall time over the runs, after checking each log line against the JSON run."""
        seconds = []
        for run in runs:
            hit = log.get((strategy, sampling, label, epochs, run["seed"]))
            if hit is None:
                continue
            acc, forg, sec = hit
            if (
                abs(acc - run["final_mean_accuracy"]) > 6e-4
                or abs(forg - run["mean_forgetting"]) > 6e-4
            ):
                problems.append(
                    f"{problems_prefix} seed {run['seed']}: log ({acc}, {forg}) does not match JSON "
                    f"({run['final_mean_accuracy']:.3f}, {run['mean_forgetting']:.3f}); time ignored"
                )
                continue
            seconds.append(sec)
        if len(seconds) != len(runs):
            problems.append(
                f"{problems_prefix}: time found for {len(seconds)}/{len(runs)} runs"
                + ("" if seconds else "; time unavailable")
            )
        return (mean(seconds), std(seconds)) if seconds else (None, None)

    # ---- 3-epoch cells: 18 vectors each, validation + test
    for path in sorted(results_dir.glob("cell_*.json")):
        data = json.loads(path.read_text())
        protocol = data.get("protocol", {})
        for cell in data["cells"]:
            strategy, sampling = cell["strategy"], cell["sampling"]
            base = cell["base_rate"]
            for label, vector in cell["vectors"].items():
                runs = cell["runs"].get(label, [])
                if not runs:
                    continue
                val = cell["validation_metrics"].get(label)
                prefix = f"{strategy}/{sampling}/{label}"
                t_mean, t_std = timing(
                    test_log,
                    strategy,
                    sampling,
                    label,
                    protocol.get("epochs", 3),
                    runs,
                    prefix + " test",
                )
                v_time = None
                if val is not None:
                    key_label = f"shared lr={base:g}" if label == "shared" else label
                    secs = [
                        v[2]
                        for (s, sp, lb, ep, _seed), v in val_log.items()
                        if (s, sp, lb, ep) == (strategy, sampling, key_label, 3)
                    ]
                    v_time = mean(secs)
                candidates.append(
                    {
                        "id": f"{strategy}|{sampling}|{label}|e{protocol.get('epochs', 3)}",
                        "source_file": path.name,
                        "strategy": strategy,
                        "sampling": sampling,
                        "vector_label": label,
                        "learning_rates_by_degree": vector,
                        "shared_base_rate": base,
                        "epochs": protocol.get("epochs", 3),
                        "batch_size": protocol.get("batch_size"),
                        "memory_per_class": protocol.get("memory_per_class"),
                        "n_ref_per_class": protocol.get("n_ref_per_class")
                        if strategy.startswith("strategy_b")
                        else None,
                        "degree_schedule": runs[0].get("active_degree_by_task"),
                        "n_features_final": feature_count(runs[0]),
                        "selected_by_core": cell.get("selected") == label,
                        "test": {
                            **summarise_runs(runs),
                            "time_seconds_mean": t_mean,
                            "time_seconds_std": t_std,
                        },
                        "validation": None
                        if val is None
                        else {
                            "accuracy_mean": val["final_mean_accuracy"],
                            "forgetting_mean": val["mean_forgetting"],
                            "current_accuracy_mean": val["final_current_accuracy"],
                            "time_seconds_mean": v_time,
                            "seeds": cell.get("selection_seeds")
                            or data.get("selection_seeds"),
                        },
                    }
                )

    # ---- 10-epoch re-runs: test only
    for path in sorted(results_dir.glob("epochs10_*.json")):
        data = json.loads(path.read_text())
        for cell in data["cells"]:
            strategy, sampling, epochs = (
                cell["strategy"],
                cell["sampling"],
                cell["epochs"],
            )
            protocol = data.get("protocol", {})
            for label, vector in cell["vectors"].items():
                runs = cell["runs"].get(label, [])
                if not runs:
                    continue
                prefix = f"{strategy}/{sampling}/{label}@e{epochs}"
                t_mean, t_std = timing(
                    test_log, strategy, sampling, label, epochs, runs, prefix
                )
                candidates.append(
                    {
                        "id": f"{strategy}|{sampling}|{label}|e{epochs}",
                        "source_file": path.name,
                        "strategy": strategy,
                        "sampling": sampling,
                        "vector_label": label,
                        "learning_rates_by_degree": vector,
                        "shared_base_rate": None,
                        "epochs": epochs,
                        "batch_size": protocol.get("batch_size"),
                        "memory_per_class": protocol.get("memory_per_class"),
                        "n_ref_per_class": protocol.get("n_ref_per_class")
                        if strategy.startswith("strategy_b")
                        else None,
                        "degree_schedule": runs[0].get("active_degree_by_task"),
                        "n_features_final": feature_count(runs[0]),
                        "selected_by_core": cell.get("selected") == label,
                        "test": {
                            **summarise_runs(runs),
                            "time_seconds_mean": t_mean,
                            "time_seconds_std": t_std,
                        },
                        "validation": None,
                    }
                )
    return candidates


# --------------------------------------------------------------------------- selection


def view(candidate, basis):
    """(accuracy, forgetting, time, current_accuracy) of a candidate on the ranking basis."""
    block = candidate["test"] if basis == "test" else candidate["validation"]
    if block is None:
        return None
    return (
        block["accuracy_mean"],
        block["forgetting_mean"],
        block["time_seconds_mean"],
        block["current_accuracy_mean"],
    )


def normalise(values, lower_is_better):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    scaled = [(v - lo) / span for v in values]  # 0 = smallest, 1 = largest
    return [s if lower_is_better else 1.0 - s for s in scaled]  # 0 = best, 1 = worst


def dominates(a, b):
    """a dominates b on (accuracy up, forgetting down, time down)."""
    ge = a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]
    gt = a[0] > b[0] or a[1] < b[1] or a[2] < b[2]
    return ge and gt


def pareto_front(pool, basis):
    front = []
    for c in pool:
        v = view(c, basis)[:3]
        if not any(dominates(view(o, basis)[:3], v) for o in pool if o is not c):
            front.append(c)
    return front


def tradeoff_scores(pool, basis, weights=(1.0, 1.0, 1.0)):
    acc, forg, time_ = zip(*(view(c, basis)[:3] for c in pool))
    badness = (
        normalise(acc, lower_is_better=False),
        normalise(forg, lower_is_better=True),
        normalise(time_, lower_is_better=True),
    )
    scores = []
    for i in range(len(pool)):
        scores.append(
            math.sqrt(sum(w * badness[k][i] ** 2 for k, w in enumerate(weights)))
        )
    return scores


def pick(pool, basis, criterion, weights=(1.0, 1.0, 1.0)):
    if criterion == "max_accuracy":
        return min(
            pool,
            key=lambda c: (-view(c, basis)[0], view(c, basis)[1], view(c, basis)[2]),
        )
    if criterion == "min_forgetting":
        return min(
            pool,
            key=lambda c: (view(c, basis)[1], -view(c, basis)[0], view(c, basis)[2]),
        )
    if criterion == "min_time":
        return min(
            pool,
            key=lambda c: (view(c, basis)[2], -view(c, basis)[0], view(c, basis)[1]),
        )
    scores = tradeoff_scores(pool, basis, weights)
    return pool[
        min(range(len(pool)), key=lambda i: (scores[i], -view(pool[i], basis)[0]))
    ]


def describe(candidate, basis, criterion):
    v = view(candidate, basis)
    t = candidate["test"]
    return {
        "criterion": criterion,
        "ranking_basis": basis,
        "candidate_id": candidate["id"],
        "strategy": candidate["strategy"],
        "sampling": candidate["sampling"],
        "vector_label": candidate["vector_label"],
        "learning_rates_by_degree": candidate["learning_rates_by_degree"],
        "epochs": candidate["epochs"],
        "batch_size": candidate["batch_size"],
        "memory_per_class": candidate["memory_per_class"],
        "n_ref_per_class": candidate["n_ref_per_class"],
        "degree_schedule": candidate["degree_schedule"],
        "n_features_final": candidate["n_features_final"],
        "ranking_values": {
            "accuracy": v[0],
            "forgetting": v[1],
            "time_seconds": v[2],
            "current_accuracy": v[3],
        },
        "test_result": {k: t[k] for k in t},
        "selected_by_core_study": candidate["selected_by_core"],
    }


def analyse(candidates, problems, accuracy_floor):
    analysis = {}
    for basis in ("validation", "test"):
        pool = [c for c in candidates if view(c, basis) is not None]
        dropped = [c["id"] for c in pool if view(c, basis)[2] is None]
        if dropped:
            problems.append(
                f"{basis} basis: {len(dropped)} candidates have no time and were excluded from ranking"
            )
        pool = [c for c in pool if view(c, basis)[2] is not None]
        if not pool:
            continue
        winners = {
            crit: describe(pick(pool, basis, crit), basis, crit) for crit in CRITERIA
        }
        sensitivity = {}
        for name, weights in {
            "equal_weights": (1.0, 1.0, 1.0),
            "accuracy_x2": (2.0, 1.0, 1.0),
            "forgetting_x2": (1.0, 2.0, 1.0),
            "time_x2": (1.0, 1.0, 2.0),
            "time_ignored": (1.0, 1.0, 0.0),
        }.items():
            w = pick(pool, basis, "best_tradeoff", weights)
            sensitivity[name] = {
                "weights_accuracy_forgetting_time": list(weights),
                "candidate_id": w["id"],
            }
        front = pareto_front(pool, basis)
        best_accuracy = max(view(c, basis)[0] for c in pool)
        floor_pool = [
            c for c in pool if view(c, basis)[0] >= best_accuracy - accuracy_floor
        ]
        constrained = {
            crit: describe(pick(floor_pool, basis, crit), basis, crit)
            for crit in ("min_forgetting", "min_time", "best_tradeoff")
        }
        analysis[basis] = {
            "n_candidates": len(pool),
            "winners": winners,
            "accuracy_floor": {
                "definition": f"candidates whose {basis} accuracy is within {accuracy_floor} (absolute) of the best",
                "floor_value": best_accuracy - accuracy_floor,
                "n_candidates": len(floor_pool),
                "winners": constrained,
            },
            "tradeoff_sensitivity": sensitivity,
            "pareto_front_ids": sorted(c["id"] for c in front),
            "distinct_winners": len({w["candidate_id"] for w in winners.values()}),
        }
    return analysis


# --------------------------------------------------------------------------- output


def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}"


def lr_text(vector):
    return "[" + ", ".join(f"{v:g}" for v in vector) + "]"


def table_row(title, w):
    t = w["test_result"]
    return (
        f"| {title} | {w['strategy']} | {w['sampling']} | {w['epochs']} | "
        f"{lr_text(w['learning_rates_by_degree'])} | "
        f"{pct(t['accuracy_mean'])} ± {pct(t['accuracy_std'])} | "
        f"{pct(t['forgetting_mean'])} ± {pct(t['forgetting_std'])} | "
        f"{pct(t['current_accuracy_mean'])} | "
        f"{'n/a' if t['time_seconds_mean'] is None else format(t['time_seconds_mean'], '.2f')} |"
    )


TABLE_HEAD = [
    "| Criterion | Strategy | Sampling | Epochs | lr (deg 0-3) | Accuracy | Forgetting | Newest-exp. acc. | Time (s) |",
    "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
]


def markdown(analysis, candidates, results_dir):
    lines = [
        "# Per-degree study: best settings by criterion",
        "",
        f"Source: `{results_dir}`. Accuracy and forgetting are percentages (mean ± std over the test seeds); "
        "time is mean wall-clock seconds per complete run (5 experiences + evaluation) from the study logs, "
        "on the machine that produced them. `lr` is the learning rate per polynomial degree group (degrees 0-3).",
        "",
    ]
    notes = {
        "validation": "Ranked on the validation split (selection seeds); the table reports the TEST result of the "
        "validation-ranked winner. Use this basis for claims.",
        "test": "Ranked on the test seeds themselves (best in hindsight, 10-epoch re-runs included). "
        "Descriptive only: optimistically biased.",
    }
    for basis in ("validation", "test"):
        if basis not in analysis:
            continue
        block = analysis[basis]
        lines += [
            f"## Ranking basis: {basis}",
            "",
            notes[basis],
            f"{block['n_candidates']} candidates; {block['distinct_winners']} distinct winners across the "
            f"four criteria; {len(block['pareto_front_ids'])} on the Pareto front.",
            "",
            *TABLE_HEAD,
        ]
        for crit in CRITERIA:
            lines.append(table_row(CRITERION_TITLES[crit], block["winners"][crit]))
        floor = block["accuracy_floor"]
        lines += [
            "",
            f"Same criteria restricted to candidates within the accuracy floor ({floor['definition']}; "
            f"{floor['n_candidates']} candidates), which removes winners that are cheap or stable only "
            "because they learn little:",
            "",
            *TABLE_HEAD,
        ]
        for crit, w in floor["winners"].items():
            lines.append(table_row(CRITERION_TITLES[crit] + " (floor)", w))
        lines += [
            "",
            "Trade-off winner under other weightings: "
            + "; ".join(
                f"{name}: `{s['candidate_id']}`"
                for name, s in block["tradeoff_sensitivity"].items()
            ),
            "",
        ]
    lines += [
        "Trade-off definition: Euclidean distance to the ideal point after min-max normalising "
        "accuracy (up), forgetting (down) and time (down) over the candidate pool. Changing the weights or "
        "the pool can change the winner (see the sensitivity line); it is a convention, not a measurement.",
        "",
        f"Candidates analysed: {len(candidates)}.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, candidates):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "strategy", "sampling", "vector_label", "epochs",
        "lr_deg0", "lr_deg1", "lr_deg2", "lr_deg3",
        "test_accuracy_mean", "test_accuracy_std", "test_forgetting_mean", "test_forgetting_std",
        "test_current_accuracy_mean", "test_time_seconds_mean", "test_n_seeds",
        "val_accuracy_mean", "val_forgetting_mean", "val_current_accuracy_mean", "val_time_seconds_mean",
        "selected_by_core",
    ]  # fmt: skip
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            t, v = c["test"], c["validation"] or {}
            lr = c["learning_rates_by_degree"]
            writer.writerow(
                {
                    "id": c["id"], "strategy": c["strategy"], "sampling": c["sampling"],
                    "vector_label": c["vector_label"], "epochs": c["epochs"],
                    "lr_deg0": lr[0], "lr_deg1": lr[1], "lr_deg2": lr[2], "lr_deg3": lr[3],
                    "test_accuracy_mean": t["accuracy_mean"], "test_accuracy_std": t["accuracy_std"],
                    "test_forgetting_mean": t["forgetting_mean"], "test_forgetting_std": t["forgetting_std"],
                    "test_current_accuracy_mean": t["current_accuracy_mean"],
                    "test_time_seconds_mean": t["time_seconds_mean"], "test_n_seeds": t["n_seeds"],
                    "val_accuracy_mean": v.get("accuracy_mean"), "val_forgetting_mean": v.get("forgetting_mean"),
                    "val_current_accuracy_mean": v.get("current_accuracy_mean"),
                    "val_time_seconds_mean": v.get("time_seconds_mean"),
                    "selected_by_core": c["selected_by_core"],
                }
            )  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="the core's results/per_degree directory",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--accuracy-floor", type=float, default=0.02,
        help="absolute accuracy margin below the best candidate for the constrained criteria (default 0.02)",
    )  # fmt: skip
    args = parser.parse_args(argv)

    if not (args.results_dir / "logs").is_dir() or not list(
        args.results_dir.glob("cell_*.json")
    ):
        parser.error(
            f"{args.results_dir} has no cell_*.json files or no logs/ directory"
        )

    problems: list = []
    candidates = load_candidates(args.results_dir, problems)
    if not candidates:
        parser.error("no candidates found")
    analysis = analyse(candidates, problems, args.accuracy_floor)

    report = {
        "schema_version": SCHEMA_VERSION,
        "source_directory": str(args.results_dir),
        "n_candidates": len(candidates),
        "criteria": list(CRITERIA),
        "tradeoff_definition": "Euclidean distance to the ideal point after min-max normalisation of "
        "accuracy (up), forgetting (down) and time (down) over the candidate pool; a convention.",
        "time_source": "wall-clock seconds per complete run parsed from logs/*.log (not stored in the result JSON); "
        "machine-dependent",
        "problems": problems,
        "analysis": analysis,
        "candidates": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown(analysis, candidates, args.results_dir))
    if args.out_csv:
        write_csv(args.out_csv, candidates)
    for p in problems:
        print("WARNING:", p, file=sys.stderr)
    print(f"{len(candidates)} candidates -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
