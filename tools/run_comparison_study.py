# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Re-run the best polynomial settings and the ER-MLP competitor under ONE protocol, ONE machine.

Inputs
    --analysis     JSON written by ``tools/analyze_per_degree_study.py`` (names the polynomial settings)
    --core-results the core's ``results/per_degree`` directory (to check re-runs against stored results)
    --data-dir     MNIST archives (sha256-verified by the protocol)

What runs (all on the protocol of ``configs/split_mnist_cil_v1.json``: Split MNIST CIL, 16-d fixed
projection, 3 epochs, batch 128, 200 replay exemplars per class, class_balanced unless a setting
says otherwise, hold-out of 10% for validation):

1. ER-MLP grid (hidden x learning rate) on the VALIDATION view, selection seeds 0-2.
2. The same grid on the TEST data, test seeds 3-9 (the core study's seeds), so ER-MLP can be put
   through the same four criteria as the polynomial method. Selection never reads these.
3. The polynomial settings named in the analysis JSON, on the test seeds (re-timed here).
4. Polynomial ablations (``fixed_linear``, ``fixed_capacity``), shared learning rate chosen on
   validation from the core study's grid {0.01 ... 30}, then run on the test seeds.
5. 10-epoch re-runs of the 3-epoch winners (no re-selection), as in the core study.

Every run is appended to ``runs.jsonl`` (resumable; a finished (spec, split, epochs, seed) is
skipped). Failed or diverged runs are recorded as failures, never dropped. Wall time is the
adapters' own ``wall_time_seconds_including_evaluation`` (identical definition for both
algorithms; one-time polynomial feature construction is excluded and timed separately).

The ER-MLP grid is wider than the maintainer-approved ``configs/er_mlp_grid.json`` (it is a
superset: adds hidden=() i.e. a linear head, hidden=512 and learning rates 3 and 10). It is a
research choice for this study and is recorded as such in ``summary.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import analyze_per_degree_study as analysis_tool

from polynomial_cl_bench import metrics
from polynomial_cl_bench.algorithms.er_mlp import ERMLPAdapter
from polynomial_cl_bench.algorithms.polynomial import PolynomialAdapter
from polynomial_cl_bench.datasets import load_split_mnist, validation_view
from polynomial_cl_bench.protocol import Protocol

SELECTION_SEEDS = (0, 1, 2)
TEST_SEEDS = (3, 4, 5, 6, 7, 8, 9)
ER_HIDDEN = ((), (64,), (128,), (256,), (512,))
ER_LRS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
ABLATION_LRS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
N_CLASSES = 10


# ------------------------------------------------------------------------ specs


def er_label(hidden, lr, replay=True):
    h = "x".join(map(str, hidden)) if hidden else "linear"
    return f"er_mlp|h{h}|lr{lr:g}" + ("" if replay else "|no_replay")


def er_spec(hidden, lr, replay=True, sampling="class_balanced"):
    return {
        "label": er_label(hidden, lr, replay),
        "algorithm": "er_mlp",
        "hidden": list(hidden),
        "learning_rate": lr,
        "replay": replay,
        "sampling": sampling,
    }


def poly_label(condition, strategy, sampling, lr):
    lr_text = (
        "[" + ",".join(f"{v:g}" for v in lr) + "]"
        if isinstance(lr, (list, tuple))
        else f"{lr:g}"
    )
    return f"poly|{condition}|{strategy}|{sampling}|lr{lr_text}"


def poly_spec(condition, strategy, sampling, lr):
    return {
        "label": poly_label(condition, strategy, sampling, lr),
        "algorithm": "polynomial_cl",
        "condition": condition,
        "strategy": strategy,
        "learning_rate": list(lr) if isinstance(lr, (list, tuple)) else lr,
        "sampling": sampling,
    }


def build_adapter(spec):
    if spec["algorithm"] == "er_mlp":
        return ERMLPAdapter(
            hidden=spec["hidden"],
            learning_rate=spec["learning_rate"],
            replay=spec["replay"],
            role="primary" if spec["replay"] else "ablation_no_replay",
        )
    lr = spec["learning_rate"]
    return PolynomialAdapter(
        condition=spec["condition"],
        strategy=spec["strategy"],
        learning_rate=np.asarray(lr, dtype=float) if isinstance(lr, list) else lr,
        feature_power=0.0,
        role="experimental",
    )


# ------------------------------------------------------------------------ storage


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[self.key(r)] = r

    @staticmethod
    def key(r):
        return (r["label"], r["split"], r["epochs"], r["sampling"], r["seed"])

    def get(self, label, split, epochs, sampling, seed):
        return self.records.get((label, split, epochs, sampling, seed))

    def add(self, record):
        self.records[self.key(record)] = record
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------------ running


class Study:
    def __init__(self, protocol, dataset, store, log):
        self.base_protocol = protocol
        self.test_data = dataset
        self.val_data = validation_view(dataset, protocol)
        self.store = store
        self.log = log
        self._adapters = {}
        self._features = {}  # dataset fingerprint -> read-only FeatureData shared by all polynomial adapters

    def protocol_for(self, epochs, sampling):
        p = dataclasses.replace(self.base_protocol, epochs=epochs, sampling=sampling)
        p.validate()
        return p

    def adapter(self, spec, split):
        """One adapter at a time; polynomial feature matrices are shared per dataset (memory)."""
        key = (spec["label"], split)
        if key not in self._adapters:
            self._adapters.clear()
            adapter = build_adapter(spec)
            if spec["algorithm"] == "polynomial_cl":
                dataset = self.test_data if split == "test" else self.val_data
                if dataset.fingerprint not in self._features:
                    self._features[dataset.fingerprint] = adapter._feature_data(dataset)
                adapter._features = (
                    dataset.fingerprint,
                    self._features[dataset.fingerprint],
                )
            self._adapters[key] = adapter
        return self._adapters[key]

    def run(self, spec, split, epochs, seed):
        sampling = spec["sampling"]
        done = self.store.get(spec["label"], split, epochs, sampling, seed)
        if done is not None:
            return done
        protocol = self.protocol_for(epochs, sampling)
        dataset = self.test_data if split == "test" else self.val_data
        adapter = self.adapter(spec, split)
        record = {
            "schema_version": "1.0",
            "label": spec["label"],
            "spec": spec,
            "split": split,
            "epochs": epochs,
            "sampling": sampling,
            "seed": seed,
            "protocol_fingerprint": protocol.fingerprint(),
        }
        try:
            result = adapter.run_sequence(dataset, protocol, seed)
            matrix = result.accuracy_matrix
            record.update(
                status="completed",
                accuracy_matrix=[
                    [None if math.isnan(v) else float(v) for v in row] for row in matrix
                ],
                final_accuracy=float(metrics.final_mean_accuracy(matrix)),
                mean_forgetting=float(metrics.mean_forgetting(matrix)),
                current_accuracy=float(matrix[-1, -1]),
                accuracy_per_experience=[float(v) for v in matrix[-1]],
                wall_seconds=float(
                    result.resources["wall_time_seconds_including_evaluation"]
                ),
                train_seconds=result.resources.get("training_time_seconds"),
                infer_seconds=result.resources.get("inference_time_seconds"),
                n_parameters=int(result.resources["n_parameters"]),
                replay_examples_by_task=result.replay_examples_used_by_task,
                plan_fingerprint=result.run_config["plan_fingerprint"],
            )
        except (RuntimeError, ValueError, OSError, ArithmeticError) as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        self.store.add(record)
        self.log(
            f"  {spec['label']} {split} e{epochs} seed {seed}: "
            + (
                f"acc {record['final_accuracy']:.3f} forg {record['mean_forgetting']:.3f} ({record['wall_seconds']:.1f}s)"
                if record["status"] == "completed"
                else record["error"]
            )
        )
        return record

    def run_many(self, spec, split, epochs, seeds):
        return [self.run(spec, split, epochs, s) for s in seeds]


def aggregate(records):
    ok = [r for r in records if r["status"] == "completed"]
    out = {"n_runs": len(records), "n_failed": len(records) - len(ok)}
    if not ok or out["n_failed"]:
        out["eligible"] = False
        return out | ({"accuracy_mean": None} if not ok else {})
    col = lambda k: [r[k] for r in ok]
    out.update(
        eligible=True,
        seeds=[r["seed"] for r in ok],
        accuracy_mean=analysis_tool.mean(col("final_accuracy")),
        accuracy_std=analysis_tool.std(col("final_accuracy")),
        forgetting_mean=analysis_tool.mean(col("mean_forgetting")),
        forgetting_std=analysis_tool.std(col("mean_forgetting")),
        current_accuracy_mean=analysis_tool.mean(col("current_accuracy")),
        current_accuracy_std=analysis_tool.std(col("current_accuracy")),
        time_seconds_mean=analysis_tool.mean(col("wall_seconds")),
        time_seconds_std=analysis_tool.std(col("wall_seconds")),
        n_parameters=ok[0]["n_parameters"],
        replay_examples_final=(ok[0]["replay_examples_by_task"] or [None])[-1],
    )
    return out


def candidate_from(spec, val_runs, test_runs, epochs):
    """A candidate in the schema of analyze_per_degree_study, so the SAME selection code applies."""
    v, t = aggregate(val_runs), aggregate(test_runs)
    if not (v.get("eligible") and t.get("eligible")):
        return None
    return {
        "id": spec["label"],
        "strategy": spec["algorithm"],
        "sampling": spec["sampling"],
        "vector_label": spec["label"],
        "learning_rates_by_degree": [spec["learning_rate"]],
        "epochs": epochs,
        "batch_size": 128,
        "memory_per_class": 200 if spec.get("replay", True) else 0,
        "n_ref_per_class": None,
        "degree_schedule": None,
        "n_features_final": None,
        "selected_by_core": False,
        "spec": spec,
        "test": {k: t[k] for k in t if k != "eligible"},
        "validation": {
            "accuracy_mean": v["accuracy_mean"],
            "forgetting_mean": v["forgetting_mean"],
            "current_accuracy_mean": v["current_accuracy_mean"],
            "time_seconds_mean": v["time_seconds_mean"],
            "seeds": list(SELECTION_SEEDS),
        },
    }


# ------------------------------------------------------------------------ main stages


def polynomial_settings_from_analysis(path: Path):
    """Distinct 3-epoch polynomial settings named by the analysis (all bases and criteria)."""
    report = json.loads(path.read_text())
    wanted = {}
    for basis, block in report["analysis"].items():
        groups = [
            ("literal", block["winners"]),
            ("floor", block["accuracy_floor"]["winners"]),
        ]
        for group_name, winners in groups:
            for criterion, w in winners.items():
                key = (
                    w["strategy"],
                    w["sampling"],
                    tuple(w["learning_rates_by_degree"]),
                    w["epochs"],
                )
                wanted.setdefault(key, []).append(f"{basis}:{group_name}:{criterion}")
    # every cell's core-selected vector and its shared-rate reference (3 epochs), for the ablation table
    for c in report["candidates"]:
        if c["epochs"] == 3 and (
            c["selected_by_core"] or c["vector_label"] == "shared"
        ):
            key = (
                c["strategy"],
                c["sampling"],
                tuple(c["learning_rates_by_degree"]),
                c["epochs"],
            )
            reason = (
                "core_selected" if c["selected_by_core"] else "shared_rate_reference"
            )
            wanted.setdefault(key, []).append(
                f"{reason}:{c['strategy']}/{c['sampling']}"
            )
    return wanted


def paired_stats(a_runs, b_runs):
    """Paired differences a - b over shared seeds (n is small: read as descriptive)."""
    by_seed = {r["seed"]: r for r in b_runs if r["status"] == "completed"}
    pairs = [
        (a, by_seed[a["seed"]])
        for a in a_runs
        if a["status"] == "completed" and a["seed"] in by_seed
    ]
    out = {"n_pairs": len(pairs)}
    for name, key in (
        ("accuracy", "final_accuracy"),
        ("forgetting", "mean_forgetting"),
    ):
        diffs = [a[key] - b[key] for a, b in pairs]
        n = len(diffs)
        m = analysis_tool.mean(diffs)
        s = analysis_tool.std(diffs)
        t_crit = {
            6: 2.4469,
            5: 2.5706,
            4: 2.7764,
            3: 3.1824,
            2: 4.3027,
            8: 2.3060,
            9: 2.2622,
        }.get(n - 1)
        half = None if (t_crit is None or s is None) else t_crit * s / math.sqrt(n)
        out[name] = {
            "mean_difference": m,
            "std_difference": s,
            "ci95": None if half is None else [m - half, m + half],
            "a_better_in_seeds": sum(d > 0 for d in diffs)
            if name == "accuracy"
            else sum(d < 0 for d in diffs),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--core-results", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mnist")
    ap.add_argument(
        "--protocol", type=Path, default=ROOT / "configs" / "split_mnist_cil_v1.json"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "comparison")
    ap.add_argument(
        "--smoke", action="store_true", help="one seed, tiny grid: wiring check only"
    )
    args = ap.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    protocol = Protocol.from_file(args.protocol)
    dataset = load_split_mnist(protocol, args.data_dir)
    store = RunStore(
        args.out_dir / ("runs_smoke.jsonl" if args.smoke else "runs.jsonl")
    )
    study = Study(protocol, dataset, store, log)
    sel_seeds = SELECTION_SEEDS[:1] if args.smoke else SELECTION_SEEDS
    test_seeds = TEST_SEEDS[:1] if args.smoke else TEST_SEEDS
    er_grid = [(h, lr) for h in ER_HIDDEN for lr in ER_LRS]
    if args.smoke:
        er_grid = er_grid[:2]

    # ---- 1+2: ER-MLP grid, validation then test
    log("== ER-MLP grid")
    er_candidates = {}
    for hidden, lr in er_grid:
        spec = er_spec(hidden, lr)
        val = study.run_many(spec, "validation", protocol.epochs, sel_seeds)
        test = study.run_many(spec, "test", protocol.epochs, test_seeds)
        cand = candidate_from(spec, val, test, protocol.epochs)
        if cand is None:
            log(f"  {spec['label']}: ineligible (failed or diverged run)")
        else:
            er_candidates[spec["label"]] = cand

    # ---- 3: polynomial settings named by the analysis (test seeds)
    log("== polynomial settings from the analysis")
    wanted = polynomial_settings_from_analysis(args.analysis)
    poly_specs = []
    for (strategy, sampling, lr, epochs), reasons in wanted.items():
        spec = poly_spec("expanding", strategy, sampling, lr)
        poly_specs.append((spec, epochs, reasons))
    if args.smoke:
        poly_specs = [t for t in poly_specs if t[1] == protocol.epochs][:1]
    poly_runs = {}
    for spec, epochs, reasons in poly_specs:
        poly_runs[(spec["label"], epochs)] = {
            "spec": spec,
            "epochs": epochs,
            "reasons": reasons,
            "test_runs": study.run_many(spec, "test", epochs, test_seeds),
        }

    # ---- 4: polynomial ablations, shared lr chosen on validation
    log("== polynomial ablations (fixed_linear, fixed_capacity): validation selection")
    ablations = {}
    for condition in ("fixed_linear", "fixed_capacity"):
        grid = ABLATION_LRS[:1] if args.smoke else ABLATION_LRS
        scored = []
        for lr in grid:
            spec = poly_spec(condition, "ordinary", "class_balanced", lr)
            val = aggregate(
                study.run_many(spec, "validation", protocol.epochs, sel_seeds)
            )
            if val.get("eligible"):
                scored.append((-val["accuracy_mean"], val["forgetting_mean"], lr, spec))
        if not scored:
            continue
        _, _, lr, spec = min(scored, key=lambda s: (s[0], s[1]))
        ablations[condition] = {
            "spec": spec,
            "selected_on": "validation accuracy (ties: lower forgetting)",
            "test_runs": study.run_many(spec, "test", protocol.epochs, test_seeds),
        }
    # poly no-replay at the study's own selected rate is already part of the analysis settings when chosen

    # ---- selection for ER-MLP with the SAME criteria and code as for the polynomial study
    er_report, er_selected_specs = {}, []
    if er_candidates:
        pool = list(er_candidates.values())
        problems = []
        er_report = analysis_tool.analyse(pool, problems, accuracy_floor=0.02)
        for basis, block in er_report.values():
            for group in (block["winners"], block["accuracy_floor"]["winners"]):
                for w in group.values():
                    er_selected_specs.append(er_candidates[w["candidate_id"]]["spec"])
        # approved-grid-only selection, for transparency
        approved = [
            c
            for c in pool
            if c["spec"]["hidden"] in ([64], [128], [256])
            and c["spec"]["learning_rate"] <= 1.0
        ]
        if approved:
            er_report["approved_grid_only"] = analysis_tool.analyse(
                approved, problems, 0.02
            )["validation"]

    # ---- 5: 10-epoch re-runs (no re-selection) + the zero-replay ER ablation
    log("== 10-epoch re-runs and ER no-replay ablation")
    ten_epoch = {}
    if not args.smoke:
        unique = {s["label"]: s for s in er_selected_specs}
        for label, spec in unique.items():
            ten_epoch[label] = {
                "spec": spec,
                "runs": study.run_many(spec, "test", 10, test_seeds),
            }
        for (label, epochs), item in poly_runs.items():
            if epochs == 10:
                continue
        er_best = er_report.get("validation", {}).get("winners", {}).get("max_accuracy")
        if er_best:
            base = er_candidates[er_best["candidate_id"]]["spec"]
            nr = er_spec(tuple(base["hidden"]), base["learning_rate"], replay=False)
            ten_epoch["_er_no_replay_3ep"] = {
                "spec": nr,
                "runs": study.run_many(nr, "test", protocol.epochs, test_seeds),
            }

    # ---- one-time polynomial feature construction cost (excluded from per-run wall time)
    from polynomial_cl.continual import make_feature_data

    tick = time.perf_counter()
    make_feature_data(dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y)
    feature_seconds = time.perf_counter() - tick

    # ---- check re-runs against the core's stored results (bit-level reproducibility)
    stored = {}
    for path in sorted(args.core_results.glob("cell_*.json")):
        for cell in json.loads(path.read_text())["cells"]:
            for label, runs in cell["runs"].items():
                stored[
                    (cell["strategy"], cell["sampling"], tuple(cell["vectors"][label]))
                ] = {
                    r["seed"]: (r["final_mean_accuracy"], r["mean_forgetting"])
                    for r in runs
                }
    checks = []
    for (label, epochs), item in poly_runs.items():
        if epochs != protocol.epochs:
            continue
        spec = item["spec"]
        ref = stored.get(
            (spec["strategy"], spec["sampling"], tuple(spec["learning_rate"]))
        )
        if ref is None:
            continue
        diffs = [
            max(
                abs(r["final_accuracy"] - ref[r["seed"]][0]),
                abs(r["mean_forgetting"] - ref[r["seed"]][1]),
            )
            for r in item["test_runs"]
            if r["status"] == "completed" and r["seed"] in ref
        ]
        checks.append(
            {
                "label": label,
                "n_seeds": len(diffs),
                "max_abs_difference": max(diffs) if diffs else None,
            }
        )

    summary = {
        "schema_version": "1.0",
        "protocol": protocol.to_dict(),
        "protocol_fingerprint": protocol.fingerprint(),
        "selection_seeds": list(sel_seeds),
        "test_seeds": list(test_seeds),
        "er_grid": {
            "hidden": [list(h) for h in ER_HIDDEN],
            "learning_rates": list(ER_LRS),
            "note": "superset of configs/er_mlp_grid.json; research choice for this study",
        },
        "polynomial_feature_construction_seconds_one_time": feature_seconds,
        "er_analysis": er_report,
        "er_candidates": list(er_candidates.values()),
        "polynomial": [
            {
                "spec": v["spec"],
                "epochs": v["epochs"],
                "named_by": v["reasons"],
                "test": aggregate(v["test_runs"]),
                "test_runs": v["test_runs"],
            }
            for v in poly_runs.values()
        ],
        "polynomial_ablations": {
            c: {
                "spec": v["spec"],
                "selected_on": v["selected_on"],
                "test": aggregate(v["test_runs"]),
                "test_runs": v["test_runs"],
            }
            for c, v in ablations.items()
        },
        "ten_epoch_and_ablation_runs": {
            k: {"spec": v["spec"], "test": aggregate(v["runs"]), "test_runs": v["runs"]}
            for k, v in ten_epoch.items()
        },
        "reproduction_checks_against_core_results": checks,
    }
    out = args.out_dir / ("summary_smoke.json" if args.smoke else "summary.json")
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
