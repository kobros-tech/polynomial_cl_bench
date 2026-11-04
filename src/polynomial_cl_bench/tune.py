# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Run validation tuning for one algorithm from a grid file.

    python -m polynomial_cl_bench.tune --grid configs/er_mlp_grid.json --data-dir data/mnist \\
        --out results/tuning_er_mlp.json

* A grid file must have ``approval_status == "approved"``; a proposed grid is refused.
* Data: the archives are verified by hash, then only the validation view is used for tuning.
* Mode: ``development`` (default) or ``official``. ``official`` applies the unchanged
  provenance rules and refuses before any training if they are not met. A development tuning
  result records its provenance and ``official_eligible`` so it cannot be mistaken for an
  officially-provenanced one.
* The complete result (every run, failed ones included) is written as JSON, with the full
  validation table as Markdown next to it. An existing output file is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .algorithms.er_mlp import ERMLPAdapter
from .algorithms.polynomial import PolynomialAdapter
from .datasets import load_split_mnist
from .protocol import Protocol
from .provenance import (
    RUN_MODES,
    benchmark_provenance,
    core_provenance,
    enforce_run_mode,
)
from .tuning import TuningResult, tune
from .verify_data import DEFAULT_PROTOCOL

ER_FIXED = {"replay": True, "optimizer": "sgd", "momentum": 0, "weight_decay": 0}
ENTRY_KEYS = {
    "er_mlp": {"hidden", "learning_rate"},
    "polynomial_cl": {"learning_rate", "feature_power"},
}


class GridError(ValueError):
    """The grid file is malformed, unapproved, or inconsistent with the protocol."""


@dataclass(frozen=True)
class GridSpec:
    algorithm: str
    approval_status: str
    fixed: dict
    entries: list
    seeds: tuple
    expected_runs: int
    margin: float
    protocol_requirements: dict
    path: str
    sha256: str


def load_grid(path) -> GridSpec:
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    algorithm = raw.get("algorithm")
    if algorithm not in ENTRY_KEYS:
        raise GridError(
            f"unknown algorithm {algorithm!r}; expected one of {sorted(ENTRY_KEYS)}"
        )
    entries = raw["grid"]
    if not entries:
        raise GridError("the grid is empty")
    for i, entry in enumerate(entries):
        if set(entry) != ENTRY_KEYS[algorithm]:
            raise GridError(
                f"grid entry {i} must have exactly the keys {sorted(ENTRY_KEYS[algorithm])}"
            )
    if len({json.dumps(e, sort_keys=True) for e in entries}) != len(entries):
        raise GridError("the grid contains duplicate configurations")
    fixed = raw.get("fixed", {})
    if algorithm == "er_mlp" and fixed != ER_FIXED:
        raise GridError(f"er_mlp fixed settings must be exactly {ER_FIXED}")
    selection = raw["selection"]
    return GridSpec(
        algorithm=algorithm,
        approval_status=raw.get("approval_status", "missing"),
        fixed=fixed,
        entries=entries,
        seeds=tuple(selection["seeds"]),
        expected_runs=int(selection["expected_runs"]),
        margin=float(selection["margin"]),
        protocol_requirements=raw.get("protocol_requirements", {}),
        path=str(path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def build_factory(spec: GridSpec, mode: str):
    """Grid entry -> adapter (role ``validation_candidate``)."""
    if spec.algorithm == "er_mlp":
        return lambda e: ERMLPAdapter(
            hidden=e["hidden"],
            learning_rate=e["learning_rate"],
            replay=spec.fixed["replay"],
            role="validation_candidate",
            hyperparameter_provenance={
                "selection": "validation_tuning_candidate",
                "grid": Path(spec.path).name,
                "grid_sha256": spec.sha256,
            },
            mode=mode,
        )
    return lambda e: PolynomialAdapter(
        condition=spec.fixed["condition"],
        strategy=spec.fixed["strategy"],
        learning_rate=e["learning_rate"],
        feature_power=e["feature_power"],
        role="validation_candidate",
        hyperparameter_provenance={
            "selection": "validation_tuning_candidate",
            "grid": Path(spec.path).name,
            "grid_sha256": spec.sha256,
        },
        mode=mode,
    )


def run_tuning(
    spec: GridSpec, protocol: Protocol, dataset, mode: str = "development"
) -> TuningResult:
    """Validate everything that can be checked up front, then tune."""
    if spec.approval_status != "approved":
        raise GridError(
            f"{Path(spec.path).name} has approval_status {spec.approval_status!r}; "
            "only an approved grid may be run"
        )
    if mode not in RUN_MODES:
        raise GridError(f"mode must be one of {RUN_MODES}")
    requirements = {
        "memory_per_class": protocol.memory_per_class,
        "sampling": protocol.sampling,
    }
    for key, wanted in spec.protocol_requirements.items():
        if requirements.get(key) != wanted:
            raise GridError(
                f"the grid requires {key}={wanted!r}; the protocol has {requirements.get(key)!r}"
            )
    core, bench = core_provenance(), benchmark_provenance()
    enforce_run_mode(mode, core, bench)  # official: refuse now, before any training
    context = {
        "run_mode": mode,
        "official_eligible": not core["official_problems"]
        and not bench["official_problems"],
        "core": core,
        "benchmark": bench,
        "grid_file": Path(spec.path).name,
        "grid_sha256": spec.sha256,
        "grid_approval_status": spec.approval_status,
    }
    return tune(
        spec.algorithm,
        build_factory(spec, mode),
        spec.entries,
        dataset,
        protocol,
        selection_seeds=spec.seeds,
        expected_runs=spec.expected_runs,
        margin=spec.margin,
        context=context,
    )


def _label(algorithm: str, config: dict) -> str:
    if algorithm == "er_mlp":
        return f"H={config['hidden'][0]}, lr={config['learning_rate']:g}"
    return f"lr={config['learning_rate']:g}, p={config['feature_power']:g}"


def validation_table(result: TuningResult) -> str:
    """The complete validation table as Markdown: every configuration, every run's status."""
    seeds = result.selection_seeds
    head = (
        ["#", "configuration"]
        + [f"acc seed {s}" for s in seeds]
        + ["mean acc", "mean forgetting", "status", "within 0.005 of max", "selected"]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in result.configs:
        runs = {
            r["seed"]: r for r in result.runs if r["config_index"] == c["config_index"]
        }
        accs = [
            f"{runs[s]['final_mean_accuracy']:.4f}"
            if runs[s]["final_mean_accuracy"] is not None
            else "—"
            for s in seeds
        ]
        status = ", ".join(sorted({r["status"] for r in runs.values()}))
        mean_acc = f"{c['mean_final_accuracy']:.4f}" if c["eligible"] else "—"
        forgetting = f"{c['mean_forgetting']:.4f}" if c["eligible"] else "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(c["config_index"]),
                    _label(result.algorithm, c["config"]),
                    *accs,
                    mean_acc,
                    forgetting,
                    status,
                    "yes" if c["within_margin_of_max"] else "no",
                    "**selected**" if c["selected"] else "",
                ]
            )
            + " |"
        )
    failed = [r for r in result.runs if r["status"] != "completed"]
    if failed:
        lines += ["", "Failed or divergent runs:"] + [
            f"- config {r['config_index']} ({_label(result.algorithm, r['config'])}), "
            f"seed {r['seed']}: {r['status']}: {r['error']}"
            for r in failed
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", required=True)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--mode", choices=RUN_MODES, default="development")
    parser.add_argument(
        "--out", required=True, help="JSON result path; never overwritten"
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    spec, protocol = load_grid(args.grid), Protocol.from_file(args.protocol)
    if spec.approval_status != "approved":  # fail before touching the data
        print(
            f"refused: {Path(args.grid).name} is {spec.approval_status!r}, not approved",
            file=sys.stderr,
        )
        return 2
    dataset = load_split_mnist(protocol, args.data_dir)  # verifies the archive hashes
    result = run_tuning(spec, protocol, dataset, args.mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_json(out)
    table = validation_table(result)
    out.with_suffix(".md").write_text(table + "\n")
    print(table)
    print(f"\nselected: {result.selected_config} (index {result.selected_index})")
    return 0 if result.selected_index is not None else 1


if __name__ == "__main__":
    sys.exit(main())
