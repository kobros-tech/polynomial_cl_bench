# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""The cells of the core's per-degree study on Split MNIST, as a job matrix.

The study is a set of independent runs of the core's experiment script
``experiments/per_degree_lr.py``: one sampling comparison, one per-degree selection per
(strategy, sampling) cell, and one 10-epoch re-run per strategy of the cells' shared and
selected rate vectors. Every run is its own GitHub job, so the whole study takes as long as its
slowest cell instead of the sum of all cells.

Parameters are the script's own defaults (selection seeds 0-2, test seeds 3-9, three epochs per
experience), so nothing but the cell is passed; only the 10-epoch stage sets ``--epochs 10``.
The file names are the ones ``tools/analyze_per_degree_study.py`` globs
(``cell_*.json``, ``epochs10_*.json``) and each run's log is named after its output, because
the analysis reads wall time from ``logs/<name>.log`` (10 epochs when it starts with
``epochs10_``).

    python tools/core_study_plan.py matrix --stage study    # sampling + the 12 cells
    python tools/core_study_plan.py matrix --stage epochs   # needs the cells' results
"""

from __future__ import annotations

import argparse
import json

DATASET = "split_mnist"
STRATEGIES = (
    "ordinary",
    "ordinary_no_replay",
    "small_old_lr",
    "frozen_old",
    "strategy_b",
    "strategy_b_new",
)
SAMPLINGS = ("class_balanced", "shuffled")
SAMPLING_SEEDS = (0, 1, 2, 3, 4)
EPOCHS_ROBUSTNESS = 10
STAGES = ("study", "epochs")


def _entry(name: str, output: str, args: list[str]) -> dict[str, str]:
    return {
        "name": name,
        "output": output,
        "args": " ".join(["--dataset", DATASET, *args]),
    }


def study_entries() -> list[dict[str, str]]:
    """Sampling comparison and the per-degree selection of every cell; all independent."""
    seeds = [str(s) for s in SAMPLING_SEEDS]
    entries = [
        _entry(
            "sampling",
            "sampling_e3_lr0.1.json",
            ["--experiment", "sampling", "--seeds", *seeds],
        )
    ]
    for strategy in STRATEGIES:
        for sampling in SAMPLINGS:
            name = f"cell_{strategy}_{sampling}"
            entries.append(
                _entry(
                    name,
                    f"{name}.json",
                    [
                        "--experiment",
                        "per-degree",
                        "--strategies",
                        strategy,
                        "--samplings",
                        sampling,
                    ],
                )
            )
    return entries


def epochs_entries() -> list[dict[str, str]]:
    """10-epoch re-runs of the stored vectors; each reads every finished cell."""
    return [
        _entry(
            f"epochs10_{strategy}",
            f"epochs10_{strategy}.json",
            [
                "--experiment",
                "epochs",
                "--epochs",
                str(EPOCHS_ROBUSTNESS),
                "--strategies",
                strategy,
            ],
        )
        for strategy in STRATEGIES
    ]


def matrix(stage: str) -> dict[str, list[dict[str, str]]]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return {"include": study_entries() if stage == "study" else epochs_entries()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("matrix",))
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(matrix(args.stage), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
