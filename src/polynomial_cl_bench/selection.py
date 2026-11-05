# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Model selection under four objectives, with optional accuracy floors.

This module is deliberately independent of the core package and of every algorithm: it ranks
:class:`Candidate` values (mean accuracy, mean forgetting, time) and nothing else. The caller
decides which data the numbers come from. **Selection must use validation results only**;
:func:`report_from_tuning` refuses a tuning result that was not evaluated on the validation split.

Objectives (tie-breaks apply in the order listed; remaining ties go to the earlier candidate):

``max_accuracy``    highest accuracy, then lower forgetting, then lower time.
``min_forgetting``  lowest forgetting, then higher accuracy, then lower time.
``min_time``        smallest time, then higher accuracy, then lower forgetting.
``best_tradeoff``   smallest weighted Euclidean distance to the ideal point after min-max
                    normalisation of accuracy (up), forgetting (down) and time (down) over the
                    candidate pool. This is a *convention*, not a measurement; the weights are
                    reported and :func:`select_all` lists the winner under other weightings.

Accuracy floor (for ``min_forgetting``, ``min_time``, ``best_tradeoff``): only candidates whose
accuracy is within ``accuracy_floor`` (absolute) of the best accuracy in the pool take part. Low
forgetting or low time is not a virtue by itself (a model that learns nothing forgets nothing),
so the floor is the way to ask for "cheap/stable *and* still accurate". ``max_accuracy`` has no
floor (it would be meaningless).

Missing values are never treated as zero. A candidate without accuracy or forgetting (a failed
run) is excluded from every objective; a candidate without time is excluded from the objectives
that use time (``min_time``; ``best_tradeoff`` when the time weight is positive). Every exclusion
is recorded with its reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SELECTION_VERSION = "1.0"
OBJECTIVES = ("max_accuracy", "min_forgetting", "min_time", "best_tradeoff")
FLOOR_OBJECTIVES = ("min_forgetting", "min_time", "best_tradeoff")
DEFAULT_ACCURACY_FLOOR = 0.02
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)  # accuracy, forgetting, time
SENSITIVITY_WEIGHTS = {
    "equal_weights": (1.0, 1.0, 1.0),
    "accuracy_x2": (2.0, 1.0, 1.0),
    "forgetting_x2": (1.0, 2.0, 1.0),
    "time_x2": (1.0, 1.0, 2.0),
    "time_ignored": (1.0, 1.0, 0.0),
}
TIME_BASES = ("wall", "cpu")
_FLOOR_EPS = 1e-12  # guards float rounding at the floor boundary


class SelectionError(ValueError):
    """The selection request is invalid, or nothing can be selected."""


@dataclass(frozen=True)
class Candidate:
    """One configuration summarised by its mean validation results."""

    id: object
    accuracy: float | None
    forgetting: float | None
    time_seconds: float | None = None
    current_accuracy: float | None = (
        None  # newest-experience accuracy (diagnostic only)
    )
    detail: Mapping = field(default_factory=dict, compare=False)


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def rule_spec(accuracy_floor, weights) -> dict:
    """Everything that determines the outcome, for the fingerprint stored with a report."""
    return {
        "selection_version": SELECTION_VERSION,
        "objectives": list(OBJECTIVES),
        "tie_breaks": {
            "max_accuracy": ["-accuracy", "forgetting", "time", "order"],
            "min_forgetting": ["forgetting", "-accuracy", "time", "order"],
            "min_time": ["time", "-accuracy", "forgetting", "order"],
            "best_tradeoff": ["tradeoff_distance", "-accuracy", "order"],
        },
        "tradeoff": "weighted euclidean distance to ideal after min-max normalisation",
        "accuracy_floor": accuracy_floor,
        "floor_definition": "accuracy >= best accuracy in the pool - floor (absolute)",
        "floor_objectives": list(FLOOR_OBJECTIVES),
        "weights_accuracy_forgetting_time": list(weights),
    }


def rule_fingerprint(accuracy_floor, weights) -> str:
    text = json.dumps(rule_spec(accuracy_floor, weights), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Selection:
    objective: str
    candidate_id: object
    accuracy: float
    forgetting: float
    time_seconds: float | None
    current_accuracy: float | None
    n_pool: int  # candidates that took part in the ranking
    accuracy_floor: float | None
    floor_value: float | None
    n_before_floor: int
    weights: tuple | None
    excluded: list  # [(candidate id, reason)]

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "candidate_id": self.candidate_id,
            "values": {
                "accuracy": self.accuracy,
                "forgetting": self.forgetting,
                "time_seconds": self.time_seconds,
                "current_accuracy": self.current_accuracy,
            },
            "n_pool": self.n_pool,
            "accuracy_floor": self.accuracy_floor,
            "floor_value": self.floor_value,
            "n_before_floor": self.n_before_floor,
            "weights_accuracy_forgetting_time": None
            if self.weights is None
            else list(self.weights),
            "excluded": [{"candidate_id": i, "reason": r} for i, r in self.excluded],
        }


# --------------------------------------------------------------------------- primitives


def normalise(values, lower_is_better: bool):
    """Min-max scale to 0 (best) .. 1 (worst); a constant column scores 0 for everyone."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    scaled = [(v - lo) / span for v in values]  # 0 = smallest, 1 = largest
    return [s if lower_is_better else 1.0 - s for s in scaled]


def tradeoff_scores(pool: Sequence[Candidate], weights=DEFAULT_WEIGHTS):
    """Weighted distance to the ideal point; needs a time value for every candidate."""
    badness = (
        normalise([c.accuracy for c in pool], lower_is_better=False),
        normalise([c.forgetting for c in pool], lower_is_better=True),
        normalise([c.time_seconds for c in pool], lower_is_better=True),
    )
    return [
        math.sqrt(sum(w * badness[k][i] ** 2 for k, w in enumerate(weights)))
        for i in range(len(pool))
    ]


def dominates(a: Candidate, b: Candidate) -> bool:
    """``a`` dominates ``b`` on (accuracy up, forgetting down, time down)."""
    ge = (
        a.accuracy >= b.accuracy
        and a.forgetting <= b.forgetting
        and a.time_seconds <= b.time_seconds
    )
    gt = (
        a.accuracy > b.accuracy
        or a.forgetting < b.forgetting
        or a.time_seconds < b.time_seconds
    )
    return ge and gt


def pareto_front(pool: Sequence[Candidate]):
    """Candidates not dominated by any other (all three values required)."""
    return [c for c in pool if not any(dominates(o, c) for o in pool if o is not c)]


def _validate(candidates, objective, accuracy_floor, weights):
    if objective not in OBJECTIVES:
        raise SelectionError(f"unknown objective {objective!r}; expected {OBJECTIVES}")
    if accuracy_floor is not None:
        if objective not in FLOOR_OBJECTIVES:
            raise SelectionError(f"an accuracy floor does not apply to {objective}")
        if not _finite(accuracy_floor) or accuracy_floor < 0:
            raise SelectionError("accuracy_floor must be a finite number >= 0")
    if len(weights) != 3 or not all(_finite(w) and w >= 0 for w in weights):
        raise SelectionError("weights must be three finite numbers >= 0")
    if not any(w > 0 for w in weights):
        raise SelectionError("at least one weight must be positive")
    ids = [c.id for c in candidates]
    if len(set(map(repr, ids))) != len(ids):
        raise SelectionError("candidate ids must be unique")


def select(
    candidates: Sequence[Candidate],
    objective: str,
    *,
    accuracy_floor: float | None = None,
    weights=DEFAULT_WEIGHTS,
) -> Selection:
    """Pick one candidate under ``objective``; see the module docstring for the rules."""
    candidates = list(candidates)
    weights = tuple(float(w) for w in weights)
    _validate(candidates, objective, accuracy_floor, weights)
    uses_time = objective == "min_time" or (
        objective == "best_tradeoff" and weights[2] > 0
    )
    pool, excluded = [], []
    for c in candidates:
        if not (_finite(c.accuracy) and _finite(c.forgetting)):
            excluded.append((c.id, "no accuracy/forgetting (failed or incomplete)"))
        elif uses_time and not _finite(c.time_seconds):
            excluded.append((c.id, "no time measurement"))
        else:
            pool.append(c)
    if not pool:
        raise SelectionError(
            f"no candidate can be ranked for {objective}: "
            + ("; ".join(f"{i}: {r}" for i, r in excluded) or "empty candidate list")
        )
    n_before = len(pool)
    floor_value = None
    if accuracy_floor is not None:
        floor_value = max(c.accuracy for c in pool) - accuracy_floor
        pool = [c for c in pool if c.accuracy >= floor_value - _FLOOR_EPS]

    t = lambda c: c.time_seconds if _finite(c.time_seconds) else math.inf
    if objective == "max_accuracy":
        best = min(pool, key=lambda c: (-c.accuracy, c.forgetting, t(c)))
    elif objective == "min_forgetting":
        best = min(pool, key=lambda c: (c.forgetting, -c.accuracy, t(c)))
    elif objective == "min_time":
        best = min(pool, key=lambda c: (t(c), -c.accuracy, c.forgetting))
    else:
        if weights[2] > 0:
            scores = tradeoff_scores(pool, weights)
        else:  # time ignored: it need not exist, and must not influence the result
            scores = tradeoff_scores_no_time(pool, weights)
        best = pool[min(range(len(pool)), key=lambda i: (scores[i], -pool[i].accuracy))]
    return Selection(
        objective=objective,
        candidate_id=best.id,
        accuracy=best.accuracy,
        forgetting=best.forgetting,
        time_seconds=best.time_seconds if _finite(best.time_seconds) else None,
        current_accuracy=best.current_accuracy,
        n_pool=len(pool),
        accuracy_floor=accuracy_floor,
        floor_value=floor_value,
        n_before_floor=n_before,
        weights=weights if objective == "best_tradeoff" else None,
        excluded=excluded,
    )


def tradeoff_scores_no_time(pool, weights):
    acc = normalise([c.accuracy for c in pool], lower_is_better=False)
    forg = normalise([c.forgetting for c in pool], lower_is_better=True)
    return [
        math.sqrt(weights[0] * acc[i] ** 2 + weights[1] * forg[i] ** 2)
        for i in range(len(pool))
    ]


def select_all(
    candidates: Sequence[Candidate],
    *,
    accuracy_floor: float | None = DEFAULT_ACCURACY_FLOOR,
    weights=DEFAULT_WEIGHTS,
) -> dict:
    """All four objectives, the floor-constrained variants, sensitivity and Pareto front.

    An objective that cannot be ranked (for example, no candidate has a time) is reported as
    ``{"error": ...}``; it never silently disappears and never raises for the others.
    """
    candidates = list(candidates)

    def attempt(objective, floor):
        try:
            return select(
                candidates, objective, accuracy_floor=floor, weights=weights
            ).to_dict()
        except SelectionError as error:
            return {"objective": objective, "error": str(error)}

    out = {
        "selection_version": SELECTION_VERSION,
        "rule_fingerprint": rule_fingerprint(accuracy_floor, weights),
        "n_candidates": len(candidates),
        "winners": {o: attempt(o, None) for o in OBJECTIVES},
        "floor_winners": {},
        "tradeoff_sensitivity": {},
        "pareto_front_ids": [],
    }
    if accuracy_floor is not None:
        out["floor_winners"] = {o: attempt(o, accuracy_floor) for o in FLOOR_OBJECTIVES}
    for name, w in SENSITIVITY_WEIGHTS.items():
        try:
            chosen = select(candidates, "best_tradeoff", weights=w).candidate_id
        except SelectionError as error:
            chosen = {"error": str(error)}
        out["tradeoff_sensitivity"][name] = {
            "weights_accuracy_forgetting_time": list(w),
            "candidate_id": chosen,
        }
    complete = [
        c
        for c in candidates
        if _finite(c.accuracy) and _finite(c.forgetting) and _finite(c.time_seconds)
    ]
    out["pareto_front_ids"] = [c.id for c in pareto_front(complete)]
    return out


# --------------------------------------------------------------------------- tuning results


def _mean(values):
    return (
        None
        if any(v is None for v in values) or not values
        else sum(values) / len(values)
    )


def candidates_from_tuning(
    tuning: Mapping, time_basis: str = "wall"
) -> list[Candidate]:
    """Candidates from a tuning result (``TuningResult.to_dict()`` / its JSON file).

    Accuracy and forgetting are the configuration's means over the selection seeds. A
    configuration is only complete if every selection-seed run completed; otherwise its
    values are ``None`` and it is excluded with a reason. Time is the mean per-run time on
    ``time_basis`` (``wall``: includes in-loop evaluation, ``cpu``: process CPU time), or
    ``None`` when any run lacks it (tuning files written before time was recorded).
    """
    if time_basis not in TIME_BASES:
        raise SelectionError(f"time_basis must be one of {TIME_BASES}")
    key = f"{time_basis}_time_seconds"
    out = []
    for cfg in tuning["configs"]:
        runs = [r for r in tuning["runs"] if r["config_index"] == cfg["config_index"]]
        complete = cfg["eligible"]
        out.append(
            Candidate(
                id=cfg["config_index"],
                accuracy=cfg["mean_final_accuracy"] if complete else None,
                forgetting=cfg["mean_forgetting"] if complete else None,
                time_seconds=_mean([r.get(key) for r in runs]) if complete else None,
                detail={"config": cfg["config"], "statuses": cfg["statuses"]},
            )
        )
    return out


def report_from_tuning(
    tuning: Mapping,
    *,
    accuracy_floor: float | None = DEFAULT_ACCURACY_FLOOR,
    time_basis: str = "wall",
    weights=DEFAULT_WEIGHTS,
) -> dict:
    """Selections for all four objectives from a validation-only tuning result."""
    split = tuning.get("validation_dataset", {}).get("evaluation_split")
    if split != "validation":
        raise SelectionError(
            f"tuning result was evaluated on {split!r}, not 'validation'; "
            "model selection must never use test results"
        )
    candidates = candidates_from_tuning(tuning, time_basis)
    report = select_all(candidates, accuracy_floor=accuracy_floor, weights=weights)
    report.update(
        {
            "algorithm": tuning["algorithm"],
            "ranking_basis": "validation",
            "time_basis": time_basis,
            "accuracy_floor": accuracy_floor,
            "protocol_fingerprint": tuning["protocol_fingerprint"],
            "selection_seeds": tuning["selection_seeds"],
            "configs": {c.id: c.detail["config"] for c in candidates},
        }
    )
    return report


def render_markdown(report: Mapping) -> str:
    def row(label, w):
        if "error" in w:
            return f"| {label} | - | - | - | - | {w['error']} |"
        v = w["values"]
        t = "n/a" if v["time_seconds"] is None else f"{v['time_seconds']:.2f}"
        cfg = json.dumps(report["configs"].get(w["candidate_id"]), sort_keys=True)
        return (
            f"| {label} | {w['candidate_id']} | {100 * v['accuracy']:.2f} | "
            f"{100 * v['forgetting']:.2f} | {t} | {cfg} |"
        )

    lines = [
        f"# Model selection: {report['algorithm']}",
        "",
        (
            f"Ranking basis: **{report['ranking_basis']}** (never test). Time basis: "
            f"`{report['time_basis']}`. Rule fingerprint "
            f"`{report['rule_fingerprint'][:12]}`."
        ),
        "",
        "| Objective | Config | Acc. (%) | Forg. (%) | Time (s) | Configuration / note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [row(o, report["winners"][o]) for o in OBJECTIVES]
    if report["floor_winners"]:
        lines += [
            "",
            f"With accuracy floor {report['accuracy_floor']} (absolute, below the best):",
            "",
            "| Objective | Config | Acc. (%) | Forg. (%) | Time (s) | Configuration / note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            row(f"{o} (floor)", report["floor_winners"][o]) for o in FLOOR_OBJECTIVES
        ]
    lines += [
        "",
        f"Pareto front (accuracy, forgetting, time): {report['pareto_front_ids']}",
    ]
    lines += ["", "Best trade-off under other weightings:"]
    lines += [
        f"- {n}: {s['candidate_id']}" for n, s in report["tradeoff_sensitivity"].items()
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Select configurations for all four objectives from a tuning result."
    )
    p.add_argument("--tuning", required=True, type=Path)
    p.add_argument("--accuracy-floor", type=float, default=DEFAULT_ACCURACY_FLOOR)
    p.add_argument("--no-floor", action="store_true")
    p.add_argument("--time-basis", choices=TIME_BASES, default="wall")
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", type=Path)
    args = p.parse_args(argv)
    for path in (args.out_json, args.out_md):
        if path is not None and path.exists():
            print(f"refusing to overwrite {path}", file=sys.stderr)
            return 2
    try:
        report = report_from_tuning(
            json.loads(args.tuning.read_text()),
            accuracy_floor=None if args.no_floor else args.accuracy_floor,
            time_basis=args.time_basis,
        )
    except SelectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report["source_tuning_file"] = args.tuning.name
    report["source_tuning_sha256"] = hashlib.sha256(
        args.tuning.read_bytes()
    ).hexdigest()
    args.out_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.out_md:
        args.out_md.write_text(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
