# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""One experience/replay plan, built once with the core's public ``make_task_plan``.

The plan fixes, for one seed, the (shuffled) training indices of every experience and the
per-class replay memory. Passing the *same* ``ExperiencePlan`` to every adapter guarantees
identical experiences and identical replay exemplars; nothing is reconstructed independently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
from polynomial_cl.continual import TASK_ORDER, make_task_plan

from .algorithms.base import AdapterError
from .datasets import SplitMNIST, array_fingerprint
from .protocol import Protocol


class PlanError(AdapterError):
    """A plan does not belong to the dataset, seed or protocol it is used with."""


class FrozenMapping(Mapping):
    """A read-only, picklable mapping (``MappingProxyType`` cannot be pickled).

    It owns a private copy of the items, so mutating the dict it was built from afterwards
    cannot change a plan whose fingerprint has already been recorded.
    """

    __slots__ = ("_items",)

    def __init__(self, items=()):
        object.__setattr__(self, "_items", dict(items))

    def __setattr__(self, name, value):
        raise AttributeError("FrozenMapping is read-only")

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self):
        return (FrozenMapping, (tuple(self._items.items()),))

    def __repr__(self) -> str:
        return f"FrozenMapping({self._items!r})"


@dataclass(frozen=True, eq=False)
class ExperiencePlan:
    seed: int
    memory_per_class: int
    task_order: tuple
    train_size: int
    label_fingerprint: str  # sha256 of the training labels the plan was cut from
    tasks: tuple  # per experience: training indices (already shuffled by the core)
    memory: (
        Mapping  # class id -> replay-memory indices (fixed per class, one random subset)
    )

    def __post_init__(self) -> None:
        # A frozen dataclass only forbids rebinding fields; a plain dict would still be
        # mutable. Replace it by a read-only copy so the fingerprint cannot go stale.
        object.__setattr__(self, "memory", FrozenMapping(self.memory))

    def replay_indices(self, experience: int) -> np.ndarray:
        """Replay rows available during ``experience``: the memories of all earlier classes,
        concatenated in class order exactly as the core does (empty for experience 0)."""
        old = [c for task in self.task_order[:experience] for c in task]
        if not old:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([self.memory[c] for c in sorted(old)])

    def fingerprint(self) -> str:
        """Identity of the plan: everything that decides which rows are trained and replayed.

        Besides the index arrays this covers the class id each memory array belongs to, the
        task order, the number of experiences and the training labels the plan was cut from,
        so two plans that assign replay exemplars to different classes cannot collide.
        """
        classes = sorted(self.memory)
        header = json.dumps(
            {
                "format": 2,
                "seed": int(self.seed),
                "memory_per_class": int(self.memory_per_class),
                "task_order": [[int(c) for c in task] for task in self.task_order],
                "train_size": int(self.train_size),
                "label_fingerprint": self.label_fingerprint,
                "n_tasks": len(self.tasks),
                "task_lengths": [len(t) for t in self.tasks],
                "memory_classes": [int(c) for c in classes],
                "memory_lengths": [len(self.memory[c]) for c in classes],
            },
            sort_keys=True,
        )
        arrays = array_fingerprint(*self.tasks, *(self.memory[c] for c in classes))
        return hashlib.sha256((header + arrays).encode()).hexdigest()


def make_plan(train_y, seed: int, memory_per_class: int) -> ExperiencePlan:
    """Build the plan with the core's ``make_task_plan`` (no re-implementation)."""
    labels = np.asarray(train_y)
    tasks, memory = make_task_plan(labels, seed, memory_per_class)
    for array in (*tasks, *memory.values()):
        array.setflags(write=False)
    return ExperiencePlan(
        seed=seed,
        memory_per_class=memory_per_class,
        task_order=tuple(tuple(t) for t in TASK_ORDER),
        train_size=len(labels),
        label_fingerprint=array_fingerprint(labels),
        tasks=tuple(tasks),
        memory=FrozenMapping(memory),
    )


def check_inputs(dataset: SplitMNIST, protocol: Protocol) -> None:
    """Refusals shared by every adapter: the protocol and dataset must be what the core fixes."""
    if tuple(tuple(t) for t in protocol.task_order) != tuple(TASK_ORDER):
        raise AdapterError(
            f"the core's task order is fixed to {TASK_ORDER}; "
            f"the protocol asks for {protocol.task_order}"
        )
    if dataset.projection_seed != protocol.projection_seed:
        raise AdapterError(
            f"dataset was projected with seed {dataset.projection_seed}; "
            f"the protocol requires {protocol.projection_seed}"
        )
    if dataset.source_hashes != protocol.dataset["files"]:
        raise AdapterError("dataset was not built from the protocol's verified files")


def resolve_plan(
    dataset: SplitMNIST, protocol: Protocol, seed: int, plan: ExperiencePlan | None
) -> ExperiencePlan:
    """Return ``plan`` after checking it belongs to this run, or build the identical one."""
    if plan is None:
        return make_plan(dataset.train_y, seed, protocol.memory_per_class)
    problems = []
    if plan.seed != seed:
        problems.append(f"plan seed {plan.seed} != run seed {seed}")
    if plan.memory_per_class != protocol.memory_per_class:
        problems.append(
            f"plan memory_per_class {plan.memory_per_class} != {protocol.memory_per_class}"
        )
    if plan.task_order != tuple(tuple(t) for t in protocol.task_order):
        problems.append("plan task order differs from the protocol")
    if plan.train_size != len(
        dataset.train_y
    ) or plan.label_fingerprint != array_fingerprint(np.asarray(dataset.train_y)):
        problems.append(
            "plan was cut from different training labels than this dataset's"
        )
    if problems:
        raise PlanError("plan does not match this run: " + "; ".join(problems))
    return plan
