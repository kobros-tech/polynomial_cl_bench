# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Benchmark protocol: an explicit, validated, algorithm-independent configuration.

The protocol owns everything that must be identical for every algorithm: data, class
order, seeds, shared training budget, memory budget, sampling policy and the evaluation rule.
Algorithm-specific settings (learning rate, strategy, ...) live in the adapter's own config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SAMPLING_POLICIES = ("shuffled", "class_balanced")
EVALUATION_SCHEDULES = ("after_each_experience",)
N_CLASSES = 10


class ProtocolError(ValueError):
    """The protocol configuration is invalid."""


@dataclass(frozen=True)
class Protocol:
    name: str
    version: str
    dataset: dict
    projection_seed: int
    n_components: int
    task_order: tuple  # tuple of tuples of class ids
    seeds: tuple
    epochs: int
    batch_size: int
    sampling: str
    memory_per_class: int
    evaluation: dict
    validation: dict
    projection_description: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Protocol:
        try:
            training, memory = raw["training"], raw["memory"]
            protocol = cls(
                name=raw["name"],
                version=raw["version"],
                dataset=raw["dataset"],
                projection_seed=raw["projection"]["seed"],
                n_components=raw["projection"]["n_components"],
                projection_description=raw["projection"].get("description", ""),
                task_order=tuple(tuple(t) for t in raw["task_order"]),
                seeds=tuple(raw["seeds"]),
                epochs=training["epochs"],
                batch_size=training["batch_size"],
                sampling=training["sampling"],
                memory_per_class=memory["examples_per_class"],
                evaluation=raw["evaluation"],
                validation=raw["validation"],
            )
        except KeyError as error:
            raise ProtocolError(f"missing protocol field: {error}") from error
        protocol.validate()
        return protocol

    @classmethod
    def from_file(cls, path) -> Protocol:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        flat = [c for task in self.task_order for c in task]
        if not self.task_order or any(len(t) == 0 for t in self.task_order):
            raise ProtocolError("task_order must contain non-empty experiences")
        if sorted(flat) != list(range(N_CLASSES)):
            raise ProtocolError(
                f"task_order must introduce each class 0..{N_CLASSES - 1} exactly once"
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ProtocolError("seeds must be a non-empty list of distinct integers")
        if not all(isinstance(s, int) and not isinstance(s, bool) for s in self.seeds):
            raise ProtocolError("seeds must be integers")
        if self.sampling not in SAMPLING_POLICIES:
            raise ProtocolError(f"sampling must be one of {SAMPLING_POLICIES}")
        for field in ("epochs", "batch_size", "memory_per_class", "n_components"):
            value = getattr(self, field)
            if not isinstance(value, int) or value < 1:
                raise ProtocolError(f"{field} must be a positive integer")
        if not isinstance(self.projection_seed, int):
            raise ProtocolError("projection_seed must be an integer")
        if self.evaluation.get("schedule") not in EVALUATION_SCHEDULES:
            raise ProtocolError(
                f"evaluation.schedule must be one of {EVALUATION_SCHEDULES}"
            )
        files = self.dataset.get("files")
        if not files or not all(
            isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef")
            for h in files.values()
        ):
            raise ProtocolError(
                "dataset.files must map file names to sha256 hex digests"
            )
        if not 0 < self.validation.get("holdout_fraction", 0) < 1:
            raise ProtocolError("validation.holdout_fraction must be in (0, 1)")

    @property
    def n_experiences(self) -> int:
        return len(self.task_order)

    def to_dict(self) -> dict:
        """Canonical JSON-able form (what is stored in result records)."""
        return {
            "name": self.name,
            "version": self.version,
            "dataset": self.dataset,
            "projection": {
                "seed": self.projection_seed,
                "n_components": self.n_components,
                "description": self.projection_description,
            },
            "task_order": [list(t) for t in self.task_order],
            "seeds": list(self.seeds),
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "sampling": self.sampling,
            },
            "memory": {
                "policy": "fixed_per_class_random_subset",
                "examples_per_class": self.memory_per_class,
            },
            "evaluation": self.evaluation,
            "validation": self.validation,
        }

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON; changes whenever any protocol field changes."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()
