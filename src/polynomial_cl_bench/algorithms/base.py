# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The adapter contract shared by every algorithm.

``run_sequence`` executes one complete benchmark sequence (all experiences, evaluated after
each) for one seed. Algorithms that expose an incremental train/predict lifecycle can build the
matrix themselves; the polynomial core only exposes a complete run, so its adapter wraps that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol as TypingProtocol

import numpy as np

from ..datasets import SplitMNIST
from ..protocol import Protocol


class AdapterError(RuntimeError):
    """The adapter cannot honour the protocol."""


@dataclass
class SequenceResult:
    """What an adapter returns for one (protocol, seed) run."""

    accuracy_matrix: np.ndarray  # (T, T), NaN where j > i; see metrics.py
    algorithm: dict  # name, configuration, provenance of the implementation
    resources: dict  # unavailable measurements are None, never 0
    sampling: str  # sampling policy actually used
    run_config: dict  # every value that determined this run (see PolynomialAdapter)
    replay_examples_used_by_task: list | None = None
    extras: dict = field(
        default_factory=dict
    )  # algorithm-reported values for cross-checks


class SequenceAdapter(TypingProtocol):
    name: str

    def describe(self) -> dict:
        """Algorithm name and full configuration, for the result record."""

    def run_sequence(
        self, dataset: SplitMNIST, protocol: Protocol, seed: int, plan=None
    ) -> SequenceResult:
        """Run every experience in order for ``seed`` and return the accuracy matrix.

        ``plan`` is an optional shared ``ExperiencePlan`` (see ``plan.py``); passing the same
        plan to every adapter guarantees identical experiences and replay exemplars. When
        omitted the adapter builds the identical plan itself.
        """
