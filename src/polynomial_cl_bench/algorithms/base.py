"""The adapter contract shared by every algorithm.

``run_sequence`` executes one complete benchmark sequence (all experiences, evaluated after
each) for one seed. Algorithms that expose an incremental train/predict lifecycle can build the
matrix themselves; the polynomial core only exposes a complete run, so its adapter wraps that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol as TypingProtocol

import numpy as np

from ..datasets import SplitMNIST
from ..protocol import Protocol


@dataclass
class SequenceResult:
    """What an adapter returns for one (protocol, seed) run."""

    accuracy_matrix: np.ndarray  # (T, T), NaN where j > i; see metrics.py
    algorithm: dict  # name, configuration, provenance of the implementation
    resources: dict  # unavailable measurements are None, never 0
    sampling: str  # sampling policy actually used
    run_config: dict  # every value that determined this run (see PolynomialAdapter)
    replay_examples_used_by_task: Optional[list] = None
    extras: dict = field(default_factory=dict)  # algorithm-reported values for cross-checks


class SequenceAdapter(TypingProtocol):
    name: str

    def describe(self) -> dict:
        """Algorithm name and full configuration, for the result record."""

    def run_sequence(
        self, dataset: SplitMNIST, protocol: Protocol, seed: int
    ) -> SequenceResult:
        """Run every experience in order for ``seed`` and return the accuracy matrix."""
