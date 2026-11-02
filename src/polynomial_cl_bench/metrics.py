"""Benchmark-owned metrics computed from an accuracy matrix.

``R[i, j]`` is the accuracy on the test examples of experience ``j`` after training on
experience ``i``, predicting over the classes introduced through experience ``i`` only.
Entries with ``j > i`` are NaN (experience not yet seen); entries with ``j <= i`` are finite.

These functions are deliberately independent of ``polynomial_cl`` so that they can be
checked against the core's own values.
"""

from __future__ import annotations

import numpy as np


class MatrixError(ValueError):
    """The accuracy matrix is malformed."""


def validate_matrix(matrix) -> np.ndarray:
    """Return ``matrix`` as a float array after checking its shape and NaN pattern."""
    r = np.asarray(matrix, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] < 1:
        raise MatrixError(f"accuracy matrix must be square, got shape {r.shape}")
    lower = np.tril(np.ones(r.shape, bool))
    if not np.all(np.isfinite(r[lower])):
        raise MatrixError("entries with j <= i must be finite")
    if not np.all(np.isnan(r[~lower])):
        raise MatrixError("entries with j > i must be NaN")
    if np.any(r[lower] < 0) or np.any(r[lower] > 1):
        raise MatrixError("accuracies must lie in [0, 1]")
    return r


def final_mean_accuracy(matrix) -> float:
    """Mean accuracy over all experiences after the last one."""
    return float(np.mean(validate_matrix(matrix)[-1]))


def accuracy_per_experience(matrix) -> list:
    """Final-row accuracy of each experience."""
    return [float(v) for v in validate_matrix(matrix)[-1]]


def current_accuracy(matrix) -> list:
    """Accuracy on experience i right after training on it (the diagonal)."""
    return [float(v) for v in np.diag(validate_matrix(matrix))]


def mean_accuracy_by_step(matrix) -> list:
    """After each experience, mean accuracy over the experiences seen so far."""
    r = validate_matrix(matrix)
    return [float(np.mean(r[i, : i + 1])) for i in range(len(r))]


def forgetting_by_experience(matrix) -> list:
    """Best accuracy experience j ever had (rows j..T-1) minus its final accuracy.

    Same definition as the core experiment; the last experience has no later evaluation, so
    its entry is always 0.
    """
    r = validate_matrix(matrix)
    last = len(r) - 1
    return [float(np.max(r[j : last + 1, j]) - r[last, j]) for j in range(len(r))]


def mean_forgetting(matrix) -> float:
    """Mean of ``forgetting_by_experience`` over experiences 0..T-2 (final one excluded)."""
    r = validate_matrix(matrix)
    if len(r) < 2:
        raise MatrixError("forgetting needs at least two experiences")
    return float(np.mean(forgetting_by_experience(r)[:-1]))
