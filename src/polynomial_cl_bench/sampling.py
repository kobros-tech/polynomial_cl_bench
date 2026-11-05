# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Mini-batch construction identical to the core's ``polynomial_cl.continual (fit_task, sampling)``.

The core exposes batching only inside ``fit_task`` (``_balanced_batches`` is private), so a
competitor that trains a different model cannot call it. This module reproduces the two
policies **draw for draw**: the same ``numpy`` Generator calls in the same order, hence the
same index batches for the same seed. ``tests/test_sampling_equivalence.py`` checks that
against the core (spying on the core's own batcher inside a real ``fit_task`` call, and
end-to-end through a linear model), so a change in the core's batching is detected.

Behaviour that matters and is therefore tested:

* ``class_balanced``: every class that has examples gets ``batch_size // n_classes`` slots;
  the ``batch_size % n_classes`` leftover slots go to classes chosen without replacement
  (``rng.choice``) afresh for every batch. Each class is read through its own shuffled cycle
  (reshuffled when exhausted); a class with fewer examples than its slots repeats and is never
  padded from another class. Every batch is full.
* ``shuffled``: a fresh permutation of all rows per epoch, cut into consecutive batches; the
  last batch of an epoch can be short.
* Both make ``ceil(rows / batch_size)`` updates per epoch, from one Generator seeded once per
  experience and shared by all epochs.
"""

from __future__ import annotations

import numpy as np

SAMPLING_POLICIES = ("shuffled", "class_balanced")


def updates_per_epoch(n_rows: int, batch_size: int) -> int:
    return -(-n_rows // batch_size)


def class_balanced_batches(labels, n_updates, batch_size, rng):
    """Yield ``n_updates`` full index batches with (near-)equal class representation."""
    classes = np.unique(labels)
    pools = {int(c): np.flatnonzero(labels == c) for c in classes}
    cycle, cursor = {}, {}
    for c, pool in pools.items():
        cycle[c], cursor[c] = pool[rng.permutation(len(pool))], 0
    base, extra = divmod(batch_size, len(classes))
    for _ in range(n_updates):
        bonus = np.zeros(len(classes), dtype=np.int64)
        bonus[rng.choice(len(classes), extra, replace=False)] = 1
        parts = []
        for c, more in zip(classes, bonus):
            c, need = int(c), base + int(more)
            while need:
                if cursor[c] == len(cycle[c]):
                    cycle[c] = pools[c][rng.permutation(len(pools[c]))]
                    cursor[c] = 0
                take = min(need, len(cycle[c]) - cursor[c])
                parts.append(cycle[c][cursor[c] : cursor[c] + take])
                cursor[c] += take
                need -= take
        yield np.concatenate(parts)


def training_batches(
    label_columns, *, epochs: int, batch_size: int, sampling: str, seed: int
):
    """All index batches of one experience, in training order (one Generator, all epochs)."""
    if sampling not in SAMPLING_POLICIES:
        raise ValueError(f"sampling must be one of {SAMPLING_POLICIES}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    rng = np.random.default_rng(seed)
    n_rows = len(label_columns)
    n_updates = updates_per_epoch(n_rows, batch_size)
    for _ in range(epochs):
        if sampling == "shuffled":
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, batch_size):
                yield order[start : start + batch_size]
        else:
            yield from class_balanced_batches(label_columns, n_updates, batch_size, rng)
