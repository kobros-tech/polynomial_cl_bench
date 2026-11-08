Running experiments
===================

An *experiment* is every algorithm x every protocol seed, producing one result record per run.
It is described by a small JSON specification and executed by
``python -m polynomial_cl_bench.runner``.

Workflow
--------

Every step is a separate command; nothing is re-run by a later step.

.. code-block:: console

   # 1. validation-only tuning, one per algorithm
   $ python -m polynomial_cl_bench.tune --grid configs/er_mlp_grid.json --out results/tuning_er_mlp.json
   $ python -m polynomial_cl_bench.tune --grid configs/polynomial_grid.json --out results/tuning_polynomial.json
   # 2. the experiment: every algorithm x every seed, one fresh process per run
   $ python -m polynomial_cl_bench.runner --spec configs/experiment_tuned_comparison.json \
         --data-dir data/mnist --out results/tuned_comparison --reproduce-check
   # 3. interrupted, or some runs failed? continue without mixing anything
   $ python -m polynomial_cl_bench.runner --spec ... --out results/tuned_comparison --resume [--retry-failed]
   # 4. validate and aggregate (also done at the end of step 2)
   $ python -m polynomial_cl_bench.summarize --dir results/tuned_comparison

Until tuning has been run on real MNIST, ``experiment_tuned_comparison.json`` cannot be used:
its tuning-result files do not exist. ``experiment_polynomial_reference.json`` is the only
experiment that currently runs (see :doc:`quickstart`).

Experiment specification
------------------------

.. code-block:: json

   {
     "name": "polynomial_reference",
     "protocol": "split_mnist_cil_v1.json",
     "seeds": null,
     "mode": "development",
     "evaluation_split": "test",
     "isolation": "process",
     "algorithms": [
       {"label": "polynomial_reference", "config": "polynomial_expanding_ordinary.json"}
     ]
   }

``seeds: null`` means the protocol's seeds. ``mode`` is ``development`` or ``official`` (see
:doc:`reproducibility`). ``isolation`` is ``process`` (each run in a fresh Python process; the
default for the command line) or ``in_process`` (for tests and quick checks). An optional
``timeout_seconds`` bounds a run. An algorithm entry may name a ``tuning_result``: its
hyper-parameters are then taken from that validation-only tuning file, after the runner checks
its protocol fingerprint and validation split, that it selected a configuration, and that its
seeds are disjoint from the experiment's. The tuning file's sha256 and selection details are
recorded in the run records. Labels and the experiment name must match
``[A-Za-z0-9][A-Za-z0-9_.-]*``.

Execution rules
---------------

**Failures are results.** An adapter exception, divergence, malformed matrix, an adapter that
did not use the shared plan, replay above the budget, a worker crash or a timeout each become a
record with status ``failed``, ``diverged``, ``crashed`` or ``timeout``. Aggregation reports
completed, failed and missing counts per algorithm and lists every failure; statistics use
completed runs only and state their ``n``. An unreadable or hand-edited record is reported as a
problem and counted as missing, not skipped. Only an official-mode provenance refusal or
Ctrl-C stops an experiment.

**Isolation and sequencing.** Process isolation prevents state leaking between runs, lets a
crash or timeout be recorded without taking the experiment down, and gives peak memory a clean
baseline. Runs are executed **sequentially on purpose**: concurrent runs would contaminate each
other's timings.

**Resume without mixing.** The output directory has a manifest with the fingerprint of the full
resolved experiment (protocol, every algorithm configuration, seeds, mode, split, isolation).
Resuming is refused unless the fingerprint matches; a completed record is reused only if its
run key and the core and benchmark commits match the current ones. A failed run is re-run only
with ``--retry-failed``, and the old record is kept as ``<name>.attempt<k>.json``.

Result records
--------------

One JSON file per run, schema version 1.0 (:mod:`polynomial_cl_bench.results`). A record holds:
the algorithm, its resolved configuration and hyper-parameter provenance; dataset and protocol
identity (fingerprints, not paths); the seed; the plan fingerprint; the training budget; the
memory policy with the budget and actual replay use per experience; the accuracy matrix and
every metric (all recomputed by this package from the matrix); resources; the environment
(Python, NumPy, platform, thread settings, **core and benchmark commits**); timestamps;
status; and error.

``null`` means *not measured*, never zero. A record is re-validated when written and when read:
its run key, every metric, the matrix shape and NaN pattern, and replay memory against its
budget. Files are written atomically, so a crash never leaves a half-written record. A
non-completed record has ``null`` metrics and an error object; it is kept, never dropped.

Resources
---------

Measured by the runner around ``run_sequence``, identically for every algorithm: wall time,
CPU time, and peak resident set size *during the run* (Linux: the kernel's peak counter is reset
just before the run; elsewhere ``null``). Peak memory is an absolute RSS and includes the loaded
dataset, so ``rss_before_run_mb`` is stored beside it; with process isolation the baseline is
the same for every algorithm. Training time, inference time, parameter count and (for the
polynomial adapter) ``feature_construction_seconds`` are adapter-reported. The polynomial
adapter reports training and inference time only when the installed core can split its run,
and ``null`` otherwise (see :doc:`algorithms`). Wall time includes in-loop evaluation and, for
the polynomial adapter, feature construction.

Statistics
----------

Per algorithm and metric: n, mean, sample standard deviation (ddof = 1), the 95% t-interval of
the mean, minimum and maximum. Between algorithms: paired per-seed differences (a - b), with
counts of the pairs where each side is higher.

These statistics are **descriptive only**. Seeds vary initialisation, sampling and replay
selection, *not* the data split. With few seeds the intervals are wide and assume roughly
normal differences. The benchmark computes no p-values and issues no "better" or "worse"
verdicts.

Experiment validation
---------------------

:func:`~polynomial_cl_bench.validation.validate_experiment` separates two kinds of finding,
both printed in ``summary.md``:

*Integrity problems* must be zero: a record contradicting the manifest, different plans for one
seed across algorithms, different dataset arrays, mixed commits, missing or unexpected runs, or a
disagreement with the core's own reported metrics.

*Publication blockers* are not errors, but the comparison is not publishable: not official
mode, unverified provenance, failed runs, a non-test split, seeds differing from the protocol,
hyper-parameters not from validation tuning, or a missing or inexact reproducibility check.
