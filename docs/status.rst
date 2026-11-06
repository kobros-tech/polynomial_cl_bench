Implementation and verification status
======================================

This page separates *what exists* from *what has been shown to work*. A component is described
by the highest verification level it has reached, using these levels (each includes the
ones before it):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Level
     - Meaning
   * - Implemented
     - The functionality exists in the repository.
   * - Unit-tested
     - It has passing isolated tests.
   * - Integration-tested
     - Components have been tested together.
   * - Core-verified
     - The integration has been tested against the actual ``polynomial_cl`` package (never a
       stand-in), installed from the pinned commit.
   * - Real-data-verified
     - Executed against the actual MNIST archives.
   * - Reproducibility-verified
     - A result has been independently reproduced.
   * - Publication-ready
     - The evidence and analysis support the claims of the associated paper.

Nothing in this repository is currently *publication-ready*.

How these statements were obtained
----------------------------------

Everything below marked as verified was executed for this documentation, on 2026-11-05, in a
fresh virtual environment (Linux, Python 3.12.3, NumPy 2.5.4) created with
``pip install -e ".[dev]"``, so that the core was installed from the pin in ``pyproject.toml``
(core commit ``f90eac99a7a6793ec93168ab6381cf0586d82199``, the merge commit of the core's
``continual``/``data`` refactor on its ``main``).

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Command
     - Result
   * - ``BENCH_REQUIRE_CORE=1 BENCH_REQUIRE_PINNED_CORE=1 pytest -q``
       (single-threaded BLAS)
     - 293 passed, 3 skipped (the skipped tests are the opt-in real-data tests below)
   * - ``python -m polynomial_cl_bench.verify_data --data-dir <mnist dir>``
     - all four archives match the protocol's sha256 hashes
   * - ``BENCH_SLOW=1 pytest tests/test_full_split_mnist.py`` (real MNIST, about 50 s;
       run with a local checkout of the core at the same commit)
     - 3 passed
   * - ``python -m polynomial_cl_bench.runner --spec configs/experiment_polynomial_reference.json
       --data-dir <mnist dir> --out <dir> --reproduce-check`` (5 seeds)
     - 5 completed, 0 failed, integrity passed; reproducibility check for seed 0:
       exact match, maximum absolute difference 0.0; ``--resume`` into the same directory
       succeeded
   * - ``ruff check .`` and ``ruff format --check .``
     - clean

The reference run reported mean final accuracy 0.6751 (sample standard deviation 0.0038 over
5 seeds) and mean forgetting 0.0734, with a wall time of about 23 s per run on the machine used.
These numbers describe a *reference configuration that was not tuned* (see
:doc:`algorithms`); they are quoted here as evidence that the pipeline runs, not as a result.
The run is correctly reported as *not publishable* by the benchmark itself.

Status by component
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Component
     - Highest level reached
     - Evidence, and what is *not* verified
   * - Protocol v1.0 (``protocol.py``)
     - Real-data-verified
     - Validation and fingerprint are unit-tested; the protocol drives every real-data run
       above.
   * - Dataset verification and loading (``datasets.py``)
     - Real-data-verified
     - Archives hashed before and after loading; loaded shapes (60000, 16) and (10000, 16)
       checked on the real data. Tests also check that no download is attempted.
   * - Validation hold-out view
     - Core-verified
     - ``datasets.validation_view`` is tested against the core's ``holdout_split`` on synthetic data. Not exercised on real
       MNIST in this review.
   * - Shared experience plan (``plan.py``)
     - Real-data-verified
     - Built with the core's ``make_task_plan``; used by every real-data run above.
   * - Polynomial adapter
     - Real-data-verified
     - On full Split MNIST the adapter reproduces the core's stored records for two
       configurations (seeds 0 and 3) to within 1e-6, and agrees with the metrics the core
       itself reports to 12 decimal places. The stored records were produced by the core
       *before* its refactor, so this also shows that the refactored core reproduces them.
       The per-degree learning-rate configuration is only checked through these stored records.
   * - ER-MLP competitor
     - Core-verified
     - Unit-tested, and run through the real runner and worker process on synthetic data with
       the real core. **Not run on real MNIST** in this review.
   * - Batch sampling (``sampling.py``)
     - Core-verified
     - Compared draw for draw with the core's batcher and with batches captured inside a real
       ``fit_task``, on synthetic data.
   * - Metrics (``metrics.py``)
     - Core-verified
     - Recomputed values agree with the core's metrics in tests.
   * - Result records, aggregation, experiment validation
     - Real-data-verified
     - Unit-tested with fake adapters; exercised by the real-data reference run above
       (records written, validated, aggregated, summarised).
   * - Experiment runner (process isolation, resume, retry)
     - Real-data-verified
     - Fresh-process runs and ``--resume`` worked in the reference run. Failure, timeout and
       crash handling is unit-tested and tested on synthetic data, not on real data.
   * - Reproducibility check
     - Reproducibility-verified
     - ``--reproduce-check``: one seed of one algorithm re-run in a fresh process with single-threaded BLAS gave
       bit-identical results. This is a same-machine, same-environment repeat, not an
       independent reproduction on other hardware.
   * - Validation tuning (``tune.py``, ``tuning.py``)
     - Core-verified
     - Tested end to end on synthetic data with the real core. **Not run on real MNIST: no
       tuning result exists**, so no hyper-parameter has been selected for either algorithm.
   * - Model selection, four objectives (``selection.py``)
     - Unit-tested
     - Tested on constructed candidate pools, for equivalence with the earlier analysis
       script, and in an end-to-end ``tune()``-to-report test with prescribed adapters. Not run
       on tuning results from real data (none exist).
   * - Official-run provenance gate
     - Core-verified
     - In the clean environment above the core is reported as a remote-verified pip VCS install
       matching the pin, with no official-run problems. Refusal paths are unit-tested.
   * - Comparison-study tools (``tools/``)
     - Implemented
     - Command-line entry points load. They consume the core's per-degree study results, which
       are not part of this repository, and were not run in this review.
   * - Research paper (``paper/``)
     - Implemented (draft)
     - A LaTeX manuscript with generated tables. The raw results its numbers were generated
       from are not in the repository, so its numbers cannot currently be traced to benchmark
       output here. **Not validated; not publication-ready.** Migration to reStructuredText
       and validation of every numerical claim are separate pieces of work.
   * - Codabench / OpenML integrations
     - Not implemented
     - Planned.

Reference values
----------------

On full Split MNIST the benchmark's polynomial adapter reproduces these stored results of the
core exactly (the real-data tests compare to within 1e-6; the quickstart run gave the first
row to full printed precision):

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Configuration
     - Seed
     - Core record
   * - ``primary_reference``
     - 0
     - accuracy 0.6770214422462445, forgetting 0.0708057983859818
   * - ``experimental``
     - 3
     - accuracy 0.7734281931972611, forgetting 0.09086581818480258

These are consistency checks of the integration, not benchmark results.

What has not been run
---------------------

* Validation tuning on real MNIST, for either algorithm.
* The ER-MLP on real MNIST, through the runner or otherwise.
* The intended comparison, ``configs/experiment_tuned_comparison.json`` (it requires tuning
  results that do not exist; it also declares ``official`` mode).
* Model selection on real tuning results.
* Any reproduction on different hardware or a different BLAS.
* The analysis and paper-asset tools in ``tools/``, and the mutation-check scripts.
