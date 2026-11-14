Architecture
============

Responsibilities
----------------

The two repositories have separate jobs and must stay independent projects.

``polynomial_cl_core`` (Python package ``polynomial_cl``, "the core")
   Owns the polynomial continual-learning *method*: polynomial feature construction and
   expansion, zero-initialisation of new parameters, the polynomial models, degree-specific
   learning rates, training, continual-learning strategies, replay and sampling mechanisms,
   synthetic experiments, and the method-level mathematical and empirical investigation.

``polynomial_cl_bench`` (this repository)
   Owns the *evaluation*: the benchmark protocol, dataset preparation and validation,
   algorithm adapters, competitor implementations, hyper-parameter tuning, experiment
   execution, reproducibility, metrics and statistics, runtime and resource measurement, model
   selection, result validation, and research reporting.

The benchmark evaluates the **actual** core implementation. It does not re-implement the
polynomial model, its features, its training or its replay, and it does not substitute a
simplified stand-in. Where the benchmark needs something the core already computes, it calls
the core's public function (for example ``make_task_plan`` and ``holdout_split``).

Dependency direction
--------------------

.. code-block:: text

   polynomial_cl_bench  ->  polynomial_cl

The core never depends on the benchmark. The benchmark pins the core to a full commit SHA in
``pyproject.toml`` (see :doc:`installation`), so that a result can be tied to the exact method
code that produced it, and records the commit actually imported in every result record
(see :doc:`reproducibility`).

Data flow of one run
--------------------

.. code-block:: text

   configs/split_mnist_cil_v1.json --> Protocol  (validated, fingerprinted)
                                           |
   MNIST archives --sha256 check--> load_split_mnist --> SplitMNIST (read-only arrays,
                                           |                         projected to 16-d)
                                           v
                        make_plan(seed)  -->  ExperiencePlan  (experiences + replay exemplars;
                                           |                    one per seed, shared)
                                           v
        adapter.run_sequence(dataset, protocol, seed, plan)  -->  SequenceResult
                                           |            (accuracy matrix + algorithm info)
                                           v
        runner: metrics recomputed from the matrix, resources measured, environment and
        provenance recorded  -->  validated result record (JSON, schema 1.0)
                                           |
                        aggregate / validate_experiment  -->  summary.json, summary.md

Every algorithm receives the same dataset object and the same plan for a given seed, so the
experiences and replay exemplars are identical by construction. Metrics are computed by this
package from the accuracy matrix alone; an adapter cannot report its own metrics into the
record.

Module map
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module (``polynomial_cl_bench.``)
     - Role
   * - ``protocol``
     - The validated, algorithm-independent protocol; canonical JSON and fingerprint.
   * - ``datasets``
     - Hash verification, loading through the core's loader (``download=False``), the
       validation hold-out view.
   * - ``plan``
     - One read-only experience/replay plan per seed, built with the core's
       ``make_task_plan``; refusals shared by all adapters.
   * - ``sampling``
     - Batch sampling identical to the core's (shuffled and class-balanced), for competitors.
   * - ``algorithms.base``
     - The adapter contract (``SequenceAdapter``, ``SequenceResult``).
   * - ``algorithms.polynomial``
     - Adapter around the core's ``run_condition``; contains no polynomial maths.
   * - ``algorithms.er_mlp``
     - Benchmark-owned NumPy ER-MLP competitor.
   * - ``metrics``
     - Accuracy-matrix metrics, independent of the core.
   * - ``tuning``, ``tune``
     - Validation-only hyper-parameter search and its command line.
   * - ``selection``
     - Model selection under four objectives (standard library only).
   * - ``results``
     - Versioned result records; validation on write and on read; atomic writes.
   * - ``resources``
     - Wall and CPU time, peak RSS, environment capture.
   * - ``runner``, ``worker``
     - Experiment specification and execution, one fresh process per run.
   * - ``aggregate``, ``validation``, ``summarize``
     - Statistics across seeds, experiment-level integrity checks, summary writing.
   * - ``provenance``
     - Which core and which benchmark commit ran, and the official-run gate.
   * - ``verify_data``
     - Command line: check the MNIST archives.

Outside the package: ``configs/`` holds the protocol, algorithm configurations, grids and
experiment specifications; ``tools/`` holds the per-degree-study analysis, the comparison
driver, the paper-asset builder and mutation-check scripts; ``paper/`` holds the manuscript.
