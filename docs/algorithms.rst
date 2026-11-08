Algorithms
==========

Supported algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Algorithm
     - Implementation
     - Notes
   * - ``polynomial_cl``
     - The core's own ``run_condition``, wrapped by ``algorithms.polynomial``
     - Condition ``expanding``, strategy ``ordinary`` in every shipped configuration. The core
       also offers other conditions and strategies; the adapter accepts any the core lists.
   * - ``er_mlp``
     - Benchmark-owned NumPy implementation (``algorithms.er_mlp``)
     - Experience replay with a small MLP. The first competitor.

No other algorithm is implemented.

The adapter contract
--------------------

Every algorithm is wrapped in an adapter that implements
:class:`~polynomial_cl_bench.algorithms.base.SequenceAdapter`:

* ``describe()`` returns the algorithm name and its complete configuration, for the result
  record.
* ``run_sequence(dataset, protocol, seed, plan=None)`` runs every experience in order for one
  seed and returns a :class:`~polynomial_cl_bench.algorithms.base.SequenceResult`: the accuracy
  matrix, the algorithm description, adapter-reported resources (``None`` where not
  measurable, never zero), the sampling policy actually used, a ``run_config`` dictionary of
  every value that determined the run, and optional extras used for cross-checks.

The ``plan`` is the shared :class:`~polynomial_cl_bench.plan.ExperiencePlan` for the seed.
Passing the same plan to every adapter guarantees identical experiences and replay exemplars.
An adapter given a plan for a different seed, memory budget, task order or label set raises
:class:`~polynomial_cl_bench.plan.PlanError`.

Adapters refuse, rather than silently adapt, a protocol or dataset they cannot honour (for
example a task order other than the core's fixed one).

Polynomial adapter
------------------

**Integration boundary.** The core exposes ``run_condition``, which trains and evaluates all
five experiences in one call. The adapter wraps one complete run and passes ``sampling``
explicitly. Consequences:

* The core evaluates internally, so the adapter cannot verify *from outside* that test data is
  untouched during training, or that future classes never compete. That holds by the core's
  code, and is checked here indirectly: tests re-evaluate the core's weights with a
  benchmark-side implementation and match every row of the matrix, check that future-class
  output columns are exactly zero at every evaluation, and check that each experience trains
  only on its own classes plus replay of old classes.
* Per-class accuracy and peak memory are unavailable and recorded as ``null``. The adapter
  reports the wall time *including evaluation*. When the installed core can report its own
  split of a run (the ``timings`` argument of its ``run_condition``), the training time is
  the time in the core's trainer and the inference time is the time spent building the
  accuracy matrix; a core without it leaves both ``null``. The one-off construction of the
  degree-3 design matrix is reported separately as ``feature_construction_seconds`` (``0.0``
  when the features of the same dataset were already available in the process).
* **Lean mode.** The benchmark reads only the accuracy matrix and the core's accuracy and
  forgetting summary, so when the core offers it (``diagnostics`` argument of its
  ``run_condition``) the adapter runs it with ``diagnostics=False``: the core's
  observation-only work (full training-set loss passes, per-degree statistics, old-class
  indicators) is skipped, because it never influences learning and ER-MLP has no counterpart.
  The accuracy matrix is the same either way; ``tests/test_lean_adapter.py`` checks it.
  ``PolynomialAdapter(diagnostics=True)`` restores the full run for research, and is not part
  of the algorithm's configuration or identity. How a run was executed (core lean mode
  available, diagnostics on or off, features from cache, the core's timing split) is stored
  in ``resources.adapter_execution``. A core without the lean mode simply runs in full.
* The adapter validates ``feature_power`` (finite, non-negative scalar), which the core does
  not.
* It checks that the core echoes back the configured learning rate, condition, strategy and
  sampling.

**Configurations.** Two configurations are shipped, both in ``configs/``. Neither is claimed
to be optimal.

``polynomial_expanding_ordinary.json``
   Role ``primary_reference``; learning rate 0.1. This is the core's default
   (``experiment.LEARNING_RATE``) and is **not** validation-selected. It is used because it
   reproduces the core's committed result, which validates the integration.

``polynomial_expanding_ordinary_per_degree.json``
   Role ``experimental``; per-degree learning rates [0.75, 1.5, 1.5, 6.0]. This is candidate
   ``combined`` from the core's per-degree study (validation split, selection seeds 0-2,
   class-balanced, ordinary): a limited candidate search, not a systematic tuning procedure,
   and not re-run here.

Both use condition ``expanding`` and strategy ``ordinary`` and carry
``optimality_claim: none`` in their provenance. No systematic validation-based tuning has been
performed for either, so **no claim of an optimal polynomial method is made**. Competitors need
tuning procedures and budgets comparable to the polynomial method's before any comparison is
published.

``polynomial_tuned_base.json`` contains no tunable values: the experiment specification fills
``learning_rate`` and ``feature_power`` from a validation tuning result (see
:doc:`tuning_and_selection`).

Roles label why a configuration is being run: ``primary_reference``, ``experimental`` and
``validation_candidate`` for the polynomial adapter.

ER-MLP
------

A benchmark-owned NumPy implementation. It shares nothing with the polynomial model except the
protocol, the projected 16-dimensional inputs and the experience/replay plan.

* Model: ``16 -> hidden -> 10`` with ReLU hidden layers and one 10-way head. ``hidden=()`` is a
  linear softmax head.
* Per experience, the current experience's examples and the replay exemplars of earlier classes
  are **concatenated** into one pool, as in the core's ``fit_task``. Exemplars come from the
  shared plan.
* Training and evaluation let only classes introduced so far compete (softmax over the eligible
  output columns); columns of future classes are never read or updated.
* Batches come from ``sampling.training_batches``, which reproduces the core's shuffled and
  class-balanced batching draw for draw, seeded with ``seed + experience`` as in the core.
  Plain SGD on batch-mean cross-entropy; no momentum, no weight decay.
* ``hidden`` and ``learning_rate`` are **required** arguments with no defaults.
* ``replay=False`` exists only as the separate role ``ablation_no_replay``, never as part of
  the primary configuration. Roles: ``primary``, ``ablation_no_replay``,
  ``validation_candidate``.
* A non-finite loss or weight raises ``TrainingDiverged``; the run becomes a ``diverged``
  record, not a silent number.

Adding an algorithm
-------------------

There is no plugin registry. Adding an algorithm currently means:

1. Implement a class satisfying ``SequenceAdapter``, with a ``from_dict`` constructor.
2. Make ``runner.build_adapter`` construct it from a configuration whose ``algorithm`` key
   names it.
3. List its tunable keys in ``runner.TUNABLE_KEYS`` and in ``tune.py`` so that tuning results
   can be applied, and add a grid file under ``configs/``.
4. Use the shared plan and the protocol's sampling and budget; do not read ``dataset.test_*``
   during training.

A new algorithm needs tests showing that it honours the plan, never lets future classes
compete, and fails loudly on a protocol it cannot honour.
