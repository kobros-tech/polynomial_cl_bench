Hyper-parameter tuning and model selection
==========================================

Two separate things happen to hyper-parameters, and they are deliberately kept apart:

* **Tuning** (``tune``, ``tuning``) runs every configuration of a grid on the *validation*
  hold-out and writes every run to a result file.
* **Selection** (``selection``) ranks the configurations in such a result under one of four
  objectives. It uses validation results only.

Neither ever reads the test split.

Validation tuning
-----------------

.. code-block:: console

   $ python -m polynomial_cl_bench.tune --grid configs/er_mlp_grid.json \
         --data-dir data/mnist --out results/tuning_er_mlp.json
   $ python -m polynomial_cl_bench.tune --grid configs/polynomial_grid.json \
         --data-dir data/mnist --out results/tuning_polynomial.json

Each grid file declares the algorithm, its fixed settings, the protocol requirements it
assumes, the selection seeds and the grid. The runner refuses any grid whose
``approval_status`` is not ``approved``. Both shipped grids are:

.. list-table::
   :header-rows: 1
   :widths: 30 46 24

   * - Grid
     - Configurations
     - Runs
   * - ``er_mlp_grid.json``
     - One ReLU hidden layer, ``H`` in {64, 128, 256} x learning rate in
       {0.01, 0.03, 0.1, 0.3, 1.0}; plain SGD, no momentum or weight decay; class-balanced
       sampling; 200 replay exemplars per class.
     - 15 configurations x 2 seeds = 30
   * - ``polynomial_grid.json``
     - Condition ``expanding``, strategy ``ordinary``; learning rate in
       {0.01, 0.03, 0.1, 0.3, 1.0} x ``feature_power`` in {0, 0.5, 1}; same split, seeds,
       budget and rule.
     - 15 configurations x 2 seeds = 30

Properties of the procedure:

* The same function runs both algorithms, with the same split, seeds and budget.
* **Selection seeds are 100 and 101**, disjoint from the evaluation seeds (0 to 4).
* The data is the validation view (see :doc:`datasets`): fit on 90% of the training data,
  evaluate on the held-out 10%. The result records ``evaluation_split = "validation"``.
* Every run is written, including failed and divergent ones. A configuration with a failed run
  is *ineligible*; it is never silently dropped.
* An existing result file is never overwritten.
* ``--mode official`` applies the provenance rules of :doc:`reproducibility`. A
  development-mode result records its provenance and whether it would have been eligible.
* Wall and CPU time are recorded for each run, measured around ``run_sequence`` (they include
  in-loop evaluation and, for the polynomial adapter, feature construction, and exclude plan
  building). Tuning files written before time was recorded have none; their time objectives are
  reported as unavailable.

The tuning rule
~~~~~~~~~~~~~~~

The result's ``selected_config`` (the one experiments consume) follows this rule: among the
configurations whose mean final accuracy is within 0.005 of the maximum, choose the lowest mean
forgetting; exact ties go to higher accuracy, then to grid order. This rule is unchanged by the
four-objective selection below.

The Gaussian-input step-multiplier computation that motivated the polynomial grid's choice of
feature powers was preliminary and is not evidence about MNIST.

Model selection under four objectives
--------------------------------------

.. code-block:: console

   $ python -m polynomial_cl_bench.selection --tuning results/tuning_er_mlp.json \
         --time-basis wall \
         --out-json results/selection_er_mlp.json --out-md results/selection_er_mlp.md

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Objective
     - Rule (ties broken in the order listed, then by earlier candidate)
   * - ``max_accuracy``
     - Highest mean accuracy; then lower forgetting; then lower time.
   * - ``min_forgetting``
     - Lowest mean forgetting; then higher accuracy; then lower time.
   * - ``min_time``
     - Smallest time; then higher accuracy; then lower forgetting.
   * - ``best_tradeoff``
     - Smallest weighted Euclidean distance to the ideal point, after min-max normalising
       accuracy (up), forgetting (down) and time (down) over the candidate pool.

**The trade-off objective is a convention, not a measurement.** The weights (default 1, 1, 1 for
accuracy, forgetting, time) are recorded, and the report lists which candidate wins under five
other weightings (accuracy x2, forgetting x2, time x2, time ignored, equal weights) so that the
dependence on the weighting is visible.

**Accuracy floor.** Without a constraint, "lowest forgetting" and "fastest" are won by models
that barely learn: a model that learns nothing forgets nothing. ``min_forgetting``,
``min_time`` and ``best_tradeoff`` can therefore be restricted by an accuracy floor: only
candidates whose accuracy is within ``--accuracy-floor`` (absolute; default 0.02) of the best
accuracy in the pool take part. ``--no-floor`` disables it. ``max_accuracy`` has no floor; it
would be meaningless. The report contains both the unconstrained winners and the
floor-constrained winners.

**Missing values are never zero.** A candidate without accuracy or forgetting (a failed
configuration) is excluded from every objective; a candidate without a time is excluded from
the objectives that use time. Each exclusion is listed with its reason, and an objective that
cannot be ranked is reported as an error entry rather than disappearing.

**Validation only.** :func:`~polynomial_cl_bench.selection.report_from_tuning` raises
:class:`~polynomial_cl_bench.selection.SelectionError` for a tuning result whose evaluation
split is not ``validation``.

The report also contains the Pareto front over (accuracy, forgetting, time) among candidates
with all three values, and a fingerprint of the selection rule (floor and weights), so that two
reports can be checked for having used the same rule. ``--time-basis`` chooses wall or CPU
time (default wall).

``selection`` depends only on the standard library. It ranks values; the caller decides which
data they came from.

.. note::

   Selection on real MNIST tuning results has not been done, because no tuning result for real
   data exists yet. An experiment specification cannot yet choose an objective other than the
   default tuning rule: it consumes ``selected_config`` only. See :doc:`limitations`.
