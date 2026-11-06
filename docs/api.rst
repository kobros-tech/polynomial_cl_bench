API reference
=============

Selected public API, generated from the docstrings. The modules documented here are the ones
a user or a contributor adding an algorithm is expected to touch. Command-line entry points
(``runner``, ``tune``, ``selection``, ``summarize``, ``verify_data``) are described in the
guide pages, and the remaining modules (``provenance``, ``resources``, ``worker``,
``sampling``) are internal to the pipeline described in :doc:`reproducibility` and
:doc:`experiments`.

Protocol
--------

.. automodule:: polynomial_cl_bench.protocol
   :members: Protocol, ProtocolError
   :member-order: bysource

Datasets
--------

.. automodule:: polynomial_cl_bench.datasets
   :members: SplitMNIST, DatasetVerificationError, verify_mnist_files, load_split_mnist, validation_view

Experience plan
---------------

.. automodule:: polynomial_cl_bench.plan
   :members: ExperiencePlan, PlanError, make_plan, check_inputs, resolve_plan

Adapter contract
----------------

.. automodule:: polynomial_cl_bench.algorithms.base
   :members: AdapterError, SequenceResult, SequenceAdapter

Metrics
-------

.. automodule:: polynomial_cl_bench.metrics
   :members:

Tuning
------

.. automodule:: polynomial_cl_bench.tuning
   :members: TuningError, TuningResult, tune

Model selection
---------------

The objectives and the accuracy floor are described in :doc:`tuning_and_selection`.

.. module:: polynomial_cl_bench.selection

.. autoclass:: polynomial_cl_bench.selection.Candidate
.. autoclass:: polynomial_cl_bench.selection.Selection
.. autoclass:: polynomial_cl_bench.selection.SelectionError
.. autofunction:: polynomial_cl_bench.selection.select
.. autofunction:: polynomial_cl_bench.selection.select_all
.. autofunction:: polynomial_cl_bench.selection.pareto_front
.. autofunction:: polynomial_cl_bench.selection.candidates_from_tuning
.. autofunction:: polynomial_cl_bench.selection.report_from_tuning

Result records
--------------

.. automodule:: polynomial_cl_bench.results
   :members: RecordError, build_record, validate_record, write_record, read_record, run_key

Aggregation
-----------

.. automodule:: polynomial_cl_bench.aggregate
   :members: summarize, aggregate_algorithm, paired_comparison, aggregate_experiment

Experiment validation
---------------------

.. automodule:: polynomial_cl_bench.validation
   :members: validate_experiment
