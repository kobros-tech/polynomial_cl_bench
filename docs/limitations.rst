Current limitations
===================

This page lists what the benchmark does not do and what has not been shown. It should be read
together with :doc:`status`.

Not implemented
---------------

* Datasets other than Split MNIST, and protocols other than ``split_mnist_cil`` 1.0.
* Competitors other than ER-MLP.
* Codabench and OpenML integrations.
* Per-class accuracy (always ``null``).
* A plugin mechanism for algorithms (see :doc:`algorithms`).

Implemented but not yet run on real data
----------------------------------------

* Official-mode validation tuning, for either algorithm. Development-mode tuning has been run
  on real MNIST for both (30 of 30 runs each) and the tuned comparison has been run from it
  (10 of 10 runs), but those records carry no core or benchmark commit, are not
  official-eligible, and the comparison is reported as not publishable. Neither has been
  repeated under the pinned core in ``official`` mode.
* Model selection under the four objectives on those tuning results (only the default
  rule's ``selected_config`` has been used).
* A reproduction on different hardware or with a different BLAS.

Known limits of the design
--------------------------

* **The polynomial core is observed, not controlled.** It trains and evaluates inside one call,
  so the adapter cannot prove from outside that test data is untouched or that future classes
  never compete. Both are checked indirectly (see :doc:`algorithms`).
* **Polynomial resources.** Separate training and inference time are unavailable, so
  training and inference seconds are ``null`` for the polynomial algorithm. Wall time includes
  in-loop evaluation and feature construction, and is therefore not directly comparable to a
  competitor's training time. Peak memory includes the loaded dataset and is ``null`` off Linux.
* **Seeds do not vary the data split or the projection.** Confidence intervals describe
  variation due to initialisation, sampling and replay selection only, and with five seeds they
  are wide. No significance tests are computed.
* **Fixed task order.** The core fixes the class order; the adapter refuses any other.
* **Model selection in experiments.** An experiment specification consumes only the tuning
  rule's ``selected_config``; it cannot yet choose another objective.
* **Reproducibility check scope.** It re-runs one seed per algorithm with exact equality, so it
  is sensitive to BLAS threading, and a pass does not show cross-machine reproducibility.
* **Remote verification** is offline evidence and cannot detect a force-push after the last
  fetch (see :doc:`reproducibility`).
* **Grid edges.** In the development-mode tuning the selected polynomial learning rate (1.0)
  is the largest value searched, and mean validation accuracy was still rising towards it
  (for ``feature_power`` 0.5: 0.7447 at 0.3, 0.7682 at 1.0). The selected ER-MLP width (256)
  is the largest searched, and accuracy rose with width at every learning rate searched (0.01 to 1.0). The optimum for either method may therefore lie outside its grid, and a gap between
  them could partly reflect which grid reaches its optimum. The approved grids were not
  changed; extending them needs a separately approved investigation.
* **Tuning budgets.** The grids are small (15 configurations, 2 seeds). The per-degree
  polynomial configuration comes from a limited candidate search in the core, not from this
  benchmark's tuning procedure. Nothing here supports a claim that either method is optimal.

Two evaluation paths
--------------------

The comparison driver in ``tools/`` uses an ER-MLP grid that is a superset of the approved
grid, and polynomial candidates from the core's own study (see :doc:`analysis_and_paper`). A
comparison produced that way is not the validation-tuned comparison defined by the runner
pipeline.

Configuration labels
--------------------

Two descriptive strings still name the core's pre-refactor module path; see
:doc:`protocol`.

Research paper
--------------

The manuscript in ``paper/`` is a draft. Its numbers come from generated macros whose raw
inputs are not in this repository, so they are not traceable to benchmark output here, and its
claims have not been audited against executed experiments. Do not cite figures from it as
verified results. It has been migrated to reStructuredText (format only, content unchanged),
and the HTML and PDF builds are the same text; the Word build is not equivalent (display
equations are unnumbered, table layout is Word's). Validating every claim is a separate step.
The draft's prose also contains a few hand-typed result figures that are not generated; they are
listed in ``paper/README.rst``.
