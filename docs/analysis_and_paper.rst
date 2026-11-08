Analysis tools and the research paper
=====================================

Besides the runner pipeline (:doc:`experiments`), the repository contains a second, older
pipeline in ``tools/`` that produced the assets of the paper draft in ``paper/``. This page
describes what it does. **None of it had been run on real results when this documentation was
written**; its command-line entry points load against the pinned core (checked in CI) and the
job plan is tested (``tests/test_study_jobs.py``), and that is the extent of the verification
(see :doc:`status`).

Producing the results: GitHub Actions
-------------------------------------

The inputs of the pipeline below, and the paper's regenerated assets, are produced by two
manually started workflows (Actions, *Run workflow*); neither runs on a push, and neither
commits anything: the results are uploaded as artifacts for review.

``paper-results``
   The core's per-degree study on Split MNIST, then everything downstream of it. The core is
   checked out at the commit pinned in ``pyproject.toml`` (``tools/pinned_core.py``), so the
   study runs against the same core the benchmark installs. The study is a set of independent
   runs of the core's ``experiments/per_degree_lr.py``, listed by
   ``tools/core_study_plan.py``: one sampling comparison, one per-degree selection for each of
   the 12 (strategy, sampling) cells, and one 10-epoch re-run per strategy of the cells'
   shared and selected rate vectors. Each run is its own job, so the study takes as long as
   its slowest cell, not the sum of all cells. The script's defaults apply (selection seeds
   0-2, test seeds 3-9, three epochs per experience). The ``analysis`` job then runs step 1
   below on the collected files, ``comparison`` runs step 2 (resumable across workflow runs:
   its progress is cached), and ``paper`` runs step 3 and builds the paper. Artifacts:
   ``analysis``, ``comparison``, ``paper-assets`` (``paper/generated`` and the figure PDFs, to
   be copied into ``paper/`` in a pull request once accepted) and ``paper-build``.

``benchmark-results``
   The runner pipeline of :doc:`experiments`: validation tuning of both algorithms in
   ``official`` mode, the tuned comparison with ``--reproduce-check``, and the polynomial
   reference experiment.

Both start from one verified copy of the MNIST archives (``verify_data``). A single cell can
also be run on its own machine with ``python tools/core_study_plan.py matrix --stage study``
listing the arguments to give ``experiments/per_degree_lr.py`` of a core checkout.

Pipeline
--------

.. code-block:: bash

   # 1. best settings by criterion from the core's per-degree study (standard library only)
   python tools/analyze_per_degree_study.py --results-dir core_results/per_degree \
         --out-json results/analysis/per_degree_analysis.json \
         --out-md results/analysis/per_degree_analysis.md \
         --out-csv results/analysis/per_degree_candidates.csv
   # 2. re-run those settings and the ER-MLP under one protocol and machine (resumable)
   python tools/run_comparison_study.py --analysis results/analysis/per_degree_analysis.json \
         --core-results core_results/per_degree --out-dir results/comparison
   # 3. tables, figures and number macros for the paper, then compile paper/main.tex
   python tools/build_paper_assets.py --analysis results/analysis/per_degree_analysis.json \
         --comparison results/comparison --out paper

The tools consume validated results and do not train anything, except through the explicit
comparison driver in step 2. They read the core's per-degree study results
(``core_results/per_degree`` above: the cell and 10-epoch files and the logs, which are the
only place the study's wall time is recorded). Those results are produced by the core's
experiments, are not committed here, and are produced by the ``paper-results`` workflow.

Differences from the runner pipeline
------------------------------------

* The ER-MLP grid used by ``run_comparison_study.py`` is a **superset** of the approved
  ``configs/er_mlp_grid.json``. It is therefore not the procedure of :doc:`tuning_and_selection`.
* ``analyze_per_degree_study.py`` delegates its ranking to the ``selection`` module; a test
  checks that the winners match the script's earlier, independent logic.
* Candidate settings for the polynomial method come from the core's own study, not from this
  repository's validation tuning.

Whether the numbers in the draft come from this pipeline cannot be confirmed here, because the
raw results (``results/``) are not committed.

The paper draft
---------------

``paper/index.rst`` is the canonical source of the manuscript, a reStructuredText document built
by Sphinx into HTML, PDF and Word (``python tools/build_paper.py``; see ``paper/README.rst``).
Its tables, figures and most numbers are generated into ``paper/generated/`` by
``tools/build_paper_assets.py`` and enter the text only as substitutions and included tables.
A few result figures in the draft's prose were typed by hand; they are listed in
``paper/README.rst`` and are not generated.

The earlier LaTeX manuscript, ``paper/main.tex``, is kept unchanged as the reference for the
format migration. ``tests/test_paper.py`` checks that the number macros, citations,
bibliography, labels, tables, figures and equations were all preserved, and that the HTML and
Word builds contain them. The Word version is not equivalent to the PDF: display equations are
unnumbered and table layout is left to Word.

The draft has **not** been validated: its claims have not been audited against executed
benchmark output, and it must not be cited as a source of verified results. Validating every
table, figure and numerical statement against actual experiments is a later step.
