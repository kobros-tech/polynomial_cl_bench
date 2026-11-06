:orphan:

Claim audit of the draft manuscript (PR 5)
==========================================

Every kind of numerical or comparative statement in ``index.rst`` is classified by whether the
benchmark output now available supports it. Numbers themselves are in
``benchmark_results.rst`` (generated) and are not repeated here.

Status vocabulary: **supported** (reproduced by executed benchmark output, same direction and
protocol), **consistent** (same direction, different protocol or magnitude not compared),
**untraceable** (comes from a study whose raw results are not in this repository),
**stale** (describes an environment that no longer applies).

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Claim in the draft
     - Status
     - Evidence and gap
   * - The polynomial method reaches higher final accuracy than a tuned ER-MLP
     - consistent
     - The executed tuned comparison (seeds 0-4, approved grids, validation-only selection)
       shows the polynomial method higher on every one of the paired seeds, with a paired
       interval that excludes zero. The draft's gap comes from a different protocol (seeds
       3-9, wider ER-MLP grid, ten-epoch runs) and its magnitude is not reproduced.
   * - Polynomial forgetting is lower than the ER-MLP's
     - consistent
     - Lower on every paired seed in the executed comparison; five pairs only.
   * - The polynomial method costs more than the ER-MLP (time, "about 1.4x" parameters)
     - consistent
     - Measured wall time and parameter counts in the executed comparison point the same way;
       the parameter ratio is read from the generated table. The "about 1.4x" in the prose is
       hand-typed and still has to be replaced by a generated value.
   * - Per-degree learning-rate vectors, update rules (``frozen_old``, ``strategy_b``,
       ``small_old_lr``), sampling-policy cells, ablations, ten-epoch re-runs, rank
       correlations, "the 60 best candidates"
     - untraceable
     - These come from the core's per-degree study (``results/per_degree``), which is in
       neither uploaded repository and which the single-CPU environment used here could not
       regenerate. ``tools/analyze_per_degree_study.py`` and
       ``tools/run_comparison_study.py`` were therefore not run.
   * - Hand-typed figures listed in ``paper/README.rst`` (about 11 points, 2.8 points, 2.6
       points, 64.8%, 78.0%, 76-78%, 17%, 5,900 examples, ...)
     - untraceable
     - Not generated from any file. They must be generated from regenerated study output or
       removed; they were not edited here because that would change scientific content
       without evidence.
   * - Appendix: core commit ``b0b917e`` plus 21 uncommitted modifications
     - stale
     - The executed runs used the core supplied as an archive, which reproduced the core's
       stored seed-0 record exactly; it is not the pinned commit and no commit could be
       determined.
   * - Seed counts and validation/test separation
     - supported (for the new runs)
     - Selection seeds 100/101 on the validation hold-out; evaluation seeds 0-4 on the test
       split; the benchmark's own integrity checks passed.

What this audit does not do: it does not decide whether the draft's conclusions are right. It
shows that the only conclusions currently traceable to executed output are the narrow ones
above.
