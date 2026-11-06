Quickstart
==========

This walks through the one experiment that can run on a fresh checkout: the polynomial
*reference* configuration on full Split MNIST. It needs the data described in
:doc:`installation`. It runs five seeds plus one reproducibility re-run; on the machine used
to verify this page each run took about 23 s.

.. code-block:: console

   $ export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   $ python -m polynomial_cl_bench.verify_data --data-dir data/mnist
   $ python -m polynomial_cl_bench.runner \
         --spec configs/experiment_polynomial_reference.json \
         --data-dir data/mnist \
         --out results/polynomial_reference \
         --reproduce-check

What happens:

1. The protocol (``configs/split_mnist_cil_v1.json``) is loaded and validated, and the MNIST
   archives are checked against its hashes.
2. For each protocol seed (0 to 4) one shared experience/replay plan is built, and the
   polynomial algorithm runs in a **fresh Python process**.
3. One JSON record per run is written to ``results/polynomial_reference/runs/``, then
   validated and aggregated into ``summary.json`` and ``summary.md``.
4. ``--reproduce-check`` re-runs seed 0 in a fresh process and records whether the result is
   bit-identical.

Output directory
----------------

.. code-block:: text

   results/polynomial_reference/
       manifest.json                 resolved experiment and its fingerprint
       runs/<label>__seed<k>.json    one record per run (earlier attempts: .attempt<n>.json)
       reproducibility/              the repeat runs
       summary.json, summary.md      validation and aggregation

The directory is git-ignored. ``summary.md`` ends with a *Validation* section. For this
experiment it reports **integrity: passed** and **publishable comparison: no**, with the
reasons: the experiment runs in ``development`` mode, and the polynomial configuration is the
core's default, not a validation-selected one. That verdict is intended, not an error.

Resuming and re-running
-----------------------

.. code-block:: console

   $ python -m polynomial_cl_bench.runner --spec configs/experiment_polynomial_reference.json \
         --data-dir data/mnist --out results/polynomial_reference --resume
   $ ... --resume --retry-failed       # also re-run runs that failed (old record is kept)
   $ python -m polynomial_cl_bench.summarize --dir results/polynomial_reference

Resuming is refused if the resolved experiment or the core/benchmark commits differ from those
that produced the existing records. See :doc:`experiments`.

Checking the integration without MNIST
--------------------------------------

The test suite needs no MNIST and drives the real core on tiny synthetic archives:

.. code-block:: console

   $ pytest -q

See :doc:`development` for the test levels and the opt-in real-data tests.
