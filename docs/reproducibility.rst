Reproducibility and provenance
==============================

A result is only useful if it can be tied to the exact code, data and configuration that
produced it. This page describes how the benchmark records and checks that.

Identity of everything that determines a run
--------------------------------------------

* **Protocol fingerprint**: SHA-256 of the canonical JSON of the protocol. Stored in every
  record and tuning result.
* **Dataset fingerprint**: SHA-256 over the four projected arrays, plus the verified archive
  hashes and the projection seed.
* **Plan fingerprint**: SHA-256 over the per-seed experiences and replay exemplars. Experiment
  validation requires every algorithm at a given seed to have the same plan fingerprint.
* **Run key**: identity of one run (algorithm configuration, protocol, seed, split, ...).
  Records re-derive it on read, so a hand-edited record is detected.
* **Experiment fingerprint**: stored in the output directory's manifest; resuming into a
  directory with a different fingerprint is refused.

Seeds and determinism
---------------------

The projection seed is fixed (0). Run seeds control experience shuffling, replay selection and
batch sampling (and ER-MLP initialisation). Selection seeds (100, 101) are disjoint from
evaluation seeds (0 to 4).

Exact repeatability depends on floating-point summation order, which multi-threaded BLAS does
not guarantee. Pin the thread counts to 1 before running anything whose repeatability you
intend to check:

.. code-block:: console

   $ export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

The thread settings are recorded in each record's environment block. ``--reproduce-check``
re-runs one seed per algorithm in a fresh process and records whether the result is
bit-identical (``exact_match``) and the maximum absolute difference. With multi-threaded BLAS
it may report a mismatch; repeat with one thread before concluding anything.

A passing check shows that a run repeats on *the same machine and software environment*. It is
not an independent reproduction on other hardware or a different BLAS, which has not been done
(see :doc:`status`).

The core pin
------------

The core is pinned to a full commit SHA in ``pyproject.toml`` (see :doc:`installation`). The
pin is never a branch. To change it:

1. Edit the SHA in ``pyproject.toml``, keeping the URL identical (provenance compares URLs).
2. Check the commit exists on the remote: ``git ls-remote
   https://github.com/kobros-tech/polynomial_cl_core.git``.
3. Reinstall (``pip install -e .``): the pin is read from installed metadata.
4. Re-run the tests with ``BENCH_REQUIRE_CORE=1 BENCH_REQUIRE_PINNED_CORE=1``.

Run modes
---------

``development`` (default)
   Any installed core is accepted. The record states the run mode, the core and benchmark
   provenance, and ``official_problems``: every reason the run would not have qualified.

``official``
   Refused with ``ProvenanceError``, before any training, unless **all** of the following hold.

   * The core commit equals the pin in this package's metadata **and** its remote identity is
     verified. That means either a pip VCS install whose recorded URL is the pinned URL, or a
     *clean* local checkout whose remote URL is the pinned URL and whose remote-tracking refs
     contain the pinned commit.
   * The benchmark has a determinable commit: a clean git checkout whose tracked file is this
     package, or a pip VCS install. A dirty benchmark checkout is rejected.

"Dirty" means modified tracked files or untracked non-ignored files, for the core and for the
benchmark. An unverified remote is an **error**; there is no warning-only path.

Remote verification is offline evidence read from pip and git. Limitation: a remote-tracking
ref shows the commit was on the remote at the last fetch or push. It cannot detect a later
force-push or deletion; ``git ls-remote`` would, but needs the network.

What a record says about provenance
-----------------------------------

``algorithm.core`` holds the commit, its source, the remote URL, ``remote_verified``,
``remote_verification``, the pin fields and ``problems``. ``algorithm.benchmark`` holds the
benchmark commit, source and problems. ``core.commit_source`` is one of:

``pip_vcs_install``
   Commit fetched from the remote URL; ``remote_verified: true``.

``local_git`` / ``local_git_dirty``
   Read from a local checkout. A dirty checkout is never eligible for an official run.

``unknown``
   The commit could not be determined.

``matches_pin`` compares against the pin in this package's metadata. For a clean install from
the pin the report is ``pip_vcs_install``, remote-verified, ``matches_pin: true``, with no
official-run problems (observed in the environment described in :doc:`status`).

Generated files
---------------

``data/`` and ``results/`` are git-ignored: archives and raw experiment outputs are not
committed. Anything that must be versioned (a protocol, a grid, a spec) lives in ``configs/``.
