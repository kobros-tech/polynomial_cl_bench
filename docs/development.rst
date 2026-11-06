Development: testing, linting and documentation
===============================================

Test levels
-----------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Command
     - What it covers
   * - ``pytest -q``
     - Unit tests (many core-free, using fake adapters) and real-core integration tests on tiny
       synthetic MNIST archives. No real data needed. Under 1 minute in the environment used
       for :doc:`status`.
   * - ``BENCH_REQUIRE_CORE=1 pytest -q``
     - The same, but a missing or incompatible core *fails* instead of skipping integration
       tests. Use this in CI.
   * - ``BENCH_REQUIRE_PINNED_CORE=1``
     - Additionally requires the installed core to be the pin from ``pyproject.toml`` as a
       remote-verified pip VCS install. Use after a clean ``pip install -e ".[dev]"``.
   * - ``BENCH_SLOW=1 BENCH_MNIST_DIR=<dir> pytest tests/test_full_split_mnist.py``
     - Real MNIST: reproduces the core's stored records for two polynomial configurations (about
       50 s). ``BENCH_MNIST_DIR`` defaults to ``data/mnist``.

Single-threaded BLAS (see :doc:`installation`) is recommended for all of them. Tests that need
the core are skipped when it is absent unless ``BENCH_REQUIRE_CORE=1``.

``tools/mutation_check.py`` and ``tools/mutation_check_phase3.py`` deliberately break the source
and check that the tests notice. They are developer aids; their figures in older notes were not
re-run when this documentation was written.

Linting
-------

.. code-block:: console

   $ ruff check .
   $ ruff format --check .

The rules are explicit in ``pyproject.toml``. A blind ``except Exception`` is allowed only
where the code records the failure and carries ``# noqa: BLE001 - <reason>``.

Continuous integration
----------------------

``.github/workflows/ci.yml`` runs lint, the test suite against the core installed from the pin
(with ``BENCH_REQUIRE_CORE`` and ``BENCH_REQUIRE_PINNED_CORE``), checks that every command-line
entry point still loads, and builds this documentation.

Building the documentation
--------------------------

.. code-block:: console

   $ pip install -e ".[docs]"
   $ sphinx-build -W --keep-going -b html docs docs/_build/html
   $ sphinx-build -W --keep-going -b latex docs docs/_build/latex
   $ make -C docs/_build/latex           # PDF; needs a TeX distribution with latexmk

``-W`` turns warnings into errors, so a broken reference or malformed docstring fails the build.
The build is hermetic: it does not fetch anything from the network (no intersphinx). The
``docs`` extra installs the core too, because the API pages import the modules.

Documentation rules for contributors:

* Describe what the code does now, and state how far it has been verified. Do not describe
  planned work as existing.
* Do not add a page without content, and do not quote a number you have not produced.
* Keep API pages to the selected, documented public surface (:doc:`api`).
