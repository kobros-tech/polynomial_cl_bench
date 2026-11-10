Installation
============

Requirements
------------

* Python 3.9 or newer.
* NumPy (installed automatically).
* Network access to GitHub *at install time only*: the core is installed from a pinned Git
  commit. The benchmark itself never uses the network when it runs and never downloads data.

Install
-------

Clone the repository and install it in a virtual environment:

.. code-block:: console

   $ python -m venv .venv && . .venv/bin/activate
   $ pip install -e .            # the benchmark + the core at the pinned commit
   $ pip install -e ".[dev]"     # ... plus pytest, ruff and scipy
   $ pip install -e ".[docs]"    # ... plus Sphinx, to build this documentation

The core dependency is declared in ``pyproject.toml`` as a ``git+https`` URL with a full
40-character commit SHA, never a branch name. It is currently:

.. code-block:: text

   polynomial-cl @ git+https://github.com/kobros-tech/polynomial_cl_core.git@2e806d7a3437e930256eb49ab7a3ec5e650a1110

That commit is the merge commit, on the core's ``main``, of the refactor that introduced the
``polynomial_cl.continual`` and ``polynomial_cl.data`` packages, which this benchmark uses.

.. important::

   The benchmark reads the pin from the **installed package metadata**. If you change the pin in
   ``pyproject.toml``, reinstall (``pip install -e .``) before running anything: otherwise
   provenance checks still compare against the old pin.

Using a local checkout of the core
----------------------------------

For development you may install the core from a local clone instead
(``pip install -e ../polynomial_cl_core``). Development-mode runs accept any installed core and
record which one was used. An ``official`` run accepts a local checkout only if it is clean, is
at the pinned commit, and its remote URL and remote-tracking refs show the commit exists on the
pinned remote; see :doc:`reproducibility`.

Data
----

The benchmark needs the four standard MNIST ``.gz`` archives, under exactly these names and
**not decompressed**, in one directory (``data/mnist`` by default; ``data/`` is git-ignored):

.. code-block:: text

   train-images-idx3-ubyte.gz  440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
   train-labels-idx1-ubyte.gz  3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
   t10k-images-idx3-ubyte.gz   8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6
   t10k-labels-idx1-ubyte.gz   f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6

The benchmark does not fetch them. Check them with:

.. code-block:: console

   $ python -m polynomial_cl_bench.verify_data --data-dir data/mnist

The command exits with status 0 only when all four hashes match the protocol.

Single-threaded BLAS
--------------------

Exact-equality reproducibility checks are only meaningful with deterministic floating-point
summation order. Before running experiments or the test suite, pin the BLAS thread count:

.. code-block:: console

   $ export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
