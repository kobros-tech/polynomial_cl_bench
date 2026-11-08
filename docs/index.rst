polynomial_cl_bench
===================

``polynomial_cl_bench`` is an independent benchmark environment for continual-learning
algorithms. It evaluates the polynomial continual-learning method implemented in
`polynomial_cl_core <https://github.com/kobros-tech/polynomial_cl_core>`_ against competing
approaches under one protocol, with one shared experience and replay plan per seed,
validation-only hyper-parameter selection, and result records that carry the provenance
needed to reproduce them.

The first (and currently only) protocol is **Split MNIST, class-incremental** (``split_mnist_cil``
version 1.0). The first competitor is an experience-replay MLP (ER-MLP).

.. note::

   This documentation describes what the repository does *today*, and says for each part how
   far it has been verified. Read :doc:`status` before relying on any component, and
   :doc:`limitations` before drawing conclusions from any result. Nothing here is a
   performance claim: the benchmark produces numbers, and the paper (a separate, not yet
   validated artefact, see :doc:`status`) is where they will be interpreted.

Two repositories, one direction of dependency
---------------------------------------------

``polynomial_cl_core`` (installed as the Python package ``polynomial_cl``, called *the core*
here) owns the method. This repository owns the evaluation. The dependency points one way:
``polynomial_cl_bench`` depends on ``polynomial_cl``, never the reverse, and the core is pinned
to an immutable commit. See :doc:`architecture`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Using the benchmark

   installation
   quickstart
   architecture
   protocol
   datasets
   algorithms
   tuning_and_selection
   experiments
   reproducibility
   analysis_and_paper

.. toctree::
   :maxdepth: 2
   :caption: Status and development

   status
   limitations
   development

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
