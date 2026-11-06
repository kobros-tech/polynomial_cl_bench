Datasets
========

Supported dataset
-----------------

**Split MNIST** is the only supported dataset. Support for other datasets is not implemented.

Preparation
-----------

:func:`polynomial_cl_bench.datasets.load_split_mnist` prepares the data in three steps.

1. **Verify.** Each of the four archive names in the protocol must exist in the data directory
   with the protocol's sha256. Any missing file or mismatch raises
   :class:`~polynomial_cl_bench.datasets.DatasetVerificationError` naming every problem.
2. **Load and project.** The core's own loader, ``polynomial_cl.data.load_projected_mnist``, is
   called with ``download=False`` and the protocol's projection seed. The benchmark therefore
   never touches the network and never uses an unverified file. The protocol's
   ``n_components`` must equal the core loader's.
3. **Verify again.** The archives are hashed once more after loading, so a file replaced
   between verification and reading is detected.

The resulting :class:`~polynomial_cl_bench.datasets.SplitMNIST` holds the four projected arrays
(60000 x 16 training, 10000 x 16 test), marked read-only because every algorithm and seed
shares them. It carries a fingerprint over the arrays, the projection seed it was built with
and the verified file hashes. Adapters refuse a dataset whose projection seed or file hashes
differ from the protocol's.

Validation view
---------------

Hyper-parameter selection must never see test data.
:func:`~polynomial_cl_bench.datasets.validation_view` calls the core's ``holdout_split`` on the
**training** data (fraction and seed from the protocol's ``validation`` block) and returns a
``SplitMNIST`` whose ``test_*`` slots hold the validation part and whose training slots hold
the rest. Any adapter can therefore be evaluated on it unchanged, and the real test arrays are
not referenced at all, so a selection procedure cannot reach them. The view records
``evaluation_split = "validation"`` and the fingerprint of the dataset it was cut from; every
result and tuning record states which split it was evaluated on.

Command-line check
------------------

.. code-block:: console

   $ python -m polynomial_cl_bench.verify_data --data-dir data/mnist

prints the expected and actual hash of every archive and exits 0 only if all four match.
