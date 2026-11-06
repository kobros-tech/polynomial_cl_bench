Benchmark protocol
==================

A *protocol* owns everything that must be identical for every algorithm: data, class order,
seeds, shared training budget, memory budget, sampling policy and evaluation rule.
Algorithm-specific settings (learning rate, strategy, ...) live in the algorithm's own
configuration, never in the protocol. The protocol is a JSON file, validated on load by
:class:`polynomial_cl_bench.protocol.Protocol`, and fingerprinted (SHA-256 of its canonical
JSON) so that any change to any field changes the fingerprint recorded in every result.

Split MNIST, class-incremental, version 1.0
-------------------------------------------

File: ``configs/split_mnist_cil_v1.json``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Item
     - Value
   * - Experiences
     - Five: classes (0, 1), (2, 3), (4, 5), (6, 7), (8, 9). One 10-way output head. The core
       fixes this order; an adapter refuses any other.
   * - Data
     - MNIST archives verified by sha256, normalised as ``(x - 0.1307) / 0.3081``, then
       projected to 16 dimensions by the core's fixed orthonormal random projection (QR of a
       seeded Gaussian matrix; not fitted to data).
   * - Projection seed
     - Fixed at 0 for every run and every algorithm. Run seeds control everything else.
   * - Run seeds
     - 0, 1, 2, 3, 4. They vary experience shuffling, replay selection and batch sampling,
       and the ER-MLP's weight initialisation (polynomial parameters start at zero). They do
       **not** vary the data split or the projection.
   * - Training budget
     - Shared: 3 epochs per experience, batch size 128.
   * - Sampling
     - ``class_balanced`` (the ``shuffled`` policy is a separate ablation, selected with
       ``training.sampling``).
   * - Memory
     - A fixed random subset of 200 examples per class (``fixed_per_class_random_subset``),
       reported per run.
   * - Evaluation
     - After every experience. See below.
   * - Validation hold-out
     - Fraction 0.1 of the *training* data, seed 123. Used for hyper-parameter selection only;
       never test data.

Evaluation rule
---------------

Let :math:`T` be the number of experiences. After training through experience :math:`i`, the
accuracy on the test examples of experience :math:`j \le i` is :math:`R_{i,j}`. The prediction
for each example is the argmax over the classes introduced **through experience** :math:`i`
**only**: classes of future experiences never compete. Entries with :math:`j > i` are undefined
(stored as NaN).

The metrics, all computed by :mod:`polynomial_cl_bench.metrics` from the matrix :math:`R`:

.. math::

   \mathrm{Acc} = \frac{1}{T}\sum_{j=0}^{T-1} R_{T-1,\,j}

.. math::

   F_j = \max_{j \le i \le T-1} R_{i,j} - R_{T-1,\,j},
   \qquad
   \mathrm{Forg} = \frac{1}{T-1}\sum_{j=0}^{T-2} F_j

Forgetting averages over experiences :math:`0 \ldots T-2` only: the last experience has no
later evaluation. This is the same definition the core uses. The benchmark also records the
accuracy of each experience after the last step, the accuracy of each experience right after it
was learned (the diagonal), and the mean accuracy over experiences seen so far after each step.
Per-class accuracy is not available (``null``): adapters return an accuracy matrix only.

Changing the protocol
---------------------

The protocol is versioned. Any change that alters what is evaluated must be justified and must
increment the version, because the fingerprint is stored in every result record, in tuning
results, and checked when an experiment consumes a tuning result. Do not edit the protocol file
in place to "fix" a description.

.. note::

   Two descriptive strings in the shipped configuration files still name the core's
   *pre-refactor* module path ``polynomial_cl.class_incremental`` (the protocol's
   ``dataset.source`` and the ``hyperparameter_provenance.note`` of the primary reference
   configuration). They are labels, not code paths, and nothing imports them. They are left
   unchanged on purpose: ``dataset.source`` is part of the protocol fingerprint, so correcting it
   is a protocol revision, not a typo fix. The code uses ``polynomial_cl.data`` and
   ``polynomial_cl.continual``.
