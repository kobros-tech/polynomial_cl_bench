.. Canonical source of the manuscript (reStructuredText).
.. Hand-written text lives here. Every number, table cell and figure comes from the generated
.. assets: ``generated_rst/`` (derived from ``generated/*.tex`` by tools/paper_rst_assets.py)
.. and ``figures/``. Build with ``python tools/build_paper.py`` (see paper/README.rst).

.. include:: generated_rst/numbers.rst

Does Incremental Polynomial Feature Expansion Help Class-Incremental Learning?
==============================================================================

**A Controlled Comparison with Experience Replay on Split MNIST**

.. admonition:: Status of this draft

   This is the original draft, migrated from LaTeX to reStructuredText without changes to its
   text, numbers, tables, figures or references. It has **not** been validated: its numbers
   have not been regenerated from, or audited against, executed benchmark output, and the
   statements in the appendix describe the environment of the original draft. Do not cite it
   as a source of verified results.

.. rubric:: Abstract

Class-incremental learning (CIL) asks a single model to absorb new classes over time without erasing old ones. A recent line of work proposes to grow a model’s capacity *explicitly*: start from a low-degree polynomial feature map, append zero-initialised higher-degree monomials as classes arrive, and protect old knowledge with replay and per-coefficient update rules. We ask whether this works, why, and at what cost. Under one protocol (Split MNIST, five two-class experiences, a fixed 16-dimensional random projection, 200 replay exemplars per class, three epochs, identical experiences and replay exemplars per seed) we compare the polynomial method, with settings chosen on a validation split from |numPolyCandidatesThree| candidate configurations, against an experience-replay MLP (ER-MLP) tuned on |numErCandidates| configurations. The validation-selected polynomial setting reaches |numPolyBestAcc|\ % final accuracy (forgetting |numPolyBestForg|\ %), against |numErBestAcc|\ % (|numErBestForg|\ %) for the validation-selected ER-MLP, and |numErHindAcc|\ % for the ER-MLP picked in hindsight on the test set. The paired accuracy difference is |numPairedERMLPhindsightbAccDiff| points (95% CI |numPairedERMLPhindsightbAccLo| to |numPairedERMLPhindsightbAccHi|) against that strongest ER-MLP, positive in all |numTestSeeds| seeds. Ablations, however, do not support the mechanism the method is built around. Most of the gain over a linear model comes from polynomial *capacity* (|numPolyLinearAcc|\ % for degree 1 versus |numPolyFixedCapAcc|\ % for degree 3 from the start), not from expanding it gradually; class-balanced sampling matters far more than any update rule (median accuracy |numMedianClassBalancedAcc|\ % versus |numMedianShuffledAcc|\ % under shuffled sampling); and per-degree learning rates change class-balanced accuracy by at most |numPerDegreeGainCbMax| points. We also show that “lowest forgetting” and “smallest time” are degenerate objectives on their own (the winners learn almost nothing), and that the best *trade-off* configuration depends on an arbitrary weighting. All claims are limited to one dataset, one projection, one competitor family and seeds that vary only shuffling and replay selection.

Introduction
============

Neural networks trained on a stream of tasks tend to overwrite what they previously learned, a failure known as catastrophic forgetting [mccloskey1989]_. In class-incremental learning (CIL) the difficulty is sharpest: the label set grows, no task identity is given at test time, and the model must separate classes that were never seen together [vandeven2019]_, [hsu2018]_. Replay of a small stored subset of earlier data is a strong and simple remedy [chaudhry2019tiny]_, [rebuffi2017]_, and strong simple baselines are repeatedly found to be hard to beat [hsu2018]_, [prabhu2020]_.

This paper evaluates a different route to the same goal. The *polynomial CIL* method studied here represents an input by its monomials up to a degree :math:`d` and learns a linear softmax head on top. Because monomials of degree :math:`d` are a prefix of those of degree :math:`d{+}1`, the model can be enlarged by appending coefficients initialised to zero, which leaves every existing output *exactly* unchanged. The hypothesis behind the method is that such incremental expansion, combined with replay and update rules that protect old coefficients, lets a model learn new classes while retaining old ones.

The hypothesis is plausible but has several separable parts, and a favourable headline number does not tell us which part matters. We therefore organise the study around five questions.

**RQ1**
   Under identical data, experiences, replay memory and training budget, does polynomial CIL outperform a standard experience-replay MLP in accuracy, forgetting and cost?

**RQ2**
   Where does any advantage come from: polynomial capacity, gradual expansion, replay, the update rule, the sampling policy, or per-degree learning rates?

**RQ3**
   What does “best” mean? How do configurations chosen for accuracy, for forgetting, for time, and for a trade-off differ, and are the extreme choices meaningful?

**RQ4**
   Is the comparison stable to the training budget (3 versus 10 epochs)?

**RQ5**
   Does selecting configurations on validation data transfer to the test set, and how large is the optimism of selecting on the test set?

**Contributions.** (i) A controlled, reproducible comparison in which both algorithms receive byte-identical experiences and replay exemplars (checked by plan fingerprints), with all hyper-parameters chosen on a validation split. (ii) An analysis tool that, from the study’s raw results, selects the highest-accuracy, lowest-forgetting, smallest-time and best-trade-off configurations together with the parameters that define them. (iii) Evidence that the polynomial method is competitive with and modestly better than an ER-MLP on this protocol, together with ablations that attribute most of the effect to capacity and balanced replay rather than to expansion or update rules. (iv) An explicit account of limitations and of what would be needed to turn these findings into general claims.

Background and related work
===========================

**CIL protocols and metrics.** Scenario taxonomies distinguish task-, domain- and class-incremental learning [vandeven2019]_, [hsu2018]_. Evaluation commonly uses an accuracy matrix :math:`R_{i,j}` and summaries of final accuracy and forgetting [chaudhry2018riemannian]_, [delange2021]_; it has been argued that forgetting alone is misleading because a model that never learns anything new cannot forget [chaudhry2018riemannian]_, [diaz2018]_, and that evaluations must be robust to protocol choices [farquhar2018]_. We adopt both points: we report forgetting only next to final and newest-experience accuracy, and we state the protocol precisely.

**Remedies for forgetting.** Regularisation (EWC, synaptic intelligence, LwF) [kirkpatrick2017]_, [zenke2017]_, [li2018]_, replay and exemplar methods [rebuffi2017]_, [lopezpaz2017]_, [chaudhry2019tiny]_, and gradient-projection methods [farajtabar2020]_, [saha2021]_ are the main families; surveys include [delange2021]_, [mai2022]_. Two of the polynomial method’s update rules (“Strategy B”) constrain gradients to the null space of the design matrix at a few stored reference inputs, which is in the spirit of the projection methods.

**Polynomial features.** Explicit low-degree polynomial maps feeding a linear model are an old idea [ivakhnenko1971]_, [chang2010]_, and polynomial networks have been revisited recently [chrysos2022]_. To our knowledge their use as an *incrementally growing* head for CIL is the specific proposal examined here; we do not survey whether others have studied it.

The method under study
======================

**Inputs and features.** An MNIST image is normalised and mapped to :math:`z\in\mathbb{R}^{16}` by a fixed random orthonormal projection that is never fitted to data. The degree-\ :math:`d` feature map is

.. math::
   :label: eq-features

   \varphi_d(z)=\Big[\,1;\;\; s_1\,m_1(z);\;\; s_2\,m_2(z);\;\dots;\;\; s_d\,m_d(z)\Big],\qquad s_k=\binom{16+k-1}{k}^{-1/2},

where :math:`m_k(z)` stacks all monomials of total degree :math:`k`. The scaling :math:`s_k` keeps high-degree groups from dominating merely by being numerous. The feature counts are 17, 153 and 969 for :math:`d=1,2,3`.

**Model, prediction and expansion.** The head is :math:`W\in\mathbb{R}^{p\times 10}` with logits :math:`\varphi_d(z)^\top W` and no separate bias (the constant monomial plays that role). Only columns of classes introduced so far take part in the softmax, in training and in prediction, so a future class can never be predicted. Starting at degree 1 the degree schedule is :math:`(1,2,3,3,3)` over the five experiences. Expansion appends zero rows to :math:`W`; the core implementation asserts that the logit change is below :math:`10^{-5}`.

**Training.** Each experience is trained with mini-batch SGD on cross-entropy over the current experience plus a fixed random replay memory of 200 examples per earlier class. *Sampling* is either ``shuffled`` (current and replay data shuffled together) or ``class_balanced`` (every batch contains almost equal numbers of every class seen so far; a class with fewer examples than its share is repeated, never padded from another class). *Update rules* decide which coefficients move: ``ordinary`` (everything), ``ordinary_no_replay``, ``frozen_old`` (a coefficient is “old” if its feature row *and* class column already existed; old coefficients do not move), ``small_old_lr`` (old coefficients move at :math:`0.1\times` the step), and two null-space rules, ``strategy_b`` and ``strategy_b_new``, that keep the logits at three stored reference inputs per old class fixed to first order. *Learning rates* may be a single shared value or one value per degree group :math:`\eta=[\eta_0,\eta_1,\eta_2,\eta_3]` for degrees 0–3.

Benchmark protocol and experimental design
==========================================

Protocol
--------

:numref:`tab-protocol` lists the protocol (version 1.0). It is defined independently of any algorithm and is implemented in the benchmark package, which depends on the polynomial package but never modifies it.

.. _tab-protocol:

.. list-table:: Split MNIST CIL protocol used for every algorithm.
   :widths: 14 86

   * - Data
     - MNIST (60,000 train, 10,000 test), archives verified by SHA-256
   * - Input
     - :math:`(x-0.1307)/0.3081`, fixed orthonormal random projection to 16-D (seed 0)
   * - Experiences
     - :math:`\{0,1\},\{2,3\},\{4,5\},\{6,7\},\{8,9\}`; one 10-way head; future classes never compete
   * - Budget
     - 3 epochs per experience, batch 128 (10 epochs as a secondary study)
   * - Memory
     - fixed random subset of 200 exemplars per class (1,600 stored at the end)
   * - Validation
     - 10% hold-out of the training set (seed 123), used only for selection
   * - Seeds
     - selection seeds 0–2 (validation); evaluation seeds 3–9 (test); disjoint
   * - Metrics
     - :math:`R_{i,j}` accuracy on experience :math:`j` after learning :math:`i`; final accuracy, forgetting

Let :math:`T=5` and :math:`R_{i,j}` be the accuracy on the test examples of experience :math:`j` after training through experience :math:`i`, predicting by argmax over the classes introduced through :math:`i`. Final mean accuracy is :math:`\mathrm{Acc}=\frac1T\sum_j R_{T-1,j}` and mean forgetting

.. math::
   :label: eq-forgetting

   \mathrm{Forg}=\frac{1}{T-1}\sum_{j=0}^{T-2}\Big(\max_{j\le i\le T-1}R_{i,j}-R_{T-1,j}\Big),

which excludes the last experience, as in the original implementation. We also report *newest-experience accuracy* :math:`R_{T-1,T-1}` to expose stability bought by not learning. Seeds change shuffling, replay selection and (for the MLP) initialisation, *not* the data or the projection; the reported spread therefore understates dataset-level variability.

Competitor
----------

The competitor is an experience-replay MLP, :math:`16\to H\to10` with one ReLU layer (or a linear head for :math:`H=\varnothing`), He initialisation, float32, plain SGD on batch-mean cross-entropy, a masked softmax over introduced classes and *the same memory and class-balanced sampler as the polynomial method* (the benchmark re-implements the core’s batching algorithm because the core exposes it only inside its training routine). For every seed both algorithms are given the same experience plan, and the paired-comparison code asserts equality of plan fingerprints before computing any difference. The ER-MLP search space is :math:`H\in\{\varnothing,64,128,256,512\}` and :math:`\eta\in\{0.01,0.03,0.1,0.3,1,3,10\}` (35 configurations; no momentum, weight decay or learning-rate schedule).

Selection, criteria and the analysis tool
-----------------------------------------

Every configuration is scored on the validation split with selection seeds, then run on the test split with the evaluation seeds. The polynomial study explored |numPolyCandidatesThree| three-epoch candidates: 6 update rules :math:`\times` 2 sampling policies :math:`\times` 18 learning-rate vectors (the shared-rate grid :math:`\{0.01,\dots,30\}`, one degree group scaled by :math:`0.25,0.5,2,4` at a time, and one combined vector). Ten-epoch re-runs of the shared and selected vector of each cell exist for the test seeds only.

A script (``tools/analyze_per_degree_study.py``) reads the study’s result files and logs and applies four criteria, each reporting the other two metrics and the configuration: highest accuracy; lowest forgetting; smallest time; and a *trade-off*, defined as the Euclidean distance to the ideal point after min–max normalising accuracy (up), forgetting (down) and time (down) over the candidate pool. The trade-off definition is a convention; the script also reports the winner under other weightings. Because “lowest forgetting” and “smallest time” are degenerate without a quality constraint, the script additionally reports them within an *accuracy floor*: candidates within 2 points of the best accuracy. Two ranking bases are produced and kept apart: **validation** (rank on validation, report test; an honest estimate) and **test** (rank on the test seeds, the 10-epoch re-runs included; hindsight, optimistic). The same code ranks the ER-MLP candidates.

Cost measurement and its limits
-------------------------------

Time is the wall-clock time of one complete run (five experiences plus in-loop evaluation) as reported by each adapter, identical in definition for both algorithms, measured in one process on one CPU core. The polynomial runs use the core’s lean benchmark mode, which skips only observation-only work (loss passes, per-degree statistics, old-class indicators) and leaves the accuracy matrix unchanged; the study’s own runs, by contrast, keep those diagnostics on. The polynomial per-run time excludes the one-time construction of the degree-3 features for the 70,000 images (|numFeatureSeconds| s here), which the ER-MLP does not need. The polynomial study’s own logs record times on a different, slower machine (:math:`t_{\mathrm{log}}`); every comparison against the ER-MLP therefore uses times re-measured here (:math:`t_{\mathrm{here}}`). Parameter counts are those of the full 10-way head (including columns of classes not yet introduced), and the stored memory is identical for all replay methods.

Results
=======

Best settings by criterion (RQ3)
--------------------------------

:numref:`tab-criteria` gives the analysis output for the polynomial method. Ranked on validation, the highest-accuracy configuration is ``frozen_old`` with class-balanced sampling and :math:`\eta=[2.5,2.5,10,10]`, at |numPolyBestAcc|\ % accuracy and |numPolyBestForg|\ % forgetting on the test seeds. The lowest-forgetting configuration (``strategy_b``) reaches |numPolyValidationminForgettingForg|\ % forgetting at |numPolyValidationminForgettingAcc|\ % accuracy, but the null-space rules take roughly 2.5–3 times as long as the other rules. The unconstrained smallest-time winner is degenerate: a no-replay model with |numPolyValidationminTimeAcc|\ % accuracy. Low forgetting is also easy to obtain by learning little (:numref:`fig-tradeoffs`\ b), which is why newest-experience accuracy is reported beside it. Within the accuracy floor, the cheapest configuration is plain ``ordinary`` replay with a single rate of 3 (|numPolyValidationFloorminTimeAcc|\ %, |numPolyValidationFloorminTimeForg|\ % forgetting).

The trade-off winner is fragile. Ranked on validation, it is |numSensValidationEqualWeights| for equal weights, accuracy, forgetting and time weighted double, but |numSensValidationTimeIgnored| when time is ignored. Ranked in hindsight on test, it varies across weightings among |numSensTestEqualWeights|, |numSensTestAccuracyX| and |numSensTestTimeIgnored|. We conclude that a “best trade-off” is only meaningful relative to a stated weighting, and that the candidates differ by less than the seed-to-seed spread in accuracy (about 0.3–0.5 points), so they are practically exchangeable. For the remainder we call the validation-selected highest-accuracy setting the *polynomial headline setting*.

.. raw:: latex

   \begingroup\scriptsize

.. _tab-criteria:

.. table:: Polynomial CIL: the four selection criteria under two ranking bases (output of the analysis script). “Floor” restricts the pool to candidates within 2 points of the best accuracy. Accuracy and forgetting: mean ± s.d. over test seeds 3–9; “Newest” is accuracy on the last experience; :math:`t_{\mathrm{log}}`: time recorded in the study logs on the original machine; :math:`t_{\mathrm{here}}`: time re-measured on the machine used for the comparison with ER-MLP.
   :widths: 16 17 15 15 9 9 7 6 6

   .. include:: generated_rst/table_criteria.rst

.. raw:: latex

   \endgroup

.. _fig-tradeoffs:

.. figure:: figures/fig_tradeoffs.*
   :width: 100%
   :alt: Two scatter plots of accuracy and newest-experience accuracy against forgetting.

   \(a) Final accuracy versus forgetting and (b) newest-experience accuracy versus forgetting for |numPolyCandidatesThree| polynomial candidates and |numErCandidates| ER-MLP configurations (test seeds). |numBelowRange| polynomial points below 55% accuracy (mainly no-replay models) are outside panel (a). Red outlines: validation-selected polynomial settings. In panel (b), configurations with the lowest forgetting have the lowest newest-experience accuracy: they forget little because they learned little.

Polynomial CIL versus experience replay (RQ1)
---------------------------------------------

:numref:`tab-h2h` places both methods on the same protocol, selection procedure and machine. The polynomial headline setting attains |numPolyBestAcc|\ % ± |numPolyBestAccSd| accuracy with |numPolyBestForg|\ % forgetting; the validation-selected ER-MLP (:math:`H{=}512`, :math:`\eta{=}1`) attains |numErBestAcc|\ % ± |numErBestAccSd| with |numErBestForg|\ % forgetting. That ER-MLP is more plastic (newest-experience accuracy |numErBestNew|\ % versus |numPolyBestNew|\ %) and forgets more. Against the ER-MLP chosen in hindsight on the test seeds (:math:`H{=}512`, :math:`\eta{=}0.3`, |numErHindAcc|\ %, |numErHindForg|\ % forgetting) the polynomial setting is still ahead, which makes the comparison conservative in the competitor’s favour.

:numref:`tab-paired` gives paired differences, which are meaningful because both methods see the same experiences and exemplars for each seed. The polynomial setting is better in accuracy by |numPairedERMLPvalidationAccDiff| points (95% CI |numPairedERMLPvalidationAccLo| to |numPairedERMLPvalidationAccHi|) against the validation-selected ER-MLP and by |numPairedERMLPhindsightbAccDiff| points (|numPairedERMLPhindsightbAccLo| to |numPairedERMLPhindsightbAccHi|) against the hindsight-best ER-MLP, in |numPairedERMLPhindsightbAccWins| of |numPairs| seeds, and it forgets less in every seed. The CIs are :math:`t`-intervals over |numPairs| seeds that vary only shuffling, replay selection and initialisation: they quantify optimiser noise, not dataset variability, and a sign test over seven seeds cannot go below :math:`p=2/2^7\approx0.016`. We therefore read the result as a *consistent small advantage on this protocol*, not as established superiority.

Cost does not separate the methods clearly. Per-run time is |numPolyBestTime| s for the polynomial setting and |numErBestTime| s for the ER-MLP with :math:`H{=}512`, but the ER-MLP with :math:`H{=}256` reaches |numErFloorTradeAcc|\ % in |numErFloorTradeTime| s with |numErFloorTradeParams| parameters (polynomial: |numPolyBestParams|). Counting the one-time feature construction of |numFeatureSeconds| s, the polynomial method is clearly the more expensive of the two. The extra accuracy of |numPairedERMLPbesttradeAccDiff| points over the :math:`H{=}256` model therefore comes with about :math:`1.4\times` the parameters and a fixed up-front cost.

.. raw:: latex

   \begingroup\scriptsize

.. _tab-h2h:

.. table:: Head-to-head on identical protocol, replay memory, seeds and machine (test seeds 3–9, 3 epochs). “val.” = ranked on validation; “floor” = within 2 points of the best accuracy; “hindsight” = ranked on the test seeds (optimistic). Params: parameters of the 10-way head. Memory: stored exemplars at the end.
   :widths: 25 30 7 7 8 8 8 7

   .. include:: generated_rst/table_head2head.rst

.. raw:: latex

   \endgroup

.. raw:: latex

   \begingroup\scriptsize

.. _tab-paired:

.. table:: Paired differences (polynomial headline setting minus competitor) over the seven test seeds. Positive accuracy difference and negative forgetting difference favour the polynomial method. Seeds do not vary the data, so intervals understate dataset-level uncertainty.
   :widths: 40 15 15 15 15

   .. include:: generated_rst/table_paired.rst

.. raw:: latex

   \endgroup

.. _fig-curves:

.. figure:: figures/fig_curves.*
   :width: 100%
   :alt: Mean accuracy over experiences seen after each experience, and final accuracy on each experience.

   \(a) Mean accuracy over experiences seen, after each experience (mean ± s.d. over seeds). (b) Final accuracy on each experience. The polynomial setting retains the middle experiences better; the ER-MLP is better on the newest one (|numPeErE|\ % versus |numPePolyE|\ %).

Where does the effect come from? (RQ2)
--------------------------------------

**Capacity.** A degree-1 polynomial model is a linear softmax classifier and, as expected, matches the ER-MLP with a linear head (|numPolyLinearAcc|\ % versus |numErLinearAcc|\ %, forgetting |numPolyLinearForg|\ % versus |numErLinearForg|\ %), which also serves as a sanity check of the two implementations. Raising the degree to 3 from the start gives |numPolyFixedCapAcc|\ %, an increase of about 11 points. Polynomial capacity is the dominant factor.

**Gradual expansion.** Expanding from degree 1 to 3 over the sequence does *not* beat starting at degree 3. With a shared rate, the expanding schedule with plain replay gives |numCellOrdinaryCbSharedAcc|\ % accuracy and |numPolyTradeForg|\ % forgetting; degree 3 from the start, with its rate chosen on validation, gives |numPolyFixedCapAcc|\ % and |numPolyFixedCapForg|\ % forgetting. Expansion is thus associated with less forgetting (by about 2.8 points) and lower newest-experience accuracy (|numPolyTradeNew|\ % versus |numPolyFixedCapNew|\ %), that is, more stability and less plasticity, with no accuracy gain. This comparison is not perfectly balanced: the fixed-capacity model was tuned only over the shared-rate grid and without the update rules, which require “old” coefficients to exist.

**Replay.** Without replay the polynomial model reaches only |numPolyNoReplayAcc|\ % (|numPolyNoReplayForg|\ % forgetting) and the ER-MLP |numErNoReplayAcc|\ %; both predict almost only the newest classes (newest-experience accuracy :math:`\ge` |numPolyNoReplayNew|\ %).

**Sampling policy, update rule and per-degree rates.** :numref:`tab-cells` shows each update rule under both sampling policies. With class-balanced sampling the five replay-based rules lie within a narrow band (about 76–78%), and the learning-rate vector chosen per degree changes accuracy by between |numPerDegreeGainCbMin| and |numPerDegreeGainCbMax| points relative to a shared rate. Under shuffled sampling accuracy collapses (median |numMedianShuffledAcc|\ % versus |numMedianClassBalancedAcc|\ %), with large seed-to-seed variance. In that regime the update rules do differ: ``strategy_b`` (64.8%) beats ``ordinary`` (|numCellOrdinaryShSharedAcc|\ %) and ``frozen_old`` (|numCellFrozenOldShSharedAcc|\ %) with a shared rate. A reading consistent with these numbers, though not proved by them, is that much of what the update rules and large per-degree rates achieve under shuffled sampling is compensation for class imbalance (about 5,900 current examples versus 200 stored per old class in the last experience), which class-balanced sampling removes directly. The headline setting is not better than the shared-rate cells (|numCellFrozenOldCbSharedAcc|\ % for ``frozen_old``, 78.0% for ``small_old_lr``); the differences among these cells are within seed noise, so the per-degree vector is not what separates the polynomial method from the ER-MLP, and the ranking of the top polynomial cells is not resolved by this study.

.. raw:: latex

   \begingroup\scriptsize

.. _tab-cells:

.. table:: Update rule × sampling policy: shared-rate reference versus the validation-selected per-degree vector (test seeds 3–9, 3 epochs; :math:`t` re-measured here for the selected vector).
   :widths: 18 16 6 8 9 1 19 8 9 6

   .. include:: generated_rst/table_cells.rst

.. raw:: latex

   \endgroup

Training budget (RQ4)
---------------------

With ten epochs (:numref:`tab-epochs`, no re-selection) the polynomial setting improves to |numPolyTenAcc|\ % (forgetting |numPolyTenForg|\ %), whereas the ER-MLPs do not improve in accuracy and forget more (for example, forgetting rises from |numErFloorTradeForg|\ % to about 17% for :math:`H{=}256`). The best ER-MLP at ten epochs reaches |numErTenBestAcc|\ %, so the gap of roughly 2.6 points persists. Time grows two- to threefold for both. Because rates were selected for three epochs, a re-tuned ER-MLP might do better at ten; we did not test this.

.. raw:: latex

   \begingroup\scriptsize

.. _tab-epochs:

.. table:: Three versus ten epochs per experience, without re-selection (test seeds 3–9).
   :widths: 13 28 10 10 10 10 10 9

   .. include:: generated_rst/table_epochs.rst

.. raw:: latex

   \endgroup

Does validation selection transfer? (RQ5)
-----------------------------------------

Across the |numCandidatesValidation| validation candidates, validation and test accuracy are almost perfectly rank-correlated (Spearman :math:`\rho=` |numSpearmanAcc|; forgetting :math:`\rho=` |numSpearmanForg|), but this is driven by the wide range that includes no-replay and unstable candidates; among the 60 best candidates it falls to :math:`\rho=` |numSpearmanAccTop|. Test accuracy is on average |numMeanAccGap| points higher than validation accuracy, plausibly because the validation models are fitted to 90% of the training data and the two splits are not equally difficult; we did not investigate this. The validation-selected setting is |numRegret| points below the best three-epoch candidate in hindsight, and the hindsight best overall (ten epochs, |numPolyTenAcc|\ %) differs from the validation-selected one only through the extra epochs. The optimism of selecting on test is thus small here, but it is not zero, and it is why we base all claims on the validation ranking.

.. _fig-transfer:

.. figure:: figures/fig_transfer.*
   :width: 100%
   :alt: Validation versus test accuracy and forgetting for all three-epoch polynomial candidates.

   Validation versus test (a) accuracy and (b) forgetting for all three-epoch polynomial candidates, coloured by update rule. The dashed line is :math:`y=x`.

Discussion
==========

**Is polynomial CIL “an answer”?** On this protocol it is a competitive and modestly better answer than a replay MLP of comparable size, with lower forgetting and a consistent paired advantage across seeds. It is not demonstrated to be an answer *for the reason it was proposed*. The method’s three distinguishing ingredients (incremental expansion, protective update rules, per-degree learning rates) contribute little or nothing measurable once class-balanced replay is used; the strong effects are polynomial capacity, replay and balanced sampling. A fair summary is that a degree-3 polynomial head on a 16-dimensional projection is a well-conditioned, cheap-to-train classifier that suits replay-based CIL, and that gradual expansion trades some plasticity for stability.

**Why might the polynomial head forget less?** The ER-MLP adapts a shared hidden representation to every experience, whereas a polynomial head with a fixed feature map changes only a linear readout of fixed features; this is a plausible reason for lower forgetting and lower plasticity (:numref:`fig-curves`\ b), but the present study does not isolate it (for example by freezing the MLP’s hidden layer or by adding random-feature baselines of matching width).

**What “best” means.** The criteria produce four different winners, two of which are degenerate (:numref:`fig-tradeoffs`\ b). A benchmark should therefore report a Pareto set with newest-experience accuracy, as the accompanying results do, and not a single scalar; the trade-off score is useful for presenting one default, provided its weighting is stated.

**Open questions.** Does the advantage persist with higher-dimensional inputs, where the number of monomials explodes and the 16-dimensional projection is not available? Is it a property of the head or of the fixed projection? How do stronger CIL methods (iCaRL-style exemplar classifiers, gradient-projection methods) compare? Would a tuned learning-rate schedule or weight decay remove the ER-MLP’s disadvantage? Would a different task order or a different number of classes per experience change the ranking of update rules?

Threats to validity and limitations
-----------------------------------

* **One dataset and a weak input representation.** Split MNIST on a 16-dimensional random projection yields about 78% accuracy; absolute numbers are not comparable to results using full-resolution inputs and deeper networks. Additional datasets will be needed if reviewers ask for them.
* **One competitor family.** Only an ER-MLP was run. EWC, LwF and other methods were not, so the comparison does not establish a position among CIL methods.
* **Unequal search effort.** The polynomial method was tuned over |numPolyCandidatesThree| candidates (including structural choices), the ER-MLP over |numErCandidates| (width and rate). The ER-MLP’s validation-selected configuration sits at the upper edge of the width grid (:math:`H{=}512`; its rate is interior), so a wider search could raise it.
* **Seed variance only.** Seeds do not vary the data split or projection.
* **Timing.** Single machine, one core, sub-second to few-second runs; excluding polynomial feature construction favours the polynomial method.
* **Numerical reproducibility.** Re-running |numReproChecks| polynomial settings against the stored core results reproduced |numReproExact| of them bit-for-bit. All class-balanced settings agree to :math:`\le 1.2\times10^{-4}`; the largest differences (:math:`\le` |numReproMaxDiff|) occur for shuffled sampling with rates :math:`\ge30`, where training is unstable and floating-point differences between machines are amplified. Conclusions rest on the class-balanced settings.
* **Fixed-capacity ablation.** It was tuned with a shared rate only.

Conclusion
==========

Under one controlled protocol, polynomial CIL with class-balanced replay is a small but consistent improvement over a tuned experience-replay MLP in accuracy and forgetting, at comparable per-run cost once features are available. The improvement is attributable chiefly to polynomial capacity and balanced replay, not to incremental expansion, update rules or per-degree learning rates, which the ablations leave unsupported as sources of accuracy. The study’s main methodological lesson is that single-number “best” claims (lowest forgetting, smallest time, best trade-off) are fragile or degenerate and should be replaced by validation-based Pareto reporting.

Appendix: Reproducibility
=========================

* **Core package** ``polynomial_cl``: commit ``b0b917ee240db7a7ea7d78cb4bf21754e99d554b`` of the checkout supplied for this study, with uncommitted working-tree changes (21 modified files). The benchmark’s imports were updated to the refactored core API (``polynomial_cl.continual``, ``polynomial_cl.data``); the core was not modified. After this change the adapter reproduces the stored seed-3 result of the combined per-degree setting exactly (accuracy 0.7734281931972611, forgetting 0.09086581818480258).

* **Study results**: produced by the core’s ``experiments/per_degree_lr.py``, one run per cell (selection seeds 0–2, test seeds 3–9), by the ``paper-results`` workflow of this repository at the core commit pinned in ``pyproject.toml``.

* **Analysis** (wall time is not stored in the result JSON; it is read from the logs and each log line is cross-checked against the JSON run before use):

  .. code-block:: bash

     python tools/analyze_per_degree_study.py \
       --results-dir <core>/results/per_degree \
       --out-json results/analysis/per_degree_analysis.json \
       --out-md   results/analysis/per_degree_analysis.md \
       --out-csv  results/analysis/per_degree_candidates.csv

* **Comparison** (resumable; every run, failed ones included, is appended to ``runs.jsonl``):

  .. code-block:: bash

     python tools/run_comparison_study.py \
       --analysis results/analysis/per_degree_analysis.json \
       --core-results <core>/results/per_degree \
       --out-dir results/comparison

* **Paper assets** (all tables, figures and quoted numbers are generated here):

  .. code-block:: bash

     python tools/build_paper_assets.py \
       --analysis results/analysis/per_degree_analysis.json \
       --comparison results/comparison --out paper

References
==========

.. [mccloskey1989] \M. McCloskey and N. J. Cohen. Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation*, 24:109–165, 1989.
.. [vandeven2019] \G. M. van de Ven and A. S. Tolias. Three scenarios for continual learning. arXiv:1904.07734, 2019.
.. [hsu2018] \Y.-C. Hsu, Y.-C. Liu, A. Ramasamy and Z. Kira. Re-evaluating continual learning scenarios: a categorization and case for strong baselines. arXiv:1810.12488, 2018.
.. [chaudhry2019tiny] \A. Chaudhry et al. On tiny episodic memories in continual learning. arXiv:1902.10486, 2019.
.. [rebuffi2017] \S.-A. Rebuffi, A. Kolesnikov, G. Sperl and C. H. Lampert. iCaRL: incremental classifier and representation learning. In *CVPR*, 2017.
.. [prabhu2020] \A. Prabhu, P. H. S. Torr and P. K. Dokania. GDumb: a simple approach that questions our progress in continual learning. In *ECCV*, 2020.
.. [chaudhry2018riemannian] \A. Chaudhry, P. K. Dokania, T. Ajanthan and P. H. S. Torr. Riemannian walk for incremental learning: understanding forgetting and intransigence. In *ECCV*, 2018.
.. [delange2021] \M. De Lange et al. A continual learning survey: defying forgetting in classification tasks. *IEEE TPAMI*, 44(7):3366–3385, 2022.
.. [diaz2018] \N. Díaz-Rodríguez, V. Lomonaco, D. Filliat and D. Maltoni. Don’t forget, there is more than forgetting: new metrics for continual learning. arXiv:1810.13166, 2018.
.. [farquhar2018] \S. Farquhar and Y. Gal. Towards robust evaluations of continual learning. arXiv:1805.09733, 2018.
.. [kirkpatrick2017] \J. Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13):3521–3526, 2017.
.. [zenke2017] \F. Zenke, B. Poole and S. Ganguli. Continual learning through synaptic intelligence. In *ICML*, 2017.
.. [li2018] \Z. Li and D. Hoiem. Learning without forgetting. *IEEE TPAMI*, 40(12):2935–2947, 2018.
.. [lopezpaz2017] \D. Lopez-Paz and M. Ranzato. Gradient episodic memory for continual learning. In *NeurIPS*, 2017.
.. [farajtabar2020] \M. Farajtabar, N. Azizan, A. Mott and A. Li. Orthogonal gradient descent for continual learning. In *AISTATS*, 2020.
.. [saha2021] \G. Saha, I. Garg and K. Roy. Gradient projection memory for continual learning. In *ICLR*, 2021.
.. [mai2022] \Z. Mai et al. Online continual learning in image classification: an empirical survey. *Neurocomputing*, 469:28–51, 2022.
.. [ivakhnenko1971] \A. G. Ivakhnenko. Polynomial theory of complex systems. *IEEE Trans. Systems, Man, and Cybernetics*, 1(4):364–378, 1971.
.. [chang2010] \Y.-W. Chang, C.-J. Hsieh, K.-W. Chang, M. Ringgaard and C.-J. Lin. Training and testing low-degree polynomial data mappings via linear SVM. *JMLR*, 11:1471–1490, 2010.
.. [chrysos2022] \G. G. Chrysos et al. Deep polynomial neural networks. *IEEE TPAMI*, 44(8):4021–4034, 2022.
