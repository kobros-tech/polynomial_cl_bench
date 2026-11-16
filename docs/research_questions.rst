Research questions and the evidence for each
============================================

The paper's questions are answered by different experiments, and this page says which. The
studies below are **exploratory, development-mode** runs. They are not maintainer-approved
experiment specifications and nothing here feeds an official comparison. All use Split MNIST
with the 16-dimensional fixed projection, five test seeds (0-4) that vary shuffling and replay
selection but not the data split, and report descriptive paired differences, not significance
tests.

.. list-table::
   :header-rows: 1
   :widths: 8 40 52

   * - RQ
     - Question
     - Where it is answered
   * - 1
     - Does expansion preserve existing predictions?
     - The core's own tests (a property of the model, not of a benchmark run). Not re-tested
       here.
   * - 2
     - Does gradual expansion help, at equal final capacity?
     - ``tools/run_research_questions.py`` (equal tuning budget).
   * - 3
     - How strong is the sampling policy?
     - ``tools/run_research_questions.py``.
   * - 4
     - Plasticity versus retention across update rules.
     - ``tools/run_research_questions.py``; arrival accuracy is reported next to forgetting.
   * - 5
     - Does degree control improve that trade-off?
     - ``tools/run_degree_control_study.py`` (needs a core with degree control).
   * - 6
     - How competitive is the method against an independent baseline?
     - The official ``tuned_comparison`` experiment (:doc:`status`).

RQ2-RQ4: equal tuning budget
----------------------------

Every cell is tuned on the validation view over the same approved 15-configuration grid
(learning rate x feature power) with the benchmark's own selection rule and seeds 100 and 101,
then run once on the test seeds. Cells differ only in the factor under study; no cell has its
own per-degree rates.

*RQ2.* With class-balanced sampling, ``fixed_linear`` is far worse than ``expanding``
(accuracy -0.110, 5 of 5 seeds). ``fixed_capacity`` has the final capacity from the start and
is **slightly better** than ``expanding`` in accuracy (+0.0030, interval [+0.0014, +0.0046],
5 of 5) and in accuracy on arrival (+0.0080), with higher forgetting (+0.0063). So the gain is
explained by final capacity, and gradual expansion buys a little less forgetting for slightly
less accuracy. The data do not support a claim that expansion itself improves accuracy.

*RQ3.* The sampling policy is the largest effect measured. Moving ``expanding / ordinary``
from ``class_balanced`` to ``shuffled`` lowers final accuracy by 0.157 and raises forgetting by
0.324 in all 5 seeds, while accuracy on arrival *rises* by 0.103, because the newest classes
dominate each batch. The same direction holds in every row of the design.

*RQ4.* Under class-balanced sampling and a tuned shared rate, ``frozen_old`` and ``strategy_b``
raise accuracy (+0.0098 and +0.0033) but also **raise** forgetting (+0.0096 and +0.0232) and
raise arrival accuracy (+0.0175 and +0.0219) relative to ``ordinary``: they act on acquisition,
not on retention. Under ``shuffled`` sampling the ranking changes (``frozen_old`` falls to
0.5065, ``strategy_b`` is 0.6360 against 0.6200 for ``ordinary``), so what an update rule does
depends on the sampling policy.

**Limit that qualifies all three.** Every cell's selection sits at the top learning rate of the
grid (1.0), and most update-rule cells also at ``feature_power`` 1. The shared-rate grid is
truncated for all of them, so differences between update rules in particular may reflect which
cell the grid serves best, not the rules. The core's per-degree rates for ``frozen_old`` and
``strategy_b`` go up to 10.

RQ5: degree control
-------------------

``tools/run_degree_control_study.py`` runs the core's degree-control protocol through the
benchmark: candidates are generated and ranked with the core's helpers (acquisition floor 0.5,
Pareto front of accuracy and forgetting, then accuracy minus forgetting), selected on the
validation view with seeds 100 and 101, and compared once on the test seeds, paired by seed
with each strategy's all-ones baseline. The base per-degree rates are the core's and are not
re-tuned.

.. list-table::
   :header-rows: 1

   * - Strategy
     - Final accuracy (change)
     - Forgetting (change)
     - Mean arrival accuracy (change)
   * - ``ordinary``
     - -0.0001 (interval spans 0, 2 of 5 seeds higher)
     - -0.0049 (lower in 5 of 5)
     - -0.0041 (lower in 5 of 5)
   * - ``frozen_old``
     - -0.0134 (lower in 5 of 5)
     - -0.0192 (lower in 5 of 5)
     - -0.0297 (lower in 5 of 5)
   * - ``strategy_b``
     - -0.0052 (lower in 5 of 5)
     - -0.0181 (lower in 5 of 5)
     - -0.0245 (lower in 5 of 5)

This independently reproduces the core's own conclusion: degree control **lowers forgetting and
also lowers accuracy on arrival**, so it exchanges plasticity for retention; it does not
improve retention in a way that costs nothing. For ``ordinary`` the final accuracy is
unchanged, which is the most favourable case, but the effect on forgetting is half a point and
arrival accuracy still falls. Control does not change the gap to the tuned ER-MLP for
``ordinary`` (+0.0098 against +0.0099 for the baseline); for ``strategy_b`` it moves the
polynomial from level with ER-MLP to below it in accuracy. Selection was under class-balanced
sampling only, and the core's own transfer check shows the trade-off does not carry over to
``shuffled`` sampling.

Running the studies
-------------------

.. code-block:: console

   python tools/run_research_questions.py --data-dir data/mnist \
       --out results/research_questions.json --cache-dir results/rq_cache
   python tools/run_degree_control_study.py --data-dir data/mnist \
       --out results/degree_control.json \
       --er-tuning results/tuning_er_mlp.json --poly-tuning results/tuning_polynomial.json

The first takes about ten minutes on one CPU and is resumable (``--cache-dir``, ``--max-seconds``).
The second takes about four minutes and requires a core with ``degree_control_factors``; with
any other core it stops with a message. Neither overwrites an existing output file.
