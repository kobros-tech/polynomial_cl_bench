# polynomial_cl_bench

Independent benchmark environment for continual-learning algorithms. Dependency direction is
strictly `polynomial_cl_bench -> polynomial_cl`; the core package is installed from a pinned
commit and is never modified or patched.

**Status: Phase 3 implemented (runner, multi-seed experiments, resource measurement, aggregation,
failure handling, experiment validation). Phase 3 has not been run against the real core or real
MNIST by its author**: see "Phase 3 status and what is unverified" at the end. Implemented before:
protocol config, dataset verification/loading, adapter contract, polynomial adapter, ER-MLP competitor,
shared experience/replay plan, validation tuning, official-run provenance rules. **Not implemented:**
Codabench/OpenML integrations (Phase 4), reporting beyond a Markdown summary (Phase 5).

## Layout

    configs/split_mnist_cil_v1.json            protocol (algorithm-independent)
    configs/polynomial_expanding_ordinary.json            primary_reference (lr 0.1)
    configs/polynomial_expanding_ordinary_per_degree.json experimental (lr [0.75, 1.5, 1.5, 6.0])
    src/polynomial_cl_bench/protocol.py        validated Protocol, canonical JSON, fingerprint
    src/polynomial_cl_bench/datasets.py        sha256 verification, core loader with download=False
    src/polynomial_cl_bench/metrics.py         accuracy-matrix metrics (independent of the core)
    src/polynomial_cl_bench/algorithms/base.py SequenceAdapter contract, SequenceResult
    src/polynomial_cl_bench/algorithms/polynomial.py  wrapper around core run_condition
    src/polynomial_cl_bench/results.py         versioned run records (schema 1.0), validation, atomic writes
    src/polynomial_cl_bench/resources.py       wall/CPU time, peak RSS during a run, environment capture
    src/polynomial_cl_bench/runner.py          experiment spec, execute_run, resume/retry, CLI
    src/polynomial_cl_bench/worker.py          one run in a fresh process (process isolation)
    src/polynomial_cl_bench/aggregate.py       mean/std/CI across seeds, paired comparisons
    src/polynomial_cl_bench/validation.py      experiment-level integrity checks and publication blockers
    src/polynomial_cl_bench/summarize.py       validate + aggregate a results directory, write summary.json/.md
    configs/experiment_tuned_comparison.json   the intended comparison: both algorithms from validation tuning
    configs/experiment_polynomial_reference.json  single-algorithm run of the (untuned) reference config

## Protocol v1.0 (Split MNIST, class-incremental)

- Five experiences (0,1) (2,3) (4,5) (6,7) (8,9); one 10-way head.
- Data: MNIST archives verified by sha256, normalised, then the core's fixed orthonormal random
  projection to 16 dimensions. **Projection seed is fixed at 0 for every run and algorithm**;
  run seeds control everything else (experience shuffling, replay selection, batch sampling).
- Shared training budget: 3 epochs per experience, batch size 128, **class_balanced sampling**
  (the `shuffled` policy is a separate ablation: set `training.sampling`).
- Memory: fixed random subset of 200 examples per class. Reported per run.
- Evaluation after every experience: `R[i, j]` = accuracy on the test examples of experience `j`
  after training through experience `i`, predicting by argmax over classes introduced through
  experience `i` only. Future classes never compete.
- Forgetting: best accuracy of experience `j` over rows `j..T-1` minus its final accuracy,
  averaged over experiences `0..T-2` (final experience excluded), as in the core.
- Hyperparameter selection must use the validation hold-out (fraction 0.1, seed 123), never test.

## Polynomial adapter: integration boundary

The core exposes `run_condition`, which trains and evaluates all five experiences in one call. The
adapter wraps one complete run and calls it with `sampling` passed explicitly. Consequences:

- The core evaluates internally, so the adapter **cannot verify from outside** that test data is
  untouched during training or that future classes never compete; this holds by the core's code
  (see `accuracy_by_task`, `fit_task(eligible_classes=...)`), and is only checked here through the
  accuracy-matrix shape and by comparison with stored core results.
- Task order is hard-coded in the core; the adapter refuses any other order.
- Per-class accuracy, separate training/inference time and peak memory are unavailable (`None`).
  `wall_time_seconds_including_evaluation` is reported instead.

## Polynomial configurations (neither is claimed optimal)

| Config | Role | Learning rate | Basis |
| --- | --- | --- | --- |
| `polynomial_expanding_ordinary.json` | `primary_reference` | 0.1 | Core default (`experiment.LEARNING_RATE`), **not** validation-selected. Used because it reproduces the core's committed result, which validates the integration. |
| `polynomial_expanding_ordinary_per_degree.json` | `experimental` | [0.75, 1.5, 1.5, 6.0] | Candidate `combined` from the core's per-degree study (validation split, selection seeds 0-2, class_balanced, ordinary). A limited candidate search, not a systematic tuning procedure; not re-run here. |

Both use condition `expanding`, strategy `ordinary`, and `optimality_claim: none` in their
provenance. No systematic validation-based tuning has been performed for either, so no claim of an
optimal polynomial method is made. Competitors will need comparable tuning procedures and budgets
before any comparison is published.

Also inspected: `experiments/rate_selection.py` validates feature power x learning rate with
`shuffled` sampling and fixed epochs; its stored selections differ from both configs above.

## What is verified, and how

- **Dataset files:** the protocol's sha256 table is checked before loading and again after it
  (a file swapped mid-load is detected). Tests confirm the table names equal the core loader's
  `FILES`, that every file the core opens is hashed against the protocol at open time, and that no
  download is attempted. The loaded dataset carries its projection seed and verified hashes; the
  adapter refuses a dataset that does not match the protocol.
- **Evaluation semantics:** tests re-evaluate the core's weights with a benchmark-side
  implementation and match every row of the matrix; check that future-class output columns are
  exactly zero at every evaluation; check that each experience trains only on its own classes plus
  old-class replay with `eligible_classes` = classes seen so far; and characterise the core's masking
  rule with weights where an unmasked future class would win.
- **Recorded values:** `SequenceResult.run_config` holds seed, protocol name/version/fingerprint,
  dataset fingerprint + file hashes + projection seed, task order, condition, strategy, learning rate,
  feature power, epochs, batch size, sampling, memory per class and role; replay examples used per
  experience and core provenance are alongside. The adapter checks that the core echoes the
  configured learning rate, condition, strategy and sampling. (Serialised result records come with the
  runner phase.)
- **Provenance:** `core.commit_source` is `pip_vcs_install` (commit fetched from `remote_url`;
  `remote_verified: true`), `local_git`/`local_git_dirty` (read from a local checkout;
  `remote_verified: false`) or `unknown`. `matches_pin` compares against the pin in this package's
  metadata. A local commit never counts as remote-verified.

## Run modes and provenance

- `development` (default): any installed core is accepted. The result records `run_mode`, core and
  benchmark provenance, and `official_problems` (every reason the run would not qualify).
- `official`: refused with `ProvenanceError`, before any training, unless **all** hold:
  - the core commit equals the pin in this package's metadata **and its remote identity is
    verified**: a pip VCS install whose recorded URL is the pinned URL, or a **clean** local checkout
    whose remote URL is the pinned URL and whose remote-tracking refs contain the pinned commit;
  - the benchmark has a determinable commit (clean git checkout whose tracked file is this package, or
    a pip VCS install). A dirty benchmark checkout is rejected.
- "Dirty" means modified tracked files **or untracked non-ignored files**, for core and benchmark.
- An unverified remote is an **error**; there is no warning-only path (`RemoteUnverifiedWarning` and
  `PIN_REMOTE_STATUS` were removed). Remote verification is offline evidence read from pip/git.
  Limitation: a remote-tracking ref shows the commit was on the remote at the last fetch/push; it
  cannot detect a later force-push or deletion (`git ls-remote` would, but needs the network).
- Records carry `algorithm.core` (commit, source, remote URL, `remote_verified`,
  `remote_verification`, pin fields, problems) and `algorithm.benchmark` (commit, source, problems).

## Reference checks (full Split MNIST, real data)

| Config | Seed | Core record | Benchmark result |
| --- | --- | --- | --- |
| primary_reference | 0 | acc 0.6770214422462445, forgetting 0.0708057983859818 | identical |
| experimental | 3 | acc 0.7734281931972611, forgetting 0.09086581818480258 | identical |

## Running

    pip install -e .                       # installs the pinned core from Git
    python3 -m unittest discover -s tests                                   # fast suite (also works with pytest)
    BENCH_SLOW=1 PYTHONPATH=tests python3 -m unittest test_full_split_mnist -v   # real data, ~45 s

Place the four MNIST `.gz` archives in `data/mnist/` (git-ignored). The benchmark never downloads.
Required files and sha256 (from `configs/split_mnist_cil_v1.json`, enforced before any loading):

    train-images-idx3-ubyte.gz  440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
    train-labels-idx1-ubyte.gz  3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
    t10k-images-idx3-ubyte.gz   8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6
    t10k-labels-idx1-ubyte.gz   f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6

Check them with `python3 -m polynomial_cl_bench.verify_data --data-dir data/mnist` (exit 0 only when all
four match). The files are the standard MNIST archives under exactly these names (not decompressed).

## Open items / limitations

- The pinned commit has not been confirmed on the GitHub remote by this project (no network when
  written). An official run now proves it per installation (see above); `git ls-remote` remains the
  stronger check before releasing.
- Test data and replay indices are handled inside the core; the adapter can only observe them.

## Phase 2 status (ER-MLP)

Implemented (all tested against the real core, `python3 -m unittest discover -s tests`):

- `plan.py`: one `ExperiencePlan` per seed from the core's public `make_task_plan`; passed to every
  adapter (`run_sequence(..., plan)`) so experiences and replay exemplars are identical by construction.
  Plans are read-only, fingerprinted, and refused if seed, memory budget, task order or labels differ.
- `datasets.validation_view`: the core's `holdout_split`, validation part placed in the `test_*` slots;
  the real test arrays are not referenced. Results record `evaluation_split`.
- `sampling.py`: the core's shuffled and class-balanced batching reproduced draw for draw (the core
  only exposes it inside `fit_task`). `tests/test_sampling_equivalence.py` checks it against the core's
  private batcher, against batches captured inside a real `fit_task`, end to end (a bias-free linear model
  reproduces `fit_task`'s weights) and via the core's own `sampling_by_task` record.
- `algorithms/er_mlp.py`: ER-MLP, concatenated-replay form, masked softmax over introduced classes,
  plain SGD. `hidden` and `learning_rate` are required arguments; there are no defaults.
  `replay=False` exists only as the separate role `ablation_no_replay`.
- `tuning.py`: validation-only selection (seeds 100/101, disjoint from the evaluation seeds, accuracy
  first; ties: lower mean forgetting, then grid order). Failed/diverged runs are kept and make a
  configuration ineligible. Same function for ER-MLP and the polynomial reference.
- Official-run provenance rules (above).

Validation tuning (`python3 -m polynomial_cl_bench.tune --grid <grid> --out results/<name>.json`):

- `configs/er_mlp_grid.json` (approved): H in {64, 128, 256} x lr in {0.01, 0.03, 0.1, 0.3, 1.0}, one ReLU
  hidden layer, plain SGD (no momentum, no weight decay), class-balanced sampling, 200 replay exemplars per
  class, selection seeds 100/101, 30 runs. Selection: candidates within 0.005 of the maximum mean final
  accuracy; among them the lowest mean forgetting (exact ties: higher accuracy, then grid order).
- `configs/polynomial_grid.json` (approved, unchanged): lr in {0.01, 0.03, 0.1, 0.3, 1.0} x
  feature_power in {0, 0.5, 1}, expanding/ordinary, same split, seeds, budget and rule. The runner refuses
  any grid whose `approval_status` is not `approved`. The Gaussian-input step-multiplier computation behind the
  choice of powers was preliminary and is not evidence about MNIST.
- The runner writes every run (failed and divergent ones included) to JSON, the complete validation table to
  Markdown beside it, and never overwrites an existing result. `--mode official` applies the unchanged
  provenance rules; a development-mode result records its provenance and `official_eligible`.

`python3 tools/mutation_check.py` (Phase 1-2 code) breaks the source in 27 ways and checks that the tests notice
(26 caught, 1 empirically equivalent over the tested cases (not proved), 0 missed).

Not run: tuning on real MNIST (the archives were not available), so no hyperparameter has been selected.

## Phase 3: reliable comparisons

Workflow (every step is a separate command; nothing is rerun by a later step):

    # 1. validation-only tuning, one per algorithm (Phase 2)
    python3 -m polynomial_cl_bench.tune --grid configs/er_mlp_grid.json --out results/tuning_er_mlp.json
    python3 -m polynomial_cl_bench.tune --grid configs/polynomial_grid.json --out results/tuning_polynomial.json
    # 2. the experiment: every algorithm x every protocol seed, one fresh process per run
    python3 -m polynomial_cl_bench.runner --spec configs/experiment_tuned_comparison.json \
        --data-dir data/mnist --out results/tuned_comparison --reproduce-check
    # 3. interrupted, or some runs failed? continue without mixing anything
    python3 -m polynomial_cl_bench.runner --spec ... --out results/tuned_comparison --resume [--retry-failed]
    # 4. validate + aggregate (also done at the end of step 2)
    python3 -m polynomial_cl_bench.summarize --dir results/tuned_comparison

Output directory (git-ignored): `manifest.json` (resolved experiment + fingerprint), `runs/<label>__seed<k>.json`
(one record per run; archived earlier attempts as `.attempt<n>.json`), `reproducibility/`, `summary.json`,
`summary.md`.

**Result records (schema 1.0).** Each record holds algorithm + resolved configuration + hyper-parameter
provenance, dataset and protocol identity, seed, plan fingerprint, training budget, memory policy with
the budget and actual replay use per experience, the accuracy matrix and every metric (all recomputed by
this package from the matrix), resources, environment (Python, numpy, platform, thread settings, **core
and benchmark commits**), timestamps, status and error. `null` means not measured, never zero. A record is
re-validated on write and on read: run key, every metric, matrix shape/NaN pattern, memory <= budget.

**Resources.** Measured by the runner around `run_sequence`, identically for every algorithm: wall time,
CPU time, and peak RSS *during the run* (Linux: the kernel's peak counter is reset just before the run;
elsewhere `null`). Peak memory is an absolute RSS and includes the loaded dataset, so
`rss_before_run_mb` is stored beside it; with process isolation that baseline is the same for every
algorithm. Training/inference time and parameter count are adapter-reported (`null` for the polynomial
core, which cannot separate them). Wall time includes in-loop evaluation and, for the polynomial adapter,
feature construction. Runs are sequential by design (concurrency would distort timings). Per-class
accuracy is `null`: adapters return an accuracy matrix only.

**Failures are results.** Adapter exceptions, divergence, malformed matrices, an adapter that did not use the
shared plan, replay above the budget, worker crashes and timeouts all become records with status
`failed`/`diverged`/`crashed`/`timeout`. Aggregation reports completed/failed/missing counts per algorithm
and lists every failure; statistics use completed runs only and state their `n`. An unreadable or
hand-edited record is reported as a problem and counted as missing, not skipped. Only an official-mode
provenance refusal or Ctrl-C stops an experiment.

**Resume without mixing.** Resuming requires the same resolved experiment (manifest fingerprint over
protocol, all algorithm configurations, seeds, mode, split, isolation), matching run keys, and the same
core/benchmark commits; otherwise it is refused. Failed runs are re-run only with `--retry-failed`, and
the old record is kept.

**Tuned configurations.** An algorithm entry with `tuning_result` takes its hyper-parameters from that
validation-only tuning file, after checking its protocol fingerprint, validation split, that it selected
something, and that its seeds are disjoint from the experiment's. Its sha256 and selection details are
recorded in the run records.

**Statistics.** Mean, sample std (ddof=1), 95% t-interval, n; paired differences per seed (a - b) with
counts of pairs where each side is higher. Descriptive only: seeds vary initialisation, sampling and
replay selection, **not** the data split; with few seeds intervals are wide and assume roughly normal
differences. No p-values, no "better/worse" verdicts.

**Experiment validation** (`validation.py`) separates *integrity problems* (must be zero: a record
contradicting the manifest, different plans for one seed across algorithms, different dataset arrays,
mixed commits, missing/unexpected runs, a disagreement with the core's own reported metrics) from
*publication blockers* (not an error, but the comparison is not publishable: not official mode,
unverified provenance, failed runs, non-test split, seeds differing from the protocol, hyper-parameters not
from validation tuning, reproducibility check missing or not exact). `summary.md` prints both.

## Phase 3 status and what is unverified

Verified (core-free tests with fake adapters, `tests/test_runner.py`, `tests/test_results_and_aggregation.py`):
record schema and tamper detection, atomic writes, multi-algorithm/multi-seed execution, plan sharing,
failure/divergence/crash/timeout records, resume/retry/mismatch refusals, aggregation numbers, paired
comparisons, experiment validation, tuning-result checks. `python3 tools/mutation_check_phase3.py` breaks
the new code in 22 ways; all 22 are noticed by these tests.

**Not verified by the author:** the Phase 3 code was written in an environment with no network, so the
pinned core could not be installed and no real MNIST was available. `tests/test_runner_integration.py`
(real worker subprocess, ER-MLP, reproducibility, peak memory; and the polynomial adapter through the
runner including the agreement check against the core's own metrics) was run only against a minimal
stand-in for the core's API written for that session, **not** the real core. Therefore:

- the polynomial adapter through the runner (`PolynomialThroughTheRunner`) has never executed;
- `PolynomialAdapter.from_dict` is a small refactor of `from_file` (same behaviour, not exercised against the core);
- no full-Split-MNIST comparison, no tuning on real data, and no timing/memory numbers on real runs exist.

Run `python3 -m unittest discover -s tests` and `tests/test_runner_integration.py` with the real core
installed before relying on any of this. Until tuning has been run on real MNIST, the configs in
`configs/experiment_tuned_comparison.json` cannot be used (the tuning result files do not exist), which is
intended: `experiment_polynomial_reference.json` is the only runnable experiment and is correctly marked
non-publishable.

Known limits: per-class accuracy and a separate polynomial training/inference time are unavailable; peak
memory is `null` off Linux; the reproducibility check re-runs one seed per algorithm (exact equality, so
multi-threaded BLAS may report a mismatch: pin `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`
to 1 and repeat).

## Phase 5: analysis and paper

Phase 5 consumes validated results only; it never trains anything except through the explicit
comparison driver.

    # 1. best settings by criterion from the core's per-degree study (stdlib only)
    python tools/analyze_per_degree_study.py --results-dir <core>/results/per_degree \
        --out-json results/analysis/per_degree_analysis.json \
        --out-md results/analysis/per_degree_analysis.md --out-csv results/analysis/per_degree_candidates.csv
    # 2. re-run those settings and the ER-MLP under one protocol/machine (resumable)
    python tools/run_comparison_study.py --analysis results/analysis/per_degree_analysis.json \
        --core-results <core>/results/per_degree --out-dir results/comparison
    # 3. tables, figures and number macros for the paper, then compile paper/main.tex
    python tools/build_paper_assets.py --analysis results/analysis/per_degree_analysis.json \
        --comparison results/comparison --out paper

Notes: the core's `main` was refactored (`class_incremental` -> `continual` + `data`); the
benchmark's imports were updated accordingly (core untouched). The unit tests under `tests/` still
import the old module path and have **not** been ported or re-run. The ER-MLP grid used by
`run_comparison_study.py` is a superset of the approved `configs/er_mlp_grid.json`.
