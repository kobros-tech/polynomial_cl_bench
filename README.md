# polynomial_cl_bench

Independent benchmark environment for continual-learning algorithms. Dependency direction is
strictly `polynomial_cl_bench -> polynomial_cl`; the core package is installed from a pinned
commit and is never modified or patched.

**Status: Phase 1 (partial); Phase 2 (ER-MLP) started, not complete.** Implemented: protocol config,
dataset verification/loading, adapter contract, polynomial adapter, benchmark-owned metrics, and the
Phase 2 official-run provenance rules (below). **Not yet implemented:** the ER-MLP competitor, the
shared experience/replay plan, sampling-equivalence tests, validation tuning, runner, result
records, platforms. See "Phase 2 status" at the end.

## Layout

    configs/split_mnist_cil_v1.json            protocol (algorithm-independent)
    configs/polynomial_expanding_ordinary.json            primary_reference (lr 0.1)
    configs/polynomial_expanding_ordinary_per_degree.json experimental (lr [0.75, 1.5, 1.5, 6.0])
    src/polynomial_cl_bench/protocol.py        validated Protocol, canonical JSON, fingerprint
    src/polynomial_cl_bench/datasets.py        sha256 verification, core loader with download=False
    src/polynomial_cl_bench/metrics.py         accuracy-matrix metrics (independent of the core)
    src/polynomial_cl_bench/algorithms/base.py SequenceAdapter contract, SequenceResult
    src/polynomial_cl_bench/algorithms/polynomial.py  wrapper around core run_condition

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

`python3 tools/mutation_check.py` breaks the source in 27 ways and checks that the tests notice
(26 caught, 1 empirically equivalent over the tested cases (not proved), 0 missed).

Not run: tuning on real MNIST (the archives were not available), so no hyperparameter has been selected.
