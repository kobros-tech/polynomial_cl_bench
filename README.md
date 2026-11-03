# polynomial_cl_bench

Independent benchmark environment for continual-learning algorithms. Dependency direction is
strictly `polynomial_cl_bench -> polynomial_cl`; the core package is installed from a pinned
commit and is never modified or patched.

**Status: Phase 1 (partial).** Implemented: protocol config, dataset verification/loading,
adapter contract, polynomial adapter, benchmark-owned metrics. Not yet implemented: runner,
result records/serialisation, competitors, resource measurement beyond wall time, platforms.

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
|---|---|---|---|
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

## Run modes and the pinned core

- `development` (default): any installed core is accepted. The result records `run_mode`, the core
  commit/source and `official_problems` (every reason the installation would not qualify).
- `official`: refused with `ProvenanceError`, before any training, unless the core commit equals the
  pin in this package's metadata and the checkout has no uncommitted tracked changes (an unknown or
  mismatched commit is also refused). `PolynomialAdapter(..., mode="official")`.
- **Remote verification is separate and currently `unverified`** (`provenance.PIN_REMOTE_STATUS`).
  Until someone confirms the pinned commit on the GitHub remote and changes that constant, every
  official run emits `RemoteUnverifiedWarning` and every record carries `pin_remote_status` and
  `remote_verified` (false for a local checkout, true only for a pip VCS install).

## Reference checks (full Split MNIST, real data)

| Config | Seed | Core record | Benchmark result |
|---|---|---|---|
| primary_reference | 0 | acc 0.6770214422462445, forgetting 0.0708057983859818 | identical |
| experimental | 3 | acc 0.7734281931972611, forgetting 0.09086581818480258 | identical |

## Running

    pip install -e .                       # installs the pinned core from Git
    python3 -m unittest discover -s tests                                   # fast suite (also works with pytest)
    BENCH_SLOW=1 PYTHONPATH=tests python3 -m unittest test_full_split_mnist -v   # real data, ~45 s

Place the four MNIST `.gz` archives in `data/mnist/` (git-ignored). The benchmark never downloads.

## Open items / limitations

- The pinned commit has **not** been verified on the GitHub remote (no network access when written).
  Run `git ls-remote https://github.com/kobros-tech/polynomial_cl_demo.git` and confirm the commit is
  reachable from a pushed branch before releasing. If the repo is private, installs need credentials.
- Test runs so far used a local editable core (`local_git`, `matches_pin: true`), not a pip VCS install.
- Official runs accept a clean local checkout at the pin (`remote_verified: false`); requiring a pip VCS install is an open choice.
- Test data and replay indices are handled inside the core; the adapter can only observe them.
