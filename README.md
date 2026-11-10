# polynomial_cl_bench

An independent benchmark for continual-learning algorithms. It evaluates the polynomial
continual-learning method from [polynomial_cl_core](https://github.com/kobros-tech/polynomial_cl_core)
against competing approaches under one protocol, with a shared experience/replay plan per seed,
validation-only hyper-parameter selection and provenance-carrying result records.

Full documentation lives in [`docs/`](docs/) (Sphinx, reStructuredText); see
[Documentation](#documentation) below. This README is a short entry point. The detailed,
verified account of what works is the documentation's *Implementation and verification status*
page.

## Research motivation

Method-level claims about the polynomial approach need to be tested against the actual
implementation, under conditions that do not favour it: the same data, the same experiences and
replay exemplars, the same training budget, hyper-parameters chosen on validation data only, and
every result tied to the exact code that produced it. The benchmark does not judge the method; it
produces numbers and the evidence needed to trust them.

## Relationship to polynomial_cl_core

`polynomial_cl_core` (Python package `polynomial_cl`, "the core") owns the method. This repository
owns the evaluation and **never re-implements** the polynomial model, features, training or
replay. The dependency is one-way, `polynomial_cl_bench -> polynomial_cl`, and the core is pinned
in `pyproject.toml` to a full commit SHA (never a branch), currently
`2e806d7a3437e930256eb49ab7a3ec5e650a1110`.

## Architecture

```text
protocol -> verified data -> per-seed shared plan -> adapter.run_sequence -> accuracy matrix
         -> metrics, resources, provenance -> validated JSON record -> aggregation and validation
```

Metrics are recomputed by this package from the accuracy matrix; an adapter cannot report its
own. Each run executes in a fresh process. See `docs/architecture.rst`.

## Supported algorithms

| Algorithm | Implementation |
| --- | --- |
| `polynomial_cl` | the core's own `run_condition` (condition `expanding`, strategy `ordinary`) |
| `er_mlp` | benchmark-owned NumPy experience-replay MLP |

No other algorithm is implemented. Neither shipped polynomial configuration is claimed optimal.

## Supported datasets

Split MNIST only (five two-class experiences, one 10-way head, 16-d projection). Other datasets
are not implemented. The benchmark never downloads data.

## Benchmark protocol

`configs/split_mnist_cil_v1.json`, version 1.0: seeds 0-4, 3 epochs, batch size 128,
class-balanced sampling, 200 replay exemplars per class, evaluation after every experience over
the classes seen so far, validation hold-out of 10% of the training data (seed 123). The
protocol is fingerprinted into every record. Changing it requires a version bump.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # benchmark + core at the pinned commit + pytest, ruff, scipy
pip install -e ".[docs]"     # ... or the documentation toolchain
```

Python 3.9 or newer. Reinstall after changing the pin: the pin is read from installed metadata.
Place the four standard MNIST `.gz` archives (not decompressed) in `data/mnist/` and run
`python -m polynomial_cl_bench.verify_data --data-dir data/mnist`.

## Quickstart

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m polynomial_cl_bench.runner \
    --spec configs/experiment_polynomial_reference.json \
    --data-dir data/mnist --out results/polynomial_reference --reproduce-check
```

Five seeds of the untuned reference configuration on real Split MNIST. The summary correctly
reports the comparison as *not publishable* (development mode, untuned configuration).

## Hyperparameter tuning

`python -m polynomial_cl_bench.tune --grid configs/<algorithm>_grid.json --out ...` runs a grid on
the validation hold-out only (selection seeds 100 and 101, disjoint from evaluation seeds), records
every run including failures, and never overwrites a result. Development-mode tuning has been run on real
MNIST for both algorithms (30 of 30 runs each); those records carry no core or benchmark commit and
are not official-eligible. **No official-mode tuning has been run.** The records live in `results/`,
which is not versioned.

## Model selection objectives

`python -m polynomial_cl_bench.selection` ranks tuned configurations under four objectives:
`max_accuracy`, `min_forgetting`, `min_time` and `best_tradeoff`, with an optional accuracy floor
(default 0.02) so that models that barely learn do not win the forgetting and time objectives.
Validation results only; missing values are never zero. Not yet applied to real tuning results.

## Reproducibility

Records carry protocol, dataset and plan fingerprints, the core and benchmark commits, and the
thread settings. `--reproduce-check` re-runs a seed in a fresh process and checks bit-identity.
`--mode official` is refused unless the core is the remote-verified pin and the benchmark checkout
is clean. See `docs/reproducibility.rst`.

## Testing

```bash
BENCH_REQUIRE_CORE=1 pytest -q                    # unit + real-core synthetic integration
BENCH_REQUIRE_PINNED_CORE=1 BENCH_REQUIRE_CORE=1 pytest -q   # after a clean install from the pin
BENCH_SLOW=1 BENCH_MNIST_DIR=data/mnist pytest tests/test_full_split_mnist.py   # real MNIST
ruff check . && ruff format --check .
```

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
sphinx-build -W --keep-going -b latex docs docs/_build/latex && make -C docs/_build/latex   # PDF
```

CI builds all three with warnings as errors.

## Producing the results

The results the paper and the documentation quote are produced by two GitHub workflows started by
hand, `paper-results` (the core's per-degree study as independent parallel jobs, then the
analysis, the comparison study and the regenerated paper assets) and `benchmark-results` (the
runner pipeline: tuning, tuned comparison, reproducibility check). They commit nothing; results
are uploaded as artifacts. See `docs/analysis_and_paper.rst`.

## Research paper

`paper/index.rst` is the canonical source of a **draft** manuscript (reStructuredText, built with
Sphinx; HTML, PDF and Word versions). Its tables and numbers come from generated assets, and
`tests/test_paper.py` checks that the migration from the earlier LaTeX (`paper/main.tex`, kept as
the reference) lost nothing. It has not been validated against executed benchmark output, and its
raw results are not in this repository; do not treat it as verified. Claim-by-claim validation is
pending. Build: `python tools/build_paper.py`; details in `paper/README.rst`.

## Status

Levels follow the documentation: implemented, unit-tested, integration-tested, core-verified,
real-data-verified, reproducibility-verified, publication-ready.

| Area | Status |
| --- | --- |
| Protocol, data verification and loading, plan, polynomial adapter, runner, records, aggregation | Real-data-verified: reference experiment executed on real MNIST; reproduces the core's stored records |
| Reproducibility check | One seed re-run bit-identical, single-threaded BLAS, same machine |
| ER-MLP, validation tuning, model selection | Run on real MNIST in development mode only (not official-eligible); **not run in official mode** |
| Tools in `tools/` | Implemented; entry points load; not run |
| Codabench / OpenML | Not implemented (planned) |
| Anything publication-ready | **None** |

## Current limitations

The only tuned comparison is a development-mode run that is reported as not publishable; the
polynomial core is
observed through one call, so some checks are indirect; seeds do not vary the data split; no
significance tests are computed; reproducibility has not been checked across machines; two
descriptive strings in the configs still name the core's pre-refactor module path (changing the
protocol's `dataset.source` needs a version bump). Details: `docs/limitations.rst`.
