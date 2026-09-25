# GP-UQ: Uncertainty-Quantified GP Regression for DPOAE I/O Functions

**Status: ongoing research project.** Core pipeline (GP fitting, sparse-acquisition
experiment, bootstrap validation, figures) is implemented and runs end-to-end;
some analyses (e.g. a proper censored-likelihood noise model) are noted as
future work below.

## Description

A Gaussian Process (GP) regression pipeline for **DPOAE input/output (I/O)
growth functions** — the curves used to characterise cochlear compression
from otoacoustic emissions. The project addresses two questions:

1. **How few stimulus levels can we measure and still recover the I/O curve
   and its features (compression knee, max-compression level, slopes) with
   usable uncertainty bounds?** (`experiment.py`)
2. **Does an active (variance-greedy) level-selection strategy outperform
   naive random/equally-spaced sampling?** (`experiment.py`, `analyse_selection.py`)

Every curve is fit with a Matérn-3/2 GP on a fixed parametric mean (linear,
cubic, two-slope, quadratic, constant), with heteroscedastic noise driven by
per-point SNR and soft left-censoring for sub-noise-floor observations. A
leave-*k*-out design compares sparse fits against a full reference fit,
evaluating curve RMSE, feature bias, and 95% CI coverage. A separate
residual-bootstrap analysis (`bootstrap_retest.py`) estimates a two-session
minimum detectable change (MDC) for the extracted features, to benchmark
against the single-session GP posterior SD.

See [`docs/algorithm_description.md`](docs/algorithm_description.md) for the
full technical specification (kernel, noise model, feature extraction,
selection strategies, evaluation design).

## Design rationale

- **Fixed parametric mean + GP on residuals** — keeps sparse fits stable at
  low *k*; the GP only has to explain what the mean function misses.
- **Left-censoring instead of discarding sub-noise-floor points** — dropping
  them biases feature estimates upward (Helsel, 2012); they're retained as
  soft-censored observations with large noise variance.
- **Reported MDC is explicitly framed as an estimated single-session lower
  bound**, not a validated test-retest MDC — there is no retest data in this
  dataset. The bootstrap analysis exists to sanity-check how close that
  lower bound gets to a bootstrapped two-session estimate.

## Repository structure

```
.
├── config.py                  # paths, stimulus protocol constants
├── load_data.py                # raw CSV / pickle -> OAEs list-of-dicts
├── gp_fitting.py                # GP model, sparse fitting, feature extraction
├── experiment.py                # leave-k-out sparse-acquisition experiment
├── bootstrap_retest.py          # residual-bootstrap two-session MDC estimate
├── analyse_selection.py         # active vs equalspaced selection overlap
├── figures.py                   # all publication figures
├── plot_algorithm.py            # pipeline flowchart diagram
├── plot_curves.py               # per-curve GP fit diagnostics
├── docs/
│   └── algorithm_description.md # full technical specification
├── data/
│   └── README.md                # expected data layout (no data included)
└── outputs/                      # figures + result CSVs (see note below)
```

## Data

No raw or individual-level data is included in this repository. `outputs/`
ships only figures aggregated across the full dataset (e.g.
`fig2_rmse_vs_sparsity`, `fig3_feature_uncertainty`, `fig5_cohort_features`);
individual-level CSVs, curve-fit plots, and a single-curve example figure
are excluded via `.gitignore`.

To run the pipeline on your own data, see [`data/README.md`](data/README.md)
— it documents the expected CSV format and the `DOPAE_DATA_DIR` environment
variable used to point at a local data folder without editing `config.py`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires Python 3.10+. Core dependencies: `numpy`, `pandas`, `torch`,
`gpytorch`, `scipy`, `pwlf`, `matplotlib`, `seaborn`.

## Usage

Each script is runnable standalone; outputs go to `outputs/`.

```bash
# 1. Sanity-check data loading
python load_data.py

# 2. Per-curve GP fit diagnostics
python plot_curves.py                        # all curves
python plot_curves.py --id <id>               # single curve

# 3. Full leave-k-out experiment (parallel)
python experiment.py --smoke                  # single-curve test first
python experiment.py                          # full run (features + RMSE)
python experiment.py --fast --workers 4       # RMSE only, override worker count

# 4. Selection-strategy diagnostics
python analyse_selection.py

# 5. Residual-bootstrap two-session MDC
python bootstrap_retest.py --smoke
python bootstrap_retest.py

# 6. Figures
python figures.py --all
python figures.py --fig 1 --show              # single figure, interactive

# Pipeline overview diagram
python plot_algorithm.py
```

## Methods

| Component | Choice | Why |
|---|---|---|
| Kernel | Matérn-3/2 over stimulus level (L2) | Once-differentiable — matches smooth compression, not over-smooth like RBF |
| Mean function | Fixed parametric (linear primary; cubic as an overconfidence cautionary example) | GP models residuals only, stabilising sparse fits |
| Noise model | Heteroscedastic, SNR-driven; soft left-censoring below noise floor | Reflects real measurement uncertainty; avoids the truncation bias of discarding sub-floor points |
| Selection strategies | Random, equally-spaced, sequential active (max posterior variance) | Tests whether informed sampling beats naive designs under a fixed level budget |
| Evaluation | Leave-*k*-out RMSE, feature bias, 95% CI coverage vs. full-data reference | Standard sparse-approximation validation design |
| MDC | 1.96·√2·(single-session posterior SD), reported as an *estimated* lower bound | No retest data available — avoids overclaiming test-retest validity |

Full derivations and parameter values: [`docs/algorithm_description.md`](docs/algorithm_description.md).

## Limitations

1. Single-session data only — GP CI widths are not yet validated against
   real test-retest variability (the bootstrap analysis approximates this).
2. Active selection is constrained to a fixed set of pre-defined stimulus
   positions, limiting its advantage over a prospective fine-grid protocol.
3. Sub-noise-floor censoring uses a soft Tobit-style variance inflation;
   a proper censored likelihood in Pascal units is future work.
4. No population pooling — features are estimated per curve, appropriate
   for individual-level monitoring but not for population-level inference.

## Reproducibility

- `N_REPEATS = 5` (random-selection strategy) and `N_SAMPLES_SPARSE = 100`
  (posterior samples per sparse fit) were chosen as the smallest values
  giving SE below the reported CI-width precision — see
  [`CHANGELOG.md`](CHANGELOG.md) for the numeric justification.
- All parameter/behaviour changes to the analysis code are logged in
  [`CHANGELOG.md`](CHANGELOG.md) with the rationale, not just the diff.

## Author

Leny Vinceslas — UCL.
