# Changelog — GP-UQ-paper code

All code changes to the Python analysis pipeline are logged here.
Format: date · file · what changed · why.

---

## 2026-09-25

### config.py
- `DATA_DIR`: hardcoded absolute local filesystem path → `Path(os.environ.get("DOPAE_DATA_DIR", _HERE / "data" / "raw"))`
  - Removes a personal filesystem path ahead of publishing the repo on GitHub; default now resolves to a repo-local, git-ignored folder and can be overridden per-machine via an env var instead of editing the file.
- `DATA_DIR_SAMPLE`, `PICKLE_PATH`: pointed at an external sibling project folder (outside this repo) → repo-local `data/sample/`, `data/processed.pkl` (both env-var overridable via `DOPAE_DATA_DIR_SAMPLE` / `DOPAE_PICKLE_PATH`)
  - Decouples the public repo from an external, unpublished folder that itself contains real individual-level data.

### Repository organisation (not code, for reference)
- Moved `algorithm_description.md` → `docs/algorithm_description.md`; added `README.md`, `data/README.md`, `.gitignore`.
- `.gitignore` excludes `outputs/curve_fits/`, the four individual-level result CSVs (`experiment_results.csv`, `reference_results.csv`, `sample_analysis.csv`, `bootstrap_retest_results.csv`), and `outputs/fig1_example_curve.*` (labels its panel with a real individual ID — see `figures.py`, `figure1_example_curve`). Aggregate figures (fig2-fig5, `selection_overlap`, `algorithm_flowchart`, `fig_bootstrap_retest`) remain tracked.
  - Data privacy: only code and aggregated results are published publicly, not individual-level outputs.

---

## 2026-06-08

### experiment.py
- `N_REPEATS`: 10 → 5
  - 5 repeats is sufficient for the SE of mean RMSE at the CI widths reported here (<0.1 dB); halves the random-strategy compute cost.
- `N_SAMPLES_SPARSE`: 200 → 100
  - MDC uses sample SD (SE ≈ 0.07–0.14 dB at n=100 vs n=200); CI percentiles have SE <0.6 dB — small relative to reported CI widths of 12–26 dB. No impact on reported numbers.
- MDC comment: updated to "estimated MDC lower bound: standard formula applied to single-session posterior SD"
  - Reflects the corrected framing that this is not an empirical test-retest MDC.

### figures.py
- Figure 3 y-axis labels: "MDC (dB SPL)" → "Estimated MDC (dB SPL)"
- Figure 3 subplot titles updated to include "est. MDC"
  - Consistent with manuscript terminology correction.

### plot_algorithm.py
- Algorithm diagram text: "MDC = 1.96√2·sigma" → "est. MDC = 1.96√2·σ̂ (lower bound)"
  - Consistent with manuscript terminology correction.

---

## 2026-06-09

### gp_fitting.py
- Added `fit_gp_conditioned(l2, dp, noise_var, n2, mean_type, hyperparams, n_samples, n_detected)`
  - Conditions a GP on new observations using FIXED hyperparameters (no Adam optimisation).
  - Fast path for bootstrap: leverages existing `_train_gp` fixed-hyperparams branch.
  - Intended for bootstrap_retest.py only; not used in the main experiment.

### bootstrap_retest.py (new file; numpy fast path added same session)
- Residual bootstrap test-retest analysis for knee and max-compression features.
- N_BOOTSTRAP=500 reps per curve, parallelised with ProcessPoolExecutor (6 workers).
- Added `_matern32_kernel()` helper and precomputed weight matrix
  `W = K_sx @ K_noisy⁻¹` once per curve; bootstrap loop reduced to a
  numpy matrix-vector product (~5 ms/rep vs ~367 ms with GPyTorch).
- Removed `fit_gp_conditioned` from the bootstrap inner loop; now pure numpy.
- Full run (146 curves × 500 reps) completed in ~6.5 min.
- Results: knee bootstrap MDC 13.31 dB, knee GP-est. MDC 15.80 dB;
           mc bootstrap MDC 8.29 dB, mc GP-est. MDC 8.86 dB.
- Outputs: outputs/bootstrap_retest_results.csv + outputs/fig_bootstrap_retest.pdf/png.

---

## 2026-06-12

### synthetic-DPOAE-dataset/synthetic_data.py (new file)
- Gorga three-segment piecewise-linear DPOAE I/O generator for 500 normal-hearing curves
  × 23 L2 levels (30–74 dB SPL, 2 dB step).
- Parameter distributions from Bhagat (2014): BP1~N(48.3,3.5²), BP2=BP1+N(15,5²),
  c~N(0.20,0.10²), G~N(-42,6²), noise_sd~N(2.0,0.5²).
- Left-censors observations below NOISE_FLOOR=-20 dB SPL (Reavis et al. 2010 criterion).
- Outputs: outputs/synthetic_curves.npz (l2_grid, curves, curves_clean) and
  outputs/synthetic_params.csv.
- Fixed ASCII-only print strings to avoid cp1252 encoding error on Windows.

### synthetic-DPOAE-dataset/validate_synthetic.py + plot_synthetic.py (revised 2026-06-13)
- Added gp_twoslope as a second mean type alongside gp_linear.
- validate_synthetic.py: MEAN_TYPES = ("linear", "twoslope"); fits two reference
  GPs per curve (separate hyperparams per mean type); model column is now
  "gp_linear" or "gp_twoslope".
- plot_synthetic.py: solid lines = linear, dashed = twoslope; shared colour per
  strategy; legend uses strategy (colour) + mean type (linestyle) convention.

### synthetic-DPOAE-dataset/synthetic_data.py (revised 2026-06-13)
- Model changed from three-segment Gorga to two-segment: removed BP2 and
  segment 3 (return to slope=1 above BP2). Rationale: BP2 ≈ 63 dB SPL lies at
  or above the top of the clinical L2 range (65 dB SPL), so segment 3 is
  never observed in the real DOPAE-Dosage dataset.
- L2 grid changed from 30-74 dB (23 levels, 2 dB step) to 20-65 dB
  (10 levels, 5 dB step), matching config.py L2_LEVELS exactly.
- BP2 and DELTA parameters removed from generator and params CSV.
- `gorga_curve()` simplified to two-segment; BP2 argument removed.

### synthetic-DPOAE-dataset/validate_synthetic.py (revised 2026-06-13)
- K_VALUES comment updated (k=10 now = full 10-level reference dataset).

### synthetic-DPOAE-dataset/plot_curves.py (revised 2026-06-13)
- Removed BP2 vertical lines and DELTA imports (three-segment model dropped).
- Parameter histogram reduced from 4 panels (BP1, BP2, c, noise_sd) to 3
  (BP1, c, noise_sd).

### synthetic-DPOAE-dataset/plot_synthetic.py (new file)
- Two publication-ready figures from outputs/synthetic_results.csv:
  fig_synthetic_coverage (4-panel: coverage + CI width vs k for knee and c)
  and fig_synthetic_rmse (2-panel: RMSE vs k, GP linear vs cubic).
- Saved as both PDF and PNG; --show flag for interactive inspection.

### synthetic-DPOAE-dataset/validate_synthetic.py (new file)
- GP pipeline coverage validation: runs fit_gp (full) + fit_sparse (k=3..10,
  equalspaced/random/active) + fit_cubic_sparse on each synthetic curve.
- Records 95% CI coverage and width for knee (BP1) and slope_high (c) vs ground truth.
- Noise2 per-point set as dp-3.5 dB so all non-NaN observations are "detected"
  (SNR=3.5 > DETECTION_DELTA_DB=3) and noise_var≈2.83 dB² (close to synthetic σ²=4).
- Parallelised with ProcessPoolExecutor (N_WORKERS=4).
- Outputs: outputs/synthetic_results.csv.
- Bugfix (same session): ground truth for slope_high changed from raw Gorga c to OLS
  slope of the clean Gorga curve over 45-65 dB (matching _SLOPE_HIGH_RANGE in
  gp_fitting.py). Raw c differs from OLS slope_high because the 45-65 dB window
  includes Gorga segments 1 and 3 (slope=1) in addition to segment 2 (slope=c),
  causing the GP CI to never cover the wrong target (c coverage = 0).

---

## 2026-08-25

### figures.py
- `figure1_example_curve`: panel (a) in-axes legend -> shared figure legend below panels (a)+(b) (`fig.legend`, `ncol=4`, fontsize 7)
  - The 7-entry legend at `loc="upper right"` sat on top of the rising I/O curve; moving it outside the axes frees the whole data area.
- `figure1_example_curve`: `figsize` (W2, 2.8) -> (W2, 3.05), GridSpec now `top=0.90, bottom=0.30`
  - Reserves the strip below the panels for the shared legend without shrinking the plotted data.
- `figure1_example_curve`: panel (c) footnote text removed; explanation folded into the legend as a grey dotted "Cubic estimate" proxy handle
  - The centred footnote overflowed left under panel (b)'s x-label.
- `figure1_example_curve`: panel (c) y-limit x1.45 headroom; feature `axvline` calls given `ymax=0.68`
  - Keeps the histogram bars and the mean/point-estimate lines clear of the upper-left legend.

### figures.py (figure 2)
- `figure2_rmse_vs_sparsity`: both `ax.legend(loc="upper right")` -> new `_legend_below()` helper (legend under each axis, `ncol=2`, fontsize 7)
  - Panel (a)'s legend overlapped the cubic/two-slope peak near k=4-5 and panel (b)'s overlapped the random-strategy cubic curve across k=5-9.
- `figure2_rmse_vs_sparsity`: `figsize` (W2, 2.8) -> (W2, 3.6); `subplots_adjust` wspace 0.40 -> 0.32, left 0.10 -> 0.09, bottom 0.15 -> 0.36
  - Makes room for the below-axis legends; wider panels since the legends no longer compete for space.
- `MODEL_STYLE["cubic"]` label: "GP cubic mean" -> "GP cubic"
  - Shorter label keeps the two-column legend compact.

### figures.py (figure 3)
- `figure3_feature_uncertainty`: both `ax.legend(loc="upper right")` -> `_legend_below()` (legend under each axis, `ncol=2`, fontsize 7)
  - Panel (a)'s legend was crossed by the GP two-slope curve at k=6-9 and panel (b)'s by both knee curves at k=3-9.
- `figure3_feature_uncertainty`: `figsize` (W2, 2.8) -> (W2, 3.6); `subplots_adjust` wspace 0.40 -> 0.32, left 0.10 -> 0.09, bottom 0.15 -> 0.36
  - Matches figure 2 geometry so the two panels-plus-legend figures form a set.
- Note: regenerating also picked up the 2026-06-08 axis/title edits ("MDC" -> "Estimated MDC"), which had never been re-rendered.

### figures.py (output formats)
- `_save`: extensions ("pdf", "png") -> ("pdf", "png", "svg")
  - SVG requested for editable vector figures; applies to all figures on next regeneration.

---

## Manuscript changes (not code, for reference)

### manuscript/submission/main.tex + reprint.tex
- MDC → "estimated MDC" throughout; new fifth Limitation paragraph added explaining single-session lower bound.
- Active vs equalspaced RMSE framing corrected: equalspaced matches/beats active at k≤5; active better at k≥7. Removed incorrect claim that active advantage is "most pronounced at k=3–4".
- Discussion §4.2 rewritten to accurately describe the retrospective constraint.
