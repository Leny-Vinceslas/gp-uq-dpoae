# Experiment Algorithm: Technical Description
## GP Regression for DPOAE I/O Functions: Sparse Acquisition & Feature UQ

---

## 1. Dataset

| Property | Value |
|---|---|
| Tasks | 4 per curve set: Left/Right × 1414 Hz / 4243 Hz |
| Total curves | 146 |
| Stimulus levels | 10 levels: L2 ∈ {65, 60, 55, 50, 45, 40, 35, 30, 25, 20} dB SPL |
| Primary tones | L1 = L2 + 10 dB; f2/f1 = 1.22 |
| Detection criterion | dp > noise2 + 3 dB (above-noise-floor) |
| Sub-noise-floor treatment | Retained as left-censored observations (not discarded) |

---

## 2. GP Model

### 2.1 Kernel

Matérn-3/2 over L2:

```
k(L2, L2') = σ_f² (1 + √3|L2 - L2'|/ℓ) exp(-√3|L2 - L2'|/ℓ)
```

- `ℓ` = length-scale, `σ_f²` = output variance
- Both optimised by maximising marginal log-likelihood (Adam, 150 steps)
- Matérn-3/2 chosen: once-differentiable sample paths match smooth cochlear compression

### 2.2 Mean Functions (5 types)

| Key | Formula | What GP explains | CI width |
|---|---|---|---|
| `cubic` | a·L2³ + b·L2² + c·L2 + d | Residuals only | Narrow |
| `twoslope` | s1·L2 + b (L2 ≤ bp); s2·(L2−bp) + y_bp (L2 > bp) | Departures from two-regime shape | Intermediate |
| `quadratic` | a·L2² + b·L2 + c | Curvature + residuals | Intermediate |
| `linear` | m·L2 + b | Compression shape + residuals | Wide |
| `constant` | c (learned) | Everything | Widest |

All parametric coefficients are fitted by OLS (or grid search for two-slope) before GP optimisation, then held fixed. The GP kernel models only what the mean leaves unexplained.

**Two-slope breakpoint search:** grid over [30, 58] dB in 29 steps; breakpoint chosen by minimum total OLS residual sum of squares with continuity constraint.

### 2.3 Noise Model (heteroscedastic)

```
σ²_noise(L2_i) = {
    clip(4·exp(-0.1·SNR_i) + 1, 1, 25)   if detected  (SNR = dp - noise2)
    36                                      if censored  (≈ Tobit approx., Gammelli 2022)
}
```

Units: dB². Detected points: variance decreases with SNR. Sub-noise-floor points: large fixed variance (soft left-censoring).

### 2.4 Posterior Sampling

- Reference fits (k=10): **300 samples** on a dense grid of 200 L2 values [17, 68] dB
- Sparse fits (k<10): **200 samples** on a grid spanning the selected levels

### 2.5 Monotone Filtering

Before feature extraction, posterior samples are filtered:
- Compute running mean of point-to-point differences over 15-step windows
- Reject samples where any window mean < −0.1 dB/step (physiologically implausible sustained decline)
- If fewer than 50% pass, filtering is skipped (safety fallback for extreme sparsity)

---

## 3. Feature Extraction

For each posterior sample, extract 4 features:

### 3.1 Compression-onset knee

Piecewise linear fit with continuous 2-segment model. Search breakpoints in **[25, 63] dB** at 57 equally-spaced positions (0.68 dB resolution). Breakpoint minimising total residual sum of squares is the knee estimate.

### 3.2 Maximum-compression level

L2 value where the posterior sample reaches its maximum amplitude in the **compressive region (L2 ≥ 40 dB)**. Uses the GP sample directly, with no re-fitting.

### 3.3 Slope features

- `slope_low`: OLS slope in L2 ∈ [25, 45] dB (linear growth region)
- `slope_high`: OLS slope in L2 ∈ [45, 65] dB (compressive region)

### 3.4 Physiological validity bounds

After extraction, values outside these ranges are treated as NaN:

| Feature | Valid range |
|---|---|
| Knee | [20, 65] dB |
| Max-compression | [30, 68] dB |
| Slope low | [−0.2, 3.0] dB/dB |
| Slope high | [−0.5, 2.5] dB/dB |

### 3.5 Summary statistics per feature

From N valid samples:
- **Mean**, **SD**
- **95% CI**: 2.5th and 97.5th percentiles
- **MDC** (minimum detectable change) = 1.96 · √2 · SD

---

## 4. Level Selection Strategies

### 4.1 Random

Sample k positions uniformly without replacement from the 10 available. Repeated 10 times per (curve, k); results averaged.

### 4.2 Equalspaced

```python
idx = round(linspace(0, 9, k))   # indices into the 10 L2 positions
```

### 4.3 Active (sequential greedy)

**Step 1 (seed):** Select 3 equalspaced positions. Fit GP with full Adam optimisation (80 steps). Cache lengthscale and outputscale.

**Steps 2 to k−3 (sequential selection):**
```
For each remaining step i = 1 … (k-3):
    Evaluate posterior variance at all remaining candidate positions
    Select the candidate with maximum variance
    Add to selected set; remove from candidates
    If not the last step:
        Refit GP posterior with FIXED hyperparameters (no Adam, so it is fast)
```

**Key property:** For Gaussian posteriors, maximising variance = maximising expected information gain (MacKay 1992). Each new measurement targets the most uncertain region of the I/O function.

**Constraint:** All strategies select from the 10 pre-defined stimulus positions. This limits active's advantage relative to a prospective fine-grid protocol.

---

## 5. Leave-k-out Evaluation

### 5.1 Loop structure

```
For each curve (146 total):
    Fit reference GP (k=10) for all 5 mean types          → reference_results.csv
    Fit deterministic cubic (k=10)

    For k in {3, 4, 5, 6, 7, 8, 9}:
        For strategy in {random (×10 repeats), equalspaced, active}:
            Select k stimulus levels using strategy
            For each mean type:
                Fit sparse GP (hyperparams cached from reference)
                Extract RMSE, features, CIs, coverage
            Fit sparse deterministic cubic
            Save row to experiment_results.csv
```

### 5.2 RMSE metrics

For each sparse fit, compute RMSE at the **held-out** (non-selected) positions:
- **Primary:** vs reference GP posterior mean at held-out L2 values
- **Secondary:** vs raw held-out DPOAE measurements

### 5.3 Feature coverage

For each feature (knee, max-compression), for each sparse fit:
```
covers = (sparse_CI_low ≤ reference_feature_mean ≤ sparse_CI_high)
```
Coverage rate = fraction of curves where `covers = True`.

### 5.4 Deterministic cubic baseline

Fit cubic polynomial to detected-only points (dp > noise2 + 3 dB). Returns NaN when fewer than 4 detected points.

---

## 6. Outputs

| File | Contents |
|---|---|
| `outputs/reference_results.csv` | k=10 feature posteriors (mean, SD, CI, MDC) for all 5 GP types + cubic; one row per curve |
| `outputs/experiment_results.csv` | One row per (curve × k × strategy × repeat); RMSE, feature posteriors, coverage flags; 121 columns |
| `outputs/fig1_example_curve.pdf` | Example GP vs cubic fit with feature posteriors |
| `outputs/fig2_rmse_vs_sparsity.pdf` | Model comparison + strategy comparison for RMSE |
| `outputs/fig3_feature_uncertainty.pdf` | MDC vs k by model and strategy |
| `outputs/fig4_coverage.pdf` | Feature CI coverage vs k for GP-linear and GP-twoslope |
| `outputs/fig5_cohort_features.pdf` | Cohort distributions of features at k=10 |
| `outputs/selection_overlap.pdf` | Active vs equalspaced selection overlap analysis |

---

## 7. Key Design Choices and Rationale

| Choice | Rationale |
|---|---|
| dB SPL (not Pascal) | Power law → linear in dB; Gaussian noise better behaved |
| Matérn-3/2 kernel | Smooth but not infinitely differentiable; matches OHC compression |
| Fixed parametric mean | Stable fitting at low k; GP models only residuals |
| Heteroscedastic noise | Reflects SNR-dependent measurement uncertainty |
| Sub-noise-floor retained | Left-censored data; discarding introduces upward bias (Helsel 2012) |
| Sequential active selection | Correctly updates variance landscape after each measurement |
| 200 posterior samples (sparse) | Reliable 2.5th/97.5th percentile estimation |
| Monotone filter | Removes physiologically implausible GP samples before feature extraction |
| Physiological bounds on features | Removes extraction failures without distorting posterior shape |
| MDC = 1.96·√2·σ | Accounts for uncertainty in both a baseline and a follow-up measurement |

---

## 8. Limitations (methodological)

1. **Single-session data**: no test-retest validation of GP CI coverage against real variability
2. **10-position grid constraint**: active sampling limited to pre-defined L2 levels
3. **Soft Tobit approximation**: a proper censored likelihood in Pa units is future work
4. **No population pooling**: by design for individual-level monitoring, not population-level inference
5. **Hyperparameter caching**: sparse fits inherit reference hyperparameters, which may not be optimal at very high sparsity
