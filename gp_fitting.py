"""
GP fitting for DPOAE I/O functions.

Model per curve:
  - Kernel : ScaleKernel(Matern32) over L2
  - Noise  : heteroscedastic from per-point SNR (detected points)
             + Tobit left-censoring for sub-noise-floor points
  - Inference: exact GP (n=10 points — no need for SVGP)

Public API
----------
fit_gp(gr, n_samples=500)
    Fit to all available L2 levels. Returns GPResult.

fit_sparse(gr, k, strategy='random', seed=None, n_samples=500)
    Fit to k levels selected by strategy. Returns GPResult.

GPResult fields
---------------
  l2_grid     : (200,)  dense L2 grid
  mean        : (200,)  posterior mean
  lower95     : (200,)  2.5th percentile
  upper95     : (200,)  97.5th percentile
  features    : dict of FeatureResult (knee, max_compression, slope_low, slope_high)
  k_used      : int     number of levels actually used
  n_detected  : int     number of levels above noise floor
  l2_used     : array   which L2 levels were used
"""

import numpy as np
import torch
import gpytorch
from dataclasses import dataclass, field
from typing import Optional
import warnings

from config import DETECTION_DELTA_DB, L2_LEVELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Observation noise floor (dB²) — minimum variance even at high SNR
_MIN_NOISE_VAR = 1.0

# Noise variance for sub-noise-floor (censored) points — large, not Tobit
# We use a simple approximation: assign high noise rather than full Tobit
# for the exact GP. See note in TobitExactGP below.
_CENSORED_NOISE_VAR = 36.0   # (6 dB)²


# GP grid resolution
_GRID_N = 200

# Posterior samples for feature extraction
_DEFAULT_N_SAMPLES = 500

# Feature extraction: L2 range for slope estimation
_SLOPE_LOW_RANGE  = (25.0, 45.0)   # linear growth region
_SLOPE_HIGH_RANGE = (45.0, 65.0)   # compressive region


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureResult:
    mean:    float
    std:     float
    ci_low:  float   # 2.5th percentile
    ci_high: float   # 97.5th percentile
    samples: np.ndarray = field(repr=False)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


@dataclass
class GPResult:
    l2_grid:    np.ndarray
    mean:       np.ndarray
    lower95:    np.ndarray
    upper95:    np.ndarray
    features:   dict           # str -> FeatureResult
    k_used:     int
    n_detected: int
    l2_used:    np.ndarray
    dp_used:    np.ndarray
    noise2_used: np.ndarray


# ---------------------------------------------------------------------------
# Noise model: heteroscedastic from per-point SNR
# ---------------------------------------------------------------------------

def _obs_noise_var(dp: np.ndarray, noise2: np.ndarray) -> np.ndarray:
    """
    Per-point observation variance.

    Detected points  (dp > noise2 + delta): variance decreases with SNR.
    Censored points  (dp <= noise2 + delta): large fixed variance.
    """
    snr = dp - noise2                          # dB above noise floor (2SD)
    detected = snr > DETECTION_DELTA_DB

    # Detected: noise var = max(1 / SNR_linear, _MIN_NOISE_VAR)
    # Simple approximation: var = 4 * exp(-0.1 * snr) + MIN
    noise_var = np.where(
        detected,
        np.clip(4.0 * np.exp(-0.1 * snr) + _MIN_NOISE_VAR, _MIN_NOISE_VAR, 25.0),
        _CENSORED_NOISE_VAR
    )
    return noise_var.astype(float)


# ---------------------------------------------------------------------------
# Mean function implementations
# ---------------------------------------------------------------------------

class _CubicMean(gpytorch.means.Mean):
    """
    Fixed cubic mean: y = a*L2^3 + b*L2^2 + c*L2 + d.
    Coefficients are fitted to all data points and held fixed.
    GP kernel models residuals only → tight, well-calibrated CIs.
    """
    def __init__(self, coeffs: np.ndarray):
        super().__init__()
        self.register_buffer("coeffs",
                             torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coeffs
        x = x.squeeze(-1)
        return c[0]*x**3 + c[1]*x**2 + c[2]*x + c[3]


class _LinearMean(gpytorch.means.Mean):
    """
    Fixed linear mean: y = m*L2 + b.
    Captures the rising slope at low L2 (fixes the flat-prior problem of
    ConstantMean) without committing to the full cubic shape.
    GP kernel models the compression nonlinearity and residuals.
    This gives wider CIs than CubicMean — the GP must explain the curvature —
    which is more honest about uncertainty when data are sparse.
    """
    def __init__(self, coeffs: np.ndarray):
        super().__init__()
        self.register_buffer("coeffs",
                             torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coeffs
        x = x.squeeze(-1)
        return c[0]*x + c[1]


class _QuadraticMean(gpytorch.means.Mean):
    """
    Fixed quadratic mean: y = a*L2^2 + b*L2 + c.
    Captures the gradual flattening at high L2 without committing to the full
    cubic inflection. Intermediate between LinearMean (too little curvature)
    and CubicMean (pre-specifies the full compression shape).
    """
    def __init__(self, coeffs: np.ndarray):
        super().__init__()
        self.register_buffer("coeffs",
                             torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coeffs
        x = x.squeeze(-1)
        return c[0]*x**2 + c[1]*x + c[2]


class _TwoSlopeMean(gpytorch.means.Mean):
    """
    Continuous two-slope (piecewise linear) mean.

    Below the breakpoint: linear growth with slope s1 (linear amplification).
    Above the breakpoint: linear compression with slope s2 < s1 (compressed).
    Continuity is enforced at the breakpoint.

    This directly encodes the two-regime model described by Janssen (2006)
    and Bhagat (2014) — the physiologically expected shape for DPOAE I/O
    functions. The GP residuals then model individual departures from this
    canonical shape only, giving well-grounded credible intervals.

    Parameters are fitted by OLS with grid search over breakpoints; the
    breakpoint with minimum total residual sum of squares is used.
    """
    def __init__(self, bp: float, s1: float, b: float, s2: float):
        super().__init__()
        self.register_buffer("bp", torch.tensor(bp,  dtype=torch.float32))
        self.register_buffer("s1", torch.tensor(s1,  dtype=torch.float32))
        self.register_buffer("b",  torch.tensor(b,   dtype=torch.float32))
        self.register_buffer("s2", torch.tensor(s2,  dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.squeeze(-1)
        bp, s1, b, s2 = self.bp, self.s1, self.b, self.s2
        y_bp = s1 * bp + b
        low  = s1 * x + b
        high = s2 * (x - bp) + y_bp
        return torch.where(x <= bp, low, high)


def _fit_poly_mean(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    """Fit polynomial of given degree; falls back gracefully if too few points."""
    n_needed = deg + 1
    if len(x) < n_needed:
        actual_deg = max(1, len(x) - 1)
        c = np.polyfit(x, y, actual_deg)
        return np.concatenate([np.zeros(deg + 1 - len(c)), c])
    return np.polyfit(x, y, deg)


def _fit_twoslope_mean(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit a continuous two-slope mean by grid search over breakpoints [30, 58] dB.
    Returns dict: bp, s1, b, s2.
    Falls back to linear if fitting fails.
    """
    best_rss    = np.inf
    best_params = None

    for bp in np.linspace(30.0, 58.0, 29):
        lo = x <= bp
        hi = x >  bp
        if lo.sum() < 2 or hi.sum() < 2:
            continue

        # Low segment: OLS for s1, b
        A  = np.column_stack([x[lo], np.ones(lo.sum())])
        try:
            s1, b_val = np.linalg.lstsq(A, y[lo], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # High segment: OLS for s2 with continuity at bp
        y_bp = s1 * bp + b_val
        x_hi = (x[hi] - bp).reshape(-1, 1)
        try:
            s2 = np.linalg.lstsq(x_hi, y[hi] - y_bp, rcond=None)[0][0]
        except np.linalg.LinAlgError:
            continue

        rss = (np.sum((y[lo] - (s1*x[lo] + b_val))**2) +
               np.sum((y[hi] - (s2*(x[hi].ravel()) + y_bp))**2))

        if rss < best_rss:
            best_rss    = rss
            best_params = {"bp": float(bp), "s1": float(s1),
                           "b": float(b_val), "s2": float(s2)}

    if best_params is None:
        s1, b_val = np.polyfit(x, y, 1) if len(x) >= 2 else (0.5, -10.0)
        best_params = {"bp": 45.0, "s1": float(s1),
                       "b": float(b_val), "s2": float(s1) * 0.3}

    return best_params


# Outputscale initialisations per mean type.
# Cubic/twoslope absorb the shape → small residuals.
# Quadratic captures curvature but not compression → medium residuals.
# Linear captures slope only → larger residuals.
# Constant captures nothing → full variance.
_OUTPUTSCALE_INIT = {
    "cubic":    4.0,
    "twoslope": 6.0,
    "quadratic": 8.0,
    "linear":  12.0,
    "constant": 25.0,
}

MEAN_TYPES = ("cubic", "twoslope", "quadratic", "linear", "constant")


# ---------------------------------------------------------------------------
# GP model (exact, fixed noise, configurable mean)
# ---------------------------------------------------------------------------

class _IoGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 mean_module: gpytorch.means.Mean,
                 outputscale_init: float = 4.0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
        self.covar_module.base_kernel.lengthscale = 15.0
        self.covar_module.outputscale = outputscale_init

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )


def _build_mean_module(mean_type: str,
                       x: np.ndarray, y: np.ndarray) -> gpytorch.means.Mean:
    """Construct the appropriate mean module for the given mean_type."""
    if mean_type == "cubic":
        return _CubicMean(_fit_poly_mean(x, y, deg=3))
    elif mean_type == "quadratic":
        return _QuadraticMean(_fit_poly_mean(x, y, deg=2))
    elif mean_type == "twoslope":
        p = _fit_twoslope_mean(x, y)
        return _TwoSlopeMean(p["bp"], p["s1"], p["b"], p["s2"])
    elif mean_type == "linear":
        return _LinearMean(_fit_poly_mean(x, y, deg=1))
    elif mean_type == "constant":
        return gpytorch.means.ConstantMean()
    else:
        raise ValueError(f"mean_type must be one of {MEAN_TYPES}, got '{mean_type}'")


def _train_gp(x: np.ndarray, y: np.ndarray, noise_var: np.ndarray,
              mean_type: str = "cubic",
              n_steps: int = 150, lr: float = 0.05,
              fixed_lengthscale: Optional[float] = None,
              fixed_outputscale: Optional[float] = None) -> tuple:
    """
    Train exact GP with configurable mean and fixed heteroscedastic noise.

    If fixed_lengthscale and fixed_outputscale are provided the Adam loop is
    skipped entirely — the model is built with those values and evaluated
    directly.  This is the fast path for sparse fits where hyperparameters
    are inherited from the reference (k=10) fit of the same curve.
    """
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    tn = torch.tensor(noise_var, dtype=torch.float32)

    mean_module      = _build_mean_module(mean_type, x, y)
    outputscale_init = fixed_outputscale if fixed_outputscale is not None \
                       else _OUTPUTSCALE_INIT[mean_type]

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=tn, learn_additional_noise=False
    )
    model = _IoGP(tx, ty, likelihood, mean_module, outputscale_init)

    if fixed_lengthscale is not None:
        # Fast path: set hyperparameters directly, skip optimisation
        model.covar_module.base_kernel.lengthscale = fixed_lengthscale
        model.covar_module.outputscale             = outputscale_init
    else:
        model.train(); likelihood.train()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_steps):
                optimiser.zero_grad()
                loss = -mll(model(tx), ty)
                loss.backward()
                optimiser.step()

    model.eval(); likelihood.eval()
    return model, likelihood, tx


def _extract_hyperparams(model) -> tuple[float, float]:
    """Extract learned lengthscale and outputscale from a trained GP model."""
    ls = float(model.covar_module.base_kernel.lengthscale.item())
    os = float(model.covar_module.outputscale.item())
    return ls, os


@torch.no_grad()
def _predict(model, likelihood, l2_grid: np.ndarray):
    """Return posterior mean and std on a dense grid."""
    tg = torch.tensor(l2_grid, dtype=torch.float32)
    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(tg))
    mean = pred.mean.numpy()
    std  = pred.stddev.numpy()
    return mean, std


@torch.no_grad()
def _sample_posterior(model, l2_grid: np.ndarray, n_samples: int) -> np.ndarray:
    """Draw n_samples functions from the posterior. Returns (n_samples, len(grid))."""
    tg = torch.tensor(l2_grid, dtype=torch.float32)
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
        post = model(tg)
        samples = post.rsample(torch.Size([n_samples]))  # (n_samples, grid)
    return samples.numpy()


# ---------------------------------------------------------------------------
# Feature extraction from posterior samples
# ---------------------------------------------------------------------------

def _knee_from_sample(l2: np.ndarray, dp_sample: np.ndarray) -> float:
    """
    PWLF knee: L2 at the breakpoint of a 2-segment continuous linear fit.
    Searches breakpoints in [25, 63] dB at 0.68 dB resolution (57 steps).
    Range extended vs earlier version to capture knees at high L2 (4243 Hz).
    """
    best_bp = np.nan
    best_sse = np.inf

    for bp in np.linspace(25.0, 63.0, 57):
        mask_low  = l2 <= bp
        mask_high = l2 >  bp

        if mask_low.sum() < 2 or mask_high.sum() < 2:
            continue

        # Fit low segment
        m1, b1 = np.polyfit(l2[mask_low], dp_sample[mask_low], 1)
        # Force continuity: high segment starts at bp
        y_bp = m1 * bp + b1
        m2, _ = np.polyfit(l2[mask_high] - bp, dp_sample[mask_high] - y_bp, 1)

        y_hat = np.where(mask_low,
                         m1 * l2 + b1,
                         m2 * (l2 - bp) + y_bp)
        sse = np.sum((dp_sample - y_hat) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_bp  = bp

    return best_bp


def _max_compression_from_sample(l2: np.ndarray, dp_sample: np.ndarray) -> float:
    """
    Max-compression L2: the stimulus level at which the GP posterior sample
    reaches its maximum amplitude in the compressive region (L2 >= 40 dB).

    Uses the sample directly rather than re-fitting a cubic, avoiding circular
    fitting and instability on non-monotone samples.
    Falls back to the full range if no points exceed 40 dB.
    """
    mask = l2 >= 40.0
    if mask.sum() < 2:
        mask = np.ones(len(l2), dtype=bool)
    idx = int(np.argmax(dp_sample[mask]))
    return float(l2[mask][idx])


def _slopes_from_sample(l2: np.ndarray, dp_sample: np.ndarray) -> tuple:
    """
    Low slope  (linear growth region, _SLOPE_LOW_RANGE)
    High slope (compressive region,   _SLOPE_HIGH_RANGE)
    Both in dB/dB.
    """
    def _slope_in_range(r):
        mask = (l2 >= r[0]) & (l2 <= r[1])
        if mask.sum() < 2:
            return np.nan
        m, _ = np.polyfit(l2[mask], dp_sample[mask], 1)
        return float(m)

    return _slope_in_range(_SLOPE_LOW_RANGE), _slope_in_range(_SLOPE_HIGH_RANGE)


# Physiologically plausible ranges for each feature (dB SPL for L2 features,
# dB/dB for slopes). Samples outside these ranges are treated as extraction
# failures and excluded from posterior summaries.
_FEATURE_BOUNDS = {
    "knee":            (20.0, 65.0),
    "max_compression": (30.0, 68.0),
    "slope_low":       (-0.2,  3.0),
    "slope_high":      (-0.5,  2.5),
}


def _filter_monotone(l2_grid: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Remove grossly non-monotone posterior samples.

    A sample is rejected if the running mean of its point-to-point differences
    over any 15-step window (≈1.7 dB on the 200-point grid) is more negative
    than -0.1 dB/step, equivalent to a sustained decline of >1.4 dB over a
    1.7-dB L2 window — physiologically implausible.

    If fewer than 50% of samples pass, filtering is skipped to avoid
    catastrophic sample loss at extreme sparsity.
    """
    diffs  = np.diff(samples, axis=1)          # (n, grid-1)
    w      = 15
    kernel = np.ones(w) / w
    # Running mean of diffs for each sample
    smooth = np.array([np.convolve(d, kernel, mode="valid") for d in diffs])
    valid  = np.all(smooth >= -0.1, axis=1)

    if valid.sum() >= max(1, len(samples) // 2):
        return samples[valid]
    return samples   # fall back: too many rejected, keep all


def _extract_features(l2_grid: np.ndarray, samples: np.ndarray) -> dict:
    """
    Extract features from posterior samples.
    Applies monotone filtering and physiological validity bounds before
    computing posterior summaries.
    Returns dict: knee, max_compression, slope_low, slope_high -> FeatureResult.
    """
    samples = _filter_monotone(l2_grid, samples)

    knees      = np.array([_knee_from_sample(l2_grid, s) for s in samples])
    max_comps  = np.array([_max_compression_from_sample(l2_grid, s) for s in samples])
    slopes_low = np.array([_slopes_from_sample(l2_grid, s)[0] for s in samples])
    slopes_hi  = np.array([_slopes_from_sample(l2_grid, s)[1] for s in samples])

    def _to_feature(arr, fname):
        arr = arr[np.isfinite(arr)]
        lo, hi = _FEATURE_BOUNDS.get(fname, (-np.inf, np.inf))
        arr = arr[(arr >= lo) & (arr <= hi)]
        if len(arr) == 0:
            return FeatureResult(np.nan, np.nan, np.nan, np.nan, arr)
        return FeatureResult(
            mean    = float(np.mean(arr)),
            std     = float(np.std(arr)),
            ci_low  = float(np.percentile(arr, 2.5)),
            ci_high = float(np.percentile(arr, 97.5)),
            samples = arr
        )

    return {
        "knee":            _to_feature(knees,      "knee"),
        "max_compression": _to_feature(max_comps,  "max_compression"),
        "slope_low":       _to_feature(slopes_low, "slope_low"),
        "slope_high":      _to_feature(slopes_hi,  "slope_high"),
    }


# ---------------------------------------------------------------------------
# Level selection strategies
# ---------------------------------------------------------------------------

def _select_levels_random(l2_all: np.ndarray, k: int,
                           seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(l2_all), size=k, replace=False)
    return np.sort(idx)


def _select_levels_equalspaced(l2_all: np.ndarray, k: int) -> np.ndarray:
    idx = np.round(np.linspace(0, len(l2_all) - 1, k)).astype(int)
    return np.unique(idx)


def _select_levels_active(l2_all: np.ndarray, dp_all: np.ndarray,
                           noise_var_all: np.ndarray, k: int,
                           n_steps_seed: int = 80) -> np.ndarray:
    """
    Sequential greedy variance-weighted active sampling.

    1. Fit a GP (with Adam optimisation) on 3 equalspaced seed points.
    2. Pick the remaining (k - 3) points one at a time: at each step, evaluate
       posterior variance at all candidates and select the highest.
    3. After each selection, refit the GP with the new point added — using
       cached hyperparameters (no Adam loop) so only the posterior updates.

    Sequential selection correctly accounts for the reduction in variance near
    an already-selected point, pushing each new selection to a genuinely
    different region of the L2 range. This differs from the previous batch
    implementation that evaluated all candidates simultaneously from the
    3-point GP, which underestimated the benefit of spread-out selection.
    """
    if k <= 3:
        return _select_levels_equalspaced(l2_all, k)

    selected_idx = list(_select_levels_equalspaced(l2_all, 3))
    remaining    = [i for i in range(len(l2_all)) if i not in selected_idx]

    if not remaining or k - 3 >= len(remaining):
        return np.arange(len(l2_all))

    try:
        # Initial fit on 3 seeds — full optimisation to learn hyperparameters
        x  = l2_all[selected_idx]
        y  = dp_all[selected_idx]
        nv = noise_var_all[selected_idx]
        model, likelihood, _ = _train_gp(x, y, nv, n_steps=n_steps_seed)
        ls, os_ = _extract_hyperparams(model)

        # Sequential greedy: one point at a time
        for step in range(k - 3):
            if not remaining:
                break

            # Evaluate posterior variance at all remaining candidates
            candidates = l2_all[remaining]
            tc = torch.tensor(candidates, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                variances = model(tc).variance.numpy()

            # Add the highest-variance candidate
            best_local = int(np.argmax(variances))
            selected_idx.append(remaining[best_local])
            remaining.pop(best_local)

            # Refit posterior with the new point — hyperparameters fixed,
            # only the posterior (kernel matrix) updates. Skip for last step.
            if step < k - 4:
                x  = l2_all[selected_idx]
                y  = dp_all[selected_idx]
                nv = noise_var_all[selected_idx]
                model, likelihood, _ = _train_gp(
                    x, y, nv, n_steps=n_steps_seed,
                    fixed_lengthscale=ls, fixed_outputscale=os_,
                )

    except Exception:
        return _select_levels_equalspaced(l2_all, k)

    return np.array(sorted(selected_idx))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_result(model, likelihood, l2_used, dp_used, noise2_used,
                  n_detected, n_samples, grid_n: int = _GRID_N) -> GPResult:
    """Build a GPResult from a trained model."""
    l2_grid = np.linspace(float(l2_used.min()), float(l2_used.max()), grid_n)

    mean, std = _predict(model, likelihood, l2_grid)
    lower95   = mean - 1.96 * std
    upper95   = mean + 1.96 * std

    if n_samples > 0:
        samples  = _sample_posterior(model, l2_grid, n_samples)
        features = _extract_features(l2_grid, samples)
    else:
        # Fast mode — skip feature extraction, return empty FeatureResult stubs
        _nan = FeatureResult(np.nan, np.nan, np.nan, np.nan, np.array([]))
        features = {k: _nan for k in
                    ("knee", "max_compression", "slope_low", "slope_high")}

    return GPResult(
        l2_grid     = l2_grid,
        mean        = mean,
        lower95     = lower95,
        upper95     = upper95,
        features    = features,
        k_used      = len(l2_used),
        n_detected  = n_detected,
        l2_used     = l2_used,
        dp_used     = dp_used,
        noise2_used = noise2_used,
    )


def fit_gp(gr: dict,
           mean_type: str = "cubic",
           n_samples: int = _DEFAULT_N_SAMPLES) -> GPResult:
    """
    Fit GP to all L2 levels in a GR dict.

    Parameters
    ----------
    gr        : dict  GR dict from load_data (must have f2, dp, noise2).
    mean_type : str   Mean function — 'cubic' | 'linear' | 'constant'.
                      'cubic'    : fixed cubic prior (tight CIs, similar to cubic fit)
                      'linear'   : fixed linear prior (GP explains curvature)
                      'constant' : flat prior (original behaviour, wide CIs)
    n_samples : int   Posterior samples for feature extraction.

    Returns
    -------
    GPResult
    """
    l2  = np.asarray(gr["f2"],    dtype=float)
    dp  = np.asarray(gr["dp"],    dtype=float)
    n2  = np.asarray(gr["noise2"],dtype=float)

    mask = np.isfinite(l2) & np.isfinite(dp) & np.isfinite(n2)
    l2, dp, n2 = l2[mask], dp[mask], n2[mask]

    n_detected = int(np.sum(dp > (n2 + DETECTION_DELTA_DB)))
    noise_var  = _obs_noise_var(dp, n2)

    model, likelihood, _ = _train_gp(l2, dp, noise_var, mean_type=mean_type)
    result = _build_result(model, likelihood, l2, dp, n2, n_detected, n_samples)
    result._hyperparams = _extract_hyperparams(model)   # cache for sparse reuse
    return result


def fit_sparse(gr: dict, k: int,
               strategy: str = "random",
               seed: Optional[int] = None,
               mean_type: str = "cubic",
               n_samples: int = _DEFAULT_N_SAMPLES,
               n_steps: int = 150,
               ref_hyperparams: Optional[tuple] = None) -> GPResult:
    """
    ref_hyperparams : (lengthscale, outputscale) from the reference k=10 fit.
    When provided the Adam optimisation loop is skipped, giving a large
    speed improvement (~5–10× per sparse fit).
    """
    """
    Fit GP to k levels selected by strategy.

    Parameters
    ----------
    gr        : dict  GR dict from load_data.
    k         : int   Number of L2 levels to retain (3 <= k <= 10).
    strategy  : str   'random' | 'equalspaced' | 'active'
    seed      : int   Random seed (only used for 'random' strategy).
    mean_type : str   Mean function — 'cubic' | 'linear' | 'constant'.
    n_samples : int   Posterior samples for feature extraction.

    Returns
    -------
    GPResult  (with k_used = k)
    """
    l2_all = np.asarray(gr["f2"],    dtype=float)
    dp_all = np.asarray(gr["dp"],    dtype=float)
    n2_all = np.asarray(gr["noise2"],dtype=float)

    mask = np.isfinite(l2_all) & np.isfinite(dp_all) & np.isfinite(n2_all)
    l2_all, dp_all, n2_all = l2_all[mask], dp_all[mask], n2_all[mask]

    k      = min(k, len(l2_all))
    nv_all = _obs_noise_var(dp_all, n2_all)

    if strategy == "random":
        idx = _select_levels_random(l2_all, k, seed)
    elif strategy == "equalspaced":
        idx = _select_levels_equalspaced(l2_all, k)
    elif strategy == "active":
        idx = _select_levels_active(l2_all, dp_all, nv_all, k)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         "Choose from: random, equalspaced, active")

    l2_sel = l2_all[idx]
    dp_sel = dp_all[idx]
    n2_sel = n2_all[idx]
    nv_sel = nv_all[idx]
    n_det  = int(np.sum(dp_sel > (n2_sel + DETECTION_DELTA_DB)))

    fixed_ls, fixed_os = ref_hyperparams if ref_hyperparams else (None, None)
    model, likelihood, _ = _train_gp(l2_sel, dp_sel, nv_sel,
                                      mean_type=mean_type, n_steps=n_steps,
                                      fixed_lengthscale=fixed_ls,
                                      fixed_outputscale=fixed_os)
    return _build_result(model, likelihood, l2_sel, dp_sel, n2_sel, n_det, n_samples)


def fit_gp_conditioned(
    l2:         np.ndarray,
    dp:         np.ndarray,
    noise_var:  np.ndarray,
    n2:         np.ndarray,
    mean_type:  str   = "linear",
    hyperparams: tuple = (None, None),
    n_samples:  int   = 100,
    n_detected: int   = 0,
) -> GPResult:
    """
    Condition a GP on observations using fixed hyperparameters (no training).

    The lengthscale and outputscale are inherited from a reference fit and the
    Adam optimisation loop is skipped entirely.  Only the mean function
    coefficients are re-estimated (by OLS) from the supplied data.

    Intended for bootstrap replication in bootstrap_retest.py: the reference
    hyperparameters characterise the curve's smoothness; each bootstrap sample
    re-estimates the mean and propagates measurement noise into the posterior.

    Parameters
    ----------
    l2          : observed L2 levels (dB SPL).
    dp          : DPOAE amplitudes — may be synthetic (bootstrap sample).
    noise_var   : per-point observation variance (pre-computed; fixed from
                  the reference session to avoid recategorising censored pts).
    n2          : noise floor values stored in GPResult.noise2_used.
    mean_type   : mean function identifier — should match the reference fit.
    hyperparams : (lengthscale, outputscale) from the reference fit.
    n_samples   : posterior samples for feature extraction.
    n_detected  : above-floor detection count (metadata only, from reference).
    """
    fixed_ls, fixed_os = hyperparams
    model, likelihood, _ = _train_gp(
        l2, dp, noise_var,
        mean_type        = mean_type,
        fixed_lengthscale = fixed_ls,
        fixed_outputscale = fixed_os,
    )
    return _build_result(model, likelihood, l2, dp, n2, n_detected, n_samples)


# ---------------------------------------------------------------------------
# Cubic baseline (current standard — no uncertainty)
# ---------------------------------------------------------------------------

@dataclass
class CubicResult:
    """
    Cubic polynomial fit — deterministic, no uncertainty.
    Comparable structure to GPResult for head-to-head comparison.
    """
    l2_grid:    np.ndarray
    mean:       np.ndarray           # cubic evaluated on grid
    features:   dict                 # str -> float (point estimates only)
    k_used:     int
    n_detected: int
    l2_used:    np.ndarray
    dp_used:    np.ndarray
    # No lower95 / upper95 — cubic gives no credible interval


def _cubic_features(l2: np.ndarray, dp: np.ndarray) -> dict:
    """Extract features from a cubic fit. Returns point estimates (no CI)."""
    if len(l2) < 4:
        return {k: np.nan for k in
                ("knee", "max_compression", "slope_low", "slope_high")}

    coeffs   = np.polyfit(l2, dp, 3)
    d_coeffs = np.polyder(coeffs)          # quadratic (derivative)

    fine  = np.linspace(float(l2.min()), float(l2.max()), 500)
    deriv = np.polyval(d_coeffs, fine)

    # Max-compression: minimum positive derivative
    pos_mask = deriv > 0
    if pos_mask.any():
        max_comp = float(fine[pos_mask][np.argmin(deriv[pos_mask])])
    else:
        sign_changes = np.where(np.diff(np.sign(deriv)))[0]
        max_comp = float(fine[sign_changes[0]]) if len(sign_changes) else np.nan

    # Knee from cubic (use same PWLF search on the fitted curve)
    dp_fitted = np.polyval(coeffs, fine)
    knee      = float(_knee_from_sample(fine, dp_fitted))

    # Slopes
    def _slope(r):
        mask = (l2 >= r[0]) & (l2 <= r[1])
        if mask.sum() < 2:
            return np.nan
        m, _ = np.polyfit(l2[mask], dp[mask], 1)
        return float(m)

    return {
        "knee":            knee,
        "max_compression": max_comp,
        "slope_low":       _slope(_SLOPE_LOW_RANGE),
        "slope_high":      _slope(_SLOPE_HIGH_RANGE),
    }


def fit_cubic(gr: dict) -> CubicResult:
    """
    Fit cubic polynomial to all L2 levels — standard baseline.

    Only uses detected points (dp > noise2 + delta), matching
    conventional practice (Janssen 2006, Bhagat 2014).
    """
    l2_all = np.asarray(gr["f2"],    dtype=float)
    dp_all = np.asarray(gr["dp"],    dtype=float)
    n2_all = np.asarray(gr["noise2"],dtype=float)

    mask = np.isfinite(l2_all) & np.isfinite(dp_all) & np.isfinite(n2_all)
    l2_all, dp_all, n2_all = l2_all[mask], dp_all[mask], n2_all[mask]

    detected = dp_all > (n2_all + DETECTION_DELTA_DB)
    l2_det   = l2_all[detected]
    dp_det   = dp_all[detected]

    l2_grid = np.linspace(float(l2_all.min()), float(l2_all.max()), _GRID_N)

    if len(l2_det) < 4:
        # Not enough detected points — return NaN curve
        return CubicResult(
            l2_grid    = l2_grid,
            mean       = np.full(_GRID_N, np.nan),
            features   = {k: np.nan for k in
                          ("knee","max_compression","slope_low","slope_high")},
            k_used     = int(detected.sum()),
            n_detected = int(detected.sum()),
            l2_used    = l2_det,
            dp_used    = dp_det,
        )

    coeffs   = np.polyfit(l2_det, dp_det, 3)
    curve    = np.polyval(coeffs, l2_grid)
    features = _cubic_features(l2_det, dp_det)

    return CubicResult(
        l2_grid    = l2_grid,
        mean       = curve,
        features   = features,
        k_used     = len(l2_det),
        n_detected = int(detected.sum()),
        l2_used    = l2_det,
        dp_used    = dp_det,
    )


def fit_cubic_sparse(gr: dict, k: int,
                     strategy: str = "random",
                     seed: Optional[int] = None) -> CubicResult:
    """
    Fit cubic to k selected L2 levels (detected points only).
    Uses same level selection strategies as fit_sparse for fair comparison.
    """
    l2_all = np.asarray(gr["f2"],    dtype=float)
    dp_all = np.asarray(gr["dp"],    dtype=float)
    n2_all = np.asarray(gr["noise2"],dtype=float)

    mask = np.isfinite(l2_all) & np.isfinite(dp_all) & np.isfinite(n2_all)
    l2_all, dp_all, n2_all = l2_all[mask], dp_all[mask], n2_all[mask]

    k = min(k, len(l2_all))
    nv_all = _obs_noise_var(dp_all, n2_all)

    if strategy == "random":
        idx = _select_levels_random(l2_all, k, seed)
    elif strategy == "equalspaced":
        idx = _select_levels_equalspaced(l2_all, k)
    elif strategy == "active":
        idx = _select_levels_active(l2_all, dp_all, nv_all, k)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    l2_sel = l2_all[idx]
    dp_sel = dp_all[idx]
    n2_sel = n2_all[idx]

    # Cubic uses only detected points from the selection
    det     = dp_sel > (n2_sel + DETECTION_DELTA_DB)
    l2_det  = l2_sel[det]
    dp_det  = dp_sel[det]
    l2_grid = np.linspace(float(l2_all.min()), float(l2_all.max()), _GRID_N)

    if len(l2_det) < 4:
        return CubicResult(
            l2_grid    = l2_grid,
            mean       = np.full(_GRID_N, np.nan),
            features   = {k: np.nan for k in
                          ("knee","max_compression","slope_low","slope_high")},
            k_used     = len(l2_sel),
            n_detected = int(det.sum()),
            l2_used    = l2_det,
            dp_used    = dp_det,
        )

    coeffs   = np.polyfit(l2_det, dp_det, 3)
    curve    = np.polyval(coeffs, l2_grid)
    features = _cubic_features(l2_det, dp_det)

    return CubicResult(
        l2_grid    = l2_grid,
        mean       = curve,
        features   = features,
        k_used     = len(l2_sel),
        n_detected = int(det.sum()),
        l2_used    = l2_det,
        dp_used    = dp_det,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from load_data import load_from_pickle

    OAEs = load_from_pickle()
    gr   = OAEs[0]["GR"][0]

    print(f"Participant {gr['ID']} | side {gr['side']} | f {gr['f']} Hz")
    print(f"L2 levels : {gr['f2']}")
    print(f"DP values : {np.round(gr['dp'], 1)}\n")

    # ---- Compare all three mean types at k=10 ----
    print(f"{'Model':<22} {'Knee mean':>10} {'Knee CI':>18}  {'CI width':>9}  {'Max-comp':>9}")
    print("-" * 75)
    for mt in MEAN_TYPES:
        r = fit_gp(gr, mean_type=mt, n_samples=200)
        k_feat = r.features["knee"]
        mc_feat = r.features["max_compression"]
        print(f"  GP ({mt:<10})     "
              f"{k_feat.mean:>8.1f}  "
              f"[{k_feat.ci_low:>5.1f}, {k_feat.ci_high:>5.1f}]  "
              f"{k_feat.ci_width:>8.1f}  "
              f"{mc_feat.mean:>8.1f}")
    cub = fit_cubic(gr)
    print(f"  Cubic (baseline)     "
          f"{cub.features['knee']:>8.1f}  {'no CI':>18}  {'—':>9}  "
          f"{cub.features['max_compression']:>8.1f}")

    print("\n--- Sparse comparison at k=5 (active) — knee feature ---")
    print(f"{'Model':<22} {'Knee mean':>10} {'Knee CI':>18}  {'CI width':>9}")
    print("-" * 65)
    for mt in MEAN_TYPES:
        r5 = fit_sparse(gr, k=5, strategy="active", mean_type=mt, n_samples=200)
        k5 = r5.features["knee"]
        print(f"  GP ({mt:<10})     "
              f"{k5.mean:>8.1f}  "
              f"[{k5.ci_low:>5.1f}, {k5.ci_high:>5.1f}]  "
              f"{k5.ci_width:>8.1f}")
    cub5 = fit_cubic_sparse(gr, k=5, strategy="active")
    print(f"  Cubic (baseline)     "
          f"{cub5.features['knee']:>8.1f}  {'no CI':>18}")

    print("\n--- Fitting full GP (k=10) ---")
    res = fit_gp(gr, mean_type="cubic", n_samples=200)
    print(f"  k_used      : {res.k_used}")
    print(f"  n_detected  : {res.n_detected}")
    print(f"  Knee        : {res.features['knee'].mean:.1f} "
          f"[{res.features['knee'].ci_low:.1f}, {res.features['knee'].ci_high:.1f}] dB")
    print(f"  Max-comp    : {res.features['max_compression'].mean:.1f} "
          f"[{res.features['max_compression'].ci_low:.1f}, "
          f"{res.features['max_compression'].ci_high:.1f}] dB")
    print(f"  Slope low   : {res.features['slope_low'].mean:.3f} dB/dB")
    print(f"  Slope high  : {res.features['slope_high'].mean:.3f} dB/dB")

    print("\n--- Fitting sparse GP (k=5, random) ---")
    res5 = fit_sparse(gr, k=5, strategy="random", seed=42, n_samples=200)
    print(f"  L2 used : {res5.l2_used}")
    print(f"  Knee    : {res5.features['knee'].mean:.1f} "
          f"[{res5.features['knee'].ci_low:.1f}, {res5.features['knee'].ci_high:.1f}] dB")

    print("\n--- Fitting sparse GP (k=5, active) ---")
    res5a = fit_sparse(gr, k=5, strategy="active", n_samples=200)
    print(f"  L2 used : {res5a.l2_used}")
    print(f"  Knee    : {res5a.features['knee'].mean:.1f} "
          f"[{res5a.features['knee'].ci_low:.1f}, {res5a.features['knee'].ci_high:.1f}] dB")

    print("\n--- Cubic baseline (k=10, detected only) ---")
    cub = fit_cubic(gr)
    print(f"  n_detected  : {cub.n_detected}")
    print(f"  Knee        : {cub.features['knee']:.1f} dB  (no CI)")
    print(f"  Max-comp    : {cub.features['max_compression']:.1f} dB  (no CI)")
    print(f"  Slope low   : {cub.features['slope_low']:.3f} dB/dB")
    print(f"  Slope high  : {cub.features['slope_high']:.3f} dB/dB")

    print("\n--- Cubic sparse (k=5, random) ---")
    cub5 = fit_cubic_sparse(gr, k=5, strategy="random", seed=42)
    print(f"  n_detected  : {cub5.n_detected}")
    print(f"  Knee        : {cub5.features['knee']:.1f} dB")

    print("\n--- GP vs Cubic feature comparison (k=10) ---")
    for feat in ("knee", "max_compression", "slope_low", "slope_high"):
        gp_val = res.features[feat]
        cu_val = cub.features[feat]
        print(f"  {feat:<20} GP: {gp_val.mean:.2f} [{gp_val.ci_low:.1f},{gp_val.ci_high:.1f}]"
              f"   Cubic: {cu_val:.2f}")

    print("\nDone.")
