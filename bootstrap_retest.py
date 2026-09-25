# Thread limits must be set before numpy/torch import any BLAS DLLs.
# Without these, 6 workers each spawn ~8 MKL threads = 48 threads on a
# machine with 8 cores → heavy context-switching, ~7% CPU per process.
import os
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

"""
Residual bootstrap test-retest analysis for DPOAE I/O compression features.

For each of the 146 I/O curves:
  1. Fit the reference GP (k=10, linear mean) -- posterior mean at observed
     L2 levels + fitted hyperparameters + posterior SD for the scatter plot.
  2. Compute residuals: r_i = DP_observed_i - GP_mean(L2_i).
  3. For N_BOOTSTRAP replicates:
     a. Resample r with replacement (parametric bootstrap).
     b. Synthetic session 2: DP_syn_j = GP_mean(L2_j) + r_boot_j.
     c. Condition the GP on synthetic data with FIXED reference hyperparameters
        (empirical-Bayes -- no re-optimisation, fast). Use n_samples=0:
        only the posterior MEAN curve is needed, not the full distribution.
        Feature extraction runs directly on the mean curve.
     d. delta = mean-curve feature_s2 - mean-curve feature_s1.
  4. Bootstrap MDC = 1.96 * SD(delta) -- a genuine two-session lower bound,
     directly comparable to Reavis (2008).

Speed note
----------
Bootstrap fits use n_samples=0 (no posterior sampling). Feature extraction
from the mean curve takes ~1 ms vs ~80 ms for 100-sample extraction.
With 500 reps x 146 curves this gives a ~50x speedup over the naive approach.

Outputs
-------
  outputs/bootstrap_retest_results.csv
  outputs/fig_bootstrap_retest.pdf / .png

Run
---
    python bootstrap_retest.py
    python bootstrap_retest.py --n-bootstrap 200 --smoke
    python bootstrap_retest.py --workers 4
    python bootstrap_retest.py --plot-only    # regenerate figure from CSV
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import OUTPUT_DIR
from load_data import load_from_pickle
from gp_fitting import (
    fit_gp,
    _obs_noise_var,
    _knee_from_sample,
    _max_compression_from_sample,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_BOOTSTRAP    = 500      # bootstrap reps per curve
MEAN_TYPE      = "linear" # must match the paper's primary model
N_SAMPLES_REF  = 100      # posterior samples for session-1 posterior SD (scatter plot)
# Bootstrap fits use n_samples=0 — features extracted from mean curve only.

N_WORKERS = max(1, min(os.cpu_count() - 2, 6))

CLINICAL_THRESHOLD = 6.0  # dB -- Reavis (2008)

RESULTS_PATH = OUTPUT_DIR / "bootstrap_retest_results.csv"
FIG_PATH     = OUTPUT_DIR / "fig_bootstrap_retest"


# ---------------------------------------------------------------------------
# Kernel helper
# ---------------------------------------------------------------------------

def _matern32_kernel(x1: np.ndarray, x2: np.ndarray,
                     ls: float, outputscale: float) -> np.ndarray:
    """Matérn-3/2 covariance matrix between x1 (m,) and x2 (n,) → (m, n)."""
    D = np.abs(x1[:, None] - x2[None, :])
    s = np.sqrt(3.0) * D / ls
    return outputscale * (1.0 + s) * np.exp(-s)


# ---------------------------------------------------------------------------
# Per-curve worker
# ---------------------------------------------------------------------------

def _worker_fn(args: tuple) -> list:
    # Set thread limits in the child process too (Windows spawn inherits env
    # vars but BLAS DLLs are re-loaded, so we re-assert the limits here).
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    gr, n_bootstrap = args
    return _bootstrap_curve(gr, n_bootstrap)


def _bootstrap_curve(gr: dict, n_bootstrap: int) -> list[dict]:
    """
    Run bootstrap replication for one I/O curve.

    Session-1 features come in two flavours:
      - Posterior-sample version (mean, std) -- stored for the scatter plot.
      - Mean-curve version                   -- used as the delta baseline.

    Session-2 features are always mean-curve (n_samples=0 per bootstrap fit).
    Using the mean curve for deltas removes within-fit Monte-Carlo noise and
    makes sessions 1 and 2 comparable.
    """
    pid  = gr["ID"]
    side = gr["side"]
    freq = gr["f"]

    # ------------------------------------------------------------------
    # Reference fit (session 1)
    # ------------------------------------------------------------------
    ref = fit_gp(gr, mean_type=MEAN_TYPE, n_samples=N_SAMPLES_REF)

    l2  = ref.l2_used      # observed L2 levels after finite-value mask
    dp  = ref.dp_used      # observed DP values
    n2  = ref.noise2_used  # noise floor values
    hp  = ref._hyperparams # (lengthscale, outputscale) -- fixed for bootstrap

    # Noise model fixed from reference: it's a property of the probe fit /
    # measurement setup, not the true DPOAE level.
    noise_var = _obs_noise_var(dp, n2)

    # GP posterior mean at observed L2 levels (dense-grid interpolation)
    gp_mean_at_obs = np.interp(l2, ref.l2_grid, ref.mean)

    # Residuals: observed DP minus GP posterior mean
    residuals = dp - gp_mean_at_obs

    # Session-1 features
    # -- posterior-sample version (for scatter plot comparison)
    knee_std_s1 = ref.features["knee"].mean            # posterior mean
    mc_std_s1   = ref.features["max_compression"].mean
    knee_gp_std = ref.features["knee"].std             # posterior SD
    mc_gp_std   = ref.features["max_compression"].std

    # -- mean-curve version (delta baseline; no Monte-Carlo noise)
    knee_s1_mc = _knee_from_sample(ref.l2_grid, ref.mean)
    mc_s1_mc   = _max_compression_from_sample(ref.l2_grid, ref.mean)

    # ------------------------------------------------------------------
    # Precompute prediction weight matrix W = K_sx @ K_noisy^{-1}
    # Shape: (n_grid, n_obs). Fixed for all bootstrap reps because
    # hyperparameters and noise are held constant (empirical Bayes).
    # Per-rep posterior mean: mean_grid + W @ (dp_syn - mean_train)
    # ------------------------------------------------------------------
    ls, outputscale = hp
    K_xx    = _matern32_kernel(l2, l2, ls, outputscale)
    K_noisy = K_xx + np.diag(noise_var)
    K_sx    = _matern32_kernel(ref.l2_grid, l2, ls, outputscale)
    # solve K_noisy @ X = K_sx.T  →  X = K_noisy^{-1} K_sx.T  →  W = X.T
    W = np.linalg.solve(K_noisy, K_sx.T).T   # (n_grid, n_obs)

    # Deterministic seed from curve identity
    seed = hash(pid + side + str(freq)) & 0x7FFFFFFF
    rng  = np.random.default_rng(seed)

    rows = []
    for rep in range(n_bootstrap):
        # Bootstrap: resample residuals with replacement
        r_boot       = rng.choice(residuals, size=len(residuals), replace=True)
        dp_synthetic = gp_mean_at_obs + r_boot

        try:
            # Re-estimate linear mean from synthetic data; kernel weight W is fixed
            coeffs     = np.polyfit(l2, dp_synthetic, 1)
            mean_train = np.polyval(coeffs, l2)
            mean_grid  = np.polyval(coeffs, ref.l2_grid)
            post_mean  = mean_grid + W @ (dp_synthetic - mean_train)

            knee_s2 = _knee_from_sample(ref.l2_grid, post_mean)
            mc_s2   = _max_compression_from_sample(ref.l2_grid, post_mean)
        except Exception:
            knee_s2 = np.nan
            mc_s2   = np.nan

        rows.append({
            "participant":  pid,
            "side":         side,
            "freq":         freq,
            "rep":          rep,
            # Session-1 posterior features (for scatter comparison)
            "knee_post_mean": knee_std_s1,
            "mc_post_mean":   mc_std_s1,
            "knee_gp_std":    knee_gp_std,
            "mc_gp_std":      mc_gp_std,
            # Session-1 mean-curve features (delta baseline)
            "knee_s1":      knee_s1_mc,
            "mc_s1":        mc_s1_mc,
            # Session-2 mean-curve features
            "knee_s2":      knee_s2,
            "mc_s2":        mc_s2,
            # Signed differences (mean curve vs mean curve)
            "delta_knee":   float(knee_s2 - knee_s1_mc)
                            if np.isfinite(knee_s2) else np.nan,
            "delta_mc":     float(mc_s2 - mc_s1_mc)
                            if np.isfinite(mc_s2) else np.nan,
        })

    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_bootstrap(n_bootstrap: int = N_BOOTSTRAP,
                  n_workers:   int = N_WORKERS) -> pd.DataFrame:

    OAEs = load_from_pickle()
    all_curves = [
        gr
        for rec in OAEs
        for gr in rec.get("GR", [])
        if len(gr.get("f2", [])) >= 3
    ]

    print(f"Curves          : {len(all_curves)}")
    print(f"Bootstrap reps  : {n_bootstrap}")
    print(f"Workers         : {n_workers}")
    print(f"Fits per curve  : 1 reference (sampled) + {n_bootstrap} conditioned "
          f"(mean-curve only)")
    print(f"Total GP fits   : ~{len(all_curves) * (1 + n_bootstrap):,}\n")

    work = [(gr, n_bootstrap) for gr in all_curves]

    all_rows    = []
    n_done      = 0
    first_write = True

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_fn, args): args[0] for args in work}

        pbar = tqdm(as_completed(futures), total=len(futures),
                    unit="curve", ncols=80)

        for future in pbar:
            gr = futures[future]
            pid, side, freq = gr["ID"], gr["side"], gr["f"]
            try:
                rows = future.result()
                all_rows.extend(rows)
                n_done += 1
                pbar.set_postfix({"last": f"{pid} {side} {freq}"})
            except Exception as e:
                tqdm.write(f"  FAILED {pid} {side} {freq}: {e}")
                continue

            if n_done % 8 == 0:
                _save(all_rows, RESULTS_PATH, append=not first_write)
                all_rows    = []
                first_write = False

    if all_rows:
        _save(all_rows, RESULTS_PATH, append=not first_write)

    print(f"\nResults saved to {RESULTS_PATH}")
    return pd.read_csv(RESULTS_PATH)


def _save(rows: list, path: Path, append: bool):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> dict:
    out = {}
    for feat, col_d, col_gp_std in [
        ("knee",            "delta_knee", "knee_gp_std"),
        ("max_compression", "delta_mc",   "mc_gp_std"),
    ]:
        deltas       = df[col_d].dropna()
        # Bootstrap MDC: 1.96 * SD(delta) -- directly comparable to Reavis (2008)
        out[f"{feat}_bootstrap_mdc"] = 1.96 * deltas.std()
        out[f"{feat}_bootstrap_sd"]  = deltas.std()

        # GP-estimated MDC: 1.96 * sqrt(2) * mean(GP posterior SD across curves)
        mean_gp_std = df.groupby(["participant","side","freq"])[col_gp_std].first().mean()
        out[f"{feat}_gp_mdc"]        = 1.96 * np.sqrt(2) * mean_gp_std
        out[f"{feat}_gp_std_mean"]   = mean_gp_std

    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def figure_bootstrap(df: pd.DataFrame, show: bool = False):
    """
    Two-panel figure.

    (a) Violin plots of |delta_knee| and |delta_mc|.
        Horizontal dashed lines at 6 dB (clinical threshold) and at the
        GP-estimated MDC (dotted, per feature), so both quantities are visible.

    (b) Per-curve scatter: bootstrap SD(delta) / sqrt(2)  vs  GP posterior SD.
        Both axes are single-session noise estimates, so they are comparable.
        If bootstrap SD / sqrt(2) ~ GP posterior SD, the GP captures most of
        the variability.
        Reference lines: y = x  and  y = sqrt(2) * x.
    """
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize":7,
        "ytick.labelsize":7,
        "legend.fontsize":7,
        "lines.linewidth":1.5,
    })

    C_KNEE = "#2166ac"
    C_MC   = "#d6604d"

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0))
    fig.subplots_adjust(wspace=0.38)

    # ----------------------------------------------------------------
    # Panel (a): violin plots of |delta|
    # ----------------------------------------------------------------
    ax = axes[0]

    abs_knee = df["delta_knee"].dropna().abs().values
    abs_mc   = df["delta_mc"].dropna().abs().values

    vp = ax.violinplot(
        [abs_knee, abs_mc], positions=[1, 2],
        showmedians=True, showextrema=False,
    )
    for body, color in zip(vp["bodies"], [C_KNEE, C_MC]):
        body.set_facecolor(color)
        body.set_alpha(0.45)
    vp["cmedians"].set_colors([C_KNEE, C_MC])
    vp["cmedians"].set_linewidth(2.0)

    # GP-estimated MDC lines (mean across curves)
    gp_std_knee = df.groupby(["participant","side","freq"])["knee_gp_std"].first().mean()
    gp_std_mc   = df.groupby(["participant","side","freq"])["mc_gp_std"].first().mean()
    gp_mdc_knee = 1.96 * np.sqrt(2) * gp_std_knee
    gp_mdc_mc   = 1.96 * np.sqrt(2) * gp_std_mc
    boot_mdc_knee = 1.96 * abs_knee.std()
    boot_mdc_mc   = 1.96 * abs_mc.std()

    ax.hlines(CLINICAL_THRESHOLD, 0.55, 2.45, colors="k",
              linestyles="--", linewidth=1.0, label=f"{CLINICAL_THRESHOLD} dB threshold")
    ax.hlines(gp_mdc_knee, 0.55, 1.45, colors=C_KNEE,
              linestyles=":", linewidth=1.2, label="GP-estimated MDC")
    ax.hlines(gp_mdc_mc,   1.55, 2.45, colors=C_MC,
              linestyles=":", linewidth=1.2)

    for x, val_gp, val_boot, color in [
        (1.0, gp_mdc_knee, boot_mdc_knee, C_KNEE),
        (2.0, gp_mdc_mc,   boot_mdc_mc,   C_MC),
    ]:
        ymax = max(val_gp, val_boot)
        ax.annotate(
            f"GP: {val_gp:.1f} dB\nBoot: {val_boot:.1f} dB",
            xy=(x, ymax), xytext=(x, ymax + 0.8),
            ha="center", va="bottom", fontsize=6, color=color,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Knee", "Max-compression"])
    ax.set_ylabel("|$\\Delta$ feature| (dB SPL)")
    ax.set_title("(a) Bootstrap test-retest differences")
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")

    # ----------------------------------------------------------------
    # Panel (b): per-curve scatter
    # ----------------------------------------------------------------
    ax = axes[1]

    per = df.groupby(["participant","side","freq"]).agg(
        boot_sd_knee = ("delta_knee", "std"),
        boot_sd_mc   = ("delta_mc",   "std"),
        gp_std_knee  = ("knee_gp_std", "first"),
        gp_std_mc    = ("mc_gp_std",   "first"),
    ).reset_index()

    # Divide by sqrt(2) to put bootstrap on single-session scale
    # (comparable to GP posterior SD, which is also per-session)
    boot_ss_knee = per["boot_sd_knee"] / np.sqrt(2)
    boot_ss_mc   = per["boot_sd_mc"]   / np.sqrt(2)

    ax.scatter(per["gp_std_knee"], boot_ss_knee,
               color=C_KNEE, alpha=0.55, s=14, label="Knee", zorder=3)
    ax.scatter(per["gp_std_mc"], boot_ss_mc,
               color=C_MC,   alpha=0.55, s=14, label="Max-compression", zorder=3)

    lim = max(
        per[["gp_std_knee","gp_std_mc"]].max().max(),
        max(boot_ss_knee.max(), boot_ss_mc.max()),
    ) * 1.08
    xs = np.array([0.0, lim])
    ax.plot(xs, xs,              color="k", lw=0.8, ls="--", label="$y=x$")
    ax.plot(xs, np.sqrt(2) * xs, color="k", lw=0.8, ls=":",
            label="$y=\\sqrt{2}\\,x$")

    ax.set_xlabel("GP posterior SD, session 1 (dB)")
    ax.set_ylabel("Bootstrap SD($\\Delta$) / $\\sqrt{2}$ (dB)")
    ax.set_title("(b) Per-curve: bootstrap vs GP uncertainty")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", fontsize=6)

    fig.savefig(str(FIG_PATH) + ".pdf", bbox_inches="tight")
    fig.savefig(str(FIG_PATH) + ".png", bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Figure saved: {FIG_PATH}.pdf / .png")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test(n_bootstrap: int = 30):
    import time
    OAEs = load_from_pickle()
    gr   = OAEs[0]["GR"][0]
    pid, side, freq = gr["ID"], gr["side"], gr["f"]

    print(f"Smoke test: {pid} {side} {freq} Hz  ({n_bootstrap} reps)")
    t0   = time.time()
    rows = _bootstrap_curve(gr, n_bootstrap)
    dt   = time.time() - t0

    df = pd.DataFrame(rows)
    print(f"  Time           : {dt:.1f} s  ({dt/n_bootstrap*1000:.0f} ms/rep)")
    print(f"  delta_knee SD  : {df['delta_knee'].std():.2f} dB")
    print(f"  delta_mc   SD  : {df['delta_mc'].std():.2f} dB")
    print(f"  Bootstrap MDC knee: {1.96*df['delta_knee'].std():.2f} dB")
    print(f"  Bootstrap MDC mc  : {1.96*df['delta_mc'].std():.2f} dB")
    print(f"  GP posterior SD knee: {df['knee_gp_std'].iloc[0]:.2f} dB "
          f"-> GP-est. MDC {1.96*np.sqrt(2)*df['knee_gp_std'].iloc[0]:.2f} dB")
    print(f"  GP posterior SD mc  : {df['mc_gp_std'].iloc[0]:.2f} dB "
          f"-> GP-est. MDC {1.96*np.sqrt(2)*df['mc_gp_std'].iloc[0]:.2f} dB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--workers",     type=int, default=N_WORKERS)
    parser.add_argument("--smoke",       action="store_true",
                        help="Single-curve smoke test (30 reps)")
    parser.add_argument("--plot-only",   action="store_true",
                        help="Regenerate figure from existing CSV")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if args.smoke:
        _smoke_test(n_bootstrap=30)

    elif args.plot_only:
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(
                f"{RESULTS_PATH} not found — run without --plot-only first.")
        df = pd.read_csv(RESULTS_PATH)
        summary = compute_summary(df)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v:.2f} dB")
        figure_bootstrap(df)

    else:
        df = run_bootstrap(n_bootstrap=args.n_bootstrap, n_workers=args.workers)
        summary = compute_summary(df)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v:.2f} dB")
        figure_bootstrap(df)
