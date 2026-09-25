"""
Publication figures for the GP DPOAE I/O paper.

Figures
-------
1  example_curve      — GP fit + 95 % CI + feature posteriors vs cubic (no CI)
2  rmse_vs_sparsity   — RMSE vs k for GP and cubic, 3 strategies
3  feature_uncertainty — CI width and MDC vs k with 6 dB clinical threshold
4  coverage           — fraction of curves where sparse GP CI covers the reference
5  cohort_features    — cohort distributions of GP features at k=10

Standalone figures (no experiment CSV needed):
    figure1_example_curve()
    figure5_cohort_features()

Experiment-dependent figures (need outputs/experiment_results.csv):
    figure2_rmse_vs_sparsity()
    figure3_feature_uncertainty()
    figure4_coverage()

Run:
    python figures.py --all          # all figures
    python figures.py --fig 1        # single figure
    python figures.py --fig 1 --show # display interactively
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

from config import OUTPUT_DIR
from load_data import load_from_pickle
from gp_fitting import fit_gp, fit_cubic, fit_sparse

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
})

# Colour palette — colour-blind safe
C_GP     = "#2166ac"   # blue  — GP
C_CUBIC  = "#d6604d"   # red   — cubic
C_ACTIVE = "#1a9850"   # green — active strategy
C_EQUAL  = "#f4a582"   # peach — equalspaced
C_RAND   = "#999999"   # grey  — random
C_NOISE  = "#cccccc"   # light grey — noise floor

# Sizes: JASA single column ~3.25 in, double column ~6.75 in
W1 = 3.25
W2 = 6.75

CLINICAL_THRESHOLD = 6.0   # dB — Reavis 2008

K_VALUES = [3, 4, 5, 6, 7, 8, 9]


def _save(fig, name: str, show: bool = False):
    exts = ("pdf", "png", "svg")
    for ext in exts:
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / name}." + " / .".join(exts))
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Example curve: GP vs cubic + feature posteriors
# ---------------------------------------------------------------------------

def figure1_example_curve(participant_idx: int = 0, task_idx: int = 1,
                           n_samples: int = 500, show: bool = False):
    """
    Three-panel figure for one I/O curve.
      (a) GP posterior mean + 95 % CI, data, noise floor, feature annotations
      (b) Cubic fit (same data, no CI) — illustrates absence of uncertainty
      (c) Posterior distributions of knee and max-compression from GP samples
    """
    OAEs = load_from_pickle()
    gr   = OAEs[participant_idx]["GR"][task_idx]
    pid, side, freq = gr["ID"], gr["side"], gr["f"]

    res_gp  = fit_gp(gr, n_samples=n_samples)
    res_cub = fit_cubic(gr)

    l2_data  = np.asarray(gr["f2"],    dtype=float)
    dp_data  = np.asarray(gr["dp"],    dtype=float)
    nf_data  = np.asarray(gr["noise2"],dtype=float)
    detected = dp_data > (nf_data + 3.0)

    fig = plt.figure(figsize=(W2, 3.05))
    gs  = gridspec.GridSpec(1, 3, wspace=0.42, left=0.075, right=0.98,
                            top=0.90, bottom=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ---- Panel (a): GP ------------------------------------------------
    for ax in (ax1, ax2):
        ax.fill_between(l2_data, nf_data, nf_data.min() - 5,
                        alpha=0.18, color=C_NOISE, label="Noise floor")
        ax.plot(l2_data, nf_data, "--", color=C_NOISE, lw=1.0, label="_nf")

    # GP posterior
    ax1.fill_between(res_gp.l2_grid, res_gp.lower95, res_gp.upper95,
                     alpha=0.25, color=C_GP, label="95 % CI")
    ax1.plot(res_gp.l2_grid, res_gp.mean, "-", color=C_GP, label="GP mean")

    # Cubic
    ax2.plot(res_cub.l2_grid, res_cub.mean, "-", color=C_CUBIC, label="Cubic fit")

    for ax in (ax1, ax2):
        ax.scatter(l2_data[detected],  dp_data[detected],
                   s=30, zorder=5, color="k", label="Detected")
        ax.scatter(l2_data[~detected], dp_data[~detected],
                   s=30, zorder=5, color="k", marker="v",
                   facecolors="none", label="Below noise floor")

    _annotate_features(ax1, res_gp)

    for ax in (ax1, ax2):
        ax.set_xlabel("Stimulus level $L_2$ (dB SPL)")
        ax.set_xlim(17, 68)

    ax1.set_ylabel("DPOAE amplitude (dB SPL)")
    ax1.set_title(f"(a) GP — {pid} {side} {freq} Hz")
    ax2.set_title("(b) Cubic fit (no CI)")

    # Shared legend for (a) and (b), placed below the axes so it never
    # overlaps the growth function.
    h_all, lab_all = ax1.get_legend_handles_labels()
    for h, lab in zip(*ax2.get_legend_handles_labels()):
        if lab not in lab_all:
            h_all.append(h)
            lab_all.append(lab)
    fig.legend(h_all, lab_all, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False, fontsize=7, columnspacing=1.6,
               handlelength=1.6, handletextpad=0.5, labelspacing=0.4)

    # ---- Panel (c): Feature posterior distributions ---------------------
    knee   = res_gp.features["knee"]
    mcomp  = res_gp.features["max_compression"]
    knee_samples  = knee.samples
    mcomp_samples = mcomp.samples

    bins = np.linspace(25, 65, 30)
    ax3.hist(knee_samples,  bins=bins, alpha=0.6, color=C_GP,    label="Knee (D)",     density=True)
    ax3.hist(mcomp_samples, bins=bins, alpha=0.6, color=C_CUBIC, label="Max-comp (*)", density=True)

    # Vertical lines: means (ymax keeps them clear of the legend)
    _VLINE_YMAX = 0.68
    ax3.axvline(knee.mean,  color=C_GP,    lw=1.5, linestyle="--", ymax=_VLINE_YMAX)
    ax3.axvline(mcomp.mean, color=C_CUBIC, lw=1.5, linestyle="--", ymax=_VLINE_YMAX)

    # Cubic point estimates
    ax3.axvline(res_cub.features["knee"],            color=C_GP,    lw=1.0,
                linestyle=":", ymax=_VLINE_YMAX)
    ax3.axvline(res_cub.features["max_compression"], color=C_CUBIC, lw=1.0,
                linestyle=":", ymax=_VLINE_YMAX)

    ax3.set_xlabel("Feature value (dB SPL)")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) Feature posteriors")

    # Headroom so the legend clears the tallest bar, and fold the
    # dotted-line explanation into the legend instead of a footnote.
    ax3.set_ylim(top=ax3.get_ylim()[1] * 1.45)
    h_c, lab_c = ax3.get_legend_handles_labels()
    h_c.append(Line2D([], [], color="grey", lw=1.0, ls=":"))
    lab_c.append("Cubic estimate")
    ax3.legend(h_c, lab_c, frameon=False, fontsize=6.5, loc="upper left",
               handlelength=1.4, handletextpad=0.5, borderaxespad=0.2,
               labelspacing=0.35)

    _save(fig, "fig1_example_curve", show)


def _annotate_features(ax, res):
    """Annotate knee (◆) and max-compression (⭐) on an axis."""
    knee  = res.features["knee"]
    mcomp = res.features["max_compression"]
    y_knee  = np.interp(knee.mean,  res.l2_grid, res.mean)
    y_mcomp = np.interp(mcomp.mean, res.l2_grid, res.mean)

    ax.plot(knee.mean,  y_knee,  marker="D", ms=8,
            color=C_GP, zorder=6, label=f"Knee  {knee.mean:.0f} dB SPL")
    ax.plot(mcomp.mean, y_mcomp, marker="*", ms=10,
            color=C_CUBIC, zorder=6, label=f"Max-comp  {mcomp.mean:.0f} dB SPL")

    # CI error bars on feature positions
    ax.errorbar(knee.mean,  y_knee,
                xerr=[[max(0, knee.mean  - knee.ci_low)],
                      [max(0, knee.ci_high  - knee.mean)]],
                fmt="none", color=C_GP, capsize=3, lw=1.2)
    ax.errorbar(mcomp.mean, y_mcomp,
                xerr=[[max(0, mcomp.mean - mcomp.ci_low)],
                      [max(0, mcomp.ci_high - mcomp.mean)]],
                fmt="none", color=C_CUBIC, capsize=3, lw=1.2)


# Colour palette for GP mean types
C_TWOSLOPE  = "#1a9850"   # green
C_QUADRATIC = "#8073ac"   # purple
C_LINEAR    = "#2166ac"   # blue  (= C_GP)
C_CONSTANT  = "#aaaaaa"   # light grey

# Model display config: (colour, label)
MODEL_STYLE = {
    "twoslope":  (C_TWOSLOPE,  "GP two-slope"),
    "linear":    (C_LINEAR,    "GP linear"),
    "quadratic": (C_QUADRATIC, "GP quadratic"),
    "cubic":     (C_CUBIC,     "GP cubic"),
    "constant":  (C_CONSTANT,  "GP constant"),
    "det_cubic": (C_CUBIC,     "Cubic (deterministic)"),
}

def _legend_below(ax, ncol=2, fontsize=7):
    """Put an axis legend underneath the axes, clear of the plotted data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol,
              frameon=False, fontsize=fontsize, columnspacing=1.2,
              handlelength=2.0, handletextpad=0.5, labelspacing=0.4)


# ---------------------------------------------------------------------------
# Figure 2 — RMSE vs sparsity k
# ---------------------------------------------------------------------------

def figure2_rmse_vs_sparsity(show: bool = False):
    """
    Two-panel figure.
      (a) Model comparison: all GP mean types + deterministic cubic, active
          strategy only (vs GP reference). Shows which mean function gives
          best reconstruction.
      (b) Strategy comparison: GP two-slope and deterministic cubic, all three
          strategies (active / equalspaced / random). Shows benefit of
          structured acquisition.
    Error bands = ±1 SE across curves.
    """
    df  = _load_results()
    col = "rmse_gp_vs_ref_held"   # primary metric — vs reference GP

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.6), sharey=False)
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.98, top=0.87, bottom=0.36)

    # ---- Panel (a): model comparison, active strategy ----------------------
    ax = axes[0]
    gp_models = ["twoslope", "linear", "quadratic", "cubic", "constant"]
    for mt in gp_models:
        c_col = col.replace("rmse_gp", f"rmse_gp_{mt}")
        if c_col not in df.columns:
            continue
        color, label = MODEL_STYLE[mt]
        ls = "-"
        _plot_agg_line(ax, df[df["strategy"] == "active"], c_col,
                       color=color, ls=ls, label=label)

    # deterministic cubic
    det_col = "rmse_cubic_vs_ref_held"
    if det_col in df.columns:
        _plot_agg_line(ax, df[df["strategy"] == "active"], det_col,
                       color=C_CUBIC, ls="--", label="Cubic (det.)")

    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Retained levels $k$")
    ax.set_ylabel("RMSE (dB SPL)")
    ax.set_title("(a) Mean function comparison\n(active acquisition)")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(bottom=0)
    _legend_below(ax, ncol=2)

    # ---- Panel (b): strategy comparison, GP linear (best calibrated model) --
    ax = axes[1]
    lin_col = col.replace("rmse_gp", "rmse_gp_linear")

    strat_style = {
        "active":      (C_LINEAR, "-",  "GP linear — active"),
        "equalspaced": (C_LINEAR, "--", "GP linear — equal-spaced"),
        "random":      (C_LINEAR, ":",  "GP linear — random"),
    }
    for strategy, (color, ls, label) in strat_style.items():
        sub = df[df["strategy"] == strategy]
        if lin_col in df.columns:
            _plot_agg_line(ax, sub, lin_col, color=color, ls=ls, label=label)

    if det_col in df.columns:
        for strategy, ls, label in [
            ("active",      "-",  "Cubic (det.) — active"),
            ("random",      ":",  "Cubic (det.) — random"),
        ]:
            _plot_agg_line(ax, df[df["strategy"] == strategy], det_col,
                           color=C_CUBIC, ls=ls, label=label)

    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Retained levels $k$")
    ax.set_ylabel("RMSE (dB SPL)")
    ax.set_title("(b) Acquisition strategy comparison\n(GP linear mean)")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(bottom=0)
    _legend_below(ax, ncol=2)

    _save(fig, "fig2_rmse_vs_sparsity", show)


def _plot_agg_line(ax, sub_df, col, color, ls, label):
    """Aggregate a column over curves (mean ± SE) and plot."""
    agg = (sub_df.groupby(["participant", "side", "freq", "k"])[col]
                 .mean().reset_index()
                 .groupby("k")[col]
                 .agg(["mean", "sem"])
                 .reindex(K_VALUES))
    ks, mu, se = agg.index.values, agg["mean"].values, agg["sem"].values
    valid = np.isfinite(mu)
    ax.plot(ks[valid], mu[valid], color=color, ls=ls, lw=1.5, label=label)
    ax.fill_between(ks[valid], (mu-se)[valid], (mu+se)[valid],
                    alpha=0.10, color=color)


# ---------------------------------------------------------------------------
# Figure 3 — Feature uncertainty vs sparsity (CI width + MDC)
# ---------------------------------------------------------------------------

def figure3_feature_uncertainty(show: bool = False):
    """
    Two-panel figure showing MDC vs k for the max-compression feature.
      (a) Model comparison: MDC for GP two-slope, linear, quadratic, cubic
          mean — active strategy. Shows how mean choice affects uncertainty.
      (b) Strategy comparison: MDC for GP two-slope — active vs random.
          Shows benefit of structured acquisition for feature precision.
    6 dB clinical threshold shown on both panels.
    """
    df = _load_results()

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.6))
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.98, top=0.87, bottom=0.36)

    feat      = "max_compression"
    feat_knee = "knee"

    # ---- Panel (a): model comparison, active, max-compression MDC ----------
    ax = axes[0]
    for mt in ["twoslope", "linear", "quadratic", "cubic"]:
        col = f"gp_{mt}_sparse_{feat}_mdc"
        if col not in df.columns:
            continue
        color, label = MODEL_STYLE[mt]
        sub = df[df["strategy"] == "active"]
        _plot_agg_line(ax, sub, col, color=color, ls="-", label=label)

    ax.axhline(CLINICAL_THRESHOLD, color="k", lw=1.2, ls="--", alpha=0.7,
               label=f"{CLINICAL_THRESHOLD} dB threshold")
    ax.set_xlabel("Retained levels $k$")
    ax.set_ylabel("Estimated MDC (dB SPL)")
    ax.set_title(f"(a) Max-compression est. MDC\n(model comparison, active)")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(bottom=0)
    _legend_below(ax, ncol=2)

    # ---- Panel (b): strategy comparison, GP two-slope, both features -------
    ax = axes[1]
    lin_mdc_mc   = f"gp_linear_sparse_{feat}_mdc"
    lin_mdc_knee = f"gp_linear_sparse_{feat_knee}_mdc"

    for strategy, ls in [("active", "-"), ("random", ":")]:
        sub = df[df["strategy"] == strategy]
        if lin_mdc_mc in df.columns:
            _plot_agg_line(ax, sub, lin_mdc_mc,
                           color=C_CUBIC,  ls=ls,
                           label=f"Max-comp — {strategy}")
        if lin_mdc_knee in df.columns:
            _plot_agg_line(ax, sub, lin_mdc_knee,
                           color=C_LINEAR, ls=ls,
                           label=f"Knee — {strategy}")

    ax.axhline(CLINICAL_THRESHOLD, color="k", lw=1.2, ls="--", alpha=0.7,
               label=f"{CLINICAL_THRESHOLD} dB threshold")
    ax.set_xlabel("Retained levels $k$")
    ax.set_ylabel("Estimated MDC (dB SPL)")
    ax.set_title("(b) Est. MDC by feature and strategy\n(GP linear mean)")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(bottom=0)
    _legend_below(ax, ncol=2)

    _save(fig, "fig3_feature_uncertainty", show)


# ---------------------------------------------------------------------------
# Figure 4 — Coverage: does sparse GP CI cover the reference value?
# ---------------------------------------------------------------------------

def figure4_coverage(show: bool = False):
    """
    Coverage rate: fraction of curves where the sparse GP 95 % CI
    contains the full-data (k=10) GP feature estimate.
    Plotted vs k for active, equalspaced, and random strategies.
    95 % nominal coverage shown as dashed horizontal line.

    Coverage is computed on-the-fly by joining experiment_results.csv
    (sparse CI bounds) with reference_results.csv (k=10 feature means).
    """
    df  = _load_results()
    ref = _load_reference()

    # Reference feature means per curve (k=10 GP-linear)
    ref_cols = {
        "knee":            "gp_linear_knee_mean",
        "max_compression": "gp_linear_max_compression_mean",
    }
    ref_key = ref[["participant", "side", "freq"]].copy()
    for fname, rcol in ref_cols.items():
        ref_key[f"ref_{fname}"] = ref[rcol]

    df = df.merge(ref_key, on=["participant", "side", "freq"], how="left").copy()

    # Compute coverage flags for GP-linear and GP-twoslope
    for model in ("linear", "twoslope"):
        for fname in ref_cols:
            ci_lo = f"gp_{model}_sparse_{fname}_ci_low"
            ci_hi = f"gp_{model}_sparse_{fname}_ci_high"
            key   = f"covers_{model}_{fname}"
            if ci_lo in df.columns and ci_hi in df.columns:
                df[key] = (
                    (df[f"ref_{fname}"] >= df[ci_lo]) &
                    (df[f"ref_{fname}"] <= df[ci_hi])
                ).astype(float)
            else:
                df[key] = np.nan
    # backward-compat alias used below
    for fname in ref_cols:
        df[f"covers_{fname}"] = df.get(f"covers_linear_{fname}", np.nan)

    features = {
        "knee":            ("Knee",      C_GP),
        "max_compression": ("Max-comp",  C_CUBIC),
    }

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.8))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.97, top=0.88, bottom=0.15)

    strat_style = {"active": "-", "equalspaced": "--", "random": ":"}
    # Show GP-linear and GP-twoslope side by side within each feature panel
    model_style = {
        "linear":   (C_LINEAR,   "GP linear"),
        "twoslope": (C_TWOSLOPE, "GP two-slope"),
    }

    for ax, (fname, (flabel, fcolor)) in zip(axes, features.items()):
        plotted = False
        for model, (mcolor, mlabel) in model_style.items():
            col = f"covers_{model}_{fname}"
            if col not in df.columns or df[col].isna().all():
                col = f"covers_{fname}"   # fallback
            if col not in df.columns or df[col].isna().all():
                continue

            for strategy, ls in strat_style.items():
                sub = df[df["strategy"] == strategy].copy()
                agg = (sub.groupby(["participant", "side", "freq", "k"])[col]
                          .mean().reset_index()
                          .groupby("k")[col].mean()
                          .reindex(K_VALUES))
                ks = agg.index.values
                cv = agg.values * 100
                valid = np.isfinite(cv)
                lbl = f"{mlabel} — {strategy}" if len(model_style) > 1 else strategy.capitalize()
                ax.plot(ks[valid], cv[valid], color=mcolor, ls=ls,
                        lw=1.5, label=lbl)
                plotted = True

        if not plotted:
            ax.set_title(f"(?) {flabel} — data not available")
            continue

        ax.axhline(95, color="k", lw=1.2, linestyle="--", alpha=0.7,
                   label="Nominal 95 %")
        ax.set_xlabel("Retained levels $k$")
        ax.set_ylabel("Coverage (%)")
        ax.set_title(f"{flabel} feature")
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 105)
        ax.legend(frameon=False, fontsize=6.5)

    fig.suptitle("Coverage: fraction of curves where sparse GP 95 % CI contains"
                 " the $k=10$ reference estimate", fontsize=8, y=1.01)

    _save(fig, "fig4_coverage", show)


# ---------------------------------------------------------------------------
# Figure 5 — Cohort feature distributions at k=10
# ---------------------------------------------------------------------------

def figure5_cohort_features(n_samples: int = 300, show: bool = False):
    """
    Cohort-level distributions of GP features (knee, max-compression)
    at k=10, grouped by frequency (1414 vs 4243 Hz).
    Individual CI bars overlaid on violin plots.
    """
    OAEs = load_from_pickle()

    records = []
    for rec in OAEs:
        for gr in rec.get("GR", []):
            if len(gr.get("f2", [])) < 4:
                continue
            try:
                res = fit_gp(gr, n_samples=n_samples)
                for fname in ("knee", "max_compression"):
                    feat = res.features[fname]
                    records.append({
                        "participant": gr["ID"],
                        "side":        gr["side"],
                        "freq":        gr["f"],
                        "feature":     fname,
                        "mean":        feat.mean,
                        "ci_low":      feat.ci_low,
                        "ci_high":     feat.ci_high,
                        "ci_width":    feat.ci_width,
                    })
            except Exception:
                continue

    if not records:
        print("  No cohort data — run after pickle is available.")
        return

    df = pd.DataFrame(records)
    df["freq_label"] = df["freq"].map({1414: "1.4 kHz", 4243: "4.2 kHz"})

    feat_labels = {"knee": "Knee $L_2$ (dB SPL)",
                   "max_compression": "Max-compression $L_2$ (dB SPL)"}

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2))
    fig.subplots_adjust(wspace=0.40, left=0.10, right=0.97, top=0.88, bottom=0.12)

    for ax, fname in zip(axes, ("knee", "max_compression")):
        sub = df[df["feature"] == fname]

        freqs  = ["1.4 kHz", "4.2 kHz"]
        colors = [C_GP, C_CUBIC]
        positions = [1, 2]

        for pos, (freq_label, color) in enumerate(zip(freqs, colors), start=1):
            data = sub[sub["freq_label"] == freq_label]["mean"].dropna().values
            if len(data) == 0:
                continue

            parts = ax.violinplot(data, positions=[pos], widths=0.6,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(2)

            # Overlay individual CI bars
            ci_sub = sub[sub["freq_label"] == freq_label]
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(ci_sub))
            for j, (_, row) in zip(jitter, ci_sub.iterrows()):
                if not np.isfinite(row["mean"]):
                    continue
                ax.plot(pos + j, row["mean"], "o", ms=3, color=color,
                        alpha=0.7, zorder=5)
                ax.plot([pos + j, pos + j],
                        [row["ci_low"], row["ci_high"]],
                        "-", lw=0.6, color=color, alpha=0.5, zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(freqs)
        ax.set_ylabel(feat_labels[fname])
        ax.set_title(fname.replace("_", " ").capitalize())

    fig.suptitle("Cohort GP feature distributions ($k=10$, $n=38$)\n"
                 "Violin = distribution; dots = individual means; bars = 95 % CI",
                 fontsize=8)

    _save(fig, "fig5_cohort_features", show)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results() -> pd.DataFrame:
    path = OUTPUT_DIR / "experiment_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment results not found: {path}\n"
            "Run:  python experiment.py --fast   (RMSE only)\n"
            "  or: python experiment.py           (full, includes features)"
        )
    return pd.read_csv(path)


def _load_reference() -> pd.DataFrame:
    path = OUTPUT_DIR / "reference_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Reference results not found: {path}\n"
            "Run:  python experiment.py"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",  action="store_true", help="Generate all figures")
    parser.add_argument("--fig",  type=int, default=None,
                        help="Generate single figure (1-5)")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")
    args = parser.parse_args()

    figs = {
        1: lambda: figure1_example_curve(show=args.show),
        2: lambda: figure2_rmse_vs_sparsity(show=args.show),
        3: lambda: figure3_feature_uncertainty(show=args.show),
        4: lambda: figure4_coverage(show=args.show),
        5: lambda: figure5_cohort_features(show=args.show),
    }

    if args.all:
        for n, fn in figs.items():
            print(f"\n--- Figure {n} ---")
            try:
                fn()
            except FileNotFoundError as e:
                print(f"  Skipped: {e}")
    elif args.fig is not None:
        if args.fig not in figs:
            print(f"Unknown figure number {args.fig}. Choose 1-5.")
        else:
            print(f"\n--- Figure {args.fig} ---")
            figs[args.fig]()
    else:
        # Default: generate figures 1 and 5 (no experiment CSV needed)
        print("Generating figures 1 and 5 (no experiment CSV needed)...")
        print("\n--- Figure 1 ---")
        figure1_example_curve(show=args.show)
        print("\n--- Figure 5 ---")
        figure5_cohort_features(show=args.show)
        print("\nFor figures 2–4, first run: python experiment.py")
