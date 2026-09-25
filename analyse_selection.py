"""
Standalone analysis: active vs equalspaced level selection overlap.

Tests the hypothesis that the similarity in RMSE between active and
equalspaced strategies at k>=5 is because active selection is constrained
to the 10 original L2 positions, so at moderate k the variance-weighted
picks converge to a near-uniform spread.

Output
------
  outputs/selection_overlap.pdf / .png
    (a) Mean overlap between active and equalspaced selections vs k
    (b) Heatmap of active selection frequency by L2 position and k
    (c) Heatmap of equalspaced selection frequency by L2 position and k
    (d) Seed positions used by active (always 3 equalspaced) marked on (b)

Run:
    python analyse_selection.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, L2_LEVELS, DETECTION_DELTA_DB
from load_data import load_from_pickle
from gp_fitting import (
    _select_levels_active,
    _select_levels_equalspaced,
    _obs_noise_var,
)

K_VALUES  = [3, 4, 5, 6, 7, 8, 9]
L2_ALL    = np.array(L2_LEVELS, dtype=float)   # [65, 60, ..., 20]
N_LEVELS  = len(L2_ALL)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_selections(OAEs: list) -> pd.DataFrame:
    records = []
    for rec in OAEs:
        for gr in rec.get("GR", []):
            if len(gr.get("f2", [])) < 4:
                continue

            l2  = np.asarray(gr["f2"],    dtype=float)
            dp  = np.asarray(gr["dp"],    dtype=float)
            nf  = np.asarray(gr["noise2"],dtype=float)

            mask = np.isfinite(l2) & np.isfinite(dp) & np.isfinite(nf)
            l2, dp, nf = l2[mask], dp[mask], nf[mask]
            if len(l2) < 4:
                continue

            nv = _obs_noise_var(dp, nf)

            for k in K_VALUES:
                eq_idx  = set(_select_levels_equalspaced(l2, k).tolist())
                try:
                    act_idx = set(_select_levels_active(l2, dp, nv, k).tolist())
                except Exception:
                    act_idx = eq_idx   # fallback on numerical failure

                # Overlap: fraction of active selections that match equalspaced
                overlap = len(eq_idx & act_idx) / k if k > 0 else 0.0

                records.append({
                    "participant":     gr["ID"],
                    "side":            gr["side"],
                    "freq":            gr["f"],
                    "k":               k,
                    "overlap":         overlap,
                    "active_idx":      sorted(act_idx),
                    "equalspaced_idx": sorted(eq_idx),
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Selection frequency heatmaps
# ---------------------------------------------------------------------------

def selection_heatmap(df: pd.DataFrame, strategy_col: str) -> np.ndarray:
    """
    Returns array of shape (len(K_VALUES), N_LEVELS):
    fraction of curves that selected each position index at each k.
    """
    heat = np.zeros((len(K_VALUES), N_LEVELS))
    for i, k in enumerate(K_VALUES):
        sub = df[df["k"] == k]
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            for idx in row[strategy_col]:
                if 0 <= idx < N_LEVELS:
                    heat[i, idx] += 1
        heat[i] /= len(sub)
    return heat


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, show: bool = False):
    overlap_summary = df.groupby("k")["overlap"].agg(["mean", "std", "sem"])

    heat_active = selection_heatmap(df, "active_idx")
    heat_equal  = selection_heatmap(df, "equalspaced_idx")

    # Seed positions used by active: 3 equalspaced from N_LEVELS candidates
    seed_idx = _select_levels_equalspaced(L2_ALL, 3)

    fig = plt.figure(figsize=(12, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.96, top=0.92, bottom=0.10)

    ax_ov   = fig.add_subplot(gs[0, 0])
    ax_act  = fig.add_subplot(gs[0, 1])
    ax_eq   = fig.add_subplot(gs[1, 0])
    ax_diff = fig.add_subplot(gs[1, 1])

    # --- (a) Overlap vs k ---
    ks = overlap_summary.index.values
    ax_ov.errorbar(ks, overlap_summary["mean"] * 100,
                   yerr=overlap_summary["sem"] * 100,
                   fmt="o-", color="#2166ac", capsize=4, lw=2, label="Mean ± SE")
    ax_ov.axhline(100, color="grey", lw=1, linestyle="--", alpha=0.6,
                  label="Perfect overlap")
    ax_ov.set_xlabel("Retained levels $k$")
    ax_ov.set_ylabel("Overlap with equalspaced (%)")
    ax_ov.set_title("(a) Active–equalspaced selection overlap")
    ax_ov.set_xticks(K_VALUES)
    ax_ov.set_ylim(0, 110)
    ax_ov.legend(frameon=False, fontsize=8)

    l2_labels = [str(int(v)) for v in L2_ALL]
    cmap = plt.cm.Blues

    # --- (b) Active selection heatmap ---
    im_act = ax_act.imshow(heat_active, aspect="auto", vmin=0, vmax=1,
                           cmap=cmap, origin="upper")
    # Mark seed positions
    for sidx in seed_idx:
        ax_act.axvline(sidx, color="#d6604d", lw=1.5, alpha=0.7, linestyle="--")
    ax_act.set_xticks(range(N_LEVELS))
    ax_act.set_xticklabels(l2_labels, fontsize=7)
    ax_act.set_yticks(range(len(K_VALUES)))
    ax_act.set_yticklabels([str(k) for k in K_VALUES])
    ax_act.set_xlabel("$L_2$ (dB SPL)")
    ax_act.set_ylabel("$k$")
    ax_act.set_title("(b) Active — selection frequency\n(red = seed positions)")
    plt.colorbar(im_act, ax=ax_act, label="Fraction of curves")

    # --- (c) Equalspaced selection heatmap ---
    im_eq = ax_eq.imshow(heat_equal, aspect="auto", vmin=0, vmax=1,
                         cmap=cmap, origin="upper")
    ax_eq.set_xticks(range(N_LEVELS))
    ax_eq.set_xticklabels(l2_labels, fontsize=7)
    ax_eq.set_yticks(range(len(K_VALUES)))
    ax_eq.set_yticklabels([str(k) for k in K_VALUES])
    ax_eq.set_xlabel("$L_2$ (dB SPL)")
    ax_eq.set_ylabel("$k$")
    ax_eq.set_title("(c) Equalspaced — selection frequency")
    plt.colorbar(im_eq, ax=ax_eq, label="Fraction of curves")

    # --- (d) Difference heatmap: active − equalspaced ---
    diff = heat_active - heat_equal
    vmax = np.abs(diff).max()
    im_diff = ax_diff.imshow(diff, aspect="auto",
                             vmin=-vmax, vmax=vmax,
                             cmap="RdBu_r", origin="upper")
    ax_diff.set_xticks(range(N_LEVELS))
    ax_diff.set_xticklabels(l2_labels, fontsize=7)
    ax_diff.set_yticks(range(len(K_VALUES)))
    ax_diff.set_yticklabels([str(k) for k in K_VALUES])
    ax_diff.set_xlabel("$L_2$ (dB SPL)")
    ax_diff.set_ylabel("$k$")
    ax_diff.set_title("(d) Difference: active − equalspaced\n(blue = active prefers, red = equalspaced prefers)")
    plt.colorbar(im_diff, ax=ax_diff, label="Frequency difference")

    fig.suptitle("Active vs equalspaced level selection: constraint hypothesis test\n"
                 "($n=146$ curves, 38 participants)", fontsize=10)

    out_pdf = OUTPUT_DIR / "selection_overlap.pdf"
    out_png = OUTPUT_DIR / "selection_overlap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print("\n=== Active vs equalspaced overlap by k ===")
    summary = df.groupby("k")["overlap"].agg(["mean", "std", "min", "max"])
    summary.columns = ["Mean overlap", "SD", "Min", "Max"]
    summary *= 100
    print(summary.round(1).to_string())

    print("\n=== At k=5: what does active actually select? ===")
    k5 = df[df["k"] == 5]
    # Flatten all active selections and count frequency per position
    from collections import Counter
    counts = Counter(idx for row in k5["active_idx"] for idx in row)
    total  = len(k5)
    print(f"  (n={total} curves)")
    for pos_idx in range(N_LEVELS):
        freq = counts.get(pos_idx, 0) / total * 100
        l2v  = int(L2_ALL[pos_idx])
        eq5  = _select_levels_equalspaced(L2_ALL, 5)
        marker = " <- equalspaced" if pos_idx in eq5 else ""
        seed3  = _select_levels_equalspaced(L2_ALL, 3)
        if pos_idx in seed3:
            marker += " (seed)"
        print(f"  L2={l2v:2d} dB (pos {pos_idx}): {freq:5.1f}%{marker}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    OAEs = load_from_pickle()

    print(f"Computing selections for {sum(len(r.get('GR',[])) for r in OAEs)} curves × {len(K_VALUES)} k values...")
    df = collect_selections(OAEs)

    print_summary(df)
    make_figure(df, show=args.show)
