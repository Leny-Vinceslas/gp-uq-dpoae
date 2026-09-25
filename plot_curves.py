"""
Per-participant GP fit figures.

For each participant, produces a 2×2 figure showing all 4 I/O curves
(Left/Right × 1414/4243 Hz) with:
  - Raw data points (detected vs below-noise-floor)
  - Noise floor shading
  - GP posterior mean + 95 % CI
  - Cubic fit overlay
  - Knee (◆) and max-compression (★) annotations with CI error bars

Output: outputs/curve_fits/{participant_id}.png

Usage
-----
    python plot_curves.py                     # all 38 participants
    python plot_curves.py --id 0089           # single participant
    python plot_curves.py --mean-type linear  # GP mean type (default: linear)
    python plot_curves.py --n-samples 200     # posterior samples (default: 300)
"""

import argparse
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.4,
})

from config import OUTPUT_DIR, DETECTION_DELTA_DB
from load_data import load_from_pickle
from gp_fitting import fit_gp, fit_cubic

OUTPUT_CURVE_DIR = OUTPUT_DIR / "curve_fits"
OUTPUT_CURVE_DIR.mkdir(exist_ok=True)

# Colours
C_GP    = "#2166ac"
C_CUBIC = "#d6604d"
C_NOISE = "#cccccc"
C_DET   = "k"
C_SUB   = "#888888"

# Task layout: (side, freq) → (row, col)
_LAYOUT = {
    ("L", 1414): (0, 0),
    ("L", 4243): (0, 1),
    ("R", 1414): (1, 0),
    ("R", 4243): (1, 1),
}
_TITLE  = {
    ("L", 1414): "Left  1.4 kHz",
    ("L", 4243): "Left  4.2 kHz",
    ("R", 1414): "Right 1.4 kHz",
    ("R", 4243): "Right 4.2 kHz",
}


def _safe_err(mean, lo, hi):
    """Return xerr [[lo_err], [hi_err]] clipped to non-negative."""
    return [[max(0., mean - lo)], [max(0., hi - mean)]]


def plot_participant(rec: dict, mean_type: str = "linear",
                     n_samples: int = 300, show: bool = False) -> Path:
    """
    Fit GP + cubic to all 4 tasks for one participant and save figure.

    Returns the path to the saved PNG.
    """
    pid = rec["ID"]
    grs = {(gr["side"], gr["f"]): gr for gr in rec.get("GR", [])}

    fig = plt.figure(figsize=(9, 5.5))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.32,
                            left=0.07, right=0.97, top=0.90, bottom=0.10)
    axes = {key: fig.add_subplot(gs[r, c])
            for key, (r, c) in _LAYOUT.items()}

    fig.suptitle(f"DPOAE I/O functions — Participant {pid}", fontsize=9,
                 fontweight="bold")

    for (side, freq), ax in axes.items():
        ax.set_title(_TITLE[(side, freq)], pad=3)
        ax.set_xlabel("$L_2$ (dB SPL)")
        ax.set_ylabel("DP (dB SPL)")
        ax.set_xlim(15, 68)

        gr = grs.get((side, freq)) or grs.get((side, int(freq)))
        if gr is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
            continue

        l2   = np.asarray(gr["f2"],    dtype=float)
        dp   = np.asarray(gr["dp"],    dtype=float)
        nf   = np.asarray(gr["noise2"],dtype=float)
        det  = dp > (nf + DETECTION_DELTA_DB)

        # --- Noise floor shading ---
        ax.fill_between(l2, nf, nf.min() - 6,
                        alpha=0.15, color=C_NOISE, zorder=0)
        ax.plot(l2, nf, "--", color=C_NOISE, lw=0.9, zorder=1)

        # --- GP fit ---
        try:
            gp = fit_gp(gr, mean_type=mean_type, n_samples=n_samples)
            ax.fill_between(gp.l2_grid, gp.lower95, gp.upper95,
                            alpha=0.22, color=C_GP, zorder=2,
                            label="GP 95 % CI")
            ax.plot(gp.l2_grid, gp.mean, "-", color=C_GP, lw=1.6,
                    zorder=3, label="GP mean")

            # Feature annotations
            for fname, marker, color in [
                ("knee",            "D", C_GP),
                ("max_compression", "*", C_CUBIC),
            ]:
                feat = gp.features[fname]
                if not np.isfinite(feat.mean):
                    continue
                y_pos = float(np.interp(feat.mean, gp.l2_grid, gp.mean))
                ax.plot(feat.mean, y_pos, marker=marker,
                        ms=7 if marker == "D" else 9,
                        color=color, zorder=6)
                ax.errorbar(feat.mean, y_pos,
                            xerr=_safe_err(feat.mean, feat.ci_low, feat.ci_high),
                            fmt="none", color=color, capsize=2.5, lw=1.0,
                            zorder=6)
        except Exception as e:
            ax.text(0.5, 0.55, f"GP failed:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="red")

        # --- Cubic fit ---
        try:
            cub = fit_cubic(gr)
            if not np.all(np.isnan(cub.mean)):
                ax.plot(cub.l2_grid, cub.mean, "--", color=C_CUBIC,
                        lw=1.0, alpha=0.7, zorder=4, label="Cubic")
        except Exception:
            pass

        # --- Data points ---
        ax.scatter(l2[det],  dp[det],  s=22, color=C_DET, zorder=5,
                   label="Detected" if (side, freq) == ("L", 1414) else "_")
        ax.scatter(l2[~det], dp[~det], s=22, color=C_SUB, marker="v",
                   facecolors="none", zorder=5,
                   label="Below NF" if (side, freq) == ("L", 1414) else "_")

    # Single legend on first axis
    axes[("L", 1414)].legend(loc="upper left", frameon=False, fontsize=6.5)

    # Annotation key
    fig.text(0.99, 0.01,
             "◆ knee  ★ max-compression  (error bars = 95 % CI)",
             ha="right", va="bottom", fontsize=6, color="grey")

    out_path = OUTPUT_CURVE_DIR / f"{pid}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def generate_all(mean_type: str = "linear", n_samples: int = 300,
                 participant_id: str = None, show: bool = False):
    """Generate figures for all participants (or one if participant_id given)."""
    OAEs = load_from_pickle()

    if participant_id:
        recs = [r for r in OAEs if r["ID"] == participant_id]
        if not recs:
            print(f"Participant {participant_id} not found.")
            return
    else:
        recs = OAEs

    print(f"Generating {len(recs)} figure(s) -> {OUTPUT_CURVE_DIR}")
    for i, rec in enumerate(recs):
        pid = rec["ID"]
        print(f"  [{i+1:3d}/{len(recs)}] {pid} ... ", end="", flush=True)
        try:
            path = plot_participant(rec, mean_type=mean_type,
                                    n_samples=n_samples, show=show)
            print(f"saved: {path.name}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",         type=str,   default=None,
                        help="Plot single participant by ID")
    parser.add_argument("--mean-type",  type=str,   default="linear",
                        choices=["cubic", "linear", "constant"],
                        help="GP mean function (default: linear)")
    parser.add_argument("--n-samples",  type=int,   default=300,
                        help="Posterior samples for feature CI (default: 300)")
    parser.add_argument("--show",       action="store_true",
                        help="Display figures interactively")
    args = parser.parse_args()

    generate_all(mean_type=args.mean_type, n_samples=args.n_samples,
                 participant_id=args.id, show=args.show)
