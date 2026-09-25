"""
Generate a flowchart diagram of the GP DPOAE experiment algorithm.

Output: outputs/algorithm_flowchart.pdf / .png

Run:
    python plot_algorithm.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_DATA     = "#f7f7f7"   # light grey   — data / IO
C_REF      = "#d1e5f0"   # light blue   — reference fitting
C_LOOP     = "#fddbc7"   # light orange — experiment loop
C_STRAT    = "#e0f3db"   # light green  — strategies
C_GP       = "#c2a5cf"   # light purple — GP model
C_FEAT     = "#ffffbf"   # light yellow — feature extraction
C_METRIC   = "#f4a582"   # salmon       — metrics
C_EDGE     = "#555555"
C_ARROW    = "#333333"

FONT      = "serif"
FS_TITLE  = 8.5
FS_BODY   = 7.5
FS_SMALL  = 6.5

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")


# ---------------------------------------------------------------------------
# Helper: draw a rounded box with title + body lines
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, title, lines, fc, ec=C_EDGE, title_bold=True,
        fontsize_body=FS_BODY):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          fc=fc, ec=ec, lw=1.2, zorder=2)
    ax.add_patch(rect)
    # Title
    weight = "bold" if title_bold else "normal"
    ax.text(x + w/2, y + h - 0.13, title, ha="center", va="top",
            fontsize=FS_TITLE, fontweight=weight, fontfamily=FONT, zorder=3)
    # Dividing line under title
    ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.26, y + h - 0.26],
            color=ec, lw=0.7, zorder=3)
    # Body text
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.44 - i * 0.175, line,
                ha="left", va="top", fontsize=fontsize_body,
                fontfamily=FONT, zorder=3)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.3, mutation_scale=12),
                zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, ha="left", va="center",
                fontsize=FS_SMALL, fontfamily=FONT, color="#444444", zorder=5)


def bracket_arrow(ax, x_from, y_from, x_to, y_to, via_x, label=""):
    """L-shaped arrow via an intermediate x position."""
    ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                arrowprops=dict(
                    arrowstyle="-|>", color=C_ARROW, lw=1.2,
                    mutation_scale=11,
                    connectionstyle=f"arc3,rad=0.0"
                ), zorder=4)


# ============================================================
# ROW 0: Title
# ============================================================
ax.text(7, 9.75, "GP DPOAE Experiment — Algorithm Overview",
        ha="center", va="top", fontsize=11, fontweight="bold",
        fontfamily=FONT)

# ============================================================
# ROW 1: Data
# ============================================================
box(ax, 4.5, 8.7, 5.0, 0.85, "Dataset",
    ["38 participants  ×  4 tasks (L/R × 1414/4243 Hz)  =  146 I/O curves",
     "10 L₂ levels: 65, 60, …, 20 dB SPL   |   Sub-noise retained (censored)"],
    C_DATA)

# ============================================================
# ROW 2: Reference fitting
# ============================================================
box(ax, 1.8, 7.15, 10.4, 1.35,
    "Reference Fit  (k = 10, all levels)",
    ["5 GP mean types × each curve:  cubic  |  two-slope  |  quadratic  |  linear  |  constant",
     "Kernel: Matérn-3/2   Noise: heteroscedastic (SNR-weighted + soft Tobit for censored)",
     "Adam optimisation — 150 steps   |   300 posterior samples",
     "→ Features + 95% CI + est. MDC   |   Cache hyperparameters (l, sigma_f2) for sparse reuse"],
    C_REF)

box(ax, 0.15, 7.15, 1.5, 1.35,
    "Cubic\nBaseline",
    ["OLS on\ndetected pts\n(NaN if <4)"],
    C_DATA, fontsize_body=FS_SMALL)

arrow(ax, 7.0, 8.7, 7.0, 8.5)   # data → reference

# ============================================================
# ROW 3: Leave-k-out loop header
# ============================================================
box(ax, 2.5, 6.45, 9.0, 0.55,
    "Leave-k-out Loop:  k ∈ {3, 4, 5, 6, 7, 8, 9}",
    [], C_LOOP)

arrow(ax, 7.0, 7.15, 7.0, 7.0)  # reference → loop

# ============================================================
# ROW 4: Three strategies side by side
# ============================================================
strat_y  = 5.0
strat_h  = 1.25
strat_xs = [1.0, 5.0, 9.0]

# Random
box(ax, strat_xs[0], strat_y, 3.5, strat_h,
    "Random  (×10 repeats)",
    ["Sample k levels uniformly",
     "without replacement",
     "→ average over 10 seeds"],
    C_STRAT)

# Equalspaced
box(ax, strat_xs[1], strat_y, 3.5, strat_h,
    "Equalspaced",
    ["linspace(0, 9, k)",
     "rounded to nearest",
     "integer index"],
    C_STRAT)

# Active
box(ax, strat_xs[2], strat_y, 3.8, strat_h,
    "Active  (sequential greedy)",
    ["1. Seed: 3 equalspaced pts, fit GP",
     "2. For each of k−3 steps:",
     "   pick max-variance candidate,",
     "   refit GP (cached hyperparams)"],
    C_STRAT)

# Arrows from loop header to strategies
for sx, sw in [(strat_xs[0], 3.5), (strat_xs[1], 3.5), (strat_xs[2], 3.8)]:
    arrow(ax, sx + sw/2, 6.45, sx + sw/2, 6.25)

# Dotted merge line below strategies
merge_y = 4.82
for sx, sw in [(strat_xs[0], 3.5), (strat_xs[1], 3.5), (strat_xs[2], 3.8)]:
    ax.plot([sx + sw/2, sx + sw/2], [strat_y, merge_y],
            color=C_ARROW, lw=1.2, linestyle="--", zorder=3)
ax.plot([strat_xs[0]+1.75, strat_xs[2]+1.9], [merge_y, merge_y],
        color=C_ARROW, lw=1.2, zorder=3)
arrow(ax, 7.0, merge_y, 7.0, 4.65)

# ============================================================
# ROW 5: Sparse GP fit
# ============================================================
box(ax, 2.5, 3.55, 9.0, 1.0,
    "Sparse GP Fit  (k selected levels)",
    ["5 mean types fitted in parallel   |   Hyperparameters inherited from reference (no Adam)",
     "200 posterior samples drawn on dense L₂ grid",
     "Monotone filtering + physiological validity bounds applied to samples"],
    C_GP)

arrow(ax, 7.0, 3.55, 7.0, 3.37)

# ============================================================
# ROW 6: Three output branches
# ============================================================
out_y = 1.85
out_h = 1.35

# RMSE
box(ax, 0.8, out_y, 3.5, out_h,
    "Curve Reconstruction",
    ["RMSE vs reference GP",
     "(primary metric)",
     "RMSE vs raw held-out",
     "(secondary metric)"],
    C_METRIC)

# Feature extraction
box(ax, 4.9, out_y, 4.0, out_h,
    "Feature Extraction (per sample)",
    ["Knee: PWLF breakpoint [25,63] dB",
     "Max-comp: argmax in L₂ ≥ 40 dB",
     "Slopes: OLS in [25,45] / [45,65] dB",
     "→ Mean, SD, 95% CI, est. MDC = 1.96√2·σ̂ (lower bound)"],
    C_FEAT)

# Coverage
box(ax, 9.4, out_y, 3.8, out_h,
    "Feature Coverage",
    ["Does sparse GP 95% CI",
     "contain the reference",
     "feature mean?",
     "→ Coverage rate (%)"],
    C_METRIC)

for ox, ow in [(0.8, 3.5), (4.9, 4.0), (9.4, 3.8)]:
    bracket_arrow(ax, 7.0, 3.55, ox + ow/2, out_y + out_h,
                  via_x=ox + ow/2)
    ax.plot([ox + ow/2, ox + ow/2], [3.55, out_y + out_h],
            color=C_ARROW, lw=1.2, linestyle="--", zorder=3)
ax.plot([0.8+1.75, 9.4+1.9], [3.55, 3.55],
        color=C_ARROW, lw=1.2, zorder=3)

# ============================================================
# ROW 7: Output CSV label
# ============================================================
ax.text(7.0, 1.6, "experiment_results.csv  (one row per curve × k × strategy × repeat)",
        ha="center", va="top", fontsize=FS_SMALL, fontfamily=FONT,
        style="italic", color="#555555",
        bbox=dict(boxstyle="round,pad=0.2", fc="#eeeeee", ec="#aaaaaa", lw=0.8))

# ============================================================
# Side panel: GP model summary
# ============================================================
box(ax, 0.05, 3.55, 2.25, 3.45,
    "GP Model",
    ["Kernel:",
     " Matérn-3/2",
     "",
     "5 Mean types:",
     " cubic (tight CI)",
     " two-slope (physiol.)",
     " quadratic",
     " linear (wide CI)",
     " constant (widest)",
     "",
     "Noise:",
     " detected: f(SNR)",
     " censored: 36 dB2"],
    C_GP, fontsize_body=6.2)

ax.annotate("", xy=(2.5, 4.05), xytext=(2.3, 4.05),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.0,
                            mutation_scale=10), zorder=4)

# ============================================================
# Legend
# ============================================================
legend_items = [
    mpatches.Patch(fc=C_DATA,  ec=C_EDGE, label="Data / I/O"),
    mpatches.Patch(fc=C_REF,   ec=C_EDGE, label="Reference fit"),
    mpatches.Patch(fc=C_LOOP,  ec=C_EDGE, label="Experiment loop"),
    mpatches.Patch(fc=C_STRAT, ec=C_EDGE, label="Level selection"),
    mpatches.Patch(fc=C_GP,    ec=C_EDGE, label="GP model"),
    mpatches.Patch(fc=C_FEAT,  ec=C_EDGE, label="Feature extraction"),
    mpatches.Patch(fc=C_METRIC,ec=C_EDGE, label="Metrics"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=FS_SMALL,
          framealpha=0.9, edgecolor=C_EDGE, ncol=7,
          bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
for ext in ("pdf", "png"):
    path = OUTPUT_DIR / f"algorithm_flowchart.{ext}"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Saved: {path}")
plt.show()
