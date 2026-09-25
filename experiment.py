"""
Leave-k-out experiment: sparse acquisition vs full GP reference.

For every participant × task (up to 152 curves):
  1. Fit reference GP (k=10) and reference cubic.
  2. For k in K_VALUES, strategy in STRATEGIES:
       - Fit sparse GP and sparse cubic to k selected levels.
       - Compute curve RMSE vs reference GP mean (primary) and vs held-out raw DP (secondary).
       - Extract features + CIs from sparse GP; point estimates from sparse cubic.
       - Compute feature bias and MDC.
  3. Save one row per (participant, task, k, strategy, repeat) to CSV.

Results file: outputs/experiment_results.csv
Reference file: outputs/reference_results.csv  (k=10 fits, one row per curve)

Run:
    python experiment.py                        # full run (features + RMSE), parallel
    python experiment.py --fast                 # RMSE only (faster)
    python experiment.py --workers 4            # override worker count
    python experiment.py --smoke                # single-curve test
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import OUTPUT_DIR, TASK_KEY
from load_data import load_from_pickle
from gp_fitting import (
    fit_gp, fit_sparse,
    fit_cubic, fit_cubic_sparse,
    GPResult, CubicResult, MEAN_TYPES,
)

# Default: leave 2 cores free for OS + main process
N_WORKERS = max(1, min(os.cpu_count() - 2, 6))

# GP mean types to run — all three for comparison
GP_MEAN_TYPES = list(MEAN_TYPES)   # ['cubic', 'linear', 'constant']

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

K_VALUES   = [3, 4, 5, 6, 7, 8, 9]
STRATEGIES = ["random", "equalspaced", "active"]
N_REPEATS  = 5     # repeats for 'random' strategy (5 is sufficient for SE at these CI widths)
N_SAMPLES_REF    = 300   # posterior samples for k=10 reference fits
N_SAMPLES_SPARSE = 100   # posterior samples for sparse fits; 100 is sufficient (CI width SE <0.6 dB vs reported widths of 12–26 dB)
N_STEPS_SPARSE   = 80    # training steps for sparse GP fits (prior mean guides optimiser)

# Suppress gpytorch numerical warnings in the loop
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 1:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def _interp_to_l2(source_grid, source_mean, target_l2):
    """Interpolate source_mean (on source_grid) to target_l2 positions."""
    return np.interp(target_l2, source_grid, source_mean)


def _feature_row(prefix: str, feat_dict, is_gp: bool) -> dict:
    """Flatten features into a dict of column-name -> value."""
    row = {}
    for fname in ("knee", "max_compression", "slope_low", "slope_high"):
        if is_gp:
            f = feat_dict[fname]
            row[f"{prefix}_{fname}_mean"]      = f.mean
            row[f"{prefix}_{fname}_std"]       = f.std
            row[f"{prefix}_{fname}_ci_low"]    = f.ci_low
            row[f"{prefix}_{fname}_ci_high"]   = f.ci_high
            row[f"{prefix}_{fname}_ci_width"]  = f.ci_width
            # estimated MDC lower bound: standard formula applied to single-session posterior SD
            row[f"{prefix}_{fname}_mdc"]       = 1.96 * np.sqrt(2) * f.std
        else:
            row[f"{prefix}_{fname}"] = feat_dict[fname]
    return row


# ---------------------------------------------------------------------------
# Top-level picklable worker (required for Windows multiprocessing / spawn)
# ---------------------------------------------------------------------------

def _worker_fn(args: tuple) -> tuple:
    """
    Worker entry point — must be a top-level function so it can be pickled
    on Windows (spawn context). Sets torch to single-threaded to avoid
    oversubscription when several workers run in parallel.
    """
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    gr, fast = args
    return _process_curve(gr, fast=fast)


# ---------------------------------------------------------------------------
# Per-curve processing
# ---------------------------------------------------------------------------

def _process_curve(gr: dict, fast: bool = False) -> tuple[dict, list[dict]]:
    """
    Fit reference models and run leave-k-out for one I/O curve.

    Returns
    -------
    ref_row   : dict  — one row for reference_results.csv
    sparse_rows : list[dict] — rows for experiment_results.csv
    """
    pid  = gr["ID"]
    side = gr["side"]
    freq = gr["f"]

    base = {"participant": pid, "side": side, "freq": freq}

    # ------------------------------------------------------------------
    # 1. Reference fits (k=10) — one per GP mean type + cubic
    # ------------------------------------------------------------------
    ref_gps = {}
    for mt in GP_MEAN_TYPES:
        ref_gps[mt] = fit_gp(gr, mean_type=mt,
                             n_samples=0 if fast else N_SAMPLES_REF)

    # Use 'linear' as the primary reference (honest uncertainty, correct prior)
    ref_gp  = ref_gps["linear"]
    ref_cub = fit_cubic(gr)

    # Cache per-mean-type hyperparameters for fast sparse fits
    ref_hyperparams = {
        mt: getattr(ref_gps[mt], "_hyperparams", None)
        for mt in GP_MEAN_TYPES
    }

    ref_row = {**base, "k": ref_gp.k_used, "n_detected": ref_gp.n_detected}
    if not fast:
        for mt in GP_MEAN_TYPES:
            ref_row.update(_feature_row(f"gp_{mt}", ref_gps[mt].features, is_gp=True))
        ref_row.update(_feature_row("cubic", ref_cub.features, is_gp=False))

    # ------------------------------------------------------------------
    # 2. Leave-k-out loop
    # ------------------------------------------------------------------
    sparse_rows = []
    all_l2 = np.asarray(gr["f2"], dtype=float)

    for k, strategy in product(K_VALUES, STRATEGIES):
        n_reps = N_REPEATS if strategy == "random" else 1

        for rep in range(n_reps):
            seed = (rep * 1000 + k * 37 + hash(pid + side) % 997) % (2**31)
            seed = seed if strategy == "random" else None

            row = {**base, "k": k, "strategy": strategy, "repeat": rep}

            # Initialise held-out arrays — populated by first successful GP fit
            held_l2     = np.array([], dtype=float)
            held_dp     = np.array([], dtype=float)
            ref_at_held = np.array([], dtype=float)

            # --- Sparse GP — all three mean types ---------------------
            for mt in GP_MEAN_TYPES:
                pfx = f"gp_{mt}"
                try:
                    sp_gp = fit_sparse(gr, k=k, strategy=strategy,
                                       seed=seed, mean_type=mt,
                                       n_samples=0 if fast else N_SAMPLES_SPARSE,
                                       n_steps=N_STEPS_SPARSE,
                                       ref_hyperparams=ref_hyperparams.get(mt))

                    # Identify held-out levels (same for all mean types — same seed)
                    if held_l2.size == 0:
                        used_l2     = sp_gp.l2_used
                        held_mask   = ~np.isin(all_l2, used_l2)
                        held_l2     = all_l2[held_mask]
                        held_dp     = np.asarray(gr["dp"], dtype=float)[held_mask]
                        if held_l2.size > 0:
                            ref_at_held = _interp_to_l2(
                                ref_gp.l2_grid, ref_gp.mean, held_l2)

                    if held_l2.size > 0:
                        sp_at_held = _interp_to_l2(sp_gp.l2_grid, sp_gp.mean, held_l2)
                        row[f"rmse_{pfx}_vs_ref_held"] = _rmse(sp_at_held, ref_at_held)
                        row[f"rmse_{pfx}_vs_raw_held"] = _rmse(sp_at_held, held_dp)
                    else:
                        row[f"rmse_{pfx}_vs_ref_held"] = np.nan
                        row[f"rmse_{pfx}_vs_raw_held"] = np.nan

                    row[f"n_detected_{pfx}"] = sp_gp.n_detected

                    if not fast:
                        row.update(_feature_row(f"{pfx}_sparse",
                                                sp_gp.features, is_gp=True))
                        for fname in ("knee", "max_compression",
                                      "slope_low", "slope_high"):
                            ref_val = ref_gps[mt].features[fname].mean
                            sp_val  = sp_gp.features[fname].mean
                            row[f"{pfx}_bias_{fname}"]       = sp_val - ref_val
                            row[f"{pfx}_covers_ref_{fname}"] = int(
                                sp_gp.features[fname].ci_low <= ref_val
                                <= sp_gp.features[fname].ci_high
                            )

                except Exception as e:
                    row[f"rmse_{pfx}_vs_ref_held"] = np.nan
                    row[f"rmse_{pfx}_vs_raw_held"] = np.nan
                    row[f"{pfx}_error"] = str(e)[:80]

            # --- Sparse Cubic -----------------------------------------
            try:
                sp_cub = fit_cubic_sparse(gr, k=k, strategy=strategy, seed=seed)

                if held_l2.size > 0:
                    cub_at_held = _interp_to_l2(sp_cub.l2_grid, sp_cub.mean, held_l2)
                    row["rmse_cubic_vs_ref_held"] = _rmse(cub_at_held, ref_at_held)
                    row["rmse_cubic_vs_raw_held"] = _rmse(cub_at_held, held_dp)
                else:
                    row["rmse_cubic_vs_ref_held"] = np.nan
                    row["rmse_cubic_vs_raw_held"] = np.nan

                if not fast:
                    row.update(_feature_row("cubic_sparse", sp_cub.features, is_gp=False))
                    for fname in ("knee", "max_compression", "slope_low", "slope_high"):
                        ref_val = ref_gp.features[fname].mean
                        cub_val = sp_cub.features[fname]
                        row[f"cubic_bias_{fname}"] = (
                            cub_val - ref_val if np.isfinite(cub_val) else np.nan
                        )

            except Exception as e:
                row["rmse_cubic_vs_ref_held"] = np.nan
                row["rmse_cubic_vs_raw_held"] = np.nan
                row["cubic_error"] = str(e)[:80]

            sparse_rows.append(row)

    return ref_row, sparse_rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(fast: bool = False, resume: bool = True,
                   n_workers: int = N_WORKERS):
    """
    Run the full experiment across all participants and tasks in parallel.

    Parameters
    ----------
    fast      : bool  Skip feature extraction (RMSE only). Much faster.
    resume    : bool  Skip curves whose results are already in the CSV.
    n_workers : int   Number of parallel worker processes.
    """
    results_path   = OUTPUT_DIR / "experiment_results.csv"
    reference_path = OUTPUT_DIR / "reference_results.csv"

    OAEs = load_from_pickle()

    # Build list of all curves to process
    all_curves = [
        gr
        for rec in OAEs
        for gr in rec.get("GR", [])
        if len(gr.get("f2", [])) >= 3
    ]

    # Load already-done keys if resuming
    done_keys = set()
    if resume and results_path.exists():
        existing  = pd.read_csv(results_path)
        done_keys = set(zip(existing["participant"],
                            existing["side"],
                            existing["freq"]))

    todo = [
        gr for gr in all_curves
        if (gr["ID"], gr["side"], gr["f"]) not in done_keys
    ]

    n_fits = len(todo) * len(K_VALUES) * (N_REPEATS + 2)
    print(f"Curves to process : {len(todo)} / {len(all_curves)}")
    print(f"Workers           : {n_workers}")
    print(f"Est. GP fits      : ~{n_fits:,}")
    print(f"Fast mode         : {fast}\n")

    if not todo:
        print("Nothing to do — all curves already processed.")
        return pd.read_csv(results_path), pd.read_csv(reference_path)

    work = [(gr, fast) for gr in todo]

    ref_batch    = []
    sparse_batch = []
    n_done       = 0
    first_write  = not (resume and results_path.exists())

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_fn, args): args[0]
                   for args in work}

        pbar = tqdm(as_completed(futures), total=len(futures),
                    unit="curve", ncols=80)

        for future in pbar:
            gr = futures[future]
            pid, side, freq = gr["ID"], gr["side"], gr["f"]

            try:
                ref_row, s_rows = future.result()
                ref_batch.append(ref_row)
                sparse_batch.extend(s_rows)
                n_done += 1
                pbar.set_postfix({"last": f"{pid} {side} {freq}"})
            except Exception as e:
                tqdm.write(f"  FAILED {pid} {side} {freq}: {e}")
                continue

            # Save incrementally every 8 curves to limit memory use
            if n_done % 8 == 0:
                _save(sparse_batch, results_path, append=not first_write)
                _save(ref_batch,    reference_path, append=not first_write)
                sparse_batch = []
                ref_batch    = []
                first_write  = False

    # Final flush
    if sparse_batch:
        _save(sparse_batch, results_path, append=not first_write)
    if ref_batch:
        _save(ref_batch, reference_path, append=not first_write)

    print(f"\nDone. Results saved to:\n"
          f"  {results_path}\n  {reference_path}")
    return pd.read_csv(results_path), pd.read_csv(reference_path)


def _save(rows: list, path: Path, append: bool):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Quick sanity check (single curve)
# ---------------------------------------------------------------------------

def _smoke_test():
    from load_data import load_from_pickle
    OAEs = load_from_pickle()
    gr   = OAEs[0]["GR"][0]

    print("=== Smoke test: single curve ===")
    ref_row, s_rows = _process_curve(gr, fast=False)

    print(f"\nReference row keys  : {list(ref_row.keys())}")
    print(f"Sparse rows         : {len(s_rows)}")
    print(f"Columns per row     : {len(s_rows[0])}")

    # Show RMSE summary for this curve
    df = pd.DataFrame(s_rows)
    rmse_cols = [c for c in df.columns if "rmse" in c and "vs_ref_held" in c]
    if rmse_cols:
        summary = df.groupby(["k", "strategy"])[rmse_cols].mean().round(3)
        print("\nMean RMSE vs reference GP (held-out levels):")
        print(summary.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",         action="store_true",
                        help="Skip feature extraction (RMSE only)")
    parser.add_argument("--smoke",        action="store_true",
                        help="Run single-curve smoke test only")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Restart from scratch (ignore existing CSV)")
    parser.add_argument("--workers",      type=int, default=N_WORKERS,
                        help=f"Number of parallel workers (default: {N_WORKERS})")
    parser.add_argument("--plot-curves",  action="store_true",
                        help="Generate per-participant GP fit figures after experiment")
    parser.add_argument("--plot-only",    action="store_true",
                        help="Skip experiment, only generate curve figures")
    parser.add_argument("--plot-id",      type=str, default=None,
                        help="With --plot-curves/--plot-only: single participant ID")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
    elif args.plot_only:
        from plot_curves import generate_all
        generate_all(participant_id=args.plot_id)
    else:
        run_experiment(fast=args.fast, resume=not args.no_resume,
                       n_workers=args.workers)
        if args.plot_curves:
            from plot_curves import generate_all
            print("\nGenerating per-participant curve figures...")
            generate_all(participant_id=args.plot_id)
