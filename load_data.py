"""
Load raw DPOAE CSV files into the standard OAEs list-of-dicts structure.

Two entry points:
    load_from_csv(data_dir)   — build OAEs from raw CSV files
    load_from_pickle(path)    — load the pre-processed pickle from GP-demo

OAEs structure (per participant):
    OAEs[i] = {
        'ID':  str,
        'GR': [{'ID', 'side', 'f', 'f1', 'f2', 'dp',
                'noise2', 'noise1', 'pdsnr', ...}, ...],   # I/O functions
        'DP': [{'ID', 'side', 'f', 'dp', 'noise2', ...}, ...]  # DP-grams
    }
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from config import L2_LEVELS, FREQS, DETECTION_DELTA_DB

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_GR_PATTERN = re.compile(r'^(\d{4})_GR_([LR])_(1414|4243)\.csv$')
_DP_PATTERN = re.compile(r'^(\d{4})_DP_([LR])\.csv$')


def _discover_files(data_dir: Path):
    """Return dicts mapping participant ID -> {GR: [...], DP: [...]} paths."""
    files = defaultdict(lambda: {"GR": [], "DP": []})

    for f in sorted(data_dir.iterdir()):
        if not f.suffix == ".csv":
            continue
        m = _GR_PATTERN.match(f.name)
        if m:
            files[m.group(1)]["GR"].append(f)
            continue
        m = _DP_PATTERN.match(f.name)
        if m:
            files[m.group(1)]["DP"].append(f)

    return files


# ---------------------------------------------------------------------------
# Single-file parsers
# ---------------------------------------------------------------------------

def _parse_gr(path: Path, participant_id: str) -> dict:
    """Parse one GR CSV into a growth-function dict."""
    df = pd.read_csv(path)
    m = _GR_PATTERN.match(path.name)
    side = m.group(2)
    freq = int(m.group(3))

    f2 = np.array(L2_LEVELS, dtype=float)
    dp = df["DP (dB)"].values.astype(float)
    noise2 = df["Noise+2sd (dB)"].values.astype(float)
    noise1 = df["Noise+1sd (dB)"].values.astype(float)

    # Sort ascending by L2 (raw files are descending)
    order = np.argsort(f2)
    f2     = f2[order]
    dp     = dp[order]
    noise2 = noise2[order]
    noise1 = noise1[order]

    # Align all other columns to the same sort order
    def _col(name):
        v = df[name].values.astype(float)
        return v[order] if len(v) == len(order) else v

    return {
        "ID":     participant_id,
        "side":   side,
        "f":      freq,
        "f1":     _col("F1 (dB)"),
        "f2":     f2,
        "dp":     dp,
        "noise2": noise2,
        "noise1": noise1,
        "f22_1":  _col("2F2-F1 (dB)"),
        "f31_22": _col("3F1-2F2 (dB)"),
        "f32_21": _col("3F2-2F1 (dB)"),
        "f41_32": _col("4F1-3F2 (dB)"),
        "pdsnr":  dp - noise1,
        # Filtered arrays (above noise floor)
        **_filter_above_noise(f2, dp, noise2),
    }


def _parse_dp(path: Path, participant_id: str) -> dict:
    """Parse one DP-gram CSV into a dp-gram dict."""
    df = pd.read_csv(path)
    m = _DP_PATTERN.match(path.name)
    side = m.group(2)

    return {
        "ID":     participant_id,
        "side":   side,
        "f":      df["Freq (Hz)"].values.astype(float),
        "f1":     df["F1 (dB)"].values.astype(float),
        "f2":     df["F2 (dB)"].values.astype(float),
        "dp":     df["DP (dB)"].values.astype(float),
        "noise2": df["Noise+2sd (dB)"].values.astype(float),
        "noise1": df["Noise+1sd (dB)"].values.astype(float),
        "pdsnr":  df["DP (dB)"].values - df["Noise+1sd (dB)"].values,
    }


def _filter_above_noise(f2, dp, noise2):
    """Compute filtered arrays (dp > noise2 + DETECTION_DELTA_DB)."""
    mask = dp > (noise2 + DETECTION_DELTA_DB)
    filt_f2 = f2[mask]
    filt_dp = dp[mask]
    # Replace empty with NaN arrays to preserve shape
    if filt_f2.size == 0:
        filt_f2 = np.full_like(f2, np.nan)
        filt_dp = np.full_like(dp, np.nan)
    return {"filt_f2": filt_f2, "filt_dp": filt_dp}


# ---------------------------------------------------------------------------
# Deduplication (keep highest mean-dp when duplicate (side, freq) exists)
# ---------------------------------------------------------------------------

def _dedup_gr(gr_list: list) -> list:
    groups = defaultdict(list)
    for i, gr in enumerate(gr_list):
        groups[(gr["side"], gr["f"])].append(i)

    keep = set()
    for indices in groups.values():
        if len(indices) == 1:
            keep.add(indices[0])
        else:
            best = max(indices,
                       key=lambda i: np.nanmean(gr_list[i]["dp"]))
            keep.add(best)

    return [gr for i, gr in enumerate(gr_list) if i in keep]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_from_csv(data_dir=None) -> list:
    """
    Load all participant CSV files from data_dir.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing {ID}_GR_{side}_{freq}.csv files.
        Defaults to config.DATA_DIR; falls back to DATA_DIR_SAMPLE if the
        primary path does not exist.

    Returns
    -------
    OAEs : list of dicts
    """
    from config import DATA_DIR, DATA_DIR_SAMPLE

    if data_dir is None:
        data_dir = DATA_DIR if DATA_DIR.exists() else DATA_DIR_SAMPLE
        if not DATA_DIR.exists():
            print(f"[load_data] DATA_DIR not found: {DATA_DIR}")
            print(f"[load_data] Falling back to sample data: {DATA_DIR_SAMPLE}")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_map = _discover_files(data_dir)
    if not file_map:
        raise FileNotFoundError(f"No GR/DP CSV files found in: {data_dir}")

    OAEs = []
    for pid in sorted(file_map.keys()):
        entry = file_map[pid]

        gr_list = [_parse_gr(p, pid) for p in entry["GR"]]
        dp_list = [_parse_dp(p, pid) for p in entry["DP"]]

        gr_list = _dedup_gr(gr_list)

        OAEs.append({"ID": pid, "GR": gr_list, "DP": dp_list})

    print(f"[load_data] Loaded {len(OAEs)} participants from {data_dir}")
    return OAEs


def load_from_pickle(path=None) -> list:
    """
    Load the pre-processed OAEs pickle produced by GP-demo.

    Parameters
    ----------
    path : str or Path, optional
        Path to .pkl file. Defaults to config.PICKLE_PATH.

    Returns
    -------
    OAEs : list of dicts
    """
    from config import PICKLE_PATH

    path = Path(path) if path else PICKLE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")

    with open(path, "rb") as fh:
        OAEs = pickle.load(fh)

    print(f"[load_data] Loaded {len(OAEs)} participants from pickle: {path.name}")
    return OAEs


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Try CSV load (full dataset if available, sample otherwise)
    try:
        OAEs = load_from_csv()
    except FileNotFoundError as e:
        print(e)
        OAEs = []

    # If CSV load returned nothing or failed, fall back to pickle
    if not OAEs:
        print("[load_data] Trying pickle fallback...")
        OAEs = load_from_pickle()

    print(f"\nParticipants loaded : {len(OAEs)}")
    if OAEs:
        p = OAEs[0]
        print(f"First participant   : {p['ID']}")
        print(f"  GR tasks          : {[(g['side'], g['f']) for g in p['GR']]}")
        print(f"  DP sides          : {[d['side'] for d in p['DP']]}")
        if p["GR"]:
            g = p["GR"][0]
            print(f"  Example GR L2     : {g['f2']}")
            print(f"  Example GR dp     : {np.round(g['dp'], 1)}")
            print(f"  Detected points   : {(~np.isnan(g['filt_dp'])).sum()} / {len(g['f2'])}")
