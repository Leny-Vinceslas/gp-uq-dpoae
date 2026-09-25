"""
Project configuration.

CSV folder must contain files named:
    {ID}_GR_{side}_{freq}.csv   — I/O growth functions (GR)
    {ID}_DP_{side}.csv          — DP-grams (DP)

where side in {L, R} and freq in {1414, 4243}.

Data access
-----------
No participant data ships with this repository (see data/README.md for
why, and for the exact CSV layout expected). To run the pipeline against
your own data, either:
  - drop CSVs into data/raw/ (already git-ignored), or
  - set the DOPAE_DATA_DIR environment variable to an external folder.
Both DATA_DIR_SAMPLE and PICKLE_PATH below are optional fast paths for a
pre-processed subset and can be left unset/empty.
"""

import os
from pathlib import Path

# --- Data paths -----------------------------------------------------------

# Root of this file
_HERE = Path(__file__).parent

# Full dataset: folder containing the raw participant CSVs.
# Override without editing this file: set the DOPAE_DATA_DIR env var.
DATA_DIR = Path(os.environ.get("DOPAE_DATA_DIR", _HERE / "data" / "raw"))

# Small fallback sample (a couple of participants) for smoke-testing the
# pipeline without full data access. Override with DOPAE_DATA_DIR_SAMPLE.
DATA_DIR_SAMPLE = Path(os.environ.get("DOPAE_DATA_DIR_SAMPLE", _HERE / "data" / "sample"))

# Optional pre-processed pickle (skips CSV parsing if you already have one).
# Override with DOPAE_PICKLE_PATH.
PICKLE_PATH = Path(os.environ.get("DOPAE_PICKLE_PATH", _HERE / "data" / "processed.pkl"))

# --- Output paths ---------------------------------------------------------

OUTPUT_DIR = _HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Stimulus protocol ----------------------------------------------------

# L2 levels forced (software returns unreliable values — see GP-demo note)
L2_LEVELS = [65, 60, 55, 50, 45, 40, 35, 30, 25, 20]  # dB SPL, descending

# Detection threshold above noise floor
DETECTION_DELTA_DB = 3.0

# Valid test frequencies (Hz)
FREQS = (1414, 4243)
SIDES = ("L", "R")

# Task mapping: (side, freq) -> int
TASK_ID  = {("L", 1414): 0, ("L", 4243): 1, ("R", 1414): 2, ("R", 4243): 3}
TASK_KEY = {0: "L_1414",    1: "L_4243",    2: "R_1414",    3: "R_4243"}
