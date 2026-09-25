# Data

No participant data is included in this repository. The underlying dataset
(38 normal-hearing adults, DPOAE I/O functions) comes from a human-subjects
study and is not publicly shareable; only code and aggregate/cohort-level
results are published here.

To run the pipeline, supply your own data in one of two ways:

## 1. Raw CSVs (`load_from_csv`)

Drop files into `data/raw/` (already git-ignored), named:

```
{ID}_GR_{side}_{freq}.csv   # I/O growth functions
{ID}_DP_{side}.csv          # DP-grams
```

where `side` is `L` or `R` and `freq` is `1414` or `4243` (Hz). See
[`load_data.py`](../load_data.py) for the exact expected columns
(`DP (dB)`, `Noise+2sd (dB)`, `Noise+1sd (dB)`, `F1 (dB)`, etc.).

A small subset for smoke-testing can instead go in `data/sample/`
(`DATA_DIR_SAMPLE` in [`config.py`](../config.py)).

## 2. Pre-processed pickle (`load_from_pickle`)

If you already have a cleaned `OAEs` list-of-dicts pickled from a prior run,
point `PICKLE_PATH` at it (see `config.py`).

## Pointing at an external folder

Rather than copying data into the repo, you can set an environment variable
instead of editing `config.py`:

```bash
export DOPAE_DATA_DIR=/path/to/your/csv/folder
python experiment.py
```

(`DOPAE_DATA_DIR_SAMPLE` and `DOPAE_PICKLE_PATH` work the same way.)
