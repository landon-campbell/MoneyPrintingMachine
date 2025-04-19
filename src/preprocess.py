from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend
from pathlib import Path

from .config import RAW_DIR, PROC_DIR

PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw(tag: str, suffix: str = "intraday") -> pd.DataFrame:
    """
    Load a CSV produced by fetch.py.  `suffix` is 'intraday' or '1d'.
    """
    fn = RAW_DIR / f"{tag}_{suffix}.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


def resample_volume(df: pd.DataFrame, rule: str) -> pd.Series:
    """
    Resample the Volume column by pandas offset alias (`'W'`, `'M'`, …).
    """
    return df["Volume"].resample(rule).sum()


def log_detrend(series: pd.Series) -> pd.Series:
    """
    Log‑transform + linear detrend to remove growth while preserving shape.
    """
    x = np.log1p(series.values)
    x_d = detrend(x)
    return pd.Series(x_d, index=series.index)
