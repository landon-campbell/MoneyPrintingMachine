"""
Pre‑processing helpers for PRISM
- split minute bars into weekday trading days
- detrend series
- load or resample volume data
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import detrend

from .config import RAW, PROC

PROC.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
def split_trading_days(df: pd.DataFrame) -> list[pd.Series]:
    """
    Split minute‑bar DataFrame into one Series per **weekday** (Mon‑Fri).
    No holiday calendar, no length check. Accepts 'Volume' or 'volume'.
    """
    vol_col = (
        "Volume"
        if "Volume" in df.columns
        else "volume" if "volume" in df.columns
        else None
    )
    if vol_col is None:
        raise KeyError("no 'Volume' or 'volume' column found")

    df = df.sort_index()

    days = [
        block[vol_col]
        for day, block in df.groupby(df.index.date)
        if block.index[0].weekday() < 5  # 0=Mon … 4=Fri
    ]
    return days

# --------------------------------------------------------------------------
def load_raw(tag: str, suffix: str = "intraday") -> pd.DataFrame:
    """Load a previously saved CSV from RAW/"""
    fn = RAW / f"{tag}_{suffix}.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


def resample_volume(df: pd.DataFrame, rule: str) -> pd.Series:
    """Sum volume by a pandas offset alias ('W', 'M', 'Q', 'A', …)."""
    return df["Volume"].resample(rule).sum()


def log_detrend(series: pd.Series) -> pd.Series:
    """log1p + linear detrend to stabilise variance."""
    x = np.log1p(series.values)
    return pd.Series(detrend(x), index=series.index)
