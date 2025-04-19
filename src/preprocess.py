import numpy as np
import pandas as pd
from scipy.signal import detrend
from pathlib import Path

from config import RAW_DIR, PROC_DIR
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_raw(tag: str, freq: str = "1d") -> pd.DataFrame:
    fn = RAW_DIR / f"{tag}_{freq}.csv"
    return pd.read_csv(fn, parse_dates=["Date" if "Date" in open(fn).read(100) else "Datetime"],
                       index_col=0)

def resample_volume(df: pd.DataFrame, rule: str) -> pd.Series:
    # Returns volume series resampled by `rule` (e.g. 'W', 'M').
    return df["Volume"].resample(rule).sum()  # pandas handles gaps :contentReference[oaicite:1]{index=1}

def log_detrend(series: pd.Series) -> pd.Series:
    # Remove slow trend (log + linear detrend).
    x = np.log1p(series.values)
    x_d = detrend(x) # removes linear component :contentReference[oaicite:2]{index=2}
    return pd.Series(x_d, index=series.index)

def inflation_adjust(series: pd.Series) -> pd.Series:
    # Placeholder volume is already a count measure (no $), so no CPI adjust
    #needed on short horizons, but left here for completeness:contentReference[oaicite:3]{index=3}.
    return series
