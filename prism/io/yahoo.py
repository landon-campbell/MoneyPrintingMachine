"""
Yahoo Finance helpers for PRISM
- download daily bars in a single call
- resample volume by week / month / quarter / year
"""

from datetime import datetime
import yfinance as yf
import pandas as pd

# --------------------------------------------------------------------------
def get_daily(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """
    Download daily OHLCV data for `ticker` from Yahoo Finance.

    Parameters
    ----------
    ticker : str
    start  : ISO date string (default "2000-01-01")

    Returns
    -------
    DataFrame indexed by naive Timestamps (no timezone)
    """
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    df.index = df.index.tz_localize(None)
    return df


# --------------------------------------------------------------------------
def resample(df: pd.DataFrame, rule: str, how: str = "sum") -> pd.Series:
    """
    Resample the volume column by a pandas offset alias: "W", "M", "Q", "A", …

    Parameters
    ----------
    df   : DataFrame that contains either "Volume" or "volume"
    rule : resample rule (e.g. "W", "M")
    how  : aggregation method (default "sum")

    Returns
    -------
    Series indexed by the resample interval
    """
    if "Volume" in df.columns:
        vol_col = "Volume"
    elif "volume" in df.columns:
        vol_col = "volume"
    else:
        # fallback: first column (for unnamed Series -> DataFrame conversion)
        vol_col = df.columns[0]

    return getattr(df[vol_col].resample(rule), how)()
