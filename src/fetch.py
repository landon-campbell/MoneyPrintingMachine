from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import (
    RAW_DIR, TICKERS, USE_ALPACA,
    ALPACA_KEY, ALPACA_SECRET,           # only defined when USE_ALPACA
)

# Optional import guarded to avoid extra dependency when Alpaca is off
if USE_ALPACA:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame



# helpers
def _save(df: pd.DataFrame, tag: str, suffix: str) -> Path:
    fn = RAW_DIR / f"{tag}_{suffix}.csv"
    df.to_csv(fn)
    return fn


# yfinance path (always available)
def _yahoo_bars(symbol: str, interval: str, start: str | None = None, end: str | None   = None) -> pd.DataFrame:
    # Wrapper around yfinance.download with timezone stripped.
    # Intraday bars limited to the last 60 days by Yahoo Finance terms :contentReference[oaicite:0]{index=0}.
    df = yf.download(
        symbol,
        interval=interval,
        start=start,
        end=end,
        progress=False,
    )
    df.index = df.index.tz_localize(None)
    return df



# Alpaca path (used for companies with greater data than Yahoo, but doesn't work for SP500, NASDAQ, DOW)
def _alpaca_bars(symbol: str, tf: TimeFrame, start: str, end: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    bars = client.get_stock_bars(symbol, tf, start=start, end=end).df
    bars.index = bars.index.tz_localize(None)
    return bars



# public API
def fetch_one(tag: str, symbol: str, *, daily_start: str = "1980-01-01", intraday_days: int = 60) -> None:
    """
    Grabs daily history back to 1980 and intraday bars.

    When USE_ALPACA is True, intraday bars span the full daily_start‑to‑today
    window; otherwise, intraday is capped at ~60 days (Yahoo limit).
    """
    # ---------- daily ----------
    daily = _yahoo_bars(symbol, "1d", start=daily_start,
                        end=datetime.today().strftime("%Y-%m-%d"))
    _save(daily, tag, "1d")

    # ---------- intraday ----------
    if USE_ALPACA:
        intraday = _alpaca_bars(
            symbol,
            TimeFrame.Minute,                # 1‑min bars
            start=daily_start,
            end=datetime.today().strftime("%Y-%m-%d"),
        )
    else:
        intraday = _yahoo_bars(
            symbol,
            "30m",
            start=None,
            end=None,                        # yfinance handles 60‑day window
        )

    _save(intraday, tag, "intraday")


def fetch_all() -> None:
    for tag, symbol in tqdm(TICKERS.items(), desc="Fetching data"):
        fetch_one(tag, symbol)


# CLI convenience
if __name__ == "__main__":
    fetch_all()
