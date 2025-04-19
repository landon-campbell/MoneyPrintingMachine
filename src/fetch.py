from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .config import (
    RAW_DIR,
    TICKERS,
    USE_ALPACA,
    ALPACA_KEY,
    ALPACA_SECRET,
)

if USE_ALPACA:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe  import TimeFrame
    from alpaca.data.requests   import StockBarsRequest


def _save(df: pd.DataFrame, tag: str, suffix: str) -> Path:
    fn = RAW_DIR / f"{tag}_{suffix}.csv"
    df.to_csv(fn)
    return fn


def _yahoo_bars(symbol: str, interval: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=interval,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    df.index = df.index.tz_localize(None)
    return df


def _alpaca_bars(symbol: str, timeframe: TimeFrame, start: str, end: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req).df

    # Alpaca returns a MultiIndex: (symbol, timestamp).  Remove symbol level.
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel(0)

    bars.index = bars.index.tz_localize(None)
    return bars


def fetch_one(tag: str, symbol: str, *, daily_start: str = "1980-01-01") -> None:
    today = datetime.today().strftime("%Y-%m-%d")

    daily_df = _yahoo_bars(symbol, "1d", daily_start, today)
    _save(daily_df, tag, "1d")

    if USE_ALPACA:
        minute_df = _alpaca_bars(symbol, TimeFrame.Minute, daily_start, today)
    else:
        minute_df = _yahoo_bars(symbol, "30m")

    _save(minute_df, tag, "intraday")


def fetch_all() -> None:
    for tag, symbol in tqdm(TICKERS.items(), desc="Fetching data"):
        fetch_one(tag, symbol)


if __name__ == "__main__":
    fetch_all()
