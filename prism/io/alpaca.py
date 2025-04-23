# prism/io/alpaca.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe  import TimeFrame
from alpaca.data.requests   import StockBarsRequest
import pandas as pd

from ..config import ALPACA_KEY, ALPACA_SEC          # ← import keys from config


def get_intraday(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download minute‑bars for a single symbol between `start` and `end`
    (ISO‑date strings).  Returns a DataFrame indexed by naive timestamps.
    """
    # pass keys explicitly so alpaca‑py never needs to search env vars
    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SEC)

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    df = client.get_stock_bars(req).df

    # Alpaca returns a MultiIndex (symbol, ts) → drop the symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)

    return df.tz_localize(None)
