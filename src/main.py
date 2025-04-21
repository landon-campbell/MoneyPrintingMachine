from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

from .config      import TICKERS
from .preprocess  import load_raw, log_detrend
from .fourier     import fft_mag, average_ffts, top_k
from .plots       import plot_spectrum


def _get_volume_column(df: pd.DataFrame) -> pd.Series:
    for col in ("Volume", "volume"):
        if col in df.columns:
            return df[col]
    raise KeyError("No volume column found in DataFrame.")


def run_pipeline() -> None:
    tag = "SP500"                       # analyse SPY minute bars
    raw = load_raw(tag, "intraday")

    # split into individual trading days
    day_series = [
        _get_volume_column(group)
        for _, group in raw.groupby(raw.index.date)
    ]

    # keep only days with the **modal length** (usually 390 minutes)
    lengths = [len(s) for s in day_series]
    mode_len = Counter(lengths).most_common(1)[0][0]
    full_days = [s for s in day_series if len(s) == mode_len]

    freqs, mags = [], []
    for day in full_days:
        series = pd.Series(day.values, index=np.arange(len(day)))
        f, m = fft_mag(log_detrend(series))
        freqs.append(f)
        mags.append(m)

    avg_mag = average_ffts(mags)
    plot_spectrum(freqs[0], avg_mag, f"{tag} – average daily spectrum")

    print("Processed days:", len(full_days), "of", len(day_series))
    print("Top‑5 daily harmonics:", top_k(freqs[0], avg_mag, 5))


if __name__ == "__main__":
    run_pipeline()
