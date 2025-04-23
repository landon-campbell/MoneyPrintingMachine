"""
Main orchestration module for PRISM
Creates spectra for
- intraday (minute bars, weekdays only)
- week / month / year / decade tiers
and writes two PNGs via visuals.save_spectrum.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .io.yahoo import get_daily, resample
from .io.alpaca import get_intraday
from .preprocess import log_detrend, split_trading_days
from .fourier import fft_mag
from .visuals import save_spectrum
from .config import PROC, USE_ALPACA


# --------------------------------------------------------------------------
def intraday_spectrum(ticker: str):
    """Return (freq, mag, label) for the latest weekday in a 60‑day window."""
    today = date.today()
    start = today - timedelta(days=60)

    df = get_intraday(ticker, start.isoformat(), today.isoformat())
    days = split_trading_days(df)

    if not days:
        logging.warning("Minute feed had no weekday data; skipping intraday tier")
        return None

    series = days[-1]  # newest weekday
    freq, mag = fft_mag(log_detrend(series))
    return freq * 390, mag, "DAY"  # convert to cycles per trading day


# --------------------------------------------------------------------------
def calendar_spectrum(volume_data, rule: str, label: str):
    """Return (freq, mag, label) after resampling daily volume."""
    frame = volume_data.to_frame() if isinstance(volume_data, pd.Series) else volume_data
    vol   = resample(frame, rule)
    freq, mag = fft_mag(log_detrend(vol).dropna())
    return freq, mag, label


# --------------------------------------------------------------------------
def run_all(ticker: str, start: str = "2000-01-01", end: str | None = None):
    spectra = []

    # intraday tier via Alpaca
    if USE_ALPACA:
        logging.info("Fetching intraday minute bars via Alpaca…")
        intraday = intraday_spectrum(ticker)
        if intraday:
            spectra.append(intraday)
    else:
        logging.info("Skipping intraday tier (PRISM_USE_ALPACA=false)")

    # daily data via Yahoo Finance
    logging.info("Fetching daily history via Yahoo Finance…")
    daily_df = get_daily(ticker, start=start)
    vol_ser  = daily_df["Volume"]

    for rule, label in [
        ("W",  "WEEK"),   # week‑end
        ("ME", "MONTH"),  # month‑end
        ("QE", "YEAR"),   # quarter‑end
        ("YE", "DECADE"), # year‑end
    ]:
        spectra.append(calendar_spectrum(vol_ser, rule, label))

    # ----- merge spectra & detect peaks -----------------------------------
    all_freqs = np.concatenate([s[0] for s in spectra])
    all_mags  = np.concatenate([s[1] for s in spectra])

    peak_idx, _   = find_peaks(all_mags, height=all_mags.max() * 0.10)
    peak_freqs    = all_freqs[peak_idx]
    peak_magnitudes = all_mags[peak_idx]

    # ----- save plots ------------------------------------------------------
    multi_png, zoom_png = save_spectrum(
        freqs=all_freqs,
        mags=all_mags,
        peak_freqs=peak_freqs,
        peak_mags=peak_magnitudes,
        output_dir=PROC,
        basename=f"{ticker}_spectrum",
    )
    logging.info(f"Saved FFT plots: {multi_png}, {zoom_png}")
