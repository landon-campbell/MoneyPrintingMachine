#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fetch_volume(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download raw (unadjusted) daily trading volume for the given ticker
    between start and end dates. Returns a pandas Series indexed by date.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    return df["Volume"].dropna()

def plot_fft(
    vol_series: pd.Series,
    title: str,
    max_period_days: float,
    color: str
) -> None:
    """
    Compute FFT of the input volume series and plot:
    - amplitude vs. period (in days, log scale on x-axis),
    - annotate the ten strongest peaks up to max_period_days,
    - use `color` for title, plot line, points, and annotations.
    """
    vol = vol_series.values
    vol_centered = vol - np.mean(vol)
    N = len(vol_centered)

    fft_vals = np.fft.fft(vol_centered)
    freq = np.fft.fftfreq(N, d=1.0)
    pos = freq > 0
    freq = freq[pos]
    power = np.abs(fft_vals[pos])

    periods = 1.0 / freq
    periods = np.asarray(periods).ravel()
    power = np.asarray(power).ravel()

    mask = periods <= max_period_days
    periods = periods[mask]
    power = power[mask]

    peaks, _ = find_peaks(power)
    if len(peaks) >= 10:
        top = peaks[np.argsort(power[peaks])][-10:]
    else:
        top = peaks

    plt.figure(figsize=(12, 6))
    plt.plot(periods, power, color=color)
    plt.scatter(periods[top], power[top], color=color, zorder=5)
    for idx in top:
        plt.annotate(f"{periods[idx]:.1f} d",
                     (periods[idx], power[idx]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", color=color)

    plt.xscale("log")
    plt.xlim(1, max_period_days)
    plt.title(title, color=color)
    plt.xlabel("Period (days, log scale)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3, which="both", linestyle="--")
    plt.tight_layout()
    plt.show()

def main():
    volume = fetch_volume("AMZN", "2014-01-01", "2024-12-31")
    plot_fft(
        volume,
        title="AMZN Daily Volume FFT (2014–2024)",
        max_period_days=3650,
        color="blue"
    )

if __name__ == "__main__":
    main()
