#!/usr/bin/env python3

import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, find_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="FFT volume analyzer")
    parser.add_argument("--ticker", default="AMZN",
                        help="Ticker symbol to fetch (e.g. AMZN)")
    return parser.parse_args()


def fetch_volume(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=False)
    vol = df["Volume"].dropna().sort_index()
    vol.index = pd.to_datetime(vol.index)
    # business-day frequency; fill any gaps (holidays) by carrying forward
    return vol.asfreq("B").ffill()


def low_pass_filter(data, cutoff, fs=1.0, requested_taps=101):
    arr = np.asarray(data).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input, got {arr.shape}")
    N = len(arr)

    numtaps = min(requested_taps, N - 1)
    if numtaps % 2 == 0:
        numtaps -= 1
    nyq = fs / 2.0
    cutoff_norm = min((cutoff/nyq)*0.99, 0.99)

    taps = firwin(numtaps, cutoff_norm, window="hamming")
    padlen = 3 * len(taps)
    while numtaps > 3 and padlen >= N:
        numtaps -= 2
        taps = firwin(numtaps, cutoff_norm, window="hamming")
        padlen = 3 * len(taps)

    data_padded = np.pad(arr, padlen, mode="reflect")
    filtered_padded = filtfilt(taps, 1.0, data_padded)
    return filtered_padded[padlen:-padlen]


def plot_fft(vol_series: pd.Series,
             title: str,
             max_period_days: float,
             cutoff_freq: float) -> None:
    # helper: greedy select peaks ensuring min separation in business days
    def greedy_select(peaks, power, periods, count, min_sep=1.1):
        selected = []
        for p in peaks:
            if len(selected) >= count:
                break
            if all(abs(periods[p] - periods[q]) > min_sep for q in selected):
                selected.append(p)
        return selected

    vals = vol_series.values.astype(float)
    centered = vals - vals.mean()
    filt = low_pass_filter(centered, cutoff=cutoff_freq)

    N = len(filt)
    fft_vals = np.fft.fft(filt)
    freq = np.fft.fftfreq(N, d=1.0)  # d=1 business day

    pos = freq > 0
    freq_pos = freq[pos]
    fft_pos = fft_vals[pos]

    power = np.abs(fft_pos)
    periods = 1.0 / freq_pos       # in business days

    # keep only up to max_period_days
    mask = periods <= max_period_days
    freq_plot = freq_pos[mask][1:-1]
    power_plot = power[mask][1:-1]
    period_plot = periods[mask][1:-1]

    # truncate to 3-year business-day span (~3*261 days)
    keep = period_plot <= (3 * 261)
    freq_plot = freq_plot[keep]
    power_plot = power_plot[keep]
    period_plot = period_plot[keep]

    plt.figure(figsize=(12,6))
    plt.plot(freq_plot, power_plot, lw=1)

    # split around 30 business-day threshold
    threshold = 29
    peaks, _ = find_peaks(power_plot)
    high = [p for p in peaks if period_plot[p] < threshold]
    low  = [p for p in peaks if period_plot[p] >= threshold]

    # take top 4 short and 4 long business-day cycles
    high_sorted = sorted(high, key=lambda p: power_plot[p], reverse=True)
    low_sorted  = sorted(low,  key=lambda p: power_plot[p], reverse=True)
    sel_high = greedy_select(high_sorted, power_plot, period_plot, 5)
    sel_low  = greedy_select(low_sorted,  power_plot, period_plot, 5)
    top = np.array(sel_high + sel_low)
    top = top[np.argsort(freq_plot[top])]

    for idx in top:
        bd = period_plot[idx]
        plt.scatter(freq_plot[idx], power_plot[idx], color="red", s=20, zorder=5)
        plt.annotate(f"{bd:.1f}d",
                     (freq_plot[idx], power_plot[idx]),
                     xytext=(0,5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, rotation=45)

    plt.title(f"{title.splitlines()[0]}\n"
              "Low-pass < 2-business-day period\n"
              "(1/business-days axis; peaks in business days, truncated to 3 yrs)")
    plt.xscale("log")
    plt.xlabel("1 / (business days)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3, which="both", linestyle="--")

    # Mark common business-day cycles
    known = {
        "Weekly":    5,
        "Monthly":  21,
        "Quarter":  63,   # ~3×21
        "Semiannual": 126,
        "Yearly":   252,  # ~52 weeks × 5
    }
    ax = plt.gca()
    for label, days in known.items():
        f = 1 / days
        ax.axvline(f, color='gray', linestyle='--', alpha=0.6)
        ax.text(f, 0.9, label,
                rotation=90, va='top', ha='right',
                fontsize=9, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    start_date, end_date = "2014-01-01", "2024-12-31"

    volume = fetch_volume(ticker, start_date, end_date)
    print(f"N = {len(volume)} samples for {ticker} "
          f"from {volume.index.min()} to {volume.index.max()}")

    cutoff = 0.5  # 1/(2 business days)
    plot_fft(volume,
             title=f"{ticker} Daily Volume FFT (2014–2024)",
             max_period_days=3650,
             cutoff_freq=cutoff)


if __name__ == "__main__":
    main()
