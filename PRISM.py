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
    # helper: greedy select peaks ensuring min distance in days
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
    freq = np.fft.fftfreq(N, d=1.0)

    pos = freq > 0
    freq_pos = freq[pos]
    fft_pos = fft_vals[pos]

    power = np.abs(fft_pos)
    periods = 1.0 / freq_pos

    mask = periods <= max_period_days
    freq2 = freq_pos[mask]
    power2 = power[mask]
    cal2 = periods[mask] * (7.0/5.0)

    # drop first and last bins
    freq_plot = freq2[1:-1]
    power_plot = power2[1:-1]
    cal_plot = cal2[1:-1]

    # truncate to max 3-year period (1095 days)
    keep = cal_plot <= (3 * 365)
    freq_plot = freq_plot[keep]
    power_plot = power_plot[keep]
    cal_plot = cal_plot[keep]

    plt.figure(figsize=(12,6))

    # shade removed high-frequency band
    plt.axvspan(cutoff_freq, freq_plot.max(),
                color='gray', alpha=0.2,
                label='Removed: period < 2 d (data constraint)')
    plt.axvline(cutoff_freq, color='gray', linestyle='--', linewidth=1)
    plt.text(cutoff_freq*1.02, plt.ylim()[1]*0.5,
             '2 d period cutoff', rotation=90,
             va='center', fontsize=8, color='gray')

    plt.plot(freq_plot, power_plot, lw=1)

    # split peaks around 45-day threshold
    threshold_days = 45.0
    peaks, _ = find_peaks(power_plot)
    high_peaks = [p for p in peaks if cal_plot[p] < threshold_days]
    low_peaks = [p for p in peaks if cal_plot[p] >= threshold_days]

    # sort candidates by descending magnitude
    high_sorted = sorted(high_peaks, key=lambda p: power_plot[p], reverse=True)
    low_sorted = sorted(low_peaks, key=lambda p: power_plot[p], reverse=True)

    # greedy select 4 high-frequency (<45d) and 5 low-frequency (>=45d) peaks
    selected_high = greedy_select(high_sorted, power_plot, cal_plot, 4)
    selected_low = greedy_select(low_sorted, power_plot, cal_plot, 5)
    top = np.array(selected_high + selected_low)
    top = top[np.argsort(freq_plot[top])]

    colors = ["red", "purple"]
    for i, idx in enumerate(top):
        days = cal_plot[idx]
        # revised labeling: <30 days => weeks; 30<=d<=366 => months; >366 => years
        if days > 366:
            label = f"{days:.1f} d ({days/365:.1f} yrs)"
        elif days >= 30:
            mo = days / 30.0
            label = f"{days:.1f} d ({mo:.1f} mo)"
        else:
            wk = days / 7.0
            label = f"{days:.1f} d ({wk:.1f} wks)"

        yoff = 8 if (i % 2 == 0) else -12
        col = colors[i % 2]

        plt.scatter(freq_plot[idx], power_plot[idx], color=col, zorder=5)
        plt.annotate(label,
                     (freq_plot[idx], power_plot[idx]),
                     textcoords="offset points",
                     xytext=(0, yoff),
                     ha="center",
                     color=col)

    plt.text(0.05, 0.95,
             "Fundamental decade-long cycle removed",
             transform=plt.gca().transAxes,
             va="top", fontsize=8, color="gray")

    plt.title(f"{title.splitlines()[0]}\n"
              "Low-pass < 2-day period\n"
              "(1/days axis; peaks in calendar days, truncated to 3 yrs)")
    plt.xscale("log")
    plt.xlabel("1/days (log scale)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3, which="both", linestyle="--")
    plt.legend(loc="lower left")


    ax = plt.gca()
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax.get_xlim())
    freq_ticks = np.geomspace(freq_plot.min(), freq_plot.max(), num=6)
    period_labels = [f"{1.0/f:.0f}" for f in freq_ticks]
    ax_top.set_xticks(freq_ticks)
    ax_top.set_xticklabels(period_labels)
    ax_top.set_xlabel("Period (days)")

    known_periods = {
        "Weekly": 5,
        "Monthly": 21,
        "Quarterly": 65,
        "Semiannual": 130,
        "Yearly": 261,
        "Biyearly" : 521
    }
    max_y = np.max(vol_series)

    for label, days in known_periods.items():
        freq = 1 / days
        plt.axvline(freq, color='gray', linestyle='--', alpha=0.6)
        plt.text(freq, 0.95, label,
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

    cutoff = 0.5  # 1/(2 d)
    plot_fft(volume,
             title=f"{ticker} Daily Volume FFT (2014–2024)",
             max_period_days=3650,
             cutoff_freq=cutoff)

if __name__ == "__main__":
    main()
