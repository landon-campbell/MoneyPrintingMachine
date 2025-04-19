from datetime import datetime
import numpy as np
import pandas as pd

from config import TICKERS
from preprocess import load_raw, resample_volume, log_detrend
from fourier import fft_mag, average_ffts, top_k, reconstruct_signal
from metrics import mse
from plots import plot_spectrum, plot_reconstruction

TAG = "SP500"
raw = load_raw(TAG, "30m")

# Split by day
daily_groups = [g["Volume"] for _, g in raw.groupby(raw.index.date)]

f_list, m_list = [], []
for day in daily_groups:
    ser = pd.Series(day.values, index=np.arange(len(day)))
    f, m = fft_mag(log_detrend(ser))
    f_list.append(f)
    m_list.append(m)

avg_mag = average_ffts(m_list)
freqs   = f_list[0]              # all same length
plot_spectrum(freqs, avg_mag, f"{TAG} average DAILY spectrum")

print("Top-5 daily frequencies:", top_k(freqs, avg_mag, 5))

# Simple reconstruction with top‑N components
N = 5
idx = np.argsort(avg_mag)[::-1][:N]
phase_template = np.angle(np.fft.rfft(log_detrend(daily_groups[-1])))  # last day
mag_sel   = np.zeros_like(avg_mag)
phase_sel = np.zeros_like(phase_template)
mag_sel[idx]   = avg_mag[idx]
phase_sel[idx] = phase_template[idx]
recon = reconstruct_signal(mag_sel, phase_sel, len(daily_groups[-1]))

real_last = pd.Series(daily_groups[-1].values)
pred_last = pd.Series(recon)
print("MSE last day:", mse(real_last, pred_last))
plot_reconstruction(real_last, pred_last, "Last day reconstruction")
