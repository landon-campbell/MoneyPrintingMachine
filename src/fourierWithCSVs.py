import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import glob
import matplotlib.ticker as ticker


# === CONFIG ===
input_dir = 'data/volume per day/'
output_dir = 'output/'
fft_plot_file = os.path.join(output_dir, 'fft_spectrum.png')
decomp_plot_file = os.path.join(output_dir, 'seasonal_decomposition')
periods_csv_file = os.path.join(output_dir, 'significant_periods.csv')

# === CONCAT CSVS ===
csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
# === 3. Read and concatenate ===
df_list = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)


# === SETUP OUTPUT DIRECTORY ===
os.makedirs(output_dir, exist_ok=True)

# === 1. Load Data ===
df = combined_df
print(f"Loaded {len(csv_files)} CSV files")
print(f"Combined DataFrame has {len(df)} rows")
df.columns = df.columns.str.strip().str.lower()
if 'date' not in df.columns:
    raise ValueError("The DataFrame does not contain a 'date' column.")
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
df = df.dropna(subset=['date'])
df.set_index('date', inplace=True)
print(type(df.index))
df.to_csv("./output/combined.csv")
df = df.sort_index()

# === 2. Resample to daily frequency ===
df = df.resample('D').mean()
df['volume'] = df['volume'].interpolate()

# === 3. Detrend the data ===
volume_detrended = detrend(df['volume'].values)

# === 4. Perform FFT ===
fft_vals = np.fft.fft(volume_detrended)
freqs = np.fft.fftfreq(len(volume_detrended), d=1)  # daily spacing

# Positive frequencies only
positive_freqs = freqs[freqs > 0]
magnitudes = np.abs(fft_vals[freqs > 0])

# === 5. Identify significant peaks ===
peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)
peak_freqs = positive_freqs[peaks]
peak_mags = magnitudes[peaks]
peak_periods = 1 / peak_freqs

# Filter and export significant periods
filtered_periods = [(1/f, m) for f, m in zip(peak_freqs, peak_mags) if 5 < 1/f < 1000]
periods_df = pd.DataFrame(filtered_periods, columns=['Period_Days', 'Magnitude'])
periods_df.sort_values(by='Magnitude', ascending=False, inplace=True)
periods_df.to_csv(periods_csv_file, index=False)

# === 6 Plotting All Bands===
# Subplot setup
fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharey=True, constrained_layout=True)

# Define ranges to zoom in on
freq_ranges = [
    (0.0000, 0.020),
    (0.020, 0.040),
    (0.040, 0.060),
    (0.060, 0.080),
    (0.080, 0.100),
    (0.100, 0.120),
    (0.140, 0.160),
    (0.160, 0.180),
    (0.180, 0.200),
    (0.990, 1.010)
]

# Known periodicities
known_periods = {
    "Daily": 1,
    "Weekly": 7,
    "Monthly": 30,
    "Quarterly": 90,
    "Semiannual": 180,
    "Yearly": 365,
    "Biyearly" : 730
}

max_y = np.max(magnitudes)

# Vectorized converters for secondary axis
def freq_to_period(x): return np.where(x != 0, 1 / x, 0)
def period_to_freq(x): return np.where(x != 0, 1 / x, 0)

# Plot each frequency segment
for ax, (fmin, fmax) in zip(axes, freq_ranges):
    # Mask data to range
    mask = (positive_freqs >= fmin) & (positive_freqs <= fmax)
    freqs_zoom = positive_freqs[mask]
    mags_zoom = magnitudes[mask]

    # Plot FFT
    ax.plot(freqs_zoom, mags_zoom, label="FFT Magnitude")
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(0, max_y * 1.15)

    # Annotate peaks within this range
    for i, (f, p, m) in enumerate(zip(peak_freqs, peak_periods, peak_mags)):
        if fmin <= f <= fmax:
            ax.scatter(f, m, color='red')
            ax.text(f, m + 0.02 * max_y, f'{p:.1f}d', fontsize=8, ha='center', va='bottom', rotation=45)

    # Known periodicity lines
    for label, days in known_periods.items():
        freq = 1 / days
        if fmin < freq < fmax:
            ax.axvline(freq, color='gray', linestyle='--', alpha=0.6)
            ax.text(freq, max_y * 0.85, label, rotation=90, va='top', ha='right', fontsize=9, color='gray')

    # Label x and y axes
    ax.set_ylabel("Magnitude")
    ax.grid(True)

    # Secondary axis
    secax = ax.secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
    secax.set_xlabel("Period (days)")
    secax.set_xticks([1, 7, 30, 90, 180, 365])
    secax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

# Add common labels
axes[-1].set_xlabel("Frequency (cycles per day)")
axes[0].set_title("Frequency Spectrum of Trading Volume")

# Save figure
fig.savefig(fft_plot_file, dpi=300)
plt.close(fig)
print(f"Split FFT spectrum saved to: {fft_plot_file}")

# Save figure
fig.savefig(fft_plot_file, bbox_inches='tight')  # ensure all elements fit
plt.close(fig)



# === 6.2 Plotting Meaningful Band ===
# === Save just the first subplot (0.0000 - 0.0200 range) ===
fig_single, ax_single = plt.subplots(figsize=(20, 6))

# Use the same logic for the first frequency range
fmin, fmax = (0.000, 0.040)
mask = (positive_freqs >= fmin) & (positive_freqs <= fmax)
freqs_zoom = positive_freqs[mask]
mags_zoom = magnitudes[mask]

# Plot
ax_single.plot(freqs_zoom, mags_zoom, label="FFT Magnitude")
ax_single.set_xlim(fmin, fmax)
ax_single.set_ylim(0, max_y * 1.15)

# Annotate peaks in this range
for f, p, m in zip(peak_freqs, peak_periods, peak_mags):
    if fmin <= f <= fmax:
        ax_single.scatter(f, m, color='red')
        ax_single.text(f, m + 0.02 * max_y, f'{p:.1f}d', fontsize=8, ha='center', va='bottom', rotation=45)

# Known periodicity lines
for label, days in known_periods.items():
    freq = 1 / days
    if fmin < freq < fmax:
        ax_single.axvline(freq, color='gray', linestyle='--', alpha=0.6)
        ax_single.text(freq, max_y * 0.85, label, rotation=90, va='top', ha='right', fontsize=9, color='gray')

# Axis settings
ax_single.set_xlabel("Frequency (cycles per day)")
ax_single.set_ylabel("Magnitude")
ax_single.set_title("Zoomed Frequency Spectrum: 0–0.04 cycles/day")
ax_single.grid(True)

# Secondary period axis
secax_single = ax_single.secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
secax_single.set_xlabel("Period (days)")
secax_single.set_xticks([1, 7, 30, 90, 180, 365])
secax_single.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

# Save just this figure
zoomed_plot_path = fft_plot_file.replace('.png', '_zoomed.png')
fig_single.savefig(zoomed_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig_single)
print(f"Zoomed-in FFT saved to: {zoomed_plot_path}")





# Plot	    Shows...	                    Useful For...
# Observed	Raw data	                    Seeing everything together
# Trend	    Long-term direction	            Detecting economic or structural shifts
# Seasonal	Repeating patterns	            Identifying cyclic market behaviors
# Residual	What's not explained by above	Spotting anomalies or randomness

# === 7.1 Seasonal Pattern: Yearly (365 days) ===
decomp_yearly = seasonal_decompose(df['volume'], model='additive', period=365)
seasonal_yearly = decomp_yearly.seasonal

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(seasonal_yearly, linewidth=2)
ax.set_title("Yearly Seasonal Component (Period = 365 days)")
ax.set_xlabel("Date")
ax.set_ylabel("Seasonal Effect")
ax.tick_params(axis='x', rotation=45)
ax.grid(True)
fig.tight_layout()
fig.savefig(f"{decomp_plot_file}_365_seasonal_only.png")
plt.close(fig)

# === 7.2 Seasonal Pattern: Monthly (30 days) ===
decomp_monthly = seasonal_decompose(df['volume'], model='additive', period=30)
seasonal_monthly = decomp_monthly.seasonal.loc['2021-01-01':'2024-01-01']

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(seasonal_monthly, linewidth=2)
ax.set_title("Monthly Seasonal Component (Period = 30 days)")
ax.set_xlabel("Date")
ax.set_ylabel("Seasonal Effect")
ax.tick_params(axis='x', rotation=45)
ax.grid(True)
fig.tight_layout()
fig.savefig(f"{decomp_plot_file}_30_seasonal_only.png")
plt.close(fig)

# === 7.3 Seasonal Pattern: Weekly (7 days) ===
decomp_weekly = seasonal_decompose(df['volume'], model='additive', period=7)
seasonal_weekly = decomp_weekly.seasonal.loc['2023-01-01':'2023-12-31']

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(seasonal_weekly, linewidth=2)
ax.set_title("Weekly Seasonal Component (Period = 7 days)")
ax.set_xlabel("Date")
ax.set_ylabel("Seasonal Effect")
ax.tick_params(axis='x', rotation=45)
ax.grid(True)
fig.tight_layout()
fig.savefig(f"{decomp_plot_file}_7_seasonal_only.png")
plt.close(fig)



print(f"Significant periods saved to: {periods_csv_file}")
print(f"FFT spectrum saved to: {fft_plot_file}")
print(f"Seasonal decomposition saved to: {decomp_plot_file}")
