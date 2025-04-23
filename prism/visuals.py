"""
Enhanced plotting for PRISM

• Produces a multi‑band frequency‑spectrum plot with:
    – per‑band log‑scaled y‑axis (better contrast)
    – red‑dot peak markers labelled with units (min or d)
    – dashed reference lines for common calendar cycles

• Saves an additional zoom‑in plot of the first band.

All numeric labels carry a unit so no bare numbers appear.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------------------------------------------------------------
KNOWN_PERIODS = {
    "Daily": 1,
    "Weekly": 7,
    "Monthly": 30,
    "Quarterly": 90,
    "Semiannual": 180,
    "Yearly": 365,
    "Biyearly": 730,
}

# first band starts above zero to avoid NaN / Inf axis limits
BANDS = [
    (0.001, 0.020),
    (0.020, 0.040),
    (0.040, 0.060),
    (0.060, 0.080),
    (0.080, 0.100),
    (0.100, 0.120),
    (0.140, 0.160),
    (0.160, 0.180),
    (0.180, 0.200),
    (0.990, 1.010),
]


# --------------------------------------------------------------------------
def _freq_to_period(freq):
    with np.errstate(divide="ignore"):
        return np.where(freq > 0, 1.0 / freq, np.nan)


def _period_to_freq(period):
    with np.errstate(divide="ignore"):
        return np.where(period > 0, 1.0 / period, 0.0)


def _period_label(period_days):
    """Return string with unit — days or minutes."""
    if period_days >= 1:
        return f"{period_days:.0f} d"
    minutes = period_days * 1440.0
    return f"{minutes:.0f} min"


def _annotate_peaks(ax, fmin, fmax, peak_freqs, peak_mags, max_y):
    """Scatter and label peaks that fall inside the current band."""
    for f, m in zip(peak_freqs, peak_mags):
        if fmin <= f <= fmax and m > 0:
            period_d = 1.0 / f
            ax.scatter(f, m, color="red", s=20, zorder=3)
            ax.text(
                f,
                m * 1.05,
                _period_label(period_d),
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=45,
            )


# --------------------------------------------------------------------------
def save_spectrum(
    freqs,
    mags,
    peak_freqs,
    peak_mags,
    output_dir,
    basename="fft_spectrum",
):
    """
    Save multi‑band + zoomed FFT plots.

    Parameters
    ----------
    freqs, mags   : 1‑D numpy arrays of positive‑frequency bins
    peak_freqs,
    peak_mags     : arrays of detected peak positions / magnitudes
    output_dir    : folder to write PNGs
    basename      : base filename without extension
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- multi‑band figure -----------------------------------------
    fig, axes = plt.subplots(
        len(BANDS), 1, figsize=(16, 20), sharey=False, constrained_layout=True
    )

    for ax, (fmin, fmax) in zip(axes, BANDS):
        mask = (freqs >= fmin) & (freqs <= fmax)
        f_zoom = freqs[mask]
        m_zoom = mags[mask]

        ax.plot(f_zoom, m_zoom, lw=1.0)
        ax.set_xlim(fmin, fmax)

        # per‑band log scale for contrast
        band_max = m_zoom.max() if m_zoom.size else 1e-6
        ax.set_yscale("log")
        ax.set_ylim(band_max * 1e-3, band_max * 1.2)
        ax.set_ylabel("Magnitude")

        _annotate_peaks(ax, fmin, fmax, peak_freqs, peak_mags, band_max)

        # reference calendar lines
        for lbl, days in KNOWN_PERIODS.items():
            f_line = 1.0 / days
            if fmin < f_line < fmax:
                ax.axvline(f_line, color="gray", linestyle="--", alpha=0.6)
                ax.text(
                    f_line,
                    band_max * 0.9,
                    lbl,
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="gray",
                )

        ax.grid(True)

        secax = ax.secondary_xaxis("top", functions=(_freq_to_period, _period_to_freq))
        secax.set_xlabel("Period (days)")
        secax.set_xticks([1, 7, 30, 90, 180, 365])
        secax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    axes[-1].set_xlabel("Frequency (cycles per day)")
    axes[0].set_title("Frequency Spectrum of Trading Volume")

    multi_path = output_dir / f"{basename}.png"
    fig.savefig(multi_path, dpi=300)
    plt.close(fig)

    # ---------- zoom‑in on first band -------------------------------------
    fmin, fmax = BANDS[0]
    mask = (freqs >= fmin) & (freqs <= fmax)
    f_zoom = freqs[mask]
    m_zoom = mags[mask]

    fig2, ax2 = plt.subplots(figsize=(20, 6))
    ax2.plot(f_zoom, m_zoom, lw=1.0)
    ax2.set_xlim(fmin, fmax)

    band_max = m_zoom.max() if m_zoom.size else 1e-6
    ax2.set_yscale("log")
    ax2.set_ylim(band_max * 1e-3, band_max * 1.2)

    _annotate_peaks(ax2, fmin, fmax, peak_freqs, peak_mags, band_max)

    for lbl, days in KNOWN_PERIODS.items():
        f_line = 1.0 / days
        if fmin < f_line < fmax:
            ax2.axvline(f_line, color="gray", linestyle="--", alpha=0.6)
            ax2.text(
                f_line,
                band_max * 0.9,
                lbl,
                rotation=90,
                ha="right",
                va="top",
                fontsize=9,
                color="gray",
            )

    ax2.set_xlabel("Frequency (cycles per day)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Zoomed Frequency Spectrum: 0.001–0.04 cycles/day")
    ax2.grid(True)

    secax2 = ax2.secondary_xaxis("top", functions=(_freq_to_period, _period_to_freq))
    secax2.set_xlabel("Period (days)")
    secax2.set_xticks([1, 7, 30, 90, 180, 365])
    secax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    zoom_path = output_dir / f"{basename}_zoomed.png"
    fig2.savefig(zoom_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return str(multi_path), str(zoom_path)
