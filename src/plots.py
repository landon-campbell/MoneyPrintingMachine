import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _minutes_for_harmonics(harmonics, samples_per_day):
    # period (minutes) = 1 / (cycles per minute)
    # cycles per minute = harmonics / (samples_per_day / 390)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(harmonics == 0,
            np.nan,
            (samples_per_day / harmonics),
        )


def plot_spectrum(freqs: np.ndarray,
                  mag:   np.ndarray,
                  title: str,
                  samples_per_day: int = 390) -> None:
    """
    freqs – numpy.fft.rfftfreq output (cycles per sample)
    mag   – magnitude array
    """
    harmonics = freqs * samples_per_day

    fig, ax1 = plt.subplots()
    ax1.plot(harmonics, mag)
    ax1.set_xlabel("cycles per trading day (harmonic number)")
    ax1.set_ylabel("magnitude")
    ax1.set_title(title)

    # secondary axis with period in minutes
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    tick_vals = ax1.get_xticks()
    period_labels = [
        f"{_minutes_for_harmonics(h, samples_per_day):.0f}"
        if h != 0 else ""
        for h in tick_vals
    ]
    ax2.set_xticks(tick_vals)
    ax2.set_xticklabels(period_labels)
    ax2.set_xlabel("equivalent period (minutes)")

    fig.tight_layout()
    plt.show()


def plot_reconstruction(real_series: pd.Series, pred_series: pd.Series, title: str):
    plt.figure()
    plt.plot(real_series.index, real_series.values, label="actual")
    plt.plot(pred_series.index, pred_series.values, label="reconstruction", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
