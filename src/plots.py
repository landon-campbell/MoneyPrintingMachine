import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_spectrum(freqs, mag, title):
    plt.figure()
    plt.plot(freqs, mag)
    plt.xlabel("cycles / sample")
    plt.ylabel("magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_reconstruction(real_series: pd.Series, pred_series: pd.Series, title: str):
    plt.figure()
    plt.plot(real_series.index, real_series.values, label="actual")
    plt.plot(pred_series.index, pred_series.values, label="reconstruction", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
