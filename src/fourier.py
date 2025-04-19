import numpy as np
import pandas as pd

def fft_mag(series: pd.Series, window: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    # Returns frequencies (cycles per sample) and normalized magnitude.
    x = series.values
    if window is not None:
        x = x * window
    n  = len(x)
    freqs = np.fft.rfftfreq(n)  # 0…0.5 cycles/sample
    mag   = np.abs(np.fft.rfft(x)) / n
    return freqs, mag

def average_ffts(list_of_mags: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(list_of_mags)
    return stacked.mean(axis=0)

def reconstruct_signal(mag: np.ndarray, phases: np.ndarray, length: int) -> np.ndarray:
    # Build synthetic signal from magnitude + phase, then IFFT.
    spectrum = mag * np.exp(1j * phases)
    full_spec = np.concatenate([spectrum,
                                np.conj(spectrum[-2:0:-1])])  # Hermitian mirror
    return np.fft.ifft(full_spec * length).real

def top_k(freqs: np.ndarray, mag: np.ndarray, k: int = 5) -> list[tuple[float, float]]:
    idx = np.argsort(mag)[::-1][:k]
    return list(zip(freqs[idx], mag[idx]))
