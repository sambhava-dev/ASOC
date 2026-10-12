"""Log-mel front end in numpy. Shared by training and inference so they can never drift apart."""
from functools import lru_cache
import numpy as np

SR, N_FFT, WIN, HOP, N_MELS = 16000, 512, 400, 160, 80


def _hz2mel(f): return 2595 * np.log10(1 + f / 700)
def _mel2hz(m): return 700 * (10 ** (m / 2595) - 1)


@lru_cache
def _fb():
    hz = _mel2hz(np.linspace(_hz2mel(20), _hz2mel(SR / 2), N_MELS + 2)); bins = np.fft.rfftfreq(N_FFT, 1 / SR)
    W = np.zeros((N_MELS, len(bins)), np.float32)
    for i in range(N_MELS):
        l, c, r = hz[i:i + 3]; W[i] = np.maximum(0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    return W


def fit_length(x, n):
    """Tile (repeat) short audio, crop long audio to exactly n samples."""
    if len(x) >= n: return x[:n]
    return np.tile(x, -(-n // len(x)))[:n]


def logmel(x):
    """float32 waveform -> (N_MELS, frames) log-mel, mean/variance normalised per window."""
    if len(x) < WIN: x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    f = x[np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]] * np.hanning(WIN)
    S = np.log(_fb() @ (np.abs(np.fft.rfft(f, N_FFT, axis=1)) ** 2).T + 1e-6)
    return ((S - S.mean()) / (S.std() + 1e-5)).astype(np.float32)


def band_edges_hz(groups=8):
    hz = _mel2hz(np.linspace(_hz2mel(20), _hz2mel(SR / 2), N_MELS + 1)); step = N_MELS // groups
    return [(int(hz[i * step]), int(hz[(i + 1) * step])) for i in range(groups)]
