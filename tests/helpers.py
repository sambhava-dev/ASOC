"""Test-only signal generator (not shipped as a detector)."""
import numpy as np
from app.audio import TARGET_SR


def make_sample(kind="human", seconds=10.0, sr=TARGET_SR, seed=7):
    """Generated voiced test signals. NOT real speech or a real deepfake: they only differ in how
    irregular pitch/amplitude are, which is exactly what the demo provider looks at.
    'mixed' = first half 'human', second half 'synthetic'."""
    if kind == "mixed":
        return np.concatenate([make_sample("human", seconds / 2, sr, seed), make_sample("synthetic", seconds / 2, sr, seed)])
    rng = np.random.default_rng(seed); n = int(seconds * sr); t = np.arange(n) / sr
    def smooth(v, k): return np.convolve(v, np.ones(k) / k, mode="same")
    if kind == "human":
        f0 = 120 + 15 * np.sin(2 * np.pi * .4 * t) + 8 * smooth(rng.standard_normal(n), sr // 8) * 12
        f0 += 2.5 * smooth(rng.standard_normal(n), 40) * 6                  # period-to-period jitter
        env = np.clip(.55 + .45 * np.sin(2 * np.pi * 3.1 * t + rng.uniform(0, 6)), 0, 1) ** 1.5
        env *= 1 + .12 * smooth(rng.standard_normal(n), 60) * 4             # shimmer
        noise = .02
    else:
        f0 = 140 + 10 * np.sin(2 * np.pi * .2 * t)
        env = np.full(n, .8); noise = .0005
    ph = np.cumsum(2 * np.pi * f0 / sr)
    x = sum(np.sin(k * ph) / k for k in range(1, 9)) * env
    x = x / np.abs(x).max() * .8 + noise * rng.standard_normal(n)
    return x.astype(np.float32)
