"""Acoustic measurements (numpy only): per-frame tracks, per-window summaries, whole-clip voice profile."""
import numpy as np

SR, FRAME, HOP = 16000, 400, 160          # 16 kHz, 25 ms frames, 10 ms hop
FRAME_S = HOP / SR


def _frames(x):
    if len(x) < FRAME: x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    return x[np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]]


ABS_FLOOR = 0.0018        # about -55 dBFS: below this a frame is treated as silence / room noise


def speech_from_rms(rms): return rms > max(ABS_FLOOR, 0.05 * rms.max())


def speech_mask(x):
    """Per-frame speech activity for a whole clip. Pass audio BEFORE peak normalisation so the absolute floor means something."""
    f = _frames(x) * np.hanning(FRAME); return speech_from_rms(np.sqrt((f ** 2).mean(1)) + 1e-9)


def frame_track(x, mask=None):
    raw = _frames(x); f = raw * np.hanning(FRAME)
    rms = np.sqrt((f ** 2).mean(1)) + 1e-9
    S = np.abs(np.fft.rfft(f, 512, axis=1)) ** 2 + 1e-12
    fr = np.fft.rfftfreq(512, 1 / SR)
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)
    centroid = (S * fr).sum(1) / S.sum(1)
    ac = np.fft.irfft(np.abs(np.fft.rfft(f, 1024, axis=1)) ** 2, 1024, axis=1)       # autocorrelation
    lo, hi = SR // 400, SR // 70
    lag = ac[:, lo:hi].argmax(1) + lo
    strength = ac[np.arange(len(f)), lag] / (ac[:, 0] + 1e-12)
    speech = speech_from_rms(rms) if mask is None else mask[:len(rms)]
    voiced = (strength > .5) & speech
    return dict(rms=rms, flat=flat, centroid=centroid, speech=speech, voiced=voiced,
                pitch=np.where(voiced, SR / lag, 0.0))


def window_features(x, mask=None):
    """Small summary used for speech-activity weighting and per-window display. `mask` = this window's slice of the clip-level speech mask."""
    t = frame_track(x, mask); sp, v = t["speech"], t["voiced"]
    return {"speech_ratio": float(sp.mean()), "voiced_ratio": float(v.mean()),
            "energy_db": float(20 * np.log10(t["rms"][sp].mean())) if sp.any() else None,
            "pitch_hz": float(t["pitch"][v].mean()) if v.sum() >= 5 else None}


def voice_profile(x, mask=None):
    """Acoustic measurements of this recording. Not identity or demographic estimates."""
    t = frame_track(x, mask); sp, v = t["speech"], t["voiced"]; n = len(sp)
    p = t["pitch"][v]; db = 20 * np.log10(t["rms"][sp]) if sp.any() else np.array([])
    env = np.convolve(t["rms"], np.ones(5) / 5, mode="same")
    peaks = int(((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > .5 * env.max())).sum()) if n > 3 else 0
    speech_s = float(sp.sum() * FRAME_S)
    d = np.diff(np.concatenate([[1], sp.astype(int), [1]])); starts, ends = np.where(d == -1)[0], np.where(d == 1)[0]
    pauses = int(sum(1 for a, b in zip(starts, ends) if (b - a) * FRAME_S >= .2 and a > 0 and b < n))
    r = lambda v_, k=2: None if v_ is None else round(float(v_), k)
    ok = v.sum() >= 5
    return {"pitch_hz_mean": r(p.mean()) if ok else None, "pitch_hz_std": r(p.std()) if ok else None,
            "pitch_range_semitones": r(12 * np.log2(np.percentile(p, 95) / np.percentile(p, 5))) if ok else None,
            "energy_db_mean": r(db.mean()) if len(db) else None, "energy_db_std": r(db.std()) if len(db) else None,
            "speech_ratio": r(sp.mean(), 3), "voiced_ratio": r(v.mean(), 3), "speech_seconds": r(speech_s),
            "pause_count": pauses, "syllable_rate_proxy_hz": r(peaks / speech_s) if speech_s > 1 else None,
            "spectral_centroid_hz_mean": r(t["centroid"][sp].mean(), 1) if sp.any() else None,
            "spectral_flatness_mean": r(t["flat"][sp].mean(), 4) if sp.any() else None,
            "note": "Measurements of this recording only. syllable_rate_proxy_hz counts loudness peaks, it is not a transcript-based speech rate."}
