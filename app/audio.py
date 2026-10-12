"""Ingestion (decode + validate) and normalization."""
import io, wave
import numpy as np

TARGET_SR = 16000
MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 600


class AudioError(ValueError):
    pass


def decode(data: bytes):
    """bytes -> (mono float32 in [-1,1], sample_rate). WAV natively; others need `soundfile`."""
    if len(data) > MAX_BYTES: raise AudioError("File too large (25 MB max).")
    if data[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(data)) as w:
                ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                raw = w.readframes(n)
        except wave.Error as e:
            raise AudioError(f"Unreadable WAV: {e}")
        if sw == 2: x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        elif sw == 1: x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
        elif sw == 4: x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
        elif sw == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)); v = np.where(v >= 2**23, v - 2**24, v)
            x = v.astype(np.float32) / 2**23
        else: raise AudioError("Unsupported WAV sample width.")
        x = x.reshape(-1, ch).mean(axis=1)
    else:
        try:
            import soundfile as sf
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True); x = x.mean(axis=1)
        except ImportError:
            raise AudioError("Only WAV is supported without the optional 'soundfile' package.")
        except Exception as e:
            raise AudioError(f"Could not decode audio: {e}")
    if len(x) == 0: raise AudioError("Audio is empty.")
    if len(x) / sr > MAX_SECONDS: raise AudioError(f"Audio longer than {MAX_SECONDS}s.")
    return x.astype(np.float32), int(sr)


def resample(x, sr, target=TARGET_SR):
    """Band-limited FFT resampling (no aliasing when downsampling)."""
    if sr == target: return x
    n, m = len(x), max(1, round(len(x) * target / sr))
    X = np.fft.rfft(x); Y = np.zeros(m // 2 + 1, dtype=complex); k = min(len(X), len(Y)); Y[:k] = X[:k]
    return (np.fft.irfft(Y, m) * (m / n)).astype(np.float32)


def to_16k(x, sr):
    """Resample to 16 kHz and remove DC offset. Absolute level is preserved (used for speech-activity detection)."""
    x = resample(x, sr); return (x - x.mean()).astype(np.float32)


def peak_normalize(x):
    p = np.abs(x).max(); return (x * (0.95 / p) if p > 1e-6 else x).astype(np.float32)


def normalize(x, sr):
    """mono -> 16 kHz -> remove DC offset -> peak-normalize to 0.95."""
    return peak_normalize(to_16k(x, sr))


def encode_wav(x, sr=TARGET_SR):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()
