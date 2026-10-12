"""Speech-to-text via faster-whisper (optional dependency). If it is not installed the transcript is reported
as unavailable; nothing is invented."""
import importlib.util, math, threading
from . import config

_lock, _model = threading.Lock(), None


def available() -> bool: return importlib.util.find_spec("faster_whisper") is not None


def transcribe(x16k):
    global _model
    from faster_whisper import WhisperModel
    with _lock:
        if _model is None: _model = WhisperModel(config.get().stt_model, device="cpu", compute_type="int8")
    segs, _ = _model.transcribe(x16k, vad_filter=True)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "confidence": round(math.exp(s.avg_logprob), 3)} for s in segs if s.text.strip()]
