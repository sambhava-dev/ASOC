"""Detection providers. The pipeline only depends on this interface, so models can be swapped."""
import hashlib, threading
from dataclasses import dataclass
import numpy as np
from .. import repo
from .melspec import SR, fit_length, logmel, band_edges_hz, N_MELS


class ProviderUnavailable(RuntimeError): pass


@dataclass
class ProviderOutput:
    p_synthetic: float                 # calibrated probability that the window is not bona fide
    source_probs: dict | None = None   # closed-set attack-system probabilities, only if the model has them


class DetectionProvider:
    meta: dict
    window_s: float
    def score_windows(self, wins): raise NotImplementedError
    def explain(self, win): return {}


class MLDetectionProvider(DetectionProvider):
    def __init__(self, row):
        try:
            import torch
            from .model import SpoofNet
        except ImportError as e:
            raise ProviderUnavailable("PyTorch is not installed (pip install torch).") from e
        self.torch = torch
        try:
            blob = open(row["path"], "rb").read()
        except OSError:
            raise ProviderUnavailable(f"Model file is missing: {row['path']}")
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise ProviderUnavailable("Model file does not match the checksum in the registry; refusing to load it.")
        ck = torch.load(row["path"], map_location="cpu", weights_only=True)
        m = ck["meta"]; self.classes = list(m["classes"]); self.T = float(m.get("temperature", 1.0))
        self.window_s = float(m["window_s"]); self.n = int(self.window_s * SR)
        self.net = SpoofNet(len(self.classes), int(m.get("width", 32))); self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.meta = {"id": row["id"], "name": row["name"], "version": row["version"], "validated": bool(row["validated"]),
                     "trained_on": row["trained_on"], "metrics": m.get("metrics", {}), "supports_source": len(self.classes) > 2}

    def _probs(self, specs):
        t = self.torch
        with t.inference_mode():
            x = t.from_numpy(np.stack(specs))[:, None]
            return t.softmax(self.net(x) / self.T, dim=1).numpy()

    def score_windows(self, wins):
        out = []
        for i in range(0, len(wins), 32):
            P = self._probs([logmel(fit_length(w, self.n)) for w in wins[i:i + 32]])
            for p in P:
                src = None
                if len(self.classes) > 2:
                    s = p[1:] / max(p[1:].sum(), 1e-9); src = {c: float(v) for c, v in zip(self.classes[1:], s)}
                out.append(ProviderOutput(float(1 - p[0]), src))
        return out

    def explain(self, win):
        """Occlusion sensitivity: how much does p(synthetic) fall when one frequency band is neutralised?
        Positive = that band pushed the score toward synthetic. A property of this model, not of the speaker."""
        S = logmel(fit_length(win, self.n)); base = 1 - self._probs([S])[0][0]
        edges, step = band_edges_hz(8), N_MELS // 8; masked = []
        for g in range(8):
            M = S.copy(); M[g * step:(g + 1) * step] = 0; masked.append(M)
        P = self._probs(masked)
        return {f"{lo}-{hi} Hz": round(float(base - (1 - p[0])), 4) for (lo, hi), p in zip(edges, P)}


class ModelManager:
    """Loads the active model from the registry once and reuses it (hash-map cache keyed by model id)."""
    def __init__(self): self.cache, self.lock = {}, threading.Lock()

    def get(self) -> DetectionProvider:
        row = repo.active_model()
        if not row:
            raise ProviderUnavailable("No trained model is registered. Train one (python -m app.ml.train) and register it (python -m app.ml.register).")
        with self.lock:
            key = (row["id"], row["sha256"])
            if key not in self.cache: self.cache = {key: MLDetectionProvider(row)}
            return self.cache[key]

    def status(self):
        try: p = self.get(); return {"available": True, **p.meta}
        except ProviderUnavailable as e: return {"available": False, "reason": str(e)}
