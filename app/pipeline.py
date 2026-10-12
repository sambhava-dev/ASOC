"""The analysis pipeline. Pure functions: no web framework, no database. numpy only (the provider brings torch)."""
import base64, time
import numpy as np
from . import audio, features
from .dsa import PriorityAlerts, Trie, kmp_search, merge_sort, sliding_windows, window_at

STAGES = ["AUDIO RECEIVED", "NORMALIZATION", "SEGMENTATION", "FEATURE EXTRACTION", "MODEL INFERENCE",
          "TEMPORAL AGGREGATION", "CONTEXT ANALYSIS", "RESULT"]
DEFAULT_KEYWORDS = ["otp", "password", "pin", "urgent", "verification", "bank", "transfer", "gift card", "wire", "account number"]


def label_of(p, lo, hi): return "synthetic" if p >= hi else "human" if p <= lo else "uncertain"
def clock(t): return f"{int(t // 60):02d}:{int(t % 60):02d}"


def make_window(i, a, b, x, out, cfg, mask=None, sps=audio.TARGET_SR):
    f = features.window_features(x, mask); no_speech = f["speech_ratio"] < .1; p = float(out.p_synthetic)
    w = {"index": i, "start": round(a / sps, 3), "end": round(b / sps, 3), "clock": clock(a / sps), "p_synthetic": round(p, 4),
         "p_human": round(1 - p, 4), "speech_ratio": round(f["speech_ratio"], 3), "no_speech": no_speech, "features": f,
         "label": "uncertain" if no_speech else label_of(p, cfg["uncertain_low"], cfg["uncertain_high"]),
         "confidence": 0.0 if no_speech else round(abs(p - .5) * 2, 4)}
    if out.source_probs: w["source_probs"] = {k: round(v, 4) for k, v in out.source_probs.items()}
    return w


def aggregate(windows, cfg):
    """Temporal aggregation: smooth neighbouring window scores, weight by speech activity, cap confidence by evidence."""
    if not windows: return None
    p = np.array([w["p_synthetic"] for w in windows]); wt = np.array([w["speech_ratio"] if not w["no_speech"] else 0 for w in windows])
    sm = np.convolve(np.pad(p, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    pa = float((sm * wt).sum() / wt.sum()) if wt.sum() > 1e-9 else .5
    speech_s = float(sum(w["speech_ratio"] * (w["end"] - w["start"]) for w in windows if not w["no_speech"]))
    evidence = min(1.0, speech_s / 5.0); conf = round(abs(pa - .5) * 2 * evidence, 4)
    dist = {k: sum(w["label"] == k for w in windows) for k in ("human", "uncertain", "synthetic")}
    lab = "uncertain" if conf < .2 else label_of(pa, cfg["uncertain_low"], cfg["uncertain_high"])
    cav = ["Scores are probabilities, not proof."]
    if evidence < 1: cav.append("Less than 5 s of detected speech: confidence is capped.")
    if dist["uncertain"] == len(windows): cav.append("No window contained enough speech to score.")
    return {"label": lab, "p_synthetic": round(pa, 4), "p_human": round(1 - pa, 4), "confidence": conf,
            "risk_score": round(100 * (.6 * pa + .4 * dist["synthetic"] / len(windows))), "window_distribution": dist,
            "speech_seconds": round(speech_s, 2), "caveats": cav}


def build_alerts(windows):
    """Priority queue (binary heap): most severe, then most confident, then earliest."""
    q = PriorityAlerts()
    for w in windows:
        if w["no_speech"]: continue
        if w["label"] == "synthetic": pr, sig = (3 if w["p_synthetic"] >= .85 else 2), "Synthetic-speech indicators"
        elif w["label"] == "uncertain": pr, sig = 1, "Ambiguous window"
        else: continue
        q.push(pr, w["confidence"], w["start"], {"signal": sig, "priority": pr, "timestamp": w["start"], "clock": w["clock"],
                                                "confidence": w["confidence"], "window": w["index"]})
    return q.drain()


def context(segments, keywords=None):
    """Keyword indicators over a timestamped transcript (supporting evidence only). Words -> Trie, phrases -> KMP."""
    kws = [k.strip().lower() for k in (keywords or DEFAULT_KEYWORDS) if k.strip()]
    trie = Trie([k for k in kws if " " not in k]); phrases = [k for k in kws if " " in k]; hits = []
    for s in segments:
        t = s["text"]
        for a, b, w in trie.find_all(t): hits.append({"keyword": w, "t": s["start"], "context": t[max(0, a - 30):b + 30]})
        for ph in phrases:
            for a in kmp_search(t, ph): hits.append({"keyword": ph, "t": s["start"], "context": t[max(0, a - 30):a + len(ph) + 30]})
    return merge_sort(hits, key=lambda h: h["t"])


def visual(x):
    """Waveform peaks + log-spectrogram, so the result pages work even when raw audio is not retained."""
    n = 1000; st = max(1, len(x) // n); pk = [[round(float(c.min()), 3), round(float(c.max()), 3)] for c in (x[i * st:(i + 1) * st] for i in range(n)) if len(c)]
    hop = max(160, len(x) // 300); xx = np.pad(x, (0, max(0, 512 - len(x))))
    f = xx[np.arange(512)[None, :] + hop * np.arange(1 + (len(xx) - 512) // hop)[:, None]] * np.hanning(512)
    S = 20 * np.log10(np.abs(np.fft.rfft(f, axis=1)) + 1e-6)[:, :256].T[::-1]            # 256 bins to 8 kHz, low frequency at the bottom
    S = np.clip((S - (S.max() - 80)) / 80, 0, 1); S = S.reshape(128, 2, S.shape[1]).mean(1)
    return {"peaks": pk, "spectrogram": {"freq_bins": 128, "frames": int(S.shape[1]), "max_hz": 8000, "hop_s": hop / audio.TARGET_SR,
                                          "data_b64": base64.b64encode((S * 255).astype(np.uint8).tobytes()).decode()}}


def assemble(windows, duration, cfg, provider_meta):
    summary = aggregate(windows, cfg); alerts = build_alerts(windows)
    events = [{"t": w["start"], "type": "window", "text": f'{w["clock"]} {w["label"]} ({w["p_synthetic"]:.0%} synthetic)'} for w in windows]
    events += [{"t": a["timestamp"], "type": "alert", "text": f'P{a["priority"]} {a["signal"]}'} for a in alerts]
    sources = []
    sp = [(w["source_probs"], w["speech_ratio"]) for w in windows if "source_probs" in w and not w["no_speech"] and w["label"] != "human"]
    if sp:
        tot = sum(wt for _, wt in sp) or 1
        agg = {k: sum(p[k] * wt for p, wt in sp) / tot for k in sp[0][0]}
        sources = sorted([{"name": k, "probability": round(v, 4)} for k, v in agg.items()], key=lambda s: -s["probability"])
    return {"summary": summary, "alerts": alerts, "events": merge_sort(events, key=lambda e: (e["t"], e["type"])), "sources": sources,
            "provider": provider_meta, "duration": round(duration, 3)}


def analyze(data, provider, cfg, transcribe=None, on_stage=None, keywords=None):
    """bytes -> full result dict. `transcribe` is a callable(x16k)->segments or None. on_stage(name, index) reports progress."""
    stages, st = [], [time.perf_counter()]
    def stage(name):
        i = STAGES.index(name); on_stage and on_stage(name, i)
        now = time.perf_counter(); stages.append({"stage": name, "ms": round((now - st[0]) * 1000, 1)}); st[0] = now
    stage("AUDIO RECEIVED"); x, sr = audio.decode(data); duration = len(x) / sr
    xr = audio.to_16k(x, sr); mask = features.speech_mask(xr); x = audio.peak_normalize(xr); stage("NORMALIZATION")
    sps = audio.TARGET_SR; spans = list(sliding_windows(len(x), int(cfg["window_s"] * sps), int(cfg["hop_s"] * sps))); stage("SEGMENTATION")
    wins_audio = [x[a:b] for a, b in spans]; stage("FEATURE EXTRACTION")
    outs = provider.score_windows(wins_audio)
    wmask = lambda a, b: mask[a // features.HOP: max(a // features.HOP + 1, (b - features.FRAME) // features.HOP + 1)]
    windows = [make_window(i, a, b, wa, o, cfg, wmask(a, b)) for i, ((a, b), wa, o) in enumerate(zip(spans, wins_audio, outs))]; stage("MODEL INFERENCE")
    res = assemble(windows, duration, cfg, provider.meta); stage("TEMPORAL AGGREGATION")
    res["windows"] = windows; res["voice_profile"] = features.voice_profile(x, mask); res["visual"] = visual(x)
    lab = res["summary"]["label"]; cand = [w for w in windows if not w["no_speech"]]
    cand = sorted([w for w in cand if w["label"] == lab] or cand, key=lambda w: -w["confidence"])[:3]
    res["evidence"] = [{"window": w["index"], "clock": w["clock"], "p_synthetic": w["p_synthetic"], "confidence": w["confidence"],
                        "band_contribution": provider.explain(wins_audio[w["index"]])} for w in cand]
    if transcribe:
        try: segs = transcribe(x); res["transcript"] = {"status": "ok", "segments": segs}; res["matches"] = context(segs, keywords)
        except Exception as e: res["transcript"] = {"status": "failed", "segments": [], "error": str(e)[:200]}; res["matches"] = []
    else: res["transcript"] = {"status": "unavailable", "segments": []}; res["matches"] = []
    stage("CONTEXT ANALYSIS"); stage("RESULT"); res["stages"] = stages
    return res


def window_lookup(windows, t):
    """Binary search over window start times."""
    i = window_at([w["start"] if isinstance(w, dict) else w for w in windows], t)
    return windows[i] if i >= 0 else None
