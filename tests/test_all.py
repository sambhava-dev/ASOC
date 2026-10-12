"""No FastAPI or PyTorch needed:  python tests/test_all.py   (or pytest tests)"""
import os, sys, tempfile
os.environ.update(AVASOC_ENV="dev", AVASOC_PBKDF2_ITERS="1000", AVASOC_DATA_DIR=tempfile.mkdtemp(prefix="avasoc-test-"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from app import audio, config, db, pipeline, repo, security, storage
from app.dsa import *
from app.ml.provider import ProviderOutput
from helpers import make_sample

CFG = {"uncertain_low": .35, "uncertain_high": .65, "window_s": 2.0, "hop_s": 1.0}


class StubProvider:
    """Test double: scores by position in the clip (first half human, second half synthetic). Never shipped."""
    meta = {"id": "stub", "name": "stub", "version": "0", "validated": False, "trained_on": None, "metrics": {}, "supports_source": True}
    def __init__(self, total): self.total = total; self.i = 0
    def score_windows(self, wins):
        out = []
        for w in wins:
            late = self.i / max(1, self.total - 1) >= .3; self.i += 1
            out.append(ProviderOutput(.95 if late else .05, {"A07": .8, "A12": .2} if late else None))
        return out
    def explain(self, win): return {"0-283 Hz": .1}


def test_security():
    h = security.hash_password("correct horse", 1000)
    assert security.verify_password("correct horse", h) and not security.verify_password("wrong", h) and not security.verify_password("x", "garbage")
    t = security.make_token("u1", 3, "s" * 40, 60, now=1000)
    assert security.read_token(t, "s" * 40, now=1030)["sub"] == "u1"
    for bad in (t[:-2] + "xx", t + "a", "a.b.c", ""):
        try: security.read_token(bad, "s" * 40, now=1030); assert False, bad
        except security.AuthError: pass
    try: security.read_token(t, "s" * 40, now=2000); assert False
    except security.AuthError: pass
    try: security.read_token(t, "o" * 40, now=1030); assert False
    except security.AuthError: pass
    rl = security.RateLimiter()
    assert [rl.allow("k", 3, 60, now=0)[0] for _ in range(4)] == [True, True, True, False]
    assert rl.allow("k", 3, 60, now=25)[0] and rl.allow("other", 3, 60, now=0)[0]      # refills over time; keys are independent
    wav = audio.encode_wav(make_sample("human", 1))
    assert security.sniff_audio(wav, 10**7) == "wav" and security.sniff_audio(b"fLaC" + b"0" * 100, 10**7) == "flac"
    for bad in (b"MZ" + b"0" * 100, b"short"):
        try: security.sniff_audio(bad, 10**7); assert False
        except security.UploadError: pass
    try: security.sniff_audio(wav, 100); assert False
    except security.UploadError: pass


def test_db_accounts_and_isolation():
    db.init()
    a = repo.create_user("a@x.io", "h", "Ann"); b = repo.create_user("b@x.io", "h", "Bob")
    try: repo.create_user("a@x.io", "h", "Dup"); assert False
    except repo.Conflict: pass
    sa = repo.create_session(a, "upload", "ann.wav"); sb = repo.create_session(b, "upload", "bob.wav")
    assert repo.get_session(a, sa) and repo.get_session(a, sb) is None and repo.get_session(b, sa) is None
    assert repo.delete_session(b, sa) is None and repo.get_session(a, sa)
    rows, total = repo.list_sessions(a); assert total == 1 and rows[0]["id"] == sa
    assert repo.list_sessions(a, q="%")[1] == 0 and repo.list_sessions(a, q="ann")[1] == 1        # LIKE wildcards are escaped
    assert repo.list_sessions(a, sort="id; DROP TABLE users")[1] == 1                               # sort column is whitelisted
    st = repo.save_investigation(a, sa, "case", ["x"]); assert st and repo.save_investigation(b, sa, "steal", []) is None
    assert repo.add_note(b, st, "no") is None and repo.add_note(a, st, "yes", 1.5)
    assert not repo.update_saved(b, st, title="hijack") and not repo.delete_saved(b, st) and len(repo.list_saved(a)) == 1 and repo.list_saved(b) == []
    repo.update_profile(a, settings={"window_s": 3.0}); p = repo.get_profile(a)
    assert p["settings"]["window_s"] == 3.0 and p["settings"]["store_audio"] is True and p["full_name"] == "Ann"
    tok = security.sha256_hex("tok"); repo.save_reset(a, tok, repo.later(minutes=30))
    assert repo.consume_reset(tok) == a and repo.consume_reset(tok) is None
    repo.save_reset(a, tok, "2000-01-01T00:00:00+00:00"); assert repo.consume_reset(tok) is None   # expired
    tv = repo.user_by_id(a)["token_version"]; repo.set_password(a, "new"); assert repo.user_by_id(a)["token_version"] == tv + 1
    repo.delete_user(a); assert repo.user_by_id(a) is None and repo.get_session(a, sa) is None and repo.get_session(b, sb)


def test_pipeline_results_and_persistence():
    x = make_sample("mixed", 12); data = audio.encode_wav(x)
    n = len(list(sliding_windows(len(audio.normalize(x, 16000)), 32000, 16000)))
    stages = []
    r = pipeline.analyze(data, StubProvider(n), CFG, on_stage=lambda name, i: stages.append(name))
    assert stages == pipeline.STAGES and len(r["windows"]) == n and r["summary"]["label"] == "synthetic"
    assert [w["label"] for w in r["windows"]][:2] == ["human", "human"] and r["windows"][-1]["label"] == "synthetic"
    assert [a["priority"] for a in r["alerts"]] == sorted((a["priority"] for a in r["alerts"]), reverse=True)
    assert r["sources"][0]["name"] == "A07" and r["evidence"] and r["evidence"][0]["band_contribution"]
    assert r["voice_profile"]["pitch_hz_mean"] and 100 < r["voice_profile"]["pitch_hz_mean"] < 200 and len(r["visual"]["peaks"]) > 500
    assert pipeline.window_lookup(r["windows"], 7.3)["start"] == 7.0 and r["transcript"]["status"] == "unavailable"
    short = pipeline.analyze(audio.encode_wav(make_sample("human", 1.0)), StubProvider(1), CFG)
    assert short["summary"]["label"] == "uncertain" and any("5 s" in c for c in short["summary"]["caveats"])      # confidence capped
    silent = pipeline.analyze(audio.encode_wav(np.full(32000, 1e-4, np.float32)), StubProvider(1), CFG)
    noise = pipeline.analyze(audio.encode_wav((np.random.default_rng(1).standard_normal(48000) * 1e-4).astype(np.float32)), StubProvider(2), CFG)
    assert all(w["no_speech"] for w in noise["windows"]) and noise["summary"]["label"] == "uncertain"      # room noise, even after it is amplified
    assert silent["summary"]["label"] == "uncertain" and all(w["no_speech"] for w in silent["windows"]) and silent["alerts"] == []
    segs = [{"start": 2, "end": 5, "text": "share the OTP and your account number"}, {"start": 0, "end": 2, "text": "Urgent gift card"}]
    assert [(g["t"], g["keyword"]) for g in pipeline.context(segs)] == [(0, "urgent"), (0, "gift card"), (2, "otp"), (2, "account number")]
    assert pipeline.analyze(data, StubProvider(n), CFG, transcribe=lambda x: segs)["matches"]
    # persistence round trip
    db.init(); u = repo.create_user("p@x.io", "h", "P"); sid = repo.create_session(u, "upload", "mixed.wav")
    repo.save_results(sid, r, "stub"); s = r["summary"]
    repo.update_session(sid, status="done", label=s["label"], confidence=s["confidence"], risk_score=s["risk_score"], duration=r["duration"])
    assert len(repo.windows(sid)) == n and repo.windows(sid)[0]["clock"] == "00:00" and repo.alerts(sid)[0]["priority"] == r["alerts"][0]["priority"]
    assert repo.sources(sid)[0]["name"] == "A07" and repo.features(sid)["visual"]["spectrogram"]["frames"] > 10
    assert repo.model_result(sid)["evidence"] and repo.list_sessions(u, label="synthetic")[1] == 1
    p = storage.save(u, sid, data, "wav", 1); assert p.exists() and oct(p.stat().st_mode)[-3:] == "600" and repo.asset_for_session(u, sid)
    with db.conn() as c: c.execute("UPDATE audio_assets SET expires_at='2000-01-01T00:00:00+00:00'")
    assert storage.purge_expired() == 1 and not p.exists() and repo.asset_for_session(u, sid) is None and len(repo.windows(sid)) == n   # audio purged, results stay


def test_audio_and_dsa():
    y = audio.resample(make_sample("human", 2), 16000, 48000); assert abs(len(y) - 96000) <= 1
    for bad in (b"not audio", b"RIFF" + b"\0" * 100):
        try: audio.decode(bad); assert False
        except audio.AudioError: pass
    assert list(sliding_windows(10, 4, 2)) == [(0, 4), (2, 6), (4, 8), (6, 10)] and list(sliding_windows(3, 4, 2)) == [(0, 3)]
    b = StreamBuffer(); b.push(np.arange(3)); b.push(np.arange(3, 8)); assert list(b.peek(5)) == [0, 1, 2, 3, 4]; b.advance(4); assert list(b.peek(2)) == [4, 5]
    q = PriorityAlerts(); q.push(1, .9, 1, "lo"); q.push(3, .5, 5, "late"); q.push(3, .5, 2, "early"); q.push(3, .8, 9, "sure")
    assert q.drain() == ["sure", "early", "late", "lo"]
    assert [m[2] for m in Trie(["otp", "bank"]).find_all("Send the OTP to my bank, notpad")] == ["otp", "bank"] and kmp_search("abababc ababc", "ababc") == [2, 8]
    assert merge_sort([3, 1, 2, 1]) == [1, 1, 2, 3] and window_at([0, 1, 2, 3], 2.5) == 2 and window_at([0, 1], -1) == -1
    s = ResultStore(max_items=2); s.put("a", 1); s.put("b", 2); s.put("c", 3); assert s.get("a") is None and s.get("c") == 3


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok  ", n)
