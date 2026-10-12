"""AV-ASOC API server.   uvicorn app.main:app --host 127.0.0.1 --port 8000
Serves the API and (if present) the frontend at /. Interactive docs at /docs."""
import asyncio, base64, logging, os, re, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from . import audio, config, db, features, mailer, pipeline, repo, security, service, storage, stt
from .dsa import StreamBuffer
from .ml.provider import ProviderUnavailable

log = logging.getLogger("avasoc"); cfg = config.get(); LIM = security.RateLimiter()
STATIC = Path(__file__).resolve().parent.parent / "static" / "index.html"
MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "flac": "audio/flac", "ogg": "audio/ogg"}
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; connect-src 'self' ws: wss:; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'")


@asynccontextmanager
async def lifespan(_):
    db.init(); repo.fail_stale_sessions(); stop = threading.Event()
    def purge():
        while not stop.wait(600):
            try: storage.purge_expired()
            except Exception: log.exception("purge failed")
    storage.purge_expired(); threading.Thread(target=purge, daemon=True).start(); yield; stop.set()


app = FastAPI(title="AV-ASOC", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=cfg.cors, allow_methods=["*"], allow_headers=["Authorization", "Content-Type"], allow_credentials=False)


@app.middleware("http")
async def headers(request: Request, call_next):
    r = await call_next(request)
    r.headers["X-Content-Type-Options"] = "nosniff"; r.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path != "/" and not request.url.path.startswith(("/docs", "/openapi")): r.headers["Cache-Control"] = "no-store"
    return r


# ---- helpers -------------------------------------------------------------------------
def rate(request: Request, name, cap, per, who=None):
    ok, retry = LIM.allow(f"{name}:{who or request.client.host}", cap, per)
    if not ok: raise HTTPException(429, "Too many requests. Try again shortly.", headers={"Retry-After": str(retry)})

def user_from_token(tok):
    try: d = security.read_token(tok, cfg.secret)
    except security.AuthError: raise HTTPException(401, "Invalid or expired session.", headers={"WWW-Authenticate": "Bearer"})
    u = repo.user_by_id(d["sub"])
    if not u or not u["is_active"] or u["token_version"] != d["tv"]: raise HTTPException(401, "Session is no longer valid.")
    return u

def current_user(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "): raise HTTPException(401, "Sign in required.", headers={"WWW-Authenticate": "Bearer"})
    return user_from_token(authorization[7:])

def token_for(u): return security.make_token(u["id"], u["token_version"], cfg.secret, cfg.token_ttl_s)
def clean_name(s): return re.sub(r"[\x00-\x1f]", "", os.path.basename(s or "upload"))[:120] or "upload"

def own(u, sid, need_done=True):
    s = repo.get_session(u["id"], sid)
    if not s: raise HTTPException(404, "Analysis not found.")
    if need_done and s["status"] != "done": raise HTTPException(409, f"Analysis is {s['status']}.")
    return s

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
class SignupIn(BaseModel): email: str = Field(max_length=254); password: str; full_name: str = Field(min_length=1, max_length=100)
class LoginIn(BaseModel): email: str; password: str
class ForgotIn(BaseModel): email: str
class ResetIn(BaseModel): token: str; new_password: str
class ChangePwIn(BaseModel): current_password: str; new_password: str
class DeleteAcctIn(BaseModel): password: str
class ProfileIn(BaseModel): full_name: str | None = Field(None, min_length=1, max_length=100); settings: dict | None = None
class SaveIn(BaseModel): session_id: str; title: str = Field(min_length=1, max_length=120); tags: list[str] = []
class SavePatch(BaseModel): title: str | None = Field(None, min_length=1, max_length=120); tags: list[str] | None = None
class NoteIn(BaseModel): text: str = Field(min_length=1, max_length=2000); t: float | None = None
class AlertPatch(BaseModel): status: str
class ContextIn(BaseModel): keywords: list[str] = Field(min_length=1, max_length=50)

RULES = {"retention_hours": (int, 1, 720), "window_s": (float, 1.0, 6.0), "hop_s": (float, .25, 6.0), "uncertain_low": (float, 0, .5), "uncertain_high": (float, .5, 1)}
def valid_settings(s):
    out = {}
    for k, v in s.items():
        if k in RULES:
            t, lo, hi = RULES[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi: raise HTTPException(422, f"{k} must be between {lo} and {hi}.")
            out[k] = t(v)
        elif k in ("store_audio", "transcript", "notifications"):
            if not isinstance(v, bool): raise HTTPException(422, f"{k} must be true or false.")
            out[k] = v
        else: raise HTTPException(422, f"Unknown setting '{k}'.")
    return out


# ---- public --------------------------------------------------------------------------
@app.get("/")
def home():
    if STATIC.exists(): return FileResponse(STATIC, headers={"Content-Security-Policy": CSP})
    return JSONResponse({"service": "AV-ASOC API", "docs": "/docs"})

@app.get("/health")
def health(): return {"status": "ok"}


# ---- accounts ------------------------------------------------------------------------
@app.post("/auth/signup", status_code=201)
def signup(b: SignupIn, request: Request):
    rate(request, "signup", 5, 3600)
    if not cfg.signup_open: raise HTTPException(403, "Sign-up is closed.")
    email = b.email.strip().lower()
    if not EMAIL.match(email): raise HTTPException(422, "Enter a valid email address.")
    try: security.check_password_policy(b.password)
    except ValueError as e: raise HTTPException(422, str(e))
    try: i = repo.create_user(email, security.hash_password(b.password, cfg.pbkdf2_iters), b.full_name.strip())
    except repo.Conflict as e: raise HTTPException(409, str(e))
    u = repo.user_by_id(i); return {"access_token": token_for(u), "profile": repo.get_profile(i)}

_DUMMY = security.hash_password("dummy-password", cfg.pbkdf2_iters)
@app.post("/auth/login")
def login(b: LoginIn, request: Request):
    email = b.email.strip().lower(); rate(request, "login", 10, 300); rate(request, "login-acct", 5, 300, email)
    u = repo.user_by_email(email)
    ok = security.verify_password(b.password, u["password_hash"] if u else _DUMMY)       # same work whether or not the user exists
    if not (u and ok and u["is_active"]): raise HTTPException(401, "Incorrect email or password.")
    return {"access_token": token_for(u), "profile": repo.get_profile(u["id"])}

@app.get("/auth/me")
def me(u=Depends(current_user)): return {**repo.get_profile(u["id"]), "id": u["id"], "stats": repo.stats(u["id"])}

@app.post("/auth/forgot-password", status_code=202)
def forgot(b: ForgotIn, request: Request, bg: BackgroundTasks):
    rate(request, "forgot", 5, 3600); u = repo.user_by_email(b.email.strip().lower())
    if u:
        tok = security.random_token(); repo.save_reset(u["id"], security.sha256_hex(tok), repo.later(minutes=30))
        bg.add_task(mailer.send_reset, u["email"], tok)
    return {"detail": "If that email has an account, a reset link has been sent."}      # same answer either way

@app.post("/auth/reset-password")
def reset(b: ResetIn, request: Request):
    rate(request, "reset", 10, 3600)
    try: security.check_password_policy(b.new_password)
    except ValueError as e: raise HTTPException(422, str(e))
    i = repo.consume_reset(security.sha256_hex(b.token))
    if not i: raise HTTPException(400, "This reset link is invalid or has expired.")
    repo.set_password(i, security.hash_password(b.new_password, cfg.pbkdf2_iters)); return {"detail": "Password updated. Sign in with the new password."}

@app.post("/auth/change-password")
def change_pw(b: ChangePwIn, request: Request, u=Depends(current_user)):
    rate(request, "chpw", 5, 600, u["id"])
    if not security.verify_password(b.current_password, u["password_hash"]): raise HTTPException(403, "Current password is incorrect.")
    try: security.check_password_policy(b.new_password)
    except ValueError as e: raise HTTPException(422, str(e))
    repo.set_password(u["id"], security.hash_password(b.new_password, cfg.pbkdf2_iters)); return {"access_token": token_for(repo.user_by_id(u["id"]))}

@app.delete("/account")
def delete_account(b: DeleteAcctIn, u=Depends(current_user)):
    if not security.verify_password(b.password, u["password_hash"]): raise HTTPException(403, "Password is incorrect.")
    storage.remove(repo.delete_user(u["id"])); return {"detail": "Account and all associated data deleted."}

@app.get("/profile")
def get_profile(u=Depends(current_user)): return {**repo.get_profile(u["id"]), "stats": repo.stats(u["id"])}

@app.patch("/profile")
def patch_profile(b: ProfileIn, u=Depends(current_user)):
    repo.update_profile(u["id"], b.full_name.strip() if b.full_name else None, valid_settings(b.settings) if b.settings else None)
    return repo.get_profile(u["id"])


# ---- analyses ------------------------------------------------------------------------
@app.post("/analyses", status_code=202)
async def create_analysis(request: Request, bg: BackgroundTasks, file: UploadFile = File(...), u=Depends(current_user)):
    rate(request, "analyze", 30, 3600, u["id"])
    data = await file.read(cfg.max_upload_bytes + 1)
    try: ext = security.sniff_audio(data, cfg.max_upload_bytes)
    except security.UploadError as e: raise HTTPException(413 if "large" in str(e) else 415, str(e))
    st = repo.user_settings(u["id"]); sid = repo.create_session(u["id"], "upload", clean_name(file.filename))
    if st["store_audio"]: storage.save(u["id"], sid, data, ext, min(st["retention_hours"], cfg.retention_h * 30))
    bg.add_task(service.process_upload, u["id"], sid, data)
    return {"id": sid, "status": "queued"}

@app.get("/analyses")
def list_analyses(q: str | None = None, label: str | None = Query(None, pattern="^(human|synthetic|uncertain)$"), status: str | None = None,
                  sort: str = "created_at", order: str = "desc", limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), u=Depends(current_user)):
    rows, total = repo.list_sessions(u["id"], q, label, status, sort, order, limit, offset); return {"total": total, "items": rows}

@app.get("/analyses/{sid}")
def get_analysis(sid: str, u=Depends(current_user)):
    s = own(u, sid, False); v = service.session_view(s); v["audio_retained"] = bool(repo.asset_for_session(u["id"], sid)); return v

@app.delete("/analyses/{sid}")
def delete_analysis(sid: str, u=Depends(current_user)):
    paths = repo.delete_session(u["id"], sid)
    if paths is None: raise HTTPException(404, "Analysis not found.")
    storage.remove(paths); return {"detail": "Deleted."}

@app.get("/analyses/{sid}/windows")
def windows(sid: str, u=Depends(current_user)): own(u, sid); return repo.windows(sid)

@app.get("/analyses/{sid}/windows/at")
def window_at(sid: str, t: float = Query(..., ge=0), u=Depends(current_user)):
    own(u, sid); ws = repo.windows(sid); w = pipeline.window_lookup([{"start": x["start_s"], **x} for x in ws], t)
    if not w: raise HTTPException(404, "No window at that time.")
    return w

@app.get("/analyses/{sid}/waveform")
def waveform(sid: str, u=Depends(current_user)): own(u, sid); return {"peaks": repo.features(sid)["visual"]["peaks"]}

@app.get("/analyses/{sid}/spectrogram")
def spectrogram(sid: str, u=Depends(current_user)): own(u, sid); return repo.features(sid)["visual"]["spectrogram"]

@app.get("/analyses/{sid}/audio")
def get_audio(sid: str, u=Depends(current_user)):
    own(u, sid, False); a = repo.asset_for_session(u["id"], sid)
    if not a or not Path(a["path"]).exists(): raise HTTPException(410, "The audio was not kept or its retention period ended. Analysis results remain available.")
    return FileResponse(a["path"], media_type=MIME.get(a["format"], "application/octet-stream"), headers={"Cache-Control": "private, no-store"})

@app.get("/analyses/{sid}/evidence")
def evidence(sid: str, u=Depends(current_user)):
    own(u, sid); mr = repo.model_result(sid)
    return {"provider": mr["provider"], "evidence": mr["evidence"],
            "note": "band_contribution is occlusion sensitivity: the drop in synthetic probability when that frequency band is neutralised. It describes the model, not the speaker."}

@app.get("/analyses/{sid}/voice-profile")
def voice_profile(sid: str, u=Depends(current_user)): own(u, sid); return repo.features(sid)["voice_profile"]

@app.get("/analyses/{sid}/source")
def source(sid: str, u=Depends(current_user)):
    own(u, sid); c = repo.sources(sid); mr = repo.model_result(sid)
    if not c: return {"source": "Unknown", "candidates": [], "reason": "The active model does not classify attack systems, or no window was flagged." if not mr["provider"].get("supports_source") else "No window was flagged as synthetic."}
    top = c[0]; return {"source": top["name"] if top["probability"] >= .5 else "Unknown", "candidates": c,
                        "note": "Closed-set guess among the attack systems the model was trained on. An unseen generator will still be forced into one of them."}

@app.get("/analyses/{sid}/transcript")
def transcript(sid: str, u=Depends(current_user)):
    own(u, sid); mr = repo.model_result(sid); return {"status": mr.get("transcript_status") or "unavailable", "segments": repo.transcript(sid)}

@app.get("/analyses/{sid}/context")
def get_context(sid: str, u=Depends(current_user)):
    own(u, sid); return {"matches": repo.matches(sid), "note": "Keyword matches are supporting indicators, not proof of intent."}

@app.post("/analyses/{sid}/context")
def post_context(sid: str, b: ContextIn, u=Depends(current_user)):
    own(u, sid); segs = [{"start": s["start_s"], "end": s["end_s"], "text": s["text"]} for s in repo.transcript(sid)]
    if not segs: raise HTTPException(409, "This analysis has no transcript.")
    m = pipeline.context(segs, b.keywords); repo.replace_matches(sid, [{"keyword": x["keyword"], "t": x["t"], "context": x["context"]} for x in m])
    return {"matches": repo.matches(sid)}

@app.get("/analyses/{sid}/risk")
def risk(sid: str, u=Depends(current_user)):
    s = own(u, sid); sm = service.session_view(s)["summary"]
    return {**sm, "definitions": {
        "p_synthetic": "Speech-weighted average of calibrated per-window scores, lightly smoothed over neighbouring windows.",
        "p_human": "1 - p_synthetic.", "confidence": "Distance of p_synthetic from 50/50, scaled down when under 5 s of speech was detected.",
        "risk_score": "0-100: 60% p_synthetic + 40% share of windows labelled synthetic. A triage indicator, not a probability of harm."}}

@app.get("/analyses/{sid}/timeline")
def timeline(sid: str, u=Depends(current_user)):
    own(u, sid); ev = repo.events(sid)
    ev += [{"t": s["start_s"], "type": "transcript", "text": s["text"]} for s in repo.transcript(sid)]
    ev += [{"t": m["start_s"], "type": "keyword", "text": m["keyword"]} for m in repo.matches(sid)]
    from .dsa import merge_sort
    return merge_sort(ev, key=lambda e: e["t"])

@app.get("/analyses/{sid}/alerts")
def alerts(sid: str, u=Depends(current_user)): own(u, sid); return repo.alerts(sid)

@app.patch("/alerts/{aid}")
def patch_alert(aid: str, b: AlertPatch, u=Depends(current_user)):
    if b.status not in ("open", "acknowledged", "dismissed"): raise HTTPException(422, "status must be open, acknowledged or dismissed.")
    if not repo.set_alert_status(u["id"], aid, b.status): raise HTTPException(404, "Alert not found.")
    return {"id": aid, "status": b.status}

@app.get("/compare")
def compare(a: str, b: str, u=Depends(current_user)):
    out = []
    for sid in (a, b):
        s = own(u, sid); v = service.session_view(s); f = repo.features(sid)
        out.append({**v, "voice_profile": f["voice_profile"], "peaks": f["visual"]["peaks"], "sources": repo.sources(sid), "evidence": repo.model_result(sid)["evidence"]})
    return out


# ---- saved investigations / notes ------------------------------------------------------
@app.post("/saved", status_code=201)
def save(b: SaveIn, u=Depends(current_user)):
    i = repo.save_investigation(u["id"], b.session_id, b.title.strip(), [t.strip()[:30] for t in b.tags][:20])
    if not i: raise HTTPException(404, "Analysis not found.")
    return {"id": i}
@app.get("/saved")
def saved(u=Depends(current_user)): return repo.list_saved(u["id"])
@app.patch("/saved/{i}")
def patch_saved(i: str, b: SavePatch, u=Depends(current_user)):
    if not repo.update_saved(u["id"], i, b.title.strip() if b.title else None, b.tags): raise HTTPException(404, "Not found.")
    return {"detail": "Updated."}
@app.delete("/saved/{i}")
def del_saved(i: str, u=Depends(current_user)):
    if not repo.delete_saved(u["id"], i): raise HTTPException(404, "Not found.")
    return {"detail": "Deleted."}
@app.post("/saved/{i}/notes", status_code=201)
def add_note(i: str, b: NoteIn, u=Depends(current_user)):
    n = repo.add_note(u["id"], i, b.text.strip(), b.t)
    if not n: raise HTTPException(404, "Not found.")
    return {"id": n}
@app.delete("/notes/{n}")
def del_note(n: str, u=Depends(current_user)):
    if not repo.delete_note(u["id"], n): raise HTTPException(404, "Not found.")
    return {"detail": "Deleted."}


# ---- models / system ---------------------------------------------------------------------
@app.get("/models")
def models(u=Depends(current_user)): return {"registry": repo.list_models(), "active": service.MODELS.status()}

@app.get("/system/status")
def system(u=Depends(current_user)):
    def ok(f):
        try: f(); return "ok"
        except Exception as e: return f"error: {str(e)[:80]}"
    def db_ok():
        with db.conn() as c: c.execute("SELECT 1")
    def disk_ok():
        p = cfg.audio_dir / ".probe"; p.write_text("x"); p.unlink()
    m = service.MODELS.status()
    return {"audio_engine": "ok", "database": ok(db_ok), "storage": ok(disk_ok), "websocket": "ok",
            "speech_to_text": "ok" if stt.available() else "not installed (faster-whisper)",
            "inference_model": "ok" if m["available"] else m["reason"], "model": m}


# ---- live microphone ------------------------------------------------------------------------
@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """1) text JSON {"token": "...", "sample_rate": 48000}   2) binary int16 little-endian mono chunks
    3) server sends {"type":"window",...} per finished window   4) text "stop" -> {"type":"final","id":...}.
    Live audio is analysed in memory and is NOT stored; only window scores are saved."""
    await ws.accept(); uid = sid = None; windows, prov, ccfg, i = [], None, None, 0
    try:
        hello = await asyncio.wait_for(ws.receive_json(), 10)
        try: u = user_from_token(str(hello.get("token", "")))
        except HTTPException: await ws.send_json({"type": "error", "detail": "Sign in required."}); await ws.close(4401); return
        sr = int(hello.get("sample_rate", 0))
        if not 8000 <= sr <= 48000: await ws.send_json({"type": "error", "detail": "sample_rate must be 8000-48000."}); await ws.close(1003); return
        try: prov = await run_in_threadpool(service.MODELS.get)
        except ProviderUnavailable as e: await ws.send_json({"type": "error", "detail": str(e)}); await ws.close(1011); return
        uid = u["id"]
        if not service.live_slot(uid, 1): uid = None; await ws.send_json({"type": "error", "detail": "Too many live sessions open."}); await ws.close(1008); return
        ccfg = service.cfg_for(repo.user_settings(uid)); sid = repo.create_session(uid, "live", "Live microphone")
        repo.update_session(sid, status="running", model_id=prov.meta["id"])
        buf, wl, hl = StreamBuffer(), int(ccfg["window_s"] * sr), int(ccfg["hop_s"] * sr); sps = audio.TARGET_SR
        await ws.send_json({"type": "ready", "id": sid, "model": prov.meta})
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect" or m.get("text") == "stop": break
            if m.get("bytes"):
                buf.push(np.frombuffer(m["bytes"][: len(m["bytes"]) // 2 * 2], "<i2").astype(np.float32) / 32768)
                while len(windows) < 1800 and (x := buf.peek(wl)) is not None:
                    xr = audio.to_16k(x, sr); mk = features.speech_mask(xr); xn = audio.peak_normalize(xr)
                    out = (await run_in_threadpool(prov.score_windows, [xn]))[0]
                    a = int(i * ccfg["hop_s"] * sps); w = pipeline.make_window(i, a, a + int(ccfg["window_s"] * sps), xn, out, ccfg, mk)
                    windows.append(w); buf.advance(hl); i += 1
                    await ws.send_json({"type": "window", "window": w, "running": pipeline.aggregate(windows, ccfg)})
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError): pass
    except Exception: log.exception("live session error")
    finally:
        if uid: service.live_slot(uid, -1)
        if sid:
            try:
                if windows:
                    dur = (len(windows) - 1) * ccfg["hop_s"] + ccfg["window_s"]; r = pipeline.assemble(windows, dur, ccfg, prov.meta)
                    r.update(windows=windows, voice_profile=None, visual=None, transcript={"status": "unavailable", "segments": []}, matches=[], evidence=[], stages=[])
                    repo.save_results(sid, r, prov.meta["id"]); s = r["summary"]
                    repo.update_session(sid, status="done", stage="RESULT", progress=1.0, duration=dur, label=s["label"], p_synthetic=s["p_synthetic"], p_human=s["p_human"],
                                        confidence=s["confidence"], risk_score=s["risk_score"], summary_json=__import__("json").dumps(s), finished_at=repo.now())
                    try: await ws.send_json({"type": "final", "id": sid, "summary": s}); await ws.close()
                    except Exception: pass
                else:
                    repo.update_session(sid, status="failed", error="Stopped before one full window (about 2 s) was captured.", finished_at=repo.now())
                    try: await ws.send_json({"type": "final", "id": sid, "summary": None, "note": "Need at least one full window."}); await ws.close()
                    except Exception: pass
            except Exception: log.exception("could not finalise live session")
