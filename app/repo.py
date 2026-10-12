"""All SQL lives here. Every function that touches user data takes user_id and filters on it
(application-level row access control: a user can never read or change another user's rows)."""
import json, sqlite3, uuid
from datetime import datetime, timedelta, timezone
from .db import conn


class Conflict(Exception): pass

DEFAULT_SETTINGS = {"retention_hours": 24, "store_audio": True, "uncertain_low": 0.35, "uncertain_high": 0.65,
                    "window_s": 2.0, "hop_s": 1.0, "transcript": True, "notifications": False}
uid = lambda: uuid.uuid4().hex
def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def later(hours=0, minutes=0): return (datetime.now(timezone.utc) + timedelta(hours=hours, minutes=minutes)).isoformat(timespec="seconds")
def _one(c, q, a=()): r = c.execute(q, a).fetchone(); return dict(r) if r else None
def _all(c, q, a=()): return [dict(r) for r in c.execute(q, a).fetchall()]
def _j(v): return json.dumps(v, separators=(",", ":"))


# ---- users / profiles ----------------------------------------------------------------
def create_user(email, pw_hash, full_name):
    i = uid()
    try:
        with conn() as c:
            c.execute("INSERT INTO users(id,email,password_hash,created_at) VALUES(?,?,?,?)", (i, email, pw_hash, now()))
            c.execute("INSERT INTO profiles(user_id,full_name,settings_json,created_at) VALUES(?,?,?,?)", (i, full_name, "{}", now()))
    except sqlite3.IntegrityError:
        raise Conflict("An account with this email already exists.")
    return i

def user_by_email(email):
    with conn() as c: return _one(c, "SELECT * FROM users WHERE email=?", (email,))
def user_by_id(i):
    with conn() as c: return _one(c, "SELECT * FROM users WHERE id=?", (i,))
def set_password(i, pw_hash):
    with conn() as c:
        c.execute("UPDATE users SET password_hash=?, token_version=token_version+1 WHERE id=?", (pw_hash, i))
        c.execute("DELETE FROM password_resets WHERE user_id=?", (i,))
def save_reset(i, token_hash, expires):
    with conn() as c:
        c.execute("DELETE FROM password_resets WHERE user_id=?", (i,))
        c.execute("INSERT INTO password_resets VALUES(?,?,?)", (token_hash, i, expires))
def consume_reset(token_hash):
    with conn() as c:
        r = _one(c, "SELECT * FROM password_resets WHERE token_hash=?", (token_hash,))
        if not r: return None
        c.execute("DELETE FROM password_resets WHERE token_hash=?", (token_hash,))
        return r["user_id"] if r["expires_at"] > now() else None

def get_profile(i):
    with conn() as c:
        r = _one(c, "SELECT u.email,u.created_at,p.full_name,p.settings_json FROM users u JOIN profiles p ON p.user_id=u.id WHERE u.id=?", (i,))
    if not r: return None
    r["settings"] = {**DEFAULT_SETTINGS, **json.loads(r.pop("settings_json"))}; return r
def user_settings(i): return get_profile(i)["settings"]
def update_profile(i, full_name=None, settings=None):
    with conn() as c:
        if full_name is not None: c.execute("UPDATE profiles SET full_name=? WHERE user_id=?", (full_name, i))
        if settings is not None:
            cur = json.loads(c.execute("SELECT settings_json FROM profiles WHERE user_id=?", (i,)).fetchone()[0])
            c.execute("UPDATE profiles SET settings_json=? WHERE user_id=?", (_j({**cur, **settings}), i))
def delete_user(i):
    """Returns audio paths to delete from disk. Cascades remove every row owned by the user."""
    with conn() as c:
        paths = [r["path"] for r in _all(c, "SELECT path FROM audio_assets WHERE user_id=? AND deleted_at IS NULL", (i,))]
        c.execute("DELETE FROM users WHERE id=?", (i,))
    return paths


# ---- sessions ------------------------------------------------------------------------
SESSION_COLS = {"status", "stage", "progress", "duration", "model_id", "label", "p_synthetic", "p_human", "confidence",
                "risk_score", "summary_json", "error", "finished_at"}

def create_session(user_id, kind, filename):
    i = uid()
    with conn() as c:
        c.execute("INSERT INTO analysis_sessions(id,user_id,kind,status,filename,created_at) VALUES(?,?,?,?,?,?)",
                  (i, user_id, kind, "queued", filename, now()))
    return i
def update_session(sid, **f):
    f = {k: v for k, v in f.items() if k in SESSION_COLS}
    if not f: return
    with conn() as c: c.execute(f"UPDATE analysis_sessions SET {','.join(k + '=?' for k in f)} WHERE id=?", (*f.values(), sid))
def get_session(user_id, sid):
    with conn() as c: return _one(c, "SELECT * FROM analysis_sessions WHERE id=? AND user_id=?", (sid, user_id))
def list_sessions(user_id, q=None, label=None, status=None, sort="created_at", order="desc", limit=50, offset=0):
    sort = sort if sort in ("created_at", "duration", "confidence", "risk_score", "filename") else "created_at"
    order = "ASC" if order == "asc" else "DESC"
    w, a = ["user_id=?"], [user_id]
    if q: w.append("filename LIKE ? ESCAPE '\\'"); a.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    if label: w.append("label=?"); a.append(label)
    if status: w.append("status=?"); a.append(status)
    with conn() as c:
        rows = _all(c, f"SELECT id,kind,status,filename,duration,label,p_synthetic,confidence,risk_score,model_id,created_at,finished_at "
                       f"FROM analysis_sessions WHERE {' AND '.join(w)} ORDER BY {sort} {order} LIMIT ? OFFSET ?", (*a, limit, offset))
        total = c.execute(f"SELECT COUNT(*) FROM analysis_sessions WHERE {' AND '.join(w)}", a).fetchone()[0]
    return rows, total
def delete_session(user_id, sid):
    with conn() as c:
        if not _one(c, "SELECT id FROM analysis_sessions WHERE id=? AND user_id=?", (sid, user_id)): return None
        paths = [r["path"] for r in _all(c, "SELECT path FROM audio_assets WHERE session_id=? AND deleted_at IS NULL", (sid,))]
        c.execute("DELETE FROM analysis_sessions WHERE id=?", (sid,))
    return paths
def fail_stale_sessions():
    """Jobs that were running when the server stopped can never finish."""
    with conn() as c:
        c.execute("UPDATE analysis_sessions SET status='failed', error='Server restarted during analysis.', finished_at=? "
                  "WHERE status IN ('queued','running') AND kind='upload'", (now(),))


# ---- results -------------------------------------------------------------------------
def save_results(sid, r, model_id):
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO audio_windows VALUES(?,?,?,?,?,?,?,?,?)",
            [(sid, w["index"], w["start"], w["end"], w["p_synthetic"], w["label"], w["confidence"], w["speech_ratio"],
              _j({k: w[k] for k in ("p_human", "clock", "no_speech", "features", "source_probs") if k in w})) for w in r["windows"]])
        c.executemany("INSERT INTO alerts VALUES(?,?,?,?,?,?,?,?,?)",
            [(uid(), sid, a["priority"], a["signal"], a["timestamp"], a["confidence"], "open", a["window"], k) for k, a in enumerate(r["alerts"])])
        c.executemany("INSERT INTO analysis_events VALUES(?,?,?,?)", [(sid, e["t"], e["type"], e["text"]) for e in r["events"]])
        c.executemany("INSERT INTO source_candidates VALUES(?,?,?)", [(sid, s["name"], s["probability"]) for s in r.get("sources", [])])
        tr = r.get("transcript", {}).get("segments", [])
        c.executemany("INSERT INTO transcript_segments VALUES(?,?,?,?,?,?)", [(sid, i, s["start"], s["end"], s["text"], s.get("confidence")) for i, s in enumerate(tr)])
        c.executemany("INSERT INTO transcript_matches VALUES(?,?,?,?)", [(sid, m["keyword"], m["t"], m["context"]) for m in r.get("matches", [])])
        c.execute("INSERT OR REPLACE INTO feature_results VALUES(?,?,?)", (sid, _j(r.get("voice_profile")), _j(r.get("visual"))))
        c.execute("INSERT OR REPLACE INTO model_results VALUES(?,?,?)", (sid, model_id, _j({"provider": r["provider"], "stages": r.get("stages", []),
                  "evidence": r.get("evidence", []), "transcript_status": r.get("transcript", {}).get("status")})))
def windows(sid):
    with conn() as c: rows = _all(c, "SELECT * FROM audio_windows WHERE session_id=? ORDER BY idx", (sid,))
    for w in rows: w.update(json.loads(w.pop("data_json") or "{}"))
    return rows
def alerts(sid):
    with conn() as c: return _all(c, "SELECT id,priority,signal,t,confidence,status,window_idx FROM alerts WHERE session_id=? ORDER BY rank", (sid,))
def set_alert_status(user_id, alert_id, status):
    with conn() as c:
        cur = c.execute("UPDATE alerts SET status=? WHERE id=? AND session_id IN (SELECT id FROM analysis_sessions WHERE user_id=?)", (status, alert_id, user_id))
        return cur.rowcount > 0
def events(sid):
    with conn() as c: return _all(c, "SELECT t,type,text FROM analysis_events WHERE session_id=? ORDER BY t, type", (sid,))
def transcript(sid):
    with conn() as c: return _all(c, "SELECT idx,start_s,end_s,text,confidence FROM transcript_segments WHERE session_id=? ORDER BY idx", (sid,))
def matches(sid):
    with conn() as c: return _all(c, "SELECT keyword,start_s,context FROM transcript_matches WHERE session_id=? ORDER BY start_s", (sid,))
def sources(sid):
    with conn() as c: return _all(c, "SELECT name,probability FROM source_candidates WHERE session_id=? ORDER BY probability DESC", (sid,))
def features(sid):
    with conn() as c: r = _one(c, "SELECT * FROM feature_results WHERE session_id=?", (sid,))
    return {"voice_profile": json.loads(r["voice_profile_json"]) if r else None, "visual": json.loads(r["visual_json"]) if r else None}
def model_result(sid):
    with conn() as c: r = _one(c, "SELECT * FROM model_results WHERE session_id=?", (sid,))
    return json.loads(r["result_json"]) if r else None
def replace_matches(sid, ms):
    with conn() as c:
        c.execute("DELETE FROM transcript_matches WHERE session_id=?", (sid,))
        c.executemany("INSERT INTO transcript_matches VALUES(?,?,?,?)", [(sid, m["keyword"], m["t"], m["context"]) for m in ms])


# ---- audio assets (retention) ----------------------------------------------------------
def add_asset(user_id, sid, path, sha, nbytes, fmt, expires):
    with conn() as c: c.execute("INSERT INTO audio_assets VALUES(?,?,?,?,?,?,?,?,?,NULL)", (uid(), user_id, sid, path, sha, nbytes, fmt, now(), expires))
def asset_for_session(user_id, sid):
    with conn() as c: return _one(c, "SELECT * FROM audio_assets WHERE session_id=? AND user_id=? AND deleted_at IS NULL", (sid, user_id))
def expired_assets():
    with conn() as c: return _all(c, "SELECT id,path FROM audio_assets WHERE deleted_at IS NULL AND expires_at<?", (now(),))
def mark_asset_deleted(aid):
    with conn() as c: c.execute("UPDATE audio_assets SET deleted_at=? WHERE id=?", (now(), aid))


# ---- saved investigations / notes ------------------------------------------------------
def save_investigation(user_id, sid, title, tags):
    if not get_session(user_id, sid): return None
    i = uid()
    with conn() as c: c.execute("INSERT INTO saved_investigations VALUES(?,?,?,?,?,?,?)", (i, user_id, sid, title, _j(tags), now(), now()))
    return i
def list_saved(user_id):
    with conn() as c:
        rows = _all(c, "SELECT s.*, a.filename, a.label, a.confidence, a.duration FROM saved_investigations s JOIN analysis_sessions a ON a.id=s.session_id "
                       "WHERE s.user_id=? ORDER BY s.updated_at DESC", (user_id,))
        for r in rows:
            r["tags"] = json.loads(r.pop("tags_json")); r["notes"] = _all(c, "SELECT id,t,text,created_at FROM notes WHERE saved_id=? AND user_id=? ORDER BY created_at", (r["id"], user_id))
    return rows
def update_saved(user_id, i, title=None, tags=None):
    with conn() as c:
        if not _one(c, "SELECT id FROM saved_investigations WHERE id=? AND user_id=?", (i, user_id)): return False
        if title is not None: c.execute("UPDATE saved_investigations SET title=? WHERE id=?", (title, i))
        if tags is not None: c.execute("UPDATE saved_investigations SET tags_json=? WHERE id=?", (_j(tags), i))
        c.execute("UPDATE saved_investigations SET updated_at=? WHERE id=?", (now(), i)); return True
def delete_saved(user_id, i):
    with conn() as c: return c.execute("DELETE FROM saved_investigations WHERE id=? AND user_id=?", (i, user_id)).rowcount > 0
def add_note(user_id, saved_id, text, t=None):
    with conn() as c:
        s = _one(c, "SELECT session_id FROM saved_investigations WHERE id=? AND user_id=?", (saved_id, user_id))
        if not s: return None
        i = uid(); c.execute("INSERT INTO notes VALUES(?,?,?,?,?,?,?)", (i, user_id, s["session_id"], saved_id, t, text, now())); return i
def delete_note(user_id, note_id):
    with conn() as c: return c.execute("DELETE FROM notes WHERE id=? AND user_id=?", (note_id, user_id)).rowcount > 0
def stats(user_id):
    with conn() as c:
        q = lambda s: c.execute(s, (user_id,)).fetchone()[0]
        return {"analyses": q("SELECT COUNT(*) FROM analysis_sessions WHERE user_id=? AND kind='upload'"),
                "live_sessions": q("SELECT COUNT(*) FROM analysis_sessions WHERE user_id=? AND kind='live'"),
                "saved": q("SELECT COUNT(*) FROM saved_investigations WHERE user_id=?")}


# ---- model registry ---------------------------------------------------------------------
def register_model(m, activate):
    i = uid()
    with conn() as c:
        if activate: c.execute("UPDATE model_registry SET status='inactive'")
        c.execute("INSERT INTO model_registry VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i, m["name"], m["version"], "pytorch", m["path"], m["sha256"], m["sample_rate"], m["window_s"],
                   "active" if activate else "inactive", int(m["validated"]), _j(m["metrics"]), m.get("trained_on"),
                   int(len(m["classes"]) > 2), _j(m["classes"]), now()))
    return i
def active_model():
    with conn() as c: return _one(c, "SELECT * FROM model_registry WHERE status='active' ORDER BY created_at DESC LIMIT 1")
def list_models():
    with conn() as c: rows = _all(c, "SELECT id,name,version,kind,sample_rate,window_s,status,validated,metrics_json,trained_on,supports_source,classes_json,created_at FROM model_registry ORDER BY created_at DESC")
    for r in rows: r["metrics"] = json.loads(r.pop("metrics_json")); r["classes"] = json.loads(r.pop("classes_json"))
    return rows
def get_model(i):
    with conn() as c: return _one(c, "SELECT * FROM model_registry WHERE id=?", (i,))
