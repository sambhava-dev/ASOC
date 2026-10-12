"""Runs analysis jobs and keeps the database in step. Jobs run in background threads (FastAPI BackgroundTasks)."""
import json, logging, threading
from . import pipeline, repo, stt, storage
from .ml.provider import ModelManager, ProviderUnavailable
from .audio import AudioError

log = logging.getLogger("avasoc")
MODELS = ModelManager()
live_counts, live_lock = {}, threading.Lock()


def cfg_for(settings):
    return {k: settings[k] for k in ("uncertain_low", "uncertain_high", "window_s", "hop_s")}


def process_upload(user_id, sid, data):
    """Background job for one uploaded file. Failures are recorded on the session, never raised to the caller."""
    try:
        repo.update_session(sid, status="running", stage=pipeline.STAGES[0], progress=0.0)
        settings = repo.user_settings(user_id)
        prov = MODELS.get()
        repo.update_session(sid, model_id=prov.meta["id"])
        on = lambda name, i: repo.update_session(sid, stage=name, progress=round(i / (len(pipeline.STAGES) - 1), 2))
        want_stt = settings["transcript"] and stt.available()
        res = pipeline.analyze(data, prov, cfg_for(settings), transcribe=stt.transcribe if want_stt else None, on_stage=on)
        repo.save_results(sid, res, prov.meta["id"]); s = res["summary"]
        repo.update_session(sid, status="done", stage="RESULT", progress=1.0, duration=res["duration"], label=s["label"],
                            p_synthetic=s["p_synthetic"], p_human=s["p_human"], confidence=s["confidence"], risk_score=s["risk_score"],
                            summary_json=json.dumps({**s, "speech_seconds": s["speech_seconds"]}), finished_at=repo.now())
    except ProviderUnavailable as e: repo.update_session(sid, status="failed", error=str(e), finished_at=repo.now())
    except AudioError as e: repo.update_session(sid, status="failed", error=str(e), finished_at=repo.now())
    except Exception:
        log.exception("analysis failed for %s", sid)
        repo.update_session(sid, status="failed", error="Analysis failed unexpectedly. See server logs.", finished_at=repo.now())


def session_view(row):
    """Public representation of a session row."""
    v = {k: row[k] for k in ("id", "kind", "status", "stage", "progress", "filename", "duration", "label", "p_synthetic", "p_human",
                              "confidence", "risk_score", "model_id", "error", "created_at", "finished_at")}
    v["summary"] = json.loads(row["summary_json"]) if row.get("summary_json") else None
    mr = repo.model_result(row["id"]) if row["status"] == "done" else None
    if mr: v["provider"] = mr["provider"]; v["stages"] = mr["stages"]; v["transcript_status"] = mr.get("transcript_status")
    return v


def live_slot(user_id, delta, cap=2):
    with live_lock:
        n = live_counts.get(user_id, 0)
        if delta > 0 and n >= cap: return False
        live_counts[user_id] = max(0, n + delta); return True
