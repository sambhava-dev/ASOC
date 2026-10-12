"""SQLite schema and connection helper (stdlib). One short-lived connection per operation, WAL mode."""
import sqlite3
from contextlib import contextmanager
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  token_version INTEGER NOT NULL DEFAULT 0, is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles(user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  full_name TEXT NOT NULL, settings_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS password_resets(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_registry(id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL,
  path TEXT NOT NULL, sha256 TEXT NOT NULL, sample_rate INTEGER NOT NULL, window_s REAL NOT NULL, status TEXT NOT NULL DEFAULT 'inactive',
  validated INTEGER NOT NULL DEFAULT 0, metrics_json TEXT NOT NULL DEFAULT '{}', trained_on TEXT, supports_source INTEGER NOT NULL DEFAULT 0,
  classes_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_sessions(id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, progress REAL NOT NULL DEFAULT 0, filename TEXT, duration REAL, model_id TEXT,
  label TEXT, p_synthetic REAL, p_human REAL, confidence REAL, risk_score INTEGER, summary_json TEXT, error TEXT,
  created_at TEXT NOT NULL, finished_at TEXT);
CREATE INDEX IF NOT EXISTS ix_sess_user ON analysis_sessions(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS audio_assets(id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, path TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
  format TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE IF NOT EXISTS audio_windows(session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, idx INTEGER NOT NULL,
  start_s REAL, end_s REAL, p_synthetic REAL, label TEXT, confidence REAL, speech_ratio REAL, data_json TEXT, PRIMARY KEY(session_id, idx));
CREATE TABLE IF NOT EXISTS feature_results(session_id TEXT PRIMARY KEY REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  voice_profile_json TEXT, visual_json TEXT);
CREATE TABLE IF NOT EXISTS model_results(session_id TEXT PRIMARY KEY REFERENCES analysis_sessions(id) ON DELETE CASCADE, model_id TEXT, result_json TEXT);
CREATE TABLE IF NOT EXISTS source_candidates(session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, name TEXT NOT NULL, probability REAL NOT NULL);
CREATE TABLE IF NOT EXISTS transcript_segments(session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, idx INTEGER NOT NULL,
  start_s REAL, end_s REAL, text TEXT, confidence REAL, PRIMARY KEY(session_id, idx));
CREATE TABLE IF NOT EXISTS transcript_matches(session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, keyword TEXT, start_s REAL, context TEXT);
CREATE TABLE IF NOT EXISTS analysis_events(session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, t REAL, type TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  priority INTEGER, signal TEXT, t REAL, confidence REAL, status TEXT NOT NULL DEFAULT 'open', window_idx INTEGER, rank INTEGER);
CREATE TABLE IF NOT EXISTS saved_investigations(id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, title TEXT NOT NULL, tags_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes(id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE, saved_id TEXT REFERENCES saved_investigations(id) ON DELETE CASCADE,
  t REAL, text TEXT NOT NULL, created_at TEXT NOT NULL);
"""


@contextmanager
def conn():
    c = sqlite3.connect(config.get().db_path, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c; c.commit()
    except Exception:
        c.rollback(); raise
    finally:
        c.close()


def init():
    s = config.get(); s.data_dir.mkdir(parents=True, exist_ok=True); s.audio_dir.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(s.db_path); c.execute("PRAGMA journal_mode=WAL"); c.executescript(SCHEMA); c.close()
