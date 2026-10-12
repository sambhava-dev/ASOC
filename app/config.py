"""Settings from environment variables. Secrets are never hard-coded."""
import logging, os, secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("avasoc")


@dataclass(frozen=True)
class Settings:
    env: str; secret: str; data_dir: Path; db_path: Path; audio_dir: Path
    max_upload_bytes: int; token_ttl_s: int; pbkdf2_iters: int; retention_h: int
    cors: list; stt_model: str; signup_open: bool


@lru_cache
def get() -> Settings:
    e = os.environ.get
    env, secret = e("AVASOC_ENV", "production"), e("AVASOC_SECRET", "")
    if len(secret) < 32:
        if env != "dev":
            raise RuntimeError("Set AVASOC_SECRET to 32+ random characters (AVASOC_ENV=dev allows an ephemeral one).")
        secret = secrets.token_urlsafe(48)
        log.warning("Using an ephemeral dev secret: tokens stop working when the server restarts.")
    data = Path(e("AVASOC_DATA_DIR", "./data")).resolve()
    return Settings(env, secret, data, data / "avasoc.db", data / "audio",
                    int(float(e("AVASOC_MAX_UPLOAD_MB", "25")) * 1024 * 1024), int(float(e("AVASOC_TOKEN_TTL_HOURS", "12")) * 3600),
                    int(e("AVASOC_PBKDF2_ITERS", "600000")), int(e("AVASOC_DEFAULT_RETENTION_HOURS", "24")),
                    [o.strip() for o in e("AVASOC_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
                    e("AVASOC_STT_MODEL", "base"), e("AVASOC_SIGNUP_OPEN", "true").lower() in ("1", "true", "yes"))
