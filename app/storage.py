"""Retained audio on disk. Paths come only from server-generated ids (never from user filenames)."""
import hashlib, os
from pathlib import Path
from . import config, repo


def save(user_id, sid, data, ext, retention_h):
    d = config.get().audio_dir / user_id; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sid}.{ext}"; tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data); os.chmod(tmp, 0o600); os.replace(tmp, p)
    repo.add_asset(user_id, sid, str(p), hashlib.sha256(data).hexdigest(), len(data), ext, repo.later(hours=retention_h))
    return p


def remove(paths):
    for p in paths:
        try: Path(p).unlink(missing_ok=True)
        except OSError: pass


def purge_expired():
    """Deletes raw audio past its retention time. Derived results (windows, scores) stay until the user deletes them."""
    n = 0
    for a in repo.expired_assets():
        remove([a["path"]]); repo.mark_asset_deleted(a["id"]); n += 1
    return n
