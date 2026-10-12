"""Password hashing, signed tokens, rate limiting and upload validation (stdlib only)."""
import base64, hashlib, hmac, json, secrets, threading, time


class AuthError(Exception): pass
class UploadError(ValueError): pass


# ---- passwords: PBKDF2-HMAC-SHA256, per-user salt, constant-time compare ----------
def hash_password(pw: str, iters: int) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iters)
    return f"pbkdf2_sha256${iters}${salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, dk = stored.split("$")
        new = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(new, bytes.fromhex(dk))
    except Exception:
        return False


def check_password_policy(pw: str):
    if len(pw) < 8: raise ValueError("Password must be at least 8 characters.")
    if len(pw) > 200: raise ValueError("Password is too long.")


# ---- tokens: HS256 JWT. `tv` is the user's token version, bumped on password reset ----
def _b64(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def _unb64(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(sub: str, tv: int, secret: str, ttl: int, now=None) -> str:
    now = int(now if now is not None else time.time())
    j = lambda d: _b64(json.dumps(d, separators=(",", ":")).encode())
    body = f'{j({"alg": "HS256", "typ": "JWT"})}.{j({"sub": sub, "tv": tv, "iat": now, "exp": now + ttl})}'
    return body + "." + _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())


def read_token(tok: str, secret: str, now=None) -> dict:
    try:
        h, p, s = tok.split(".")
        if json.loads(_unb64(h)).get("alg") != "HS256": raise AuthError("bad alg")
        good = _b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(s, good): raise AuthError("bad signature")
        d = json.loads(_unb64(p))
    except AuthError: raise
    except Exception: raise AuthError("malformed token")
    if d["exp"] < (now if now is not None else time.time()): raise AuthError("expired")
    return d


def random_token() -> str: return secrets.token_urlsafe(32)
def sha256_hex(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()


# ---- rate limiting: token bucket per key ------------------------------------------
class RateLimiter:
    def __init__(self): self.b, self.l = {}, threading.Lock()

    def allow(self, key, capacity, per_seconds, now=None):
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now; rate = capacity / per_seconds
        with self.l:
            if len(self.b) > 20000: self.b = {k: v for k, v in self.b.items() if now - v[1] < per_seconds}
            tokens, last = self.b.get(key, (capacity, now)); tokens = min(capacity, tokens + (now - last) * rate)
            if tokens >= 1: self.b[key] = (tokens - 1, now); return True, 0
            self.b[key] = (tokens, now); return False, int((1 - tokens) / rate) + 1


# ---- upload validation: size + magic bytes (never trust the filename or Content-Type) ----
def sniff_audio(data: bytes, max_bytes: int) -> str:
    if len(data) > max_bytes: raise UploadError(f"File too large (limit {max_bytes // 2**20} MB).")
    if len(data) < 64: raise UploadError("File is too small to be audio.")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE": return "wav"
    if data[:4] == b"fLaC": return "flac"
    if data[:4] == b"OggS": return "ogg"
    if data[:3] == b"ID3" or (data[0] == 0xFF and data[1] & 0xE0 == 0xE0): return "mp3"
    if data[4:8] == b"ftyp": return "m4a"
    raise UploadError("Unsupported file type. Use WAV, MP3, M4A, FLAC or OGG.")
