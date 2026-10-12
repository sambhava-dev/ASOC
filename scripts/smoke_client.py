"""End-to-end check against a running server (stdlib only):
  python scripts/smoke_client.py http://127.0.0.1:8000 path/to/speech.wav
Creates a throw-away account, uploads the file, waits for the job and prints the result."""
import json, sys, time, uuid, urllib.request, urllib.error

base, path = sys.argv[1].rstrip("/"), sys.argv[2]
def call(method, p, body=None, token=None, raw=None, ctype="application/json"):
    h = {"Content-Type": ctype}
    if token: h["Authorization"] = "Bearer " + token
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(urllib.request.Request(base + p, data, h, method=method)) as r: return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e: sys.exit(f"{method} {p} -> {e.code}: {e.read().decode()[:300]}")

email = f"smoke-{uuid.uuid4().hex[:8]}@example.com"
tok = call("POST", "/auth/signup", {"email": email, "password": "smoke-test-pass", "full_name": "Smoke Test"})["access_token"]
print("signed up", email); wav = open(path, "rb").read(); bd = "----avasoc"
body = (f'--{bd}\r\nContent-Disposition: form-data; name="file"; filename="{path.split("/")[-1]}"\r\nContent-Type: audio/wav\r\n\r\n').encode() + wav + f"\r\n--{bd}--\r\n".encode()
sid = call("POST", "/analyses", token=tok, raw=body, ctype=f"multipart/form-data; boundary={bd}")["id"]
while True:
    v = call("GET", f"/analyses/{sid}", token=tok); print(" ", v["status"], v.get("stage") or "")
    if v["status"] in ("done", "failed"): break
    time.sleep(1)
if v["status"] == "failed": sys.exit("FAILED: " + v["error"])
print(json.dumps(v["summary"], indent=2)); print("windows:", len(call("GET", f"/analyses/{sid}/windows", token=tok)), "alerts:", len(call("GET", f"/analyses/{sid}/alerts", token=tok)))
call("DELETE", f"/analyses/{sid}", token=tok); print("cleaned up analysis")
