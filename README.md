# AV-ASOC backend

FastAPI server for AI-Driven Deepfake Voice & Audio Anti-Spoofing Operation Centre: accounts, database, background analysis jobs, live WebSocket analysis, model registry, and the web app (served at `/`).

**There is no demo mode.** The detector is a PyTorch model you train and register. Until one is registered, `POST /analyses` accepts the upload and the job fails with a clear message instead of producing a made-up score.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                         # set AVASOC_SECRET (python -c "import secrets;print(secrets.token_urlsafe(48))")
export $(grep -v '^#' .env | sed 's/ *#.*//' | xargs)       # or set the variables your own way; AVASOC_ENV=dev for local play
python tests/test_all.py                                     # self-checks, no server needed
uvicorn app.main:app --host 127.0.0.1 --port 8000            # open http://127.0.0.1:8000
```
Behind a reverse proxy add `--proxy-headers --forwarded-allow-ips=<proxy ip>` so rate limits see real client IPs, and terminate HTTPS at the proxy. Run **one** worker: rate limits and live-session counts are in process memory.

## Get a working detector
1. Get labelled data: genuine speech and spoofed speech (for example the ASVspoof 2019 LA corpus, WaveFake, or In-the-Wild). Lay it out as `corpus/bonafide/**.wav|flac` and `corpus/spoof/**.wav|flac`, or pass an ASVspoof-style protocol file.
2. Train: `python -m app.ml.train --data corpus --out models/spoofnet-v1.pt --epochs 30` (add `--multiclass` with `corpus/spoof/<SYSTEM>/…` folders to also learn source attribution).
3. Register: `python -m app.ml.register models/spoofnet-v1.pt --name spoofnet --version 1 --validated`
4. Check: `GET /system/status` shows the model, then `python scripts/smoke_client.py http://127.0.0.1:8000 some.wav`.

Be honest with yourself about accuracy: the EER printed by training comes from a held-out split of the same corpus. It does not show the model works on unseen generators, phone codecs, other languages or noisy rooms. Evaluate on a different dataset before trusting results. The registry stores `validated` and the metrics, and the UI shows them.

## What it does
`decode + validate (magic bytes, size) -> 16 kHz mono -> clip-level speech activity -> 2 s windows / 1 s hop (sliding window) -> log-mel -> model (calibrated by temperature scaling) -> speech-weighted temporal aggregation -> confidence, risk -> alerts (heap), timeline (merge sort), evidence (frequency-band occlusion), voice profile, source candidates, transcript + keyword context (Trie + KMP)`. Data structures live in `app/dsa.py`.

## API (full interactive docs at /docs)
| Area | Endpoints |
|---|---|
| Accounts | `POST /auth/signup` `/auth/login` `/auth/forgot-password` `/auth/reset-password` `/auth/change-password`, `GET /auth/me`, `DELETE /account` |
| Profile | `GET/PATCH /profile` (name, settings: retention_hours, store_audio, window_s, hop_s, uncertain_low/high, transcript) |
| Analyses | `POST /analyses` (multipart `file`) -> 202; `GET /analyses` (q, label, status, sort, order, limit, offset); `GET/DELETE /analyses/{id}` |
| Result parts | `/windows` `/windows/at?t=` `/waveform` `/spectrogram` `/audio` `/evidence` `/voice-profile` `/source` `/transcript` `/context` (GET, POST keywords) `/risk` `/timeline` `/alerts`; `PATCH /alerts/{id}`; `GET /compare?a=&b=` |
| Saved | `POST/GET /saved`, `PATCH/DELETE /saved/{id}`, `POST /saved/{id}/notes`, `DELETE /notes/{id}` |
| System | `GET /models`, `GET /system/status`, `GET /health`, `WS /ws/live` |

Live protocol: send `{"token": "...", "sample_rate": 48000}`, then binary int16 mono chunks; receive `window` messages; send text `stop` for `final`. Live audio is never stored.

## Security and privacy
PBKDF2-SHA256 passwords (600k iterations), signed expiring tokens with a version that is bumped on password change/reset, login timing equalised, generic errors, rate limits on auth/upload, upload sniffing by magic bytes (filenames never touch paths), parameterised SQL with whitelisted sort columns, every query scoped by `user_id`, model files verified by SHA-256 and loaded with `weights_only`, CSP + `nosniff` headers, secrets only from the environment. Raw audio is kept only if `store_audio` is on, for `retention_hours` (default 24, purged every 10 min); derived scores, waveform peaks and the spectrogram stay until the user deletes the analysis. Deleting an account removes all of it. The browser keeps the session token in `localStorage`; the CSP limits script sources, but any XSS would expose it.

## What has and has not been verified
Verified by running code: password/token/rate-limit/upload checks, database schema and per-user isolation, retention purge, the full pipeline (with a test double standing in for the model), speech-activity detection, keyword context, all data structures (`python tests/test_all.py`).
**Not run in the environment this was written in:** the FastAPI routes, WebSocket, PyTorch model/training/provider, faster-whisper, SMTP, and the web app in a browser (JavaScript was only syntax-checked). Expect to fix small things on first run; `scripts/smoke_client.py` is the quickest way to find them.

## Not included
Google sign-in (needs your OAuth credentials), a trained model, a settings/compare/saved/models/system/profile UI (the API for all of them exists; the web app currently has landing, sign-in, analyzer, history and result pages), multi-node scaling (SQLite, in-process jobs and rate limits), refresh tokens. Jobs interrupted by a restart are marked failed.
