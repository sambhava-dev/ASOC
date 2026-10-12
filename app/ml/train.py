"""Train the anti-spoofing model.

  python -m app.ml.train --data ./corpus --out models/spoofnet-v1.pt --epochs 30
    corpus/bonafide/**/*.wav|flac           genuine speech
    corpus/spoof/**/*.wav|flac              synthetic / converted / replayed speech
    (optional) corpus/spoof/<SYSTEM>/...    with --multiclass, sub-folder names become source classes

  ASVspoof 2019 LA style protocol:
  python -m app.ml.train --protocol ASVspoof2019.LA.cm.train.trn.txt --audio-dir LA/ASVspoof2019_LA_train/flac --out ...

Metrics are computed on a held-out split of the SAME corpus. They do not prove the model generalises to unseen
generators, codecs or languages: evaluate on a separate dataset before relying on it."""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ..audio import resample
from .melspec import SR, fit_length, logmel
from .model import SpoofNet


def read(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True); return resample(x.mean(1), sr)


def index_folder(root, multiclass):
    root, items = Path(root), []; classes = ["bonafide", "spoof"]
    for p in sorted((root / "bonafide").rglob("*")):
        if p.suffix.lower() in (".wav", ".flac"): items.append((p, 0))
    if multiclass:
        classes = ["bonafide"] + sorted(d.name for d in (root / "spoof").iterdir() if d.is_dir())
    for p in sorted((root / "spoof").rglob("*")):
        if p.suffix.lower() in (".wav", ".flac"): items.append((p, classes.index(p.relative_to(root / "spoof").parts[0]) if multiclass else 1))
    return items, classes


def index_protocol(proto, audio_dir, multiclass):
    rows = [l.split() for l in open(proto) if l.strip()]            # SPEAKER FILE - SYSTEM KEY
    systems = sorted({r[3] for r in rows if r[4] != "bonafide"}); classes = ["bonafide"] + (systems if multiclass else ["spoof"])
    return [(Path(audio_dir) / f"{r[1]}.flac", 0 if r[4] == "bonafide" else (classes.index(r[3]) if multiclass else 1)) for r in rows], classes


class DS(Dataset):
    def __init__(s, items, n, train): s.items, s.n, s.train = items, n, train
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i]; x = read(p)
        if len(x) > s.n:
            a = random.randint(0, len(x) - s.n) if s.train else (len(x) - s.n) // 2; x = x[a:a + s.n]
        x = fit_length(x, s.n)
        if s.train:
            x = x * 10 ** (random.uniform(-6, 6) / 20)
            if random.random() < .5:
                snr = random.uniform(10, 40); x = x + np.random.randn(len(x)).astype(np.float32) * np.sqrt((x ** 2).mean() / 10 ** (snr / 10))
        return torch.from_numpy(logmel(x.astype(np.float32)))[None], y


def eer(p_spoof, y_spoof):
    """Equal error rate and its threshold. y_spoof: 1 = spoof."""
    o = np.argsort(-p_spoof); y = np.asarray(y_spoof)[o]; P, N = y.sum(), (1 - y).sum()
    tp = np.cumsum(y); fp = np.cumsum(1 - y); frr = 1 - tp / max(P, 1); far = fp / max(N, 1)
    i = int(np.argmin(np.abs(frr - far))); return float((frr[i] + far[i]) / 2), float(p_spoof[o][i])


def fit_temperature(logits, y):
    best, bt = 1e9, 1.0
    for t in np.linspace(.5, 5, 46):
        nll = F.cross_entropy(logits / t, y).item()
        if nll < best: best, bt = nll, float(t)
    return bt


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data"); a.add_argument("--protocol"); a.add_argument("--audio-dir"); a.add_argument("--out", required=True)
    a.add_argument("--epochs", type=int, default=30); a.add_argument("--batch", type=int, default=32); a.add_argument("--window-s", type=float, default=2.0)
    a.add_argument("--width", type=int, default=32); a.add_argument("--lr", type=float, default=1e-3); a.add_argument("--workers", type=int, default=2)
    a.add_argument("--multiclass", action="store_true", help="learn which attack system made a spoof (enables source candidates)")
    a.add_argument("--val-frac", type=float, default=.1); a.add_argument("--seed", type=int, default=1)
    a = a.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    items, classes = index_protocol(a.protocol, a.audio_dir, a.multiclass) if a.protocol else index_folder(a.data, a.multiclass)
    if len(items) < 20: raise SystemExit("Too few audio files found.")
    key = lambda it: int(hashlib.md5(str(it[0]).encode()).hexdigest(), 16) % 1000 / 1000          # deterministic split by file
    val = [i for i in items if key(i) < a.val_frac]; tr = [i for i in items if key(i) >= a.val_frac]
    n = int(a.window_s * SR); dev = "cuda" if torch.cuda.is_available() else "cpu"
    cnt = np.bincount([y for _, y in tr], minlength=len(classes)); w = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(classes), dtype=torch.float32, device=dev)
    dl = DataLoader(DS(tr, n, True), a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(DS(val, n, False), a.batch, num_workers=a.workers)
    net = SpoofNet(len(classes), a.width).to(dev); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); best = (9, None, None)
    print(f"{len(tr)} train / {len(val)} val files, classes={classes}, device={dev}")
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); l = F.cross_entropy(net(x), y, weight=w); l.backward(); opt.step(); tot += l.item()
        sched.step(); net.eval(); L, Y = [], []
        with torch.no_grad():
            for x, y in vl: L.append(net(x.to(dev)).cpu()); Y.append(y)
        L, Y = torch.cat(L), torch.cat(Y); ps = 1 - torch.softmax(L, 1)[:, 0].numpy(); e, thr = eer(ps, (Y > 0).numpy().astype(int))
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f} val EER {e:.4f} ({time.time() - t0:.0f}s)")
        if e < best[0]: best = (e, {k: v.cpu().clone() for k, v in net.state_dict().items()}, (L.clone(), Y.clone(), thr))
    e, state, (L, Y, thr) = best; T = fit_temperature(L, Y)
    meta = {"classes": classes, "window_s": a.window_s, "width": a.width, "temperature": T, "sample_rate": SR,
            "trained_on": a.protocol or a.data, "metrics": {"val_eer": round(e, 5), "eer_threshold": round(thr, 4), "n_train": len(tr), "n_val": len(val),
                                                         "eval_note": "held-out split of the training corpus; not an independent benchmark"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": state, "meta": meta}, a.out)
    print("saved", a.out, json.dumps(meta["metrics"]))


if __name__ == "__main__": main()
