"""python -m app.ml.register models/spoofnet-v1.pt --name spoofnet --version 1 [--validated] [--no-activate]
Adds a trained checkpoint to the model registry (with its SHA-256) and makes it the active model.
--validated is only accepted when the checkpoint carries evaluation metrics."""
import argparse, hashlib
from .. import db, repo


def main():
    a = argparse.ArgumentParser(); a.add_argument("path"); a.add_argument("--name", required=True); a.add_argument("--version", required=True)
    a.add_argument("--validated", action="store_true"); a.add_argument("--no-activate", action="store_true"); a = a.parse_args()
    import torch
    ck = torch.load(a.path, map_location="cpu", weights_only=True); m = ck["meta"]
    if a.validated and "val_eer" not in m.get("metrics", {}): raise SystemExit("Checkpoint has no evaluation metrics; cannot mark it validated.")
    db.init()
    i = repo.register_model({"name": a.name, "version": a.version, "path": str(__import__("pathlib").Path(a.path).resolve()),
                             "sha256": hashlib.sha256(open(a.path, "rb").read()).hexdigest(), "sample_rate": m["sample_rate"], "window_s": m["window_s"],
                             "validated": a.validated, "metrics": m.get("metrics", {}), "trained_on": m.get("trained_on"), "classes": m["classes"]}, not a.no_activate)
    print("registered", i, "active" if not a.no_activate else "inactive")


if __name__ == "__main__": main()
