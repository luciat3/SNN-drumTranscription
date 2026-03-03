# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import librosa

from models.cnn.dataset import SpecConfig, compute_log_mel

# to avoid recalculating mels for every window, we precompute them for the whole track and save as .npy files

def main():
    index_jsonl = Path("data/processed/index.jsonl")
    out_root = Path("data/processed/mels")
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = SpecConfig(win_seconds=1.0, tol_frames=1)

    n_ok, n_skip = 0, 0

    with index_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            wav_path = Path(row["wav"])
            track_id = row["id"]

            out_path = out_root / f"{track_id}.npy"
            if out_path.exists():
                n_skip += 1
                continue

            y, _ = librosa.load(wav_path, sr=cfg.sr, mono=True)
            X = compute_log_mel(y, cfg)  # (M, T) float32

            np.save(out_path, X)
            n_ok += 1

            if (n_ok % 100) == 0:
                print(f"Done {n_ok} tracks...")

    print(f"Finished. Saved: {n_ok}, skipped: {n_skip}")

if __name__ == "__main__":
    main()
