###############################################################################
# aux to check spectrogram extraction from index                              #
###############################################################################

import argparse
import json
import random
from pathlib import Path
from typing import Counter

import librosa
import numpy as np

# Sampling Rate: number of samples per second
# Human hearing < 20 kHz
# A SR too high will increase computation time without improving results
SR = 22050
# FFT Size: number of samples per frame
# 22050/1024 ~= 46 ms per frame
N_FFT = 1024
# Hop Length: number of samples between frames
# 256/22050 ~= 11.6 ms between frames
HOP = 256
# Number of Mel bands
# Typical values: 40, 80, 128
# 80 is the standard for music
N_MELS = 80

CLASSES = ["kick", "snare", "hihat_closed", "hihat_open", "tom", "floor_tom", "crash", "ride"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Window size in seconds for checking
WIN_SECONDS = 1.0
WIN_FRAMES = int(round(WIN_SECONDS * SR / HOP))  
TOL = 1  


def log_mel(y, sr):
    """
    Computes log-mel spectrogram from audio signal
    compresses with log(1+x) to avoid issues with log(0)
    to improve learning of temporal patterns
    
    :param y: Audio signal
    :param sr: Sampling rate
    :return: log-mel spectrogram (n_mels, T)
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    return np.log1p(S).astype(np.float32)

def onsets_to_frame_targets(onsets_sec: dict, T: int) -> np.ndarray:
    """
    Converts onset times in seconds to frame-level binary targets (y_frames)

    :param onsets_sec: {class_name: [onset_seconds, ...], ...}
    :param T: number of frames in the spectrogram
    :return: y_frames: (T, C) binary array where C is number of classes
    """
    y = np.zeros((T, len(CLASSES)), dtype=np.float32)
    for cls, times in onsets_sec.items():
        if cls not in CLASS_TO_IDX:
            continue
        ci = CLASS_TO_IDX[cls]
        for t_sec in times:
            frame = int(round(t_sec * SR / HOP))
            for k in range(-TOL, TOL + 1):
                f = frame + k
                if 0 <= f < T:
                    y[f, ci] = 1.0
    return y

def load_index(index_path: Path):
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

"""
def main():
    index_path = Path("data/processed/index.jsonl")
    rows = [json.loads(l) for l in index_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    row = random.choice(rows)

    wav_path = Path(row["wav"])
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    X = log_mel(y, sr)
    print("WAV:", wav_path)
    print("Audio:", y.shape, "sr:", sr, "dur(s):", len(y)/sr)
    print("Log-mel:", X.shape, "dtype:", X.dtype, "min/max:", float(X.min()), float(X.max()))
"""

"""
def main():
    
    #Selects a random track from the index, computes log-mel spectrogram and frame targets,
    #selects a random window, and prints info about active classes at the center frame.
    
    rows = [json.loads(l) for l in Path("data/processed/index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    row = random.choice(rows)

    wav_path = Path(row["wav"])
    y_audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    X = log_mel(y_audio, sr) 
    T = X.shape[1]

    y_frames = onsets_to_frame_targets(row["onsets_sec"], T) 

    start = random.randint(0, max(0, T - WIN_FRAMES - 1))
    Xw = X[:, start:start + WIN_FRAMES] 
    center = start + WIN_FRAMES // 2
    yw = y_frames[center]  

    print("WAV:", wav_path)
    # X: (n_mels, T), y_frames: (T, C)
    # T is the number of frames
    # C is the number of classes
    print("X:", X.shape, "y_frames:", y_frames.shape)
    print("Window:", Xw.shape, "center frame:", center)
    active = [CLASSES[i] for i, v in enumerate(yw) if v > 0.5]
    print("Active classes at center:", active if active else "none")

    pos_counts = y_frames.sum(axis=0)
    print("Positive frames per class (this track):")
    for c, n in zip(CLASSES, pos_counts):
        print(f"  {c:12s} {int(n)}")
"""

def main():
    """
    Checks class distribution in random windows sampled from the dataset index.
    For each sampled window, it checks which classes are active at the center frame.
    It prints statistics about class occurrences and combinations.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/processed/index.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--win_seconds", type=float, default=1.0)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--max_tracks", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    win_frames = int(round(args.win_seconds * SR / HOP))

    rows = load_index(Path(args.index))

    if len(rows) > args.max_tracks:
        rows = rng.sample(rows, args.max_tracks)

    none_count = 0
    class_counts = Counter()
    combo_counts = Counter()

    cache = {}  

    for i in range(args.n):
        row = rng.choice(rows)
        track_id = row["id"]

        if track_id in cache:
            X, y_frames = cache[track_id]
        else:
            y_audio, _ = librosa.load(row["wav"], sr=SR, mono=True)
            X = log_mel(y_audio, SR)
            T = X.shape[1]
            y_frames = onsets_to_frame_targets(row["onsets_sec"], T)
            cache[track_id] = (X, y_frames)

        T = y_frames.shape[0]
        if T <= win_frames:
            start = 0
        else:
            start = rng.randint(0, T - win_frames - 1)

        center = start + win_frames // 2
        yw = y_frames[center]

        active = tuple([CLASSES[j] for j, v in enumerate(yw) if v > 0.5])

        if len(active) == 0:
            none_count += 1
            combo_counts[("none",)] += 1
        else:
            for a in active:
                class_counts[a] += 1
            combo_counts[active] += 1

    pct_none = 100.0 * none_count / args.n
    pct_active = 100.0 - pct_none

    print(f"Analized windows: {args.n}")
    print(f"Win seconds: {args.win_seconds} (frames={win_frames}), tol={args.tol}")
    print(f"Number of analyzed tracks: {len(rows)} (cache size={len(cache)})")
    print()
    print(f"Center active: {pct_active:.2f}% | none: {pct_none:.2f}%")
    print()

    print("Top classes:")
    for cls, c in class_counts.most_common(10):
        print(f"  {cls:12s} {c}")

    print()
    print("Top combinations of classes at the center:")
    for combo, c in combo_counts.most_common(15):
        print(f"  {', '.join(combo):25s} {c}")

if __name__ == "__main__":
    main()