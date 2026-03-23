# -*- coding: utf-8 -*-
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse


@dataclass
class SpecConfig:
    # Sampling Rate: number of samples per second
    # Human hearing < 20 kHz
    # A SR too high will increase computation time without improving results
    sr: int = 22050
    # FFT Size: number of samples per frame
    # 22050/1024 ~= 46 ms per frame
    n_fft: int = 1024
    # Hop Length: number of samples between frames
    # 256/22050 ~= 11.6 ms between frames
    hop_length: int = 256
    # Number of Mel bands
    # Typical values: 40, 80, 128
    # 80 is the standard for music
    n_mels: int = 256
    # window size
    win_seconds: float = 1.0
    # tolerance t, t+1, t-1, t+2, t-2
    tol_frames: int = 1 
    # Labeling strategy:
    # - "center": a window is positive for a class if an onset falls within
    #             [center_sec - radius, center_sec + radius]
    # - "window": a window is positive for a class if an onset falls within
    #             [start_sec, end_sec]
    label_mode: str = "center"


# maped classes 
CLASSES = ["kick", "xstick", "snare", "hihat_closed", "hihat_open", "hihat_pedal", "tom", "floor_tom", "vibraslap", "crash", "ride_bow", "ride_bell", "chinese_cymbal", "splash_cymbal"]
CYMBAL_CLASSES = ["crash", "ride_bow", "ride_bell", "chinese_cymbal", "splash_cymbal"]
HIHAT_CLASSES  = ["hihat_closed", "hihat_open", "hihat_pedal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def _iter_onset_times(times: Iterable) -> Iterable[float]:
    """
    Yield onset times in seconds.

    Supports:
      - list[float] <- old version on index.jsonl
      - list[{"t": float, "vel": int}] <- new version that takes in consideration velocity

    Ignores malformed entries.
    """
    for x in times:
        if isinstance(x, dict):
            t = x.get("t", None)
            if t is None:
                continue
            try:
                yield float(t)
            except Exception:
                continue
        else:
            try:
                yield float(x)
            except Exception:
                continue

def compute_log_mel(y: np.ndarray, cfg: SpecConfig) -> np.ndarray:
    """
    Computes log-mel spectrogram from audio signal
    compresses with log(1+x) to avoid issues with log(0)
    to improve learning of temporal patterns 

    Now used only once before training runs (mel data of every track)
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    return np.log1p(S).astype(np.float32)  # (M, T)

def compute_mel(y: np.ndarray, cfg: SpecConfig) -> np.ndarray:
    """
    Computes mel spectrogram from audio signal
    to improve learning of temporal patterns 

    Now used only once before training runs (mel data of every track)
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    return S.astype(np.float32)  # (M, T)


# To compare results, we can also compute the log spectrogram without mel scaling
def compute_log_spectrogram(y: np.ndarray, cfg) -> np.ndarray:
    D = librosa.stft(
        y=y,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window="hann",
        center=True,
    )
    S = np.abs(D) ** 2
    return np.log1p(S).astype(np.float32)



def onsets_to_frame_targets(onsets_sec: Dict[str, List], T: int, cfg: SpecConfig) -> np.ndarray:
    """
    Converts onset times in seconds to frame-level binary targets (y_frames)

    :param onsets_sec: {class_name: [onset_seconds, ...], ...}
    :param T: number of frames in the spectrogram
    :param cfg: spectrogram configuration
    :return: y_frames: (T, C) binary array where C is number of classes
    """
    y = np.zeros((T, len(CLASSES)), dtype=np.float32)
    for cls, times in onsets_sec.items():
        # ignores classes that aren't maped (shouldn't be any)
        if cls not in CLASS_TO_IDX:
            continue
        ci = CLASS_TO_IDX[cls]
        for t_sec in _iter_onset_times(times):
            # turns seconds to spectrogram frames
            frame = int(round(t_sec * cfg.sr / cfg.hop_length))
            # allows a tolerance to neighbouring frames
            for k in range(-cfg.tol_frames, cfg.tol_frames + 1):
                f = frame + k
                # avoid access outside spectrogram
                if 0 <= f < T:
                    y[f, ci] = 1.0
    return y


def onsets_to_positive_frames(onsets_sec: Dict[str, List], cfg: SpecConfig) -> List[int]:
    """
    Converts onsets (lists of classes where an instrument appears) into frames
    """
    frames = []
    for cls, times in onsets_sec.items():
        # ignores classes that aren't maped (shouldn't be any)
        if cls not in CLASS_TO_IDX:
            continue
        # iterates every second
        for t_sec in _iter_onset_times(times):
            # turns seconds to spectrogram frames
            frames.append(int(round(t_sec * cfg.sr / cfg.hop_length)))
    return sorted(set(frames))


def onsets_to_window_targets(onsets_sec_by_class: Dict[str, List],
                             center_sec: float,
                             radius_sec: float) -> torch.Tensor:
    """
    Build multi-label target for a window centered at center_sec.
    A class is 1 if any onset falls within [center_sec - radius_sec, center_sec + radius_sec].
    """
    y = torch.zeros(len(CLASSES), dtype=torch.float32)
    lo = center_sec - radius_sec
    hi = center_sec + radius_sec

    for cls_name, onsets in onsets_sec_by_class.items():
        if cls_name not in CLASS_TO_IDX:
            continue
        c = CLASS_TO_IDX[cls_name]
        for t in _iter_onset_times(onsets):
            if lo <= t <= hi:
                y[c] = 1.0
                break

    return y

def onsets_to_window_interval_targets(onsets_sec_by_class: Dict[str, List],
                                        start_sec: float,
                                        end_sec: float) -> torch.Tensor:
    """
    Build multi-label target for a window [start_sec, end_sec].
    A class is 1 if any onset falls within the window.
    """
    y = torch.zeros(len(CLASSES), dtype=torch.float32)

    for cls_name, onsets in onsets_sec_by_class.items():
        if cls_name not in CLASS_TO_IDX:
            continue
        c = CLASS_TO_IDX[cls_name]
        for t in _iter_onset_times(onsets):
            if start_sec <= t <= end_sec:
                y[c] = 1.0
                break

    return y


class DrumOnsetWindowDataset(Dataset):
    """
    Class to generate training examples
    """

    def __init__(
        self,
        # tracks information
        index_jsonl: str,
        # splits train-test-val
        ids: List[str],
        cfg: SpecConfig,
        max_windows_per_track: int = 8,
        seed: int = 42,
        p_pos: float = 0.8,
        sampling: str = "random",          
        stride_frames: int = 0,
        debug: bool = False
    ):
        super().__init__()

        self.cfg = cfg
        # random generator for sampling windows
        self.rng = random.Random(seed)
        self.p_pos = float(p_pos)
        self.sampling = sampling
        self.stride_frames = int(stride_frames)
        # set to True to print window and center spectrograms
        self.debug = debug

        rows = []
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("id") in set(ids):
                    rows.append(r)
        self.rows = rows

        if not self.rows:
            raise RuntimeError("No rows found.")
        
        self.win_frames = max(1, int(round(cfg.win_seconds * cfg.sr / cfg.hop_length)))
        self.radius_sec = float(cfg.tol_frames) * (cfg.hop_length / cfg.sr)

        # onsets list
        self.pos_frames = []
        for r in self.rows:
            pf = onsets_to_positive_frames(r.get("onsets_sec", {}), cfg)
            self.pos_frames.append(pf)

        # Per-track per-class onset frames for balanced sampling
        self.pos_frames_by_class = []
        self.class_counts = np.zeros(len(CLASSES), dtype=np.int64)

        for r in self.rows:
            per_class = {c: [] for c in CLASSES}
            for cls, times in r["onsets_sec"].items():
                if cls not in CLASS_TO_IDX:
                    continue
                ci = CLASS_TO_IDX[cls]
                for x in times:
                    t_sec = float(x["t"]) if isinstance(x, dict) else float(x)
                    f = int(round(t_sec * cfg.sr / cfg.hop_length))
                    per_class[cls].append(f)
                    self.class_counts[ci] += 1
            # unique + sorted
            for cls in per_class:
                per_class[cls] = sorted(set(per_class[cls]))
            self.pos_frames_by_class.append(per_class)

        # Sampling weights: inverse frequency (use sqrt to avoid overfocusing rarest)
        counts = self.class_counts.astype(np.float32)
        self.class_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        self.class_weights = self.class_weights / self.class_weights.sum()

        self.samples = []
        if self.sampling == "all":
            stride = self.stride_frames if self.stride_frames > 0 else max(1, self.win_frames // 2)

            for i, r in enumerate(self.rows):
                # if mel exists, prefer its T; otherwise approximate from audio duration
                mel_path = Path(r.get("mel", ""))
                if mel_path.exists():
                    X = np.load(mel_path)
                    T = int(X.shape[1])
                else:
                    dur = float(r.get("duration_sec", 0.0))
                    if dur <= 0.0:
                        dur = librosa.get_duration(path=r["wav"])
                    T = max(1, int(math.ceil(dur * cfg.sr / cfg.hop_length)))

                if T <= self.win_frames:
                    self.samples.append((i, 0))
                else:
                    last = T - self.win_frames
                    for start in range(0, last + 1, stride):
                        self.samples.append((i, start))
        else:
            # random: fixed number of samples per track, start chosen in __getitem__
            for i in range(len(self.rows)):
                for _ in range(max_windows_per_track):
                    self.samples.append((i, -1))  

    # total number of tracks
    def __len__(self) -> int:
        return len(self.samples)

    def _choose_start(self, T: int, track_i: int) -> int:
        """
        Chooses the initial frame of the window

        :param T: Total number of spectrogram frames 
        :param pos_frames: frames with onsets
        """
        if T <= self.win_frames:
            return 0

        # helper clamp
        def clamp_start(s: int) -> int:
            return max(0, min(s, T - self.win_frames))
            
        frame_dur_ms = (self.cfg.hop_length / self.cfg.sr) * 1000.0
        def ms_to_frames(ms: float) -> int:
            return max(1, int(round(ms / frame_dur_ms)))
        
        # decide positive/negative
        want_pos = (self.rng.random() < self.p_pos)

        if want_pos:
            per_class = self.pos_frames_by_class[track_i]

            # pick a class (rare classes get higher probability)
            # but only among classes that actually exist in this track
            available = [cls for cls in CLASSES if len(per_class[cls]) > 0]
            if available:
                avail_idx = [CLASS_TO_IDX[c] for c in available]
                w = self.class_weights[avail_idx]
                w = w / w.sum()
                chosen_cls = self.rng.choices(available, weights=w.tolist(), k=1)[0]

                half = self.win_frames // 2
                # choose an onset frame that can be centered without clamping
                valid = [f for f in per_class[chosen_cls] if half <= f <= (T - half - 1)]
                center = self.rng.choice(valid) if valid else self.rng.choice(per_class[chosen_cls])
                start = center - half
                return clamp_start(start)

            # fallback if no onsets at all
            return self.rng.randint(0, T - self.win_frames)

        per_class = self.pos_frames_by_class[track_i]

        # ---- Tail hard negatives for cymbals (reduces crash/ride FP)
        # take a real cymbal onset, move the center into its ringing tail (no onset there)
        if self.rng.random() < 0.25:
            cym_avail = [c for c in CYMBAL_CLASSES if len(per_class.get(c, [])) > 0]
            if cym_avail:
                cym_cls = self.rng.choice(cym_avail)
                onset_f = self.rng.choice(per_class[cym_cls])

                # 120..700 ms after onset
                off_lo = max(self.cfg.tol_frames + 2, ms_to_frames(120))
                off_hi = ms_to_frames(700)
                offset = self.rng.randint(off_lo, max(off_lo + 1, off_hi))
                center = onset_f + offset

                # make sure we're not accidentally within tol of another cymbal onset
                if all(abs(f - center) > self.cfg.tol_frames for f in per_class[cym_cls]):
                    half = self.win_frames // 2
                    if center < half or center > (T - half - 1):
                        pass  # skip this attempt
                    else:
                        start = center - half
                        return clamp_start(start)

        # ---- Tail/texture hard negatives for hi-hats (reduces hat confusion + duplicates)
        if self.rng.random() < 0.25:
            hat_avail = [c for c in HIHAT_CLASSES if len(per_class.get(c, [])) > 0]
            if hat_avail:
                hat_cls = self.rng.choice(hat_avail)
                onset_f = self.rng.choice(per_class[hat_cls])

                # class-specific offsets into tail/texture:
                if hat_cls == "hihat_open":
                    # open rings longer
                    off_lo, off_hi = ms_to_frames(80), ms_to_frames(450)
                elif hat_cls == "hihat_closed":
                    # closed ticks/texture: short offsets to teach "don't fire again immediately"
                    off_lo, off_hi = ms_to_frames(25), ms_to_frames(120)
                else:  # hihat_pedal
                    # pedal is often confused with closed ticks; sample near but not at the onset
                    off_lo, off_hi = ms_to_frames(30), ms_to_frames(200)

                off_lo = max(self.cfg.tol_frames + 2, off_lo)
                offset = self.rng.randint(off_lo, max(off_lo + 1, off_hi))
                center = onset_f + offset

                if all(abs(f - center) > self.cfg.tol_frames for f in per_class[hat_cls]):
                    half = self.win_frames // 2
                    if center < half or center > (T - half - 1):
                        pass  # skip this attempt
                    else:
                        start = center - half
                        return clamp_start(start)
        # NEGATIVE: mix easy + hard negatives
        # 50% hard negative near an onset but not within tolerance of center
        if self.rng.random() < 0.5 and len(self.pos_frames[track_i]) > 0:
            center_onset = self.rng.choice(self.pos_frames[track_i])
            # shift the center outside tolerance band
            shift = self.rng.choice(list(range(self.cfg.tol_frames + 2, self.cfg.tol_frames + 30)))
            shift *= self.rng.choice([-1, 1])
            center = center_onset + shift
            start = center - self.win_frames // 2
            return clamp_start(start)

        # easy negative
        return self.rng.randint(0, T - self.win_frames)

    def __getitem__(self, idx: int):
        # selects track
        track_i, start = self.samples[idx]
        row = self.rows[track_i]

        mel_path = Path(row.get("mel", ""))

        if mel_path and mel_path.exists():
            X = np.load(mel_path).astype(np.float32)  # (M, T)
        else:
            wav_path = Path(row["wav"])
            y_audio, _ = librosa.load(wav_path, sr=self.cfg.sr, mono=True)
            X = compute_log_mel(y_audio, self.cfg)  # (M, T)

        T = int(X.shape[1])

        onsets = row.get("onsets_sec", {})
        y_frames = onsets_to_frame_targets(onsets, T, self.cfg)
        
        if T < self.win_frames:
            pad = self.win_frames - T
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
            y_frames = np.pad(y_frames, ((0, pad), (0, 0)), mode="constant")
            T = int(X.shape[1])

        if start < 0:  # random mode
            start = self._choose_start(T, track_i)
        else:
            # all mode: start is already set in __init__, but clamp just in case
            start = max(0, min(start, T - self.win_frames))
        Xw = X[:, start : start + self.win_frames]  # (M, W) 

        center_frame = start + self.win_frames // 2
        center_sec = center_frame * (self.cfg.hop_length / self.cfg.sr)

        start_sec = start * (self.cfg.hop_length / self.cfg.sr)
        end_sec = (start + self.win_frames) * (self.cfg.hop_length / self.cfg.sr)

        if getattr(self.cfg, "label_mode", "window") == "center":
            yw_t = onsets_to_window_targets(
                onsets_sec_by_class=onsets,
                center_sec=center_sec,
                radius_sec=self.radius_sec,
            )
        else:
            yw_t = onsets_to_window_interval_targets(
                onsets_sec_by_class=onsets,
                start_sec=start_sec,
                end_sec=end_sec,
            )


        Xw_t = torch.from_numpy(np.ascontiguousarray(Xw)).unsqueeze(0).clone()

        meta = None
        if self.debug:
            meta = {
                "track_id": row.get("id"),
                "start": int(start),
                "win_frames": int(self.win_frames),
                "center_frame": int(center_frame),
                "center_sec": float(center_sec),
                "T": int(T),
            }

        if self.debug:
            return Xw_t, yw_t, meta
        return Xw_t, yw_t


# -------------------------
# Debug visualization (script mode only)
# -------------------------

def _load_row_by_id(index_jsonl: str, track_id: str) -> dict:
    with open(index_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("id") == track_id:
                return r
    raise RuntimeError(f"Track id not found in index: {track_id}")

def _center_radius_seconds(cfg: SpecConfig) -> float:
    frame_dur = cfg.hop_length / cfg.sr
    return float(cfg.tol_frames) * frame_dur



def debug_plot_center_selection(
    index_jsonl: str,
    track_id: str,
    cfg: SpecConfig,
    p_pos: float = 1.0,
    seed: int = 0,
    classes_to_show: Optional[List[str]] = None,
    context_windows: int = 2,   # how many window-lengths to show around the selected window
):
    """
    Debug plot to inspect how the window start/center is chosen.

    Produces two figures:
      (1) Full mel, zoomed around the sampled window (context view)
      (2) Only the sampled window (window view)
    """
    import matplotlib.pyplot as plt

    row = _load_row_by_id(index_jsonl, track_id)

    mel_path = Path(row.get("mel", ""))
    if mel_path.exists():
        X = np.load(mel_path).astype(np.float32)
    else:
        y_audio, _ = librosa.load(Path(row["wav"]), sr=cfg.sr, mono=True)
        X = compute_log_mel(y_audio, cfg)

    T = int(X.shape[1])
    frame_dur = cfg.hop_length / cfg.sr

    # Use the real dataset sampling logic
    ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=[track_id],
        cfg=cfg,
        max_windows_per_track=1,
        seed=seed,
        p_pos=p_pos,
        sampling="random",
        debug=True,
    )
    Xw_t, yw_t, meta = ds[0]

    start = meta["start"]
    win_frames = meta["win_frames"]
    end = start + win_frames
    center_frame = meta["center_frame"]

    start_sec = start * frame_dur
    end_sec = end * frame_dur
    center_sec = center_frame * frame_dur

    if classes_to_show is None:
        classes_to_show = ["kick", "snare", "hihat_closed", "hihat_open"]

    # Helper: convert sec -> frame (float)
    def sec_to_frame(t_sec: float) -> float:
        return t_sec / frame_dur

    # Tolerance bounds (only in center mode)
    radius_sec = _center_radius_seconds(cfg) if cfg.label_mode == "center" else 0.0
    tol_lo = sec_to_frame(center_sec - radius_sec) if radius_sec > 0 else None
    tol_hi = sec_to_frame(center_sec + radius_sec) if radius_sec > 0 else None

    pad = context_windows * win_frames
    x0 = max(0, start - pad)
    x1 = min(T, end + pad)

    plt.figure(figsize=(14, 4))
    plt.imshow(X, aspect="auto", origin="lower")
    plt.xlim(x0, x1)

    plt.title(
        "Context view (full track, zoomed)\n"
        f"start={start} ({start_sec:.3f}s)  end={end} ({end_sec:.3f}s)  "
        f"center={center_frame} ({center_sec:.3f}s)  win_frames={win_frames}"
    )
    plt.xlabel("Frame")
    plt.ylabel("Mel band")

    # Window shading + center line
    plt.axvspan(start, end, alpha=0.15)
    plt.axvline(center_frame, linewidth=2)

    # Tolerance band (center mode)
    if cfg.label_mode == "center" and radius_sec > 0:
        plt.axvspan(tol_lo, tol_hi, alpha=0.12)

    # Onset markers (only those inside zoom region to reduce clutter)
    onsets = row.get("onsets_sec", {})
    for cls in classes_to_show:
        for t in _iter_onset_times(onsets.get(cls, [])):
            f = sec_to_frame(t)
            if x0 <= f <= x1:
                plt.axvline(f, linewidth=1, alpha=0.25)

    plt.tight_layout()

    Xwin = X[:, start:end]
    plt.figure(figsize=(14, 4))
    plt.imshow(
        Xwin,
        aspect="auto",
        origin="lower",
        extent=[start, end, 0, X.shape[0]],  # keep absolute frame numbers
    )

    plt.title(
        "Window view (only the sampled window)\n"
        f"center={center_frame} ({center_sec:.3f}s)  "
        f"label={yw_t.numpy().astype(int).tolist()}"
    )
    plt.xlabel("Frame (absolute)")
    plt.ylabel("Mel band")

    # Center line (DON'T shade start..end; the image already is the window)
    plt.axvline(center_frame, linewidth=2)

    if cfg.label_mode == "center" and radius_sec > 0:
        plt.axvspan(tol_lo, tol_hi, alpha=0.12)

    # Onsets within the window only (much cleaner)
    for cls in classes_to_show:
        for t in _iter_onset_times(onsets.get(cls, [])):
            f = sec_to_frame(t)
            if start <= f <= end:
                plt.axvline(f, linewidth=1, alpha=0.25)

    plt.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dataset.py debug utilities")
    p.add_argument("--index", default="data/processed/index.jsonl")
    p.add_argument("--id", required=True, help="Track id to visualize")
    p.add_argument("--p_pos", type=float, default=1.0, help="Probability to force positive-centered sampling")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--win_seconds", type=float, default=1.0, help="Window size in seconds")
    p.add_argument("--hop_length", type=int, default=256, help="Hop length in samples")
    p.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    p.add_argument("--n_mels", type=int, default=80, help="Number of mel bands")
    p.add_argument("--sr", type=int, default=22050, help="Sampling rate")
    p.add_argument("--tol_frames", type=int, default=1, help="Tolerance in frames")
    p.add_argument("--label_mode", choices=["center", "window"], default="center", help="Label mode")
    p.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Classes to draw onset markers for (default: kick snare hihat_closed hihat_open)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = SpecConfig(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        win_seconds=args.win_seconds,
        tol_frames=args.tol_frames,
        label_mode=args.label_mode,
    )
    debug_plot_center_selection(
        index_jsonl=args.index,
        track_id=args.id,
        cfg=cfg,
        p_pos=args.p_pos,
        seed=args.seed,
        classes_to_show=args.classes,
    )
