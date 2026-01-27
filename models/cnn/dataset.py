# -*- coding: utf-8 -*-
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


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
    n_mels: int = 80
    # window size
    win_seconds: float = 1.0
    # tolerance t, t+1, t-1, t+2, t-2
    tol_frames: int = 1 


# maped classes 
CLASSES = ["kick", "snare", "hihat_closed", "hihat_open", "tom", "floor_tom", "crash", "ride"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def compute_log_mel(y: np.ndarray, cfg: SpecConfig) -> np.ndarray:
    """
    Computes log-mel spectrogram from audio signal
    compresses with log(1+x) to avoid issues with log(0)
    to improve learning of temporal patterns 
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


def onsets_to_frame_targets(onsets_sec: Dict[str, List[float]], T: int, cfg: SpecConfig) -> np.ndarray:
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
        for t_sec in times:
            # turns seconds to spectrogram frames
            frame = int(round(t_sec * cfg.sr / cfg.hop_length))
            # allows a tolerance to neighbouring frames
            for k in range(-cfg.tol_frames, cfg.tol_frames + 1):
                f = frame + k
                # avoid access outside spectrogram
                if 0 <= f < T:
                    y[f, ci] = 1.0
    return y


def onsets_to_positive_frames(onsets_sec: Dict[str, List[float]], cfg: SpecConfig) -> List[int]:
    """
    Converts onsets (lists of classes where an instrument appears) into frames
    """
    frames = []
    for cls, times in onsets_sec.items():
        # ignores classes that aren't maped (shouldn't be any)
        if cls not in CLASS_TO_IDX:
            continue
        # iterates every second
        for t_sec in times:
            # turns seconds to spectrogram frames
            frames.append(int(round(t_sec * cfg.sr / cfg.hop_length)))
    return sorted(set(frames))


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
    ):
        self.cfg = cfg
        self.win_frames = int(round(cfg.win_seconds * cfg.sr / cfg.hop_length))

        # in order to avoid dataset bias, we force 80% positive windows
        if not (0.0 <= p_pos <= 1.0):
            raise ValueError("p_pos must be between 0 and 1")
        self.p_pos = p_pos

        self.rng = random.Random(seed)
        ids_set = set(ids)

        self.rows = []
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["id"] in ids_set:
                    self.rows.append(row)

        if not self.rows:
            raise RuntimeError("No rows found.")

        # onsets list
        self.pos_frames = [onsets_to_positive_frames(r["onsets_sec"], cfg) for r in self.rows]

        self.samples: List[Tuple[int, int]] = []
        for ti in range(len(self.rows)):
            for wi in range(max_windows_per_track):
                self.samples.append((ti, wi))

    # total number of tracks
    def __len__(self) -> int:
        return len(self.samples)

    def _choose_start(self, T: int, pos_frames: List[int]) -> int:
        """
        Chooses the initial frame of the window

        :param T: Total number of spectrogram frames 
        :param pos_frames: frames with onsets
        """
        if T <= self.win_frames:
            return 0

        # we choose the frame depending if it's positive or negative (not always centered to avoid bias)
        # oversampling of positives 
        want_pos = (self.rng.random() < self.p_pos) and (len(pos_frames) > 0)
        if want_pos:
            center = self.rng.choice(pos_frames)
            start = center - self.win_frames // 2
            start = max(0, min(start, T - self.win_frames))
            return start

        return self.rng.randint(0, T - self.win_frames - 1)

    def __getitem__(self, idx: int):
        # selects track
        track_i, _ = self.samples[idx]
        row = self.rows[track_i]

        wav_path = Path(row["wav"])
        y_audio, _ = librosa.load(wav_path, sr=self.cfg.sr, mono=True)
        # converts intro spectrogram
        X = compute_log_mel(y_audio, self.cfg)  # (M, T)
        T = X.shape[1]
        # generates targets
        y_frames = onsets_to_frame_targets(row["onsets_sec"], T, self.cfg)  # (T, C)

        # selects window
        start = self._choose_start(T, self.pos_frames[track_i])
        Xw = X[:, start : start + self.win_frames]  # (M, W)
        center = min(max(start + self.win_frames // 2, 0), T - 1)
        yw = y_frames[center]  # (C,)

        # converts into pyTorch tensors
        Xw_t = torch.from_numpy(Xw).unsqueeze(0)  # [1, M, W] = [Channel, #Mel bands, Window frames]
        yw_t = torch.from_numpy(yw)               # [C] = [Bool active classes]
        return Xw_t, yw_t
