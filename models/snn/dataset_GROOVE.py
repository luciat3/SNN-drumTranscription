import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_delta(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] == 1:
        return d
    d[1:-1] = 0.5 * (x[2:] - x[:-2])
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    return d


class GrooveWindowDataset(Dataset):
    """
    Window-level dataset for SNN training.

    It reads full-track MFCCs and framewise multi-label targets, then samples or
    enumerates fixed-length windows and converts each window into a single
    multi-label target vector [C].
    """

    DEFAULT_CLASS_NAMES = [
        "kick",
        "xstick",
        "snare",
        "hihat_pedal",
        "hihat_closed",
        "hihat_open",
        "tom",
        "floor_tom",
        "vibraslap",
        "crash",
        "ride_bow",
        "ride_bell",
        "chinese_cymbal",
        "splash_cymbal",
    ]

    def __init__(
        self,
        root,
        split="train",
        add_deltas=True,
        sr=22050,
        hop_length=256,
        win_seconds=1.0,
        tol_frames=3,
        label_mode="center",
        sampling="random",
        max_windows_per_track=8,
        stride_frames=0,
        p_pos=0.6,
        seed=42,
    ):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.add_deltas = add_deltas

        self.sr = int(sr)
        self.hop_length = int(hop_length)
        self.win_seconds = float(win_seconds)
        self.tol_frames = int(tol_frames)
        self.label_mode = str(label_mode)
        self.sampling = str(sampling)
        self.max_windows_per_track = int(max_windows_per_track)
        self.stride_frames = int(stride_frames)
        self.p_pos = float(p_pos)
        self.rng = random.Random(seed)

        self.window_frames = max(1, int(round(self.win_seconds * self.sr / self.hop_length)))

        index_path = self.split_dir / "index.jsonl"
        print(
            f"Initializing GrooveWindowDataset with "
            f"root={self.root}, split={split}, add_deltas={add_deltas}, "
            f"sampling={sampling}, window_frames={self.window_frames}"
        )
        print(f"Looking for index file at {index_path}")

        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")

        self.items = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                mfcc_path = row.get("mfcc_path") or row.get("feature_path") or row.get("features_path")
                label_path = row.get("label_path") or row.get("labels_path")

                if mfcc_path is None or label_path is None:
                    raise ValueError(
                        "Each JSONL row must contain 'mfcc_path' and 'label_path' "
                        "(or compatible aliases)."
                    )

                mfcc_path = Path(mfcc_path)
                label_path = Path(label_path)

                if not mfcc_path.is_absolute():
                    mfcc_path = self.root / mfcc_path
                if not label_path.is_absolute():
                    label_path = self.root / label_path

                if not mfcc_path.exists():
                    raise FileNotFoundError(f"Missing MFCC file: {mfcc_path}")
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label file: {label_path}")

                self.items.append({
                    "mfcc_path": mfcc_path,
                    "label_path": label_path,
                    "meta": row,
                })

        if not self.items:
            raise ValueError(f"No items found in {index_path}")

        first_y = np.load(self.items[0]["label_path"]).astype(np.float32)
        if first_y.ndim != 2:
            raise ValueError(f"Expected labels [T, C], got {first_y.shape}")

        self.n_classes = first_y.shape[1]
        self.class_names = self.DEFAULT_CLASS_NAMES.copy()
        if len(self.class_names) != self.n_classes:
            raise ValueError(
                f"Number of class names ({len(self.class_names)}) does not match "
                f"label dimension ({self.n_classes})."
            )

        self.track_lengths = []
        self.pos_frames = []
        self.pos_frames_by_class = []
        self.class_counts = np.zeros(self.n_classes, dtype=np.int64)

        for item_idx, item in enumerate(self.items):
            x = np.load(item["mfcc_path"]).astype(np.float32)
            y = np.load(item["label_path"]).astype(np.float32)

            if x.ndim != 2:
                raise ValueError(f"Features must be 2D, got shape {x.shape} at item={item_idx}")
            if y.ndim != 2:
                raise ValueError(f"Labels must be [T, C], got shape {y.shape} at item={item_idx}")

            if x.shape[0] != y.shape[0] and x.shape[1] == y.shape[0]:
                x = x.T

            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Length mismatch for item={item_idx}: x={x.shape}, y={y.shape}")

            T = int(y.shape[0])
            self.track_lengths.append(T)

            frame_pos = np.where(y.any(axis=1))[0].astype(np.int64).tolist()
            self.pos_frames.append(frame_pos)

            per_class = {}
            for c in range(self.n_classes):
                frames_c = np.where(y[:, c] > 0.5)[0].astype(np.int64).tolist()
                per_class[c] = frames_c
                self.class_counts[c] += len(frames_c)
            self.pos_frames_by_class.append(per_class)

        counts = self.class_counts.astype(np.float32)
        class_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        self.class_weights = class_weights / class_weights.sum()

        self.samples = []
        if self.sampling == "all":
            stride = self.stride_frames if self.stride_frames > 0 else max(1, self.window_frames // 2)
            for item_idx, T in enumerate(self.track_lengths):
                if T <= self.window_frames:
                    self.samples.append((item_idx, 0))
                else:
                    last = T - self.window_frames
                    for start in range(0, last + 1, stride):
                        self.samples.append((item_idx, start))
        elif self.sampling == "random":
            for item_idx in range(len(self.items)):
                for _ in range(self.max_windows_per_track):
                    self.samples.append((item_idx, -1))
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling}")

        print(f"Built {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def _clamp_start(self, start: int, T: int) -> int:
        if T <= self.window_frames:
            return 0
        return max(0, min(int(start), T - self.window_frames))

    def _choose_positive_start(self, track_i: int, T: int) -> int:
        per_class = self.pos_frames_by_class[track_i]
        available_classes = [c for c in range(self.n_classes) if len(per_class[c]) > 0]

        if not available_classes:
            return self.rng.randint(0, max(T - self.window_frames, 0))

        w = self.class_weights[available_classes]
        w = w / w.sum()
        chosen_class = self.rng.choices(available_classes, weights=w.tolist(), k=1)[0]

        onset_frame = self.rng.choice(per_class[chosen_class])
        center = int(onset_frame)
        start = center - self.window_frames // 2
        return self._clamp_start(start, T)

    def _choose_negative_start(self, track_i: int, T: int) -> int:
        if T <= self.window_frames:
            return 0

        pos = self.pos_frames[track_i]
        if not pos:
            return self.rng.randint(0, T - self.window_frames)

        if self.rng.random() < 0.5:
            center_onset = self.rng.choice(pos)
            shift = self.rng.randint(self.tol_frames + 2, self.tol_frames + 30)
            shift *= self.rng.choice([-1, 1])
            center = center_onset + shift
            start = center - self.window_frames // 2
            return self._clamp_start(start, T)

        return self.rng.randint(0, T - self.window_frames)

    def _choose_start(self, track_i: int, T: int) -> int:
        if T <= self.window_frames:
            return 0

        want_pos = self.rng.random() < self.p_pos
        if want_pos:
            return self._choose_positive_start(track_i, T)
        return self._choose_negative_start(track_i, T)

    def _window_target(self, y_frames: np.ndarray, start: int, end: int) -> np.ndarray:
        if self.label_mode == "center":
            center = start + (end - start) // 2
            lo = max(0, center - self.tol_frames)
            hi = min(y_frames.shape[0], center + self.tol_frames + 1)
            y_window = y_frames[lo:hi].max(axis=0)
        elif self.label_mode == "window":
            y_window = y_frames[start:end].max(axis=0)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        return y_window.astype(np.float32)

    def __getitem__(self, idx):
        track_i, start = self.samples[idx]
        item = self.items[track_i]

        x = np.load(item["mfcc_path"], mmap_mode="r")
        y_frames = np.load(item["label_path"], mmap_mode="r")

        if x.ndim != 2:
            raise ValueError(f"Features must be 2D, got shape {x.shape} at idx={idx}")
        if y_frames.ndim != 2:
            raise ValueError(f"Labels must be [T, C], got shape {y_frames.shape} at idx={idx}")

        if x.shape[0] != y_frames.shape[0] and x.shape[1] == y_frames.shape[0]:
            x = x.T

        if x.shape[0] != y_frames.shape[0]:
            raise ValueError(
                f"Length mismatch at idx={idx}: x has shape {x.shape}, y has shape {y_frames.shape}"
            )

        T = x.shape[0]
        if T < self.window_frames:
            pad = self.window_frames - T
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            y_frames = np.pad(y_frames, ((0, pad), (0, 0)), mode="constant")
            T = x.shape[0]

        if start < 0:
            start = self._choose_start(track_i, T)
        else:
            start = self._clamp_start(start, T)

        end = min(start + self.window_frames, T)

        xw = np.asarray(x[start:end], dtype=np.float32)
        yw = self._window_target(np.asarray(y_frames), start, end)
        
        if self.add_deltas:
            dx = compute_delta(xw)
            ddx = compute_delta(dx)
            xw = np.concatenate([xw, dx, ddx], axis=1)

        return (
            torch.tensor(xw, dtype=torch.float32),
            torch.tensor(yw, dtype=torch.float32),
            torch.tensor(xw.shape[0], dtype=torch.long),
        )


def groove_window_collate_fn(batch):
    xs, ys, lengths = zip(*batch)

    lengths = torch.stack(lengths)
    B = len(xs)
    T_max = max(x.shape[0] for x in xs)
    n_in = xs[0].shape[1]

    features = torch.zeros(B, T_max, n_in, dtype=torch.float32)
    labels = torch.stack(ys, dim=0).to(torch.float32)

    for i, x in enumerate(xs):
        T_i = x.shape[0]
        features[i, :T_i] = x

    return features, labels, lengths