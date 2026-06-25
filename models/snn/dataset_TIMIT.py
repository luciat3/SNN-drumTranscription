import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_delta(x: np.ndarray) -> np.ndarray:
    """
    Simple time derivative for [T, F].
    Uses centered differences in the middle and one-sided at boundaries.
    """
    d = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] == 1:
        return d

    d[1:-1] = 0.5 * (x[2:] - x[:-2])
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    return d


class TimitFramewiseDataset(Dataset):
    def __init__(self, root, split="train", add_deltas=True):
        split_dir = os.path.join(root, split)

        with open(os.path.join(split_dir, "mfccs.pickle"), "rb") as f:
            self.features = pickle.load(f)

        with open(os.path.join(split_dir, "phonems.pickle"), "rb") as f:
            self.labels = pickle.load(f)

        with open(os.path.join(split_dir, "phonem_list.json"), "r") as f:
            self.phoneme_list = json.load(f)

        self.add_deltas = add_deltas

        self.sr = 22050
        self.hop_length = 256
        self.win_seconds = 1.0
        self.window_frames = int(round(self.win_seconds * self.sr / self.hop_length))  # 86
        self.window_hop_frames = self.window_frames  # non-overlapping -> IN CNN THERE'S SOME OVERLAPPING

        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].astype(np.float32)   # [T, 13] in your current files
        y = self.labels[idx].astype(np.int64)       # [T]

        assert len(x) == len(y), f"Length mismatch at idx={idx}: {len(x)} vs {len(y)}"

        if self.add_deltas:
            if x.shape[1] == 13:
                dx = compute_delta(x)
                ddx = compute_delta(dx)
                x = np.concatenate([x, dx, ddx], axis=1)   # [T, 39]
            elif x.shape[1] != 39:
                raise ValueError(f"Expected 13 or 39 features, got {x.shape[1]}")

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def timit_collate_fn(batch):
    """
    Pads variable-length TIMIT sequences in a batch.

    Input:
        batch = list of tuples (x, y)
            x: [T_i, 13]
            y: [T_i]

    Returns:
        features: [B, T_max, 13]
        labels:   [B, T_max]
        lengths:  [B]
    """
    xs, ys = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    B = len(xs)
    T_max = max(lengths).item()
    n_in = xs[0].shape[1]

    features = torch.zeros(B, T_max, n_in, dtype=torch.float32)
    labels = torch.zeros(B, T_max, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        T_i = x.shape[0]
        features[i, :T_i] = x
        labels[i, :T_i] = y

    return features, labels, lengths