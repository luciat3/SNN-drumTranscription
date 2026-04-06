import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class TimitFramewiseDataset(Dataset):
    """
    Minimal PyTorch dataset for processed framewise TIMIT.

    Uses:
      - mfccs.pickle   -> list of [T, 13] feature arrays
      - phonems.pickle -> list of [T] label arrays
    """

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        split_dir = os.path.join(root, split)

        with open(os.path.join(split_dir, "mfccs.pickle"), "rb") as f:
            self.features = pickle.load(f)

        with open(os.path.join(split_dir, "phonems.pickle"), "rb") as f:
            self.labels = pickle.load(f)

        with open(os.path.join(split_dir, "phonem_list.json"), "r") as f:
            self.phoneme_list = json.load(f)

        assert len(self.features) == len(self.labels), \
            "Mismatch between number of feature sequences and label sequences"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]   # numpy array [T, 13]
        y = self.labels[idx]     # numpy array [T]

        # Safety check: framewise alignment
        assert len(x) == len(y), f"Length mismatch at idx={idx}: {len(x)} vs {len(y)}"

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

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