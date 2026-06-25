# -*- coding: utf-8 -*-
"""
Fine-tune the CNN drum onset model on the RWC-style dataset using MFCC features.
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.cnn.model import DrumCNN


CLASSES = [
    "kick",
    "xstick",
    "snare",
    "hihat_closed",
    "hihat_open",
    "hihat_pedal",
    "tom",
    "floor_tom",
    "vibraslap",
    "crash",
    "ride_bow",
    "ride_bell",
    "chinese_cymbal",
    "splash_cymbal",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class FineTuneConfig:
    sr: int = 22050
    hop_length: int = 256
    win_seconds: float = 1.0
    tol_frames: int = 3
    label_mode: str = "center" 


def load_splits(splits_path: str):
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    return data["train"], data["val"], data["test"]


def iter_onset_times(times: Iterable):
    """Supports [0.52, ...] and [{"t": 0.52, "vel": ...}, ...]."""
    for x in times:
        if isinstance(x, dict):
            t = x.get("t")
            if t is not None:
                yield float(t)
        else:
            yield float(x)


def load_index_rows(index_jsonl: str, ids: List[str]) -> List[dict]:
    wanted = set(ids)
    rows = []
    with open(index_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("id") in wanted:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found for split in {index_jsonl}")
    return rows


def get_mfcc_path(row: dict) -> Path:
    """Find MFCC feature path in the new index format or older aliases."""
    if "feature_paths" in row and "mfcc" in row["feature_paths"]:
        return Path(row["feature_paths"]["mfcc"])
    for key in ("mfcc_path", "feature_path", "features_path"):
        if key in row:
            return Path(row[key])
    raise KeyError(
        "Expected row['feature_paths']['mfcc'] or one of: "
        "row['mfcc_path'], row['feature_path'], row['features_path']."
    )


def get_onsets(row: dict) -> Dict[str, list]:
    if "onsets" in row:
        return row["onsets"]
    if "onsets_sec" in row:
        return row["onsets_sec"]
    raise KeyError("Expected row['onsets'] or row['onsets_sec'].")


def normalize_window(x: np.ndarray, mode: str = "per_window") -> np.ndarray:
    x = x.astype(np.float32)

    if mode == "none":
        return x

    if mode == "per_window":
        return (x - x.mean()) / (x.std() + 1e-6)

    if mode == "per_feature_window":
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

    if mode == "per_track":
        return x

    raise ValueError(f"Unknown normalization mode: {mode}")


def make_target_center(onsets_by_class: Dict[str, list], center_sec: float, radius_sec: float) -> torch.Tensor:
    y = torch.zeros(len(CLASSES), dtype=torch.float32)
    lo = center_sec - radius_sec
    hi = center_sec + radius_sec

    for cls_name, onset_list in onsets_by_class.items():
        if cls_name not in CLASS_TO_IDX:
            continue
        ci = CLASS_TO_IDX[cls_name]
        for t in iter_onset_times(onset_list):
            if lo <= t <= hi:
                y[ci] = 1.0
                break
    return y


def make_target_window(onsets_by_class: Dict[str, list], start_sec: float, end_sec: float) -> torch.Tensor:
    y = torch.zeros(len(CLASSES), dtype=torch.float32)
    for cls_name, onset_list in onsets_by_class.items():
        if cls_name not in CLASS_TO_IDX:
            continue
        ci = CLASS_TO_IDX[cls_name]
        for t in iter_onset_times(onset_list):
            if start_sec <= t <= end_sec:
                y[ci] = 1.0
                break
    return y


class RWCMFCCWindowDataset(Dataset):
    """
    Dataset for the new RWC index.jsonl using MFCC features.

    It reads:
      row["feature_paths"]["mfcc"]
      row["onsets"]

    Returns:
      X: [1, n_mfcc, win_frames]
      y: [14]
    """

    def __init__(
        self,
        index_jsonl: str,
        ids: List[str],
        cfg: FineTuneConfig,
        sampling: str = "random",
        max_windows_per_track: int = 8,
        stride_frames: int = 0,
        p_pos: float = 0.7,
        seed: int = 42,
        norm: str = "per_window",
    ):
        super().__init__()
        assert sampling in ["random", "all"]

        self.index_root = Path(index_jsonl).resolve().parent
        self.rows = load_index_rows(index_jsonl, ids)
        self.cfg = cfg
        self.sampling = sampling
        self.max_windows_per_track = int(max_windows_per_track)
        self.stride_frames = int(stride_frames)
        self.p_pos = float(p_pos)
        self.rng = random.Random(seed)
        self.norm = norm

        self.win_frames = max(1, int(round(cfg.win_seconds * cfg.sr / cfg.hop_length)))
        self.radius_sec = float(cfg.tol_frames) * (cfg.hop_length / cfg.sr)

        self.track_lengths = []
        self.pos_frames_by_class = []
        self.pos_frames = []
        self.class_counts = np.zeros(len(CLASSES), dtype=np.int64)

        for row in self.rows:
            X = np.load(self._resolve_path(get_mfcc_path(row)), mmap_mode="r")
            T = self._infer_time_frames(X)
            self.track_lengths.append(T)

            onsets = get_onsets(row)
            per_class = {cls: [] for cls in CLASSES}
            all_pos = []

            for cls_name, onset_list in onsets.items():
                if cls_name not in CLASS_TO_IDX:
                    continue
                ci = CLASS_TO_IDX[cls_name]
                for t in iter_onset_times(onset_list):
                    frame = int(round(t * cfg.sr / cfg.hop_length))
                    if 0 <= frame < T:
                        per_class[cls_name].append(frame)
                        all_pos.append(frame)
                        self.class_counts[ci] += 1

            for cls_name in per_class:
                per_class[cls_name] = sorted(set(per_class[cls_name]))

            self.pos_frames_by_class.append(per_class)
            self.pos_frames.append(sorted(set(all_pos)))

        counts = self.class_counts.astype(np.float32)
        self.class_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        self.class_weights = self.class_weights / self.class_weights.sum()

        self.samples = []
        if self.sampling == "all":
            stride = self.stride_frames if self.stride_frames > 0 else max(1, self.win_frames // 2)
            for track_i, T in enumerate(self.track_lengths):
                if T <= self.win_frames:
                    self.samples.append((track_i, 0))
                else:
                    last = T - self.win_frames
                    for start in range(0, last + 1, stride):
                        self.samples.append((track_i, start))
        else:
            for track_i in range(len(self.rows)):
                for _ in range(self.max_windows_per_track):
                    self.samples.append((track_i, -1))

        print(f"Loaded {len(self.rows)} tracks")
        print(f"Created {len(self.samples)} windows")
        print("Class counts:", {CLASSES[i]: int(self.class_counts[i]) for i in range(len(CLASSES))})

    def _resolve_path(self, p: Path) -> Path:
        if p.is_absolute():
            return p
        if p.exists():
            return p
        return self.index_root / p

    @staticmethod
    def _infer_time_frames(X: np.ndarray) -> int:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D MFCC array, got shape {X.shape}")
        if X.shape[0] <= X.shape[1] and X.shape[0] <= 256:
            return int(X.shape[1])
        return int(X.shape[0])

    def _load_mfcc_freq_time(self, row: dict) -> np.ndarray:
        X = np.load(self._resolve_path(get_mfcc_path(row))).astype(np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D MFCC array, got shape {X.shape}")
        # Return [F, T]. If the stored file is [T, F], transpose it.
        if not (X.shape[0] <= X.shape[1] and X.shape[0] <= 256):
            X = X.T
        return X

    def __len__(self):
        return len(self.samples)

    def _choose_start(self, T: int, track_i: int) -> int:
        if T <= self.win_frames:
            return 0

        def clamp_start(s: int) -> int:
            return max(0, min(int(s), T - self.win_frames))

        want_pos = self.rng.random() < self.p_pos

        if want_pos:
            per_class = self.pos_frames_by_class[track_i]
            available = [cls for cls in CLASSES if len(per_class.get(cls, [])) > 0]

            if available:
                avail_idx = [CLASS_TO_IDX[c] for c in available]
                w = self.class_weights[avail_idx]
                w = w / w.sum()
                chosen_cls = self.rng.choices(available, weights=w.tolist(), k=1)[0]

                half = self.win_frames // 2
                frames = per_class[chosen_cls]
                valid = [f for f in frames if half <= f <= T - half - 1]
                center = self.rng.choice(valid) if valid else self.rng.choice(frames)
                return clamp_start(center - half)

        # Negative sampling: half of negatives are close to an onset, but not centered on it.
        if self.rng.random() < 0.5 and len(self.pos_frames[track_i]) > 0:
            onset_f = self.rng.choice(self.pos_frames[track_i])
            shift = self.rng.choice(list(range(self.cfg.tol_frames + 2, self.cfg.tol_frames + 30)))
            shift *= self.rng.choice([-1, 1])
            center = onset_f + shift
            return clamp_start(center - self.win_frames // 2)

        return self.rng.randint(0, T - self.win_frames)

    def __getitem__(self, idx: int):
        track_i, start = self.samples[idx]
        row = self.rows[track_i]

        X = self._load_mfcc_freq_time(row)  # [F, T]

        if self.norm == "per_track":
            X = (X - X.mean()) / (X.std() + 1e-6)

        T = int(X.shape[1])

        if T < self.win_frames:
            pad = self.win_frames - T
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
            T = int(X.shape[1])

        if start < 0:
            start = self._choose_start(T, track_i)
        else:
            start = max(0, min(start, T - self.win_frames))

        end = start + self.win_frames
        Xw = X[:, start:end]

        if self.norm in ("per_window", "per_feature_window"):
            Xw = normalize_window(Xw, mode=self.norm)

        onsets = get_onsets(row)
        start_sec = start * self.cfg.hop_length / self.cfg.sr
        end_sec = end * self.cfg.hop_length / self.cfg.sr
        center_frame = start + self.win_frames // 2
        center_sec = center_frame * self.cfg.hop_length / self.cfg.sr

        if self.cfg.label_mode == "center":
            y = make_target_center(onsets, center_sec, self.radius_sec)
        else:
            y = make_target_window(onsets, start_sec, end_sec)

        Xw_t = torch.from_numpy(np.ascontiguousarray(Xw)).unsqueeze(0)  # [1, F, T]
        return Xw_t.float(), y.float()


def load_pretrained_cnn(model: nn.Module, checkpoint_path: str, device: str, strict: bool = True):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if strict:
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint strictly: {checkpoint_path}")
        return

    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)
    print(f"Loaded compatible tensors: {len(compatible)}/{len(model_state)}")
    skipped = sorted(set(model_state.keys()) - set(compatible.keys()))
    if skipped:
        print("Skipped incompatible/missing tensors:", skipped)


@torch.no_grad()
def f1_stats_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: Union[float, torch.Tensor] = 0.5):
    probs = torch.sigmoid(logits)

    if isinstance(thr, torch.Tensor):
        thr = thr.to(probs.device).view(1, -1)
        preds = (probs >= thr).to(targets.dtype)
    else:
        preds = (probs >= float(thr)).to(targets.dtype)

    eps = 1e-8
    tp_c = (preds * targets).sum(dim=0)
    fp_c = (preds * (1 - targets)).sum(dim=0)
    fn_c = ((1 - preds) * targets).sum(dim=0)
    tn_c = ((1 - preds) * (1 - targets)).sum(dim=0)

    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c = tp_c / (tp_c + fn_c + eps)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    tp, fp, fn, tn = tp_c.sum(), fp_c.sum(), fn_c.sum(), tn_c.sum()

    return {
        "micro_f1": float(2 * tp / (2 * tp + fp + fn + eps)),
        "micro_precision": float(tp / (tp + fp + eps)),
        "micro_recall": float(tp / (tp + fn + eps)),
        "macro_f1": float(f1_c.mean()),
        "macro_precision": float(prec_c.mean()),
        "macro_recall": float(rec_c.mean()),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision_by_class": {CLASSES[i]: float(prec_c[i]) for i in range(len(CLASSES))},
        "recall_by_class": {CLASSES[i]: float(rec_c[i]) for i in range(len(CLASSES))},
        "f1_by_class": {CLASSES[i]: float(f1_c[i]) for i in range(len(CLASSES))},
        "tp_by_class": {CLASSES[i]: float(tp_c[i]) for i in range(len(CLASSES))},
        "fp_by_class": {CLASSES[i]: float(fp_c[i]) for i in range(len(CLASSES))},
        "fn_by_class": {CLASSES[i]: float(fn_c[i]) for i in range(len(CLASSES))},
        "tn_by_class": {CLASSES[i]: float(tn_c[i]) for i in range(len(CLASSES))},
        "support_by_class": {CLASSES[i]: float(tp_c[i] + fn_c[i]) for i in range(len(CLASSES))},
    }


@torch.no_grad()
def find_best_threshold_per_class(logits: torch.Tensor, targets: torch.Tensor, thresholds=None, min_support: int = 5):
    if thresholds is None:
        thresholds = torch.linspace(0.20, 0.95, 16)

    probs = torch.sigmoid(logits)
    C = targets.shape[1]
    support = targets.sum(dim=0)

    best_thr = torch.full((C,), 0.5, dtype=torch.float32)
    best_f1 = torch.full((C,), -1.0, dtype=torch.float32)

    eps = 1e-8
    for thr in thresholds:
        preds = (probs >= float(thr)).to(targets.dtype)
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)

        improved = f1 > best_f1
        best_f1[improved] = f1[improved]
        best_thr[improved] = float(thr)

    # Avoid extremely low thresholds for classes absent or nearly absent in validation.
    best_thr[support < min_support] = 0.95
    return best_thr, best_f1


def estimate_pos_weight_from_loader(loader: DataLoader, num_classes: int, max_batches: int = 9999, eps: float = 1e-6):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    n = 0
    for batch_idx, (_, y) in enumerate(loader):
        pos += y.sum(dim=0).double()
        n += y.shape[0]
        if batch_idx + 1 >= max_batches:
            break
    neg = n - pos
    pos_weight = neg / (pos + eps)
    pos_rate = pos / max(n, 1)
    return pos_weight.float(), pos_rate.float()


def freeze_backbone(model: DrumCNN):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model: DrumCNN):
    for p in model.parameters():
        p.requires_grad = True


def make_optimizer(model, lr: float, weight_decay: float):
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, log_every: int = 20):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

        if log_every > 0 and batch_idx % log_every == 0:
            print(f"  batch {batch_idx:4d}/{len(loader)-1:4d} | loss={loss.item():.4f}")

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    stats = f1_stats_from_logits(logits_all, targets_all, thr=0.5)
    stats["loss"] = total_loss / max(len(loader.dataset), 1)
    return stats


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    stats = f1_stats_from_logits(logits_all, targets_all, thr=threshold)
    stats["loss"] = total_loss / max(len(loader.dataset), 1)
    return stats, logits_all, targets_all


def json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index_path", default="data/processed/RWC/index.jsonl")
    parser.add_argument("--splits_path", default="data/processed/RWC/splits.json")
    parser.add_argument("--checkpoint", default="models/cnn/runs/run12/best.pt")
    parser.add_argument("--out_dir", default="models/cnn/runs/finetune_rwc_mfcc")

    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_seconds", type=float, default=1.0)
    parser.add_argument("--tol_frames", type=int, default=3)
    parser.add_argument("--label_mode", choices=["center", "window"], default="center")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--frozen_epochs", type=int, default=3)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_full", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--train_windows_per_track", type=int, default=32)
    parser.add_argument("--eval_stride_frames", type=int, default=43)
    parser.add_argument("--p_pos", type=float, default=0.75)

    parser.add_argument(
        "--norm",
        choices=["none", "per_window", "per_feature_window", "per_track"],
        default="per_window",
    )
    parser.add_argument("--pos_weight_mode", choices=["log", "sqrt", "none"], default="log")
    parser.add_argument("--pos_weight_max", type=float, default=8.0)
    parser.add_argument("--min_threshold", type=float, default=0.20)
    parser.add_argument("--low_support_threshold", type=int, default=5)

    parser.add_argument("--non_strict_load", action="store_true", help="Use this if checkpoint keys/shapes differ.")
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print("Device:", device)

    cfg = FineTuneConfig(
        sr=args.sr,
        hop_length=args.hop_length,
        win_seconds=args.win_seconds,
        tol_frames=args.tol_frames,
        label_mode=args.label_mode,
    )

    train_ids, val_ids, test_ids = load_splits(args.splits_path)

    train_ds = RWCMFCCWindowDataset(
        index_jsonl=args.index_path,
        ids=train_ids,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=args.train_windows_per_track,
        p_pos=args.p_pos,
        seed=42,
        norm=args.norm,
    )

    val_ds = RWCMFCCWindowDataset(
        index_jsonl=args.index_path,
        ids=val_ids,
        cfg=cfg,
        sampling="all",
        stride_frames=args.eval_stride_frames,
        seed=123,
        norm=args.norm,
    )

    test_ds = RWCMFCCWindowDataset(
        index_jsonl=args.index_path,
        ids=test_ids,
        cfg=cfg,
        sampling="all",
        stride_frames=args.eval_stride_frames,
        seed=456,
        norm=args.norm,
    )

    pin_memory = device == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=pin_memory, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         pin_memory=pin_memory)

    model = DrumCNN(num_classes=len(CLASSES), dropout=0.3).to(device)
    load_pretrained_cnn(model=model, checkpoint_path=args.checkpoint, device=device, strict=not args.non_strict_load)

    pw_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    pos_weight, pos_rate = estimate_pos_weight_from_loader(pw_loader, num_classes=len(CLASSES))

    if args.pos_weight_mode == "log":
        pos_weight = torch.log1p(pos_weight)
    elif args.pos_weight_mode == "sqrt":
        pos_weight = torch.sqrt(pos_weight)
    else:
        pos_weight = torch.ones_like(pos_weight)

    pos_weight = pos_weight.clamp(max=args.pos_weight_max)

    print("Estimated pos_rate:", {CLASSES[i]: float(pos_rate[i]) for i in range(len(CLASSES))})
    print("Using pos_weight:", {CLASSES[i]: float(pos_weight[i]) for i in range(len(CLASSES))})

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_thresholds = torch.full((len(CLASSES),), 0.5)
    bad_epochs = 0
    optimizer = None
    scheduler = None
    start_time = time.time()

    threshold_grid = torch.linspace(args.min_threshold, 0.95, int(round((0.95 - args.min_threshold) / 0.05)) + 1)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if epoch == 1 and args.frozen_epochs > 0:
            print("Freezing CNN feature extractor; training classifier only.")
            freeze_backbone(model)
            optimizer = make_optimizer(model, lr=args.lr_head, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2, threshold=1e-3
            )

        if epoch == args.frozen_epochs + 1:
            print("Unfreezing full CNN for fine-tuning.")
            unfreeze_all(model)
            optimizer = make_optimizer(model, lr=args.lr_full, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2, threshold=1e-3
            )

        if optimizer is None:
            unfreeze_all(model)
            optimizer = make_optimizer(model, lr=args.lr_full, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2, threshold=1e-3
            )

        train_stats = train_one_epoch(
            model=model,
            loader=train_dl,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            log_every=args.log_every,
        )

        val_stats_raw, logits_val, targets_val = evaluate(
            model=model,
            loader=val_dl,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )

        thr_c, _ = find_best_threshold_per_class(
            logits_val,
            targets_val,
            thresholds=threshold_grid,
            min_support=args.low_support_threshold,
        )

        val_stats, _, _ = evaluate(
            model=model,
            loader=val_dl,
            criterion=criterion,
            device=device,
            threshold=thr_c,
        )

        scheduler.step(val_stats["macro_f1"])
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_microF1={train_stats['micro_f1']:.4f} | "
            f"train_macroF1={train_stats['macro_f1']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_microF1={val_stats['micro_f1']:.4f} | "
            f"val_macroF1={val_stats['macro_f1']:.4f} | "
            f"thr_mean={float(thr_c.mean()):.2f} | "
            f"lr={lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        metrics_row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "lr": lr,
            "train": train_stats,
            "val": val_stats,
            "threshold_by_class": {CLASSES[i]: float(thr_c[i]) for i in range(len(CLASSES))},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row, default=json_safe) + "\n")

        if val_stats["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_stats["macro_f1"]
            best_thresholds = thr_c.clone()
            bad_epochs = 0

            test_stats, _, _ = evaluate(
                model=model,
                loader=test_dl,
                criterion=criterion,
                device=device,
                threshold=best_thresholds,
            )

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "cfg": cfg.__dict__,
                "source_checkpoint": args.checkpoint,
                "feature_type": "mfcc",
                "normalization": args.norm,
                "pos_weight_mode": args.pos_weight_mode,
                "threshold_by_class": {CLASSES[i]: float(best_thresholds[i]) for i in range(len(CLASSES))},
                "val_macroF1": val_stats["macro_f1"],
                "val_microF1": val_stats["micro_f1"],
                "test_macroF1": test_stats["macro_f1"],
                "test_microF1": test_stats["micro_f1"],
                "test_stats": test_stats,
                "total_time_sec": time.time() - start_time,
            }

            torch.save(ckpt, out_dir / "best.pt")
            with (out_dir / "best_test_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(test_stats, f, indent=2, default=json_safe)

            print(
                f"Saved best.pt | "
                f"val_macroF1={val_stats['macro_f1']:.4f} | "
                f"test_macroF1={test_stats['macro_f1']:.4f} | "
                f"test_microF1={test_stats['micro_f1']:.4f}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no val_macroF1 improvement for {args.patience} epochs.")
                break

    print(f"Best val_macroF1: {best_val_macro_f1:.4f}")
    print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
