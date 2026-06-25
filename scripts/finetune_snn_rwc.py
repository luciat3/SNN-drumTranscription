# -*- coding: utf-8 -*-
"""
Fine-tune the SNN drum onset model on the new RWC-style dataset using MFCCs.
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.snn.model import LSNN
from models.snn.poisson_encoder import encode_window_sequence, effective_repeat_steps, normalize_input_encoding


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
GROOVE_PRETRAIN_CLASS_ORDER = [
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
REPEAT_STEPS = 5


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
    """Find an MFCC feature path in the new index format or older aliases."""
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


def compute_delta(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] == 1:
        return d
    d[1:-1] = 0.5 * (x[2:] - x[:-2])
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    return d


def normalize_features(x: np.ndarray, mode: str) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "none":
        return x
    if mode in ("per_window", "per_track"):
        return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    if mode == "global_per_window":
        return (x - x.mean()) / (x.std() + 1e-6)
    raise ValueError(f"Unknown normalization mode: {mode}")


def make_target_center(onsets_by_class: Dict[str, list], center_sec: float, radius_sec: float):
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


def make_target_window(onsets_by_class: Dict[str, list], start_sec: float, end_sec: float):
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
    RWC-style MFCC dataset for LSNN fine-tuning.

    Returns:
      X: [T_frames, n_features] float32
      y: [14] float32
      length: scalar long
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
        add_deltas: bool = True,
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
        self.add_deltas = bool(add_deltas)
        self.win_frames = max(1, int(round(cfg.win_seconds * cfg.sr / cfg.hop_length)))
        self.radius_sec = float(cfg.tol_frames) * (cfg.hop_length / cfg.sr)

        self.pos_frames_by_class = []
        self.pos_frames = []
        self.class_counts = np.zeros(len(CLASSES), dtype=np.int64)
        self.track_lengths = []

        for row in self.rows:
            mfcc_path = self._resolve_path(get_mfcc_path(row))
            X = np.load(mfcc_path, mmap_mode="r")
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
        if sampling == "all":
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
        # First try relative to cwd, then relative to index directory.
        if p.exists():
            return p
        return self.index_root / p

    @staticmethod
    def _infer_time_frames(X: np.ndarray) -> int:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D MFCC array, got shape {X.shape}")
        # MFCCs are usually [n_mfcc, T]. If first dim is small, time is axis 1.
        return int(X.shape[1] if X.shape[0] <= X.shape[1] and X.shape[0] <= 256 else X.shape[0])

    def _load_mfcc_time_major(self, row: dict) -> np.ndarray:
        X = np.load(self._resolve_path(get_mfcc_path(row))).astype(np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D MFCC array, got shape {X.shape}")
        # Convert [F, T] -> [T, F] when needed.
        if X.shape[0] <= X.shape[1] and X.shape[0] <= 256:
            X = X.T
        return X

    def __len__(self):
        return len(self.samples)

    def _clamp_start(self, start: int, T: int) -> int:
        if T <= self.win_frames:
            return 0
        return max(0, min(int(start), T - self.win_frames))

    def _choose_start(self, T: int, track_i: int) -> int:
        if T <= self.win_frames:
            return 0

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
                return self._clamp_start(center - half, T)

        if self.rng.random() < 0.5 and len(self.pos_frames[track_i]) > 0:
            onset_f = self.rng.choice(self.pos_frames[track_i])
            shift = self.rng.randint(self.cfg.tol_frames + 2, self.cfg.tol_frames + 30)
            shift *= self.rng.choice([-1, 1])
            return self._clamp_start(onset_f + shift - self.win_frames // 2, T)

        return self.rng.randint(0, T - self.win_frames)

    def __getitem__(self, idx: int):
        track_i, start = self.samples[idx]
        row = self.rows[track_i]
        X = self._load_mfcc_time_major(row)  # [T, F]

        if self.norm == "per_track":
            X = normalize_features(X, mode="per_track")

        T = int(X.shape[0])
        if T < self.win_frames:
            pad = self.win_frames - T
            X = np.pad(X, ((0, pad), (0, 0)), mode="constant")
            T = int(X.shape[0])

        if start < 0:
            start = self._choose_start(T, track_i)
        else:
            start = self._clamp_start(start, T)

        end = start + self.win_frames
        Xw = np.asarray(X[start:end], dtype=np.float32)

        if self.norm in ("per_window", "global_per_window"):
            Xw = normalize_features(Xw, mode=self.norm)

        if self.add_deltas:
            dx = compute_delta(Xw)
            ddx = compute_delta(dx)
            Xw = np.concatenate([Xw, dx, ddx], axis=1)

        onsets = get_onsets(row)
        start_sec = start * self.cfg.hop_length / self.cfg.sr
        end_sec = end * self.cfg.hop_length / self.cfg.sr
        center_frame = start + self.win_frames // 2
        center_sec = center_frame * self.cfg.hop_length / self.cfg.sr

        if self.cfg.label_mode == "center":
            y = make_target_center(onsets, center_sec, self.radius_sec)
        else:
            y = make_target_window(onsets, start_sec, end_sec)

        return (
            torch.from_numpy(np.ascontiguousarray(Xw)).float(),
            y.float(),
            torch.tensor(Xw.shape[0], dtype=torch.long),
        )


def snn_window_collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    lengths = torch.stack(lengths)
    B = len(xs)
    T_max = max(x.shape[0] for x in xs)
    n_in = xs[0].shape[1]
    features = torch.zeros(B, T_max, n_in, dtype=torch.float32)
    labels = torch.stack(ys, dim=0).float()
    for i, x in enumerate(xs):
        features[i, : x.shape[0]] = x
    return features, labels, lengths


def prepare_window_batch(
    features,
    repeat_steps=REPEAT_STEPS,
    input_encoding="analog",
    poisson_steps=None,
    poisson_max_rate=0.30,
    poisson_silence_steps=0,
    poisson_collapse="none",
    poisson_normalize="window",
):
    input_encoding = normalize_input_encoding(input_encoding)
    if poisson_steps is None:
        poisson_steps = repeat_steps

    features = encode_window_sequence(
        features,
        image_feature_size=features.shape[-1],
        encoding=input_encoding,
        poisson_steps=poisson_steps,
        poisson_max_rate=poisson_max_rate,
        poisson_silence_steps=poisson_silence_steps,
        poisson_collapse=poisson_collapse,
        poisson_normalize=poisson_normalize,
    )

    x = features.transpose(0, 1).contiguous()  # [T_snn, B, n_in]
    if not (input_encoding == "poisson" and str(poisson_collapse).lower() == "none") and repeat_steps > 1:
        x = x.repeat_interleave(repeat_steps, dim=0)
    return x


@torch.no_grad()
def f1_stats_from_logits(logits, targets, thr: Union[float, torch.Tensor] = 0.5):
    probs = torch.sigmoid(logits)
    if isinstance(thr, torch.Tensor):
        preds = (probs >= thr.to(probs.device).view(1, -1)).to(targets.dtype)
    else:
        preds = (probs >= float(thr)).to(targets.dtype)

    eps = 1e-8
    tp_c = (preds * targets).sum(dim=0)
    fp_c = (preds * (1.0 - targets)).sum(dim=0)
    fn_c = ((1.0 - preds) * targets).sum(dim=0)
    tn_c = ((1.0 - preds) * (1.0 - targets)).sum(dim=0)

    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c = tp_c / (tp_c + fn_c + eps)
    f1_c = 2.0 * prec_c * rec_c / (prec_c + rec_c + eps)

    tp, fp, fn, tn = tp_c.sum(), fp_c.sum(), fn_c.sum(), tn_c.sum()
    return {
        "micro_f1": float(2.0 * tp / (2.0 * tp + fp + fn + eps)),
        "micro_precision": float(tp / (tp + fp + eps)),
        "micro_recall": float(tp / (tp + fn + eps)),
        "macro_f1": float(f1_c.mean()),
        "macro_precision": float(prec_c.mean()),
        "macro_recall": float(rec_c.mean()),
        "exact_match": float((preds == targets).all(dim=1).float().mean()),
        "precision_by_class": {CLASSES[i]: float(prec_c[i]) for i in range(len(CLASSES))},
        "recall_by_class": {CLASSES[i]: float(rec_c[i]) for i in range(len(CLASSES))},
        "f1_by_class": {CLASSES[i]: float(f1_c[i]) for i in range(len(CLASSES))},
        "tp_by_class": {CLASSES[i]: float(tp_c[i]) for i in range(len(CLASSES))},
        "fp_by_class": {CLASSES[i]: float(fp_c[i]) for i in range(len(CLASSES))},
        "fn_by_class": {CLASSES[i]: float(fn_c[i]) for i in range(len(CLASSES))},
        "tn_by_class": {CLASSES[i]: float(tn_c[i]) for i in range(len(CLASSES))},
        "support_by_class": {CLASSES[i]: float(tp_c[i] + fn_c[i]) for i in range(len(CLASSES))},
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


@torch.no_grad()
def find_best_threshold_per_class(logits, targets, thresholds=None, min_support=5):
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
        fp = (preds * (1.0 - targets)).sum(dim=0)
        fn = ((1.0 - preds) * targets).sum(dim=0)
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
        improved = f1 > best_f1
        best_f1[improved] = f1[improved]
        best_thr[improved] = float(thr)

    best_thr[support < int(min_support)] = 0.95
    return best_thr, best_f1


@torch.no_grad()
def estimate_pos_weight_from_loader(dl, num_classes, max_batches=400, eps=1e-6):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    n = 0
    for i, (_, y, _) in enumerate(dl):
        pos += y.sum(dim=0).double()
        n += y.shape[0]
        if i + 1 >= max_batches:
            break
    neg = n - pos
    return (neg / (pos + eps)).float(), (pos / max(n, 1)).float()


def build_lsnn(n_in: int, n_out: int, args) -> LSNN:
    return LSNN(
        n_in=n_in,
        n_regular=args.n_regular,
        n_adaptive=args.n_adaptive,
        n_out=n_out,
        tau_out=args.tau_out,
        dt=1.0,
        beta=args.beta,
        tau_m=args.tau_m,
        tau_a=args.tau_a,
        thr=args.thr,
        dampening_factor=args.dampening_factor,
        n_refractory=args.n_refractory,
        rec=not args.no_rec,
    )


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def get_checkpoint_classes(ckpt):
    if not isinstance(ckpt, dict):
        return None

    for key in ("classes", "class_names"):
        value = ckpt.get(key)
        if isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
            return list(value)

    return None


def remap_checkpoint_output_order(state, source_classes, target_classes):
    if source_classes == target_classes:
        return state

    if sorted(source_classes) != sorted(target_classes):
        print(
            "Checkpoint class names do not match target classes; "
            "leaving output tensors unchanged."
        )
        print("Checkpoint classes:", source_classes)
        print("Target classes:", target_classes)
        return state

    index = [source_classes.index(cls) for cls in target_classes]
    remapped = dict(state)

    for key in ("w_out", "B"):
        value = remapped.get(key)
        if value is not None and value.ndim == 2 and value.shape[1] == len(index):
            remapped[key] = value[:, index].clone()

    value = remapped.get("b_out")
    if value is not None and value.ndim == 1 and value.shape[0] == len(index):
        remapped["b_out"] = value[index].clone()

    print("Remapped checkpoint output order:")
    print("  from:", source_classes)
    print("  to:  ", target_classes)
    return remapped


def load_pretrained_snn(
    model: LSNN,
    checkpoint_path: str,
    device,
    strict: bool = True,
    remap_classes: bool = True,
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = extract_state_dict(ckpt)
    source_classes = get_checkpoint_classes(ckpt) if remap_classes else None

    if remap_classes and source_classes is None and model.n_out == len(GROOVE_PRETRAIN_CLASS_ORDER):
        source_classes = GROOVE_PRETRAIN_CLASS_ORDER
        print(
            "Checkpoint has no class metadata; assuming legacy GROOVE "
            "pretrain class order."
        )

    if source_classes is not None:
        state = remap_checkpoint_output_order(
            state=state,
            source_classes=source_classes,
            target_classes=CLASSES,
        )

    if strict:
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint strictly: {checkpoint_path}")
        return ckpt

    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)
    print(f"Loaded compatible tensors: {len(compatible)}/{len(model_state)}")
    missing = sorted(set(model_state.keys()) - set(compatible.keys()))
    if missing:
        print("Skipped incompatible/missing tensors:", missing)
    return ckpt


def make_optimizer(model: LSNN, args, readout_only: bool):
    if readout_only:
        groups = [
            {"params": [model.w_out, model.b_out, model.B], "lr": args.lr_readout, "weight_decay": args.weight_decay_out},
        ]
    else:
        rec_params = [model.alif.w_in]
        if model.alif.w_rec is not None:
            rec_params.append(model.alif.w_rec)
        groups = [
            {"params": rec_params, "lr": args.lr_rec, "weight_decay": args.weight_decay_rec},
            {"params": [model.w_out, model.b_out, model.B], "lr": args.lr_readout, "weight_decay": args.weight_decay_out},
        ]
    return torch.optim.AdamW(groups, eps=1e-5)


def train_one_epoch(model, loader, optimizer, device, pos_weight, args, threshold=0.5, log_every=50):
    model.train()
    total_loss = total_pred = total_reg = 0.0
    all_logits, all_targets = [], []
    fr_avg_sum = fr_max_sum = 0.0
    n_batches = 0

    for batch_idx, (features, labels, _) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x_seq = prepare_window_batch(
            features,
            repeat_steps=args.snn_repeat_steps,
            input_encoding=args.input_encoding,
            poisson_steps=args.poisson_steps,
            poisson_max_rate=args.poisson_max_rate,
            poisson_silence_steps=args.poisson_silence_steps,
            poisson_collapse=args.poisson_collapse,
            poisson_normalize=args.poisson_normalize,
        )

        out = model.eprop_update_window(
            x_seq=x_seq,
            targets=labels,
            optimizer=optimizer,
            pos_weight=pos_weight,
            tol_steps=args.tol_frames,
            repeat_steps=args.snn_repeat_steps,
            reg_rate=args.reg_rate,
            reg_voltage=args.reg_voltage,
            f_target_hz=args.f_target_hz,
            dt_seconds=1e-3,
            homeo_lr=args.homeo_lr,
        )

        bs = features.shape[0]
        total_loss += out["loss"] * bs
        total_pred += out["loss_pred"] * bs
        total_reg += out["loss_reg"] * bs
        all_logits.append(out["logits_window"].detach().cpu())
        all_targets.append(labels.detach().cpu())
        fr_avg_sum += out["spike_rate_hz"]
        fr_max_sum += out.get("spike_rate_max_hz", out["spike_rate_hz"])
        n_batches += 1

        if log_every > 0 and batch_idx % log_every == 0:
            tmp_stats = f1_stats_from_logits(torch.cat(all_logits), torch.cat(all_targets), thr=threshold)
            print(
                f"  batch {batch_idx:4d}/{len(loader)-1:4d} | "
                f"loss={out['loss']:.4f} | pred={out['loss_pred']:.4f} | reg={out['loss_reg']:.4f} | "
                f"microF1={tmp_stats['micro_f1']:.4f} | macroF1={tmp_stats['macro_f1']:.4f} | "
                f"fr_avg={out['spike_rate_hz']:.1f}Hz | fr_max={out.get('spike_rate_max_hz', 0.0):.1f}Hz"
            )

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    stats = f1_stats_from_logits(logits_all, targets_all, thr=threshold)
    n = max(len(loader.dataset), 1)
    stats.update({
        "loss": total_loss / n,
        "loss_pred": total_pred / n,
        "loss_reg": total_reg / n,
        "fr_avg": fr_avg_sum / max(n_batches, 1),
        "fr_max": fr_max_sum / max(n_batches, 1),
    })
    return stats


@torch.no_grad()
def evaluate(model, loader, device, pos_weight, args, threshold=0.5, return_logits=False):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x_seq = prepare_window_batch(
            features,
            repeat_steps=args.snn_repeat_steps,
            input_encoding=args.input_encoding,
            poisson_steps=args.poisson_steps,
            poisson_max_rate=args.poisson_max_rate,
            poisson_silence_steps=args.poisson_silence_steps,
            poisson_collapse=args.poisson_collapse,
            poisson_normalize=args.poisson_normalize,
        )
        T, B, _ = x_seq.shape
        C = model.n_out
        kappa = model.kappa.to(device)

        state = model.alif.zero_state(B, device=device)
        y_prev = torch.zeros(B, C, device=device)

        tol_ms = args.tol_frames * args.snn_repeat_steps
        center = T // 2
        lo = max(0, center - tol_ms)
        hi = min(T, center + tol_ms + 1)
        n_sup = max(1, hi - lo)
        y_sum = torch.zeros(B, C, device=device)

        for t in range(T):
            state, _ = model.alif(x_seq[t], state)
            y_prev = kappa * y_prev + state["z"] @ model.w_out + model.b_out
            if lo <= t < hi:
                y_sum += y_prev

        logits = y_sum / n_sup
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight.to(device) if pos_weight is not None else None,
            reduction="mean",
        )
        total_loss += float(loss) * features.shape[0]
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    stats = f1_stats_from_logits(logits_all, targets_all, thr=threshold)
    stats["loss"] = total_loss / max(len(loader.dataset), 1)
    if return_logits:
        return stats, logits_all, targets_all
    return stats


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
    parser.add_argument("--checkpoint", default="models/snn/runs/snnrun8FULLTRAINING/best_groove_window.pt")
    parser.add_argument("--out_dir", default="models/snn/runs/finetune_rwc_mfcc")

    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_seconds", type=float, default=1.0)
    parser.add_argument("--tol_frames", type=int, default=3)
    parser.add_argument("--label_mode", choices=["center", "window"], default="center")
    parser.add_argument("--add_deltas", action="store_true", default=True)
    parser.add_argument("--no_deltas", dest="add_deltas", action="store_false")
    parser.add_argument("--norm", choices=["none", "per_window", "per_track", "global_per_window"], default="per_window")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--frozen_epochs", type=int, default=3, help="Train only readout/B for these first epochs.")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train_windows_per_track", type=int, default=16)
    parser.add_argument("--eval_stride_frames", type=int, default=43)
    parser.add_argument("--p_pos", type=float, default=0.75)
    parser.add_argument("--append_metrics", action="store_true", help="Append to an existing metrics.jsonl instead of archiving it.")

    parser.add_argument("--repeat_steps", type=int, default=5)
    parser.add_argument("--input_encoding", choices=["analog", "poisson"], default="analog")
    parser.add_argument("--poisson_steps", type=int, default=None)
    parser.add_argument("--poisson_max_rate", type=float, default=0.30)
    parser.add_argument("--poisson_silence_steps", type=int, default=0)
    parser.add_argument("--poisson_collapse", choices=["none", "mean", "any"], default="none")
    parser.add_argument("--poisson_normalize", choices=["window", "feature", "sigmoid", "clamp"], default="window")
    parser.add_argument("--reg_rate", type=float, default=10.0)
    parser.add_argument("--reg_voltage", type=float, default=1e-4)
    parser.add_argument("--f_target_hz", type=float, default=15.0)
    parser.add_argument("--homeo_lr", type=float, default=1e-4)

    parser.add_argument("--lr_rec", type=float, default=1e-4)
    parser.add_argument("--lr_readout", type=float, default=3e-4)
    parser.add_argument("--weight_decay_rec", type=float, default=1e-5)
    parser.add_argument("--weight_decay_out", type=float, default=1e-4)
    parser.add_argument("--min_threshold", type=float, default=0.20)
    parser.add_argument("--low_support_threshold", type=int, default=5)

    parser.add_argument("--n_regular", type=int, default=192)
    parser.add_argument("--n_adaptive", type=int, default=64)
    parser.add_argument("--tau_out", type=float, default=20.0)
    parser.add_argument("--beta", type=float, default=0.184)
    parser.add_argument("--tau_m", type=float, default=20.0)
    parser.add_argument("--tau_a", type=float, default=200.0)
    parser.add_argument("--thr", type=float, default=1.4)
    parser.add_argument("--dampening_factor", type=float, default=0.3)
    parser.add_argument("--n_refractory", type=int, default=3)
    parser.add_argument("--no_rec", action="store_true")

    parser.add_argument("--non_strict_load", action="store_true", help="Load only matching checkpoint tensors.")
    parser.add_argument(
        "--no_checkpoint_class_remap",
        action="store_true",
        help="Disable output-column remapping when loading the pretrained checkpoint.",
    )
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()
    if args.poisson_steps is None:
        args.poisson_steps = args.repeat_steps
    args.input_encoding = normalize_input_encoding(args.input_encoding)
    args.snn_repeat_steps = effective_repeat_steps(
        repeat_steps=args.repeat_steps,
        encoding=args.input_encoding,
        poisson_steps=args.poisson_steps,
        poisson_silence_steps=args.poisson_silence_steps,
        poisson_collapse=args.poisson_collapse,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.append_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_metrics_path = out_dir / f"metrics_{timestamp}.jsonl"
        metrics_path.rename(archived_metrics_path)
        print(f"Archived previous metrics: {archived_metrics_path}")

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": CLASSES,
                "legacy_groove_pretrain_class_order": GROOVE_PRETRAIN_CLASS_ORDER,
                "args": vars(args),
            },
            f,
            indent=2,
            default=json_safe,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        "input_encoding:", args.input_encoding,
        "| repeat_steps:", args.repeat_steps,
        "| poisson_steps:", args.poisson_steps,
        "| poisson_max_rate:", args.poisson_max_rate,
        "| poisson_collapse:", args.poisson_collapse,
        "| poisson_normalize:", args.poisson_normalize,
        "| effective_repeat_steps:", args.snn_repeat_steps,
    )

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
        add_deltas=args.add_deltas,
    )
    val_ds = RWCMFCCWindowDataset(
        index_jsonl=args.index_path,
        ids=val_ids,
        cfg=cfg,
        sampling="all",
        stride_frames=args.eval_stride_frames,
        seed=123,
        norm=args.norm,
        add_deltas=args.add_deltas,
    )
    test_ds = RWCMFCCWindowDataset(
        index_jsonl=args.index_path,
        ids=test_ids,
        cfg=cfg,
        sampling="all",
        stride_frames=args.eval_stride_frames,
        seed=456,
        norm=args.norm,
        add_deltas=args.add_deltas,
    )

    pin_memory = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=pin_memory, collate_fn=snn_window_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=pin_memory, collate_fn=snn_window_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         pin_memory=pin_memory, collate_fn=snn_window_collate_fn)

    pw_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                           pin_memory=pin_memory, collate_fn=snn_window_collate_fn)

    n_in = train_ds[0][0].shape[1]
    n_out = len(CLASSES)
    model = build_lsnn(n_in=n_in, n_out=n_out, args=args).to(device)

    source_ckpt = load_pretrained_snn(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        strict=not args.non_strict_load,
        remap_classes=not args.no_checkpoint_class_remap,
    )

    pos_weight, pos_rate = estimate_pos_weight_from_loader(pw_loader, num_classes=n_out)
    pos_weight = torch.sqrt(pos_weight).clamp(max=8.0).to(device)
    print("Estimated pos_rate:", {CLASSES[i]: float(pos_rate[i]) for i in range(n_out)})
    print("Using pos_weight:", {CLASSES[i]: float(pos_weight[i].cpu()) for i in range(n_out)})

    best_val_macro_f1 = -1.0
    best_thresholds = torch.full((n_out,), 0.5, dtype=torch.float32)
    threshold_grid = torch.linspace(
        args.min_threshold,
        0.95,
        int(round((0.95 - args.min_threshold) / 0.05)) + 1,
    )
    bad_epochs = 0
    optimizer = None
    scheduler = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        readout_only = epoch <= args.frozen_epochs

        if optimizer is None or epoch == args.frozen_epochs + 1:
            if readout_only:
                print("Training readout/B only before full SNN fine-tuning.")
            else:
                print("Fine-tuning full LSNN.")
            optimizer = make_optimizer(model, args, readout_only=readout_only)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3, threshold=1e-3
            )

        # Same idea as your SNN training curriculum, but starting from the requested p_pos.
        if epoch <= 5:
            train_ds.p_pos = max(args.p_pos, 0.85)
        elif epoch <= 15:
            train_ds.p_pos = args.p_pos
        else:
            train_ds.p_pos = max(0.4, args.p_pos * 0.75)

        print(f"\n=== Epoch {epoch:03d} | p_pos={train_ds.p_pos:.2f} ===")
        train_stats = train_one_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            args=args,
            threshold=best_thresholds,
            log_every=args.log_every,
        )

        val_raw, logits_val, targets_val = evaluate(
            model=model,
            loader=val_dl,
            device=device,
            pos_weight=pos_weight,
            args=args,
            threshold=0.5,
            return_logits=True,
        )
        thr_c, _ = find_best_threshold_per_class(
            logits_val,
            targets_val,
            thresholds=threshold_grid,
            min_support=args.low_support_threshold,
        )
        val_stats = f1_stats_from_logits(logits_val, targets_val, thr=thr_c)
        val_stats["loss"] = val_raw["loss"]

        scheduler.step(val_stats["macro_f1"])
        epoch_time = time.time() - epoch_start
        lr_values = [float(g["lr"]) for g in optimizer.param_groups]

        metrics_row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "readout_only": readout_only,
            "lr": lr_values,
            "train": train_stats,
            "val": val_stats,
            "threshold_by_class": {CLASSES[i]: float(thr_c[i]) for i in range(n_out)},
            "classes": CLASSES,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row, default=json_safe) + "\n")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | train_microF1={train_stats['micro_f1']:.4f} | "
            f"train_macroF1={train_stats['macro_f1']:.4f} | val_loss={val_stats['loss']:.4f} | "
            f"val_microF1={val_stats['micro_f1']:.4f} | val_macroF1={val_stats['macro_f1']:.4f} | "
            f"thr_mean={float(thr_c.mean()):.2f} | fr_avg={train_stats['fr_avg']:.1f}Hz | "
            f"time={epoch_time:.1f}s"
        )

        if val_stats["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_stats["macro_f1"]
            best_thresholds = thr_c.clone()
            bad_epochs = 0

            test_stats = evaluate(
                model=model,
                loader=test_dl,
                device=device,
                pos_weight=pos_weight,
                args=args,
                threshold=best_thresholds,
            )
            test_stats["epoch"] = epoch
            test_stats["classes"] = CLASSES
            test_stats["threshold_by_class"] = {CLASSES[i]: float(best_thresholds[i]) for i in range(n_out)}
            test_stats["threshold_per_class"] = [float(x) for x in best_thresholds.tolist()]
            test_stats["threshold_mean"] = float(best_thresholds.mean())
            test_stats["min_threshold"] = args.min_threshold
            test_stats["low_support_threshold"] = args.low_support_threshold

            ckpt = {
                "epoch": epoch,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "classes": CLASSES,
                "cfg": cfg.__dict__,
                "source_checkpoint": args.checkpoint,
                "feature_type": "mfcc",
                "add_deltas": args.add_deltas,
                "normalization": args.norm,
                "input_encoding": args.input_encoding,
                "repeat_steps": args.repeat_steps,
                "effective_repeat_steps": args.snn_repeat_steps,
                "poisson_steps": args.poisson_steps,
                "poisson_max_rate": args.poisson_max_rate,
                "poisson_silence_steps": args.poisson_silence_steps,
                "poisson_collapse": args.poisson_collapse,
                "poisson_normalize": args.poisson_normalize,
                "threshold_by_class": {CLASSES[i]: float(best_thresholds[i]) for i in range(n_out)},
                "threshold_per_class": [float(x) for x in best_thresholds.tolist()],
                "val_macroF1": val_stats["macro_f1"],
                "val_microF1": val_stats["micro_f1"],
                "test_macroF1": test_stats["macro_f1"],
                "test_microF1": test_stats["micro_f1"],
                "test_stats": test_stats,
                "min_threshold": args.min_threshold,
                "low_support_threshold": args.low_support_threshold,
                "n_in": n_in,
                "n_out": n_out,
                "lsnn_args": {
                    "n_regular": args.n_regular,
                    "n_adaptive": args.n_adaptive,
                    "tau_out": args.tau_out,
                    "beta": args.beta,
                    "tau_m": args.tau_m,
                    "tau_a": args.tau_a,
                    "thr": args.thr,
                    "dampening_factor": args.dampening_factor,
                    "n_refractory": args.n_refractory,
                    "rec": not args.no_rec,
                },
                "total_time_sec": time.time() - start_time,
            }
            torch.save(ckpt, out_dir / "best.pt")
            with (out_dir / "best_test_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(test_stats, f, indent=2, default=json_safe)
            print(
                f"Saved best.pt | val_macroF1={val_stats['macro_f1']:.4f} | "
                f"test_macroF1={test_stats['macro_f1']:.4f} | test_microF1={test_stats['micro_f1']:.4f}"
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
