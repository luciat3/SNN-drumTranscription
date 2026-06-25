# -*- coding: utf-8 -*-

"""
Generates onset prediction from a WAV file using either the CNN or SNN model, 
and saves the predictions to a json file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Union, Optional

import librosa
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CNN
from models.cnn.model import DrumCNN
from models.cnn.dataset import SpecConfig, compute_log_mel, CLASSES

# SNN
from models.snn.model import LSNN as DrumSNN
from models.snn.dataset_GROOVE import compute_delta

from scripts.groove_processing import compute_mfcc, MFCCConfig


EXPECTED_CLASSES = [
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

THRESHOLD_PROFILES = {
    "balanced": {
        "kick": 0.88,
        "xstick": 0.80,
        "snare": 0.86,
        "hihat_pedal": 0.72,
        "hihat_closed": 0.62,
        "hihat_open": 0.80,
        "tom": 0.80,
        "floor_tom": 0.80,
        "vibraslap": 0.80,
        "crash": 0.88,
        "ride_bow": 0.84,
        "ride_bell": 0.84,
        "chinese_cymbal": 0.90,
        "splash_cymbal": 0.90,
    },
    "sensitive": {
        "kick": 0.82,
        "xstick": 0.72,
        "snare": 0.78,
        "hihat_pedal": 0.65,
        "hihat_closed": 0.68,
        "hihat_open": 0.72,
        "tom": 0.72,
        "floor_tom": 0.74,
        "vibraslap": 0.72,
        "crash": 0.78,
        "ride_bow": 0.76,
        "ride_bell": 0.76,
        "chinese_cymbal": 0.80,
        "splash_cymbal": 0.80,
    },
}


# Minimum distance between onsets of the same class, in milliseconds.
MIN_DIST_BY_CLASS_MS = {
    "kick": 120.0,
    "snare": 180.0,
    "xstick": 180.0,
    "hihat_pedal": 90.0,
    "hihat_closed": 90.0,
    "hihat_open": 120.0,
    "tom": 160.0,
    "floor_tom": 160.0,
    "vibraslap": 200.0,
    "crash": 250.0,
    "ride_bow": 90.0,
    "ride_bell": 120.0,
    "chinese_cymbal": 250.0,
    "splash_cymbal": 250.0,
}


# Groups of instruments that can't be played simultaneously (different parts of a same instrument),
# so we can suppress duplicates in a small time window.
DEFAULT_SUPPRESSION_GROUPS = [
    ["hihat_pedal", "hihat_closed", "hihat_open"],
    ["ride_bow", "ride_bell"],
]


def load_drum_map(drum_map_path: str) -> Dict[str, int]:
    data = json.loads(Path(drum_map_path).read_text(encoding="utf-8"))

    out = {}

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, int):
                out[k] = int(v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], int):
                out[k] = int(v[0])

    return out


def load_mfcc_normalization(processed_root: str, n_mfcc: int):
    summary_path = Path(processed_root) / "preprocessing_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"No encuentro {summary_path}. "
            "Necesito preprocessing_summary.json para usar la misma normalización que entrenamiento."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    mean = np.asarray(summary["mfcc_mean"], dtype=np.float32)
    std = np.asarray(summary["mfcc_std"], dtype=np.float32)

    if mean.shape[0] != n_mfcc:
        raise ValueError(f"mfcc_mean tiene {mean.shape[0]} valores, esperaba {n_mfcc}")

    if std.shape[0] != n_mfcc:
        raise ValueError(f"mfcc_std tiene {std.shape[0]} valores, esperaba {n_mfcc}")

    return mean, std


def mfcc_normalization_from_cfg(cfg_dict: dict, n_mfcc: int):
    mean = cfg_dict.get("mfcc_mean")
    std = cfg_dict.get("mfcc_std")

    if mean is None or std is None:
        return None

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.shape[0] != n_mfcc:
        raise ValueError(f"mfcc_mean tiene {mean.shape[0]} valores, esperaba {n_mfcc}")

    if std.shape[0] != n_mfcc:
        raise ValueError(f"mfcc_std tiene {std.shape[0]} valores, esperaba {n_mfcc}")

    return mean, std


def normalize_mfcc_like_training(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    X = [n_mfcc, T].
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D MFCC array, got {X.shape}")

    if X.shape[0] != mean.shape[0]:
        raise ValueError(
            "Para normalizar espero X=[n_mfcc,T]. "
            f"X shape={X.shape}, mean shape={mean.shape}"
        )

    return ((X - mean[:, None]) / (std[:, None] + 1e-8)).astype(np.float32)


def ensure_time_first(X: np.ndarray, n_features: int) -> np.ndarray:
    """
    [F,T] to [T,F]
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.shape}")

    if X.shape[0] == n_features and X.shape[1] != n_features:
        X = X.T

    return X.astype(np.float32)


def build_windows_from_spec(X: np.ndarray, win_frames: int) -> np.ndarray:
    """
    CNN.
    X: [M, T]
    returns: [T, M, W]
    """
    M, T = X.shape
    pad = win_frames // 2
    Xp = np.pad(X, ((0, 0), (pad, pad)), mode="reflect")

    windows = np.empty((T, M, win_frames), dtype=np.float32)

    for t in range(T):
        windows[t] = Xp[:, t : t + win_frames]

    return windows


def normalize_windows(windows: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("", "none", None):
        return windows.astype(np.float32)

    if mode == "per_window":
        mean = windows.mean(axis=(1, 2), keepdims=True)
        std = windows.std(axis=(1, 2), keepdims=True)
        return ((windows - mean) / (std + 1e-6)).astype(np.float32)

    if mode == "per_feature_window":
        mean = windows.mean(axis=2, keepdims=True)
        std = windows.std(axis=2, keepdims=True)
        return ((windows - mean) / (std + 1e-6)).astype(np.float32)

    raise ValueError(f"Normalización CNN-MFCC no soportada en inferencia: {mode}")


def build_centered_windows_snn(X: np.ndarray, win_frames: int) -> np.ndarray:
    """
    SNN.
    X: [T, F]
    returns: [T, W, F]
    """
    T, F = X.shape
    pad = win_frames // 2

    Xp = np.pad(X, ((pad, pad), (0, 0)), mode="reflect")

    windows = np.empty((T, win_frames, F), dtype=np.float32)

    for t in range(T):
        windows[t] = Xp[t : t + win_frames]

    return windows


def add_deltas_like_training(xw: np.ndarray) -> np.ndarray:
    """
    [MFCC] -> [MFCC, delta, delta-delta]
    """
    dx = compute_delta(xw)
    ddx = compute_delta(dx)

    return np.concatenate([xw, dx, ddx], axis=1).astype(np.float32)


def smooth_probs(probs: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    if kernel_size <= 1:
        return probs

    if kernel_size % 2 == 0:
        raise ValueError("--smooth_kernel debe ser impar: 1, 3, 5, ...")

    pad = kernel_size // 2
    padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(probs)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)

    for c in range(probs.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")

    return out.astype(np.float32)


def predict_frame_probs_cnn(
    model: DrumCNN,
    X: np.ndarray,
    cfg,
    device: str,
    class_names: List[str],
    batch_size: int = 128,
    window_norm: str = "none",
) -> np.ndarray:
    model.eval()

    win_frames = int(round(cfg.win_seconds * cfg.sr / cfg.hop_length))
    windows = build_windows_from_spec(X, win_frames)
    windows = normalize_windows(windows, window_norm)

    T = windows.shape[0]
    C = len(class_names)

    probs = np.zeros((T, C), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, T, batch_size):
            chunk = windows[i : i + batch_size]
            xb = torch.from_numpy(chunk).unsqueeze(1).to(device)

            logits = model(xb)

            if logits.shape[1] != C:
                raise ValueError(
                    f"Modelo devuelve {logits.shape[1]} clases, "
                    f"pero class_names tiene {C}: {class_names}"
                )

            pb = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

            if i == 0:
                print("\n=== CNN FIRST BATCH DEBUG ===")
                print("xb shape:", tuple(xb.shape))
                print("logits shape:", tuple(logits.shape))
                print("logits min:", float(logits.min()))
                print("logits max:", float(logits.max()))
                print("logits mean:", float(logits.mean()))
                print("probs min:", float(torch.sigmoid(logits).min()))
                print("probs max:", float(torch.sigmoid(logits).max()))
                print("probs mean:", float(torch.sigmoid(logits).mean()))

            probs[i : i + batch_size] = pb

    return probs


def predict_frame_probs_snn(
    model: DrumSNN,
    X: np.ndarray,
    cfg: MFCCConfig,
    device: str,
    n_in: int,
    batch_size: int = 64,
    repeat_steps: int = 5,
    tol_steps: int = 5,
) -> np.ndarray:
    model.eval()

    win_frames = max(1, int(round(cfg.win_seconds * cfg.sr / cfg.hop_length)))

    n_mfcc = n_in // 3
    X = ensure_time_first(X, n_features=n_mfcc)  # [T, n_mfcc]

    windows = build_centered_windows_snn(X, win_frames)  # [T, W, n_mfcc]

    T_total = windows.shape[0]
    C = model.n_out

    probs = np.zeros((T_total, C), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, T_total, batch_size):
            chunk = windows[i : i + batch_size]

            chunk_delta = np.stack(
                [add_deltas_like_training(w) for w in chunk],
                axis=0,
            )

            if chunk_delta.shape[-1] != n_in:
                raise ValueError(
                    f"n_in mismatch: checkpoint n_in={n_in}, "
                    f"but prediction features have {chunk_delta.shape[-1]}"
                )

            features = torch.from_numpy(chunk_delta).to(device)  # [B, W, n_in]

            x_seq = features.transpose(0, 1).contiguous()  # [W, B, n_in]

            if repeat_steps > 1:
                x_seq = x_seq.repeat_interleave(repeat_steps, dim=0)

            T_snn, B, _ = x_seq.shape
            C = model.n_out

            kappa = model.kappa.to(device)
            state = model.alif.zero_state(B, device=device)

            y_prev = torch.zeros(B, C, device=device)
            y_sum = torch.zeros(B, C, device=device)

            tol_ms = tol_steps * repeat_steps
            center = T_snn // 2
            lo = max(0, center - tol_ms)
            hi = min(T_snn, center + tol_ms + 1)
            n_sup = max(1, hi - lo)

            for t in range(T_snn):
                state, _ = model.alif(x_seq[t], state)
                y_prev = kappa * y_prev + state["z"] @ model.w_out + model.b_out

                if lo <= t < hi:
                    y_sum += y_prev

            logits = y_sum / n_sup
            pb = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

            if i == 0:
                print("\n=== FIRST BATCH DEBUG ===")
                print("logits shape:", logits.shape)
                print("logits min:", float(logits.min()))
                print("logits max:", float(logits.max()))
                print("logits mean:", float(logits.mean()))
                print("probs min:", float(torch.sigmoid(logits).min()))
                print("probs max:", float(torch.sigmoid(logits).max()))
                print("probs mean:", float(torch.sigmoid(logits).mean()))

            probs[i : i + batch_size] = pb

    return probs


def peak_pick_1d(
    p: np.ndarray,
    thr: float,
    min_dist: int,
) -> List[int]:
    T = p.shape[0]
    peaks: List[int] = []

    for t in range(1, T - 1):
        if p[t] <= thr:
            continue

        if not (p[t] >= p[t - 1] and p[t] >= p[t + 1]):
            continue

        if peaks and (t - peaks[-1]) < min_dist:
            if p[t] > p[peaks[-1]]:
                peaks[-1] = t
            continue

        peaks.append(t)

    return peaks


def thresholds_from_dict(class_names: List[str], threshold_dict: Dict[str, float]) -> np.ndarray:
    missing = [c for c in class_names if c not in threshold_dict]
    if missing:
        raise ValueError(f"Faltan thresholds para clases: {missing}")

    return np.asarray([threshold_dict[c] for c in class_names], dtype=np.float32)


def load_thresholds_json(path: str, class_names: List[str]) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("--thresholds_json debe ser un diccionario {class_name: threshold}")

    return thresholds_from_dict(class_names, {k: float(v) for k, v in data.items()})


def checkpoint_thresholds(ckpt: dict, class_names: List[str]) -> Optional[np.ndarray]:
    by_class = ckpt.get("threshold_by_class")
    if isinstance(by_class, dict):
        missing = [c for c in class_names if c not in by_class]
        if missing:
            raise ValueError(f"Checkpoint threshold_by_class no contiene clases: {missing}")
        return np.asarray([float(by_class[c]) for c in class_names], dtype=np.float32)

    per_class = ckpt.get("threshold_per_class")
    if per_class is not None:
        thr = np.asarray(per_class, dtype=np.float32)
        if thr.shape[0] != len(class_names):
            raise ValueError(
                f"Checkpoint threshold_per_class has {thr.shape[0]} values, "
                f"but model has {len(class_names)} classes."
            )
        return thr

    return None


def apply_adaptive_thresholds(
    probs: np.ndarray,
    base_thr: np.ndarray,
    class_names: List[str],
    quantile: float = 0.995,
    ratio: float = 0.80,
    max_thr: float = 0.97,
) -> np.ndarray:
    out = base_thr.astype(np.float32).copy()

    q = np.quantile(probs, quantile, axis=0).astype(np.float32)
    adaptive = np.minimum(q * float(ratio), float(max_thr))
    out = np.maximum(out, adaptive)

    print("\nAdaptive thresholds:")
    for i, name in enumerate(class_names):
        print(
            f"  {i:02d} {name:15s}: "
            f"base={base_thr[i]:.4f} q{quantile:.3f}={q[i]:.4f} final={out[i]:.4f}"
        )

    return out


def resolve_thresholds(
    probs: np.ndarray,
    ckpt: dict,
    args,
    class_names: List[str],
    model_type: str,
) -> Union[float, np.ndarray]:

    n_classes = len(class_names)

    if args.thr is not None:
        print(f"\nUsing CLI global threshold: {args.thr}")
        return float(args.thr)

    if args.thresholds_json is not None:
        thr = load_thresholds_json(args.thresholds_json, class_names)
        print(f"\nUsing thresholds JSON: {args.thresholds_json}")
    elif args.use_ckpt_thresholds:
        thr = checkpoint_thresholds(ckpt, class_names)
        if thr is None:
            raise ValueError(
                "Has usado --use_ckpt_thresholds, pero el checkpoint no tiene "
                "threshold_by_class ni threshold_per_class"
            )
        print("\nUsing checkpoint per-class thresholds")
    elif args.threshold_profile == "auto":
        ckpt_thr = checkpoint_thresholds(ckpt, class_names)
        if model_type == "cnn" and ckpt_thr is not None:
            thr = ckpt_thr
            print("\nUsing CNN checkpoint per-class thresholds (auto)")
        elif model_type == "cnn":
            thr = 0.5
            print("\nUsing CNN default global threshold: 0.5 (auto)")
        else:
            thr = thresholds_from_dict(class_names, THRESHOLD_PROFILES["money_beat"])
            print("\nUsing SNN inference threshold profile: money_beat (auto)")
    else:
        if args.threshold_profile not in THRESHOLD_PROFILES:
            raise ValueError(
                f"Perfil desconocido: {args.threshold_profile}. "
                f"Disponibles: {['auto'] + list(THRESHOLD_PROFILES.keys())}"
            )
        thr = thresholds_from_dict(class_names, THRESHOLD_PROFILES[args.threshold_profile])
        print(f"\nUsing inference threshold profile: {args.threshold_profile}")

    if not isinstance(thr, np.ndarray):
        return thr

    if args.adaptive_thresholds:
        thr = apply_adaptive_thresholds(
            probs=probs,
            base_thr=thr,
            class_names=class_names,
            quantile=args.adaptive_quantile,
            ratio=args.adaptive_ratio,
            max_thr=args.adaptive_max,
        )

    print("\nFinal per-class thresholds:")
    for i, v in enumerate(thr):
        print(f"  {i:02d} {class_names[i]:15s}: {v:.4f}")

    return thr


def build_min_dist_by_class(
    class_names: List[str],
    default_min_dist_ms: float,
    snare_min_dist_ms: Optional[float],
    kick_min_dist_ms: Optional[float],
    hihat_min_dist_ms: Optional[float],
) -> Dict[str, float]:
    out = {c: float(default_min_dist_ms) for c in class_names}

    for c, v in MIN_DIST_BY_CLASS_MS.items():
        if c in out:
            out[c] = float(v)

    if snare_min_dist_ms is not None and "snare" in out:
        out["snare"] = float(snare_min_dist_ms)

    if kick_min_dist_ms is not None and "kick" in out:
        out["kick"] = float(kick_min_dist_ms)

    if hihat_min_dist_ms is not None:
        for c in ("hihat_closed", "hihat_open", "hihat_pedal"):
            if c in out:
                out[c] = float(hihat_min_dist_ms)

    return out


def probs_to_events(
    probs: np.ndarray,
    cfg,
    class_names: List[str],
    thr: Union[float, np.ndarray, List[float]],
    min_dist_ms: float = 60.0,
    min_dist_by_class_ms: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    T, C = probs.shape

    if isinstance(thr, (list, tuple, np.ndarray)):
        thr_arr = np.asarray(thr, dtype=np.float32)

        if thr_arr.shape[0] != C:
            raise ValueError(
                f"threshold_per_class has {thr_arr.shape[0]} values, expected {C}"
            )
    else:
        thr_arr = np.full((C,), float(thr), dtype=np.float32)

    events: List[Dict] = []

    for ci, cls in enumerate(class_names):
        cls_min_dist_ms = float(min_dist_ms)
        if min_dist_by_class_ms is not None and cls in min_dist_by_class_ms:
            cls_min_dist_ms = float(min_dist_by_class_ms[cls])

        min_dist_frames = max(
            1,
            int(round((cls_min_dist_ms / 1000.0) * cfg.sr / cfg.hop_length)),
        )

        peaks = peak_pick_1d(
            probs[:, ci],
            thr=float(thr_arr[ci]),
            min_dist=min_dist_frames,
        )

        for f in peaks:
            t_sec = (f * cfg.hop_length) / cfg.sr

            events.append(
                {
                    "time_sec": float(t_sec),
                    "frame": int(f),
                    "class": cls,
                    "prob": float(probs[f, ci]),
                    "threshold": float(thr_arr[ci]),
                    "min_dist_ms": float(cls_min_dist_ms),
                }
            )

    events.sort(key=lambda e: e["time_sec"])

    return events


def suppress_same_group_events(
    events: List[Dict],
    group_names: List[str],
    window_ms: float = 50.0,
) -> List[Dict]:
    if not events:
        return events

    group = set(group_names)
    kept: List[Dict] = []
    used = [False] * len(events)

    for i, ev in enumerate(events):
        if used[i]:
            continue

        if ev["class"] not in group:
            kept.append(ev)
            used[i] = True
            continue

        cluster = [i]
        used[i] = True

        for j in range(i + 1, len(events)):
            if used[j]:
                continue

            if events[j]["class"] not in group:
                continue

            dt_ms = abs(events[j]["time_sec"] - ev["time_sec"]) * 1000.0

            if dt_ms <= window_ms:
                cluster.append(j)
                used[j] = True

        best_i = max(cluster, key=lambda idx: events[idx]["prob"])
        kept.append(events[best_i])

    kept.sort(key=lambda e: e["time_sec"])
    return kept


def apply_group_suppression(
    events: List[Dict],
    enabled: bool,
    window_ms: float,
) -> List[Dict]:
    if not enabled:
        return events

    out = events

    before = len(out)
    for group in DEFAULT_SUPPRESSION_GROUPS:
        out = suppress_same_group_events(out, group_names=group, window_ms=window_ms)

    after = len(out)
    print(f"\nGroup suppression: {before} -> {after} events")

    return out


def print_prob_stats(probs: np.ndarray, class_names: List[str], thresholds=None) -> None:
    print("\n=== PROB STATS ===")
    print("shape:", probs.shape)
    print("global min:", float(probs.min()))
    print("global max:", float(probs.max()))
    print("global mean:", float(probs.mean()))

    for ci, cls in enumerate(class_names):
        extra = ""

        if thresholds is not None:
            if isinstance(thresholds, np.ndarray):
                extra = f" thr={thresholds[ci]:.4f}"
            else:
                extra = f" thr={float(thresholds):.4f}"

        print(
            f"{ci:02d} {cls:15s} "
            f"min={probs[:, ci].min():.6f} "
            f"max={probs[:, ci].max():.6f} "
            f"mean={probs[:, ci].mean():.6f} "
            f"p99={np.quantile(probs[:, ci], 0.99):.6f} "
            f"p995={np.quantile(probs[:, ci], 0.995):.6f}"
            f"{extra}"
        )


def print_debug_frames(probs: np.ndarray, class_names: List[str], debug_frames: str) -> None:
    if not debug_frames.strip():
        return

    frames_to_check = [int(x.strip()) for x in debug_frames.split(",") if x.strip()]

    print("\n=== TOP CLASSES AT DEBUG FRAMES ===")

    for f in frames_to_check:
        if f < 0 or f >= probs.shape[0]:
            print(f"\nFrame {f}: fuera de rango")
            continue

        top = np.argsort(probs[f])[::-1][:8]

        print(f"\nFrame {f}:")
        for ci in top:
            print(
                f"  idx={ci:02d} "
                f"class={class_names[ci]:15s} "
                f"prob={probs[f, ci]:.6f}"
            )


def print_checkpoint_debug(ckpt: dict, ckpt_path: str) -> None:
    print("\n=== CHECKPOINT DEBUG ===")
    print("ckpt:", ckpt_path)
    print("epoch:", ckpt.get("epoch"))
    print("n_in:", ckpt.get("n_in"))
    print("n_out:", ckpt.get("n_out"))
    print("class_names:", ckpt.get("class_names"))
    print("threshold_per_class:", ckpt.get("threshold_per_class"))
    print("val_macro_f1:", ckpt.get("val_macro_f1"))
    print("val_micro_f1:", ckpt.get("val_micro_f1"))
    print("test_macro_f1:", ckpt.get("test_macro_f1"))
    print("test_micro_f1:", ckpt.get("test_micro_f1"))


def build_snn_model(n_in: int, n_out: int) -> DrumSNN:
    """
    Arquitectura igual que train_GROOVE.py.
    """
    return DrumSNN(
        n_in=n_in,
        n_regular=192,
        n_adaptive=64,
        n_out=n_out,
        tau_out=20.0,
        dt=1.0,
        beta=0.184,
        tau_m=20.0,
        tau_a=200.0,
        thr=1.4,
        dampening_factor=0.3,
        n_refractory=3,
        rec=True,
    )


def save_json(path: str, payload) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--wav", required=True, help="Ruta al wav de entrada")
    ap.add_argument("--ckpt", required=True, help="Ruta al checkpoint del modelo")

    ap.add_argument(
        "--model_type",
        choices=["cnn", "snn"],
        required=True,
        help="Tipo de modelo: cnn o snn",
    )

    ap.add_argument("--out_events", required=True, help="Salida eventos JSON limpio")
    ap.add_argument("--out_raw_events", default=None, help="Salida JSON antes de supresión de grupos")
    ap.add_argument("--out_probs", default=None, help="Guarda probabilidades framewise como .npy")
    ap.add_argument("--drum_map", default="experiments/drum_map.json")

    # Thresholds.
    ap.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Threshold global manual. Si se pasa, ignora thresholds por clase.",
    )
    ap.add_argument(
        "--threshold_profile",
        choices=["auto"] + list(THRESHOLD_PROFILES.keys()),
        default="auto",
        help=(
            "Perfil de thresholds por clase. auto usa thresholds del checkpoint "
            "en CNN cuando existen."
        ),
    )
    ap.add_argument(
        "--thresholds_json",
        default=None,
        help="JSON opcional con thresholds por clase. Tiene prioridad sobre threshold_profile.",
    )
    ap.add_argument(
        "--use_ckpt_thresholds",
        action="store_true",
        help="Usa threshold_per_class del checkpoint. Normalmente es demasiado permisivo para partitura.",
    )
    ap.add_argument(
        "--adaptive_thresholds",
        action="store_true",
        help="Sube thresholds automáticamente según la distribución de probs del audio.",
    )
    ap.add_argument("--adaptive_quantile", type=float, default=0.995)
    ap.add_argument("--adaptive_ratio", type=float, default=0.80)
    ap.add_argument("--adaptive_max", type=float, default=0.97)

    # Peak picking.
    ap.add_argument("--min_dist_ms", type=float, default=130.0)
    ap.add_argument("--snare_min_dist_ms", type=float, default=180.0)
    ap.add_argument("--kick_min_dist_ms", type=float, default=120.0)
    ap.add_argument("--hihat_min_dist_ms", type=float, default=90.0)
    ap.add_argument("--smooth_kernel", type=int, default=3)

    # Inferencia.
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--repeat_steps", type=int, default=5)
    ap.add_argument("--tol_steps", type=int, default=5)

    ap.add_argument(
        "--processed_root",
        type=str,
        default="data/processed/groove_processed_mfcc",
        help="Carpeta donde está preprocessing_summary.json",
    )

    ap.add_argument(
        "--no_group_suppression",
        action="store_true",
        help="Desactiva supresión de grupos como kick/tom/floor_tom.",
    )
    ap.add_argument("--group_window_ms", type=float, default=50.0)

    ap.add_argument(
        "--wrapped_json",
        action="store_true",
        help="Guarda {meta, events} en lugar de una lista directa.",
    )

    ap.add_argument("--debug_frames", type=str, default="")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Model type:", args.model_type)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    print_checkpoint_debug(ckpt, args.ckpt)

    if args.model_type == "cnn":
        cfg_dict = ckpt.get("cfg", {})

        print("\n=== CNN CHECKPOINT STRUCTURE DEBUG ===")
        print("ckpt keys:", list(ckpt.keys()))
        print("ckpt classes:", ckpt.get("classes"))
        print("ckpt class_names:", ckpt.get("class_names"))
        print("dataset CLASSES:", CLASSES)

        class_names = ckpt.get("classes", None)
        if class_names is None:
            class_names = ckpt.get("class_names", None)
        if class_names is None:
            class_names = CLASSES

        class_names = list(class_names)

        print("class_names used:", class_names)
        print("num class_names:", len(class_names))

        cnn_window_norm = "none"
        is_cnn_mfcc = "n_mfcc" in cfg_dict or ckpt.get("normalization") in {
            "per_window",
            "per_feature_window",
            "per_track",
        }

        if is_cnn_mfcc:
            print("\nDetected CNN checkpoint trained with MFCC features")

            cfg = MFCCConfig(
                sr=cfg_dict.get("sr", 22050),
                n_fft=cfg_dict.get("n_fft", 1024),
                hop_length=cfg_dict.get("hop_length", 256),
                n_mfcc=cfg_dict.get("n_mfcc", 40),
                win_seconds=cfg_dict.get("win_seconds", 1.0),
            )

            _, X, _ = compute_mfcc(args.wav, cfg)
            X = X.astype(np.float32)

            print("\n=== CNN MFCC DEBUG BEFORE NORMALIZATION ===")
            print("X shape:", X.shape)
            print("X min:", float(X.min()))
            print("X max:", float(X.max()))
            print("X mean:", float(X.mean()))
            print("X std:", float(X.std()))

            normalization_mode = ckpt.get("normalization")
            if normalization_mode == "per_track":
                X = ((X - X.mean()) / (X.std() + 1e-6)).astype(np.float32)
                print("\nApplied CNN MFCC per-track normalization")
            elif normalization_mode in ("per_window", "per_feature_window"):
                cnn_window_norm = normalization_mode
                print(f"\nWill apply CNN MFCC {normalization_mode} normalization per window")
            else:
                norm_from_ckpt = mfcc_normalization_from_cfg(cfg_dict, cfg.n_mfcc)
                if norm_from_ckpt is None:
                    norm_from_ckpt = load_mfcc_normalization(
                        processed_root=args.processed_root,
                        n_mfcc=cfg.n_mfcc,
                    )

                mfcc_mean, mfcc_std = norm_from_ckpt
                X = normalize_mfcc_like_training(X, mfcc_mean, mfcc_std)
                print("\nApplied CNN MFCC train-set normalization")

            print("\n=== CNN MFCC DEBUG AFTER NORMALIZATION SETUP ===")
            print("X shape:", X.shape)
            print("X min:", float(X.min()))
            print("X max:", float(X.max()))
            print("X mean:", float(X.mean()))
            print("X std:", float(X.std()))
            print("per-coeff mean first 5:", X.mean(axis=1)[:5])
            print("per-coeff std first 5:", X.std(axis=1)[:5])

        else:
            print("\nDetected CNN checkpoint trained with log-mel features")

            allowed_spec_keys = {
                "sr",
                "n_mels",
                "n_fft",
                "hop_length",
                "fmin",
                "fmax",
                "win_seconds",
            }

            cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed_spec_keys}
            cfg = SpecConfig(**cfg_dict) if cfg_dict else SpecConfig(win_seconds=1.0)

            y_audio, _ = librosa.load(args.wav, sr=cfg.sr, mono=True)
            X = compute_log_mel(y_audio, cfg).astype(np.float32)

            print("\n=== CNN LOG-MEL INPUT DEBUG ===")
            print("X shape:", X.shape)
            print("X min:", float(X.min()))
            print("X max:", float(X.max()))
            print("X mean:", float(X.mean()))
            print("X std:", float(X.std()))

        print("\n=== CNN INPUT FINAL DEBUG ===")
        print("X shape:", X.shape)
        print("X min:", float(X.min()))
        print("X max:", float(X.max()))
        print("X mean:", float(X.mean()))
        print("X std:", float(X.std()))

        model = DrumCNN(num_classes=len(class_names), dropout=0.0)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        probs = predict_frame_probs_cnn(
            model=model,
            X=X,
            cfg=cfg,
            device=device,
            class_names=class_names,
            batch_size=args.batch_size,
            window_norm=cnn_window_norm,
        )

    elif args.model_type == "snn":
        n_in = int(ckpt["n_in"])
        n_out = int(ckpt["n_out"])

        class_names = ckpt.get("class_names", EXPECTED_CLASSES)

        if class_names != EXPECTED_CLASSES:
            raise ValueError(
                "El orden de class_names del checkpoint NO coincide con el esperado.\n"
                f"Checkpoint: {class_names}\n"
                f"Esperado:   {EXPECTED_CLASSES}"
            )

        if n_out != len(EXPECTED_CLASSES):
            raise ValueError(
                f"n_out={n_out}, pero se esperaban {len(EXPECTED_CLASSES)} clases."
            )

        if n_in % 3 != 0:
            raise ValueError(
                f"n_in={n_in}. Esperaba n_in divisible entre 3 "
                f"porque se usa MFCC + delta + delta-delta."
            )

        n_mfcc = n_in // 3

        cfg = MFCCConfig(
            sr=22050,
            hop_length=256,
            n_mfcc=n_mfcc,
            win_seconds=1.0,
        )

        _, X, _ = compute_mfcc(args.wav, cfg)
        X = X.astype(np.float32)  # [n_mfcc, T]

        mfcc_mean, mfcc_std = load_mfcc_normalization(
            processed_root=args.processed_root,
            n_mfcc=n_mfcc,
        )

        X = normalize_mfcc_like_training(X, mfcc_mean, mfcc_std)

        print("\n=== SNN MFCC DEBUG AFTER TRAIN NORMALIZATION ===")
        print("X shape:", X.shape)
        print("X min:", float(X.min()))
        print("X max:", float(X.max()))
        print("X mean:", float(X.mean()))
        print("X std:", float(X.std()))
        print("per-coeff mean first 5:", X.mean(axis=1)[:5])
        print("per-coeff std first 5:", X.std(axis=1)[:5])
        print("repeat_steps:", args.repeat_steps)
        print("tol_steps:", args.tol_steps)

        model = build_snn_model(n_in=n_in, n_out=n_out)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        probs = predict_frame_probs_snn(
            model=model,
            X=X,
            cfg=cfg,
            device=device,
            n_in=n_in,
            batch_size=args.batch_size,
            repeat_steps=args.repeat_steps,
            tol_steps=args.tol_steps,
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if args.out_probs is not None:
        out_probs = Path(args.out_probs)
        out_probs.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_probs, probs)
        print(f"Saved probs: {out_probs}")

    if args.smooth_kernel > 1:
        probs = smooth_probs(probs, kernel_size=args.smooth_kernel)
        print(f"Applied probability smoothing: kernel={args.smooth_kernel}")

    thresholds = resolve_thresholds(
        probs=probs,
        ckpt=ckpt,
        args=args,
        class_names=class_names,
        model_type=args.model_type,
    )

    print_prob_stats(probs=probs, class_names=class_names, thresholds=thresholds)
    print_debug_frames(probs=probs, class_names=class_names, debug_frames=args.debug_frames)

    min_dist_by_class = build_min_dist_by_class(
        class_names=class_names,
        default_min_dist_ms=args.min_dist_ms,
        snare_min_dist_ms=args.snare_min_dist_ms,
        kick_min_dist_ms=args.kick_min_dist_ms,
        hihat_min_dist_ms=args.hihat_min_dist_ms,
    )

    print("\nMin distance by class:")
    for c in class_names:
        print(f"  {c:15s}: {min_dist_by_class[c]:.1f} ms")

    events_raw = probs_to_events(
        probs=probs,
        cfg=cfg,
        class_names=class_names,
        thr=thresholds,
        min_dist_ms=args.min_dist_ms,
        min_dist_by_class_ms=min_dist_by_class,
    )

    if args.out_raw_events is not None:
        save_json(args.out_raw_events, events_raw)
        print(f"Saved raw events: {args.out_raw_events} ({len(events_raw)} events)")

    events = apply_group_suppression(
        events_raw,
        enabled=not args.no_group_suppression,
        window_ms=args.group_window_ms,
    )

    meta = {
        "wav": args.wav,
        "ckpt": args.ckpt,
        "model_type": args.model_type,
        "processed_root": args.processed_root,
        "threshold_profile": args.threshold_profile,
        "thresholds_json": args.thresholds_json,
        "use_ckpt_thresholds": args.use_ckpt_thresholds,
        "adaptive_thresholds": args.adaptive_thresholds,
        "min_dist_ms": args.min_dist_ms,
        "snare_min_dist_ms": args.snare_min_dist_ms,
        "kick_min_dist_ms": args.kick_min_dist_ms,
        "hihat_min_dist_ms": args.hihat_min_dist_ms,
        "smooth_kernel": args.smooth_kernel,
        "repeat_steps": args.repeat_steps,
        "tol_steps": args.tol_steps,
        "group_suppression": not args.no_group_suppression,
        "group_window_ms": args.group_window_ms,
        "num_raw_events": len(events_raw),
        "num_events": len(events),
    }

    if args.wrapped_json:
        payload = {"meta": meta, "events": events}
    else:
        payload = events

    save_json(args.out_events, payload)
    print(f"\nSaved events: {args.out_events} ({len(events)} events)")


if __name__ == "__main__":
    main()
