# -*- coding: utf-8 -*-
"""
Few-shot SNN training.

Run examples:
    python -m scripts.train_snn_fewshot --shots 1
    python -m scripts.train_snn_fewshot --shots 5
    python -m scripts.train_snn_fewshot --shots 10

Important:
    CNN and SNN must use the same data for fair comparison.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cnn.dataset_mfcc import CLASSES
from models.snn.dataset_GROOVE import GrooveWindowDataset, groove_window_collate_fn
from models.snn.model import LSNN
from models.snn.poisson_encoder import encode_window_sequence, effective_repeat_steps, normalize_input_encoding

from scripts.fewshot_datasets import FewShotSNNDataset


REPEAT_STEPS = 5


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
    """
    features: [B, T_frames, n_in]
    returns:  [T_snn, B, n_in]
    """
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

    x = features.transpose(0, 1).contiguous()

    if not (input_encoding == "poisson" and str(poisson_collapse).lower() == "none") and repeat_steps > 1:
        x = x.repeat_interleave(repeat_steps, dim=0)
    return x


@torch.no_grad()
def f1_stats_from_logits(logits, targets, thr: Union[float, torch.Tensor] = 0.5):
    probs = torch.sigmoid(logits)

    if isinstance(thr, torch.Tensor):
        thr_use = thr.to(probs.device).view(1, -1)
        preds = (probs >= thr_use).to(targets.dtype)
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

    tp = tp_c.sum()
    fp = fp_c.sum()
    fn = fn_c.sum()
    tn = tn_c.sum()

    return {
        "micro_f1": float(2.0 * tp / (2.0 * tp + fp + fn + eps)),
        "micro_precision": float(tp / (tp + fp + eps)),
        "micro_recall": float(tp / (tp + fn + eps)),
        "macro_f1": float(f1_c.mean()),
        "macro_precision": float(prec_c.mean()),
        "macro_recall": float(rec_c.mean()),
        "exact_match": float((preds == targets).all(dim=1).float().mean()),
        "precision_per_class": [float(x) for x in prec_c.tolist()],
        "recall_per_class": [float(x) for x in rec_c.tolist()],
        "f1_per_class": [float(x) for x in f1_c.tolist()],
        "tp_per_class": [float(x) for x in tp_c.tolist()],
        "fp_per_class": [float(x) for x in fp_c.tolist()],
        "fn_per_class": [float(x) for x in fn_c.tolist()],
        "tn_per_class": [float(x) for x in tn_c.tolist()],
        "support_per_class": [float(x) for x in (tp_c + fn_c).tolist()],
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def add_derived_rates(stats, eps=1e-8):
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]

    stats["micro_fpr"] = fp / (fp + tn + eps)
    stats["micro_fnr"] = fn / (fn + tp + eps)
    stats["micro_specificity"] = tn / (tn + fp + eps)
    stats["micro_balanced_acc"] = 0.5 * (stats["micro_recall"] + stats["micro_specificity"])

    C = len(stats["tp_per_class"])
    fpr_c, fnr_c, spec_c = [], [], []

    for i in range(C):
        tp_i = stats["tp_per_class"][i]
        fp_i = stats["fp_per_class"][i]
        fn_i = stats["fn_per_class"][i]
        tn_i = stats["tn_per_class"][i]

        fpr_c.append(fp_i / (fp_i + tn_i + eps))
        fnr_c.append(fn_i / (fn_i + tp_i + eps))
        spec_c.append(tn_i / (tn_i + fp_i + eps))

    stats["fpr_per_class"] = fpr_c
    stats["fnr_per_class"] = fnr_c
    stats["specificity_per_class"] = spec_c

    return stats


@torch.no_grad()
def find_best_threshold_per_class(logits, targets, thresholds=None):
    if thresholds is None:
        thresholds = torch.linspace(0.10, 0.90, 33)

    probs = torch.sigmoid(logits)
    C = targets.shape[1]

    best_thr = torch.full((C,), 0.5, dtype=torch.float32)
    best_f1 = torch.full((C,), -1.0, dtype=torch.float32)

    eps = 1e-8

    for thr in thresholds:
        preds = (probs >= float(thr)).to(targets.dtype)

        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1.0 - targets)).sum(dim=0)
        fn = ((1.0 - preds) * targets).sum(dim=0)

        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

        improved = f1 > best_f1
        best_f1[improved] = f1[improved]
        best_thr[improved] = float(thr)

    return best_thr, best_f1


@torch.no_grad()
def estimate_pos_weight_from_loader(dl, num_classes, max_batches=9999, eps=1e-6):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    n = 0

    for i, (_, y, _) in enumerate(dl):
        pos += y.sum(dim=0).double()
        n += y.shape[0]

        if i + 1 >= max_batches:
            break

    neg = n - pos
    pos_weight = neg / (pos + eps)
    pos_rate = pos / max(n, 1)

    return pos_weight.float(), pos_rate.float()


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    pos_weight,
    threshold=0.5,
    tol_steps=5,
    repeat_steps=REPEAT_STEPS,
    reg_rate=10.0,
    reg_voltage=1e-4,
    f_target_hz=15.0,
    homeo_lr=1e-4,
    log_every=1,
    input_encoding="analog",
    poisson_steps=None,
    poisson_max_rate=0.30,
    poisson_silence_steps=0,
    poisson_collapse="none",
    poisson_normalize="window",
):
    model.train()

    total_loss = 0.0
    total_loss_pred = 0.0
    total_loss_reg = 0.0

    all_logits = []
    all_targets = []

    fr_avg_sum = 0.0
    fr_max_sum = 0.0
    n_batches = 0

    for batch_idx, (features, labels, _) in enumerate(train_loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x_seq = prepare_window_batch(
            features,
            repeat_steps=repeat_steps,
            input_encoding=input_encoding,
            poisson_steps=poisson_steps,
            poisson_max_rate=poisson_max_rate,
            poisson_silence_steps=poisson_silence_steps,
            poisson_collapse=poisson_collapse,
            poisson_normalize=poisson_normalize,
        )

        out = model.eprop_update_window(
            x_seq=x_seq,
            targets=labels,
            optimizer=optimizer,
            pos_weight=pos_weight,
            tol_steps=tol_steps,
            repeat_steps=repeat_steps,
            reg_rate=reg_rate,
            reg_voltage=reg_voltage,
            f_target_hz=f_target_hz,
            dt_seconds=1e-3,
            homeo_lr=homeo_lr,
        )

        logits = out["logits_window"].detach().cpu()

        total_loss += out["loss"] * features.shape[0]
        total_loss_pred += out["loss_pred"] * features.shape[0]
        total_loss_reg += out["loss_reg"] * features.shape[0]

        all_logits.append(logits)
        all_targets.append(labels.detach().cpu())

        fr_avg_sum += out["spike_rate_hz"]
        fr_max_sum += out.get("spike_rate_max_hz", out["spike_rate_hz"])
        n_batches += 1

        if batch_idx % log_every == 0:
            tmp_logits = torch.cat(all_logits, dim=0)
            tmp_targets = torch.cat(all_targets, dim=0)
            tmp_stats = f1_stats_from_logits(tmp_logits, tmp_targets, thr=threshold)

            print(
                f"batch {batch_idx:4d}/{len(train_loader)-1:4d} | "
                f"loss={out['loss']:.4f} | "
                f"pred={out['loss_pred']:.4f} | "
                f"reg={out['loss_reg']:.4f} | "
                f"micro_f1={tmp_stats['micro_f1']:.4f} | "
                f"macro_f1={tmp_stats['macro_f1']:.4f} | "
                f"fr_avg={out['spike_rate_hz']:.1f}Hz | "
                f"fr_max={out.get('spike_rate_max_hz', out['spike_rate_hz']):.1f}Hz"
            )

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    stats = f1_stats_from_logits(logits_all, targets_all, thr=threshold)
    stats["loss"] = total_loss / max(len(train_loader.dataset), 1)
    stats["loss_pred"] = total_loss_pred / max(len(train_loader.dataset), 1)
    stats["loss_reg"] = total_loss_reg / max(len(train_loader.dataset), 1)
    stats["fr_avg"] = fr_avg_sum / max(n_batches, 1)
    stats["fr_max"] = fr_max_sum / max(n_batches, 1)
    stats["spike_rate_hz"] = stats["fr_avg"]

    return add_derived_rates(stats)


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    pos_weight,
    threshold=0.5,
    tol_steps=5,
    repeat_steps=REPEAT_STEPS,
    return_logits=False,
    input_encoding="analog",
    poisson_steps=None,
    poisson_max_rate=0.30,
    poisson_silence_steps=0,
    poisson_collapse="none",
    poisson_normalize="window",
):
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for features, labels, _ in data_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x_seq = prepare_window_batch(
            features,
            repeat_steps=repeat_steps,
            input_encoding=input_encoding,
            poisson_steps=poisson_steps,
            poisson_max_rate=poisson_max_rate,
            poisson_silence_steps=poisson_silence_steps,
            poisson_collapse=poisson_collapse,
            poisson_normalize=poisson_normalize,
        )

        T, B, _ = x_seq.shape
        C = model.n_out
        kappa = model.kappa.to(device)

        state = model.alif.zero_state(B, device=device)
        y_prev = torch.zeros(B, C, device=device)

        tol_ms = tol_steps * repeat_steps
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
    stats["loss"] = total_loss / max(len(data_loader.dataset), 1)
    stats["loss_pred"] = stats["loss"]
    stats["loss_reg"] = 0.0
    stats["fr_avg"] = 0.0
    stats["fr_max"] = 0.0
    stats["spike_rate_hz"] = 0.0
    stats = add_derived_rates(stats)

    if return_logits:
        return stats, logits_all, targets_all

    return stats


def write_metrics_csv(metrics, class_names, csv_epoch_path, csv_per_class_path):
    aggregate_rows = []
    per_class_rows = []

    for split_name, split_history in metrics.items():
        if split_name not in ("train", "validation", "test"):
            continue

        for em in split_history:
            aggregate_rows.append({
                "epoch": em["epoch"],
                "split": split_name,
                "loss": em["loss"],
                "micro_precision": em["micro_precision"],
                "micro_recall": em["micro_recall"],
                "micro_f1": em["micro_f1"],
                "macro_precision": em["macro_precision"],
                "macro_recall": em["macro_recall"],
                "macro_f1": em["macro_f1"],
                "exact_match": em["exact_match"],
                "fr_avg": em.get("fr_avg", 0.0),
                "fr_max": em.get("fr_max", 0.0),
                "threshold_mean": em.get("threshold_mean"),
                "lr": em.get("lr"),
            })

            for ci, cn in enumerate(class_names):
                per_class_rows.append({
                    "epoch": em["epoch"],
                    "split": split_name,
                    "class_idx": ci,
                    "class_name": cn,
                    "precision": em["precision_per_class"][ci],
                    "recall": em["recall_per_class"][ci],
                    "f1": em["f1_per_class"][ci],
                    "threshold": em["threshold_per_class"][ci],
                    "tp": em["tp_per_class"][ci],
                    "fp": em["fp_per_class"][ci],
                    "fn": em["fn_per_class"][ci],
                    "tn": em["tn_per_class"][ci],
                    "support": em["support_per_class"][ci],
                })

    csv_epoch_path.parent.mkdir(parents=True, exist_ok=True)

    if aggregate_rows:
        with csv_epoch_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            w.writeheader()
            w.writerows(aggregate_rows)

    if per_class_rows:
        with csv_per_class_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
            w.writeheader()
            w.writerows(per_class_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, choices=[1, 5, 10], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", type=str, default="data/processed/groove_processed_mfcc")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeat-steps", type=int, default=5)
    parser.add_argument("--input-encoding", choices=["analog", "poisson"], default="analog")
    parser.add_argument("--poisson-steps", type=int, default=None)
    parser.add_argument("--poisson-max-rate", type=float, default=0.30)
    parser.add_argument("--poisson-silence-steps", type=int, default=0)
    parser.add_argument("--poisson-collapse", choices=["none", "mean", "any"], default="none")
    parser.add_argument("--poisson-normalize", choices=["window", "feature", "sigmoid", "clamp"], default="window")
    args = parser.parse_args()
    if args.poisson_steps is None:
        args.poisson_steps = args.repeat_steps

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = args.base_dir
    base_dir = Path(args.base_dir)

    manifest_path = Path(f"data/fewshot/fewshot_{args.shots}shot_seed{args.seed}.json")

    out_dir = Path(f"models/snn/noDeltas/{args.shots}shot_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = out_dir / "metrics_pytorch_groove_window.json"
    out_csv_epoch_path = out_dir / "metrics_pytorch_groove_window_epoch.csv"
    out_csv_per_class_path = out_dir / "metrics_pytorch_groove_window_per_class.csv"
    out_model_path = out_dir / "best_groove_window.pt"

    train_dataset = FewShotSNNDataset(
        manifest_path=manifest_path,
        feature_base_dir=base_dir,
    )

    # Keep your original validation/test distribution from the SNN trainer.
    val_dataset = GrooveWindowDataset(
        root=root,
        split="validation",
        add_deltas=False,
        win_seconds=1.0,
        tol_frames=3,
        label_mode="center",
        sampling="random",
        stride_frames=0,
        p_pos=0.5,
        seed=123,
    )

    test_dataset = GrooveWindowDataset(
        root=root,
        split="test",
        add_deltas=False,
        win_seconds=1.0,
        tol_frames=3,
        label_mode="center",
        sampling="random",
        stride_frames=0,
        p_pos=0.5,
        seed=456,
    )

    batch_size = min(args.batch_size, len(train_dataset))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=groove_window_collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=groove_window_collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )

    pos_weight_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    n_in = train_dataset[0][0].shape[1]
    n_out = train_dataset.n_classes
    class_names = train_dataset.class_names

    print(f"Few-shot manifest: {manifest_path}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"n_in={n_in}, n_out={n_out}")

    input_encoding = normalize_input_encoding(args.input_encoding)
    snn_repeat_steps = effective_repeat_steps(
        repeat_steps=args.repeat_steps,
        encoding=input_encoding,
        poisson_steps=args.poisson_steps,
        poisson_silence_steps=args.poisson_silence_steps,
        poisson_collapse=args.poisson_collapse,
    )
    print(
        "input_encoding:", input_encoding,
        "| repeat_steps:", args.repeat_steps,
        "| poisson_steps:", args.poisson_steps,
        "| poisson_max_rate:", args.poisson_max_rate,
        "| poisson_collapse:", args.poisson_collapse,
        "| poisson_normalize:", args.poisson_normalize,
        "| effective_repeat_steps:", snn_repeat_steps,
    )

    model = LSNN(
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
    ).to(device)

    pos_weight, pos_rate = estimate_pos_weight_from_loader(
        pos_weight_loader,
        num_classes=n_out,
        max_batches=9999,
    )

    pos_weight = torch.sqrt(pos_weight).clamp(max=8.0)

    print("pos_rate:", {class_names[i]: float(pos_rate[i]) for i in range(n_out)})
    print("pos_weight:", {class_names[i]: float(pos_weight[i]) for i in range(n_out)})

    pos_weight = pos_weight.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": [model.alif.w_in, model.alif.w_rec], "lr": 3e-4, "weight_decay": 1e-5},
            {"params": [model.w_out, model.b_out, model.B], "lr": 1e-3, "weight_decay": 1e-4},
        ],
        eps=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-3,
    )

    metrics = {
        "train": [],
        "validation": [],
        "test": [],
        "class_names": class_names,
        "shots_per_class": args.shots,
        "seed": args.seed,
        "manifest_path": str(manifest_path),
        "input_encoding": input_encoding,
        "repeat_steps": args.repeat_steps,
        "effective_repeat_steps": snn_repeat_steps,
        "poisson_steps": args.poisson_steps,
        "poisson_max_rate": args.poisson_max_rate,
        "poisson_silence_steps": args.poisson_silence_steps,
        "poisson_collapse": args.poisson_collapse,
        "poisson_normalize": args.poisson_normalize,
    }

    best_val = -1.0
    best_thresholds = torch.full((n_out,), 0.5, dtype=torch.float32)
    bad_epochs = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== SNN Few-shot {args.shots}-shot | Epoch {epoch:03d} ===")
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            threshold=best_thresholds,
            tol_steps=5,
            repeat_steps=snn_repeat_steps,
            reg_rate=10.0,
            reg_voltage=1e-4,
            f_target_hz=15.0,
            homeo_lr=1e-4,
            log_every=1,
            input_encoding=input_encoding,
            poisson_steps=args.poisson_steps,
            poisson_max_rate=args.poisson_max_rate,
            poisson_silence_steps=args.poisson_silence_steps,
            poisson_collapse=args.poisson_collapse,
            poisson_normalize=args.poisson_normalize,
        )

        val_stats_raw, logits_val, targets_val = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            threshold=best_thresholds,
            tol_steps=5,
            repeat_steps=snn_repeat_steps,
            return_logits=True,
            input_encoding=input_encoding,
            poisson_steps=args.poisson_steps,
            poisson_max_rate=args.poisson_max_rate,
            poisson_silence_steps=args.poisson_silence_steps,
            poisson_collapse=args.poisson_collapse,
            poisson_normalize=args.poisson_normalize,
        )

        thr_c, _ = find_best_threshold_per_class(logits_val, targets_val)

        val_metrics = f1_stats_from_logits(
            logits_val,
            targets_val,
            thr=thr_c,
        )

        val_metrics["loss"] = val_stats_raw["loss"]
        val_metrics["loss_pred"] = val_stats_raw["loss_pred"]
        val_metrics["loss_reg"] = 0.0
        val_metrics["fr_avg"] = 0.0
        val_metrics["fr_max"] = 0.0
        val_metrics["spike_rate_hz"] = 0.0
        val_metrics = add_derived_rates(val_metrics)

        threshold_list = [float(x) for x in thr_c.tolist()]
        threshold_mean = float(thr_c.mean())
        lr = float(optimizer.param_groups[0]["lr"])

        for pack in (train_metrics, val_metrics):
            pack["epoch"] = epoch
            pack["threshold_per_class"] = threshold_list
            pack["threshold_mean"] = threshold_mean
            pack["lr"] = lr

        metrics["train"].append(train_metrics)
        metrics["validation"].append(val_metrics)

        score = val_metrics["macro_f1"]
        scheduler.step(score)

        print(
            f"Epoch {epoch:03d} | "
            f"train_f1={train_metrics['micro_f1']:.3f} | "
            f"val_micro={val_metrics['micro_f1']:.3f} | "
            f"val_macro={val_metrics['macro_f1']:.3f} | "
            f"thr_mean={threshold_mean:.2f} | "
            f"lr={lr:.1e} | "
            f"fr_avg={train_metrics['fr_avg']:.1f}Hz | "
            f"fr_max={train_metrics['fr_max']:.1f}Hz | "
            f"time={time.time() - t0:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_thresholds = thr_c.clone()
            bad_epochs = 0

            test_metrics = evaluate(
                model=model,
                data_loader=test_loader,
                device=device,
                pos_weight=pos_weight,
                threshold=thr_c,
                tol_steps=5,
                repeat_steps=snn_repeat_steps,
                input_encoding=input_encoding,
                poisson_steps=args.poisson_steps,
                poisson_max_rate=args.poisson_max_rate,
                poisson_silence_steps=args.poisson_silence_steps,
                poisson_collapse=args.poisson_collapse,
                poisson_normalize=args.poisson_normalize,
            )

            test_metrics["epoch"] = epoch
            test_metrics["threshold_per_class"] = threshold_list
            test_metrics["threshold_mean"] = threshold_mean
            test_metrics["lr"] = lr

            metrics["test"].append(test_metrics)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "threshold_per_class": threshold_list,
                    "threshold_by_class": {
                        class_names[i]: float(thr_c[i])
                        for i in range(len(class_names))
                    },
                    "class_names": class_names,
                    "shots_per_class": args.shots,
                    "seed": args.seed,
                    "manifest_path": str(manifest_path),
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_micro_f1": val_metrics["micro_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_micro_f1": test_metrics["micro_f1"],
                    "n_in": n_in,
                    "n_out": n_out,
                    "input_encoding": input_encoding,
                    "repeat_steps": args.repeat_steps,
                    "effective_repeat_steps": snn_repeat_steps,
                    "poisson_steps": args.poisson_steps,
                    "poisson_max_rate": args.poisson_max_rate,
                    "poisson_silence_steps": args.poisson_silence_steps,
                    "poisson_collapse": args.poisson_collapse,
                    "poisson_normalize": args.poisson_normalize,
                    "total_training_time_sec": time.time() - start_time,
                },
                out_model_path,
            )

            print(
                f"Saved best model -> {out_model_path} | "
                f"test_macro={test_metrics['macro_f1']:.4f} | "
                f"test_micro={test_metrics['micro_f1']:.4f}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {args.patience} non-improving epochs.")
                break

        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        write_metrics_csv(
            metrics,
            class_names,
            out_csv_epoch_path,
            out_csv_per_class_path,
        )

    print(f"Best val_macro_f1 = {best_val:.4f}")
    print(f"Total time = {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
