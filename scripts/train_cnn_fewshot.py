# -*- coding: utf-8 -*-
"""
Few-shot CNN training.

Run examples:
    python -m scripts.train_cnn_fewshot --shots 1
    python -m scripts.train_cnn_fewshot --shots 5
    python -m scripts.train_cnn_fewshot --shots 10

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
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cnn.dataset_mfcc import DrumOnsetWindowDataset, SpecConfig, CLASSES
from models.cnn.model import DrumCNN

from scripts.fewshot_datasets import FewShotCNNDataset


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def f1_stats_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: Union[float, torch.Tensor] = 0.5):
    probs = torch.sigmoid(logits)

    if isinstance(thr, torch.Tensor):
        thr = thr.to(probs.device).view(1, -1)
        preds = (probs >= thr).to(targets.dtype)
    else:
        preds = (probs >= float(thr)).to(targets.dtype)

    eps = 1e-8
    C = targets.shape[1]

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
        "precision_by_class": {i: float(prec_c[i]) for i in range(C)},
        "recall_by_class": {i: float(rec_c[i]) for i in range(C)},
        "f1_by_class": {i: float(f1_c[i]) for i in range(C)},
        "tp_by_class": {i: float(tp_c[i]) for i in range(C)},
        "fp_by_class": {i: float(fp_c[i]) for i in range(C)},
        "fn_by_class": {i: float(fn_c[i]) for i in range(C)},
        "tn_by_class": {i: float(tn_c[i]) for i in range(C)},
        "support_by_class": {i: float(tp_c[i] + fn_c[i]) for i in range(C)},
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def add_derived_rates(stats: dict, eps: float = 1e-8) -> dict:
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]

    stats["micro_fpr"] = fp / (fp + tn + eps)
    stats["micro_fnr"] = fn / (fn + tp + eps)
    stats["micro_specificity"] = tn / (tn + fp + eps)
    stats["micro_balanced_acc"] = 0.5 * (stats["micro_recall"] + stats["micro_specificity"])

    C = len(stats["tp_by_class"])
    stats["fpr_by_class"] = {}
    stats["fnr_by_class"] = {}
    stats["specificity_by_class"] = {}

    for i in range(C):
        tp_i = stats["tp_by_class"][i]
        fp_i = stats["fp_by_class"][i]
        fn_i = stats["fn_by_class"][i]
        tn_i = stats["tn_by_class"][i]

        stats["fpr_by_class"][i] = fp_i / (fp_i + tn_i + eps)
        stats["fnr_by_class"][i] = fn_i / (fn_i + tp_i + eps)
        stats["specificity_by_class"][i] = tn_i / (tn_i + fp_i + eps)

    return stats


@torch.no_grad()
def find_best_threshold_per_class(logits: torch.Tensor, targets: torch.Tensor, thresholds=None):
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, 19)

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
def estimate_pos_weight_from_loader(dl, num_classes: int, max_batches: int = 9999, eps: float = 1e-6):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    n = 0

    for i, (_, y) in enumerate(dl):
        pos += y.sum(dim=0).double()
        n += y.shape[0]

        if i + 1 >= max_batches:
            break

    neg = n - pos
    pos_weight = neg / (pos + eps)
    pos_rate = pos / max(n, 1)

    return pos_weight.float(), pos_rate.float()


def train_one_epoch_cnn(model, train_dl, criterion, optimizer, scaler, device, use_amp, threshold=0.5, log_every=1):
    model.train()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch_idx, (X, y) in enumerate(train_dl):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss) * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

        if batch_idx % log_every == 0:
            tmp_logits = torch.cat(all_logits, dim=0)
            tmp_targets = torch.cat(all_targets, dim=0)
            tmp_stats = f1_stats_from_logits(tmp_logits, tmp_targets, thr=threshold)

            print(
                f"batch {batch_idx:4d}/{len(train_dl)-1:4d} | "
                f"loss={float(loss):.4f} | "
                f"micro_f1={tmp_stats['micro_f1']:.4f} | "
                f"macro_f1={tmp_stats['macro_f1']:.4f}"
            )

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    stats = f1_stats_from_logits(logits_all, targets_all, thr=threshold)
    stats = add_derived_rates(stats)
    stats["loss"] = total_loss / max(len(train_dl.dataset), 1)

    return stats


@torch.no_grad()
def evaluate_cnn(model, data_loader, criterion, device, threshold, return_logits=False):
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for X, y in data_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += float(loss) * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    stats = f1_stats_from_logits(logits_all, targets_all, thr=threshold)
    stats = add_derived_rates(stats)
    stats["loss"] = total_loss / max(len(data_loader.dataset), 1)

    if return_logits:
        return stats, logits_all, targets_all

    return stats


def write_epoch_csv(csv_path, row, class_names, write_header):
    base_cols = [
        "epoch", "epoch_time_sec",
        "train_loss", "train_micro_f1", "train_macro_f1",
        "val_loss", "val_micro_f1", "val_macro_f1",
        "test_loss", "test_micro_f1", "test_macro_f1",
        "threshold_mean", "lr",
    ]

    per_class_cols = []
    for name in class_names:
        per_class_cols += [
            f"val_f1_{name}",
            f"test_f1_{name}",
            f"threshold_{name}",
        ]

    cols = base_cols + per_class_cols

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in cols})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, choices=[1, 5, 10], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", type=str, default="data/processed/groove_processed_mfcc")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("Device:", device)

    base_dir = Path(args.base_dir)
    preprocessing_summary = base_dir / "preprocessing_summary.json"
    summary = load_json(preprocessing_summary)
    summary_cfg = summary.get("config", {})

    cfg = SpecConfig(
        sr=int(summary_cfg.get("sr", 22050)),
        n_fft=int(summary_cfg.get("n_fft", 1024)),
        hop_length=int(summary_cfg.get("hop_length", 256)),
        n_mfcc=int(summary_cfg.get("n_mfcc", 40)),
        win_seconds=1.0,
        tol_frames=3,
        label_mode="center",
        feature_type="mfcc",
        feature_base_dir=str(base_dir),
        mfcc_mean=summary.get("mfcc_mean"),
        mfcc_std=summary.get("mfcc_std"),
    )

    manifest_path = Path(f"data/fewshot/fewshot_{args.shots}shot_seed{args.seed}.json")

    out_dir = Path(f"models/cnn/runs/run_mfcc_{args.shots}shot_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl_path = out_dir / "metrics.jsonl"
    metrics_csv_path = out_dir / "metrics.csv"
    write_header = not metrics_csv_path.exists()

    train_ds = FewShotCNNDataset(
        manifest_path=manifest_path,
        feature_base_dir=base_dir,
    )

    # Keep your original validation/test distribution.
    val_ds = DrumOnsetWindowDataset(
        index_jsonl=str(base_dir / "validation" / "index.jsonl"),
        ids=None,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=8,
        seed=123,
        p_pos=0.5,
    )

    test_ds = DrumOnsetWindowDataset(
        index_jsonl=str(base_dir / "test" / "index.jsonl"),
        ids=None,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=8,
        seed=456,
        p_pos=0.5,
    )

    print(f"Few-shot manifest: {manifest_path}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    pin_memory = device == "cuda"
    batch_size = min(args.batch_size, len(train_ds))

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    pw_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    pos_weight, pos_rate = estimate_pos_weight_from_loader(
        pw_dl,
        num_classes=len(CLASSES),
        max_batches=9999,
    )

    # Few-shot can make pos_weight huge because each class appears only K times.
    pos_weight = torch.log1p(pos_weight).clamp(max=8.0)

    print("Estimated pos_rate per class:", {CLASSES[i]: float(pos_rate[i]) for i in range(len(CLASSES))})
    print("Using pos_weight:", {CLASSES[i]: float(pos_weight[i]) for i in range(len(CLASSES))})

    model = DrumCNN(num_classes=len(CLASSES), dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-3,
    )

    best_val_macro = -1.0
    bad_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print(f"\n=== CNN Few-shot {args.shots}-shot | Epoch {epoch:03d} ===")

        train_stats = train_one_epoch_cnn(
            model=model,
            train_dl=train_dl,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            threshold=0.5,
            log_every=1,
        )

        val_raw, logits_val, targets_val = evaluate_cnn(
            model=model,
            data_loader=val_dl,
            criterion=criterion,
            device=device,
            threshold=0.5,
            return_logits=True,
        )

        thr_c, _ = find_best_threshold_per_class(logits_val, targets_val)

        val_stats = f1_stats_from_logits(logits_val, targets_val, thr=thr_c)
        val_stats = add_derived_rates(val_stats)
        val_stats["loss"] = val_raw["loss"]

        test_stats = evaluate_cnn(
            model=model,
            data_loader=test_dl,
            criterion=criterion,
            device=device,
            threshold=thr_c,
            return_logits=False,
        )

        score = val_stats["macro_f1"]
        scheduler.step(score)

        epoch_time = time.time() - epoch_start
        threshold_mean = float(thr_c.mean())
        lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_micro={train_stats['micro_f1']:.4f} | "
            f"train_macro={train_stats['macro_f1']:.4f} | "
            f"val_micro={val_stats['micro_f1']:.4f} | "
            f"val_macro={val_stats['macro_f1']:.4f} | "
            f"test_micro={test_stats['micro_f1']:.4f} | "
            f"test_macro={test_stats['macro_f1']:.4f} | "
            f"thr_mean={threshold_mean:.2f} | "
            f"lr={lr:.1e} | "
            f"time={epoch_time:.2f}s"
        )

        metrics_row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "shots": args.shots,
            "seed": args.seed,
            "manifest_path": str(manifest_path),
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
            "threshold_by_class": {CLASSES[i]: float(thr_c[i]) for i in range(len(CLASSES))},
            "threshold_mean": threshold_mean,
            "lr": lr,
        }

        with metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row) + "\n")

        csv_row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_stats["loss"],
            "train_micro_f1": train_stats["micro_f1"],
            "train_macro_f1": train_stats["macro_f1"],
            "val_loss": val_stats["loss"],
            "val_micro_f1": val_stats["micro_f1"],
            "val_macro_f1": val_stats["macro_f1"],
            "test_loss": test_stats["loss"],
            "test_micro_f1": test_stats["micro_f1"],
            "test_macro_f1": test_stats["macro_f1"],
            "threshold_mean": threshold_mean,
            "lr": lr,
        }

        for i, name in enumerate(CLASSES):
            csv_row[f"val_f1_{name}"] = val_stats["f1_by_class"][i]
            csv_row[f"test_f1_{name}"] = test_stats["f1_by_class"][i]
            csv_row[f"threshold_{name}"] = float(thr_c[i])

        write_epoch_csv(metrics_csv_path, csv_row, CLASSES, write_header)
        write_header = False

        if score > best_val_macro:
            best_val_macro = score
            bad_epochs = 0

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "classes": CLASSES,
                "shots_per_class": args.shots,
                "seed": args.seed,
                "manifest_path": str(manifest_path),
                "val_micro_f1": val_stats["micro_f1"],
                "val_macro_f1": val_stats["macro_f1"],
                "test_micro_f1": test_stats["micro_f1"],
                "test_macro_f1": test_stats["macro_f1"],
                "threshold_by_class": {CLASSES[i]: float(thr_c[i]) for i in range(len(CLASSES))},
                "threshold_per_class": [float(x) for x in thr_c.tolist()],
                "total_training_time_sec": time.time() - start_time,
                "lr": lr,
            }

            torch.save(ckpt, out_dir / "best.pt")

            with (out_dir / "best_test_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "test_stats": test_stats,
                        "threshold_by_class": ckpt["threshold_by_class"],
                    },
                    f,
                    indent=2,
                )

            print(f"Saved best.pt -> {out_dir / 'best.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {args.patience} non-improving epochs.")
                break

    print(f"Best val_macro_f1 = {best_val_macro:.4f}")
    print(f"Total time = {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
