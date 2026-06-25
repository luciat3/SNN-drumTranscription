# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Union

from models.cnn.dataset import DrumOnsetWindowDataset, SpecConfig, CLASSES
from models.cnn.fewshot import FewShotMultiLabelDataset
from models.cnn.model import DrumCNN

import time

def load_splits(splits_path: str):
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    return data["train"], data["val"], data["test"]


@torch.no_grad()
def f1_stats_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thr: Union[float, torch.Tensor] = 0.5,
):
    probs = torch.sigmoid(logits)
    if isinstance(thr, torch.Tensor):
        # thr: [C] -> broadcast to [N,C]
        thr = thr.to(probs.device).view(1, -1)
        preds = (probs >= thr).to(targets.dtype)
    else:
        preds = (probs >= float(thr)).to(targets.dtype)

    eps = 1e-8
    C = targets.shape[1]

    tp_c = (preds * targets).sum(dim=0)
    fp_c = (preds * (1 - targets)).sum(dim=0)
    fn_c = ((1 - preds) * targets).sum(dim=0)
    tn_c = ((1 - preds) * (1 - targets)).sum(dim=0)

    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c  = tp_c / (tp_c + fn_c + eps)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    # micro totals
    tp = tp_c.sum()
    fp = fp_c.sum()
    fn = fn_c.sum()
    tn = tn_c.sum()

    micro_prec = (tp / (tp + fp + eps)).item()
    micro_rec  = (tp / (tp + fn + eps)).item()
    micro_f1   = (2 * tp / (2 * tp + fp + fn + eps)).item()

    macro_prec = prec_c.mean().item()
    macro_rec  = rec_c.mean().item()
    macro_f1   = f1_c.mean().item()

    prec_dict = {i: prec_c[i].item() for i in range(C)}
    rec_dict  = {i: rec_c[i].item()  for i in range(C)}
    f1_dict   = {i: f1_c[i].item()   for i in range(C)}

    # counts by class
    tp_dict = {i: tp_c[i].item() for i in range(C)}
    fp_dict = {i: fp_c[i].item() for i in range(C)}
    fn_dict = {i: fn_c[i].item() for i in range(C)}
    tn_dict = {i: tn_c[i].item() for i in range(C)}

    return {
        "micro_f1": micro_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "precision_by_class": prec_dict,
        "recall_by_class": rec_dict,
        "f1_by_class": f1_dict,
        "tp_by_class": tp_dict,
        "fp_by_class": fp_dict,
        "fn_by_class": fn_dict,
        "tn_by_class": tn_dict,
    }


def add_derived_rates(stats: dict, eps: float = 1e-8) -> dict:
    # micro
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    stats["micro_fpr"] = fp / (fp + tn + eps)
    stats["micro_fnr"] = fn / (fn + tp + eps)
    stats["micro_specificity"] = tn / (tn + fp + eps)
    stats["micro_balanced_acc"] = 0.5 * (stats["micro_recall"] + stats["micro_specificity"])

    # per-class
    C = len(stats["tp_by_class"])
    fpr_c, fnr_c, spec_c, supp_pos_c = {}, {}, {}, {}
    for i in range(C):
        tp_i = stats["tp_by_class"][i]
        fp_i = stats["fp_by_class"][i]
        fn_i = stats["fn_by_class"][i]
        tn_i = stats["tn_by_class"][i]

        fpr_c[i] = fp_i / (fp_i + tn_i + eps)
        fnr_c[i] = fn_i / (fn_i + tp_i + eps)
        spec_c[i] = tn_i / (tn_i + fp_i + eps)
        supp_pos_c[i] = tp_i + fn_i  # nº de positivos reales

    stats["fpr_by_class"] = fpr_c
    stats["fnr_by_class"] = fnr_c
    stats["specificity_by_class"] = spec_c
    stats["support_pos_by_class"] = supp_pos_c
    return stats

@torch.no_grad()
def find_best_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds=None,
    optimize: str = "micro",
):
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95

    best_thr = 0.5
    best_micro = -1.0
    best_macro = -1.0

    for thr in thresholds:
        s = f1_stats_from_logits(logits, targets, thr=float(thr))
        micro = s["micro_f1"]
        macro = s["macro_f1"]
        score = micro if optimize == "micro" else macro

        best_score = best_micro if optimize == "micro" else best_macro
        if score > best_score:
            best_thr = float(thr)
            best_micro = micro
            best_macro = macro



    return best_thr, best_micro, best_macro

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
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)

        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        improved = f1 > best_f1
        best_f1[improved] = f1[improved]
        best_thr[improved] = float(thr)

    return best_thr, best_f1


# to avoid class imbalance issues, we can weight the positive examples more in the loss function
def estimate_pos_weight_from_loader(dl, num_classes: int, max_batches: int = 200, eps: float = 1e-6):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    n = 0

    for i, (_, y) in enumerate(dl):
        # y: [B, C] 
        pos += y.sum(dim=0).double()
        n += y.shape[0]
        if i + 1 >= max_batches:
            break

    pos_rate = pos / max(n, 1)
    neg = n - pos
    pos_weight = neg / (pos + eps)

    return pos_weight.float(), pos_rate.float()

def train_one_epoch_cnn(
    model,
    train_dl,
    criterion,
    optim,
    scaler,
    device,
    use_amp,
    threshold=0.5,
    log_every=1,
):
    model.train()

    train_loss = 0.0
    all_logits = []
    all_targets = []

    for batch_idx, (X, y) in enumerate(train_dl):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        train_loss += loss.item() * X.size(0)

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

        if batch_idx % log_every == 0:
            tmp_logits = torch.cat(all_logits, dim=0)
            tmp_targets = torch.cat(all_targets, dim=0)

            tmp_stats = f1_stats_from_logits(
                tmp_logits,
                tmp_targets,
                thr=threshold,
            )

            probs = torch.sigmoid(tmp_logits)
            if isinstance(threshold, torch.Tensor):
                thr_use = threshold.cpu().view(1, -1)
                preds = (probs >= thr_use).to(tmp_targets.dtype)
            else:
                preds = (probs >= float(threshold)).to(tmp_targets.dtype)

            pos_rate_y = tmp_targets.mean().item()
            pos_rate_pred = preds.mean().item()

            print(
                f"batch {batch_idx:4d}/{len(train_dl)-1:4d} | "
                f"loss={loss.item():.4f} | "
                f"micro_f1={tmp_stats['micro_f1']:.4f} | "
                f"macro_f1={tmp_stats['macro_f1']:.4f} | "
                f"micro_prec={tmp_stats['micro_precision']:.4f} | "
                f"micro_rec={tmp_stats['micro_recall']:.4f} | "
                f"pos(y)={pos_rate_y:.3f} | "
                f"pos(pred)={pos_rate_pred:.3f}"
            )

    train_loss /= max(len(train_dl.dataset), 1)

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    train_stats = f1_stats_from_logits(
        logits_all,
        targets_all,
        thr=threshold,
    )
    train_stats = add_derived_rates(train_stats)
    train_stats["loss"] = train_loss

    return train_stats

@torch.no_grad()
def evaluate_cnn(model, data_loader, criterion, device, threshold):
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for X, y in data_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    stats = f1_stats_from_logits(
        logits_all,
        targets_all,
        thr=threshold,
    )
    stats = add_derived_rates(stats)
    stats["loss"] = total_loss / max(len(data_loader.dataset), 1)

    return stats, logits_all, targets_all

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("Device:", device)

    # Paths
    index_jsonl = "data/processed/index.jsonl"
    splits_json = "data/processed/splits.json"
    out_dir = Path("models/cnn/runs/run3_fewshot")
    out_dir.mkdir(parents=True, exist_ok=True)
    # save results
    metrics_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"

    write_header = not csv_path.exists()

    # Config
    SHOTS_PER_CLASS = 1
    MAX_NEGATIVES = 25
    SEED = 42

    cfg = SpecConfig(
        win_seconds=1.0,
        tol_frames=3,
        label_mode="center",
    )

    train_ids, val_ids, test_ids = load_splits(splits_json)

    #window_frames = int(round(cfg.win_seconds * cfg.sr / cfg.hop_length))
    #stride_frames = max(1, window_frames // 2)

    """
    # Dataset, p_pos to avoid center class: none
    train_ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=train_ids,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=8,
        seed=42,
        p_pos=0.6,
    )
    """
    base_train_ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=train_ids,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=1,
        seed=SEED,
        p_pos=0.9,
    )
    train_ds = FewShotMultiLabelDataset(
        base_dataset=base_train_ds,
        shots_per_class=SHOTS_PER_CLASS,
        max_negatives=MAX_NEGATIVES,
        seed=SEED,
    )
    BATCH_SIZE = 128

    print("\nFew-shot configuration:")
    print("  shots_per_class:", SHOTS_PER_CLASS)
    print("  max_negatives:", MAX_NEGATIVES)
    print("  seed:", SEED)
    print("  final train windows:", len(train_ds))
    print("  batch_size:", BATCH_SIZE)
    print("  batches per epoch:", math.ceil(len(train_ds) / BATCH_SIZE))

    val_ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=val_ids,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=8,
        seed=123,
        p_pos=0.5,
    )

    test_ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=test_ids,
        cfg=cfg,
        sampling="random",
        max_windows_per_track=8,
        seed=456,
        p_pos=0.5,
    )

    print("Tracks train:", len(base_train_ds.rows))
    print("Samples train:", len(train_ds), "=> batches/epoch:", len(train_ds) // 128)
    print("Tracks val:", len(val_ds.rows))
    print("Tracks test:", len(test_ds.rows))
    print("Example row keys:", base_train_ds.rows[0].keys())
    print("Example mel path:", base_train_ds.rows[0].get("mel"))


    pin_memory = (device == "cuda")

    # DataLoader gets X and y and turns it into a Batch X = [B, 1, 80, 86]
    # B: number of windows, 32 
    # 1: channels
    # 80: Mel bands
    # 86: frames per window
    # and y = [B, C] where C = len(classes)
    train_dl = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=pin_memory,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
    )

    model = DrumCNN(num_classes=len(CLASSES), dropout=0.3).to(device)   

    """
    pw_dl = DataLoader(pw_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=pin_memory)

    pos_weight, pos_rate = estimate_pos_weight_from_loader(pw_dl, num_classes=len(CLASSES), max_batches=400)

    # clamp pos_weight to avoid too large values which can cause instability in training
    pos_weight = torch.log1p(pos_weight)
    """

    pw_dl = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    pos_weight, pos_rate = estimate_pos_weight_from_loader(
        pw_dl,
        num_classes=len(CLASSES),
        max_batches=9999,
    )

    pos_weight = torch.log1p(pos_weight).clamp(max=8.0)

    print("Estimated pos_rate per class:", {CLASSES[i]: float(pos_rate[i]) for i in range(len(CLASSES))})
    print("Using pos_weight:", {CLASSES[i]: float(pos_weight[i]) for i in range(len(CLASSES))})

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # Optimizes weights
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
    )

    best_val_f1 = -1.0
    epochs = 50
    patience = 20
    bad_epochs = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        epoch_start = time.time()

        """
        if epoch <= 5:
            train_ds.p_pos = 0.8
        elif epoch <= 10:
            train_ds.p_pos = 0.5
        else:
            train_ds.p_pos = 0.3
        """

        # ---- train
        train_stats = train_one_epoch_cnn(
            model=model,
            train_dl=train_dl,
            criterion=criterion,
            optim=optim,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            threshold=0.5,      # durante training usa 0.5, como referencia estable
            log_every=1,        # few-shot: normalmente hay pocos batches
        )

        train_loss = train_stats["loss"]
        train_micro_f1 = train_stats["micro_f1"]
        train_macro_f1 = train_stats["macro_f1"]

        # ---- val
        model.eval()
        val_loss = 0.0

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * X.size(0)

                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())

        val_loss /= len(val_ds)

        logits_val = torch.cat(all_logits, dim=0)    # [N, C]
        targets_val = torch.cat(all_targets, dim=0)  # [N, C]

        #bad results with per-class thresholds, maybe due to small validation set? or because it optimizes micro-F1 which is more affected by common classes?
        thr_c, best_f1_c = find_best_threshold_per_class(logits_val, targets_val)

        #best_thr, best_micro, best_macro = find_best_threshold(logits_val, targets_val, optimize="micro")


        #thr_fixed = best_thr
        thr = float(thr_c.mean())
        stats = f1_stats_from_logits(logits_val, targets_val, thr=thr_c)
        stats = add_derived_rates(stats)

        val_micro_f1 = stats["micro_f1"]
        val_macro_f1 = stats["macro_f1"]

        micro_prec = stats["micro_precision"]
        micro_rec = stats["micro_recall"]
        tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]

        thr_by_name = {CLASSES[i]: float(thr_c[i]) for i in range(len(CLASSES))}

        score = val_macro_f1
        scheduler.step(score)

        # positive ratio
        probs_val = torch.sigmoid(logits_val)
        preds_val = (probs_val >= thr_c).to(targets_val.dtype)

        pos_rate_y = targets_val.mean().item()
        pos_rate_pred = preds_val.mean().item()

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_microF1={train_micro_f1:.4f} | "
            f"train_macroF1={train_macro_f1:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_microF1={val_micro_f1:.4f} | "
            f"val_macroF1={val_macro_f1:.4f} | "
            f"pos(y)={pos_rate_y:.3f} | "
            f"pos(pred)={pos_rate_pred:.3f} | "
            f"lr={optim.param_groups[0]['lr']:.1e} | "
            f"time={epoch_time:.2f}s"
        )

        # record metrics
        prec_by_name = {CLASSES[i]: stats["precision_by_class"][i] for i in range(len(CLASSES))}
        rec_by_name  = {CLASSES[i]: stats["recall_by_class"][i]    for i in range(len(CLASSES))}
        f1_by_name   = {CLASSES[i]: stats["f1_by_class"][i]        for i in range(len(CLASSES))}



        metrics_row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,

            "train_loss": train_loss,
            "train_microF1": train_micro_f1,
            "train_macroF1": train_macro_f1,

            "val_loss": val_loss,
            "val_microF1": val_micro_f1,
            "val_macroF1": val_macro_f1,

            "pos_rate_y": pos_rate_y,
            "pos_rate_pred": pos_rate_pred,
            "threshold": thr,
            "precision_by_name": prec_by_name,
            "recall_by_name": rec_by_name,
            "f1_by_name": f1_by_name,
            "lr": optim.param_groups[0]["lr"],
            "threshold_by_class": thr_by_name,
            **stats,
        }


        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row) + "\n")

        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                base_cols = [
                    "epoch","epoch_time_sec","train_loss","val_loss","val_microF1","val_macroF1",
                    "pos_rate_y","pos_rate_pred","threshold","lr","micro_precision","micro_recall",
                    "tp","fp","fn","tn"
                ]
                per_class_cols = []
                for name in CLASSES:
                    per_class_cols += [f"tp_{name}", f"fp_{name}", f"fn_{name}", f"tn_{name}"]
                    # per_class_cols += [f"prec_{name}", f"rec_{name}", f"f1_{name}"]
                f.write(",".join(base_cols + per_class_cols) + "\n")
                write_header = False

            # --- each epoch row
            row_vals = [
                epoch, epoch_time, train_loss, val_loss, val_micro_f1, val_macro_f1,
                pos_rate_y, pos_rate_pred, thr, optim.param_groups[0]["lr"],
                micro_prec, micro_rec, tp, fp, fn, tn,
            ]

            # add per-class confusion values
            for i, name in enumerate(CLASSES):
                row_vals += [
                    stats["tp_by_class"][i],
                    stats["fp_by_class"][i],
                    stats["fn_by_class"][i],
                    stats["tn_by_class"][i],
                ]

            f.write(",".join(map(str, row_vals)) + "\n")

        # Save best
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            bad_epochs = 0

            best_thresholds = thr_c.clone()

            test_stats, _, _ = evaluate_cnn(
                model=model,
                data_loader=test_dl,
                criterion=criterion,
                device=device,
                threshold=best_thresholds,
            )

            test_f1_by_name = {
                CLASSES[i]: test_stats["f1_by_class"][i]
                for i in range(len(CLASSES))
            }

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "classes": CLASSES,

                "shots_per_class": SHOTS_PER_CLASS,
                "max_negatives": MAX_NEGATIVES,
                "seed": SEED,

                "val_microF1": val_micro_f1,
                "val_macroF1": val_macro_f1,

                "test_loss": test_stats["loss"],
                "test_microF1": test_stats["micro_f1"],
                "test_macroF1": test_stats["macro_f1"],
                "test_micro_precision": test_stats["micro_precision"],
                "test_micro_recall": test_stats["micro_recall"],
                "test_f1_by_name": test_f1_by_name,

                "threshold": thr,
                "threshold_by_class": {
                    CLASSES[i]: float(best_thresholds[i])
                    for i in range(len(CLASSES))
                },

                "total_training_time_sec": epoch_time,
                "lr": optim.param_groups[0]["lr"],
            }

            torch.save(ckpt, out_dir / "best.pt")

            print(
                f"Saved best.pt | "
                f"val_macroF1={val_macro_f1:.4f} | "
                f"test_macroF1={test_stats['macro_f1']:.4f} | "
                f"test_microF1={test_stats['micro_f1']:.4f}"
            )
            test_metrics_path = out_dir / "best_test_metrics.json"

            with open(test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "test_stats": test_stats,
                        "threshold_by_class": {
                            CLASSES[i]: float(best_thresholds[i])
                            for i in range(len(CLASSES))
                        },
                        "test_f1_by_name": test_f1_by_name,
                    },
                    f,
                    indent=2,
                )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: isn't improving for {patience} epochs.")
                break

    total_time = time.time() - start_time

    print(f"Total training time: {total_time:.2f} seconds "
        f"({total_time/60:.2f} minutes)")

    print("Best val_microF1 =", best_val_f1)


if __name__ == "__main__":
    main()
