# -*- coding: utf-8 -*-
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict, Tuple

from models.cnn.dataset import DrumOnsetWindowDataset, SpecConfig, CLASSES
from models.cnn.model import DrumCNN

import time



def load_splits(splits_path: str):
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    return data["train"], data["val"], data["test"]


@torch.no_grad()
def f1_stats_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
) -> Tuple[float, float, Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Returns:
      micro_f1, macro_f1,
      class precission, class recall, class f1
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).to(targets.dtype)

    eps = 1e-8
    C = targets.shape[1]

    tp_c = (preds * targets).sum(dim=0)
    fp_c = (preds * (1 - targets)).sum(dim=0)
    fn_c = ((1 - preds) * targets).sum(dim=0)

    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c  = tp_c / (tp_c + fn_c + eps)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    # micro
    tp = tp_c.sum()
    fp = fp_c.sum()
    fn = fn_c.sum()
    micro_f1 = (2 * tp / (2 * tp + fp + fn + eps)).item()

    # macro
    macro_f1 = f1_c.mean().item()

    prec_dict = {i: prec_c[i].item() for i in range(C)}
    rec_dict  = {i: rec_c[i].item()  for i in range(C)}
    f1_dict   = {i: f1_c[i].item()   for i in range(C)}
    return micro_f1, macro_f1, prec_dict, rec_dict, f1_dict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Paths
    index_jsonl = "data/processed/index.jsonl"
    splits_json = "data/processed/splits.json"
    out_dir = Path("models/cnn/runs/run1")
    out_dir.mkdir(parents=True, exist_ok=True)
    # save results
    metrics_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"

    write_header = not csv_path.exists()

    # Config
    cfg = SpecConfig(win_seconds=1.0, tol_frames=1)
    train_ids, val_ids, _ = load_splits(splits_json)

    # Dataset, p_pos to avoid center class: none
    train_ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=train_ids,
        cfg=cfg,
        max_windows_per_track=8,
        seed=42,
        p_pos=0.8,
    )
    val_ds = DrumOnsetWindowDataset(
        index_jsonl=index_jsonl,
        ids=val_ids,
        cfg=cfg,
        max_windows_per_track=4,
        seed=123,
        p_pos=0.8,
    )

    pin_memory = (device == "cuda")

    # DataLoader gets X and y and turns it into a Batch X = [B, 1, 80, 86]
    # B: number of windows, 32 
    # 1: channels
    # 80: Mel bands
    # 86: frames per window
    # and y = [B, C] where C = len(classes)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=pin_memory)

    model = DrumCNN(num_classes=len(CLASSES), dropout=0.3).to(device)           

    criterion = nn.BCEWithLogitsLoss()
    # Optimizes weights
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_f1 = -1.0
    epochs = 30
    patience = 6
    bad_epochs = 0
    thr = 0.5

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        epoch_start = time.time()

        # ---- train
        model.train()
        train_loss = 0.0
        # iterates batches
        for X, y in train_dl:
            X = X.to(device)               # [B,1,80,86]
            y = y.to(device)               # [B,8]

            # reset gradient
            optim.zero_grad(set_to_none=True)

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            # adjusts weight using gradient
            optim.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_ds)

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

        val_micro_f1, val_macro_f1, prec_dict, rec_dict, f1_dict = f1_stats_from_logits(
            logits_val, targets_val, thr=thr
        )

        # positive ratio
        probs_val = torch.sigmoid(logits_val)
        preds_val = (probs_val >= thr).to(targets_val.dtype)

        pos_rate_y = targets_val.mean().item()
        pos_rate_pred = preds_val.mean().item()

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_microF1={val_micro_f1:.4f} | val_macroF1={val_macro_f1:.4f} | "
            f"pos(y)={pos_rate_y:.3f} pos(pred)={pos_rate_pred:.3f} | "
            f"time={epoch_time:.2f}s"
        )

        # record metrics
        prec_by_name = {CLASSES[i]: prec_dict[i] for i in range(len(CLASSES))}
        rec_by_name  = {CLASSES[i]: rec_dict[i]  for i in range(len(CLASSES))}
        f1_by_name   = {CLASSES[i]: f1_dict[i]   for i in range(len(CLASSES))}

        metrics_row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_microF1": val_micro_f1,
            "val_macroF1": val_macro_f1,
            "pos_rate_y": pos_rate_y,
            "pos_rate_pred": pos_rate_pred,
            "threshold": thr,
            "precision_by_class": prec_by_name,
            "recall_by_class": rec_by_name,
            "f1_by_class": f1_by_name,
        }

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row) + "\n")

        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,epoch_time_sec,train_loss,val_loss,val_microF1,val_macroF1,pos_rate_y,pos_rate_pred,threshold\n")
                write_header = False
            f.write(
                f"{epoch},{epoch_time},{train_loss},{val_loss},{val_micro_f1},{val_macro_f1},{pos_rate_y},{pos_rate_pred},{thr}\n"
            )

        # Save best
        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            bad_epochs = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "classes": CLASSES,
                "val_microF1": best_val_f1,
                "val_macroF1": val_macro_f1,
                "threshold": thr,
                "total_training_time_sec": epoch_time,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best.pt (val_microF1={best_val_f1:.4f})")
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
