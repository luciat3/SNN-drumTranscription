import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cnn import model
from models.snn.dataset_TIMIT import TimitFramewiseDataset, timit_collate_fn
from models.snn.model import LSNN


def prepare_timit_batch(features, labels, lengths, repeat_steps=5):
    """
    features: [B, T_frames, F]
    labels:   [B, T_frames]
    lengths:  [B] in frame units

    Returns:
        x_seq:          [T_ms, B, F]   repeated features
        targets_frame:  [T_frames, B]  original frame labels
        mask_frame:     [T_frames, B]  original weighted mask
    """
    B, T_frames, F = features.shape

    # Frame-level weighted mask, matching the TF code
    time_idx = torch.arange(T_frames, device=features.device).unsqueeze(0)   # [1, T_frames]
    valid = (time_idx < lengths.unsqueeze(1)).float()                        # [B, T_frames]
    weighted_mask = valid / lengths.unsqueeze(1).float()                     # [B, T_frames]

    # Repeat only features to ms resolution
    if repeat_steps > 1:
        features = features.repeat_interleave(repeat_steps, dim=1)           # [B, T_ms, F]

    x_seq = features.transpose(0, 1).contiguous()                            # [T_ms, B, F]
    targets_frame = labels.transpose(0, 1).contiguous()                      # [T_frames, B]
    weighted_mask = weighted_mask.transpose(0, 1).contiguous()               # [T_frames, B]

    return x_seq, targets_frame, weighted_mask

def downsample_repeated_time(z_rec_ms, repeat_steps):
    """
    z_rec_ms: [T_ms, B, N]
    returns:  [T_frames, B, N]
    """
    T_ms, B, N = z_rec_ms.shape
    assert T_ms % repeat_steps == 0, (
        f"T_ms={T_ms} not divisible by repeat_steps={repeat_steps}"
    )
    T_frames = T_ms // repeat_steps
    z_ds = z_rec_ms.view(T_frames, repeat_steps, B, N).mean(dim=1)
    return z_ds

def masked_metrics(y_out, targets, weighted_mask, z_rec, dt=1e-3):
    T, B, n_out = y_out.shape

    loss_per_frame = F.cross_entropy(
        y_out.reshape(T * B, n_out),
        targets.reshape(T * B),
        reduction="none"
    ).reshape(T, B)

    # Original-style weighted loss: sum over time, mean over batch
    loss_pred = (loss_per_frame * weighted_mask).sum(dim=0).mean()

    preds = y_out.argmax(dim=-1)  # [T, B]
    acc = (((preds == targets).float()) * weighted_mask).sum(dim=0).mean()
    ler = 1.0 - acc

    # Per-neuron firing rate in Hz
    fr_per_neuron = z_rec.mean(dim=(0, 1)) / dt
    fr_avg = fr_per_neuron.mean()
    fr_max = fr_per_neuron.max()

    spike_rate_hz = z_rec.mean().item() / dt

    return {
        "loss_pred": float(loss_pred.item()),
        "acc": float(acc.item()),
        "ler": float(ler.item()),
        "fr_avg": float(fr_avg.item()),
        "fr_max": float(fr_max.item()),
        "spike_rate_hz": float(spike_rate_hz),
    }


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    metrics,
    epoch,
    run_start_time,
    repeat_steps=5,
    log_every=10
):
    model.train()

    totals = {
        "loss": 0.0,
        "loss_pred": 0.0,
        "acc": 0.0,
        "ler": 0.0,
        "fr_avg": 0.0,
        "fr_max": 0.0,
        "spike_rate_hz": 0.0,
    }
    n_batches = 0
    last_grad_nonzero = 0.0

    epoch_start = time.time()

    for batch_idx, (features, labels, lengths) in enumerate(train_loader):
        batch_start = time.time()

        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        x_seq, targets, weighted_mask = prepare_timit_batch(
            features, labels, lengths, repeat_steps=repeat_steps
        )

        z_rec_ms = model(x_seq)                            # [T_ms, B, n_rec]
        z_rec_frame = downsample_repeated_time(z_rec_ms, repeat_steps=repeat_steps)   # [T_frames, B, n_rec]
        y_out = model.readout_from_spikes(z_rec_frame)    # [T_frames, B, n_out]

        loss = model.eprop_update(
            x_seq=x_seq,
            y_out=y_out,
            z_rec=z_rec_ms,
            targets=targets,
            weighted_mask=weighted_mask,
            optimizer=optimizer,
        )

        with torch.no_grad():
            m = masked_metrics(y_out, targets, weighted_mask, z_rec_ms)

            if model.alif.w_rec is not None and model.alif.w_rec.grad is not None:
                last_grad_nonzero = float((model.alif.w_rec.grad != 0).float().mean().item())
            else:
                last_grad_nonzero = 0.0

        totals["loss"] += float(loss)
        totals["loss_pred"] += m["loss_pred"]
        totals["acc"] += m["acc"]
        totals["ler"] += m["ler"]
        totals["fr_avg"] += m["fr_avg"]
        totals["fr_max"] += m["fr_max"]
        totals["spike_rate_hz"] += m["spike_rate_hz"]
        n_batches += 1

        if batch_idx % log_every == 0:
            elapsed = time.time() - epoch_start
            batch_time = time.time() - batch_start
            lr = optimizer.param_groups[0]["lr"]

            global_iteration = epoch * len(train_loader) + batch_idx

            metrics["loss_list"].append(float(loss))
            metrics["train_ler_list"].append(m["ler"])
            metrics["iteration_list"].append(global_iteration)
            metrics["epoch_list"].append(epoch)
            metrics["training_time_list"].append(time.time() - run_start_time)
            metrics["fr_max_list"].append(m["fr_max"])
            metrics["fr_avg_list"].append(m["fr_avg"])

            print(
                f"  batch {batch_idx:4d}/{len(train_loader)-1:4d} | "
                f"loss={loss:.4f} | "
                f"loss_pred={m['loss_pred']:.4f} | "
                f"ler={m['ler']:.4f} | "
                f"fr_avg={m['fr_avg']:.2f}Hz | "
                f"fr_max={m['fr_max']:.2f}Hz | "
                f"grad_nonzero={last_grad_nonzero:.3f} | "
                f"lr={lr:.5f} | "
                f"batch_time={batch_time:.2f}s | "
                f"elapsed={elapsed:.1f}s"
            )

    for k in totals:
        totals[k] /= n_batches

    totals["grad_nonzero"] = last_grad_nonzero
    return totals

@torch.no_grad()
def evaluate(model, data_loader, device, repeat_steps=5):
    model.eval()

    totals = {
        "loss_pred": 0.0,
        "acc": 0.0,
        "ler": 0.0,
        "fr_avg": 0.0,
        "fr_max": 0.0,
        "spike_rate_hz": 0.0,
    }
    n_batches = 0

    for features, labels, lengths in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        x_seq, targets, weighted_mask = prepare_timit_batch(
            features, labels, lengths, repeat_steps=repeat_steps
        )

        z_rec_ms = model(x_seq)                            # [T_ms, B, n_rec]
        z_rec_frame = downsample_repeated_time(z_rec_ms, repeat_steps=repeat_steps)   # [T_frames, B, n_rec]
        y_out = model.readout_from_spikes(z_rec_frame)    # [T_frames, B, n_out]
        m = masked_metrics(y_out, targets, weighted_mask, z_rec_ms)

        totals["loss_pred"] += m["loss_pred"]
        totals["acc"] += m["acc"]
        totals["ler"] += m["ler"]
        totals["fr_avg"] += m["fr_avg"]
        totals["fr_max"] += m["fr_max"]
        totals["spike_rate_hz"] += m["spike_rate_hz"]
        n_batches += 1

    for k in totals:
        totals[k] /= n_batches

    return totals


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "data/processed/timit_processed"
    out_path = Path("models/snn/metrics_pytorch_timit.json")

    train_dataset = TimitFramewiseDataset(root=root, split="train")
    develop_dataset = TimitFramewiseDataset(root=root, split="develop")
    test_dataset = TimitFramewiseDataset(root=root, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=timit_collate_fn,
    )

    develop_loader = DataLoader(
        develop_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=timit_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=timit_collate_fn,
    )


    model = LSNN(
        n_in=39,          # mfccs run
        n_regular=300,
        n_adaptive=100,
        n_out=61,
        tau_out=3.0,
        dt=1.0,
        beta=0.184,
        tau_m=20.0,
        tau_a=200.0,
        thr=0.60,
        dampening_factor=0.3,
        n_refractory=2,
        rec=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-5)

    metrics = {
        "loss_list": [],
        "ler_list": [],
        "ler_test_list": [],
        "train_ler_list": [],
        "n_synapse": [],
        "iteration_list": [],
        "epoch_list": [],
        "training_time_list": [],
        "fr_max_list": [],
        "fr_avg_list": [],
        "early_stopping_test_ler0": None,
    }

    global_step = 0
    best_test_ler = None
    run_start_time = time.time()

    for epoch in range(80):
        print(f"\n=== Epoch {epoch:3d} ===")
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            metrics=metrics,
            epoch=epoch,
            run_start_time=run_start_time,
            repeat_steps=5,
            log_every=10,
        )     
        dev_metrics = evaluate(model, develop_loader, device, repeat_steps=5)
        test_metrics = evaluate(model, test_loader, device, repeat_steps=5)

        epoch_time = time.time() - t0
        global_step += len(train_loader)


        metrics["loss_list"].append(train_metrics["loss"])
        metrics["ler_list"].append(dev_metrics["ler"])
        metrics["ler_test_list"].append(test_metrics["ler"])
        metrics["n_synapse"].append([])
        metrics["iteration_list"].append(global_step)
        metrics["epoch_list"].append(epoch)
        metrics["training_time_list"].append(epoch_time)
        metrics["fr_max_list"].append(train_metrics["fr_max"])
        metrics["fr_avg_list"].append(train_metrics["fr_avg"])

        if best_test_ler is None or test_metrics["ler"] < best_test_ler:
            best_test_ler = test_metrics["ler"]
        metrics["early_stopping_test_ler0"] = best_test_ler

        with out_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_loss_pred={train_metrics['loss_pred']:.4f} | "
            f"train_ler={train_metrics['ler']:.4f} | "
            f"train_fr_avg={train_metrics['fr_avg']:.2f}Hz | "
            f"train_fr_max={train_metrics['fr_max']:.2f}Hz | "
            f"dev_ler={dev_metrics['ler']:.4f} | "
            f"test_ler={test_metrics['ler']:.4f} | "
            f"grad_nonzero={train_metrics['grad_nonzero']:.3f} | "
            f"lr={lr:.5f} | "
            f"time={epoch_time:.2f}s"
        )

    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    train()