import torch
import torch.nn as nn
from models.snn.model import LSNN

# ---------------------------------------------------------------------------------
# 1) Toy dataset
# ---------------------------------------------------------------------------------
# Corresponds to the simple pattern task described in supp. note 6 / fig 7
# but simplified to binary classification.
# Class 0: input neuron 0 fires at high rate, neuron 1 at low rate
# Class 1: input neuron 0 fires at low rate,  neuron 1 at high rate

def make_batch(batch_size, T=50, n_in=10, device='cpu'):
    """
    Generates Poisson spike trains.
    Half the batch is class 0, half is class 1.
    
    Returns:
        x:       [T, B, n_in]  binary spike trains
        targets: [T, B]        integer labels (same label at every time step,
                               matching the framewise TIMIT setup, supp. p.18)
    """
    labels = torch.randint(0, 2, (batch_size,), device=device)  # [B]

    # Base firing rates — class 0: [0.8, 0.1, ...], class 1: [0.1, 0.8, ...]
    rates = torch.zeros(batch_size, n_in, device=device)
    rates[labels == 0, 0] = 0.8   # neuron 0 fires fast for class 0
    rates[labels == 0, 1] = 0.1
    rates[labels == 1, 0] = 0.1
    rates[labels == 1, 1] = 0.8   # neuron 1 fires fast for class 1
    rates[:, 2:] = 0.1            # background noise for remaining neurons

    # Sample Poisson spikes: x~Bernoulli(rate) at each time step
    rates_expanded = rates.unsqueeze(0).expand(T, -1, -1)  # [T, B, n_in]
    x = torch.bernoulli(rates_expanded)

    # Framewise targets: same label repeated at every time step
    # This matches eq. (29) where loss is summed over all t
    targets = labels.unsqueeze(0).expand(T, -1)            # [T, B]

    return x, targets


# ---------------------------------------------------------------------------------
# 2) Training loop
# ---------------------------------------------------------------------------------

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters from supp. p.18 (scaled down for toy task)
    T        = 50     # sequence length (ms)
    B        = 32     # batch size
    n_in     = 10     # input neurons
    n_regular    = 30     # recurrent neurons (paper uses 300+100 for TIMIT)
    n_adaptive   = 20     # recurrent neurons with adaptation (ALIF)
    n_out    = 2      # output classes (paper uses 61 for TIMIT)
    n_epochs = 200

    model = LSNN(
        n_in=n_in,
        n_regular=n_regular,
        n_adaptive=n_adaptive,
        n_out=n_out,
        tau_out=3.0,
        dt=1.0,
        # ALIFStep kwargs:
        tau_m=20.0,
        tau_a=200.0,
        thr=0.45,
        beta=0.184,
        dampening_factor=0.3,
        n_refractory=2,
        rec=True,
    ).to(device)

    # Adam with lr=0.01 — supp. p.18 "learning rate was initialized to 0.01"
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for epoch in range(n_epochs):
        x, targets = make_batch(B, T=T, n_in=n_in, device=device)

        # Forward pass — lsnn.py LSNN.forward()
        y_out, z_rec, e_bar = model(x)
        # y_out:   [T, B, n_out]
        # z_rec:   [T, B, n_rec]
        # e_bar:   [T, B, n_rec, n_rec]

        # E-prop weight update — lsnn.py LSNN.eprop_update()
        loss = model.eprop_update(y_out, z_rec, targets, e_bar, optimizer)

        # ---- diagnostics ----
        if epoch % 10 == 0:
            with torch.no_grad():
                # Accuracy: take prediction at last time step
                # (or mean over time — both valid for framewise setup)
                preds = y_out[-1].argmax(dim=-1)        # [B]
                acc = (preds == targets[-1]).float().mean().item()

                # Spike rate — should be 10-20Hz (not silent, not saturated)
                # Paper enforces this via firing rate regularization, supp. eq. (5)
                mean_rate = z_rec.mean().item() / 0.001  # convert to Hz (dt=1ms)

                # Fraction of neurons with nonzero gradient — health check
                if model.alif.w_rec.grad is not None:
                    grad_nonzero = (model.alif.w_rec.grad != 0).float().mean().item()
                else:
                    grad_nonzero = 0.0

            print(f"Epoch {epoch:4d} | loss={loss:.4f} | acc={acc:.3f} | "
                  f"spike_rate={mean_rate:.1f}Hz | grad_nonzero={grad_nonzero:.3f}")

    print("\nDone. Expected outcome:")
    print("  loss should decrease from ~0.69 (random) toward ~0.1")
    print("  acc  should increase from ~0.50 toward ~0.95+")
    print("  spike_rate should stay between 5-50 Hz")
    print("  grad_nonzero should be > 0 from epoch 0")


if __name__ == '__main__':
    train()