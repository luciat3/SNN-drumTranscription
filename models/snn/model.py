import torch
import torch.nn as nn
from models.snn.alif import ALIFStep
from models.snn.eprop import compute_psi, update_alif_traces_recurrent, exp_convolve
import torch.nn.functional as F

class LSNN(nn.Module):
    """
    Single forward LSNN with adaptive e-prop.
    The weight update is defined in eq. (29) from the original paper:
    ΔW_ji = η · Σ_t  L^t_j · ē^t_ji
    where L^t_j is the learning signal (eq. (4)) and ē^t_ji is the filtered 
    elegibility trace.
    """
    def __init__(
            self,
            n_in, # number of input neurons
            n_regular,
            n_adaptive,
            n_out, # number of output neurons
            tau_out=3.0, # output neuron time constant (ms)
            dt=1.0, # time step (ms)
            beta=0.184, # adaptation strength for ALIF neurons
            **alif_kwargs # additional arguments for ALIF neuron
    ):
        
        super().__init__()

        self.n_regular = n_regular
        self.n_adaptive = n_adaptive
        self.n_rec = n_regular + n_adaptive
        self.n_out = n_out

        # Build beta weights for ALIF neurons, set to zero for regular LIF neurons
        beta_vec = [0.0] * n_regular + [float(beta)] * n_adaptive

        # Output neuron decay factor for eq. (11)
        self.kappa = torch.exp(torch.tensor(-dt / tau_out))

        # ALIF recurrent layer
        self.alif = ALIFStep(n_in=n_in, n_rec=self.n_rec, dt=dt, beta=beta_vec, **alif_kwargs)

        # Output weights for eq. (11)
        # Random initialization with scaling factor for better convergence
        self.w_out = nn.Parameter(torch.randn(self.n_rec, self.n_out) / (self.n_rec ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(n_out))

        # Broadcast weights, used for learning signal computation in eq. (4)
        # Randomly initialized with scaling factor for adaptive e-prop
        self.B = nn.Parameter(torch.randn(self.n_rec, n_out) / (self.n_rec ** 0.5), requires_grad=False)

    def forward(self, x_seq):
        """
        Forward pass through the LSNN for a sequence of inputs.
        x_seq: [T, B, n_in] input sequence of length T, batch size B
        """
        T, B, _ = x_seq.shape
        device = x_seq.device

        # Initialise state
        state = self.alif.zero_state(B, device=device)

        """
        # Elegibility trace accumulators
        # z_bar: filtered pre-synaptic activity for recurrent connections (eq. (22))
        z_bar = torch.zeros(B, self.n_rec, device=device)
        # eps_a: elegibility vector for the adaptation variable (eq. (24))
        eps_a = torch.zeros(B, self.n_rec, self.n_rec, device=device)
        # e_bar: filtered elegibility trace for output weights (used in eq. (29))
        e_bar = torch.zeros(B, self.n_rec, self.n_rec, device=device)
        """
        # Leaky output neuron state for eq. (11)
        y_prev = torch.zeros(B, self.n_out, device=device)

        all_y, all_z = [], []

        for t in range(T):

            # 1) ALIF step forward
            new_state, _ = self.alif(x_seq[t], state)
            z = new_state["z"]  # Spike output at time t -> [B, n_rec]

            # 2) Leaky output neuron update eq. (11)
            y = self.kappa.to(device) * y_prev + z @ self.w_out + self.b_out

            """
            # 3) Compute pseudo-derivative for learning signal and elegibility trace updates
            psi, _, _ = compute_psi(state["v"], state["a"], self.alif.thr, self.alif.beta, self.alif.dampening_factor)

            # 4) Update elegibility traces for recurrent weights -> eqs. (24), (25)
            z_bar, eps_a, e_trace = update_alif_traces_recurrent(
                z_bar=z_bar, eps_a=eps_a, z_pre=state["z"], psi=psi, alpha=self.alif.alpha, rho=self.alif.rho, beta=self.alif.beta
            )
            
            # 5) Update filtered elegibility trace for output weights
            e_bar = self.kappa.to(device) * e_bar + e_trace
            """
            # Store for later use in learning signal computation
            state = new_state
            y_prev = y
            all_y.append(y)
            all_z.append(z)
            #all_ebar.append(e_bar)

        return (torch.stack(all_y), torch.stack(all_z))
        
    def eprop_update(self, x_seq, y_out, z_rec, targets, weighted_mask, optimizer, reg=5e-6, f_target_hz=10.0, dt=1e-3):
        """
        Compute learning signal and perform weight update using e-prop.

        y_out: [T, B, n_out] output sequence from forward pass
        z_rec: [T, B, n_rec] recurrent spike sequence from forward pass
        targets: [T, B] target sequence for supervised learning
        e_bar: [T, B, n_rec, n_rec] elegibility traces from forward pass

        Implements eq. (29):
            ΔW_ji = -η · Σ_t  L^t_j · ē^t_ji
        with learning signal eq. (4):
            L^t_j = Σ_k B_jk · (π^t_k - π*^t_k)
        
        Corresponds to the weight update block in solve_timit_with_framewise_lsnn.py,
        using broadcast weights instead of symmetric feedback.
        """
        T, B, _ = y_out.shape
        device = y_out.device

        #print("y_out shape:", y_out.shape)
        #print("targets shape:", targets.shape)
        #print("e_bar shape:", e_bar.shape)

        # π^t_k — softmax over readout, as used in eq. (29), is the probability for class k
        pi = torch.softmax(y_out, dim=-1)

        # π*^t_k — target one-hot encoding
        pi_star = torch.zeros_like(pi)
        pi_star.scatter_(-1, targets.unsqueeze(-1), 1.0)

        # Compute learning signal L^t_j for each recurrent neuron j at each time step t (eq. (4))
        error = pi - pi_star  # [T, B, n_out]
        error = error * weighted_mask.unsqueeze(-1)
        L = (error @ self.B.T).detach()  # [T, B, n_rec]

        # Compute elegibility traces online
        state = self.alif.zero_state(B, device=device)
        z_bar = torch.zeros(B, self.n_rec, device=device)
        eps_a = torch.zeros(B, self.n_rec, self.n_rec, device=device)
        e_bar = torch.zeros(B, self.n_rec, self.n_rec, device=device)

        grad_w_rec = torch.zeros_like(self.alif.w_rec) if self.alif.w_rec is not None else None
        
        with torch.no_grad():
            for t in range(T):
                # pseudo-derivative from current state
                psi, _, _ = compute_psi(state["v"], state["a"], self.alif.thr, self.alif.beta, self.alif.dampening_factor)
                # recurrent elegibility traces
                z_bar, eps_a, e_trace = update_alif_traces_recurrent(
                    z_bar=z_bar, eps_a=eps_a, z_pre=state["z"], psi=psi, alpha=self.alif.alpha, rho=self.alif.rho, beta=self.alif.beta
                )
                # filtered recurrent trace
                e_bar = self.kappa.to(device) * e_bar + e_trace

                # acumulate recurrent gradient online
                L_t = L[t]  # [B, n_rec]
                grad_w_rec = grad_w_rec - (L_t.unsqueeze(-1) * e_bar).sum(dim=0)  # [n_rec, n_rec]

                state, _ = self.alif(x_seq[t], state)

        loss_per_frame = F.cross_entropy(
                y_out.reshape(T * B, -1),
                targets.reshape(T * B),
                reduction="none"
            ).reshape(T, B)

        loss_pred = (loss_per_frame * weighted_mask).sum(dim=0).mean()
        
        # Average firing rate
        f_av = z_rec.mean(dim=(0, 1)) / dt  # [n_rec]

        f_target = torch.full_like(f_av, f_target_hz)
        loss_reg = 0.5 * reg * torch.sum((f_av - f_target) ** 2)

        loss = loss_pred + loss_reg

        # Optimizer applies the learning rate η, in the paper they use ADAM with η=10^⁻5
        optimizer.zero_grad()
        loss.backward()

        # Override gradients for recurrent weights with e-prop computed gradients
        if self.alif.w_rec is not None:
            self.alif.w_rec.grad = grad_w_rec / B

        # Adaptive e-prop mirrors the W update to B
        with torch.no_grad():
            if self.w_out.grad is not None:
                lr = optimizer.param_groups[0]["lr"]
                c_decay = 1e-2 # from supp. note 2, same value used for TIMIT
                self.B.data += lr * self.w_out.grad.detach() # match w_out scale
                self.B.data -= c_decay * self.B.data # L2 decay

        optimizer.step()

        return loss.item()