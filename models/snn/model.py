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

    # ------------------------------------------------------------------
    # Plain inference forward (used only at eval time)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x_seq):
        """x_seq: [T, B, n_in] -> z_seq: [T, B, n_rec]"""
        T, B, _ = x_seq.shape
        device = x_seq.device
        state = self.alif.zero_state(B, device=device)
        all_z = []
        for t in range(T):
            state, _ = self.alif(x_seq[t], state)
            all_z.append(state["z"])
        return torch.stack(all_z)

    @torch.no_grad()
    def readout_from_spikes(self, z_seq):
        T, B, _ = z_seq.shape
        device = z_seq.device
        kappa = self.kappa.to(device)
        y_prev = torch.zeros(B, self.n_out, device=device)
        all_y = []
        for t in range(T):
            y = kappa * y_prev + z_seq[t] @ self.w_out + self.b_out
            y_prev = y
            all_y.append(y)
        return torch.stack(all_y)


    def eprop_update(self, x_seq, y_out, z_rec, targets, weighted_mask, optimizer, repeat_steps=5, reg=50, f_target_hz=10.0, dt=1e-3):
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
        T_frame, B, _ = y_out.shape
        T_ms = x_seq.shape[0]

        device = y_out.device

        # π^t_k — softmax over readout, as used in eq. (29), is the probability for class k
        pi = torch.softmax(y_out, dim=-1)

        # π*^t_k — target one-hot encoding
        pi_star = torch.zeros_like(pi)
        pi_star.scatter_(-1, targets.unsqueeze(-1), 1.0)

        # Compute learning signal L^t_j for each recurrent neuron j at each time step t (eq. (4))
        error = pi - pi_star  # [T, B, n_out]
        error = error * weighted_mask.unsqueeze(-1)
        L_frame = (error @ self.B.T).detach()  # [B] correctly detach broadcast path
        L_ms = L_frame.repeat_interleave(repeat_steps, dim=0) / repeat_steps

        # Compute elegibility traces online
        state = self.alif.zero_state(B, device=device)
        z_bar = torch.zeros(B, self.n_rec, device=device)
        eps_a = torch.zeros(B, self.n_rec, self.n_rec, device=device)
        e_bar = torch.zeros(B, self.n_rec, self.n_rec, device=device)

        grad_w_rec = torch.zeros_like(self.alif.w_rec) if self.alif.w_rec is not None else None
        
        # Average firing rate
        f_av = z_rec.mean(dim=(0, 1)) / dt  # [n_rec]

        f_target = torch.full_like(f_av, f_target_hz)
        
        # local regularization learning signal
        L_reg = reg * (f_target - f_av) / T_ms  # [n_rec]

        with torch.no_grad():
            for t in range(T_ms):
                # pseudo-derivative from current state
                psi, _, _ = compute_psi(state["v"], state["a"], self.alif.thr, self.alif.beta, self.alif.dampening_factor)
                # recurrent elegibility traces
                z_bar, eps_a, e_trace = update_alif_traces_recurrent(
                    z_bar=z_bar, eps_a=eps_a, z_pre=state["z"], psi=psi, alpha=self.alif.alpha, rho=self.alif.rho, beta=self.alif.beta
                )
                # filtered recurrent trace
                e_bar = self.kappa.to(device) * e_bar + e_trace

                # acumulate recurrent gradient online
                L_t_total = L_ms[t] + L_reg.unsqueeze(0)  # [B, n_rec]
                grad_w_rec = grad_w_rec - (L_t_total.unsqueeze(-1) * e_bar).sum(dim=0)  # [n_rec, n_rec]

                state, _ = self.alif(x_seq[t], state)

        loss_per_frame = F.cross_entropy(
                y_out.reshape(T_frame * B, -1),
                targets.reshape(T_frame * B),
                reduction="none"
            ).reshape(T_frame, B)

        loss_pred = (loss_per_frame * weighted_mask).sum(dim=0).mean()
        

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

    # ------------------------------------------------------------------
    # Window-level adaptive e-prop update
    # ------------------------------------------------------------------
    def eprop_update_window(
        self,
        x_seq,                 # [T, B, n_in]  (already repeated if repeat_steps>1)
        targets,               # [B, n_out] multi-hot
        optimizer,
        pos_weight=None,       # [n_out]
        tol_steps=3,           # supervised radius around center, in FRAMES (before repeat)
        repeat_steps=1,        # number of SNN steps per input frame
        reg_rate=50.0,         # firing-rate regularisation coefficient
        reg_voltage=1e-4,      # voltage regularisation coefficient
        f_target_hz=10.0,
        dt_seconds=1e-3,       # physical duration of ONE SNN step (= paper's dt = 1 ms)
        homeo_lr=0.0,          # optional homeostatic threshold adjustment
    ):
        """
        Performs a single adaptive e-prop update on a batch of windows.

        Gradient accumulation is done ONLINE during the temporal loop, so
        we never materialize [T, B, N, N] eligibility-trace tensors. This
        only works because the supervised delta is a per-sample constant
        over the supervised window: Δ_t = Δ_center / n_sup for t in [lo, hi],
        and Δ_t = 0 elsewhere. Under this structure:

            grad_rec = L_const[b,j] * sum_{t in [lo,hi]} e_bar_rec[t, b, j, i]
                     + L_reg[j]    * sum_{t in [0, T]}   e_bar_rec[t, b, j, i]
                       (analogously for w_in)

        so we only need two running tensors of shape [B, N, N] (and similar
        for w_in) instead of a full [T, B, N, N] history.
        """
        device = x_seq.device
        T, B, n_in = x_seq.shape
        C = self.n_out
        N = self.n_rec
        dtype = x_seq.dtype

        if pos_weight is None:
            pos_weight = torch.ones(C, device=device, dtype=dtype)
        else:
            pos_weight = pos_weight.to(device=device, dtype=dtype)

        alpha = self.alif.alpha.to(device)
        rho = self.alif.rho.to(device)
        beta = self.alif.beta.to(device)
        kappa = self.kappa.to(device)
        thr = self.alif.thr.to(device)

        # -------------------------------------------------------------
        # 1) Supervised window in SNN-step units
        # -------------------------------------------------------------
        # tol_steps is specified in MFCC frames; convert to SNN steps
        tol_ms = tol_steps * repeat_steps
        center = T // 2
        lo = max(0, center - tol_ms)
        hi = min(T, center + tol_ms + 1)
        n_sup = max(1, hi - lo)

        # -------------------------------------------------------------
        # 2) Init state + eligibility accumulators
        # -------------------------------------------------------------
        state = self.alif.zero_state(B, device=device)
        y_prev = torch.zeros(B, C, device=device, dtype=dtype)

        # recurrent eligibility state
        z_bar_rec = torch.zeros(B, N, device=device, dtype=dtype)
        eps_a_rec = torch.zeros(B, N, N, device=device, dtype=dtype)
        e_bar_rec = torch.zeros(B, N, N, device=device, dtype=dtype)

        # input eligibility state
        x_bar_in = torch.zeros(B, n_in, device=device, dtype=dtype)
        eps_a_in = torch.zeros(B, N, n_in, device=device, dtype=dtype)
        e_bar_in = torch.zeros(B, N, n_in, device=device, dtype=dtype)

        # readout eligibility (filtered spikes)
        z_filt_out = torch.zeros(B, N, device=device, dtype=dtype)

        # running sums of filtered eligibility traces
        sum_e_rec_window = torch.zeros(B, N, N, device=device, dtype=dtype)
        sum_e_rec_all = torch.zeros(B, N, N, device=device, dtype=dtype)
        sum_e_in_window = torch.zeros(B, N, n_in, device=device, dtype=dtype)
        sum_e_in_all = torch.zeros(B, N, n_in, device=device, dtype=dtype)

        # voltage-reg accumulators (see Supp Note 2)
        grad_w_in_vreg = torch.zeros(N, n_in, device=device, dtype=dtype)
        grad_w_rec_vreg = (
            torch.zeros_like(self.alif.w_rec) if self.alif.w_rec is not None else None
        )

        # running spike count and outputs for readout loss
        spike_count = torch.zeros(N, device=device, dtype=dtype)
        y_center_sum = torch.zeros(B, C, device=device, dtype=dtype)
        sum_z_filt_window = torch.zeros(B, N, device=device, dtype=dtype)

        # -------------------------------------------------------------
        # 3) Single temporal pass
        # -------------------------------------------------------------
        with torch.no_grad():
            for t in range(T):
                v = state["v"]
                a = state["a"]
                z_prev = state["z"]

                # pseudo-derivative at current state
                psi, _, _ = compute_psi(v, a, thr, beta, self.alif.dampening_factor)

                # --- recurrent eligibility (eqs. 22, 24, 25) ---
                z_bar_rec, eps_a_rec, e_rec = update_alif_traces_recurrent(
                    z_bar=z_bar_rec,
                    eps_a=eps_a_rec,
                    z_pre=z_prev,
                    psi=psi,
                    alpha=alpha,
                    rho=rho,
                    beta=beta,
                )
                e_bar_rec = kappa * e_bar_rec + e_rec

                # --- input eligibility ---
                x_bar_old = x_bar_in
                x_bar_in = alpha * x_bar_in + x_seq[t]
                psi_exp = psi.unsqueeze(-1)                    # [B, N, 1]
                beta_exp = beta.unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
                eps_a_in = (
                    psi_exp * x_bar_old.unsqueeze(1)
                    + (rho - beta_exp * psi_exp) * eps_a_in
                )
                e_in = psi_exp * (x_bar_in.unsqueeze(1) - beta_exp * eps_a_in)
                e_bar_in = kappa * e_bar_in + e_in

                # --- step the recurrent dynamics ---
                new_state, _ = self.alif(x_seq[t], state)
                z_t = new_state["z"]

                # --- readout ---
                y_t = kappa * y_prev + z_t @ self.w_out + self.b_out
                y_prev = y_t

                z_filt_out = kappa * z_filt_out + z_t

                # --- voltage regularization (Supp Note 2) ---
                # Penalize |v| > thr with a direct local gradient on e_v & e_a.
                # dE_V / dv = 2 * ([v - thr]_+  -  [-v - thr]_+)
                if reg_voltage > 0.0:
                    v_now = new_state["v"]
                    over = (v_now - thr.unsqueeze(0)).clamp(min=0.0)
                    under = (-v_now - thr.unsqueeze(0)).clamp(min=0.0)
                    dv = (over - under)                        # [B, N]
                    # eligibility components: eps_v = x_bar  (for w_in) and
                    # eps_v = z_bar (for w_rec). We use the current eligibility
                    # vector for the adaptation var too, with the minus sign
                    # from Supp eq. (11).
                    # grad_w[post, pre] += dv[b, post] * (eps_v - eps_a)[b, post, pre]
                    vreg_factor = reg_voltage * dv.unsqueeze(-1)  # [B, N, 1]
                    eps_v_rec = z_bar_rec.unsqueeze(1).expand(B, N, N)
                    grad_w_rec_vreg_step = (
                        vreg_factor * (eps_v_rec - eps_a_rec)
                    ).sum(dim=0)
                    eps_v_in = x_bar_in.unsqueeze(1).expand(B, N, n_in)
                    grad_w_in_vreg_step = (
                        vreg_factor * (eps_v_in - eps_a_in)
                    ).sum(dim=0)
                    if grad_w_rec_vreg is not None:
                        grad_w_rec_vreg = grad_w_rec_vreg + grad_w_rec_vreg_step
                    grad_w_in_vreg = grad_w_in_vreg + grad_w_in_vreg_step

                # --- accumulate running sums ---
                sum_e_rec_all = sum_e_rec_all + e_bar_rec
                sum_e_in_all = sum_e_in_all + e_bar_in
                spike_count = spike_count + z_t.sum(dim=0)

                if lo <= t < hi:
                    y_center_sum = y_center_sum + y_t
                    sum_z_filt_window = sum_z_filt_window + z_filt_out
                    sum_e_rec_window = sum_e_rec_window + e_bar_rec
                    sum_e_in_window = sum_e_in_window + e_bar_in

                state = new_state

        # -------------------------------------------------------------
        # 4) Window-level logits, BCE loss, and per-sample delta
        # -------------------------------------------------------------
        # FIX: detach logits_window to prevent accidental autograd graph
        # construction outside no_grad. y_center_sum was built inside
        # torch.no_grad() so it has no grad_fn, but the explicit .detach()
        # guarantees this even if that assumption ever changes.
        logits_window = (y_center_sum / n_sup).detach()         # [B, C]

        loss_pred = F.binary_cross_entropy_with_logits(
            logits_window,
            targets,
            pos_weight=pos_weight,
            reduction="mean",
        )

        probs_window = torch.sigmoid(logits_window)

        # analytical d/d logit of BCEWithLogits(pos_weight):
        #   d/dx = sigmoid(x) * [1 + (w-1)*t] - w*t
        class_scale = 1.0 + (pos_weight.unsqueeze(0) - 1.0) * targets   # [B, C]
        delta_window = (
            probs_window * class_scale - targets * pos_weight.unsqueeze(0)
        )
        # normalize like reduction='mean': /B/C
        delta_window = delta_window / (B * C)

        # per-step delta: constant over [lo, hi]
        delta_center = delta_window / n_sup                      # [B, C]

        # L_const[b, j] = sum_k B[j,k] * delta_center[b,k]
        # B is requires_grad=False; .detach() is defensive but correct.
        L_const = delta_center @ self.B.detach().T               # [B, N]

        # -------------------------------------------------------------
        # 5) Firing-rate regularization (Supp Note 2)
        # -------------------------------------------------------------
        # Average spike probability per SNN step, not Hz.
        # This is in [0, 1], so the regularizer has a sane numerical scale.
        f_av_step = (spike_count / (B * T)).detach()          # [N], spikes / step
        f_target_step = torch.full_like(
            f_av_step,
            f_target_hz * dt_seconds                         # e.g. 15 Hz * 0.001 = 0.015
        )

        loss_reg = 0.5 * reg_rate * ((f_av_step - f_target_step) ** 2).mean()

        # Local e-prop regularization signal.
        # Keep this in step units too.
        L_reg = reg_rate * (f_av_step - f_target_step) / max(T, 1)

        # For logging only.
        f_av_hz = f_av_step / dt_seconds
        # -------------------------------------------------------------
        # 6) Gradient assembly
        # -------------------------------------------------------------
        # Recurrent gradient. sum_e_rec_* has shape [B, N_post, N_pre].
        # We need dE/dw_rec[i, j] where w_rec is [N_pre, N_post].
        # -> compute [N_post, N_pre] and transpose at the end.
        grad_w_rec_post_pre = (
            L_const.unsqueeze(-1) * sum_e_rec_window
        ).sum(dim=0) + L_reg.view(N, 1) * sum_e_rec_all.sum(dim=0)
        if grad_w_rec_vreg is not None:
            grad_w_rec_post_pre = grad_w_rec_post_pre + grad_w_rec_vreg
        grad_w_rec = grad_w_rec_post_pre.transpose(0, 1).contiguous()    # [pre, post]

        # Input gradient. sum_e_in_* has shape [B, N_post, n_in_pre].
        # w_in is [n_in, N] -> transpose to [n_in, N] at the end.
        grad_w_in_post_pre = (
            L_const.unsqueeze(-1) * sum_e_in_window
        ).sum(dim=0) + L_reg.view(N, 1) * sum_e_in_all.sum(dim=0)
        grad_w_in_post_pre = grad_w_in_post_pre + grad_w_in_vreg
        grad_w_in = grad_w_in_post_pre.transpose(0, 1).contiguous()

        # Readout gradients.
        # dE/dw_out[j, k] = sum_{t in [lo,hi]} delta_center[b,k] * z_filt[t, b, j]
        # = delta_center[b, k] * sum_z_filt_window[b, j]
        grad_w_out = sum_z_filt_window.transpose(0, 1) @ delta_center    # [N, C]
        grad_b_out = delta_window.sum(dim=0)                             # [C]

        # -------------------------------------------------------------
        # 7) Assign grads and step
        # -------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        self.w_out.grad = grad_w_out
        self.b_out.grad = grad_b_out
        self.alif.w_in.grad = grad_w_in
        if self.alif.w_rec is not None:
            self.alif.w_rec.grad = grad_w_rec

        # Adaptive e-prop: mirror W_out update INTO B.grad so Adam applies
        # an identical update to both (Supp Note 2: "identical weight update").
        self.B.grad = grad_w_out.detach().clone()

        # FIX: clip_grad_norm_(self.parameters()) silently skips self.B because
        # B has requires_grad=False. We must clip B explicitly to prevent large
        # broadcast-weight updates from destabilising adaptive e-prop.
        params_to_clip = list(self.parameters()) + [self.B]
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

        optimizer.step()

        # Optional homeostatic threshold drift (kept tiny by default)
        if homeo_lr > 0.0:
            with torch.no_grad():
                excess = f_av_hz - f_target_hz
                self.alif.thr.data += homeo_lr * excess
                self.alif.thr.data.clamp_(min=0.3, max=2.5)

        loss = loss_pred + loss_reg
        return {
            "loss": float(loss.item()),
            "loss_pred": float(loss_pred.item()),
            "loss_reg": float(loss_reg.item()),
            "logits_window": logits_window.detach(),
            "f_av_hz": f_av_hz.detach(),
            "spike_rate_hz": float(f_av_hz.mean()),
            "spike_rate_max_hz": float(f_av_hz.max()),
        }
