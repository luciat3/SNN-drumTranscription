
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# 1) Custom Gradient Spike Function for ALIF
# --------------------------------------------------------------------------

class SpikeFunction(torch.autograd.Function):
    """
    Custom spike function with:
    - Forward: Heaviside step function (spike generation) 
        as of eqs. (7) -> LIF and (9) -> ALIF
    - Backward: surrogate gradient (pseudo-derivative)

    This reproduces the TensorFlow @tf.custom_gradient SpikeFunction behavior 
    in PyTorch.
    """
    @staticmethod
    def forward(ctx, v_scaled, dampening_factor):
        ctx.save_for_backward(v_scaled, dampening_factor)

        z = (v_scaled > 0.0).to(v_scaled.dtype)  # Heaviside step function
        return z

    @staticmethod
    def backward(ctx, grad_output):

        # dampening_factor = gamma in the equation, fixed to 0.3
        v_scaled, dampening_factor = ctx.saved_tensors

        # surrogate gradient: max(0, 1 - abs(v_scaled)) * gamma
        surrogate_grad = torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)
        surrogate_grad = surrogate_grad * dampening_factor

        grad_v_scaled = grad_output * surrogate_grad

        grad_dampening_factor = torch.zeros_like(dampening_factor)

        return grad_v_scaled, grad_dampening_factor
    
def spike_function(v_scaled, dampening_factor):
    return SpikeFunction.apply(v_scaled, dampening_factor)

# --------------------------------------------------------------------------
# 2) ALIF Neuron Model
# --------------------------------------------------------------------------

class ALIFStep(nn.Module):
    """
    Hidden state per neuron j defined as:
    - v_j(t): membrane potential
    - a_j(t): adaptive threshold component
    - r_j(t): refractory counter

    Observable state:
    - z_j(t): spike output

    This class reproduces CustomALIF from the original TensorFlow code.
    """

    def __init__(
            self,
            n_in, # number of input neurons
            n_rec, # number of recurrent neurons
            tau_m=20.0, # membrane time constant (ms)
            tau_a=200.0, # adaptation time constant (ms)
            thr=0.60, # base threshold, paper defines 1.6
            beta=0.184, # adaptation strength
            dt=1.0, # time step (ms)
            dampening_factor=0.3, # surrogate gradient dampening factor (gamma)
            n_refractory=2, # refractory period in time steps
            rec=True, # whether to include recurrent connections
        ):
        """
        Corresponds to the initialization of CustomALIF in the original
        TensorFlow code, but adapted to PyTorch.
        """

        super().__init__()

        self.n_in = n_in
        self.n_rec = n_rec
        self.rec = rec
        self.dt = dt
        self.n_refractory = n_refractory

        # Decay factors as stated in the paper for eqs. (6) and (10)
        # alpha = exp(-dt / tau_m) for membrane potential decay
        self.alpha = torch.exp(torch.tensor(-dt / tau_m, dtype=torch.float32))
        # rho = exp(-dt / tau_a) for adaptation decay
        self.rho = torch.exp(torch.tensor(-dt / tau_a, dtype=torch.float32))

        # Base threshold and adaptation strength
        self.thr = nn.Parameter(torch.full((n_rec,), float(thr)), requires_grad=False)

        # In order to have both LIF and ALIF, we can set beta to zero for LIF behavior, 
        # and to 0.184 for ALIF as in the paper
        if isinstance(beta, (float, int)):
            beta_tensor = torch.full((n_rec,), float(beta), dtype=torch.float32)
        else:
            beta_tensor = torch.as_tensor(beta, dtype=torch.float32)
            if beta_tensor.numel() != n_rec:
                raise ValueError(
                    f"beta must be a scalar or have length n_rec={n_rec}, "
                    f"but got shape {tuple(beta_tensor.shape)}"
                )

        self.beta = nn.Parameter(beta_tensor, requires_grad=False)

        # Dampening factor for surrogate gradient
        self.dampening_factor = nn.Parameter(torch.tensor(float(dampening_factor), dtype=torch.float32), requires_grad=False)

        # Input weights [n_in, n_rec]
        self.w_in = nn.Parameter(torch.randn(n_in, n_rec) / (n_in ** 0.5))

        # Recurrent weights [n_rec, n_rec], only if rec=True
        if rec:
            w_rec = torch.randn(n_rec, n_rec) / (n_rec ** 0.5)

            # Mask to prevent self-connections (diagonal should be zero)
            mask = torch.eye(n_rec, dtype=torch.bool)
            self.register_buffer('rec_mask', mask)

            # Remove self-connections by zeroing the diagonal
            w_rec.fill_diagonal_(0.0)
            self.w_rec = nn.Parameter(w_rec)
        else:
            self.w_rec = None

    # --------------------------------------------------------------------------
    # 3) Initial state
    # --------------------------------------------------------------------------
    def zero_state(self, batch_size, device=None):
        """
        Returns initial state:
        - v = 0 (membrane potential)
        - a = 0 (adaptation variable)
        - z = 0 (spike output)
        - r = 0 (refractory counter)
        """
        if device is None:
            device = self.w_in.device

        v = torch.zeros(batch_size, self.n_rec, device=device)
        a = torch.zeros(batch_size, self.n_rec, device=device)
        z = torch.zeros(batch_size, self.n_rec, device=device)
        r = torch.zeros(batch_size, self.n_rec, device=device)
        return {"v": v, "a": a, "z": z, "r": r}

    # --------------------------------------------------------------------------
    # 4) Spike computation
    # --------------------------------------------------------------------------
    def compute_z(self, v, a):
        """
        Following paper equation (8), this function calculates the threshold 
        potential:
            A_j(t) = thr + beta * a_j(t)
        and then computes the scaled membrane potential:
            v_scaled = (v_j(t) - A_j(t)) / A_j(t)
        Finally, it applies the custom spike function to get the binary spike 
        output z_j(t), calculating eq (9):
            z_j(t) = H(v_j(t) - A_j(t)) 
        """
        adaptive_thr = self.thr.unsqueeze(0) + self.beta.unsqueeze(0) * a  
        v_scaled = (v - adaptive_thr) / self.thr.unsqueeze(0)
        z = spike_function(v_scaled, self.dampening_factor)

        return z, adaptive_thr, v_scaled
    
    # --------------------------------------------------------------------------
    # 5) One recurrent step
    # --------------------------------------------------------------------------
    def forward(self, x_t, state):
        """
        Performs one time step update.
        Inputs:
        - x_t: input at time t, shape (batch_size, n_in)
        - state: dict with:
            - v: membrane potential at time t, shape (batch_size, n_rec)
            - a: adaptation variable at time t, shape (batch_size, n_rec)
            - z: previous spike output, shape (batch_size, n_rec)
            - r: refractory counter, shape (batch_size, n_rec)
        Returns the updated dictionary

        This function corresponds to the call method of CustomALIF in the 
        original TensorFlow code, but adapted to PyTorch.
        """
        v = state["v"]
        a = state["a"]
        z = state["z"]
        r = state["r"]

        # 1) Compute spikes for current state
        old_z, adaptive_thr_t, v_scaled_t = self.compute_z(v, a)
        # 2) Adaptation update, eq. (10)
        new_a = self.rho.to(a.device) * a + old_z
        # 3) Input current, corresponds to the input sum in eq. (6)
        i_in = x_t @ self.w_in
        # 4) Recurrent current, corresponds to the recurrent sum in eq. (6)
        if self.rec:
            w_rec = self.w_rec.masked_fill(self.rec_mask, 0.0)  # Ensure no self-connections

            i_rec = z @ w_rec
            i_t = i_in + i_rec
        else:
            i_t = i_in
        # 5) Reset current after spike
        I_reset = z * self.thr.unsqueeze(0) * self.dt
        # 6) Membrane potential update, eq. (6)
        new_v = self.alpha.to(v.device) * v + i_t - I_reset
        # 7) Refractory period
        is_refractory = r > 0.1

        candidate_new_z, adaptive_thr_next, v_scaled_next = self.compute_z(new_v, new_a)

        # During refractory period, force new_z to be zero
        new_z = torch.where(is_refractory, torch.zeros_like(candidate_new_z), candidate_new_z)

        # 8) Update refractory counter
        new_r = torch.clamp(r + self.n_refractory * new_z - 1.0, min=0.0, max=float(self.n_refractory))

        new_state = {
            "v": new_v,
            "a": new_a,
            "z": new_z,
            "r": new_r
        }

        aux = {
            "old_z": old_z,
            "adaptive_thr_t": adaptive_thr_t,
            "v_scaled_t": v_scaled_t,
            "i_in": i_in,
            "i_t": i_t,
            "I_reset": I_reset,
            "candidate_new_z": candidate_new_z,
            "adaptive_thr_next": adaptive_thr_next,
            "v_scaled_next": v_scaled_next,
            "is_refractory": is_refractory
        }

        return new_state, aux