"""
Through the TensorFlow file "alif_eligibility_propagation.py", the pseudo-derivate,
eligibility traces, ALIF hidden state and the local Jacobian are implicitly calculated.
This code will explicitly do that. It also includes the Learning signal computation.
"""
import torch

# ---------------------------------------------------------------------------------
# 1) Pseudo-derivative 
# ---------------------------------------------------------------------------------
def compute_psi(v, a, thr, beta, dampening_factor):
    """
    Similar to the backward pass from the costume gradient
    """
    # Eq. (8): Adaptive thresholds
    adaptive_thr = thr.unsqueeze(0) + beta.unsqueeze(0) * a

    # Pseudo-derivate equation
    v_scaled = (v - adaptive_thr) / thr.unsqueeze(0)
    pseudo = torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)
    psi = (dampening_factor * pseudo) / thr.unsqueeze(0)

    return psi, v_scaled, adaptive_thr

# ---------------------------------------------------------------------------------
# 2) Hidden-state Jacobian
# ---------------------------------------------------------------------------------
def build_alif_jacobian(psi, alpha, rho, beta):
    """
    The elegibility trace update requires the Jacobian of the hidden state,
    which is ∂h_j^(t+1)/∂h_j^t​​. It is a 2x2 matrix, since each neuron has 
    two hidden state variables: v and a. 
    """
    B, N = psi.shape
    device = psi.device
    dtype = psi.dtype

    J = torch.zeros(B, N, 2, 2, device=device, dtype=dtype)

    # ∂v_j^(t+1)/∂v_j^t​​ = α 
    J[..., 0, 0] = alpha

    # ∂v_j^(t+1)/∂a_j^t​​ = 0
    J[..., 0, 1] = 0.0

    # ∂a_j^(t+1)/∂v_j^t​​ = ψ_j^t
    J[..., 1, 0] = psi

    # ∂a_j^(t+1)/∂a_j^t​​ = ρ - β * ψ_j^t
    J[..., 1, 1] = rho - beta.unsqueeze(0) * psi

    return J

# ---------------------------------------------------------------------------------
# 3) Elegibility Trace and vector update
# ---------------------------------------------------------------------------------
def update_alif_traces_recurrent(
        z_bar, # filtered pre-synaptic activity
        eps_a, # elegibility trace for the adaptation variable
        z_pre, # pre-synaptic spikes at time t
        psi,  # pseudo-derivative at time t
        alpha, # voltage decay
        rho,   # adaptation decay
        beta   
    ):

    # Low pass filter F_alpha eq. (12)
    new_z_bar = alpha * z_bar + z_pre

    # Expand for broadcasting
    psi_exp = psi.unsqueeze(-1)  # [B, N_post, 1]
    beta_exp = beta.unsqueeze(0).unsqueeze(-1)  # [1, N_post,1]

    # Update elegibility vector eq. (24)
    new_eps_a = psi_exp * z_bar.unsqueeze(1) + (rho - beta_exp * psi_exp) * eps_a

    # Elegibility trace eq. (25)
    e_trace = psi_exp * (new_z_bar.unsqueeze(1) - beta_exp * eps_a)

    return new_z_bar, new_eps_a, e_trace

# ---------------------------------------------------------------------------------
# 4) Leaky output neuron
# ---------------------------------------------------------------------------------
def exp_convolve(x, decay):
    """
    Leaky output neurons are defined in the original paper in eq. (11).
    This function performs the exponential convolution of the input x with the 
    given decay factor.
    The low pass filter F_alpha is defined as:
    F_alpha(x^(t)) = α * F_alpha(x^(t-1)) + x^(t) -> eq. (12)
    here, alpha is the decay factor.
    """
    B, T, N = x.shape # Batch size, Time steps, Number of neurons
    y = torch.zeros_like(x)
    prev = torch.zeros(B, N, device=x.device, dtype=x.dtype)

    for t in range(T):
        prev = decay * prev + x[:, t]
        y[:, t] = prev

    return y
