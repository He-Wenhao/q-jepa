"""
Model: f(γ_t, H_vec) → γ_{t+1}

No encoder, no latent space. Direct γ-space dynamics.
Residual architecture: output γ_t + Δ (predict the change).
H_vec is normalized by running statistics stored at training time.
"""
import torch
import torch.nn as nn
import numpy as np


class DeltaPredictor(nn.Module):
    """
    Predicts γ_{t+1} = γ_t + f(γ_t, H_vec).
    The residual structure biases toward small changes (good inductive bias
    for small δτ imaginary-time steps and for fine-tuning from γ_0 → γ_GS).
    """
    def __init__(self, rdm_dim: int, h_dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.rdm_dim = rdm_dim
        in_dim  = rdm_dim ** 2 + h_dim
        out_dim = rdm_dim ** 2

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        # H normalization stats (set after training data is known)
        self.register_buffer("h_mean", torch.zeros(h_dim))
        self.register_buffer("h_std",  torch.ones(h_dim))

    def set_h_stats(self, h_mean: np.ndarray, h_std: np.ndarray):
        self.h_mean.copy_(torch.tensor(h_mean, dtype=torch.float32))
        self.h_std.copy_(torch.tensor(np.maximum(h_std, 1e-6), dtype=torch.float32))

    def forward(self, gamma: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        gamma: (B, rdm_dim, rdm_dim)
        h:     (B, h_dim)
        returns γ_{t+1}: (B, rdm_dim, rdm_dim)
        """
        B = gamma.shape[0]
        g_flat = gamma.reshape(B, -1)
        h_norm = (h - self.h_mean) / self.h_std
        x      = torch.cat([g_flat, h_norm], dim=-1)
        delta  = self.net(x).reshape(B, self.rdm_dim, self.rdm_dim)
        out    = gamma + delta
        # Symmetrize: physical γ is Hermitian (real case: symmetric)
        return (out + out.transpose(-2, -1)) * 0.5
