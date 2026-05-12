"""
Q-JEPA model — corrected design:

  Encoder:   γ → z          (universal state compression, NO Hamiltonian params)
  Predictor: (z, H) → z'    (H-conditioned evolution, analogous to KS iteration)
  Denoiser:  (z, H) → z*    (H-conditioned one-shot ground-state finder)
  Head:      z* → property  (universal functional, NO Hamiltonian params)

Analogy to DFT:
  Encoder   ↔  density functional ρ[ψ]       (state → descriptor)
  Predictor ↔  KS iteration                   (descriptor update, needs v_ext)
  Head      ↔  E[ρ] universal functional      (descriptor → energy, no v_ext)
"""
import torch
import torch.nn as nn
import copy


class RDMEncoder(nn.Module):
    """γ → z  (universal, no Hamiltonian parameters)."""
    def __init__(self, rdm_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = rdm_dim * rdm_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        """gamma: (B, rdm_dim, rdm_dim) or (B, rdm_dim²)"""
        return self.net(gamma.reshape(gamma.shape[0], -1))


class Predictor(nn.Module):
    """(z, H_params) → z'  (one imaginary-time step, H-conditioned)."""
    def __init__(self, latent_dim: int, h_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + h_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim), h: (B, h_dim)"""
        return self.net(torch.cat([z, h], dim=-1))


class Denoiser(nn.Module):
    """(z, H_params) → z*  (one-shot ground-state finder, H-conditioned)."""
    def __init__(self, latent_dim: int, h_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + h_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h], dim=-1))


class QJEPA(nn.Module):
    """
    Full Q-JEPA model with corrected information flow.

    Training losses:
      L_JEPA    : predictor(z_t, H) ≈ z_{t+1}   (dynamics learning)
      L_denoise : denoiser(z_t, H)  ≈ z_GS       (ground-state finding)

    EMA target encoder prevents representation collapse.
    """
    def __init__(self, rdm_dim: int, latent_dim: int = 64,
                 h_dim: int = 1, hidden: int = 256, ema_decay: float = 0.996):
        super().__init__()
        self.encoder   = RDMEncoder(rdm_dim, latent_dim, hidden)
        self.predictor = Predictor(latent_dim, h_dim, hidden)
        self.denoiser  = Denoiser(latent_dim, h_dim, hidden)

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for p_o, p_t in zip(self.encoder.parameters(),
                             self.target_encoder.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_o.data, alpha=1 - self.ema_decay)

    def forward(self, gamma_t, gamma_tp, gamma_gs, H):
        """
        gamma_t, gamma_tp, gamma_gs : (B, rdm_dim, rdm_dim)
        H : (B, h_dim)  Hamiltonian parameters (e.g. U/t)
        Returns: (loss_jepa, loss_denoise)
        """
        z_t = self.encoder(gamma_t)

        # JEPA: predict next latent step
        z_pred_next = self.predictor(z_t, H)
        with torch.no_grad():
            z_tp_target = self.target_encoder(gamma_tp).detach()
        loss_jepa = nn.functional.mse_loss(z_pred_next, z_tp_target)

        # Denoiser: one-shot map to ground-state latent
        z_pred_gs = self.denoiser(z_t, H)
        with torch.no_grad():
            z_gs_target = self.target_encoder(gamma_gs).detach()
        loss_denoise = nn.functional.mse_loss(z_pred_gs, z_gs_target)

        return loss_jepa, loss_denoise

    # ── Inference helpers ──────────────────────────────────────────────────
    def encode(self, gamma: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(gamma)

    def denoise_to_gs(self, gamma0: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Encode γ₀, apply denoiser → z* (ground-state latent estimate)."""
        with torch.no_grad():
            z0 = self.encoder(gamma0)
            return self.denoiser(z0, H)


class EnergyHead(nn.Module):
    """z* → E₀  (universal functional, no Hamiltonian parameters)."""
    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class DensityEncoder(nn.Module):
    """
    DFT-analog baseline: use only the diagonal of γ (density ρ) as the descriptor.
    ρ → z_density
    """
    def __init__(self, rdm_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rdm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        """Extract diagonal (density) and encode."""
        B = gamma.shape[0]
        rho = gamma.reshape(B, gamma.shape[-1], gamma.shape[-1]).diagonal(dim1=-2, dim2=-1)
        return self.net(rho)


class RDMDecoder(nn.Module):
    """z* → γ_GS  (reconstruction for verification)."""
    def __init__(self, latent_dim: int, rdm_dim: int, hidden: int = 256):
        super().__init__()
        self.rdm_dim = rdm_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rdm_dim * rdm_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(-1, self.rdm_dim, self.rdm_dim)
