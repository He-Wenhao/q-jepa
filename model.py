"""Q-JEPA model: Encoder + Predictor for 1-RDM imaginary-time prediction."""
import torch
import torch.nn as nn
import copy


class RDMEncoder(nn.Module):
    """Encodes a flattened 1-RDM + U/t scalar into a latent vector."""
    def __init__(self, rdm_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = rdm_dim * rdm_dim + 1  # flattened RDM + U/t
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, gamma: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        gamma: (B, rdm_dim, rdm_dim) or (B, rdm_dim*rdm_dim)
        U:     (B,) scalar U/t value
        """
        B = gamma.shape[0]
        x = gamma.reshape(B, -1)
        u = U.reshape(B, 1)
        x = torch.cat([x, u], dim=-1)
        return self.net(x)


class Predictor(nn.Module):
    """Predicts next latent z_{t+1} from z_t."""
    def __init__(self, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Denoiser(nn.Module):
    """One-shot denoiser: maps z_t (any trajectory step) → z_GS estimate."""
    def __init__(self, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class QJEPA(nn.Module):
    """
    Encoder + Predictor (next-step) + Denoiser (one-shot → z_GS).

    Training losses:
      L_JEPA:   predictor(z_t) ≈ z_{t+1}  (dynamics)
      L_denoise: denoiser(z_t) ≈ z_GS     (ground-state finding)
    """
    def __init__(self, rdm_dim: int, latent_dim: int = 64, hidden: int = 256,
                 ema_decay: float = 0.996):
        super().__init__()
        self.encoder   = RDMEncoder(rdm_dim, latent_dim, hidden)
        self.predictor = Predictor(latent_dim, hidden)
        self.denoiser  = Denoiser(latent_dim, hidden)

        # Target encoder: EMA copy
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for p_online, p_target in zip(self.encoder.parameters(),
                                       self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_decay).add_(p_online.data, alpha=1 - self.ema_decay)

    def forward(self, gamma_t, gamma_tp, gamma_gs, U):
        """Returns (loss_jepa, loss_denoise)."""
        z_t  = self.encoder(gamma_t,  U)
        z_tp = self.encoder(gamma_tp, U)

        # JEPA: predict next step
        z_pred_next = self.predictor(z_t)
        with torch.no_grad():
            z_tp_target = self.target_encoder(gamma_tp, U).detach()
        loss_jepa = nn.functional.mse_loss(z_pred_next, z_tp_target)

        # Denoiser: one-shot map to z_GS
        z_pred_gs = self.denoiser(z_t)
        with torch.no_grad():
            z_gs_target = self.target_encoder(gamma_gs, U).detach()
        loss_denoise = nn.functional.mse_loss(z_pred_gs, z_gs_target)

        return loss_jepa, loss_denoise

    def encode(self, gamma, U):
        with torch.no_grad():
            return self.encoder(gamma, U)

    def denoise_to_gs(self, gamma0, U):
        """One-shot: encode gamma0, apply denoiser → z_GS estimate."""
        with torch.no_grad():
            z0 = self.encoder(gamma0, U)
            return self.denoiser(z0)


class EnergyHead(nn.Module):
    """Lightweight MLP to predict E0 from latent z*."""
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class RDMDecoder(nn.Module):
    """Decode latent z* back to ground-state 1-RDM."""
    def __init__(self, latent_dim: int, rdm_dim: int, hidden: int = 256):
        super().__init__()
        out_dim = rdm_dim * rdm_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.rdm_dim = rdm_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.reshape(-1, self.rdm_dim, self.rdm_dim)
