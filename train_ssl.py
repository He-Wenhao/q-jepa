"""
Q-JEPA self-supervised pretraining.

Information flow (corrected):
  Encoder:   γ → z          (universal, no H)
  Predictor: (z, H) → z'    (H-conditioned dynamics)
  Denoiser:  (z, H) → z*    (H-conditioned ground-state finder)
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import QJEPA

LATENT_DIM = 64
H_DIM      = 1        # Hamiltonian parameter dimension (just U/t here)
HIDDEN     = 256
EMA_DECAY  = 0.996
LR         = 3e-4
BATCH_SIZE = 256
EPOCHS     = 300
LAMBDA_DN  = 2.0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "checkpoints/jepa_pretrained.pt"
DATA_PATH  = "data/hubbard_ssl.npz"

os.makedirs("checkpoints", exist_ok=True)

print("Loading SSL data...")
d        = np.load(DATA_PATH)
gamma_t  = torch.tensor(d["gamma_t"],  dtype=torch.float32)
gamma_tp = torch.tensor(d["gamma_tp"], dtype=torch.float32)
gamma_gs = torch.tensor(d["gamma_gs"], dtype=torch.float32)
U_vals   = torch.tensor(d["U"],        dtype=torch.float32).unsqueeze(-1)  # (N,1)

rdm_dim = gamma_t.shape[-1]
print(f"  N={len(gamma_t)}, rdm_dim={rdm_dim}, h_dim={H_DIM}, device={DEVICE}")

dataset = TensorDataset(gamma_t, gamma_tp, gamma_gs, U_vals)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True)

model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM,
              hidden=HIDDEN, ema_decay=EMA_DECAY).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(model.encoder.parameters()) +
    list(model.predictor.parameters()) +
    list(model.denoiser.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_j = tot_d = n_total = 0
    for gt, gtp, gGs, H in loader:
        gt, gtp, gGs, H = gt.to(DEVICE), gtp.to(DEVICE), gGs.to(DEVICE), H.to(DEVICE)
        optimizer.zero_grad()
        loss_j, loss_d = model(gt, gtp, gGs, H)
        loss = loss_j + LAMBDA_DN * loss_d
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target()
        B = gt.shape[0]
        tot_j += loss_j.item() * B
        tot_d += loss_d.item() * B
        n_total += B

    scheduler.step()
    avg_j = tot_j / n_total
    avg_d = tot_d / n_total
    avg   = avg_j + LAMBDA_DN * avg_d

    if epoch % 30 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{EPOCHS}  "
              f"jepa={avg_j:.5f}  denoise={avg_d:.5f}  total={avg:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), SAVE_PATH)

print(f"\nDone. Best loss={best_loss:.6f}  ->  {SAVE_PATH}")
