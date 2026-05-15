"""
Train a decoder: z_GS → γ_GS (1-RDM reconstruction).
Also evaluates reconstruction quality by U/t regime.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import QJEPA, RDMDecoder

LATENT_DIM  = 64
HIDDEN      = 256
LR          = 1e-3
EPOCHS      = 500
BATCH_SIZE  = 64
TEST_FRAC   = 0.2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN    = "checkpoints/jepa_pretrained.pt"
GS_PATH     = "data/hubbard_gs.npz"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

d         = np.load(GS_PATH)
U_all     = torch.tensor(d["U"],        dtype=torch.float32)
gamma_all = torch.tensor(d["gamma_gs"], dtype=torch.float32)
n_total   = len(U_all)
rdm_dim   = gamma_all.shape[-1]
print(f"Dataset: N={n_total}, rdm_dim={rdm_dim}")

rng = np.random.default_rng(0)
idx = np.arange(n_total); rng.shuffle(idx)
n_test   = int(n_total * TEST_FRAC)
test_idx = idx[:n_test]; train_idx = idx[n_test:]

gamma_tr = gamma_all[train_idx].to(DEVICE)
U_tr     = U_all[train_idx].to(DEVICE)
gamma_te = gamma_all[test_idx].to(DEVICE)
U_te     = U_all[test_idx].to(DEVICE)
U_te_np  = U_all[test_idx].numpy()

# Load pretrained encoder
model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, hidden=HIDDEN)
model.load_state_dict(torch.load(PRETRAIN, map_location=DEVICE))
model = model.to(DEVICE).eval()

# Encode all training/test 1-RDMs
with torch.no_grad():
    z_tr = model.encoder(gamma_tr, U_tr)  # (N_tr, latent)
    z_te = model.encoder(gamma_te, U_te)

# Train decoder: z_GS → γ_GS
decoder = RDMDecoder(latent_dim=LATENT_DIM, rdm_dim=rdm_dim, hidden=HIDDEN).to(DEVICE)
opt   = torch.optim.Adam(decoder.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

ds = TensorDataset(z_tr, gamma_tr)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    decoder.train()
    for z_b, g_b in dl:
        opt.zero_grad()
        g_pred = decoder(z_b)
        loss = nn.functional.mse_loss(g_pred, g_b)
        loss.backward()
        opt.step()
    sched.step()

    if epoch % 100 == 0:
        decoder.eval()
        with torch.no_grad():
            loss_te = nn.functional.mse_loss(decoder(z_te), gamma_te).item()
        print(f"Epoch {epoch:4d}/{EPOCHS}  test_loss={loss_te:.6f}")
        if loss_te < best_loss:
            best_loss = loss_te
            torch.save(decoder.state_dict(), "checkpoints/decoder.pt")

# ── Evaluation ──────────────────────────────────────────────────────────────
decoder.load_state_dict(torch.load("checkpoints/decoder.pt", map_location=DEVICE))
decoder.eval()

with torch.no_grad():
    gamma_pred = decoder(z_te)

# Per-element MAE
mae_overall = (gamma_pred - gamma_te).abs().mean().item()

# MAE by U/t regime
masks = {
    "weak   (U<4)":   U_te_np < 4,
    "medium (4≤U<8)": (U_te_np >= 4) & (U_te_np < 8),
    "strong (U≥8)":   U_te_np >= 8,
}

print(f"\n--- 1-RDM Reconstruction MAE ---")
print(f"Overall:  {mae_overall:.5f}")
for regime, mask in masks.items():
    if mask.sum() == 0:
        continue
    m = (gamma_pred[mask] - gamma_te[mask]).abs().mean().item()
    print(f"{regime}: {m:.5f}")

# Also check diagonal (density) separately vs off-diagonal (correlations)
diag_err  = (gamma_pred.diagonal(dim1=-2, dim2=-1) -
             gamma_te.diagonal(dim1=-2, dim2=-1)).abs().mean().item()
# Off-diagonal
gamma_pred_np = gamma_pred.cpu().numpy()
gamma_te_np   = gamma_te.cpu().numpy()
off_err = np.abs(gamma_pred_np - gamma_te_np).mean() - diag_err
print(f"\nDiagonal (density):     {diag_err:.5f}")
print(f"Off-diagonal (correl.): {off_err:.5f}  (off-diag captures beyond-density info)")

# Save
np.savez(
    os.path.join(RESULTS_DIR, "decoder_eval.npz"),
    U_test=U_te_np,
    gamma_true=gamma_te_np,
    gamma_pred=gamma_pred_np,
    mae_overall=mae_overall,
)
print(f"\nSaved -> {RESULTS_DIR}/decoder_eval.npz")
