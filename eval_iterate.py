"""
Core evaluation of Q-JEPA (corrected architecture).

Methods compared:
  A) Q-JEPA denoiser  : γ₀ → encoder → z₀ → denoiser(z₀, H) → z* → head(z*) → E₀
                        (head has NO H input — universal functional)
  B) Oracle           : γ_GS → encoder → z* → head(z*) → E₀
                        (upper bound: cheats by using exact ground-state γ)
  C) DFT analog       : ρ = diag(γ_GS) → MLP → E₀
                        (density as latent, half-filling → ρ=const → should fail)
  D) Full-γ supervised: flatten(γ_GS) → MLP → E₀
                        (full 1-RDM as descriptor, no SSL, direct supervised)
  E) Hartree-Fock     : fixed MAE from physics (no labels)

N_labels ∈ {10, 20, 50, 100}, 5 random seeds.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import QJEPA, EnergyHead
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        compute_1rdm_fast, random_fock_state)

LATENT_DIM    = 64
H_DIM         = 1
HIDDEN        = 256
EPOCHS        = 600
LR            = 1e-3
BATCH_SIZE    = 32
N_LABEL_LIST  = [10, 20, 50, 100]
N_SEEDS       = 5
TEST_FRAC     = 0.2
N_FOCK_AVG    = 5      # Fock states to average for z₀ estimate
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN_PATH = "checkpoints/jepa_pretrained.pt"
GS_PATH       = "data/hubbard_gs.npz"
HF_PATH       = "data/hubbard_hf.npz"
RESULTS_DIR   = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
d         = np.load(GS_PATH)
U_all     = d["U"].astype(np.float32)
E0_all    = d["E0"].astype(np.float32)
gamma_all = d["gamma_gs"].astype(np.float32)
n_total   = len(U_all)
rdm_dim   = gamma_all.shape[-1]
L         = 6
print(f"Dataset: N={n_total}, rdm_dim={rdm_dim}, device={DEVICE}")

# HF MAE
E_hf    = np.load(HF_PATH)["E0_HF"].astype(np.float32)
HF_MAE  = float(np.abs(E_hf - E0_all).mean())

# Build Hubbard structures for random Fock states
basis     = build_basis(L, 3, 3)
basis_idx = make_basis_index(basis)
ops       = build_all_cdagger_c(L, basis, basis_idx)

# Fixed 80/20 split
rng_split = np.random.default_rng(0)
all_idx   = np.arange(n_total); rng_split.shuffle(all_idx)
n_test    = int(n_total * TEST_FRAC)
test_idx  = all_idx[:n_test];  pool_idx = all_idx[n_test:]

# Test tensors
gamma_test = torch.tensor(gamma_all[test_idx], dtype=torch.float32, device=DEVICE)
H_test     = torch.tensor(U_all[test_idx],     dtype=torch.float32, device=DEVICE).unsqueeze(-1)
E0_test    = torch.tensor(E0_all[test_idx],    dtype=torch.float32, device=DEVICE)
U_test_np  = U_all[test_idx]

# Load pretrained model
model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM, hidden=HIDDEN)
model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()


def sample_avg_z_star(U_np, rng):
    """For each U value, sample N_FOCK_AVG Fock states and average z*."""
    z_list = []
    for U_val in U_np:
        H_t = torch.tensor([[U_val]], dtype=torch.float32, device=DEVICE)
        zs  = []
        for _ in range(N_FOCK_AVG):
            psi0 = random_fock_state(basis, rng)
            g0   = torch.tensor(compute_1rdm_fast(psi0, ops, L)[None],
                                 dtype=torch.float32, device=DEVICE)
            zs.append(model.denoise_to_gs(g0, H_t))
        z_list.append(torch.stack(zs).mean(0))
    return torch.cat(z_list)   # (B, latent_dim)


def train_head(z_tr, E0_tr, epochs=EPOCHS):
    """Train EnergyHead: z* → E₀  (no H input)."""
    head = EnergyHead(latent_dim=LATENT_DIM, hidden=128).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B    = len(z_tr)
    ds   = TensorDataset(z_tr.detach(), E0_tr)
    dl   = DataLoader(ds, batch_size=min(BATCH_SIZE, B), shuffle=True)
    for _ in range(epochs):
        for z_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(head(z_b), e_b).backward()
            opt.step()
        sch.step()
    return head


def train_mlp(X_tr, E0_tr, in_dim, epochs=EPOCHS):
    """Generic MLP regression: X → E₀."""
    mlp = nn.Sequential(
        nn.Linear(in_dim, 256), nn.GELU(),
        nn.Linear(256, 128),    nn.GELU(),
        nn.Linear(128, 1)
    ).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B   = len(X_tr)
    ds  = TensorDataset(X_tr, E0_tr)
    dl  = DataLoader(ds, batch_size=min(BATCH_SIZE, B), shuffle=True)
    for _ in range(epochs):
        for x_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(mlp(x_b).squeeze(-1), e_b).backward()
            opt.step()
        sch.step()
    return mlp


def mae(pred, true): return float((pred - true).abs().mean())


# ── Run experiments ───────────────────────────────────────────────────────────
methods  = ["denoiser", "oracle", "dft_analog", "full_gamma"]
results  = {m: {n: [] for n in N_LABEL_LIST} for m in methods}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 100)
        idx = rng.choice(pool_idx, size=n_labels, replace=False)

        U_tr_np  = U_all[idx]
        E0_tr    = torch.tensor(E0_all[idx], dtype=torch.float32, device=DEVICE)
        H_tr     = torch.tensor(U_tr_np, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        gamma_tr = torch.tensor(gamma_all[idx], dtype=torch.float32, device=DEVICE)

        # A) Q-JEPA denoiser (from random γ₀, no H in head)
        z_tr_dn   = sample_avg_z_star(U_tr_np, rng)
        head_dn   = train_head(z_tr_dn, E0_tr)
        z_te_dn   = sample_avg_z_star(U_test_np, rng)
        with torch.no_grad():
            pred_dn = head_dn(z_te_dn)
        results["denoiser"][n_labels].append(mae(pred_dn, E0_test))

        # B) Oracle: encode γ_GS directly (no H in head)
        with torch.no_grad():
            z_tr_or = model.encode(gamma_tr)
            z_te_or = model.encode(gamma_test)
        head_or = train_head(z_tr_or, E0_tr)
        with torch.no_grad():
            pred_or = head_or(z_te_or)
        results["oracle"][n_labels].append(mae(pred_or, E0_test))

        # C) DFT analog: only density ρ = diag(γ_GS) → E₀
        B_tr = gamma_tr.shape[0]
        rho_tr = gamma_tr.reshape(B_tr, rdm_dim, rdm_dim).diagonal(dim1=-2, dim2=-1)
        B_te = gamma_test.shape[0]
        rho_te = gamma_test.reshape(B_te, rdm_dim, rdm_dim).diagonal(dim1=-2, dim2=-1)
        mlp_dft = train_mlp(rho_tr, E0_tr, in_dim=rdm_dim)
        with torch.no_grad():
            pred_dft = mlp_dft(rho_te).squeeze(-1)
        results["dft_analog"][n_labels].append(mae(pred_dft, E0_test))

        # D) Full γ supervised (flatten γ_GS → MLP → E₀, no SSL)
        g_tr_flat = gamma_tr.reshape(B_tr, -1)
        g_te_flat = gamma_test.reshape(B_te, -1)
        mlp_full  = train_mlp(g_tr_flat, E0_tr, in_dim=rdm_dim * rdm_dim)
        with torch.no_grad():
            pred_full = mlp_full(g_te_flat).squeeze(-1)
        results["full_gamma"][n_labels].append(mae(pred_full, E0_test))

    row = "  ".join(
        f"{m}={np.mean(results[m][n_labels]):.4f}±{np.std(results[m][n_labels]):.4f}"
        for m in methods
    )
    print(f"N={n_labels:4d}  {row}")

np.save(os.path.join(RESULTS_DIR, "iterate_results.npy"), results)
print(f"\nHF baseline (no labels): MAE={HF_MAE:.4f}")
print(f"Saved -> {RESULTS_DIR}/iterate_results.npy")
