"""
Experiment 1: Multi-filling evaluation.

Key insight: at any filling n with PBC, ρ_i = N_total/L = const for all U.
So DFT analog (ρ → E₀) can distinguish fillings but NOT U at fixed filling.
Q-JEPA denoiser uses full γ → should outperform within each filling.

Methods:
  A) Q-JEPA denoiser  : γ₀ → encoder → z₀ → denoiser(z₀, H) → z* → head(z*)
  B) Oracle           : γ_GS → encoder → z_GS → head(z_GS)
  C) DFT analog       : ρ = diag(γ_GS) → MLP → E₀   (should fail within filling)
  D) Full-γ supervised: flatten(γ_GS) → MLP → E₀

H = [U/t, filling_n]  (h_dim=2)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import QJEPA, EnergyHead
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        compute_1rdm_fast, random_fock_state)

LATENT_DIM   = 64
H_DIM        = 2
HIDDEN       = 256
EPOCHS       = 600
LR           = 1e-3
BATCH_SIZE   = 32
N_LABEL_LIST = [10, 20, 50, 100]
N_SEEDS      = 5
N_FOCK_AVG   = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN_PATH= "checkpoints/jepa_filling.pt"
GS_PATH      = "data/hubbard_filling_gs.npz"
RESULTS_DIR  = "results"
L            = 6

FILLINGS = [(1, 1, 1/3), (2, 2, 2/3), (3, 3, 1.0)]
FILLING_VALS = [f for _, _, f in FILLINGS]

os.makedirs(RESULTS_DIR, exist_ok=True)

d          = np.load(GS_PATH)
U_all      = d["U"].astype(np.float32)
fill_all   = d["filling"].astype(np.float32)
E0_all     = d["E0"].astype(np.float32)
gamma_all  = d["gamma_gs"].astype(np.float32)
is_test    = d["is_test"].astype(bool)
H_all      = np.stack([U_all, fill_all], axis=-1).astype(np.float32)  # (N, 2)
n_total    = len(U_all)
rdm_dim    = gamma_all.shape[-1]
print(f"Dataset: N={n_total}, rdm_dim={rdm_dim}, device={DEVICE}")
print(f"Fillings: {np.unique(fill_all)}")
print(f"OOD test: {is_test.sum()} samples | SSL pool: {(~is_test).sum()} samples")

# Build ED structures per filling for random Fock states
# Use rounded keys to avoid float32 vs float64 mismatch
ED_STRUCTS = {}
for N_up, N_dn, fn in FILLINGS:
    basis     = build_basis(L, N_up, N_dn)
    basis_idx = make_basis_index(basis)
    ops       = build_all_cdagger_c(L, basis, basis_idx)
    ED_STRUCTS[round(fn, 4)] = (basis, ops)

def _get_ed(fn):
    return ED_STRUCTS[min(ED_STRUCTS, key=lambda k: abs(k - fn))]

# Load model
model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM, hidden=HIDDEN)
model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# OOD split: test = U values never seen during SSL pretraining
test_idx = np.where(is_test)[0]
pool_idx = np.where(~is_test)[0]

gamma_test = torch.tensor(gamma_all[test_idx], dtype=torch.float32, device=DEVICE)
H_test     = torch.tensor(H_all[test_idx],     dtype=torch.float32, device=DEVICE)
E0_test    = torch.tensor(E0_all[test_idx],    dtype=torch.float32, device=DEVICE)
fill_test  = fill_all[test_idx]


def sample_avg_z_star(U_np, fill_np, rng):
    z_list = []
    for U_val, fn in zip(U_np, fill_np):
        basis, ops = _get_ed(float(fn))
        H_t = torch.tensor([[U_val, fn]], dtype=torch.float32, device=DEVICE)
        zs  = []
        for _ in range(N_FOCK_AVG):
            psi0 = random_fock_state(basis, rng)
            g0   = torch.tensor(compute_1rdm_fast(psi0, ops, L)[None],
                                 dtype=torch.float32, device=DEVICE)
            zs.append(model.denoise_to_gs(g0, H_t))
        z_list.append(torch.stack(zs).mean(0))
    return torch.cat(z_list)


def train_head(z_tr, E0_tr):
    head = EnergyHead(latent_dim=LATENT_DIM, hidden=128).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ds   = TensorDataset(z_tr.detach(), E0_tr)
    dl   = DataLoader(ds, batch_size=min(BATCH_SIZE, len(z_tr)), shuffle=True)
    for _ in range(EPOCHS):
        for z_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(head(z_b), e_b).backward()
            opt.step()
        sch.step()
    return head


def train_mlp(X_tr, E0_tr, in_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, 256), nn.GELU(),
        nn.Linear(256, 128),    nn.GELU(),
        nn.Linear(128, 1)
    ).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ds  = TensorDataset(X_tr, E0_tr)
    dl  = DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True)
    for _ in range(EPOCHS):
        for x_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(mlp(x_b).squeeze(-1), e_b).backward()
            opt.step()
        sch.step()
    return mlp


def mae(pred, true): return float((pred - true).abs().mean())
def mae_by_filling(pred, true, fill):
    out = {}
    for fn in FILLING_VALS:
        m = np.abs(fill - fn) < 1e-4
        if m.sum() > 0:
            out[fn] = float(np.abs(pred[m] - true[m]).mean())
    return out


methods  = ["denoiser", "oracle", "dft_analog", "full_gamma"]
results  = {m: {n: [] for n in N_LABEL_LIST} for m in methods}
# Per-filling breakdown
results_by_fill = {m: {n: {fn: [] for fn in FILLING_VALS} for n in N_LABEL_LIST}
                   for m in methods}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 100)
        # Stratified sampling: n_labels // 3 per filling
        idx = []
        for fn in FILLING_VALS:
            mask = np.where(np.abs(fill_all[pool_idx] - fn) < 1e-4)[0]
            k    = min(n_labels // len(FILLING_VALS), len(mask))
            idx.extend(pool_idx[rng.choice(mask, size=k, replace=False)])
        idx = np.array(idx)

        U_tr_np   = U_all[idx]
        fill_tr_np= fill_all[idx]
        E0_tr     = torch.tensor(E0_all[idx],    dtype=torch.float32, device=DEVICE)
        H_tr      = torch.tensor(H_all[idx],     dtype=torch.float32, device=DEVICE)
        gamma_tr  = torch.tensor(gamma_all[idx], dtype=torch.float32, device=DEVICE)
        B_tr      = len(idx)

        # A) Q-JEPA denoiser
        z_tr_dn = sample_avg_z_star(U_tr_np, fill_tr_np, rng)
        head_dn = train_head(z_tr_dn, E0_tr)
        z_te_dn = sample_avg_z_star(U_all[test_idx], fill_test, rng)
        with torch.no_grad():
            pred_dn = head_dn(z_te_dn)
        results["denoiser"][n_labels].append(mae(pred_dn, E0_test))
        for fn, v in mae_by_filling(pred_dn.cpu().numpy(), E0_test.cpu().numpy(), fill_test).items():
            results_by_fill["denoiser"][n_labels][fn].append(v)

        # B) Oracle
        with torch.no_grad():
            z_tr_or = model.encode(gamma_tr)
            z_te_or = model.encode(gamma_test)
        head_or = train_head(z_tr_or, E0_tr)
        with torch.no_grad():
            pred_or = head_or(z_te_or)
        results["oracle"][n_labels].append(mae(pred_or, E0_test))
        for fn, v in mae_by_filling(pred_or.cpu().numpy(), E0_test.cpu().numpy(), fill_test).items():
            results_by_fill["oracle"][n_labels][fn].append(v)

        # C) DFT analog (density ρ = diag(γ))
        B_te  = gamma_test.shape[0]
        rho_tr = gamma_tr.reshape(B_tr, rdm_dim, rdm_dim).diagonal(dim1=-2, dim2=-1)
        rho_te = gamma_test.reshape(B_te, rdm_dim, rdm_dim).diagonal(dim1=-2, dim2=-1)
        mlp_dft = train_mlp(rho_tr, E0_tr, in_dim=rdm_dim)
        with torch.no_grad():
            pred_dft = mlp_dft(rho_te).squeeze(-1)
        results["dft_analog"][n_labels].append(mae(pred_dft, E0_test))
        for fn, v in mae_by_filling(pred_dft.cpu().numpy(), E0_test.cpu().numpy(), fill_test).items():
            results_by_fill["dft_analog"][n_labels][fn].append(v)

        # D) Full γ supervised
        g_tr_flat = gamma_tr.reshape(B_tr, -1)
        g_te_flat = gamma_test.reshape(B_te, -1)
        mlp_full  = train_mlp(g_tr_flat, E0_tr, in_dim=rdm_dim * rdm_dim)
        with torch.no_grad():
            pred_full = mlp_full(g_te_flat).squeeze(-1)
        results["full_gamma"][n_labels].append(mae(pred_full, E0_test))
        for fn, v in mae_by_filling(pred_full.cpu().numpy(), E0_test.cpu().numpy(), fill_test).items():
            results_by_fill["full_gamma"][n_labels][fn].append(v)

    row = "  ".join(
        f"{m}={np.mean(results[m][n_labels]):.4f}±{np.std(results[m][n_labels]):.4f}"
        for m in methods
    )
    print(f"N={n_labels:4d}  {row}")
    for fn in FILLING_VALS:
        row2 = "  ".join(
            f"{m}={np.mean(results_by_fill[m][n_labels][fn]):.4f}"
            for m in methods
        )
        print(f"  fill={fn:.2f}  {row2}")

np.save(os.path.join(RESULTS_DIR, "filling_results.npy"), results)
np.save(os.path.join(RESULTS_DIR, "filling_results_byfill.npy"), results_by_fill)
print(f"\nSaved -> {RESULTS_DIR}/filling_results.npy")
