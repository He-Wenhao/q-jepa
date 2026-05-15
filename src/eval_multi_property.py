"""
Multi-property prediction from the SAME z*.

This tests the "universal functional" claim:
  - ONE encoder + ONE denoiser trained with SSL (no labels)
  - For each property, train a SMALL HEAD: z* → property
  - The head never sees H = (U, filling) — only z*

Properties predicted:
  E₀    ground-state energy  (total)
  E_kin kinetic energy        (hopping: measures delocalization)
  D     double occupancy      (⟨n↑n↓⟩: hallmark of Mott physics, D→0 at large U)

All three vary strongly with U, so the DFT analog (ρ=const) fails for all.

Uses multi-filling model (h_dim=2) and all 3 filling fractions.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import QJEPA, EnergyHead
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        compute_1rdm_fast, random_fock_state)

# ── Config ────────────────────────────────────────────────────────────────────
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
PRETRAIN_PATH= os.path.join(ROOT, "checkpoints", "jepa_filling.pt")
GS_PATH      = os.path.join(ROOT, "data", "hubbard_filling_gs.npz")
RESULTS_DIR  = os.path.join(ROOT, "results")
L            = 6

FILLINGS     = [(1, 1, 1/3), (2, 2, 2/3), (3, 3, 1.0)]
FILLING_VALS = [fn for _, _, fn in FILLINGS]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
d          = np.load(GS_PATH)
U_all      = d["U"].astype(np.float32)
fill_all   = d["filling"].astype(np.float32)
E0_all     = d["E0"].astype(np.float32)
gamma_all  = d["gamma_gs"].astype(np.float32)
is_test    = d["is_test"].astype(bool)
H_all      = np.stack([U_all, fill_all], axis=-1).astype(np.float32)
rdm_dim    = gamma_all.shape[-1]

# ── Compute physical observables from γ_GS ────────────────────────────────────
def kinetic_energy(gamma_batch):
    """E_kin = -t Σ_{<ij>,σ} (γ_{ij} + γ_{ji})"""
    E = np.zeros(len(gamma_batch))
    for sigma in range(2):
        g = gamma_batch[:, sigma*L:(sigma+1)*L, sigma*L:(sigma+1)*L]
        for i in range(L):
            j = (i + 1) % L
            E += -(g[:, i, j] + g[:, j, i])
    return E

Ekin_all = kinetic_energy(gamma_all)

# Double occupancy: D = (E0 - E_kin) / U  for U > 0
# At U=0, D = (n↑)(n↓) per site = (fill/2)² / fill... simplify:
# non-interacting D_i = (N_up/L)(N_dn/L) = (fill/2)^2? No:
# D_i = n_{i↑} * n_{i↓} = (N_up/L)(N_dn/L) for non-interacting at uniform filling
# For balanced spin (N_up=N_dn), D_i = (N_up/L)² at U=0
# fill = N_total/L = 2*N_up/L → N_up/L = fill/2 → D = (fill/2)²
D_all = np.where(
    U_all > 0.2,
    (E0_all - Ekin_all) / U_all,
    (fill_all / 2) ** 2          # non-interacting limit at U=0
).astype(np.float32)

print(f"Dataset: N={len(U_all)}, rdm_dim={rdm_dim}, device={DEVICE}")
print(f"E₀   range: [{E0_all.min():.3f}, {E0_all.max():.3f}]")
print(f"E_kin range: [{Ekin_all.min():.3f}, {Ekin_all.max():.3f}]")
print(f"D    range: [{D_all.min():.3f}, {D_all.max():.3f}]")

# ── Build ED structures ───────────────────────────────────────────────────────
ED_STRUCTS = {}
for N_up, N_dn, fn in FILLINGS:
    basis = build_basis(L, N_up, N_dn)
    ops   = build_all_cdagger_c(L, basis, make_basis_index(basis))
    ED_STRUCTS[round(fn, 4)] = (basis, ops)

def _get_ed(fn):
    return ED_STRUCTS[min(ED_STRUCTS, key=lambda k: abs(k - fn))]

# ── Load model ────────────────────────────────────────────────────────────────
model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM, hidden=HIDDEN)
model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# ── OOD split: test = U values never seen during SSL pretraining ──────────────
test_idx = np.where(is_test)[0]
pool_idx = np.where(~is_test)[0]

gamma_test = torch.tensor(gamma_all[test_idx], dtype=torch.float32, device=DEVICE)
H_test     = torch.tensor(H_all[test_idx],     dtype=torch.float32, device=DEVICE)
fill_test  = fill_all[test_idx]
U_test_np  = U_all[test_idx]

targets_test = {
    "E0":   torch.tensor(E0_all[test_idx],   dtype=torch.float32, device=DEVICE),
    "Ekin": torch.tensor(Ekin_all[test_idx], dtype=torch.float32, device=DEVICE),
    "D":    torch.tensor(D_all[test_idx],    dtype=torch.float32, device=DEVICE),
}


def sample_z_star(U_np, fill_np, rng):
    z_list = []
    for U_val, fn in zip(U_np, fill_np):
        basis, ops = _get_ed(float(fn))
        H_t = torch.tensor([[U_val, fn]], dtype=torch.float32, device=DEVICE)
        zs  = []
        for _ in range(N_FOCK_AVG):
            psi0 = random_fock_state(basis, rng)
            g0   = torch.tensor(compute_1rdm_fast(psi0, ops, L)[None],
                                 dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                z0 = model.encoder(g0)
                zs.append(model.denoiser(z0, H_t))
        z_list.append(torch.stack(zs).mean(0))
    return torch.cat(z_list)


def train_head(z_tr, y_tr):
    head = EnergyHead(latent_dim=LATENT_DIM, hidden=128).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ds   = TensorDataset(z_tr.detach(), y_tr)
    dl   = DataLoader(ds, batch_size=min(BATCH_SIZE, len(z_tr)), shuffle=True)
    for _ in range(EPOCHS):
        for z_b, y_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(head(z_b), y_b).backward()
            opt.step()
        sch.step()
    return head


def train_mlp(X_tr, y_tr, in_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, 256), nn.GELU(),
        nn.Linear(256, 128),    nn.GELU(),
        nn.Linear(128, 1)
    ).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ds  = TensorDataset(X_tr, y_tr)
    dl  = DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True)
    for _ in range(EPOCHS):
        for x_b, y_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(mlp(x_b).squeeze(-1), y_b).backward()
            opt.step()
        sch.step()
    return mlp


def mae(pred, true): return float((pred - true).abs().mean())


# ── Run experiments ───────────────────────────────────────────────────────────
methods  = ["denoiser", "oracle", "full_gamma", "dft_analog"]
prop_names = ["E0", "Ekin", "D"]

# results[method][property][n_labels] = list of MAE across seeds
results = {m: {p: {n: [] for n in N_LABEL_LIST} for p in prop_names}
           for m in methods}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 500)
        idx = []
        for fn in FILLING_VALS:
            mask = np.where(np.abs(fill_all[pool_idx] - fn) < 1e-4)[0]
            k    = min(n_labels // len(FILLING_VALS), len(mask))
            idx.extend(pool_idx[rng.choice(mask, size=k, replace=False)])
        idx = np.array(idx)

        U_tr    = U_all[idx]; fill_tr = fill_all[idx]
        gamma_tr= torch.tensor(gamma_all[idx], dtype=torch.float32, device=DEVICE)
        B_tr    = len(idx)
        targets_tr = {
            "E0":   torch.tensor(E0_all[idx],   dtype=torch.float32, device=DEVICE),
            "Ekin": torch.tensor(Ekin_all[idx], dtype=torch.float32, device=DEVICE),
            "D":    torch.tensor(D_all[idx],    dtype=torch.float32, device=DEVICE),
        }

        # ── A) Q-JEPA denoiser: one z*, separate head per property
        z_tr_dn = sample_z_star(U_tr, fill_tr, rng)
        z_te_dn = sample_z_star(U_test_np, fill_test, rng)
        for prop in prop_names:
            head = train_head(z_tr_dn, targets_tr[prop])
            with torch.no_grad():
                pred = head(z_te_dn)
            results["denoiser"][prop][n_labels].append(mae(pred, targets_test[prop]))

        # ── B) Oracle: encode γ_GS
        with torch.no_grad():
            z_tr_or = model.encoder(gamma_tr)
            z_te_or = model.encoder(gamma_test)
        for prop in prop_names:
            head = train_head(z_tr_or, targets_tr[prop])
            with torch.no_grad():
                pred = head(z_te_or)
            results["oracle"][prop][n_labels].append(mae(pred, targets_test[prop]))

        # ── C) Full-γ supervised
        g_tr_f = gamma_tr.reshape(B_tr, -1)
        g_te_f = gamma_test.reshape(len(test_idx), -1)
        for prop in prop_names:
            mlp = train_mlp(g_tr_f, targets_tr[prop], in_dim=rdm_dim * rdm_dim)
            with torch.no_grad():
                pred = mlp(g_te_f).squeeze(-1)
            results["full_gamma"][prop][n_labels].append(mae(pred, targets_test[prop]))

        # ── D) DFT analog: density ρ = diag(γ)
        B_te   = len(test_idx)
        rho_tr = gamma_tr.reshape(B_tr, rdm_dim, rdm_dim).diagonal(dim1=-2, dim2=-1)
        rho_te = gamma_test.reshape(B_te, rdm_dim, rdm_dim).diagonal(dim1=-2, dim2=-1)
        for prop in prop_names:
            mlp = train_mlp(rho_tr, targets_tr[prop], in_dim=rdm_dim)
            with torch.no_grad():
                pred = mlp(rho_te).squeeze(-1)
            results["dft_analog"][prop][n_labels].append(mae(pred, targets_test[prop]))

    # Print progress
    print(f"\nN={n_labels}")
    for prop in prop_names:
        row = "  ".join(
            f"{m[:6]}={np.mean(results[m][prop][n_labels]):.4f}"
            for m in methods
        )
        print(f"  {prop:5s}: {row}")

np.save(os.path.join(RESULTS_DIR, "multi_property_results.npy"), results)
print(f"\nSaved -> results/multi_property_results.npy")
