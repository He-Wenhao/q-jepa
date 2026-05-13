"""
Experiment 3: Iterative refinement of z*.

Idea: instead of one-shot denoiser(z₀, H) → z*, first apply the predictor
K steps (mimicking KS iteration), then apply the denoiser once.

z₀ → P(z₀,H) → P²(z₀,H) → ... → Pᴷ(z₀,H) → D(·,H) → z*_refined

Compare:
  A) One-shot denoiser       (K=0 predictor steps)
  B) Iterative K=3
  C) Iterative K=8
  D) Iterative K=20
  E) Oracle (encode γ_GS)
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
H_DIM        = 1
HIDDEN       = 256
EPOCHS       = 600
LR           = 1e-3
BATCH_SIZE   = 32
N_LABEL_LIST = [10, 20, 50, 100]
N_SEEDS      = 5
TEST_FRAC    = 0.2
N_FOCK_AVG   = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN_PATH= "checkpoints/jepa_pretrained.pt"
GS_PATH      = "data/hubbard_gs.npz"
RESULTS_DIR  = "results"
L            = 6
K_STEPS      = [0, 3, 8, 20]   # predictor steps before denoiser

os.makedirs(RESULTS_DIR, exist_ok=True)

d         = np.load(GS_PATH)
U_all     = d["U"].astype(np.float32)
E0_all    = d["E0"].astype(np.float32)
gamma_all = d["gamma_gs"].astype(np.float32)
n_total   = len(U_all)
rdm_dim   = gamma_all.shape[-1]
print(f"Dataset: N={n_total}, rdm_dim={rdm_dim}, device={DEVICE}")

basis     = build_basis(L, 3, 3)
basis_idx = make_basis_index(basis)
ops       = build_all_cdagger_c(L, basis, basis_idx)

model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM, hidden=HIDDEN)
model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

rng_split = np.random.default_rng(0)
all_idx   = np.arange(n_total); rng_split.shuffle(all_idx)
n_test    = int(n_total * TEST_FRAC)
test_idx  = all_idx[:n_test];  pool_idx = all_idx[n_test:]

gamma_test = torch.tensor(gamma_all[test_idx], dtype=torch.float32, device=DEVICE)
E0_test    = torch.tensor(E0_all[test_idx],    dtype=torch.float32, device=DEVICE)
U_test_np  = U_all[test_idx]


def get_z_star(U_np, rng, k_steps):
    """Encode random γ₀, apply k_steps predictor, then denoiser."""
    z_list = []
    for U_val in U_np:
        H_t = torch.tensor([[U_val]], dtype=torch.float32, device=DEVICE)
        zs  = []
        for _ in range(N_FOCK_AVG):
            psi0 = random_fock_state(basis, rng)
            g0   = torch.tensor(compute_1rdm_fast(psi0, ops, L)[None],
                                 dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                z = model.encoder(g0)
                for _ in range(k_steps):
                    z = model.predictor(z, H_t)
                z = model.denoiser(z, H_t)
            zs.append(z)
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


def mae(pred, true): return float((pred - true).abs().mean())


methods = [f"K={k}" for k in K_STEPS] + ["oracle"]
results = {m: {n: [] for n in N_LABEL_LIST} for m in methods}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 100)
        idx = rng.choice(pool_idx, size=n_labels, replace=False)

        E0_tr    = torch.tensor(E0_all[idx],    dtype=torch.float32, device=DEVICE)
        U_tr_np  = U_all[idx]
        gamma_tr = torch.tensor(gamma_all[idx], dtype=torch.float32, device=DEVICE)

        # Iterative refinement methods
        for k in K_STEPS:
            key = f"K={k}"
            z_tr = get_z_star(U_tr_np, rng, k)
            z_te = get_z_star(U_test_np, rng, k)
            head = train_head(z_tr, E0_tr)
            with torch.no_grad():
                pred = head(z_te)
            results[key][n_labels].append(mae(pred, E0_test))

        # Oracle
        with torch.no_grad():
            z_tr_or = model.encode(gamma_tr)
            z_te_or = model.encode(gamma_test)
        head_or = train_head(z_tr_or, E0_tr)
        with torch.no_grad():
            pred_or = head_or(z_te_or)
        results["oracle"][n_labels].append(mae(pred_or, E0_test))

    row = "  ".join(
        f"{m}={np.mean(results[m][n_labels]):.4f}±{np.std(results[m][n_labels]):.4f}"
        for m in methods
    )
    print(f"N={n_labels:4d}  {row}")

np.save(os.path.join(RESULTS_DIR, "iterative_refine_results.npy"), results)

# Diagnostic: how does z* quality change with K?
print("\n=== z* quality (linear probe 5-fold CV) ===")
rng_d = np.random.default_rng(42)
n = len(U_all)
from numpy.linalg import lstsq
idx_all = np.arange(n)
fold = n // 5
for k in K_STEPS:
    errs = []
    for f in range(5):
        te = idx_all[f*fold:(f+1)*fold]
        tr = np.concatenate([idx_all[:f*fold], idx_all[(f+1)*fold:]])
        z_tr_ = get_z_star(U_all[tr], rng_d, k).cpu().numpy()
        z_te_ = get_z_star(U_all[te], rng_d, k).cpu().numpy()
        A_tr = np.hstack([z_tr_, np.ones((len(tr),1))])
        coef, _, _, _ = lstsq(A_tr, E0_all[tr], rcond=None)
        A_te = np.hstack([z_te_, np.ones((len(te),1))])
        pred_ = A_te @ coef
        errs.append(np.abs(pred_ - E0_all[te]).mean())
    print(f"  K={k:2d}: MAE={np.mean(errs):.4f}±{np.std(errs):.4f}")

print(f"\nSaved -> {RESULTS_DIR}/iterative_refine_results.npy")
