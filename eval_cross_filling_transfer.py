"""
Cross-filling transfer experiment (cleaner Exp 2 design).

Setup:
  - SSL pretraining on n=1 (half-filling) ONLY  →  jepa_halffilling.pt
  - Few-shot fine-tune on n=1/3 (quarter-filling) labels
  - Test on n=1/3

vs.

  - SSL pretrained on ALL fillings  →  jepa_filling.pt
  - Few-shot fine-tune on n=1/3 labels
  - Test on n=1/3

vs. Scratch baselines (no SSL)

This tests: does SSL on half-filling learn a universal encoder
that transfers to quarter-filling, despite different physics?
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
H_DIM_SINGLE = 1    # half-filling only model: H=[U]
H_DIM_MULTI  = 2    # multi-filling model: H=[U, filling_n]
HIDDEN       = 256
EPOCHS_PRETRAIN = 300
EPOCHS_HEAD  = 600
LR           = 1e-3
BATCH_SIZE   = 32
N_LABEL_LIST = [5, 10, 20, 50]
N_SEEDS      = 5
N_FOCK_AVG   = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
L            = 6

TARGET_FILL  = 1/3      # target domain for few-shot fine-tuning + testing
TARGET_N_UP  = 1; TARGET_N_DN = 1

CKPT_HALF  = "checkpoints/jepa_pretrained.pt"   # trained on n=1 half-filling only
CKPT_MULTI = "checkpoints/jepa_filling.pt"       # trained on all fillings
GS_FILL    = "data/hubbard_filling_gs.npz"
SSL_HALF   = "data/hubbard_ssl.npz"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load target-domain ground states (n=1/3) ──────────────────────────────────
d         = np.load(GS_FILL)
fill_all  = d["filling"].astype(np.float32)
U_all     = d["U"].astype(np.float32)
E0_all    = d["E0"].astype(np.float32)
gamma_all = d["gamma_gs"].astype(np.float32)
H_all     = np.stack([U_all, fill_all], axis=-1).astype(np.float32)
rdm_dim   = gamma_all.shape[-1]

target_mask = np.abs(fill_all - TARGET_FILL) < 1e-4
tgt_idx     = np.where(target_mask)[0]
U_tgt       = U_all[tgt_idx]; E0_tgt = E0_all[tgt_idx]
gamma_tgt   = gamma_all[tgt_idx]; H_tgt = H_all[tgt_idx]
print(f"Target domain n={TARGET_FILL:.2f}: {len(tgt_idx)} samples")

rng_split  = np.random.default_rng(0)
perm       = rng_split.permutation(len(tgt_idx))
n_test     = int(len(tgt_idx) * 0.2)
te_local   = perm[:n_test]; pool_local = perm[n_test:]

U_test     = U_tgt[te_local]
E0_test_t  = torch.tensor(E0_tgt[te_local],    dtype=torch.float32, device=DEVICE)
gamma_test = torch.tensor(gamma_tgt[te_local], dtype=torch.float32, device=DEVICE)
H_test_2d  = torch.tensor(H_tgt[te_local],     dtype=torch.float32, device=DEVICE)

# ED structures for target domain
basis_tgt = build_basis(L, TARGET_N_UP, TARGET_N_DN)
ops_tgt   = build_all_cdagger_c(L, basis_tgt, make_basis_index(basis_tgt))

# ED structures for half-filling (for random Fock states from source domain)
basis_half = build_basis(L, 3, 3)
ops_half   = build_all_cdagger_c(L, basis_half, make_basis_index(basis_half))

# ── Load models ───────────────────────────────────────────────────────────────
model_half = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM_SINGLE, hidden=HIDDEN)
model_half.load_state_dict(torch.load(CKPT_HALF, map_location=DEVICE))
model_half = model_half.to(DEVICE).eval()

model_multi = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM_MULTI, hidden=HIDDEN)
model_multi.load_state_dict(torch.load(CKPT_MULTI, map_location=DEVICE))
model_multi = model_multi.to(DEVICE).eval()


def sample_z_star_half(U_np, rng):
    """Use half-filling model: only H=[U] (dim 1)."""
    z_list = []
    for U_val in U_np:
        H_t = torch.tensor([[U_val]], dtype=torch.float32, device=DEVICE)
        zs  = []
        for _ in range(N_FOCK_AVG):
            # Use TARGET domain Fock states (n=1/3), encode with half-filling model
            psi0 = random_fock_state(basis_tgt, rng)
            g0   = torch.tensor(compute_1rdm_fast(psi0, ops_tgt, L)[None],
                                 dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                zs.append(model_half.denoise_to_gs(g0, H_t))
        z_list.append(torch.stack(zs).mean(0))
    return torch.cat(z_list)


def sample_z_star_multi(U_np, rng):
    """Use multi-filling model: H=[U, n=1/3] (dim 2)."""
    z_list = []
    for U_val in U_np:
        H_t = torch.tensor([[U_val, TARGET_FILL]], dtype=torch.float32, device=DEVICE)
        zs  = []
        for _ in range(N_FOCK_AVG):
            psi0 = random_fock_state(basis_tgt, rng)
            g0   = torch.tensor(compute_1rdm_fast(psi0, ops_tgt, L)[None],
                                 dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                z0 = model_multi.encoder(g0)
                zs.append(model_multi.denoiser(z0, H_t))
        z_list.append(torch.stack(zs).mean(0))
    return torch.cat(z_list)


def train_head(z_tr, E0_tr, latent_dim):
    head = EnergyHead(latent_dim=latent_dim, hidden=128).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_HEAD)
    ds   = TensorDataset(z_tr.detach(), E0_tr)
    dl   = DataLoader(ds, batch_size=min(BATCH_SIZE, len(z_tr)), shuffle=True)
    for _ in range(EPOCHS_HEAD):
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
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_HEAD)
    ds  = TensorDataset(X_tr, E0_tr)
    dl  = DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True)
    for _ in range(EPOCHS_HEAD):
        for x_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(mlp(x_b).squeeze(-1), e_b).backward()
            opt.step()
        sch.step()
    return mlp


def mae(pred, true): return float((pred - true).abs().mean())


methods  = ["transfer_half", "transfer_multi", "oracle_multi", "scratch_gamma"]
results  = {m: {n: [] for n in N_LABEL_LIST} for m in methods}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 300)
        tr_local = rng.choice(pool_local, size=min(n_labels, len(pool_local)), replace=False)

        U_tr      = U_tgt[tr_local]
        E0_tr_t   = torch.tensor(E0_tgt[tr_local],    dtype=torch.float32, device=DEVICE)
        gamma_tr  = torch.tensor(gamma_tgt[tr_local], dtype=torch.float32, device=DEVICE)
        B_tr      = len(tr_local)

        # A) Transfer from half-filling model (never saw n=1/3 during SSL)
        z_tr_h = sample_z_star_half(U_tr, rng)
        z_te_h = sample_z_star_half(U_test, rng)
        head_h = train_head(z_tr_h, E0_tr_t, LATENT_DIM)
        with torch.no_grad():
            pred_h = head_h(z_te_h)
        results["transfer_half"][n_labels].append(mae(pred_h, E0_test_t))

        # B) Multi-filling model (SSL saw n=1/3 during training)
        z_tr_m = sample_z_star_multi(U_tr, rng)
        z_te_m = sample_z_star_multi(U_test, rng)
        head_m = train_head(z_tr_m, E0_tr_t, LATENT_DIM)
        with torch.no_grad():
            pred_m = head_m(z_te_m)
        results["transfer_multi"][n_labels].append(mae(pred_m, E0_test_t))

        # C) Oracle: encode γ_GS with multi-filling model
        with torch.no_grad():
            z_tr_or = model_multi.encoder(gamma_tr)
            z_te_or = model_multi.encoder(gamma_test)
        head_or = train_head(z_tr_or, E0_tr_t, LATENT_DIM)
        with torch.no_grad():
            pred_or = head_or(z_te_or)
        results["oracle_multi"][n_labels].append(mae(pred_or, E0_test_t))

        # D) Scratch: flatten(γ_GS) → MLP (no SSL)
        g_tr_f  = gamma_tr.reshape(B_tr, -1)
        g_te_f  = gamma_test.reshape(n_test, -1)
        mlp_g   = train_mlp(g_tr_f, E0_tr_t, in_dim=rdm_dim * rdm_dim)
        with torch.no_grad():
            pred_g = mlp_g(g_te_f).squeeze(-1)
        results["scratch_gamma"][n_labels].append(mae(pred_g, E0_test_t))

    row = "  ".join(
        f"{m}={np.mean(results[m][n_labels]):.4f}±{np.std(results[m][n_labels]):.4f}"
        for m in methods
    )
    print(f"N={n_labels:3d}  {row}")

np.save(os.path.join(RESULTS_DIR, "cross_filling_transfer_results.npy"), results)
print(f"\nSaved -> {RESULTS_DIR}/cross_filling_transfer_results.npy")

# Print interpretation
print("\nKey comparison (transfer_half: SSL never saw n=1/3, must generalize)")
print("vs scratch_gamma: supervised only, no SSL")
print("vs transfer_multi: SSL DID see n=1/3 during pretraining")
