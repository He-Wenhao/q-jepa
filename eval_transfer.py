"""
Experiment 2: Transfer learning.

Setup:
  - SSL pretraining on U ∈ [0, U_pretrain_max] (weak-to-medium correlation)
  - Few-shot fine-tune on U ∈ [U_target_min, 12] (strong correlation) labels
  - Test on strong-correlation systems

Methods compared:
  A) Q-JEPA transfer: pretrained on U<4, few-shot head on U>=8
  B) Scratch MLP: direct (U, E0) mapping from scratch using only strong-U labels
  C) Q-JEPA full: pretrained on ALL U, few-shot head on U>=8  (upper bound)

Shows: SSL pretraining on unlabeled weak-U trajectories transfers to strong-U.
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
EPOCHS_PRETRAIN = 300
EPOCHS_HEAD   = 600
LR            = 1e-3
BATCH_SIZE    = 32
N_LABEL_LIST  = [5, 10, 20, 50]
N_SEEDS       = 5
N_FOCK_AVG    = 5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

U_PRETRAIN_MAX  = 4.0   # SSL pretraining domain
U_TARGET_MIN    = 8.0   # few-shot target domain
L               = 6

SSL_PATH  = "data/hubbard_ssl.npz"    # existing half-filling data
GS_PATH   = "data/hubbard_gs.npz"
CKPT_WEAK = "checkpoints/jepa_weak_u.pt"      # pretrained on U<4
CKPT_FULL = "checkpoints/jepa_pretrained.pt"  # pretrained on all U
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
d_ssl = np.load(SSL_PATH)
gamma_t_all  = d_ssl["gamma_t"]
gamma_tp_all = d_ssl["gamma_tp"]
gamma_gs_all = d_ssl["gamma_gs"]
U_ssl_all    = d_ssl["U"]

d_gs       = np.load(GS_PATH)
U_gs       = d_gs["U"].astype(np.float32)
E0_gs      = d_gs["E0"].astype(np.float32)
gamma_gs_d = d_gs["gamma_gs"].astype(np.float32)

rdm_dim = gamma_t_all.shape[-1]
print(f"Total SSL pairs: {len(U_ssl_all)}, GS: {len(U_gs)}, rdm_dim={rdm_dim}")

# Filter: strong-U ground states for few-shot + test
strong_mask  = U_gs >= U_TARGET_MIN
strong_idx   = np.where(strong_mask)[0]
U_strong     = U_gs[strong_idx]
E0_strong    = E0_gs[strong_idx]
gamma_strong = gamma_gs_d[strong_idx]
print(f"Strong-U (U>={U_TARGET_MIN}) samples: {len(strong_idx)}")

# Filter: weak-U SSL data for pretraining
weak_ssl_mask = U_ssl_all <= U_PRETRAIN_MAX
print(f"Weak-U (U<={U_PRETRAIN_MAX}) SSL pairs: {weak_ssl_mask.sum()}")

# ── Step 1: pretrain on weak-U data ──────────────────────────────────────────
if not os.path.exists(CKPT_WEAK):
    print(f"\n--- Pretraining on U<={U_PRETRAIN_MAX} ---")
    gt  = torch.tensor(gamma_t_all[weak_ssl_mask],  dtype=torch.float32)
    gtp = torch.tensor(gamma_tp_all[weak_ssl_mask], dtype=torch.float32)
    gGs = torch.tensor(gamma_gs_all[weak_ssl_mask], dtype=torch.float32)
    Hv  = torch.tensor(U_ssl_all[weak_ssl_mask],    dtype=torch.float32).unsqueeze(-1)

    ds = TensorDataset(gt, gtp, gGs, Hv)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    model_w = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM,
                    hidden=HIDDEN, ema_decay=0.996).to(DEVICE)
    opt = torch.optim.AdamW(
        list(model_w.encoder.parameters()) +
        list(model_w.predictor.parameters()) +
        list(model_w.denoiser.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PRETRAIN)
    best = float("inf")
    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        model_w.train()
        tot_j = tot_d = nb = 0
        for b_gt, b_gtp, b_gGs, b_H in dl:
            b_gt, b_gtp, b_gGs, b_H = (x.to(DEVICE) for x in (b_gt, b_gtp, b_gGs, b_H))
            opt.zero_grad()
            lj, ld = model_w(b_gt, b_gtp, b_gGs, b_H)
            loss = lj + 2.0 * ld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_w.parameters(), 1.0)
            opt.step(); model_w.update_target()
            tot_j += lj.item() * len(b_gt); tot_d += ld.item() * len(b_gt)
            nb += len(b_gt)
        sch.step()
        avg = (tot_j + 2.0 * tot_d) / nb
        if epoch % 60 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{EPOCHS_PRETRAIN}  loss={avg:.5f}")
        if avg < best:
            best = avg; torch.save(model_w.state_dict(), CKPT_WEAK)
    print(f"Saved -> {CKPT_WEAK}  (best={best:.6f})")
else:
    print(f"Using cached {CKPT_WEAK}")

# Build ED structures for strong-U Fock states
basis     = build_basis(L, 3, 3)   # half-filling (same as training data)
basis_idx = make_basis_index(basis)
ops       = build_all_cdagger_c(L, basis, basis_idx)


def sample_z_star(model, U_np, rng):
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
    return torch.cat(z_list)


def train_head(z_tr, E0_tr):
    head = EnergyHead(latent_dim=LATENT_DIM, hidden=128).to(DEVICE)
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


def train_mlp_scalar(X_tr, E0_tr, in_dim):
    """Simple MLP from X (scalar U or flattened γ) to E₀."""
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

# ── Load models ───────────────────────────────────────────────────────────────
model_weak = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM, hidden=HIDDEN)
model_weak.load_state_dict(torch.load(CKPT_WEAK, map_location=DEVICE))
model_weak = model_weak.to(DEVICE).eval()

model_full = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, h_dim=H_DIM, hidden=HIDDEN)
model_full.load_state_dict(torch.load(CKPT_FULL, map_location=DEVICE))
model_full = model_full.to(DEVICE).eval()

# Fixed test: hold out last 20% of strong-U samples
rng_split = np.random.default_rng(0)
perm      = rng_split.permutation(len(strong_idx))
n_test    = int(len(strong_idx) * 0.2)
te_idx    = perm[:n_test]
pool_strong = perm[n_test:]

U_test_strong     = U_strong[te_idx]
E0_test_strong    = E0_strong[te_idx]
gamma_test_strong = gamma_strong[te_idx]
E0_test_t  = torch.tensor(E0_test_strong, dtype=torch.float32, device=DEVICE)
gamma_test_t = torch.tensor(gamma_test_strong, dtype=torch.float32, device=DEVICE)

methods  = ["transfer_weak", "transfer_full", "scratch_U", "scratch_gamma"]
results  = {m: {n: [] for n in N_LABEL_LIST} for m in methods}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 200)
        tr_idx_local = rng.choice(pool_strong, size=min(n_labels, len(pool_strong)), replace=False)

        U_tr    = U_strong[tr_idx_local]
        E0_tr_t = torch.tensor(E0_strong[tr_idx_local], dtype=torch.float32, device=DEVICE)
        gamma_tr_t = torch.tensor(gamma_strong[tr_idx_local], dtype=torch.float32, device=DEVICE)

        # A) Transfer from weak-U pretraining
        z_tr_w = sample_z_star(model_weak, U_tr, rng)
        z_te_w = sample_z_star(model_weak, U_test_strong, rng)
        head_w = train_head(z_tr_w, E0_tr_t)
        with torch.no_grad():
            pred_w = head_w(z_te_w)
        results["transfer_weak"][n_labels].append(mae(pred_w, E0_test_t))

        # B) Transfer from full pretraining
        z_tr_f = sample_z_star(model_full, U_tr, rng)
        z_te_f = sample_z_star(model_full, U_test_strong, rng)
        head_f = train_head(z_tr_f, E0_tr_t)
        with torch.no_grad():
            pred_f = head_f(z_te_f)
        results["transfer_full"][n_labels].append(mae(pred_f, E0_test_t))

        # C) Scratch: U → E₀ MLP (no SSL, no γ)
        U_tr_t  = torch.tensor(U_tr[:, None],                  dtype=torch.float32, device=DEVICE)
        U_te_t  = torch.tensor(U_test_strong[:, None],         dtype=torch.float32, device=DEVICE)
        mlp_U   = train_mlp_scalar(U_tr_t, E0_tr_t, in_dim=1)
        with torch.no_grad():
            pred_U = mlp_U(U_te_t).squeeze(-1)
        results["scratch_U"][n_labels].append(mae(pred_U, E0_test_t))

        # D) Scratch: γ_GS → E₀ MLP (supervised, no SSL)
        g_tr_flat = gamma_tr_t.reshape(len(tr_idx_local), -1)
        g_te_flat = gamma_test_t.reshape(n_test, -1)
        mlp_g     = train_mlp_scalar(g_tr_flat, E0_tr_t, in_dim=rdm_dim * rdm_dim)
        with torch.no_grad():
            pred_g = mlp_g(g_te_flat).squeeze(-1)
        results["scratch_gamma"][n_labels].append(mae(pred_g, E0_test_t))

    row = "  ".join(
        f"{m}={np.mean(results[m][n_labels]):.4f}±{np.std(results[m][n_labels]):.4f}"
        for m in methods
    )
    print(f"N={n_labels:3d}  {row}")

np.save(os.path.join(RESULTS_DIR, "transfer_results.npy"), results)
print(f"\nSaved -> {RESULTS_DIR}/transfer_results.npy")
