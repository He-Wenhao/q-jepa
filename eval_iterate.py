"""
Proper evaluation of Q-JEPA:

For each test (U, E0) pair:
  1. Start from random Fock state gamma_0
  2. Encode z_0 = encoder(gamma_0, U)
  3. Iterate predictor K steps: z_0 -> z_1 -> ... -> z_K
  4. Predict E0 = head(z_K)

Compare to baselines:
  - Direct U->E0 regression (polynomial / MLP)
  - JEPA but using z(gamma_GS) instead of iterating (upper bound)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import QJEPA
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        compute_1rdm_fast, random_fock_state)

# ── Config ──────────────────────────────────────────────────────────────────
LATENT_DIM    = 64
HIDDEN        = 256
ITER_STEPS    = 50      # predictor unrolling steps at inference
EPOCHS_HEAD   = 500
LR_HEAD       = 1e-3
BATCH_SIZE    = 32
N_LABEL_LIST  = [10, 20, 50, 100]
N_SEEDS       = 5
TEST_FRAC     = 0.2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN_PATH = "checkpoints/jepa_pretrained.pt"
GS_PATH       = "data/hubbard_gs.npz"
RESULTS_DIR   = "results"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load ground-state data
d = np.load(GS_PATH)
U_all     = d["U"].astype(np.float32)
E0_all    = d["E0"].astype(np.float32)
gamma_all = d["gamma_gs"].astype(np.float32)
n_total   = len(U_all)
L         = 6
rdm_dim   = gamma_all.shape[-1]

# Build Hubbard structures (for generating random Fock states)
basis     = build_basis(L, 3, 3)
basis_idx = make_basis_index(basis)
ops       = build_all_cdagger_c(L, basis, basis_idx)
print(f"Dataset: N={n_total}, rdm_dim={rdm_dim}, Hilbert dim={len(basis)}")

# Fixed random split
rng_split = np.random.default_rng(0)
all_idx   = np.arange(n_total)
rng_split.shuffle(all_idx)
n_test    = int(n_total * TEST_FRAC)
test_idx  = all_idx[:n_test]
pool_idx  = all_idx[n_test:]

U_test  = torch.tensor(U_all[test_idx],  dtype=torch.float32, device=DEVICE)
E0_test = torch.tensor(E0_all[test_idx], dtype=torch.float32, device=DEVICE)
# gamma_GS for test (used only in upper-bound evaluation)
gamma_test_gs = torch.tensor(gamma_all[test_idx], dtype=torch.float32, device=DEVICE)

# Load model
model = QJEPA(rdm_dim=rdm_dim, latent_dim=LATENT_DIM, hidden=HIDDEN)
model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()


def get_iterated_z(gamma0_batch, U_batch, n_steps=ITER_STEPS):
    """One-shot denoiser: encode gamma0, apply denoiser → z_GS estimate."""
    with torch.no_grad():
        return model.denoise_to_gs(gamma0_batch, U_batch)


def sample_random_rdm_batch(U_batch_np, rng, n_fock=1):
    """For each U value, sample n_fock random Fock states and average their z."""
    gammas = []
    for _ in range(n_fock):
        batch_gammas = []
        for _ in U_batch_np:
            psi0 = random_fock_state(basis, rng)
            g = compute_1rdm_fast(psi0, ops, L).astype(np.float32)
            batch_gammas.append(g)
        gammas.append(np.stack(batch_gammas))  # (B, rdm, rdm)
    return np.mean(gammas, axis=0)  # average over Fock states → (B, rdm, rdm)


def make_head():
    return nn.Sequential(
        nn.Linear(LATENT_DIM, 128), nn.GELU(),
        nn.Linear(128, 64), nn.GELU(),
        nn.Linear(64, 1)
    ).to(DEVICE)


def train_head(z_tr, E0_tr, epochs=EPOCHS_HEAD):
    head = make_head()
    opt  = torch.optim.Adam(head.parameters(), lr=LR_HEAD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = len(z_tr)
    ds = TensorDataset(z_tr.detach(), E0_tr)
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, B), shuffle=True)
    for _ in range(epochs):
        for z_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(head(z_b).squeeze(-1), e_b).backward()
            opt.step()
        sched.step()
    return head


results = {"iterate": {}, "upperbound": {}, "direct_U": {}}

for n_labels in N_LABEL_LIST:
    iterate_maes = []
    upper_maes   = []
    direct_maes  = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 100)
        idx = rng.choice(pool_idx, size=n_labels, replace=False)

        U_tr_np  = U_all[idx]
        E0_tr_np = E0_all[idx]
        U_tr     = torch.tensor(U_tr_np,  dtype=torch.float32, device=DEVICE)
        E0_tr    = torch.tensor(E0_tr_np, dtype=torch.float32, device=DEVICE)

        # ── JEPA iterate: start from random gamma_0, iterate to z* ──────────
        gamma0_np = sample_random_rdm_batch(U_tr_np, rng, n_fock=5)
        gamma0_t  = torch.tensor(gamma0_np, dtype=torch.float32, device=DEVICE)
        z_tr_iter = get_iterated_z(gamma0_t, U_tr)
        head_iter = train_head(z_tr_iter, E0_tr)

        # Test: also start from random gamma_0
        test_U_np  = U_test.cpu().numpy()
        gamma0_te  = sample_random_rdm_batch(test_U_np, rng, n_fock=5)
        gamma0_te_t = torch.tensor(gamma0_te, dtype=torch.float32, device=DEVICE)
        z_te_iter = get_iterated_z(gamma0_te_t, U_test)
        with torch.no_grad():
            pred_iter = head_iter(z_te_iter).squeeze(-1)
        mae_iter = (pred_iter - E0_test).abs().mean().item()
        iterate_maes.append(mae_iter)

        # ── Upper bound: encode gamma_GS directly (oracle access) ──────────
        z_tr_gs = model.encoder(torch.tensor(gamma_all[idx], dtype=torch.float32, device=DEVICE), U_tr)
        head_gs  = train_head(z_tr_gs, E0_tr)
        z_te_gs  = model.encoder(gamma_test_gs, U_test)
        with torch.no_grad():
            pred_gs = head_gs(z_te_gs).squeeze(-1)
        mae_gs = (pred_gs - E0_test).abs().mean().item()
        upper_maes.append(mae_gs)

        # ── Direct U -> E0 (polynomial MLP, no 1-RDM) ───────────────────────
        U_tr_t = U_tr.unsqueeze(-1)   # (B, 1)
        head_u = nn.Sequential(nn.Linear(1, 64), nn.GELU(),
                               nn.Linear(64, 64), nn.GELU(),
                               nn.Linear(64, 1)).to(DEVICE)
        opt_u  = torch.optim.Adam(head_u.parameters(), lr=LR_HEAD)
        sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, T_max=EPOCHS_HEAD)
        ds_u = TensorDataset(U_tr_t, E0_tr)
        dl_u = DataLoader(ds_u, batch_size=min(BATCH_SIZE, n_labels), shuffle=True)
        for _ in range(EPOCHS_HEAD):
            for u_b, e_b in dl_u:
                opt_u.zero_grad()
                nn.functional.mse_loss(head_u(u_b).squeeze(-1), e_b).backward()
                opt_u.step()
            sched_u.step()
        with torch.no_grad():
            pred_u = head_u(U_test.unsqueeze(-1)).squeeze(-1)
        mae_u = (pred_u - E0_test).abs().mean().item()
        direct_maes.append(mae_u)

    results["iterate"][n_labels]    = iterate_maes
    results["upperbound"][n_labels] = upper_maes
    results["direct_U"][n_labels]   = direct_maes

    print(f"N={n_labels:4d} | "
          f"iterate={np.mean(iterate_maes):.4f}±{np.std(iterate_maes):.4f}  "
          f"upper(encode_gs)={np.mean(upper_maes):.4f}±{np.std(upper_maes):.4f}  "
          f"direct_U={np.mean(direct_maes):.4f}±{np.std(direct_maes):.4f}")

np.save(os.path.join(RESULTS_DIR, "iterate_results.npy"), results)
print(f"\nSaved -> {RESULTS_DIR}/iterate_results.npy")
