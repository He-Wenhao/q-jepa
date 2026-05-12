"""
Downstream evaluation for Q-JEPA.

Setup:
  - Random 80/20 train/test split
  - N_labels subsampled from training set
  - Report MAE overall + by U/t regime (weak / medium / strong correlation)

Methods:
  A) finetune  -- pretrained encoder (low LR) + head, end-to-end
  B) scratch   -- same architecture, random init, end-to-end
  C) direct    -- raw 1-RDM (flattened) -> MLP, no encoder pretraining
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import QJEPA, RDMEncoder

LATENT_DIM    = 64
HIDDEN        = 256
LR_HEAD       = 1e-3
LR_ENC_FT     = 5e-5      # fine-tuning LR for pretrained encoder
EPOCHS        = 500
BATCH_SIZE    = 32
N_LABEL_LIST  = [20, 50, 100, 200]
N_SEEDS       = 5
TEST_FRAC     = 0.2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN_PATH = "checkpoints/jepa_pretrained.pt"
GS_PATH       = "data/hubbard_gs.npz"
RESULTS_DIR   = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

d = np.load(GS_PATH)
U_all     = torch.tensor(d["U"],        dtype=torch.float32)
E0_all    = torch.tensor(d["E0"],       dtype=torch.float32)
gamma_all = torch.tensor(d["gamma_gs"], dtype=torch.float32)
n_total   = len(U_all)
rdm_dim   = gamma_all.shape[-1]
print(f"Dataset: N={n_total}, rdm_dim={rdm_dim}, device={DEVICE}")

# Fixed random test/train split (same across all experiments)
rng_split = np.random.default_rng(0)
all_idx   = np.arange(n_total)
rng_split.shuffle(all_idx)
n_test    = int(n_total * TEST_FRAC)
test_idx  = all_idx[:n_test]
pool_idx  = all_idx[n_test:]

gamma_test = gamma_all[test_idx].to(DEVICE)
U_test     = U_all[test_idx].to(DEVICE)
E0_test    = E0_all[test_idx].to(DEVICE)
U_test_np  = U_all[test_idx].numpy()

# U/t regime masks for test set
mask_weak   = U_test_np < 4
mask_medium = (U_test_np >= 4) & (U_test_np < 8)
mask_strong = U_test_np >= 8


def _make_head():
    return nn.Sequential(nn.Linear(LATENT_DIM, 64), nn.GELU(), nn.Linear(64, 1)).to(DEVICE)


def _make_direct_head(in_dim):
    return nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(),
                         nn.Linear(128, 64), nn.GELU(),
                         nn.Linear(64, 1)).to(DEVICE)


def _report_mae(pred, true, mask=None):
    if mask is not None:
        pred, true = pred[mask.astype(bool)], true[mask.astype(bool)]
    if len(pred) == 0:
        return float("nan")
    return float(np.abs(pred - true).mean())


def run_e2e(enc_state_or_none, n_labels, seed):
    """Train encoder + head end-to-end. enc_state_or_none: pretrained or None (scratch)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool_idx, size=n_labels, replace=False)
    gamma_tr = gamma_all[idx].to(DEVICE)
    U_tr     = U_all[idx].to(DEVICE)
    E0_tr    = E0_all[idx].to(DEVICE)

    enc  = RDMEncoder(rdm_dim, LATENT_DIM, HIDDEN).to(DEVICE)
    if enc_state_or_none is not None:
        enc.load_state_dict(enc_state_or_none)
        lr_enc = LR_ENC_FT
    else:
        lr_enc = LR_HEAD

    head = _make_head()
    opt  = torch.optim.Adam([
        {"params": enc.parameters(),  "lr": lr_enc},
        {"params": head.parameters(), "lr": LR_HEAD},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    ds = TensorDataset(gamma_tr, U_tr, E0_tr)
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, n_labels), shuffle=True)

    for _ in range(EPOCHS):
        for g_b, u_b, e_b in dl:
            opt.zero_grad()
            pred = head(enc(g_b, u_b)).squeeze(-1)
            nn.functional.mse_loss(pred, e_b).backward()
            opt.step()
        sched.step()

    enc.eval(); head.eval()
    with torch.no_grad():
        pred_te = head(enc(gamma_test, U_test)).squeeze(-1)
    return pred_te.cpu()


def run_direct(n_labels, seed):
    """Direct regression: flattened 1-RDM + U -> E0."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool_idx, size=n_labels, replace=False)
    gamma_tr = gamma_all[idx].to(DEVICE)
    U_tr     = U_all[idx].to(DEVICE)
    E0_tr    = E0_all[idx].to(DEVICE)

    in_dim = rdm_dim * rdm_dim + 1
    head   = _make_direct_head(in_dim)
    opt    = torch.optim.Adam(head.parameters(), lr=LR_HEAD)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    B = gamma_tr.shape[0]
    X_tr = torch.cat([gamma_tr.reshape(B, -1), U_tr.reshape(B, 1)], dim=-1)
    Bt = gamma_test.shape[0]
    X_te = torch.cat([gamma_test.reshape(Bt, -1), U_test.reshape(Bt, 1)], dim=-1)

    ds = TensorDataset(X_tr, E0_tr)
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, n_labels), shuffle=True)

    for _ in range(EPOCHS):
        for x_b, e_b in dl:
            opt.zero_grad()
            nn.functional.mse_loss(head(x_b).squeeze(-1), e_b).backward()
            opt.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        pred_te = head(X_te).squeeze(-1)
    return pred_te.cpu()


# Load pretrained encoder weights
full_state = torch.load(PRETRAIN_PATH, map_location=DEVICE)
pretrained_enc_state = {k[len("encoder."):]: v
                        for k, v in full_state.items() if k.startswith("encoder.")}

E0_test_np = E0_test.cpu().numpy()

results = {m: {n: {"all": [], "weak": [], "medium": [], "strong": []}
               for n in N_LABEL_LIST}
           for m in ["finetune", "scratch", "direct"]}

for n_labels in N_LABEL_LIST:
    for seed in range(N_SEEDS):
        for method, fn_args in [
            ("finetune", (pretrained_enc_state, n_labels, seed)),
            ("scratch",  (None, n_labels, seed)),
        ]:
            pred = run_e2e(*fn_args).numpy()
            for key, mask in [("all", None), ("weak", mask_weak),
                               ("medium", mask_medium), ("strong", mask_strong)]:
                results[method][n_labels][key].append(_report_mae(pred, E0_test_np, mask))

        pred_d = run_direct(n_labels, seed).numpy()
        for key, mask in [("all", None), ("weak", mask_weak),
                          ("medium", mask_medium), ("strong", mask_strong)]:
            results["direct"][n_labels][key].append(_report_mae(pred_d, E0_test_np, mask))

    ft_all = np.mean(results["finetune"][n_labels]["all"])
    sc_all = np.mean(results["scratch"][n_labels]["all"])
    di_all = np.mean(results["direct"][n_labels]["all"])
    ft_str = np.mean(results["finetune"][n_labels]["strong"])
    sc_str = np.mean(results["scratch"][n_labels]["strong"])
    di_str = np.mean(results["direct"][n_labels]["strong"])
    print(f"N={n_labels:4d} | finetune {ft_all:.4f} ({ft_str:.4f}@strong)"
          f"  scratch {sc_all:.4f} ({sc_str:.4f}@strong)"
          f"  direct {di_all:.4f} ({di_str:.4f}@strong)")

np.save(os.path.join(RESULTS_DIR, "downstream_results.npy"), results)
print(f"\nSaved -> {RESULTS_DIR}/downstream_results.npy")
