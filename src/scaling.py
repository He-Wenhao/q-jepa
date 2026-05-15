"""
Experiment 3: SSL data volume ablation.

Fix the fine-tuning/test set (labeled_gs.npz).
Vary the number of SSL Hamiltonians used for trajectory pretraining.
Show how fine-tuning MAE decreases as N_ssl grows.

N_ssl_list × N_labels_list grid, averaged over N_SEEDS seeds.
Saves results/scaling_results.npy
"""
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeltaPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN        = 256
N_LAYERS      = 4
PRETRAIN_EPOCHS  = 100
FINETUNE_EPOCHS  = 300
LR_PT         = 3e-4
LR_FT         = 1e-4
BATCH_SIZE    = 256
N_SSL_LIST    = [50, 100, 200, 500]
N_LABEL_LIST  = [5, 10, 20, 50]
N_SEEDS       = 3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TRAJ_PATH     = os.path.join(ROOT, "data", "trajectories.npz")
LABELED_PATH  = os.path.join(ROOT, "data", "labeled_gs.npz")
RESULTS_DIR   = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Pretrain on subset of trajectories ────────────────────────────────────────

def pretrain_subset(gamma_curr, gamma_next, h_vec,
                    rdm_dim, h_dim, h_mean, h_std, seed: int) -> DeltaPredictor:
    torch.manual_seed(seed)
    model = DeltaPredictor(rdm_dim=rdm_dim, h_dim=h_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    model.set_h_stats(h_mean, h_std)
    model = model.to(DEVICE)

    gc_t = torch.tensor(gamma_curr, device=DEVICE)
    gn_t = torch.tensor(gamma_next, device=DEVICE)
    hv_t = torch.tensor(h_vec,      device=DEVICE)
    ds   = TensorDataset(gc_t, gn_t, hv_t)
    dl   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR_PT, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    best_loss = float("inf")
    best_state = None
    for _ in range(PRETRAIN_EPOCHS):
        model.train()
        total = 0.0
        for gc_b, gn_b, hv_b in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(gc_b, hv_b), gn_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sch.step()
        avg = total / len(dl)
        if avg < best_loss:
            best_loss = avg
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def finetune_model(pretrained: DeltaPredictor,
                   gamma_0: torch.Tensor,
                   h_tr: torch.Tensor, gamma_tr: torch.Tensor,
                   seed: int) -> DeltaPredictor:
    torch.manual_seed(seed + 10000)
    model = copy.deepcopy(pretrained)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    ds    = TensorDataset(h_tr, gamma_tr)
    dl    = DataLoader(ds, batch_size=min(BATCH_SIZE, len(h_tr)), shuffle=True)
    model.train()
    for _ in range(FINETUNE_EPOCHS):
        for h_b, gstar_b in dl:
            B = len(h_b)
            g0_b = gamma_0.unsqueeze(0).expand(B, -1, -1)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(g0_b, h_b), gstar_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


def evaluate(model, gamma_0, h_te, gamma_te):
    model.eval()
    g0 = gamma_0.unsqueeze(0).expand(len(h_te), -1, -1)
    with torch.no_grad():
        pred = model(g0, h_te)
    return float((pred - gamma_te).abs().mean())


def main():
    print(f"Scaling ablation  device: {DEVICE}")

    # Load trajectory data once
    d_traj     = np.load(TRAJ_PATH)
    gc_all     = d_traj["gamma_curr"].astype(np.float32)
    gn_all     = d_traj["gamma_next"].astype(np.float32)
    hv_all     = d_traj["h_vec"].astype(np.float32)
    ham_id_all = d_traj["ham_id"]

    # Load labeled data
    d_lbl      = np.load(LABELED_PATH)
    h_vec_all  = d_lbl["h_vec"].astype(np.float32)
    gamma_gs   = d_lbl["gamma_gs"].astype(np.float32)
    is_test    = d_lbl["is_test"].astype(bool)
    gamma_0_np = d_lbl["gamma_0"].astype(np.float32)
    rdm_dim    = gamma_gs.shape[-1]
    h_dim      = h_vec_all.shape[-1]

    test_idx = np.where(is_test)[0]
    pool_idx = np.where(~is_test)[0]

    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vec_all[test_idx],  device=DEVICE)
    g_te    = torch.tensor(gamma_gs[test_idx],    device=DEVICE)

    # H stats from full training data
    h_mean = hv_all.mean(0)
    h_std  = hv_all.std(0)

    # results[n_ssl][n_labels] = list of MAE across seeds
    results = {n: {k: [] for k in N_LABEL_LIST} for n in N_SSL_LIST}

    n_ssl_max = max(N_SSL_LIST)
    # pairs_per_ham ≈ total_pairs / n_ham_ssl
    pairs_per_ham = len(gc_all) // (ham_id_all.max() + 1)

    for n_ssl in N_SSL_LIST:
        # Subset: keep pairs from the first n_ssl Hamiltonians
        mask    = ham_id_all < n_ssl
        gc_sub  = gc_all[mask]
        gn_sub  = gn_all[mask]
        hv_sub  = hv_all[mask]
        n_pairs = mask.sum()
        print(f"\nN_ssl={n_ssl:4d}  pairs={n_pairs:6,}")

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 7777)
            # Pretrain on subset
            model_pt = pretrain_subset(gc_sub, gn_sub, hv_sub,
                                        rdm_dim, h_dim, h_mean, h_std, seed)

            for n_labels in N_LABEL_LIST:
                idx = rng.choice(pool_idx, size=min(n_labels, len(pool_idx)), replace=False)
                h_tr = torch.tensor(h_vec_all[idx], device=DEVICE)
                g_tr = torch.tensor(gamma_gs[idx],  device=DEVICE)
                model_ft = finetune_model(model_pt, gamma_0, h_tr, g_tr, seed)
                mae = evaluate(model_ft, gamma_0, h_te, g_te)
                results[n_ssl][n_labels].append(mae)

        for n_labels in N_LABEL_LIST:
            vals = results[n_ssl][n_labels]
            print(f"  N_labels={n_labels:3d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out_path = os.path.join(RESULTS_DIR, "scaling_results.npy")
    np.save(out_path, results)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
