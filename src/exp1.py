"""
Experiment 1: Trajectory SSL vs endpoint-only supervised learning.

Same simulation budget N on the x-axis. Two curves:
  endpoint : train f(γ_0, H) → γ_GS on N labeled (H,γ_GS) from scratch
  trajectory: pretrain f(γ_t,H)→γ_{t+1} on N trajectories (cold start),
              then fine-tune f(γ_0,H)→γ_GS on same N endpoints

At every N, both methods have run exactly N expensive quantum simulations.
The trajectory method additionally uses the intermediate states.

Saves results/exp1_results.npy
"""
import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeltaPredictor

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data", "exp1_combined.npz")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN          = 256
N_LAYERS        = 4
PRETRAIN_EPOCHS = 200     # pretrain on trajectories (cold start per run)
FINETUNE_EPOCHS = 500
LR_PT           = 3e-4
LR_FT           = 1e-4
BATCH_PT        = 256
BATCH_FT        = 32
N_TRAIN_LIST    = [5, 10, 20, 50, 100, 200]
N_SEEDS         = 5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ── Training helpers ──────────────────────────────────────────────────────────

def make_model(rdm_dim, h_dim, h_mean, h_std):
    m = DeltaPredictor(rdm_dim=rdm_dim, h_dim=h_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    m.set_h_stats(h_mean, h_std)
    return m.to(DEVICE)


def pretrain(model, gc, gn, hv, epochs):
    """Train f(γ_t,H)→γ_{t+1} on trajectory pairs."""
    model.train()
    gc_t = torch.tensor(gc, device=DEVICE)
    gn_t = torch.tensor(gn, device=DEVICE)
    hv_t = torch.tensor(hv, device=DEVICE)
    ds   = TensorDataset(gc_t, gn_t, hv_t)
    dl   = DataLoader(ds, batch_size=min(BATCH_PT, len(gc)), shuffle=True, drop_last=False)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR_PT, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        total = 0.0
        for gc_b, gn_b, hv_b in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(gc_b, hv_b), gn_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sch.step()
        if total < best_loss:
            best_loss = total
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def finetune(model, gamma_0, h_tr, g_tr, epochs):
    """Fine-tune f(γ_0,H)→γ_GS on labeled pairs."""
    model = copy.deepcopy(model)
    model.train()
    ds  = TensorDataset(h_tr, g_tr)
    dl  = DataLoader(ds, batch_size=min(BATCH_FT, len(h_tr)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for h_b, g_b in dl:
            B    = len(h_b)
            g0_b = gamma_0.unsqueeze(0).expand(B, -1, -1)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(g0_b, h_b), g_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


def evaluate(model, gamma_0, h_te, g_te):
    model.eval()
    g0 = gamma_0.unsqueeze(0).expand(len(h_te), -1, -1)
    with torch.no_grad():
        pred = model(g0, h_te)
    return float((pred - g_te).abs().mean())


def traj_pairs_from_idx(gamma_traj, h_vecs, idx):
    """Extract consecutive (γ_t, γ_{t+1}, H) pairs from trajectory array."""
    gc_list, gn_list, hv_list = [], [], []
    for i in idx:
        trajs = gamma_traj[i]          # (N_TRAJ, T+1, rdm, rdm)
        h     = h_vecs[i]
        for traj in trajs:
            for t in range(len(traj) - 1):
                gc_list.append(traj[t])
                gn_list.append(traj[t + 1])
                hv_list.append(h)
    return (np.array(gc_list, dtype=np.float32),
            np.array(gn_list, dtype=np.float32),
            np.array(hv_list, dtype=np.float32))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Experiment 1: trajectory vs endpoint  device={DEVICE}")
    d = np.load(DATA_PATH)
    h_vecs     = d["h_vecs"].astype(np.float32)
    gamma_gs   = d["gamma_gs"].astype(np.float32)
    gamma_traj = d["gamma_traj"].astype(np.float32)
    is_test    = d["is_test"].astype(bool)
    gamma_0_np = d["gamma_0"].astype(np.float32)
    rdm_dim    = gamma_gs.shape[-1]
    h_dim      = h_vecs.shape[-1]

    test_idx = np.where(is_test)[0]
    pool_idx = np.where(~is_test)[0]

    # H normalization stats from full pool
    h_mean = h_vecs[pool_idx].mean(0)
    h_std  = h_vecs[pool_idx].std(0)

    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vecs[test_idx],  device=DEVICE)
    g_te    = torch.tensor(gamma_gs[test_idx], device=DEVICE)

    print(f"  rdm_dim={rdm_dim}  h_dim={h_dim}")
    print(f"  pool={len(pool_idx)}  test={len(test_idx)}")

    methods  = ["endpoint", "trajectory"]
    results  = {m: {n: [] for n in N_TRAIN_LIST} for m in methods}

    for n_train in N_TRAIN_LIST:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 31415)
            idx = rng.choice(pool_idx, size=min(n_train, len(pool_idx)), replace=False)

            h_tr = torch.tensor(h_vecs[idx],  device=DEVICE)
            g_tr = torch.tensor(gamma_gs[idx], device=DEVICE)

            # ── Endpoint-only ──────────────────────────────────────────────
            model_ep = make_model(rdm_dim, h_dim, h_mean, h_std)
            model_ep = finetune(model_ep, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            mae_ep   = evaluate(model_ep, gamma_0, h_te, g_te)
            results["endpoint"][n_train].append(mae_ep)

            # ── Trajectory pretrain + fine-tune ───────────────────────────
            gc, gn, hv = traj_pairs_from_idx(gamma_traj, h_vecs, idx)
            model_tr   = make_model(rdm_dim, h_dim, h_mean, h_std)
            model_tr   = pretrain(model_tr, gc, gn, hv, PRETRAIN_EPOCHS)
            model_tr   = finetune(model_tr, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            mae_tr     = evaluate(model_tr, gamma_0, h_te, g_te)
            results["trajectory"][n_train].append(mae_tr)

        ep_m  = np.mean(results["endpoint"][n_train])
        ep_s  = np.std(results["endpoint"][n_train])
        tr_m  = np.mean(results["trajectory"][n_train])
        tr_s  = np.std(results["trajectory"][n_train])
        print(f"N={n_train:4d}  endpoint={ep_m:.4f}±{ep_s:.4f}  "
              f"trajectory={tr_m:.4f}±{tr_s:.4f}")

    out = os.path.join(RESULTS_DIR, "exp1_results.npy")
    np.save(out, results)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
