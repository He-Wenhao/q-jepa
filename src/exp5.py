"""
Figure 5 experiment: Does trajectory structure matter?

Four methods at equal simulation budget N:
  endpoint   : f(γ_0, H) → γ_GS from scratch (N labeled pairs)
  imag_time  : pretrain on ordered imaginary-time pairs, fine-tune
  shuffled   : pretrain on same γ frames but randomly paired (breaks time order)
  power_iter : pretrain on power-iteration trajectories, fine-tune

Saves results/exp5_results.npy
"""
import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeltaPredictor

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_EXP1   = os.path.join(ROOT, "data", "exp1_combined.npz")
DATA_FIG5   = os.path.join(ROOT, "data", "fig5_power.npz")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HIDDEN          = 256
N_LAYERS        = 4
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 500
LR_PT           = 3e-4
LR_FT           = 1e-4
BATCH_PT        = 256
BATCH_FT        = 32
N_TRAIN_LIST    = [5, 10, 20, 50, 100, 200]
N_SEEDS         = 5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(rdm_dim, h_dim, h_mean, h_std):
    m = DeltaPredictor(rdm_dim=rdm_dim, h_dim=h_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    m.set_h_stats(h_mean, h_std)
    return m.to(DEVICE)


def pretrain(model, gc, gn, hv, epochs):
    model.train()
    ds  = TensorDataset(torch.tensor(gc, device=DEVICE),
                        torch.tensor(gn, device=DEVICE),
                        torch.tensor(hv, device=DEVICE))
    dl  = DataLoader(ds, batch_size=min(BATCH_PT, len(gc)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PT, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        total = 0.0
        for gc_b, gn_b, hv_b in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(gc_b, hv_b), gn_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        sch.step()
        if total < best:
            best = total; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def finetune(model, gamma_0, h_tr, g_tr, epochs):
    model = copy.deepcopy(model)
    model.train()
    ds  = TensorDataset(h_tr, g_tr)
    dl  = DataLoader(ds, batch_size=min(BATCH_FT, len(h_tr)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for h_b, g_b in dl:
            B = len(h_b)
            g0_b = gamma_0.unsqueeze(0).expand(B, -1, -1)
            opt.zero_grad()
            nn.functional.mse_loss(model(g0_b, h_b), g_b).backward()
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


def ordered_pairs(gamma_traj, h_vecs, idx):
    """Consecutive (γ_t, γ_{t+1}, H) pairs — temporal order preserved."""
    gc, gn, hv = [], [], []
    for i in idx:
        for traj in gamma_traj[i]:          # (N_TRAJ, T+1, 12, 12)
            for t in range(len(traj) - 1):
                gc.append(traj[t]); gn.append(traj[t+1]); hv.append(h_vecs[i])
    return (np.array(gc, np.float32), np.array(gn, np.float32),
            np.array(hv, np.float32))


def shuffled_pairs(gamma_traj, h_vecs, idx, rng):
    """Same γ frames but randomly paired — breaks temporal order.
    For each Hamiltonian, collect all frames then pair randomly.
    Input (γ, H) is consistent, but target γ' is a random other frame."""
    gc, gn, hv = [], [], []
    for i in idx:
        frames = gamma_traj[i].reshape(-1, gamma_traj.shape[-2], gamma_traj.shape[-1])
        n = len(frames)
        perm = rng.permutation(n)
        for t in range(n):
            gc.append(frames[t])
            gn.append(frames[perm[t]])     # random other frame, same H
            hv.append(h_vecs[i])
    return (np.array(gc, np.float32), np.array(gn, np.float32),
            np.array(hv, np.float32))


def power_pairs(gamma_traj_power, h_vecs, idx):
    """Consecutive pairs from power-iteration trajectories."""
    gc, gn, hv = [], [], []
    for i in idx:
        traj = gamma_traj_power[i]          # (T+1, 12, 12)
        for t in range(len(traj) - 1):
            gc.append(traj[t]); gn.append(traj[t+1]); hv.append(h_vecs[i])
    return (np.array(gc, np.float32), np.array(gn, np.float32),
            np.array(hv, np.float32))


def main():
    print(f"Experiment 5: trajectory structure  device={DEVICE}")
    d1 = np.load(DATA_EXP1)
    d5 = np.load(DATA_FIG5)

    h_vecs          = d1["h_vecs"].astype(np.float32)
    gamma_gs        = d1["gamma_gs"].astype(np.float32)
    gamma_traj      = d1["gamma_traj"].astype(np.float32)
    gamma_traj_pw   = d5["gamma_traj_power"].astype(np.float32)
    is_test         = d1["is_test"].astype(bool)
    gamma_0_np      = d1["gamma_0"].astype(np.float32)

    rdm_dim = gamma_gs.shape[-1]
    h_dim   = h_vecs.shape[-1]

    test_idx = np.where(is_test)[0]
    pool_idx = np.where(~is_test)[0]
    h_mean   = h_vecs[pool_idx].mean(0)
    h_std    = h_vecs[pool_idx].std(0)

    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vecs[test_idx],  device=DEVICE)
    g_te    = torch.tensor(gamma_gs[test_idx], device=DEVICE)

    print(f"  rdm_dim={rdm_dim}  h_dim={h_dim}  pool={len(pool_idx)}  test={len(test_idx)}")

    methods = ["endpoint", "imag_time", "shuffled", "power_iter"]
    results = {m: {n: [] for n in N_TRAIN_LIST} for m in methods}

    for n_train in N_TRAIN_LIST:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 31415)
            idx = rng.choice(pool_idx, size=min(n_train, len(pool_idx)), replace=False)
            h_tr = torch.tensor(h_vecs[idx],  device=DEVICE)
            g_tr = torch.tensor(gamma_gs[idx], device=DEVICE)

            # 1. Endpoint-only
            m = make_model(rdm_dim, h_dim, h_mean, h_std)
            m = finetune(m, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            results["endpoint"][n_train].append(evaluate(m, gamma_0, h_te, g_te))

            # 2. Imaginary-time SSL (ordered)
            gc, gn, hv = ordered_pairs(gamma_traj, h_vecs, idx)
            m = make_model(rdm_dim, h_dim, h_mean, h_std)
            m = pretrain(m, gc, gn, hv, PRETRAIN_EPOCHS)
            m = finetune(m, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            results["imag_time"][n_train].append(evaluate(m, gamma_0, h_te, g_te))

            # 3. Shuffled SSL (same frames, random pairing)
            gc, gn, hv = shuffled_pairs(gamma_traj, h_vecs, idx, rng)
            m = make_model(rdm_dim, h_dim, h_mean, h_std)
            m = pretrain(m, gc, gn, hv, PRETRAIN_EPOCHS)
            m = finetune(m, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            results["shuffled"][n_train].append(evaluate(m, gamma_0, h_te, g_te))

            # 4. Power iteration SSL
            gc, gn, hv = power_pairs(gamma_traj_pw, h_vecs, idx)
            m = make_model(rdm_dim, h_dim, h_mean, h_std)
            m = pretrain(m, gc, gn, hv, PRETRAIN_EPOCHS)
            m = finetune(m, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            results["power_iter"][n_train].append(evaluate(m, gamma_0, h_te, g_te))

        ep = np.mean(results["endpoint"][n_train])
        it = np.mean(results["imag_time"][n_train])
        sh = np.mean(results["shuffled"][n_train])
        pw = np.mean(results["power_iter"][n_train])
        print(f"N={n_train:4d}  ep={ep:.4f}  imag={it:.4f}  shuf={sh:.4f}  pow={pw:.4f}")

    np.save(os.path.join(RESULTS_DIR, "exp5_results.npy"), results)
    print(f"\nSaved results/exp5_results.npy")


if __name__ == "__main__":
    main()
