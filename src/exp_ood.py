"""
Figure 4 experiment: equal-budget OOD generalization.

Same as exp1.py but test set is U∈[6,10] (Mott insulator regime).
Pool and pretraining use U∈[0,6] only.

Saves results/exp_ood_results.npy
"""
import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeltaPredictor

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data", "ood_combined.npz")
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
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
    return model


def evaluate(model, gamma_0, h_te, g_te):
    model.eval()
    g0 = gamma_0.unsqueeze(0).expand(len(h_te), -1, -1)
    with torch.no_grad():
        pred = model(g0, h_te)
    return float((pred - g_te).abs().mean())


def main():
    print(f"OOD experiment: train U∈[0,6] → test U∈[6,10]  device={DEVICE}")
    d = np.load(DATA_PATH)
    h_vecs     = d["h_vecs"].astype(np.float32)
    gamma_gs   = d["gamma_gs"].astype(np.float32)
    gamma_traj = d["gamma_traj"].astype(np.float32)
    is_test    = d["is_test"].astype(bool)
    gamma_0_np = d["gamma_0"].astype(np.float32)

    rdm_dim  = gamma_gs.shape[-1]
    h_dim    = h_vecs.shape[-1]
    test_idx = np.where(is_test)[0]
    pool_idx = np.where(~is_test)[0]
    h_mean   = h_vecs[pool_idx].mean(0)
    h_std    = h_vecs[pool_idx].std(0)

    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vecs[test_idx],  device=DEVICE)
    g_te    = torch.tensor(gamma_gs[test_idx], device=DEVICE)

    print(f"  pool={len(pool_idx)} (U∈[0,6])  test={len(test_idx)} (U∈[6,10])")

    methods = ["endpoint", "trajectory"]
    results = {m: {n: [] for n in N_TRAIN_LIST} for m in methods}

    for n_train in N_TRAIN_LIST:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 31415)
            idx = rng.choice(pool_idx, size=min(n_train, len(pool_idx)), replace=False)
            h_tr = torch.tensor(h_vecs[idx],  device=DEVICE)
            g_tr = torch.tensor(gamma_gs[idx], device=DEVICE)

            # Endpoint-only
            m = make_model(rdm_dim, h_dim, h_mean, h_std)
            m = finetune(m, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            results["endpoint"][n_train].append(evaluate(m, gamma_0, h_te, g_te))

            # Trajectory SSL
            gc_list, gn_list, hv_list = [], [], []
            for i in idx:
                for traj in gamma_traj[i]:
                    for t in range(len(traj) - 1):
                        gc_list.append(traj[t]); gn_list.append(traj[t+1])
                        hv_list.append(h_vecs[i])
            gc = np.array(gc_list, np.float32)
            gn = np.array(gn_list, np.float32)
            hv = np.array(hv_list, np.float32)
            m = make_model(rdm_dim, h_dim, h_mean, h_std)
            m = pretrain(m, gc, gn, hv, PRETRAIN_EPOCHS)
            m = finetune(m, gamma_0, h_tr, g_tr, FINETUNE_EPOCHS)
            results["trajectory"][n_train].append(evaluate(m, gamma_0, h_te, g_te))

        ep = np.mean(results["endpoint"][n_train])
        tr = np.mean(results["trajectory"][n_train])
        print(f"N={n_train:4d}  endpoint={ep:.4f}  trajectory={tr:.4f}  ×{ep/tr:.2f}")

    np.save(os.path.join(RESULTS_DIR, "exp_ood_results.npy"), results)
    print("\nSaved results/exp_ood_results.npy")


if __name__ == "__main__":
    main()
