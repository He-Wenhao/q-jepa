"""
Compute all data for Figure 4.

Panel A: Iterated prediction — iterate pretrained f^k(γ_0, H) and track MAE vs k.
         Compare to true imaginary-time trajectory convergence.
Panel B: Physical observables — predicted vs true E_kin, colored by U.
         Shows whether trajectory SSL captures Mott physics better.
Panel C: Fine-tuning convergence curves — validation MAE vs epoch for
         endpoint-only vs trajectory pretrain (N=20).

Saves:
  results/fig4a_iter.npy
  results/fig4b_obs.npy
  results/fig4c_conv.npy
"""
import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from model import DeltaPredictor

ROOT   = os.path.join(os.path.dirname(__file__), "..", "..")
CKPT   = os.path.join(ROOT, "checkpoints")
RES    = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN  = 256
N_LAYERS = 4

# ─────────────────────────────────────────────────────────────────────────────

def load_exp1():
    d = np.load(os.path.join(ROOT, "data", "exp1_combined.npz"))
    return (d["h_vecs"].astype(np.float32),
            d["gamma_gs"].astype(np.float32),
            d["gamma_traj"].astype(np.float32),   # (M, N_TRAJ, T+1, 12, 12)
            d["is_test"].astype(bool),
            d["gamma_0"].astype(np.float32))


def make_model(rdm_dim, h_dim, h_mean, h_std):
    m = DeltaPredictor(rdm_dim=rdm_dim, h_dim=h_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    m.set_h_stats(h_mean, h_std)
    return m.to(DEVICE)


def pretrain_on_pool(gc, gn, hv, rdm_dim, h_dim, h_mean, h_std,
                     epochs=150, seed=0):
    torch.manual_seed(seed)
    model = make_model(rdm_dim, h_dim, h_mean, h_std)
    ds = TensorDataset(torch.tensor(gc, device=DEVICE),
                       torch.tensor(gn, device=DEVICE),
                       torch.tensor(hv, device=DEVICE))
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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


def finetune_with_log(init_model, gamma_0_t, h_tr, g_tr, h_te, g_te,
                      epochs=500, log_every=5):
    model = copy.deepcopy(init_model)
    ds = TensorDataset(h_tr, g_tr)
    dl = DataLoader(ds, batch_size=min(32, len(h_tr)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for h_b, g_b in dl:
            B = len(h_b)
            g0_b = gamma_0_t.unsqueeze(0).expand(B, -1, -1)
            opt.zero_grad()
            nn.functional.mse_loss(model(g0_b, h_b), g_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if ep % log_every == 0 or ep == 1:
            model.eval()
            g0 = gamma_0_t.unsqueeze(0).expand(len(h_te), -1, -1)
            with torch.no_grad():
                mae = float((model(g0, h_te) - g_te).abs().mean())
            curve.append((ep, mae))
    return model, np.array(curve)


def compute_ekin(gamma_batch, h_vecs, L=6):
    """E_kin = -Σ_{<ij>,σ} t_ij (γ_ij + γ_ji). gamma_batch: (B,12,12)"""
    B = len(gamma_batch)
    ekin = np.zeros(B)
    t_bonds = h_vecs[:, :L]         # (B, L)
    for sigma in range(2):
        g = gamma_batch[:, sigma*L:(sigma+1)*L, sigma*L:(sigma+1)*L]
        for i in range(L):
            j = (i + 1) % L
            ekin += -t_bonds[:, i] * (g[:, i, j] + g[:, j, i])
    return ekin


# ── Panel A: Iterated Prediction ─────────────────────────────────────────────

def panel_a(h_vecs, gamma_gs, gamma_traj, test_idx, pool_idx, gamma_0_np,
            rdm_dim, h_dim, h_mean, h_std):
    print("Panel A: iterated prediction...")
    K_MAX = 80    # number of iterations
    N_TEST_USE = 50   # use first 50 test Hamiltonians for speed

    # Build trajectory pairs from pool for pretraining
    gc_list, gn_list, hv_list = [], [], []
    for i in pool_idx:
        for traj in gamma_traj[i]:          # (N_TRAJ, T+1, 12, 12)
            for t in range(len(traj) - 1):
                gc_list.append(traj[t]); gn_list.append(traj[t+1])
                hv_list.append(h_vecs[i])
    gc = np.array(gc_list, dtype=np.float32)
    gn = np.array(gn_list, dtype=np.float32)
    hv = np.array(hv_list, dtype=np.float32)

    model = pretrain_on_pool(gc, gn, hv, rdm_dim, h_dim, h_mean, h_std,
                              epochs=150, seed=0)
    model.eval()

    test_use = test_idx[:N_TEST_USE]
    gamma_0  = torch.tensor(gamma_0_np, device=DEVICE)
    h_te_t   = torch.tensor(h_vecs[test_use], device=DEVICE)
    g_te_t   = torch.tensor(gamma_gs[test_use], device=DEVICE)

    # Model iteration
    model_mae = []
    gamma_k = gamma_0.unsqueeze(0).expand(len(test_use), -1, -1).clone()
    with torch.no_grad():
        for k in range(K_MAX + 1):
            mae = float((gamma_k - g_te_t).abs().mean())
            model_mae.append(mae)
            if k < K_MAX:
                gamma_k = model(gamma_k, h_te_t)

    # True trajectory: MAE(gamma_traj[i,traj,t], gamma_gs[i]) vs t
    T_max = gamma_traj.shape[2]
    true_mae = []
    for t in range(T_max):
        frames = gamma_traj[test_use, :, t, :, :]   # (N_TEST_USE, N_TRAJ, 12, 12)
        gs     = gamma_gs[test_use][:, None, :, :]   # broadcast
        true_mae.append(float(np.abs(frames - gs).mean()))

    result = {"model_mae": np.array(model_mae),
              "true_mae":  np.array(true_mae),
              "k_axis":    np.arange(K_MAX + 1),
              "t_axis":    np.arange(T_max)}
    np.save(os.path.join(RES, "fig4a_iter.npy"), result)
    print(f"  Saved fig4a_iter.npy  (model final MAE={model_mae[-1]:.4f}, "
          f"true traj final MAE={true_mae[-1]:.4f})")


# ── Panel B: Physical Observables ─────────────────────────────────────────────

def panel_b(h_vecs, gamma_gs, test_idx, pool_idx, gamma_0_np,
            rdm_dim, h_dim, h_mean, h_std, N_TRAIN=20, N_SEEDS=5):
    print("Panel B: physical observables...")
    L = 6
    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vecs[test_idx], device=DEVICE)
    g_te    = torch.tensor(gamma_gs[test_idx], device=DEVICE)
    U_test  = h_vecs[test_idx, -1]

    # Build trajectory pairs from pool
    gc_list, gn_list, hv_list = [], [], []
    d_traj = np.load(os.path.join(ROOT, "data", "exp1_combined.npz"))
    gamma_traj = d_traj["gamma_traj"].astype(np.float32)
    for i in pool_idx:
        for traj in gamma_traj[i]:
            for t in range(len(traj) - 1):
                gc_list.append(traj[t]); gn_list.append(traj[t+1])
                hv_list.append(h_vecs[i])

    gc = np.array(gc_list, dtype=np.float32)
    gn = np.array(gn_list, dtype=np.float32)
    hv_traj = np.array(hv_list, dtype=np.float32)

    ep_preds, tr_preds = [], []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 42)
        idx = rng.choice(pool_idx, size=min(N_TRAIN, len(pool_idx)), replace=False)
        h_tr = torch.tensor(h_vecs[idx], device=DEVICE)
        g_tr = torch.tensor(gamma_gs[idx], device=DEVICE)

        # Endpoint
        ep_model = make_model(rdm_dim, h_dim, h_mean, h_std)
        ep_model, _ = finetune_with_log(ep_model, gamma_0, h_tr, g_tr,
                                         h_te, g_te, epochs=500, log_every=500)
        ep_model.eval()
        g0 = gamma_0.unsqueeze(0).expand(len(test_idx), -1, -1)
        with torch.no_grad():
            ep_preds.append(ep_model(g0, h_te).cpu().numpy())

        # Trajectory
        pt_model = pretrain_on_pool(gc, gn, hv_traj, rdm_dim, h_dim,
                                     h_mean, h_std, epochs=100, seed=seed)
        tr_model, _ = finetune_with_log(pt_model, gamma_0, h_tr, g_tr,
                                         h_te, g_te, epochs=500, log_every=500)
        tr_model.eval()
        with torch.no_grad():
            tr_preds.append(tr_model(g0, h_te).cpu().numpy())

    # Average predicted gamma over seeds
    ep_gamma = np.mean(ep_preds, axis=0)   # (N_test, 12, 12)
    tr_gamma = np.mean(tr_preds, axis=0)

    true_gamma = gamma_gs[test_idx]
    true_ekin  = compute_ekin(true_gamma, h_vecs[test_idx])
    ep_ekin    = compute_ekin(ep_gamma,   h_vecs[test_idx])
    tr_ekin    = compute_ekin(tr_gamma,   h_vecs[test_idx])

    result = {"U": U_test, "true_ekin": true_ekin,
              "ep_ekin": ep_ekin, "tr_ekin": tr_ekin}
    np.save(os.path.join(RES, "fig4b_obs.npy"), result)
    print(f"  Saved fig4b_obs.npy  E_kin MAE: "
          f"endpoint={np.abs(ep_ekin-true_ekin).mean():.4f}  "
          f"traj={np.abs(tr_ekin-true_ekin).mean():.4f}")


# ── Panel C: Convergence Curves ───────────────────────────────────────────────

def panel_c(h_vecs, gamma_gs, gamma_traj, test_idx, pool_idx, gamma_0_np,
            rdm_dim, h_dim, h_mean, h_std, N_TRAIN=20, N_SEEDS=3):
    """Equal-budget convergence curves: both methods use the same N_TRAIN Hamiltonians."""
    print("Panel C: convergence curves (equal budget)...")
    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vecs[test_idx], device=DEVICE)
    g_te    = torch.tensor(gamma_gs[test_idx], device=DEVICE)

    EPOCHS = 500; LOG_EVERY = 5
    ep_curves, tr_curves = [], []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 99)
        idx = rng.choice(pool_idx, size=min(N_TRAIN, len(pool_idx)), replace=False)
        h_tr = torch.tensor(h_vecs[idx], device=DEVICE)
        g_tr = torch.tensor(gamma_gs[idx], device=DEVICE)

        # Endpoint: fine-tune from random init
        ep_model = make_model(rdm_dim, h_dim, h_mean, h_std)
        _, ep_curve = finetune_with_log(ep_model, gamma_0, h_tr, g_tr,
                                         h_te, g_te, EPOCHS, LOG_EVERY)
        ep_curves.append(ep_curve[:, 1])

        # Trajectory: pretrain on same N_TRAIN Hamiltonians' trajectories, then fine-tune
        gc_list, gn_list, hv_list = [], [], []
        for i in idx:
            for traj in gamma_traj[i]:
                for t in range(len(traj) - 1):
                    gc_list.append(traj[t]); gn_list.append(traj[t+1])
                    hv_list.append(h_vecs[i])
        gc = np.array(gc_list, dtype=np.float32)
        gn = np.array(gn_list, dtype=np.float32)
        hv_traj_seed = np.array(hv_list, dtype=np.float32)

        pt_model = pretrain_on_pool(gc, gn, hv_traj_seed, rdm_dim, h_dim,
                                     h_mean, h_std, epochs=200, seed=seed)
        _, tr_curve = finetune_with_log(pt_model, gamma_0, h_tr, g_tr,
                                         h_te, g_te, EPOCHS, LOG_EVERY)
        tr_curves.append(tr_curve[:, 1])

    epochs_logged = np.array([ep_curve[0] for ep_curve in
                               [finetune_with_log.__wrapped__ if hasattr(
                                finetune_with_log, '__wrapped__') else None]
                               ]) if False else \
                   np.arange(LOG_EVERY, EPOCHS + 1, LOG_EVERY)
    # Correct epoch axis
    n_pts = len(ep_curves[0])
    ep_axis = np.linspace(LOG_EVERY, EPOCHS, n_pts)

    result = {"epochs": ep_axis,
              "ep_mean": np.mean(ep_curves, axis=0),
              "ep_std":  np.std(ep_curves, axis=0),
              "tr_mean": np.mean(tr_curves, axis=0),
              "tr_std":  np.std(tr_curves, axis=0)}
    np.save(os.path.join(RES, "fig4c_conv.npy"), result)
    print(f"  Saved fig4c_conv.npy  final MAE: "
          f"endpoint={np.mean(ep_curves, axis=0)[-1]:.4f}  "
          f"traj={np.mean(tr_curves, axis=0)[-1]:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Computing Figure 4  device={DEVICE}")
    h_vecs, gamma_gs, gamma_traj, is_test, gamma_0 = load_exp1()
    test_idx = np.where(is_test)[0]
    pool_idx = np.where(~is_test)[0]
    rdm_dim  = gamma_gs.shape[-1]
    h_dim    = h_vecs.shape[-1]
    h_mean   = h_vecs[pool_idx].mean(0)
    h_std    = h_vecs[pool_idx].std(0)

    print(f"  pool={len(pool_idx)}  test={len(test_idx)}  "
          f"rdm_dim={rdm_dim}  h_dim={h_dim}")

    panel_c(h_vecs, gamma_gs, gamma_traj, test_idx, pool_idx, gamma_0,
            rdm_dim, h_dim, h_mean, h_std)
    print("\nAll done.")


if __name__ == "__main__":
    main()
