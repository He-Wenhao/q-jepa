"""
Stage 1: SSL pretraining on trajectory pairs.

Modes:
  --mode traj    (C) Train f(γ_t, H) → γ_{t+1} on consecutive trajectory pairs.
  --mode rand    (B) Train on random pairs: same γ_t and H_vec, but the target
                     γ_next is drawn from a DIFFERENT Hamiltonian's trajectory.
                     Same data volume, no temporal/physical coherence.

Saves checkpoint to checkpoints/pretrain_{mode}.pt
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeltaPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN      = 256
N_LAYERS    = 4
EPOCHS      = 100
LR          = 3e-4
BATCH_SIZE  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TRAJ_PATH   = os.path.join(ROOT, "data", "trajectories.npz")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def load_data(mode: str, rng: np.random.Generator, traj_path: str = TRAJ_PATH):
    d = np.load(traj_path)
    gamma_curr = d["gamma_curr"].astype(np.float32)  # (N, rdm, rdm)
    gamma_next = d["gamma_next"].astype(np.float32)
    h_vec      = d["h_vec"].astype(np.float32)
    ham_id     = d["ham_id"]                          # (N,)

    if mode == "rand":
        # Shuffle targets across different Hamiltonians.
        # For each pair, sample a random target from a DIFFERENT Hamiltonian.
        n = len(gamma_curr)
        shuffled_next = np.empty_like(gamma_next)
        for i in range(n):
            hid = ham_id[i]
            # Draw a random pair from a different Hamiltonian
            candidates = np.where(ham_id != hid)[0]
            j = rng.choice(candidates)
            shuffled_next[i] = gamma_next[j]
        gamma_next = shuffled_next

    return gamma_curr, gamma_next, h_vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["traj", "rand"], default="traj")
    parser.add_argument("--epochs",    type=int, default=EPOCHS)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--traj_path", type=str, default=TRAJ_PATH)
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="Checkpoint filename stem (default: pretrain_{mode})")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ckpt_stem = args.ckpt_name or f"pretrain_{args.mode}"
    print(f"Pretraining mode: {args.mode}  device: {DEVICE}  ckpt: {ckpt_stem}")
    gamma_curr, gamma_next, h_vec = load_data(args.mode, rng, traj_path=args.traj_path)
    n, rdm_dim, _ = gamma_curr.shape
    h_dim = h_vec.shape[-1]
    print(f"  Pairs: {n:,}  rdm_dim: {rdm_dim}  h_dim: {h_dim}")

    # Compute H normalization stats from training data
    h_mean = h_vec.mean(0)
    h_std  = h_vec.std(0)

    model = DeltaPredictor(rdm_dim=rdm_dim, h_dim=h_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    model.set_h_stats(h_mean, h_std)
    model = model.to(DEVICE)

    gc_t  = torch.tensor(gamma_curr, device=DEVICE)
    gn_t  = torch.tensor(gamma_next, device=DEVICE)
    hv_t  = torch.tensor(h_vec,      device=DEVICE)

    ds = TensorDataset(gc_t, gn_t, hv_t)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, f"{ckpt_stem}.pt")
    _save = lambda: torch.save({"model": model.state_dict(),
                                "h_mean": h_mean.tolist(), "h_std": h_std.tolist(),
                                "rdm_dim": rdm_dim, "h_dim": h_dim,
                                "mode": args.mode}, ckpt_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for gc_b, gn_b, hv_b in dl:
            opt.zero_grad()
            pred = model(gc_b, hv_b)
            loss = nn.functional.mse_loss(pred, gn_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sch.step()

        avg = total_loss / len(dl)
        if avg < best_loss:
            best_loss = avg
            _save()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  loss={avg:.6f}  best={best_loss:.6f}")

    print(f"\nSaved -> {ckpt_path}  (best_loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
