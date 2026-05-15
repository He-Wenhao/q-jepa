"""
Stage 2: Fine-tune f(γ_0, H) → γ_GS on N labeled (H, γ_GS) pairs.

Three-way comparison (all share the same architecture and fine-tuning loop):
  A  no_pretrain   Random initialization
  B  rand_pretrain Pretrained on random (mismatched) trajectory pairs
  C  traj_pretrain Pretrained on consecutive trajectory pairs  ← our method

Metric: MAE between predicted γ_GS and true γ_GS, averaged over test set.
Also reports implied E₀ error via E = Tr(h_1e @ γ) + U * D (if single-band).

Saves results to results/finetune_results.npy
"""
import os
import sys
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeltaPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN       = 256
N_LAYERS     = 4
EPOCHS_FT    = 300
LR_FT        = 1e-4
BATCH_SIZE   = 32
N_LABEL_LIST = [5, 10, 20, 50]
N_SEEDS      = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

LABELED_PATH = os.path.join(ROOT, "data", "labeled_gs.npz")
CKPT_DIR     = os.path.join(ROOT, "checkpoints")
RESULTS_DIR  = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_pretrained(mode: str, device: str,
                    traj_ckpt: str = "pretrain_traj",
                    rand_ckpt: str = "pretrain_rand") -> DeltaPredictor | None:
    if mode == "no_pretrain":
        return None
    stem = traj_ckpt if "traj" in mode else rand_ckpt
    path = os.path.join(CKPT_DIR, f"{stem}.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DeltaPredictor(rdm_dim=ckpt["rdm_dim"], h_dim=ckpt["h_dim"],
                           hidden=HIDDEN, n_layers=N_LAYERS)
    model.load_state_dict(ckpt["model"])
    model.set_h_stats(ckpt["h_mean"], ckpt["h_std"])
    return model.to(device)


def make_fresh_model(rdm_dim: int, h_dim: int, h_mean, h_std, device: str) -> DeltaPredictor:
    model = DeltaPredictor(rdm_dim=rdm_dim, h_dim=h_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    model.set_h_stats(h_mean, h_std)
    return model.to(device)


def finetune(model: DeltaPredictor, gamma_0: torch.Tensor,
             h_tr: torch.Tensor, gamma_tr: torch.Tensor,
             epochs: int) -> DeltaPredictor:
    model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Expand gamma_0 to match batch
    g0 = gamma_0.unsqueeze(0).expand(len(h_tr), -1, -1)
    ds = TensorDataset(h_tr, gamma_tr)
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, len(h_tr)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for h_b, gstar_b in dl:
            B = len(h_b)
            g0_b = gamma_0.unsqueeze(0).expand(B, -1, -1)
            opt.zero_grad()
            pred = model(g0_b, h_b)
            loss = nn.functional.mse_loss(pred, gstar_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


def evaluate(model: DeltaPredictor, gamma_0: torch.Tensor,
             h_te: torch.Tensor, gamma_te: torch.Tensor) -> float:
    model.eval()
    g0 = gamma_0.unsqueeze(0).expand(len(h_te), -1, -1)
    with torch.no_grad():
        pred = model(g0, h_te)
    return float((pred - gamma_te).abs().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int, default=N_SEEDS)
    parser.add_argument("--labeled_path", type=str, default=LABELED_PATH)
    parser.add_argument("--traj_ckpt",    type=str, default="pretrain_traj",
                        help="Stem of traj checkpoint (no .pt)")
    parser.add_argument("--rand_ckpt",    type=str, default="pretrain_rand",
                        help="Stem of rand checkpoint (no .pt)")
    parser.add_argument("--out_name",     type=str, default="finetune_results")
    args = parser.parse_args()

    print(f"Fine-tuning evaluation  device: {DEVICE}")
    d = np.load(args.labeled_path)
    h_vec_all   = d["h_vec"].astype(np.float32)
    gamma_gs_all = d["gamma_gs"].astype(np.float32)
    is_test      = d["is_test"].astype(bool)
    gamma_0_np   = d["gamma_0"].astype(np.float32)
    rdm_dim      = gamma_gs_all.shape[-1]
    h_dim        = h_vec_all.shape[-1]

    # H stats from pretraining data (load from traj pretrain checkpoint)
    traj_ckpt = torch.load(os.path.join(CKPT_DIR, f"{args.traj_ckpt}.pt"), map_location=DEVICE, weights_only=False)
    h_mean = traj_ckpt["h_mean"]
    h_std  = traj_ckpt["h_std"]

    test_idx = np.where(is_test)[0]
    pool_idx = np.where(~is_test)[0]

    gamma_0 = torch.tensor(gamma_0_np, device=DEVICE)
    h_te    = torch.tensor(h_vec_all[test_idx],    device=DEVICE)
    g_te    = torch.tensor(gamma_gs_all[test_idx], device=DEVICE)

    n_test = len(test_idx)
    n_pool = len(pool_idx)
    print(f"  rdm_dim={rdm_dim}  h_dim={h_dim}")
    print(f"  test={n_test}  pool={n_pool}")

    methods = ["no_pretrain", "rand_pretrain", "traj_pretrain"]
    results = {m: {n: [] for n in N_LABEL_LIST} for m in methods}

    # Pre-load pretrained models (once each)
    pretrained = {}
    for m in methods:
        pt = load_pretrained(m, DEVICE, traj_ckpt=args.traj_ckpt, rand_ckpt=args.rand_ckpt)
        if pt is None:
            pt = make_fresh_model(rdm_dim, h_dim, h_mean, h_std, DEVICE)
        pretrained[m] = pt

    for n_labels in N_LABEL_LIST:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed + 999)
            idx = rng.choice(pool_idx, size=min(n_labels, n_pool), replace=False)
            h_tr = torch.tensor(h_vec_all[idx],    device=DEVICE)
            g_tr = torch.tensor(gamma_gs_all[idx], device=DEVICE)

            for m in methods:
                model = finetune(pretrained[m], gamma_0, h_tr, g_tr, EPOCHS_FT)
                mae   = evaluate(model, gamma_0, h_te, g_te)
                results[m][n_labels].append(mae)

        print(f"\nN={n_labels}")
        for m in methods:
            vals = results[m][n_labels]
            print(f"  {m:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out_path = os.path.join(RESULTS_DIR, f"{args.out_name}.npy")
    np.save(out_path, results)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
