"""
Latent analysis for Q-JEPA:
1. PCA of z vs U/t
2. Linear probe: z → {off-diagonal γ, kinetic energy, E0}
3. Denoiser z* vs oracle z(γ_GS): quality of physical observables
4. Compare to HF baseline
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import QJEPA
from hubbard_ed import build_basis, make_basis_index, build_all_cdagger_c, compute_1rdm_fast, random_fock_state

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

L  = 6
t  = 1.0
d  = np.load("data/hubbard_gs.npz")
U_all     = d["U"]
E0_all    = d["E0"]
gamma_all = d["gamma_gs"]   # (250, 12, 12)
n         = len(U_all)

# Kinetic energy from 1-RDM: E_kin = -t * Σ_{<ij>,σ} (γ_{ij} + γ_{ji})
def kinetic_energy(gamma_batch):
    """gamma_batch: (N, 12, 12) -> E_kin (N,)"""
    E = np.zeros(len(gamma_batch))
    for sigma in range(2):
        g = gamma_batch[:, sigma*L:(sigma+1)*L, sigma*L:(sigma+1)*L]
        for i in range(L):
            j = (i+1) % L
            E += -t * (g[:, i, j] + g[:, j, i])
    return E

E_kin_all = kinetic_energy(gamma_all)

# Off-diagonal elements (vary with U)
def off_diagonal_flat(gamma_batch):
    N, D, _ = gamma_batch.shape
    mask = ~np.eye(D, dtype=bool)
    return gamma_batch[:, mask]   # (N, D²-D)

gamma_off = off_diagonal_flat(gamma_all)  # (250, 132)

# Load model
model = QJEPA(rdm_dim=12, latent_dim=64, hidden=256)
model.load_state_dict(torch.load("checkpoints/jepa_pretrained.pt", map_location="cpu"))
model.eval()

# Encode z(γ_GS) for all U values
U_t     = torch.tensor(U_all, dtype=torch.float32)
gamma_t = torch.tensor(gamma_all, dtype=torch.float32)
with torch.no_grad():
    z_gs = model.encoder(gamma_t).numpy()

# Denoiser z* from random γ₀ (5 Fock states, averaged)
basis     = build_basis(L, 3, 3)
basis_idx = make_basis_index(basis)
ops       = build_all_cdagger_c(L, basis, basis_idx)
rng       = np.random.default_rng(42)

z_star_list = []
for i in range(n):
    U_ti = torch.tensor([[U_all[i]]], dtype=torch.float32)
    zs   = []
    for _ in range(5):
        psi0 = random_fock_state(basis, rng)
        g0   = torch.tensor(compute_1rdm_fast(psi0, ops, L)[None], dtype=torch.float32)
        with torch.no_grad():
            zs.append(model.denoise_to_gs(g0, U_ti))
    z_star_list.append(torch.stack(zs).mean(0))

z_star = torch.cat(z_star_list).numpy()  # (250, 64)

# ── Linear probe: R² ─────────────────────────────────────────────────────────
def linear_r2_per_col(X, Y):
    """Mean R² of linear regression X → each column of Y."""
    from numpy.linalg import lstsq
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, _, _, _ = lstsq(A, Y, rcond=None)
    Y_pred = A @ coef
    ss_res = np.mean((Y - Y_pred)**2, axis=0)
    ss_tot = np.var(Y, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return r2.mean(), r2

targets = {
    "E₀":          E0_all.reshape(-1,1),
    "E_kin":        E_kin_all.reshape(-1,1),
    "off-diag γ":  gamma_off,
}

print("=== Linear R² from z → targets ===")
print(f"{'Target':20s}  oracle z(γ_GS)  denoiser z*")
for name, Y in targets.items():
    # Remove zero-variance columns
    mask = np.var(Y, axis=0) > 1e-10
    if not mask.any():
        continue
    Y = Y[:, mask]
    r2_gs,  _ = linear_r2_per_col(z_gs,   Y)
    r2_st,  _ = linear_r2_per_col(z_star, Y)
    print(f"  {name:18s}  {r2_gs:.4f}          {r2_st:.4f}")

# By U/t regime
print("\nBy regime (E₀):")
for name, mask in [("weak U<4", U_all<4), ("medium 4-8", (U_all>=4)&(U_all<8)), ("strong U≥8", U_all>=8)]:
    Y = E0_all[mask].reshape(-1,1)
    r_gs, _ = linear_r2_per_col(z_gs[mask],   Y)
    r_st, _ = linear_r2_per_col(z_star[mask], Y)
    print(f"  {name:16s}  oracle={r_gs:.4f}  denoiser={r_st:.4f}")

# ── Physical observable comparison ───────────────────────────────────────────
from numpy.linalg import lstsq

print("\n=== Physical observable prediction (linear probe, 5-fold CV) ===")
rng2 = np.random.default_rng(0)
idx  = np.arange(n); rng2.shuffle(idx)
fold_size = n // 5
for name, Y_full in [("E₀", E0_all), ("E_kin", E_kin_all)]:
    errs_gs = []; errs_st = []
    for f in range(5):
        te = idx[f*fold_size:(f+1)*fold_size]
        tr = np.concatenate([idx[:f*fold_size], idx[(f+1)*fold_size:]])
        for Z, errs in [(z_gs, errs_gs), (z_star, errs_st)]:
            A_tr = np.hstack([Z[tr], np.ones((len(tr),1))])
            coef, _, _, _ = lstsq(A_tr, Y_full[tr], rcond=None)
            A_te = np.hstack([Z[te], np.ones((len(te),1))])
            pred = A_te @ coef
            errs.append(np.abs(pred - Y_full[te]).mean())
    print(f"  {name}: oracle={np.mean(errs_gs):.4f}  denoiser={np.mean(errs_st):.4f}")

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# Panel (a): PCA colored by U/t
ax = axes[0]
z_c = z_gs - z_gs.mean(0)
_, _, Vt = np.linalg.svd(z_c, full_matrices=False)
z_2d = z_c @ Vt[:2].T
sc = ax.scatter(z_2d[:,0], z_2d[:,1], c=U_all, cmap='plasma', s=15)
plt.colorbar(sc, ax=ax, label='U/t', shrink=0.8)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("(a) z(γ_GS) PCA colored by U/t")

# Panel (b): E_kin vs U from oracle and denoiser
ax2 = axes[1]
A = np.hstack([z_gs,   np.ones((n,1))]); coef_gs, _, _, _ = lstsq(A, E_kin_all, rcond=None)
A = np.hstack([z_star, np.ones((n,1))]); coef_st, _, _, _ = lstsq(A, E_kin_all, rcond=None)

E_kin_pred_gs = np.hstack([z_gs,   np.ones((n,1))]) @ coef_gs
E_kin_pred_st = np.hstack([z_star, np.ones((n,1))]) @ coef_st

ax2.plot(U_all, E_kin_all,        "k-",  lw=2, label="Exact E_kin")
ax2.plot(U_all, E_kin_pred_gs, "g--", lw=1.5, label="Oracle z(γ_GS) probe", alpha=0.8)
ax2.plot(U_all, E_kin_pred_st, "b:",  lw=1.5, label="Denoiser z* probe", alpha=0.8)
ax2.set_xlabel("U/t"); ax2.set_ylabel("Kinetic energy")
ax2.set_title("(b) Kinetic energy: exact vs linear probe")
ax2.legend(fontsize=8)

# Panel (c): |z* - z_gs| vs U/t
ax3 = axes[2]
dists = np.linalg.norm(z_star - z_gs, axis=1)
norms = np.linalg.norm(z_gs, axis=1)
ax3.plot(U_all, dists / norms * 100, 'C0-', lw=2, label='Relative distance (%)')
ax3.set_xlabel("U/t"); ax3.set_ylabel("||z* - z(γ_GS)|| / ||z(γ_GS)|| (%)")
ax3.set_title("(c) Denoiser z* accuracy vs U/t")
ax3.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("results/fig5_latent_analysis.png", dpi=150, bbox_inches="tight")
plt.savefig("results/fig5_latent_analysis.pdf", bbox_inches="tight")
print("\nSaved fig5_latent_analysis.png")
