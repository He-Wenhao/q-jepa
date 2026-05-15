"""
Figure 4: Latent space analysis.

Examines what information is encoded in the Q-JEPA latent z:
  (a) PCA of z(γ_GS) colored by U/t — shows smooth U-ordering
  (b) Linear probe: predicted vs true E_kin across U
  (c) Relative distance ||z*(denoiser) - z(γ_GS)|| / ||z(γ_GS)||  vs U/t
      — quantifies denoiser z* accuracy

Also prints R² and 5-fold CV MAE for linear probes on E₀, E_kin, off-diag γ.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import QJEPA
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        compute_1rdm_fast, random_fock_state)

L  = 6
d  = np.load(os.path.join(ROOT, "data", "hubbard_gs.npz"))
U_all     = d["U"].astype(np.float32)
E0_all    = d["E0"].astype(np.float32)
gamma_all = d["gamma_gs"].astype(np.float32)
n         = len(U_all)

def kinetic_energy(gamma_batch):
    E = np.zeros(len(gamma_batch))
    for sigma in range(2):
        g = gamma_batch[:, sigma*L:(sigma+1)*L, sigma*L:(sigma+1)*L]
        for i in range(L):
            j = (i+1) % L
            E += -(g[:, i, j] + g[:, j, i])
    return E

E_kin_all = kinetic_energy(gamma_all)

model = QJEPA(rdm_dim=12, latent_dim=64, hidden=256)
model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "jepa_pretrained.pt"),
                                  map_location="cpu"))
model.eval()

gamma_t = torch.tensor(gamma_all, dtype=torch.float32)
with torch.no_grad():
    z_gs = model.encoder(gamma_t).numpy()   # (n, 64)

# Denoiser z* — average 5 Fock states per sample
basis     = build_basis(L, 3, 3)
basis_idx = make_basis_index(basis)
ops       = build_all_cdagger_c(L, basis, basis_idx)
rng       = np.random.default_rng(42)
z_star    = np.zeros_like(z_gs)
for i in range(n):
    H_t = torch.tensor([[U_all[i]]], dtype=torch.float32)
    zs  = []
    for _ in range(5):
        psi0 = random_fock_state(basis, rng)
        g0   = torch.tensor(compute_1rdm_fast(psi0, ops, L)[None], dtype=torch.float32)
        with torch.no_grad():
            zs.append(model.denoise_to_gs(g0, H_t).numpy())
    z_star[i] = np.mean(zs, axis=0)

# ── Linear probe R² ────────────────────────────────────────────────────────────
def r2_mean(X, Y):
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, _, _, _ = lstsq(A, Y, rcond=None)
    Y_pred = A @ coef
    ss_res = np.mean((Y - Y_pred)**2, axis=0)
    ss_tot = np.var(Y, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return r2.mean()

off_diag_mask = ~np.eye(12, dtype=bool)
targets = {
    "E₀":         E0_all.reshape(-1, 1),
    "E_kin":      E_kin_all.reshape(-1, 1),
    "off-diag γ": gamma_all.reshape(n, 12, 12)[:, off_diag_mask],
}
print("=== Linear probe R² ===")
print(f"  {'Target':15s}  oracle z(γ_GS)  denoiser z*")
for name, Y in targets.items():
    mask = np.var(Y, axis=0) > 1e-10
    Y_ = Y[:, mask] if Y.ndim > 1 else Y
    print(f"  {name:15s}  {r2_mean(z_gs, Y_):.4f}          {r2_mean(z_star, Y_):.4f}")

# 5-fold CV MAE
idx_all = np.arange(n); fold = n // 5
rng2 = np.random.default_rng(0); rng2.shuffle(idx_all)
print("\n=== 5-fold CV MAE (linear probe → E₀) ===")
for label, Z in [("oracle", z_gs), ("denoiser", z_star)]:
    errs = []
    for f in range(5):
        te = idx_all[f*fold:(f+1)*fold]
        tr = np.concatenate([idx_all[:f*fold], idx_all[(f+1)*fold:]])
        A_tr = np.hstack([Z[tr], np.ones((len(tr),1))])
        coef, _, _, _ = lstsq(A_tr, E0_all[tr], rcond=None)
        pred = np.hstack([Z[te], np.ones((len(te),1))]) @ coef
        errs.append(np.abs(pred - E0_all[te]).mean())
    print(f"  {label:10s}  MAE={np.mean(errs):.4f}±{np.std(errs):.4f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): PCA of z_gs colored by U
ax = axes[0]
z_c  = z_gs - z_gs.mean(0)
_, _, Vt = np.linalg.svd(z_c, full_matrices=False)
z_2d = z_c @ Vt[:2].T
sc   = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=U_all, cmap="plasma", s=18)
plt.colorbar(sc, ax=ax, label="U/t", shrink=0.85)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("(a) PCA of z(γ_GS) — colored by U/t")

# Panel (b): Kinetic energy linear probe
ax2 = axes[1]
A_gs = np.hstack([z_gs,   np.ones((n,1))]); coef_gs, _, _, _ = lstsq(A_gs, E_kin_all, rcond=None)
A_st = np.hstack([z_star, np.ones((n,1))]); coef_st, _, _, _ = lstsq(A_st, E_kin_all, rcond=None)
ax2.plot(U_all, E_kin_all, "k-",  lw=2.0, label="Exact E_kin")
ax2.plot(U_all, A_gs @ coef_gs, "g--", lw=1.5, label="Oracle probe",   alpha=0.9)
ax2.plot(U_all, A_st @ coef_st, "b:",  lw=1.5, label="Denoiser probe", alpha=0.9)
ax2.set_xlabel("U/t"); ax2.set_ylabel("Kinetic energy")
ax2.set_title("(b) E_kin: exact vs linear probe from z")
ax2.legend(fontsize=9)

# Panel (c): Relative ||z* - z_GS|| / ||z_GS||
ax3 = axes[2]
dists = np.linalg.norm(z_star - z_gs, axis=1)
norms = np.linalg.norm(z_gs, axis=1)
ax3.plot(U_all, dists / norms * 100, "C0-", lw=2)
ax3.set_xlabel("U/t")
ax3.set_ylabel("‖z* − z(γ_GS)‖ / ‖z(γ_GS)‖  (%)")
ax3.set_title("(c) Denoiser z* accuracy vs correlation strength")
ax3.set_ylim(bottom=0)

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(OUT, f"fig4.{ext}"), dpi=150, bbox_inches="tight")
print(f"\nSaved fig4.png / fig4.pdf")
