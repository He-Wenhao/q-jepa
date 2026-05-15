"""
Figure 1: Motivation — exact ED vs Hartree-Fock ground-state energy.

Shows why HF (the DFT workhorse for correlated systems) fails in the
strong-correlation regime (U/t ≥ 4), motivating the need for a
beyond-density ML framework.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.dirname(os.path.abspath(__file__))

d_gs = np.load(os.path.join(ROOT, "data", "hubbard_gs.npz"))
d_hf = np.load(os.path.join(ROOT, "data", "hubbard_hf.npz"))
U_all  = d_gs["U"]
E0_all = d_gs["E0"]
E_hf   = d_hf["E0_HF"]
L      = 6

HF_MAE = float(np.abs(E_hf - E0_all).mean())

plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(U_all, E0_all / L, "k-",  lw=2.5, label="Exact ED (ground truth)")
ax.plot(U_all, E_hf   / L, "r--", lw=1.8, label=f"Hartree-Fock (AFM)")
ax.fill_between(U_all, E0_all / L, E_hf / L,
                alpha=0.18, color="red", label=f"HF error  (overall MAE={HF_MAE:.3f})")

ax.axvline(4, color="gray", linestyle="--", alpha=0.5, lw=1)
ax.axvline(8, color="gray", linestyle=":", alpha=0.5, lw=1)
ax.text(1.5, -0.45, "weak\ncorr.", ha="center", color="gray", fontsize=10)
ax.text(6,   -0.45, "medium",      ha="center", color="gray", fontsize=10)
ax.text(10.5,-0.45, "strong\ncorr.",ha="center",color="gray", fontsize=10)

ax.set_xlabel("Hubbard interaction U/t", fontsize=13)
ax.set_ylabel("Ground-state energy E₀/L (per site)", fontsize=13)
ax.set_title("1D Hubbard model (L=6, half-filling): exact vs Hartree-Fock", fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(0, 12)

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(OUT, f"fig1.{ext}"),
                dpi=150, bbox_inches="tight")
print(f"Saved fig1.png / fig1.pdf  (HF MAE = {HF_MAE:.4f})")
