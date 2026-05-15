"""
Figure 2: Main result — few-shot E₀ prediction, 4-method comparison.

Methods:
  A) Q-JEPA denoiser  : random γ₀ → encoder → denoiser(z₀,H) → z* → head(z*)
  B) Oracle           : γ_GS → encoder → z_GS → head(z_GS)   [upper bound]
  C) DFT analog       : ρ = diag(γ_GS) → MLP → E₀            [should fail]
  D) Full-γ supervised: flatten(γ_GS) → MLP → E₀              [no SSL]
  E) Hartree-Fock     : physics baseline, no labels
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.dirname(os.path.abspath(__file__))

r    = np.load(os.path.join(ROOT, "results", "iterate_results.npy"), allow_pickle=True).item()
d_gs = np.load(os.path.join(ROOT, "data", "hubbard_gs.npz"))
d_hf = np.load(os.path.join(ROOT, "data", "hubbard_hf.npz"))
HF_MAE = float(np.abs(d_hf["E0_HF"] - d_gs["E0"]).mean())

Ns = sorted(r["denoiser"].keys())

plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})
fig, ax = plt.subplots(figsize=(7, 5))

configs = [
    ("denoiser",   "Q-JEPA denoiser (from random γ₀)",  "C0", "o-",  2.2),
    ("oracle",     "Oracle (encode exact γ_GS)",          "C2", "s--", 1.6),
    ("full_gamma", "Full-γ supervised (no SSL)",          "C1", "D:",  1.6),
    ("dft_analog", "DFT analog — ρ only (fails here)",   "C3", "^:",  1.6),
]
for key, label, color, fmt, lw in configs:
    means = [np.mean(r[key][n]) for n in Ns]
    stds  = [np.std(r[key][n])  for n in Ns]
    ax.errorbar(Ns, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=5, linewidth=lw, markersize=9)

ax.axhline(HF_MAE, color="gray", linestyle="--", linewidth=1.8,
           label=f"Hartree-Fock (no labels), MAE={HF_MAE:.3f}")

ax.set_xlabel("Number of labeled (U, E₀) pairs", fontsize=13)
ax.set_ylabel("Test MAE of E₀ prediction", fontsize=13)
ax.set_title("Q-JEPA: few-shot E₀ prediction (L=6, half-filling)", fontsize=12)
ax.set_xticks(Ns)
ax.legend(fontsize=10, loc="upper right")
ax.set_yscale("log")

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(OUT, f"fig2.{ext}"),
                dpi=150, bbox_inches="tight")

print("=== Results table ===")
print(f"{'Method':35s}  N=10    N=20    N=50   N=100")
for key, label, *_ in configs:
    row = "  ".join(f"{np.mean(r[key][n]):.4f}" for n in [10, 20, 50, 100])
    print(f"  {label[:33]:33s}  {row}")
print(f"  {'HF baseline':33s}  " + "  ".join(f"{HF_MAE:.4f}" for _ in range(4)))
print(f"\nSaved fig2.png / fig2.pdf")
