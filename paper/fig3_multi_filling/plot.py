"""
Figure 3: Multi-filling experiment — DFT analog fails at ALL filling fractions.

Key insight: With PBC and fixed particle number, translational symmetry forces
ρᵢ = N_total / L = const for every U/t at a given filling. So a density-based
functional (DFT analog) cannot distinguish different U values within a filling.

This is tested across three filling fractions:
  n = 1/3  (N_up=1, N_dn=1)
  n = 2/3  (N_up=2, N_dn=2)
  n = 1    (N_up=3, N_dn=3, half-filling — Mott insulator transition)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.dirname(os.path.abspath(__file__))

r    = np.load(os.path.join(ROOT, "results", "filling_results.npy"),
               allow_pickle=True).item()
r_bf = np.load(os.path.join(ROOT, "results", "filling_results_byfill.npy"),
               allow_pickle=True).item()

FILLING_VALS   = [1/3, 2/3, 1.0]
FILLING_LABELS = ["n = 1/3\n(quarter)", "n = 2/3", "n = 1\n(half, Mott)"]
Ns = sorted(r["denoiser"].keys())

plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel (a): Overall learning curves ────────────────────────────────────────
ax = axes[0]
configs = [
    ("denoiser",   "Q-JEPA denoiser",   "C0", "o-",  2.2),
    ("oracle",     "Oracle (γ_GS)",     "C2", "s--", 1.6),
    ("full_gamma", "Full-γ supervised", "C1", "D:",  1.6),
    ("dft_analog", "DFT analog (ρ)",    "C3", "^:",  1.6),
]
for key, label, color, fmt, lw in configs:
    means = [np.mean(r[key][n]) for n in Ns]
    stds  = [np.std(r[key][n])  for n in Ns]
    ax.errorbar(Ns, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=5, linewidth=lw, markersize=9)
ax.set_xlabel("Labeled pairs per filling fraction", fontsize=12)
ax.set_ylabel("Test MAE of E₀ prediction", fontsize=12)
ax.set_title("(a) Multi-filling: all fillings combined", fontsize=12)
ax.set_xticks(Ns)
ax.legend(fontsize=10)
ax.set_yscale("log")

# ── Panel (b): DFT analog vs Q-JEPA by filling (N=100) ────────────────────────
ax2 = axes[1]
x     = np.arange(len(FILLING_VALS))
width = 0.35
n_show = 100

for i, (key, label, color, *_) in enumerate(
        [(k, l, c) for k, l, c, *_ in configs if k in ("denoiser", "dft_analog")]):
    means = [np.mean(r_bf[key][n_show][fn]) for fn in FILLING_VALS]
    stds  = [np.std(r_bf[key][n_show][fn])  for fn in FILLING_VALS]
    ax2.bar(x + (i - 0.5) * width, means, width, yerr=stds,
            capsize=5, color=color, label=label, alpha=0.85)

# annotate: ρ = const value for each filling
for xi, fn in zip(x, FILLING_VALS):
    ax2.text(xi, 0.02, f"ρᵢ≡{fn:.2f}", ha="center", fontsize=9,
             color="C3", style="italic")

ax2.set_xlabel("Filling fraction", fontsize=12)
ax2.set_ylabel(f"Test MAE of E₀  (N={n_show} labels)", fontsize=12)
ax2.set_title("(b) DFT analog (ρ-only) vs Q-JEPA by filling\n"
              "ρᵢ = const within each filling → zero U information", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(FILLING_LABELS)
ax2.legend(fontsize=10)

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(OUT, f"fig3.{ext}"),
                dpi=150, bbox_inches="tight")

print("=== DFT analog vs Q-JEPA at N=100, by filling ===")
for fn, lbl in zip(FILLING_VALS, FILLING_LABELS):
    dn  = np.mean(r_bf["denoiser"][100][fn])
    dft = np.mean(r_bf["dft_analog"][100][fn])
    print(f"  {lbl.replace(chr(10),' '):20s}  Q-JEPA={dn:.4f}  DFT-analog={dft:.4f}  "
          f"ratio={dft/dn:.1f}×")
print(f"\nSaved fig3.png / fig3.pdf")
