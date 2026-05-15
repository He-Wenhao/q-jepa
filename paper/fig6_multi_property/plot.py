"""
Figure 6 — Universal functional: one z*, multiple properties.

Loads results/multi_property_results.npy and plots:
  Panel (a): Learning curves (MAE vs N_labels) for E₀
  Panel (b): Learning curves for E_kin
  Panel (c): Learning curves for D (double occupancy)

4 methods in each panel:
  Q-JEPA denoiser (blue)
  Oracle          (green dashed)
  Full-γ supervised (orange)
  DFT analog      (red triangles)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

# ── Load ──────────────────────────────────────────────────────────────────────
data = np.load(os.path.join(RESULTS_DIR, "multi_property_results.npy"),
               allow_pickle=True).item()

methods    = ["denoiser", "oracle", "full_gamma", "dft_analog"]
prop_names = ["E0", "Ekin", "D"]
prop_labels = {
    "E0":   r"$E_0$ (ground-state energy)",
    "Ekin": r"$E_\mathrm{kin}$ (kinetic energy)",
    "D":    r"$D$ (double occupancy)",
}
N_LABEL_LIST = sorted(data["denoiser"]["E0"].keys())

style = {
    "denoiser":  dict(color="#2196F3", marker="o",       ls="-",  lw=2,   label="Q-JEPA denoiser"),
    "oracle":    dict(color="#4CAF50", marker="s",       ls="--", lw=1.5, label="Oracle (γ_GS enc.)"),
    "full_gamma":dict(color="#FF9800", marker="D",       ls="-",  lw=1.5, label="Full-γ supervised"),
    "dft_analog":dict(color="#F44336", marker="^",       ls="-",  lw=1.5, label="DFT analog (ρ only)"),
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
fig.suptitle("Universal functional: one $z^*$, multiple physical observables",
             fontsize=13, y=1.01)

for ax, prop in zip(axes, prop_names):
    for m in methods:
        means = [np.mean(data[m][prop][n]) for n in N_LABEL_LIST]
        sems  = [np.std(data[m][prop][n]) / np.sqrt(len(data[m][prop][n]))
                 for n in N_LABEL_LIST]
        ax.errorbar(N_LABEL_LIST, means, yerr=sems,
                    capsize=3, markersize=5, **style[m])
    ax.set_xlabel("Number of labelled samples", fontsize=11)
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title(prop_labels[prop], fontsize=11)
    ax.set_xticks(N_LABEL_LIST)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig6.{ext}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
