"""
Comprehensive plot for all three experiments.

Exp 1: Multi-filling (DFT analog fails within each filling)
Exp 2: Transfer learning (weak-U SSL → strong-U few-shot)
Exp 3: Iterative refinement (K predictor steps before denoiser)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
RESULTS_DIR = "results"
L = 6

# ── Exp 1: Multi-filling ──────────────────────────────────────────────────────
r_fill    = np.load(f"{RESULTS_DIR}/filling_results.npy",        allow_pickle=True).item()
r_fill_bf = np.load(f"{RESULTS_DIR}/filling_results_byfill.npy", allow_pickle=True).item()

FILLING_VALS = [1/3, 2/3, 1.0]
FILL_LABELS  = {1/3: "n=1/3", 2/3: "n=2/3", 1.0: "n=1 (half)"}
methods_fill = ["denoiser", "oracle", "dft_analog", "full_gamma"]
method_style = {
    "denoiser":   ("Q-JEPA denoiser",   "C0", "o-",  2.0),
    "oracle":     ("Oracle (γ_GS)",     "C2", "s--", 1.5),
    "dft_analog": ("DFT analog (ρ)",    "C3", "^:",  1.5),
    "full_gamma": ("Full-γ supervised", "C1", "D:",  1.5),
}

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.8))

# Panel (a): Overall MAE vs N_labels across all fillings
ax = axes1[0]
Ns_fill = sorted(r_fill["denoiser"].keys())
for m, (label, color, fmt, lw) in method_style.items():
    means = [np.mean(r_fill[m][n]) for n in Ns_fill]
    stds  = [np.std(r_fill[m][n])  for n in Ns_fill]
    ax.errorbar(Ns_fill, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=5, linewidth=lw, markersize=8)
ax.set_xlabel("Labeled (U, E₀) pairs per filling", fontsize=12)
ax.set_ylabel("MAE of E₀ prediction", fontsize=12)
ax.set_title("(a) Multi-filling: all methods (ρ_i = const within each filling)", fontsize=11)
ax.set_xticks(Ns_fill)
ax.legend(fontsize=9)
ax.set_yscale("log")

# Panel (b): DFT analog vs Q-JEPA breakdown by filling (N=100)
ax2 = axes1[1]
n_show = 100
filling_labels = [FILL_LABELS[fn] for fn in FILLING_VALS]
x = np.arange(len(FILLING_VALS))
width = 0.35
for i, (m, (label, color, fmt, lw)) in enumerate(
        [(k, method_style[k]) for k in ["denoiser", "dft_analog"]]):
    means = [np.mean(r_fill_bf[m][n_show][fn]) for fn in FILLING_VALS]
    stds  = [np.std(r_fill_bf[m][n_show][fn])  for fn in FILLING_VALS]
    ax2.bar(x + (i - 0.5) * width, means, width, yerr=stds,
            capsize=4, color=color, label=label, alpha=0.85)
ax2.set_xlabel("Filling fraction", fontsize=12)
ax2.set_ylabel(f"MAE of E₀ (N={n_show} labels)", fontsize=12)
ax2.set_title("(b) DFT analog fails at ALL fillings\n(ρ=const within each filling → zero U info)", fontsize=10)
ax2.set_xticks(x); ax2.set_xticklabels(filling_labels)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_exp1_filling.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{RESULTS_DIR}/fig_exp1_filling.pdf", bbox_inches="tight")
print("Saved fig_exp1_filling")

# Print summary table
print("\n=== Exp 1: Multi-filling results ===")
print(f"{'Method':20s}  N=10    N=20    N=50   N=100")
for m, (label, *_) in method_style.items():
    row = "  ".join(f"{np.mean(r_fill[m][n]):.4f}" for n in [10, 20, 50, 100])
    print(f"  {label:18s}  {row}")
print("\nBy filling (N=50):")
for fn in FILLING_VALS:
    row = "  ".join(f"{m[:6]}={np.mean(r_fill_bf[m][50][fn]):.4f}" for m in methods_fill)
    print(f"  {FILL_LABELS[fn]}: {row}")

# ── Exp 2: Transfer learning ──────────────────────────────────────────────────
r_tr = np.load(f"{RESULTS_DIR}/transfer_results.npy", allow_pickle=True).item()
Ns_tr = sorted(r_tr["transfer_weak"].keys())

fig2, ax = plt.subplots(figsize=(7, 4.5))
tr_configs = [
    ("transfer_weak", "Q-JEPA transfer (pretrain U<4, fine-tune U≥8)", "C0", "o-",  2.0),
    ("transfer_full", "Q-JEPA full (pretrain all U, fine-tune U≥8)",    "C2", "s--", 1.5),
    ("scratch_gamma", "Scratch: γ_GS → E₀ (no SSL)",                   "C1", "D:",  1.5),
    ("scratch_U",     "Scratch: U → E₀ (knows U directly)",             "gray","^:", 1.2),
]
for key, label, color, fmt, lw in tr_configs:
    means = [np.mean(r_tr[key][n]) for n in Ns_tr]
    stds  = [np.std(r_tr[key][n])  for n in Ns_tr]
    ax.errorbar(Ns_tr, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=5, linewidth=lw, markersize=8)
ax.set_xlabel("Labeled pairs from strong-U (U≥8) domain", fontsize=12)
ax.set_ylabel("MAE of E₀ prediction (U≥8 test set)", fontsize=12)
ax.set_title("Transfer learning: SSL on weak-U → few-shot to strong-U", fontsize=12)
ax.set_xticks(Ns_tr)
ax.legend(fontsize=9)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_exp2_transfer.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{RESULTS_DIR}/fig_exp2_transfer.pdf", bbox_inches="tight")
print("Saved fig_exp2_transfer")

print("\n=== Exp 2: Transfer learning (strong-U test MAE) ===")
print(f"{'Method':45s}  N=5     N=10    N=20    N=50")
for key, label, *_ in tr_configs:
    row = "  ".join(f"{np.mean(r_tr[key][n]):.4f}" for n in Ns_tr)
    print(f"  {label:43s}  {row}")

# ── Exp 3: Iterative refinement ───────────────────────────────────────────────
r_it3 = np.load(f"{RESULTS_DIR}/iterative_refine_results.npy", allow_pickle=True).item()
K_STEPS = [0, 3, 8, 20]
Ns_it3 = sorted(r_it3["K=0"].keys())

fig3, ax = plt.subplots(figsize=(7, 4.5))
colors = ["C0", "C1", "C2", "C3"]
for k, color in zip(K_STEPS, colors):
    key   = f"K={k}"
    means = [np.mean(r_it3[key][n]) for n in Ns_it3]
    stds  = [np.std(r_it3[key][n])  for n in Ns_it3]
    lbl   = f"K={k} predictor steps + denoiser" if k > 0 else "K=0 (one-shot denoiser)"
    ax.errorbar(Ns_it3, means, yerr=stds, fmt=f"o-", color=color, label=lbl,
                capsize=5, linewidth=2 if k==0 else 1.5, markersize=8)
# Oracle
means_or = [np.mean(r_it3["oracle"][n]) for n in Ns_it3]
stds_or  = [np.std(r_it3["oracle"][n])  for n in Ns_it3]
ax.errorbar(Ns_it3, means_or, yerr=stds_or, fmt="s--", color="C5", label="Oracle (γ_GS)",
            capsize=5, linewidth=1.5, markersize=8)
ax.set_xlabel("Labeled pairs", fontsize=12)
ax.set_ylabel("MAE of E₀ prediction", fontsize=12)
ax.set_title("Iterative refinement: K predictor steps before denoiser", fontsize=12)
ax.set_xticks(Ns_it3)
ax.legend(fontsize=9)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_exp3_iterative.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{RESULTS_DIR}/fig_exp3_iterative.pdf", bbox_inches="tight")
print("Saved fig_exp3_iterative")

print("\n=== Exp 3: Iterative refinement ===")
print(f"{'Method':35s}  N=10    N=20    N=50   N=100")
for k in K_STEPS:
    key = f"K={k}"
    row = "  ".join(f"{np.mean(r_it3[key][n]):.4f}" for n in [10, 20, 50, 100])
    lbl = f"K={k} (one-shot)" if k == 0 else f"K={k}"
    print(f"  {lbl:33s}  {row}")
row = "  ".join(f"{np.mean(r_it3['oracle'][n]):.4f}" for n in [10, 20, 50, 100])
print(f"  {'oracle':33s}  {row}")
