"""Generate result figures for Q-JEPA experiments."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

# ── Load results ─────────────────────────────────────────────────────────────
r_it = np.load("results/iterate_results.npy", allow_pickle=True).item()
r_ds = np.load("results/downstream_results.npy", allow_pickle=True).item()

d = np.load("data/hubbard_gs.npz")
U_all, E0_all = d["U"], d["E0"]

# ── Figure 1: Learning curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): iterate evaluation
ax = axes[0]
methods_it = {
    "JEPA (iterate from γ₀)":     ("iterate",    "C0", "o-"),
    "JEPA (encode γ_GS, oracle)":  ("upperbound", "C2", "s--"),
    "Direct U → E₀":               ("direct_U",   "C3", "^:"),
}
Ns = sorted(r_it["iterate"].keys())
for label, (key, color, fmt) in methods_it.items():
    means = [np.mean(r_it[key][n]) for n in Ns]
    stds  = [np.std(r_it[key][n])  for n in Ns]
    ax.errorbar(Ns, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=4, linewidth=2, markersize=7)

ax.set_xlabel("Number of labeled (U, E₀) pairs")
ax.set_ylabel("MAE of E₀ prediction")
ax.set_title("(a) JEPA iterate vs baselines")
ax.set_xticks(Ns)
ax.legend(fontsize=9)
ax.set_yscale("log")

# Panel (b): fine-tune vs scratch
ax = axes[1]
methods_ds = {
    "Pretrained (fine-tune)": ("finetune", "C0", "o-"),
    "Scratch (e2e)":           ("scratch",  "C1", "s--"),
    "Direct RDM → E₀":        ("direct",   "C3", "^:"),
}
Ns_ds = sorted(r_ds["finetune"].keys())
for label, (key, color, fmt) in methods_ds.items():
    means = [np.mean(r_ds[key][n]["all"]) for n in Ns_ds]
    stds  = [np.std(r_ds[key][n]["all"])  for n in Ns_ds]
    ax.errorbar(Ns_ds, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=4, linewidth=2, markersize=7)

ax.set_xlabel("Number of labeled (U, E₀) pairs")
ax.set_ylabel("MAE of E₀ prediction")
ax.set_title("(b) Fine-tune vs scratch (input: γ_GS)")
ax.set_xticks(Ns_ds)
ax.legend(fontsize=9)
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("results/fig1_learning_curves.pdf", bbox_inches="tight")
plt.savefig("results/fig1_learning_curves.png", dpi=150, bbox_inches="tight")
print("Saved fig1_learning_curves")

# ── Figure 2: E₀ vs U (ground truth curve) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(U_all, E0_all / 6, "k-", linewidth=2, label="ED exact (E₀/L)")
ax.axvline(4,  color="gray", linestyle="--", alpha=0.5, label="U/t = 4 (weak/medium)")
ax.axvline(8,  color="gray", linestyle=":",  alpha=0.5, label="U/t = 8 (medium/strong)")
ax.set_xlabel("U/t")
ax.set_ylabel("E₀/L (per site)")
ax.set_title("Ground-state energy of 1D Hubbard model (L=6, half-filling)")
ax.legend()
plt.tight_layout()
plt.savefig("results/fig2_E0_vs_U.pdf", bbox_inches="tight")
plt.savefig("results/fig2_E0_vs_U.png", dpi=150, bbox_inches="tight")
print("Saved fig2_E0_vs_U")

# ── Figure 3: Strong-correlation breakdown ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
Ns_ds = sorted(r_ds["finetune"].keys())
regimes = [("weak", "U<4", "C0"), ("medium", "4≤U<8", "C1"), ("strong", "U≥8", "C2")]
for key, label, color in regimes:
    means = [np.mean(r_ds["finetune"][n][key]) for n in Ns_ds]
    stds  = [np.std(r_ds["finetune"][n][key])  for n in Ns_ds]
    ax.errorbar(Ns_ds, means, yerr=stds, label=f"Pretrained, {label}",
                fmt="o-", color=color, capsize=4, linewidth=2, markersize=6)

for key, label, color in [("weak", "U<4", "C0"), ("medium", "4≤U<8", "C1"), ("strong", "U≥8", "C2")]:
    means = [np.mean(r_ds["scratch"][n][key]) for n in Ns_ds]
    ax.plot(Ns_ds, means, linestyle="--", color=color, alpha=0.6, label=f"Scratch, {label}")

ax.set_xlabel("Number of labeled pairs")
ax.set_ylabel("MAE of E₀")
ax.set_title("(c) MAE by correlation regime")
ax.set_xticks(Ns_ds)
ax.legend(ncol=2, fontsize=8)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("results/fig3_by_regime.pdf", bbox_inches="tight")
plt.savefig("results/fig3_by_regime.png", dpi=150, bbox_inches="tight")
print("Saved fig3_by_regime")

print("\nAll figures saved to results/")
