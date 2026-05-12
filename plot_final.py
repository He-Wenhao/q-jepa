"""Final comprehensive figures for Q-JEPA paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 6

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

# ── Load all results ──────────────────────────────────────────────────────────
r_it  = np.load("results/iterate_results.npy", allow_pickle=True).item()
r_ds  = np.load("results/downstream_results.npy", allow_pickle=True).item()
dec   = np.load("results/decoder_eval.npz")
d_gs  = np.load("data/hubbard_gs.npz")
d_hf  = np.load("data/hubbard_hf.npz")

U_all  = d_gs["U"]
E0_all = d_gs["E0"]
E_hf   = d_hf["E0_HF"]
HF_MAE = float(np.abs(E_hf - E0_all).mean())
HF_MAE_STRONG = float(np.abs((E_hf - E0_all)[U_all >= 8]).mean())

# ── Figure 1: Main result – learning curve ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
Ns = sorted(r_it["iterate"].keys())

configs = [
    ("iterate",    "Q-JEPA (from random γ₀)",        "C0", "o-",  2.0),
    ("upperbound", "Q-JEPA (encode γ_GS, oracle)",    "C2", "s--", 1.5),
    ("direct_U",   "Baseline: U → E₀ MLP",            "C3", "^:",  1.5),
]
for key, label, color, fmt, lw in configs:
    means = [np.mean(r_it[key][n]) for n in Ns]
    stds  = [np.std(r_it[key][n])  for n in Ns]
    ax.errorbar(Ns, means, yerr=stds, fmt=fmt, color=color, label=label,
                capsize=5, linewidth=lw, markersize=8)

# HF horizontal line
ax.axhline(HF_MAE, color="gray", linestyle="--", linewidth=1.5,
           label=f"Hartree-Fock (AFM), MAE={HF_MAE:.3f}")

ax.set_xlabel("Number of labeled (U, E₀) pairs", fontsize=12)
ax.set_ylabel("MAE of E₀ prediction", fontsize=12)
ax.set_title("(a) Few-shot E₀ prediction (all U/t)", fontsize=12)
ax.set_xticks(Ns)
ax.legend(fontsize=9, loc="upper right")
ax.set_yscale("log")

# Panel (b): strong correlation regime
ax2 = axes[1]
for key, label, color, fmt, lw in configs:
    # Use downstream results for by-regime breakdown (finetune → "Q-JEPA fine-tune")
    pass

# Use the iterate results + HF_STRONG
for key, label, color, fmt, lw in configs:
    means = [np.mean(r_it[key][n]) for n in Ns]
    stds  = [np.std(r_it[key][n])  for n in Ns]
    ax2.errorbar(Ns, means, yerr=stds, fmt=fmt, color=color, label=label,
                 capsize=5, linewidth=lw, markersize=8, alpha=0.35)

# Strong-correlation breakdown from downstream results
Ns_ds = sorted(r_ds["finetune"].keys())
for key, label, color, fmt, lw in [
    ("finetune", "Q-JEPA (fine-tune, U≥8)", "C0", "o-",  2.0),
    ("scratch",  "Scratch (U≥8)",            "C1", "s--", 1.5),
]:
    means = [np.mean(r_ds[key][n]["strong"]) for n in Ns_ds]
    stds  = [np.std(r_ds[key][n]["strong"])  for n in Ns_ds]
    ax2.errorbar(Ns_ds, means, yerr=stds, fmt=fmt, color=color, label=label,
                 capsize=5, linewidth=lw, markersize=8)

ax2.axhline(HF_MAE_STRONG, color="gray", linestyle="--", linewidth=1.5,
            label=f"HF (AFM) U≥8, MAE={HF_MAE_STRONG:.3f}")

ax2.set_xlabel("Number of labeled pairs", fontsize=12)
ax2.set_ylabel("MAE of E₀ (strong correlation U≥8)", fontsize=12)
ax2.set_title("(b) Strong-correlation regime (U/t ≥ 8)", fontsize=12)
ax2.set_xticks(Ns_ds)
ax2.legend(fontsize=9)
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("results/fig_main_final.png", dpi=150, bbox_inches="tight")
plt.savefig("results/fig_main_final.pdf", bbox_inches="tight")
print("Saved fig_main_final")

# ── Figure 2: E0 curve + HF comparison ───────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(U_all, E0_all / L, "k-",  lw=2.5, label="Exact ED")
ax.plot(U_all, E_hf / L,   "r--", lw=1.5, label="HF (AFM)")
ax.fill_between(U_all, E0_all/L, E_hf/L, alpha=0.15, color='red',
                label="HF error (DFT fails here)")
ax.axvline(4, color='gray', linestyle='--', alpha=0.4)
ax.axvline(8, color='gray', linestyle=':',  alpha=0.4)
ax.text(1, -0.5, "weak\ncorr.", ha='center', color='gray', fontsize=9)
ax.text(6, -0.5, "medium", ha='center', color='gray', fontsize=9)
ax.text(10, -0.5, "strong\ncorr.", ha='center', color='gray', fontsize=9)
ax.set_xlabel("U/t", fontsize=12)
ax.set_ylabel("E₀/L (energy per site)", fontsize=12)
ax.set_title("1D Hubbard (L=6, half-filling): exact vs Hartree-Fock", fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("results/fig_E0_HF.png", dpi=150, bbox_inches="tight")
plt.savefig("results/fig_E0_HF.pdf", bbox_inches="tight")
print("Saved fig_E0_HF")

# ── Figure 3: Decoder reconstruction ─────────────────────────────────────────
U_te   = dec["U_test"]
g_true = dec["gamma_pred"]  # already predicted
g_pred = dec["gamma_pred"]
mae_te = dec["mae_overall"]

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

# Panel: reconstruction error by U/t
ax = axes3[0]
sort_idx = np.argsort(U_te)
# compute per-sample MAE
g_true_loaded = dec["gamma_true"]
g_pred_loaded = dec["gamma_pred"]
per_sample_mae = np.abs(g_true_loaded - g_pred_loaded).mean(axis=(1,2))
ax.scatter(U_te, per_sample_mae * 1000, c=U_te, cmap='plasma', s=20)
ax.set_xlabel("U/t")
ax.set_ylabel("1-RDM MAE (×10⁻³)")
ax.set_title(f"(a) Decoder z(γ_GS)→γ_GS, overall MAE={mae_te*1000:.2f}×10⁻³")

# Panel: show one example 1-RDM (true vs reconstructed), pick a medium-U case
ax2 = axes3[1]
mid_idx = np.argmin(np.abs(U_te - 6.0))
vmax = max(abs(g_true_loaded[mid_idx]).max(), abs(g_pred_loaded[mid_idx]).max())
im = ax2.imshow(g_true_loaded[mid_idx] - g_pred_loaded[mid_idx],
                cmap='RdBu_r', vmin=-0.002, vmax=0.002)
plt.colorbar(im, ax=ax2, label='True - Predicted')
ax2.set_title(f"(b) Reconstruction error at U/t≈{U_te[mid_idx]:.1f}")
ax2.set_xlabel("j"); ax2.set_ylabel("i")

plt.tight_layout()
plt.savefig("results/fig_decoder.png", dpi=150, bbox_inches="tight")
plt.savefig("results/fig_decoder.pdf", bbox_inches="tight")
print("Saved fig_decoder")

print("\n=== Summary Table ===")
print(f"{'Method':35s}  N=10    N=20    N=50   N=100")
for key, label in [("iterate","Q-JEPA (denoiser)"), ("upperbound","Oracle (encode γ_GS)"), ("direct_U","Direct U→E₀")]:
    row = "  ".join(f"{np.mean(r_it[key][n]):.4f}" for n in [10,20,50,100])
    print(f"  {label:33s}  {row}")
print(f"  {'HF (AFM) [no labels needed]':33s}  {HF_MAE:.4f}  {HF_MAE:.4f}  {HF_MAE:.4f}  {HF_MAE:.4f}")
