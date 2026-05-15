"""
Figure 5 (supplementary): Decoder — latent z(γ_GS) → reconstructed γ_GS.

Demonstrates that the encoder is information-preserving: the 64-dimensional
latent z retains enough information to reconstruct the full 12×12 1-RDM.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.dirname(os.path.abspath(__file__))

dec   = np.load(os.path.join(ROOT, "results", "decoder_eval.npz"))
U_te  = dec["U_test"]
g_true= dec["gamma_true"]
g_pred= dec["gamma_pred"]
mae_overall = float(dec["mae_overall"])

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Panel (a): reconstruction MAE by U/t
ax = axes[0]
per_mae = np.abs(g_true - g_pred).mean(axis=(1, 2))
sc = ax.scatter(U_te, per_mae * 1000, c=U_te, cmap="plasma", s=20)
plt.colorbar(sc, ax=ax, label="U/t")
ax.set_xlabel("U/t")
ax.set_ylabel("1-RDM reconstruction MAE  (×10⁻³)")
ax.set_title(f"(a) Decoder: z(γ_GS) → γ_GS\nOverall MAE = {mae_overall*1000:.2f}×10⁻³")

# Panel (b): error matrix at U/t ≈ 6
ax2 = axes[1]
mid = np.argmin(np.abs(U_te - 6.0))
err = g_true[mid] - g_pred[mid]
im  = ax2.imshow(err, cmap="RdBu_r", vmin=-0.003, vmax=0.003)
plt.colorbar(im, ax=ax2, label="True − Predicted")
ax2.set_title(f"(b) 1-RDM error matrix at U/t ≈ {U_te[mid]:.1f}")
ax2.set_xlabel("j"); ax2.set_ylabel("i")

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(OUT, f"fig5.{ext}"), dpi=150, bbox_inches="tight")
print(f"Saved fig5.png / fig5.pdf  (overall MAE = {mae_overall*1000:.3f}×10⁻³)")
