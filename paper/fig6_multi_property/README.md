# Figure 6 — Universal Functional: one z*, multiple physical observables

## What this figure shows

Tests the core "universal functional" claim of Q-JEPA: a **single** pretrained
encoder + denoiser produces one latent vector z*, and separate small heads trained
on z* can predict multiple physically distinct observables from the same z*.

**Three properties predicted simultaneously from z*:**

| Observable | Physical meaning | Why it's hard |
|------------|-----------------|---------------|
| E₀         | Total ground-state energy | Requires capturing both kinetic and interaction effects |
| E_kin      | Kinetic energy = −t Σ_{<ij>,σ}(γ_{ij}+γ_{ji}) | Measures electron delocalization |
| D          | Double occupancy ⟨n↑n↓⟩ = (E₀−E_kin)/U | Key Mott physics signature; D→0 at large U |

All three vary strongly with U/t, so the DFT analog (ρ=const) fails for all.
Uses the multi-filling model (h_dim=2) across all three fillings n ∈ {1/3, 2/3, 1}.

---

## Four methods compared

### A) Q-JEPA denoiser (blue) — our method
Same encoder+denoiser as used in Fig 3. One z* per sample (averaged over
N_FOCK_AVG=5 random Fock states). Separate small MLP head per property.
**The head never sees H = (U, filling).**

### B) Oracle (green dashed) — upper bound
Encodes the exact ground-state γ_GS directly. Cheating: requires solving the
quantum many-body problem first. Upper bound for this framework.

### C) Full-γ supervised (orange)
Flattens γ_GS to 144-dim, trains MLP directly. No SSL. Requires exact γ_GS
at inference, and needs fresh training per property.

### D) DFT analog (red) — baseline that fails
Uses only ρ = diag(γ_GS) as input. Due to translational symmetry, ρᵢ = const
for all U at fixed filling. Predicts only the mean value → catastrophic failure
for all three properties.

---

## Core scientific conclusion

**The universal functional claim holds**: one z* (same encoder+denoiser, no
retraining) predicts E₀, E_kin, and D with comparable accuracy for all.
The denoiser z* approaches oracle quality at N ≥ 50, and beats full_gamma
(which needs exact γ_GS) due to SSL-structured latent space.

DFT analog fails for **all** observables, not just E₀, because ρ carries no U
information at fixed filling under PBC — reinforcing that z (from full 1-RDM γ)
is the right representation.

---

## How to reproduce

```bash
# Requires: checkpoints/jepa_filling.pt and data/hubbard_filling_gs.npz
python src/eval_multi_property.py    # → results/multi_property_results.npy
python paper/fig6_multi_property/plot.py   # → paper/fig6_multi_property/fig6.{png,pdf}
```

## Dependencies
- `results/multi_property_results.npy` — evaluation results
- `checkpoints/jepa_filling.pt` — pretrained multi-filling Q-JEPA
- `data/hubbard_filling_gs.npz` — ground-state data for all fillings
