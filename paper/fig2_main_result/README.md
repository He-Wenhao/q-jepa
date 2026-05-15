# Figure 2 — Main result: few-shot E₀ prediction

## What this figure shows

Few-shot ground-state energy prediction on the 1D Hubbard model (L=6,
half-filling), comparing four methods as a function of the number of labeled
(U, E₀) pairs used for fine-tuning (N ∈ {10, 20, 50, 100}, 5 random seeds):

| Method | Description |
|--------|-------------|
| **Q-JEPA denoiser** | Starts from random Fock state γ₀ → encoder → denoiser(z₀, H) → z* → EnergyHead(z*) |
| **Oracle** | Uses exact ground-state γ_GS → encoder → z_GS → EnergyHead(z_GS) — upper bound |
| **DFT analog (ρ only)** | Diagonal of γ_GS (electron density) → MLP → E₀ — should fail |
| **Full-γ supervised** | Flatten(γ_GS) → MLP → E₀ — supervised baseline, no SSL |
| **HF baseline** | Hartree-Fock energy, no labels needed |

## Scientific conclusions

1. **DFT analog fails** (MAE ≈ 1.55, off the chart): at half-filling with PBC,
   ρᵢ = 0.5 for all U/t values by translational symmetry — the density is a
   constant and carries zero information about U. This directly motivates using
   the full 1-RDM γ as a descriptor instead of just ρ.

2. **Q-JEPA beats HF at N ≥ 50**: with only 50 labeled pairs (vs. solving the
   many-body problem directly), the SSL-pretrained denoiser reaches lower MAE
   than Hartree-Fock (which requires full SCF convergence).

3. **Q-JEPA ≈ Oracle at large N**: the denoiser's z* approaches the quality of
   directly encoding the exact γ_GS, confirming that the SSL objective successfully
   teaches the model to find ground-state latent representations.

4. **Q-JEPA beats full-γ supervised at N ≥ 50**: the SSL pretraining provides a
   data-efficient inductive bias that plain supervised learning lacks.

## Architecture recap (corrected DFT analogy)

```
Encoder:   γ → z          (universal state compression, NO Hamiltonian params)
Predictor: (z, H) → z'    (H-conditioned dynamics, KS iteration analog)
Denoiser:  (z, H) → z*    (H-conditioned ground-state finder)
Head:      z* → E₀        (universal energy functional, NO Hamiltonian params)
```

## How to reproduce

```bash
# 1. Generate data (from repo root)
python src/generate_data.py          # → data/hubbard_ssl.npz, hubbard_gs.npz
python src/hf_baseline.py            # → data/hubbard_hf.npz

# 2. Pretrain Q-JEPA
python src/train_ssl.py              # → checkpoints/jepa_pretrained.pt

# 3. Run few-shot evaluation
python src/eval_iterate.py           # → results/iterate_results.npy

# 4. Generate this figure
python paper/fig2_main_result/plot.py
```

## Dependencies
- `results/iterate_results.npy` — pre-computed evaluation results (5 seeds × 4 N values)
- `data/hubbard_gs.npz`, `data/hubbard_hf.npz` — for HF baseline MAE
