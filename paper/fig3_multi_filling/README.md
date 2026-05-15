# Figure 3 — Multi-filling: DFT analog fails at ALL filling fractions

## What this figure shows

Extends the half-filling experiment to three different filling fractions
(n = 1/3, 2/3, 1) with H = [U/t, n] as the 2-dimensional Hamiltonian parameter.

**Panel (a)**: Learning curves (MAE vs N_labels) for all methods combined across
all three fillings.

**Panel (b)**: Bar chart comparing DFT analog vs Q-JEPA at N=100, broken down
by individual filling fraction. Each bar also shows the constant density value ρᵢ
for that filling.

## Scientific conclusions

**The key physical argument**: In a periodic system (PBC) with fixed particle
number and translational symmetry, the electron density is strictly uniform:

```
ρᵢ = ⟨n̂ᵢ⟩ = N_total / L = const   for ALL U/t values, at a given filling n
```

Therefore a density-based functional (DFT analog) receives the SAME input for
all U values → it predicts the mean E₀ over all U → catastrophic failure.

**Results at N=100**:
| Filling | Q-JEPA denoiser | DFT analog | Ratio |
|---------|----------------|------------|-------|
| n=1/3   | 0.030          | 0.110      | 3.7×  |
| n=2/3   | 0.131          | 0.536      | 4.1×  |
| n=1 (half) | 0.187       | 1.535      | 8.2×  |

The failure is most severe at half-filling where the Mott insulator physics
drives large E₀ variations with U. At any filling, the 1-RDM γ's off-diagonal
elements encode the quantum correlations that ρ misses.

**This result directly answers**: "is a density-based descriptor insufficient in
general, or only at half-filling?" Answer: insufficient at ALL fillings with PBC.
The failure is a consequence of translational symmetry, not a special property
of half-filling.

## How to reproduce

```bash
# 1. Generate multi-filling data (from repo root)
python src/generate_data_filling.py       # → data/hubbard_filling_{ssl,gs}.npz

# 2. Pretrain Q-JEPA on all fillings (h_dim=2)
python src/train_ssl_filling.py           # → checkpoints/jepa_filling.pt

# 3. Run few-shot evaluation
python src/eval_filling.py                # → results/filling_results{,_byfill}.npy

# 4. Generate this figure
python paper/fig3_multi_filling/plot.py
```

## Dependencies
- `results/filling_results.npy` — overall MAE per method per N
- `results/filling_results_byfill.npy` — MAE broken down by filling fraction
