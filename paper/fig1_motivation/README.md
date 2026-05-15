# Figure 1 — Motivation: Why HF fails at strong correlation

## What this figure shows

Ground-state energy per site (E₀/L) of the 1D Hubbard model (L=6, half-filling,
PBC) as a function of interaction strength U/t, comparing:

- **Exact ED** (exact diagonalization, ground truth)
- **Hartree-Fock AFM** (mean-field approximation, standard DFT proxy)

The shaded region shows the HF error, which grows dramatically for U/t ≥ 4
(medium → strong correlation regime) and reaches an overall MAE of ~0.54 eV·t.

## Scientific conclusion

Hartree-Fock — the backbone of standard DFT exchange-correlation functionals —
fails precisely in the physically interesting strongly-correlated regime (Mott
insulator physics at half-filling). This motivates Q-JEPA: a self-supervised
framework that learns a richer latent descriptor (the full 1-RDM γ rather than
just the electron density ρ) to predict ground-state properties beyond the reach
of mean-field methods.

## How to reproduce

```bash
# 1. Generate ground-state and HF data (from repo root)
python src/generate_data.py      # → data/hubbard_gs.npz
python src/hf_baseline.py        # → data/hubbard_hf.npz

# 2. Generate this figure
python paper/fig1_motivation/plot.py  # → paper/fig1_motivation/fig1.{png,pdf}
```

## Dependencies
- `data/hubbard_gs.npz` — exact ED ground states
- `data/hubbard_hf.npz` — Hartree-Fock energies
