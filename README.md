# Q-JEPA → Trajectory SSL for Quantum Many-Body Systems

## Core Scientific Question

**Can unlabeled imaginary-time trajectories substitute for labeled ground-state data?**

In quantum chemistry and condensed matter, computing ground states is expensive (DMRG, QMC, CCSD(T)). But the *convergence trajectories* of these solvers are generated for free — they are discarded intermediate states on the way to the solution. This project asks: can we learn from those trajectories to reduce the number of expensive labeled ground states needed?

## The Experiment

### Task
Given a Hamiltonian H and a canonical initial state γ₀, predict the ground-state 1-RDM γ_GS.

The 1-RDM (one-particle reduced density matrix) encodes all one-body observables and, via functionals, approximates many-body properties. For the Hubbard model:

```
γ_{σ,ij} = ⟨c†_{iσ} c_{jσ}⟩_GS
```

### Three-Way Comparison

All three methods use the **same model** f(γ, H) → γ' and the **same fine-tuning** on N labeled (H, γ_GS) pairs. The only difference is initialization:

| Method | Pretraining | What it learns |
|--------|-------------|----------------|
| **A: No pretrain** | Random init | Fine-tune only on N labels |
| **B: Random-pair pretrain** | f(γ_rand, H) → γ_rand' on random unrelated (γ, γ') pairs | Same data volume, no temporal structure |
| **C: Trajectory pretrain** | f(γ_t, H) → γ_{t+1} on imaginary-time trajectories | Temporal structure of convergence |

If **C > B > A**: the trajectory *structure* (not just data volume) is what helps.
If **C > B ≈ A**: trajectory structure helps, random pairs do not.
If **C ≈ B > A**: unlabeled data in general helps, structure irrelevant.

### Model Architecture

No encoder, no latent space, no JEPA. Direct γ-space learning:

```
f : (γ_t, H_vec) → γ_{t+1}
```

where H_vec is the full Hamiltonian parameter vector (hopping t_{ij}, onsite ε_i, interaction U).

- f is a simple MLP (or attention over orbital pairs)
- γ is flattened as input; output is reshaped and symmetrized
- Fine-tuning: f(γ_0_canonical, H) → γ_GS with N labeled pairs

### Hamiltonian Space

Diverse Hubbard-like Hamiltonians with random parameters:

```
H = Σ_{ij,σ} t_{ij} c†_{iσ} c_{jσ} + Σ_{i,σ} ε_i n_{iσ} + U Σ_i n_{i↑} n_{i↓}
```

- t_{ij}: random hopping (nearest-neighbor + small next-nearest-neighbor)
- ε_i: random onsite energies (disorder)
- U: random interaction strength

This tests genuine generalization across Hamiltonian space, not just interpolation in U.

## Why This Matters

**Data efficiency in correlated quantum systems.** Current ML functionals (DM21, etc.) train on endpoint labels only — the CCSD(T) converged density. Every trajectory step is discarded. If trajectory pretraining transfers to fewer labels needed at fine-tuning, one can:

1. Run existing expensive calculations (DMRG/QMC/CCSD(T)) for their trajectories
2. Pretrain on trajectories of cheap instances
3. Fine-tune on very few expensive labeled examples of the hard instances

## Why γ (1-RDM) and Not ρ (Density)?

For periodic systems with PBC, ρᵢ = N/L = const for all U. The density is informationally useless — it cannot distinguish a Mott insulator (large U) from a metal (small U). The off-diagonal elements γ_{ij} (i≠j) encode quantum coherence and are sensitive to interactions.

For molecules: γ_{μν} = Σ_k f_k C_{μk} C*_{νk} (MO-basis density matrix), where off-diagonals encode bond order and correlation.

## Connection to 1-RDMFT

In 1-RDM functional theory, the ground-state energy is:

```
E[γ] = T[γ] + E_ne[γ] + E_x[γ] + E_c[γ]
```

The first three terms are *exact* functionals of γ. Only E_c[γ] needs approximation — a smaller ML target than the full E_xc[ρ] in DFT. Learning γ_GS directly sidesteps the functional approximation entirely.

## Previous Work (v1, archived in git)

The first design used a JEPA architecture (encoder + EMA target encoder + predictor + denoiser) with a latent space. Results were positive (denoiser ≈ oracle at N=100 labels), but the architecture had a fundamental issue: the predictor and denoiser did not share parameters, so the "trajectory data is useful" claim was not cleanly isolated. The new design (this README) removes the latent space entirely, making the three-way comparison unambiguous.

See `git log` for archived results.

## Directory Structure (new)

```
src/
  generate_data.py     # diverse Hubbard trajectories + labeled GS
  model.py             # f(γ, H) → γ' (MLP, no encoder)
  pretrain.py          # Stage 1: SSL on trajectory pairs
  finetune.py          # Stage 2: fine-tune on N labeled pairs
  eval.py              # three-way comparison A/B/C
data/
  trajectories/        # unlabeled imaginary-time paths
  labeled_gs/          # (H, γ_GS) pairs for fine-tuning/test
results/
checkpoints/
```
