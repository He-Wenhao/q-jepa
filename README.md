# Trajectory SSL for Quantum Many-Body Systems

## Core Scientific Question

**Can unlabeled imaginary-time trajectories substitute for labeled ground-state data?**

In quantum chemistry and condensed matter, computing ground states is expensive (DMRG, QMC, CCSD(T)). But the *convergence trajectories* of these solvers are generated for free — they are discarded intermediate states on the way to the solution. This project asks: can we learn from those trajectories to reduce the number of expensive labeled ground states needed?

---

## Experiment 1 — Trajectory vs Endpoint-Only (Primary Result)

**Direct comparison at equal computational budget.**

- x-axis: number of simulations N (each simulation = run one Hamiltonian to convergence)
- y-axis: MAE on predicted γ_GS (ground-state 1-RDM)

| Method | What it uses |
|--------|--------------|
| **Endpoint-only** | N labeled (H, γ_GS) pairs, trained from scratch |
| **Trajectory** | N trajectories γ_t (unlabeled) → pretrain; then fine-tune on same N endpoints |

Both methods run exactly N expensive quantum simulations. The only difference: trajectory method also uses the intermediate steps.

**Results** (`results/exp1_results.npy`): see `paper/fig1_exp1.{pdf,png}` (generated after run completes).

---

## Experiment 2 — Ablation & Generalization (Support)

Three-way comparison on a fixed labeled test set, with 500 unlabeled SSL Hamiltonians
separate from the 200 labeled ones.

Methods:
- **A (no_pretrain)**: Random init → fine-tune on N labels
- **B (rand_pretrain)**: Pretrain on mismatched trajectory pairs → fine-tune
- **C (traj_pretrain)**: Pretrain on consecutive trajectory pairs → fine-tune

### 2a — In-distribution (U∈[0,8] everywhere)

| N | A | B | C |
|---|---|---|---|
| 5  | 0.148 | 0.020 | **0.018** |
| 10 | 0.130 | 0.016 | **0.014** |
| 20 | 0.095 | 0.013 | **0.011** |
| 50 | 0.049 | 0.009 | **0.008** |

### 2b — Extrapolation (SSL/fine-tune U∈[0,6] → test U∈[6,10])

| N | A | B | C |
|---|---|---|---|
| 5  | 0.150 | 0.022 | **0.017** |
| 10 | 0.134 | 0.020 | **0.015** |
| 20 | 0.113 | 0.016 | **0.013** |
| 50 | 0.064 | 0.011 | **0.010** |

Extrapolation performance is nearly identical to interpolation — the learned representation
generalizes to the Mott insulator regime (U > t) without seeing it during SSL.

### 2c — SSL data volume scaling (N_ssl ∈ {50,100,200,500}, N_labels=10)

| N_ssl | MAE (traj_pretrain) |
|-------|---------------------|
| 50    | 0.027 |
| 100   | 0.021 |
| 200   | 0.017 |
| 500   | 0.015 |

No-pretrain baseline at N_labels=10: **0.130** — even 50 unlabeled Hamiltonians
give 6× improvement.

**Why is B so good?** Experiment 2 has a known confound: the canonical γ_0 (fine-tuning
input) equals the mean γ across the training distribution. So B, which learns to predict
the mean, gets a "free" good initialization. Experiment 1 avoids this by using the same
N Hamiltonians for trajectories and endpoints, so there is no separate "random-pairs" pool.

---

## System & Architecture

**Physical system**: 1D Hubbard model, L=6, half-filling (N_up=N_dn=3), PBC.
Random Hamiltonians: t_bonds ∈ [0.5,1.5], ε_i ∈ [-0.5,0.5], U ∈ [0,8].

**Model**: `DeltaPredictor` — MLP with residual connection, no encoder, no latent space.
```
f(γ_t, H_vec) → γ_{t+1} = γ_t + MLP([flatten(γ_t), normalize(H_vec)])
```
Output symmetrized: (γ + γᵀ)/2. 4 hidden layers × 256 units, LayerNorm + GELU.

**1-RDM γ** (12×12 matrix) encodes quantum coherence including off-diagonal elements.
Unlike ρ = diag(γ) (density), γ can distinguish metal from Mott insulator.

---

## Why γ and Not ρ?

For periodic systems with PBC, ρᵢ = N/L = const for all U — the density is
informationally useless. The off-diagonal elements γ_{ij} (i≠j) encode hopping
amplitudes and are strongly suppressed in the Mott regime (large U).

---

## Connection to 1-RDMFT

Ground-state energy: E[γ] = T[γ] + E_ne[γ] + E_x[γ] + E_c[γ].
The first three terms are exact functionals of γ. Only E_c[γ] needs approximation.
Learning γ_GS directly sidesteps functional approximation entirely.

---

## Directory Structure

```
src/
  generate_data.py         — diverse Hubbard SSL + labeled data (Exp 2)
  generate_data_exp1.py    — combined trajectory+endpoint data (Exp 1)
  model.py                 — DeltaPredictor f(γ, H) → γ'
  pretrain.py              — SSL pretraining (--mode traj|rand)
  finetune.py              — three-way fine-tuning evaluation (Exp 2)
  exp1.py                  — trajectory vs endpoint at equal budget (Exp 1)
  scaling.py               — SSL data volume ablation (Exp 2c)
  hubbard_ed.py            — exact diagonalization utilities
data/
  exp1_combined.npz        — Exp 1 data: trajectories + endpoints per Hamiltonian
  trajectories.npz         — Exp 2 SSL trajectory pairs (500 Hamiltonians)
  labeled_gs.npz           — Exp 2 labeled GS (200 Hamiltonians, held out from SSL)
  trajectories_extrap.npz  — same but U∈[0,6] only
  labeled_gs_extrap.npz    — extrap test: U∈[6,10]
results/
  exp1_results.npy         — Exp 1 main result
  finetune_results.npy     — Exp 2a in-distribution
  finetune_extrap_results.npy — Exp 2b extrapolation
  scaling_results.npy      — Exp 2c scaling
paper/
  plot_all.py              — generates all Exp 2 figures
  fig1_main.{pdf,png}      — Exp 2a in-distribution
  fig2_extrap.{pdf,png}    — Exp 2b extrapolation
  fig3_scaling.{pdf,png}   — Exp 2c scaling
  fig_combined.{pdf,png}   — 3-panel combined (Exp 2)
checkpoints/
  pretrain_traj.pt         — Exp 2, trajectory pretrain (500 SSL Hamiltonians)
  pretrain_rand.pt         — Exp 2, random-pair pretrain
  pretrain_traj_extrap.pt  — Exp 2b extrap version
  pretrain_rand_extrap.pt  — Exp 2b extrap version
```

---

## Data & Reproducibility

### What's on GitHub vs Google Drive

Large binary files (`data/`, `results/`, `checkpoints/`, `logs/`) are excluded from git
and stored on Google Drive at:

```
gdrive:research/q-jepa/
  data/          — all .npz datasets (~120 MB)
  results/       — all .npy result arrays
  checkpoints/   — pretrained model weights (.pt)
  logs/          — training logs
```

### Download data from Google Drive (rclone)

**One-time setup**: install and authenticate rclone with Google Drive.
```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash   # or: conda install -c conda-forge rclone

# Authenticate (opens browser)
rclone config
# -> choose "n" (new remote), name it "gdrive", type "drive", follow OAuth prompts
```

**Download all data to your local clone**:
```bash
# From the project root
rclone copy gdrive:research/q-jepa/data/        data/        --progress
rclone copy gdrive:research/q-jepa/results/     results/     --progress
rclone copy gdrive:research/q-jepa/checkpoints/ checkpoints/ --progress
```

Or download everything at once:
```bash
rclone copy gdrive:research/q-jepa/ . \
  --include "data/**" --include "results/**" --include "checkpoints/**" \
  --progress
```

### Quick start (after data download)

```bash
# Reproduce main result (Fig 2) — ~20 min on GPU
python src/exp1.py

# Reproduce OOD result (Fig 4) — ~40 min on GPU
python src/exp_ood.py

# Reproduce mechanism analysis (Fig 5) — ~60 min on GPU
python src/exp5.py

# Plot all figures
python paper/fig1/plot_fig1.py
python paper/fig2/plot_exp1.py
python paper/fig4_ood/plot_fig4_ood.py
python paper/fig5/plot_fig5.py
```

### Regenerate data from scratch

```bash
# Main dataset (Exp 1 + Exp 2) — ~5 min
python src/generate_data_exp1.py

# OOD dataset — ~30 s
python src/generate_data_ood.py

# Fig 5 power-iteration dataset — ~2 min
python src/generate_data_fig5.py
```

### Paper

The manuscript draft is at `paper/manuscript/main.tex` (compiled PDF: `paper/manuscript/main.pdf`).
Compile with:
```bash
# Using tectonic (downloads only needed packages, no full texlive install)
curl -fsSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz | tar -xz -C /tmp
/tmp/tectonic paper/manuscript/main.tex
```
