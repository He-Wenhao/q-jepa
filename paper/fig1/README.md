# Figure 1 — Method Schematic (Cartoon)

## What This Figure Shows

A conceptual illustration of the core idea: imaginary-time convergence trajectories
are generated for free during any ground-state solver, but are traditionally discarded.
We propose to keep them and use them for self-supervised pretraining.

## Suggested Layout (three panels, left to right)

**Panel (a): The waste in current practice**
- A quantum solver (ED / DMRG / CCSD(T)) converging iteratively
- Trajectory: γ_0 → γ_1 → γ_2 → ... → γ_GS
- Intermediate states γ_t (t < T) shown grayed out / discarded
- Only γ_GS and its label E_0 are kept for training

**Panel (b): Our approach**
- Same solver, but intermediate states are saved (colored)
- Stage 1 (SSL, no labels): learn f(γ_t, H) → γ_{t+1} from trajectory pairs
- Stage 2 (fine-tune, N labels): fine-tune f(γ_0, H) → γ_GS on N endpoints

**Panel (c): Equal-budget comparison**
- Both methods run N simulations (same computational cost)
- Endpoint-only: uses only N final states
- Ours: uses the same N final states + intermediate steps (free)
- Ours achieves lower error at every N

## Reproduction

Hand-drawn schematic. No code needed.
Recommended: Inkscape, Illustrator, or draw.io.

## Proposed Caption

**Figure 1. Trajectory self-supervised learning for quantum ground-state prediction.**
(a) Conventional practice: iterative quantum solvers produce convergence trajectories
γ₀→γ₁→⋯→γ_GS, but only the final ground state is retained as a training label.
(b) Our approach: consecutive trajectory pairs (γ_t, γ_{t+1}) serve as free
self-supervised signal. A neural network f(γ_t, H)→γ_{t+1} is pretrained on
unlabeled trajectories, then fine-tuned on N labeled endpoints to predict γ_GS
for new Hamiltonians. (c) At equal simulation budget N, trajectory pretraining
consistently outperforms endpoint-only supervised learning.
