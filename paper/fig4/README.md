# Figure 4 — Why Does Trajectory Pretraining Work?

Three planned experiments, likely combined into one multi-panel figure.

## Panel A: Iterated Prediction (Zero-Shot)

After pretraining f(γ_t, H) → γ_{t+1} on trajectories (no fine-tuning),
iterate the model starting from γ_0:

    γ_0 → f(γ_0, H) → f²(γ_0, H) → f³(γ_0, H) → ...

If the model has learned the imaginary-time dynamics, this should converge
to γ_GS without any labeled data.

Proposed plot: MAE vs iteration step k, for several Hamiltonians.
Compare model trajectory against true imaginary-time trajectory.

## Panel B: Physical Observables (Mott Physics)

From the predicted γ̂_GS, compute:
  - E_kin = -t Σ_{<ij>} (γ_{ij} + γ_{ji})     (kinetic energy)
  - D = (E_0 - E_kin) / U                       (double occupancy → Mott indicator)

Plot D vs U for endpoint-only and trajectory methods, compared to exact ED.
Show that trajectory method correctly captures D → 0 as U → ∞ (Mott transition).

## Panel C: Fine-Tuning Convergence

Compare training loss curves during fine-tuning for:
  - A: no pretrain (starts from random)
  - C: traj pretrain (starts from good initialization)

Show that C reaches a given accuracy with far fewer fine-tuning steps.
This reveals that trajectory pretraining provides a better loss landscape,
not just a different inductive bias.
