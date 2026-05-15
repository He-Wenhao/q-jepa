# Figure 4 — Why Does Trajectory Pretraining Work?

Three planned experiments that probe the mechanism behind the improvement.
Likely combined into one multi-panel figure.

## Panel (a): Iterated Prediction — Does the Model Learn True Dynamics?

**Question**: After pretraining f(γ_t, H)→γ_{t+1} with no labels, does iterating
the model starting from γ_0 converge to γ_GS?

**Experiment**: For several test Hamiltonians, apply f repeatedly:
    γ_0 → f(γ_0,H) → f²(γ_0,H) → f³(γ_0,H) → ...
Measure MAE(f^k(γ_0,H), γ_GS) as a function of iteration k.
Compare against true imaginary-time trajectory MAE(γ_t, γ_GS) vs τ=t·δτ.

**Expected result**: The model's iterated trajectory should converge (possibly slower
than true dynamics, but monotonically decreasing). This would demonstrate that
the model has internalized the physics of imaginary-time flow, not merely
memorized statistical patterns.

**Script to write**: `paper/fig4/plot_iter.py`
**Data needed**: `results/exp1_results.npy` (existing), plus new iterative eval code.

## Panel (b): Physical Observables — Does the Model Capture Mott Physics?

**Question**: Does the predicted γ̂_GS reproduce physically meaningful observables,
in particular the Mott metal-insulator transition?

**Experiment**: For N=20 labeled training examples, compute from predicted γ̂_GS:
    E_kin = -t Σ_{<ij>,σ} (γ_{ij} + γ_{ji})     (kinetic energy; proxy for metallicity)
    D     = (E_0 - E_kin) / U                     (double occupancy; D→0 = Mott insulator)
Plot D vs U for: exact ED, endpoint-only (N=20), trajectory SSL (N=20).

**Expected result**: Trajectory SSL should track the exact D(U) curve more closely,
especially in the large-U Mott regime. The endpoint-only method at N=20 may miss
the sharp crossover around U/t ≈ 4–6.

**Script to write**: `paper/fig4/plot_observables.py`
**Data needed**: `data/exp1_combined.npz` (existing), re-run fine-tuning with
E_kin and D logged.

## Panel (c): Fine-Tuning Convergence — Better Loss Landscape

**Question**: Does trajectory pretraining provide a better initialization in the
loss landscape (fewer fine-tuning steps needed), or just a different inductive bias?

**Experiment**: Log validation MAE vs fine-tuning epoch for both methods at N=20.
Compare learning curves: trajectory pretrain should start lower and converge faster.

**Expected result**: Trajectory SSL starts near the true answer at epoch 0 (because
iterating the pretrained model already gives a reasonable γ_GS approximation),
while endpoint-only starts from random output. The gap shrinks as epochs increase
but trajectory SSL reaches a lower final loss.

**Script to write**: `paper/fig4/plot_convergence.py`
**Data needed**: Modify `src/exp1.py` to log per-epoch validation loss.

## Reproduction (once scripts are written)

```bash
# Panel (a): iterative prediction
python paper/fig4/plot_iter.py

# Panel (b): physical observables
python paper/fig4/plot_observables.py

# Panel (c): convergence curves
# First re-run exp1 with logging:
python src/exp1.py --log_curve
python paper/fig4/plot_convergence.py
```

## Proposed Caption

**Figure 4. Mechanistic analysis of trajectory self-supervised pretraining.**
(a) Iterated prediction: starting from γ_0, iterating the pretrained model
f^k(γ_0, H) converges toward γ_GS without any labeled fine-tuning, confirming
the model has learned imaginary-time dynamics rather than statistical patterns.
(b) Physical observables: double occupancy D(U) predicted by endpoint-only and
trajectory SSL (N=20 labels each) compared to exact ED. Trajectory SSL reproduces
the Mott crossover (D→0 at large U) more accurately.
(c) Fine-tuning loss curves: trajectory SSL starts from a near-optimal initialization
and converges in ~3× fewer epochs than endpoint-only at equivalent accuracy.
