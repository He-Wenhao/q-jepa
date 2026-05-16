# Figure 4 — Fine-Tuning Convergence (Equal Simulation Budget)

## What This Figure Shows

Validation MAE vs fine-tuning epoch for endpoint-only and trajectory SSL,
both at N=20 labeled examples and equal simulation budget. Shows that trajectory
SSL reaches a lower error faster, providing a better loss-landscape initialization.

## Experimental Design

**Setup**: Same equal-budget framework as Fig 2 (N_TRAJ=1).
For each of 3 seeds: sample N=20 Hamiltonians from pool.
- Endpoint: fine-tune from random init, log val MAE every 5 epochs
- Trajectory: pretrain on same 20 trajectories (200 epochs), then fine-tune,
  log val MAE every 5 epochs
Fine-tuning: 500 epochs, AdamW lr=1e-4, cosine schedule.
Test set: 100 held-out Hamiltonians (same as Fig 2).

## Reproduction

```bash
python paper/fig4/compute_fig4.py   # runs panel_c only
python paper/fig4/plot_fig4.py
```

## Results

Final MAE after 500 epochs (N=20, mean over 3 seeds):
- Endpoint-only: 0.0995
- Trajectory SSL: 0.0514
Ratio: ×1.9 — consistent with Fig 2 at N=20 (×1.8)

The trajectory SSL curve starts lower (better initialization from pretraining)
and converges to a lower plateau. The gap is present from epoch 1 and is
maintained throughout fine-tuning.

## Key Insights

1. **Better initialization**: Trajectory pretraining places the model parameters
   in a region of loss landscape that is already close to the fine-tuning optimum,
   even before seeing any labeled data.

2. **Same final learning rate**: Both methods use the same fine-tuning hyperparameters,
   so the advantage is not due to any asymmetry in optimization.

3. **Gap maintained**: The ~2× advantage is present at every epoch, not just at
   convergence, confirming this is an initialization effect rather than a
   generalization effect.

## Proposed Caption

**Figure 4. Trajectory SSL provides better fine-tuning initialization.**
Validation MAE vs fine-tuning epoch for endpoint-only (gray, dashed) and trajectory
SSL (blue, solid) at N=20 labeled examples (equal simulation budget). Shaded regions:
±1 std over 3 random seeds. Trajectory SSL starts from a near-optimal initialization
provided by the pretrained model and converges to a ~2× lower plateau, consistent
with the equal-budget comparison in Fig. 2.
