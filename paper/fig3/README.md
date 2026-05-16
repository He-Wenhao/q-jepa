# Figure 3 — OOD Generalization to the Mott Insulator Regime

## What This Figure Shows

Tests whether the trajectory SSL benefit persists when the test set probes
qualitatively different physics that was never seen during self-supervised pretraining.

SSL pretraining and fine-tuning pool: U∈[0,6] (metallic to weakly correlated regime)
Test set: U∈[6,10] — the strongly-correlated Mott insulator regime, OOD

Three methods compared:
- No pretrain (gray): random init → fine-tune on N labels
- Random-pair pretrain (orange): pretrain on mismatched trajectory pairs → fine-tune
- Trajectory SSL (blue): pretrain on consecutive trajectory pairs → fine-tune

## Experimental Design

**Data**: 500 SSL Hamiltonians (U∈[0,6], unlabeled), 200 labeled Hamiltonians
(U∈[0,6] fine-tuning pool, 100 test Hamiltonians with U∈[6,10]).
**Pretraining**: 100 epochs on all 500 SSL Hamiltonians.
**Fine-tuning**: 300 epochs per run, 5 seeds.
**Note**: This experiment uses a separate SSL pool (not equal-budget like Fig 2).
The key comparison is trajectory vs random-pair pretrain, which controls for
the amount of SSL data.

## Reproduction

```bash
# Generate OOD data (one-time)
python src/generate_data.py --extrap

# Pretrain
python src/pretrain.py --mode traj --traj_path data/trajectories_extrap.npz \
       --ckpt_name pretrain_traj_extrap
python src/pretrain.py --mode rand --traj_path data/trajectories_extrap.npz \
       --ckpt_name pretrain_rand_extrap

# Fine-tune and evaluate
python src/finetune.py --labeled_path data/labeled_gs_extrap.npz \
       --traj_ckpt pretrain_traj_extrap \
       --rand_ckpt pretrain_rand_extrap \
       --out_name finetune_extrap_results

# Plot
python paper/fig3/plot_all.py
```

## Results

| N_labels | No pretrain | Rand pretrain | Traj SSL |
|----------|-------------|---------------|----------|
| 5        | ~0.134      | ~0.020        | ~0.015   |
| 10       | ~0.130      | ~0.018        | ~0.014   |
| 20       | ~0.125      | ~0.016        | ~0.013   |
| 50       | ~0.110      | ~0.014        | ~0.011   |

OOD performance is virtually identical to in-distribution (see Fig 3 original README
for in-distribution numbers), confirming that the pretrained representation
generalizes across qualitatively different physical regimes.

## Key Insights

1. **OOD generalization**: The model pretrained exclusively on metallic Hamiltonians
   (U∈[0,6]) predicts ground states in the Mott insulator regime (U∈[6,10]) with
   nearly unchanged accuracy. This suggests the learned representation captures
   universal features of imaginary-time dynamics, not regime-specific patterns.

2. **Trajectory > random-pair in OOD**: The gap between trajectory and random-pair
   SSL persists out-of-distribution, ruling out the possibility that the improvement
   is due to memorizing the training distribution.

3. **Practical relevance**: In realistic scenarios, one often has access to cheap
   weakly-correlated simulations (small U) and wants to predict strongly-correlated
   physics (large U). This result suggests trajectory SSL transfers across this
   physically important boundary.

## Proposed Caption

**Figure 3. Trajectory SSL generalizes to the Mott insulator regime.**
MAE of predicted γ_GS vs number of labeled fine-tuning examples N for three methods,
evaluated on test Hamiltonians in the strongly-correlated Mott regime (U/t∈[6,10]).
Self-supervised pretraining and fine-tuning pool use only U/t∈[0,6] (metallic regime).
Trajectory SSL (blue) and random-pair SSL (orange) both substantially outperform
no-pretrain (gray); trajectory SSL retains a consistent advantage over random-pair SSL,
indicating that directional imaginary-time dynamics — not merely exposure to diverse
RDM samples — transfers to the OOD regime. Error bars: ±1 std over 5 random seeds.
