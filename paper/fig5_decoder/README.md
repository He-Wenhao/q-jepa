# Figure 5 (Supplementary) — Decoder: latent → 1-RDM reconstruction

## What this figure shows

A linear decoder network trained on top of the frozen encoder verifies that the
64-dimensional latent z is **information-preserving**: it can reconstruct the
full 12×12 one-body reduced density matrix (1-RDM) with high fidelity.

**Panel (a)**: Per-sample reconstruction MAE (×10⁻³) as a function of U/t.
Color encodes U/t. The overall MAE across all test samples is ~0.34×10⁻³.

**Panel (b)**: Element-wise error matrix (True − Predicted) for a
representative sample at U/t ≈ 6 (medium correlation).

## Scientific conclusions

1. The encoder is **information-preserving**: a simple MLP decoder can recover
   the 12×12 = 144-dimensional γ matrix from a 64-dimensional z with average
   error < 0.001 per element.

2. This justifies the use of z as a sufficient statistic for γ — whatever
   physical observable depends on γ can in principle be recovered from z.

3. Reconstruction error is slightly higher at intermediate U/t (crossover
   region), consistent with the increased sensitivity of γ to U near the
   Mott transition.

## How to reproduce

```bash
# Requires pretrained model (run train_ssl.py first)
python train_decoder.py              # → results/decoder_eval.npz

python paper/fig5_decoder/plot.py   # → paper/fig5_decoder/fig5.{png,pdf}
```

## Dependencies
- `results/decoder_eval.npz` — decoder evaluation results
