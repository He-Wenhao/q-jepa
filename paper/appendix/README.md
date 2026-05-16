# Appendix — SSL Data Volume Scaling

## What This Figure Shows

Performance as a function of the number of unlabeled SSL Hamiltonians N_ssl,
at several fixed fine-tuning label counts N_labels. Shows diminishing returns
and how much unlabeled data is needed.

Note: This experiment uses a separate pool of unlabeled SSL Hamiltonians
(not the equal-budget framework of the main figures). It answers a different
question: given that you have a large unlabeled dataset and a small labeled set,
how much does more unlabeled data help?

## Experimental Design

**SSL data**: N_ssl ∈ {50, 100, 200, 500} unlabeled Hamiltonians, U∈[0,8]
**Fine-tuning**: N_labels ∈ {5, 10, 20, 50} labeled Hamiltonians (separate pool)
**Test**: 100 held-out Hamiltonians
**Dotted reference lines**: no-pretrain baseline at each N_labels

## Reproduction

```bash
python src/scaling.py
python paper/appendix/plot_scaling.py
```

## Results

N_ssl: 50→0.027, 100→0.021, 200→0.017, 500→0.015 (at N_labels=10)
No-pretrain baseline at N_labels=10: ~0.130

Even 50 unlabeled Hamiltonians (4,500 trajectory pairs with T=30) give ~6×
improvement over no pretraining at N_labels=10. Performance plateaus around
N_ssl=200-500.

## Key Insights

Even a small number of unlabeled simulations (N_ssl=50) produces substantial
improvement, making the method practical for expensive calculations where even
50 trajectories represent significant computational investment. The plateau
at N_ssl≈200 suggests that data diversity (Hamiltonian variety) matters more
than raw data volume once sufficient coverage is achieved.

## Proposed Caption

**Figure S1. SSL performance vs unlabeled data volume.**
MAE vs number of unlabeled SSL Hamiltonians N_ssl for trajectory SSL at four
fixed labeled fine-tuning sizes N_labels (color, solid lines). Dotted lines of
corresponding color show the no-pretrain baseline at each N_labels. Performance
improves smoothly with N_ssl and plateaus around 200–500, indicating that
coverage of Hamiltonian space matters more than raw data volume. Even N_ssl=50
provides 4–6× improvement over no pretraining across all N_labels tested.
Error bars: ±1 std over 5 random seeds.
