"""
Generate power-iteration trajectories for Figure 5.
Reuses same Hamiltonians and pool/test split from exp1_combined.npz.

Saves data/fig5_power.npz:
  gamma_traj_power : (M, T+1, rdm_dim, rdm_dim) — power iteration trajectories
  (plus h_vecs, gamma_gs, is_test, gamma_0 copied from exp1 for convenience)
"""
import os, sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        build_hubbard_general, compute_1rdm_fast,
                        power_iteration_evolve, random_superposition)

L       = 6
N_UP = N_DN = 3
TAU_STEPS   = 30     # same length as imaginary-time trajectories
POWER_SHIFT = 20.0   # must exceed max eigenvalue of H

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

T_MEAN = 1.0; T_SPREAD = 0.5
EPS_MAX = 0.5; U_MAX = 8.0


def main():
    src = np.load(os.path.join(DATA_DIR, "exp1_combined.npz"))
    h_vecs   = src["h_vecs"]
    gamma_gs = src["gamma_gs"]
    is_test  = src["is_test"]
    gamma_0  = src["gamma_0"]
    M        = len(h_vecs)

    print("Building ED structures...")
    basis     = build_basis(L, N_UP, N_DN)
    basis_idx = make_basis_index(basis)
    ops       = build_all_cdagger_c(L, basis, basis_idx)
    print(f"  Hilbert space dim={len(basis)},  M={M}")

    rng = np.random.default_rng(2025)
    traj_power = []

    for idx in tqdm(range(M)):
        h = h_vecs[idx]
        t_bonds = h[:L]
        eps     = h[L:2*L]
        U       = float(h[-1])
        H_sp    = build_hubbard_general(L, t_bonds, eps, U, basis, basis_idx)

        psi0   = random_superposition(basis, rng)
        frames = [compute_1rdm_fast(psi, ops, L).astype(np.float32)
                  for _, psi in power_iteration_evolve(H_sp, psi0, TAU_STEPS, POWER_SHIFT)]
        traj_power.append(np.stack(frames))   # (T+1, 12, 12)

    out = os.path.join(DATA_DIR, "fig5_power.npz")
    np.savez(out,
             h_vecs          = h_vecs,
             gamma_gs        = gamma_gs,
             is_test         = is_test,
             gamma_0         = gamma_0,
             gamma_traj_power = np.array(traj_power))
    print(f"\nSaved {out}")
    print(f"  gamma_traj_power shape: {np.array(traj_power).shape}")


if __name__ == "__main__":
    main()
