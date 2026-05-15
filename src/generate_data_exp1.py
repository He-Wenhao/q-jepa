"""
Generate combined data for Experiment 1.

For each Hamiltonian we save BOTH:
  - imaginary-time trajectory γ_t  (unlabeled)
  - ground-state γ_GS              (label)

This lets us compare, at a fixed simulation budget N:
  - endpoint-only  : train f(γ_0, H) → γ_GS on N labeled pairs from scratch
  - trajectory     : pretrain f(γ_t,H)→γ_{t+1} on N trajectories,
                     then fine-tune f(γ_0,H)→γ_GS on the same N endpoints

Output: data/exp1_combined.npz
  h_vecs    : (M, h_dim)
  gamma_gs  : (M, rdm_dim, rdm_dim)
  E0        : (M,)
  gamma_traj: (M, n_traj, T_steps, rdm_dim, rdm_dim)
  is_test   : (M,) bool  — 100 held-out test Hamiltonians
  gamma_0   : (rdm_dim, rdm_dim)  canonical initial state
"""
import os, sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        build_hubbard_general, compute_1rdm_fast,
                        ground_state, imaginary_time_evolve, random_superposition)

# ── Config ────────────────────────────────────────────────────────────────────
L            = 6
N_UP = N_DN  = 3
RDM_DIM      = 2 * L

M_TOTAL      = 300        # total Hamiltonians (200 pool + 100 test)
N_TEST       = 100
N_TRAJ       = 1          # trajectories per Hamiltonian
TAU_STEPS    = 30
DELTA_TAU    = 0.1

T_MEAN  = 1.0;  T_SPREAD = 0.5
EPS_MAX = 0.5;  U_MAX    = 8.0
H_DIM   = L + L + 1      # t_bonds(6) + eps(6) + U(1)
SEED    = 2024

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def canonical_gamma_0():
    g = np.zeros((RDM_DIM, RDM_DIM), dtype=np.float32)
    for s in range(2):
        n_s = N_UP if s == 0 else N_DN
        for i in range(L):
            g[s * L + i, s * L + i] = n_s / L
    return g


def sample_ham(rng):
    t = T_MEAN + T_SPREAD * (rng.random(L) * 2 - 1)
    e = EPS_MAX * (rng.random(L) * 2 - 1)
    u = rng.random() * U_MAX
    h = np.concatenate([t, e, [u]]).astype(np.float32)
    return t, e, u, h


def main():
    rng = np.random.default_rng(SEED)
    print("Building ED structures...")
    basis     = build_basis(L, N_UP, N_DN)
    basis_idx = make_basis_index(basis)
    ops       = build_all_cdagger_c(L, basis, basis_idx)
    print(f"  Hilbert space dim = {len(basis)},  M = {M_TOTAL}")

    h_vecs_list    = []
    gamma_gs_list  = []
    E0_list        = []
    gamma_traj_list = []   # shape per sample: (N_TRAJ, TAU_STEPS+1, RDM_DIM, RDM_DIM)

    for _ in tqdm(range(M_TOTAL)):
        t_bonds, eps, U, h_vec = sample_ham(rng)
        H_sp = build_hubbard_general(L, t_bonds, eps, U, basis, basis_idx)
        E0, psi_gs = ground_state(H_sp)
        gam_gs = compute_1rdm_fast(psi_gs, ops, L).astype(np.float32)

        trajs = []
        for _ in range(N_TRAJ):
            psi0 = random_superposition(basis, rng)
            steps = list(imaginary_time_evolve(H_sp, psi0, TAU_STEPS, DELTA_TAU))
            frames = np.stack([compute_1rdm_fast(p, ops, L).astype(np.float32)
                               for _, p in steps])  # (TAU_STEPS+1, RDM, RDM)
            trajs.append(frames)

        h_vecs_list.append(h_vec)
        gamma_gs_list.append(gam_gs)
        E0_list.append(float(E0))
        gamma_traj_list.append(np.stack(trajs))   # (N_TRAJ, T+1, RDM, RDM)

    is_test = np.zeros(M_TOTAL, dtype=bool)
    is_test[:N_TEST] = True   # first N_TEST are test (shuffled below)
    perm = rng.permutation(M_TOTAL)
    is_test = is_test[np.argsort(perm)]   # random assignment

    out = os.path.join(DATA_DIR, "exp1_combined.npz")
    np.savez(out,
             h_vecs     = np.array(h_vecs_list),
             gamma_gs   = np.array(gamma_gs_list),
             E0         = np.array(E0_list, dtype=np.float32),
             gamma_traj = np.array(gamma_traj_list),
             is_test    = is_test,
             gamma_0    = canonical_gamma_0())
    print(f"\nSaved {out}")
    print(f"  pool={int((~is_test).sum())}  test={int(is_test.sum())}")
    print(f"  gamma_traj shape: {np.array(gamma_traj_list).shape}")


if __name__ == "__main__":
    main()
