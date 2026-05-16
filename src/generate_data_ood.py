"""
Generate equal-budget OOD dataset for Figure 4.

Pool (training): 200 Hamiltonians with U∈[0,6]  (metallic regime)
Test:            100 Hamiltonians with U∈[6,10]  (Mott insulator regime, OOD)
N_TRAJ=1 per Hamiltonian (equal-budget, same as exp1).

Saves data/ood_combined.npz with same structure as exp1_combined.npz.
"""
import os, sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hubbard_ed import (build_basis, make_basis_index, build_all_cdagger_c,
                        build_hubbard_general, compute_1rdm_fast,
                        ground_state, imaginary_time_evolve, random_superposition)

L = 6; N_UP = N_DN = 3; RDM_DIM = 2 * L
M_POOL   = 200
M_TEST   = 100
N_TRAJ   = 1
TAU_STEPS = 30; DELTA_TAU = 0.1
T_MEAN = 1.0; T_SPREAD = 0.5; EPS_MAX = 0.5
SEED = 2026

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


def sample_ham(rng, u_lo, u_hi):
    t = T_MEAN + T_SPREAD * (rng.random(L) * 2 - 1)
    e = EPS_MAX * (rng.random(L) * 2 - 1)
    u = u_lo + rng.random() * (u_hi - u_lo)
    return t, e, u, np.concatenate([t, e, [u]]).astype(np.float32)


def main():
    rng = np.random.default_rng(SEED)
    basis     = build_basis(L, N_UP, N_DN)
    basis_idx = make_basis_index(basis)
    ops       = build_all_cdagger_c(L, basis, basis_idx)
    print(f"Building OOD dataset  pool U∈[0,6]  test U∈[6,10]")
    print(f"  Hilbert dim={len(basis)}  pool={M_POOL}  test={M_TEST}  N_TRAJ={N_TRAJ}")

    h_vecs_list, gamma_gs_list, gamma_traj_list, is_test_list = [], [], [], []

    # Pool: U∈[0,6]
    for _ in tqdm(range(M_POOL), desc="Pool U∈[0,6]"):
        t_b, eps, U, h_vec = sample_ham(rng, 0.0, 6.0)
        H_sp = build_hubbard_general(L, t_b, eps, U, basis, basis_idx)
        E0, psi_gs = ground_state(H_sp)
        gamma_gs = compute_1rdm_fast(psi_gs, ops, L).astype(np.float32)
        trajs = []
        for _ in range(N_TRAJ):
            psi0   = random_superposition(basis, rng)
            frames = np.stack([compute_1rdm_fast(p, ops, L).astype(np.float32)
                               for _, p in imaginary_time_evolve(H_sp, psi0, TAU_STEPS, DELTA_TAU)])
            trajs.append(frames)
        h_vecs_list.append(h_vec); gamma_gs_list.append(gamma_gs)
        gamma_traj_list.append(np.stack(trajs)); is_test_list.append(False)

    # Test: U∈[6,10]
    for _ in tqdm(range(M_TEST), desc="Test  U∈[6,10]"):
        t_b, eps, U, h_vec = sample_ham(rng, 6.0, 10.0)
        H_sp = build_hubbard_general(L, t_b, eps, U, basis, basis_idx)
        E0, psi_gs = ground_state(H_sp)
        gamma_gs = compute_1rdm_fast(psi_gs, ops, L).astype(np.float32)
        h_vecs_list.append(h_vec); gamma_gs_list.append(gamma_gs)
        gamma_traj_list.append(np.zeros((N_TRAJ, TAU_STEPS+1, RDM_DIM, RDM_DIM), np.float32))
        is_test_list.append(True)

    out = os.path.join(DATA_DIR, "ood_combined.npz")
    np.savez(out,
             h_vecs     = np.array(h_vecs_list),
             gamma_gs   = np.array(gamma_gs_list),
             gamma_traj = np.array(gamma_traj_list),
             is_test    = np.array(is_test_list, dtype=bool),
             gamma_0    = canonical_gamma_0())
    print(f"\nSaved {out}")
    print(f"  gamma_traj shape: {np.array(gamma_traj_list).shape}")


if __name__ == "__main__":
    main()
