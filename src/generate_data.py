"""
Generate data for the trajectory SSL experiment.

Hamiltonians: diverse 1D Hubbard with random t_bonds, eps, U
  H = Σ_{<ij>} t_{ij} c†c + Σ_i ε_i n_i + U Σ_i n_{i↑} n_{i↓}

Outputs:
  data/trajectories.npz  — unlabeled trajectory pairs (SSL pretraining)
  data/labeled_gs.npz    — labeled (H, γ_GS) pairs (fine-tuning / test)

The 200 labeled Hamiltonians are held out from SSL entirely (OOD test).
"""
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hubbard_ed import (build_basis, make_basis_index,
                        build_all_cdagger_c, build_hubbard_general,
                        compute_1rdm_fast, ground_state,
                        imaginary_time_evolve, random_superposition)

# ── Config ────────────────────────────────────────────────────────────────────
L        = 6
N_UP     = 3          # half-filling
N_DN     = 3
RDM_DIM  = 2 * L     # 12; γ is (12, 12)

N_HAM_SSL     = 500   # Hamiltonians used for SSL pretraining
N_HAM_LABELED = 200   # Hamiltonians held out for labeled fine-tuning / test
N_TEST        = 100   # of the labeled, these are fixed test Hamiltonians
N_TRAJ_PER_HAM = 3    # random initial states per Hamiltonian
TAU_STEPS     = 30    # imaginary-time steps per trajectory
DELTA_TAU     = 0.1   # step size

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Hamiltonian parameter ranges ──────────────────────────────────────────────
T_MEAN   = 1.0
T_SPREAD = 0.5   # t_bonds ~ U[T_MEAN - T_SPREAD, T_MEAN + T_SPREAD]
EPS_MAX  = 0.5   # eps_i   ~ U[-EPS_MAX, EPS_MAX]
U_MAX    = 8.0   # U       ~ U[0, U_MAX]

H_DIM = L + L + 1  # t_bonds(L) + eps(L) + U(1) = 13


def sample_hamiltonian(rng):
    t_bonds = T_MEAN + T_SPREAD * (rng.random(L) * 2 - 1)
    eps     = EPS_MAX * (rng.random(L) * 2 - 1)
    U       = rng.random() * U_MAX
    h_vec   = np.concatenate([t_bonds, eps, [U]]).astype(np.float32)
    return t_bonds, eps, U, h_vec


def canonical_gamma_0():
    """Uniform diagonal 1-RDM: N_σ/L on diagonal, zeros off-diagonal."""
    gamma = np.zeros((RDM_DIM, RDM_DIM), dtype=np.float32)
    for sigma in range(2):
        n_sigma = N_UP if sigma == 0 else N_DN
        for i in range(L):
            gamma[sigma * L + i, sigma * L + i] = n_sigma / L
    return gamma


def main():
    rng  = np.random.default_rng(SEED)
    N_total = N_HAM_SSL + N_HAM_LABELED

    # Build ED structures (shared across all Hamiltonians at same filling)
    print("Building ED structures...")
    basis     = build_basis(L, N_UP, N_DN)
    basis_idx = make_basis_index(basis)
    ops       = build_all_cdagger_c(L, basis, basis_idx)
    print(f"  Hilbert space dim = {len(basis)}")

    # Sample Hamiltonian parameters
    print(f"Sampling {N_total} Hamiltonians...")
    all_t_bonds = []
    all_eps     = []
    all_U       = []
    all_h_vecs  = []
    for _ in range(N_total):
        t, e, u, hv = sample_hamiltonian(rng)
        all_t_bonds.append(t)
        all_eps.append(e)
        all_U.append(u)
        all_h_vecs.append(hv)

    # Shuffle and split
    idx_perm      = rng.permutation(N_total)
    ssl_idx       = idx_perm[:N_HAM_SSL]
    labeled_idx   = idx_perm[N_HAM_SSL:]          # N_HAM_LABELED total
    test_labeled  = labeled_idx[:N_TEST]           # fixed test set
    pool_labeled  = labeled_idx[N_TEST:]           # fine-tuning pool

    # ── Generate labeled GS data ───────────────────────────────────────────────
    print(f"\nGenerating {N_HAM_LABELED} labeled ground states...")
    lbl_h_vecs  = []
    lbl_gamma   = []
    lbl_E0      = []
    lbl_is_test = []

    for k, i in enumerate(tqdm(labeled_idx)):
        H_sparse = build_hubbard_general(
            L, all_t_bonds[i], all_eps[i], all_U[i], basis, basis_idx)
        E0, psi_gs = ground_state(H_sparse)
        gamma_gs   = compute_1rdm_fast(psi_gs, ops, L).astype(np.float32)
        lbl_h_vecs.append(all_h_vecs[i])
        lbl_gamma.append(gamma_gs)
        lbl_E0.append(float(E0))
        lbl_is_test.append(i in set(test_labeled))

    labeled_path = os.path.join(DATA_DIR, "labeled_gs.npz")
    np.savez(labeled_path,
             h_vec    = np.array(lbl_h_vecs),
             gamma_gs = np.array(lbl_gamma),
             E0       = np.array(lbl_E0, dtype=np.float32),
             is_test  = np.array(lbl_is_test, dtype=bool),
             gamma_0  = canonical_gamma_0(),
             L        = L, N_up = N_UP, N_dn = N_DN,
             h_dim    = H_DIM, rdm_dim = RDM_DIM)
    print(f"  Saved {labeled_path}  (test={sum(lbl_is_test)}, pool={sum(~np.array(lbl_is_test))})")

    # ── Generate SSL trajectory pairs ─────────────────────────────────────────
    print(f"\nGenerating SSL trajectories for {N_HAM_SSL} Hamiltonians...")
    gc_list  = []  # gamma_curr
    gn_list  = []  # gamma_next
    hv_list  = []  # h_vec for the pair
    hid_list = []  # Hamiltonian index (for constructing random-pair baseline)

    for k, i in enumerate(tqdm(ssl_idx)):
        H_sparse = build_hubbard_general(
            L, all_t_bonds[i], all_eps[i], all_U[i], basis, basis_idx)

        for traj_id in range(N_TRAJ_PER_HAM):
            psi0 = random_superposition(basis, rng)
            steps = list(imaginary_time_evolve(H_sparse, psi0, TAU_STEPS, DELTA_TAU))
            gammas = [compute_1rdm_fast(psi, ops, L).astype(np.float32)
                      for _, psi in steps]
            for t in range(len(gammas) - 1):
                gc_list.append(gammas[t])
                gn_list.append(gammas[t + 1])
                hv_list.append(all_h_vecs[i])
                hid_list.append(k)  # Hamiltonian index (0..N_HAM_SSL-1)

    traj_path = os.path.join(DATA_DIR, "trajectories.npz")
    np.savez(traj_path,
             gamma_curr = np.array(gc_list),
             gamma_next = np.array(gn_list),
             h_vec      = np.array(hv_list),
             ham_id     = np.array(hid_list, dtype=np.int32))
    n_pairs = len(gc_list)
    print(f"  Saved {traj_path}  ({n_pairs:,} pairs from {N_HAM_SSL} Hamiltonians)")
    print(f"\nDone. H_dim={H_DIM}, rdm_dim={RDM_DIM}")


if __name__ == "__main__":
    main()
