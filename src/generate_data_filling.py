"""
Generate SSL + ground-state data for multiple filling fractions.

Fillings: n = 1/3 (N_up=1,N_dn=1), 2/3 (N_up=2,N_dn=2), 1 (N_up=3,N_dn=3)
H parameters: [U/t, filling_n]  → h_dim = 2

Output:
  data/hubbard_filling_ssl.npz  -- (gamma_t, gamma_tp, gamma_gs, H)  H=(U,n)
  data/hubbard_filling_gs.npz   -- (U, filling_n, E0, gamma_gs)
"""
import os
import numpy as np
from tqdm import tqdm
from hubbard_ed import (
    build_basis, make_basis_index, build_all_cdagger_c,
    build_hubbard, ground_state, imaginary_time_evolve,
    compute_1rdm_fast, random_fock_state, random_superposition,
)

L = 6
t = 1.0
FILLINGS = [
    (1, 1, 1/3),
    (2, 2, 2/3),
    (3, 3, 1.0),
]
U_VALUES  = np.linspace(0.0, 12.0, 100)   # 100 U per filling → 300 total
# Every 5th U value (indices 4,9,14,...,99) is held out as OOD test set.
# These U values are NEVER seen during SSL pretraining — only used for evaluation.
_TEST_SEL = np.zeros(100, dtype=bool)
_TEST_SEL[4::5] = True   # 20 OOD test values, 80 SSL values
N_INIT    = 10
TAU_STEPS = 30
DELTA_TAU = 0.3
SEED      = 42
DATA_DIR  = "data"

os.makedirs(DATA_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

ssl_gamma_t  = []
ssl_gamma_tp = []
ssl_gamma_gs = []
ssl_H        = []   # (U, filling_n)

gs_U       = []
gs_filling = []
gs_E0      = []
gs_gamma   = []
gs_is_test = []   # True = OOD test (never seen in SSL)

for N_up, N_dn, filling_n in FILLINGS:
    print(f"\nFilling n={filling_n:.3f}  (N_up={N_up}, N_dn={N_dn})")
    basis     = build_basis(L, N_up, N_dn)
    basis_idx = make_basis_index(basis)
    ops       = build_all_cdagger_c(L, basis, basis_idx)
    print(f"  Hilbert dim = {len(basis)}")

    for u_idx, U in enumerate(tqdm(U_VALUES, desc=f"  U/t (n={filling_n:.2f})")):
        H_mat = build_hubbard(L, t, U, basis, basis_idx)

        E0, psi_gs = ground_state(H_mat)
        gamma_gs   = compute_1rdm_fast(psi_gs, ops, L)
        gs_U.append(U)
        gs_filling.append(filling_n)
        gs_E0.append(E0)
        gs_gamma.append(gamma_gs)
        gs_is_test.append(bool(_TEST_SEL[u_idx]))

        # SSL trajectory data only for non-test U values
        if _TEST_SEL[u_idx]:
            continue
        for _ in range(N_INIT):
            psi0 = (random_fock_state(basis, rng) if rng.random() < 0.5
                    else random_superposition(basis, rng))
            traj   = list(imaginary_time_evolve(H_mat, psi0, TAU_STEPS, DELTA_TAU))
            gammas = [compute_1rdm_fast(psi, ops, L) for _, psi in traj]
            for k in range(len(gammas) - 1):
                ssl_gamma_t.append(gammas[k])
                ssl_gamma_tp.append(gammas[k + 1])
                ssl_gamma_gs.append(gamma_gs)
                ssl_H.append([U, filling_n])

ssl_path = os.path.join(DATA_DIR, "hubbard_filling_ssl.npz")
gs_path  = os.path.join(DATA_DIR, "hubbard_filling_gs.npz")

np.savez(ssl_path,
         gamma_t  = np.array(ssl_gamma_t,  dtype=np.float32),
         gamma_tp = np.array(ssl_gamma_tp, dtype=np.float32),
         gamma_gs = np.array(ssl_gamma_gs, dtype=np.float32),
         H        = np.array(ssl_H,        dtype=np.float32))
np.savez(gs_path,
         U        = np.array(gs_U,       dtype=np.float32),
         filling  = np.array(gs_filling, dtype=np.float32),
         E0       = np.array(gs_E0,      dtype=np.float32),
         gamma_gs = np.array(gs_gamma,   dtype=np.float32),
         is_test  = np.array(gs_is_test, dtype=bool))

n_test = sum(gs_is_test)
print(f"\nSaved {len(ssl_gamma_t)} SSL pairs  -> {ssl_path}")
print(f"Saved {len(gs_U)} ground states -> {gs_path}")
print(f"  SSL train: {len(gs_U)-n_test} samples | OOD test: {n_test} samples")
print(f"  Test U values (first 5): {np.array(gs_U)[np.array(gs_is_test)][:5]}")
print(f"1-RDM shape: {np.array(ssl_gamma_t).shape[1:]}")
