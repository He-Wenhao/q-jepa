"""
Generate self-supervised training data and ground-state labels for the 1D Hubbard model.

Output: data/
  hubbard_ssl.npz   -- self-supervised pairs (gamma_t, gamma_t+dt, U_over_t)
  hubbard_gs.npz    -- ground-state (E0, gamma_GS) per U/t value
"""
import os
import numpy as np
from tqdm import tqdm
from hubbard_ed import (
    build_basis, make_basis_index, build_hubbard, build_all_cdagger_c,
    compute_1rdm_fast, ground_state, imaginary_time_evolve,
    random_fock_state, random_superposition,
)

# ── Config ──────────────────────────────────────────────────────────────────
L = 6           # sites
N_up = 3        # half-filling
N_dn = 3
t = 1.0         # hopping (energy unit)

U_VALUES = np.linspace(0.0, 12.0, 250)  # 250 U/t values
N_INIT = 10      # initial states per U value
TAU_STEPS = 30   # imaginary-time steps per trajectory
DELTA_TAU = 0.3  # step size

SEED = 42
DATA_DIR = "data"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# Build basis once (same for all U)
print(f"Building basis: L={L}, N_up={N_up}, N_dn={N_dn}")
basis = build_basis(L, N_up, N_dn)
basis_idx = make_basis_index(basis)
dim = len(basis)
print(f"  Hilbert space dim = {dim}")

print("Precomputing single-body operators...")
ops = build_all_cdagger_c(L, basis, basis_idx)
rdm_dim = 2 * L  # 1-RDM is (rdm_dim x rdm_dim)

# ── Self-supervised data ─────────────────────────────────────────────────
ssl_gamma_t   = []   # 1-RDM at tau
ssl_gamma_tp  = []   # 1-RDM at tau + delta_tau
ssl_gamma_gs  = []   # ground-state 1-RDM (target for convergence loss)
ssl_U         = []   # U/t label

# ── Ground-state data ────────────────────────────────────────────────────
gs_U    = []
gs_E0   = []
gs_gamma = []

for U in tqdm(U_VALUES, desc="U/t values"):
    H = build_hubbard(L, t, U, basis, basis_idx)

    # Ground state
    E0, psi_gs = ground_state(H)
    gamma_gs = compute_1rdm_fast(psi_gs, ops, L)
    gs_U.append(U)
    gs_E0.append(E0)
    gs_gamma.append(gamma_gs)

    # Self-supervised trajectories
    for _ in range(N_INIT):
        # Mix Fock states and random superpositions
        if rng.random() < 0.5:
            psi0 = random_fock_state(basis, rng)
        else:
            psi0 = random_superposition(basis, rng)

        traj = list(imaginary_time_evolve(H, psi0, TAU_STEPS, DELTA_TAU))
        gammas = [compute_1rdm_fast(psi, ops, L) for _, psi in traj]

        for k in range(len(gammas) - 1):
            ssl_gamma_t.append(gammas[k])
            ssl_gamma_tp.append(gammas[k + 1])
            ssl_gamma_gs.append(gamma_gs)
            ssl_U.append(U)

ssl_gamma_t  = np.array(ssl_gamma_t,  dtype=np.float32)
ssl_gamma_tp = np.array(ssl_gamma_tp, dtype=np.float32)
ssl_gamma_gs = np.array(ssl_gamma_gs, dtype=np.float32)
ssl_U        = np.array(ssl_U,        dtype=np.float32)

gs_U     = np.array(gs_U,    dtype=np.float32)
gs_E0    = np.array(gs_E0,   dtype=np.float32)
gs_gamma = np.array(gs_gamma, dtype=np.float32)

ssl_path = os.path.join(DATA_DIR, "hubbard_ssl.npz")
gs_path  = os.path.join(DATA_DIR, "hubbard_gs.npz")

np.savez(ssl_path, gamma_t=ssl_gamma_t, gamma_tp=ssl_gamma_tp, gamma_gs=ssl_gamma_gs, U=ssl_U)
np.savez(gs_path,  U=gs_U, E0=gs_E0, gamma_gs=gs_gamma)

print(f"\nSaved {len(ssl_gamma_t)} SSL pairs -> {ssl_path}")
print(f"Saved {len(gs_U)} ground states   -> {gs_path}")
print(f"1-RDM shape: {ssl_gamma_t.shape[1:]}")
