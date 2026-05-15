"""Exact diagonalization of 1D Hubbard model + imaginary time evolution + 1-RDM extraction."""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
from itertools import combinations


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def build_basis(L, N_up, N_dn):
    """Build Fock basis as list of (up_tuple, dn_tuple)."""
    up_states = [tuple(sorted(c)) for c in combinations(range(L), N_up)]
    dn_states = [tuple(sorted(c)) for c in combinations(range(L), N_dn)]
    basis = [(u, d) for u in up_states for d in dn_states]
    return basis


def make_basis_index(basis):
    return {state: i for i, state in enumerate(basis)}


# ---------------------------------------------------------------------------
# Fermionic sign helpers
# ---------------------------------------------------------------------------

def _annihilate_sign(occ_tuple, site):
    """Sign from annihilating 'site' in ordered occupied tuple."""
    pos = occ_tuple.index(site)
    return (-1) ** pos


def _create_sign(new_occ_set, site):
    """Sign from creating at 'site' (site not in set before creation)."""
    pos = sum(1 for x in new_occ_set if x < site)
    return (-1) ** pos


# ---------------------------------------------------------------------------
# Single-body operator as sparse matrix: c†_{i,sigma} c_{j,sigma}
# ---------------------------------------------------------------------------

def build_cdagger_c(L, i, j, sigma, basis, basis_idx):
    """
    Build sparse matrix for c†_{i,sigma} c_{j,sigma}.
    sigma=0: spin-up, sigma=1: spin-dn.
    """
    dim = len(basis)
    mat = lil_matrix((dim, dim), dtype=np.float64)
    for col_idx, (up, dn) in enumerate(basis):
        occ = dn if sigma else up
        if j not in occ:
            continue
        if i == j:
            # number operator: maps state to itself with sign +1
            mat[col_idx, col_idx] += 1.0
            continue
        if i in occ:
            continue  # c†_i would give 0
        sign_ann = _annihilate_sign(occ, j)
        new_occ_set = (set(occ) - {j}) | {i}
        sign_cre = _create_sign(new_occ_set - {i}, i)
        sign = sign_ann * sign_cre
        new_occ = tuple(sorted(new_occ_set))
        if sigma == 0:
            new_state = (new_occ, dn)
        else:
            new_state = (up, new_occ)
        row_idx = basis_idx.get(new_state)
        if row_idx is not None:
            mat[row_idx, col_idx] = sign
    return csr_matrix(mat)


def build_all_cdagger_c(L, basis, basis_idx):
    """Precompute all 2L x 2L single-body operators as (sigma, i, j) -> sparse matrix."""
    ops = {}
    for sigma in range(2):
        for i in range(L):
            for j in range(L):
                ops[(sigma, i, j)] = build_cdagger_c(L, i, j, sigma, basis, basis_idx)
    return ops


def compute_1rdm_fast(psi, ops, L):
    """Compute 1-RDM using precomputed operators. Returns (2L,2L) real array."""
    gamma = np.zeros((2 * L, 2 * L))
    psi_norm = psi / np.linalg.norm(psi)
    for sigma in range(2):
        for i in range(L):
            for j in range(L):
                op = ops[(sigma, i, j)]
                val = psi_norm @ (op @ psi_norm)
                gamma[sigma * L + i, sigma * L + j] = val
    return gamma


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

def build_hubbard(L, t, U, basis, basis_idx):
    """Build 1D Hubbard Hamiltonian with PBC."""
    t_bonds = np.full(L, t)
    eps = np.zeros(L)
    return build_hubbard_general(L, t_bonds, eps, U, basis, basis_idx)


def build_hubbard_general(L, t_bonds, eps, U, basis, basis_idx):
    """
    General 1D Hubbard Hamiltonian with PBC.
    t_bonds: (L,) hopping for bond (i, i+1 mod L)
    eps:     (L,) onsite energies
    U:       scalar Hubbard interaction
    """
    dim = len(basis)
    H = lil_matrix((dim, dim), dtype=np.float64)

    for col_idx, (up, dn) in enumerate(basis):
        up_set = set(up)
        dn_set = set(dn)

        # Diagonal: Hubbard U + onsite energies
        H[col_idx, col_idx] += U * len(up_set & dn_set)
        for i in up_set:
            H[col_idx, col_idx] += eps[i]
        for i in dn_set:
            H[col_idx, col_idx] += eps[i]

        # Hopping
        for sigma, occ in [(0, up), (1, dn)]:
            occ_set = set(occ)
            for i in range(L):
                j = (i + 1) % L
                t_ij = t_bonds[i]
                for src, dst in [(i, j), (j, i)]:
                    if src in occ_set and dst not in occ_set:
                        sign_ann = _annihilate_sign(occ, src)
                        new_set = (occ_set - {src}) | {dst}
                        sign_cre = _create_sign(new_set - {dst}, dst)
                        sign = sign_ann * sign_cre
                        new_occ = tuple(sorted(new_set))
                        if sigma == 0:
                            new_state = (new_occ, dn)
                        else:
                            new_state = (up, new_occ)
                        row_idx = basis_idx.get(new_state)
                        if row_idx is not None:
                            H[row_idx, col_idx] -= t_ij * sign
    return csr_matrix(H)


# ---------------------------------------------------------------------------
# Ground state & imaginary time evolution
# ---------------------------------------------------------------------------

def ground_state(H_sparse):
    E, V = eigsh(H_sparse, k=1, which='SA')
    return float(E[0]), V[:, 0]


def imaginary_time_evolve(H_sparse, psi0, tau_steps, delta_tau):
    """Generator: yields (tau, psi_normalized) at each step including tau=0."""
    psi = psi0.astype(np.float64)
    psi /= np.linalg.norm(psi)
    yield 0.0, psi.copy()
    for step in range(tau_steps):
        psi = expm_multiply(-delta_tau * H_sparse, psi)
        norm = np.linalg.norm(psi)
        if norm < 1e-14:
            break
        psi /= norm
        yield (step + 1) * delta_tau, psi.copy()


def random_fock_state(basis, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    psi = np.zeros(len(basis))
    psi[rng.integers(len(basis))] = 1.0
    return psi


def random_superposition(basis, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    psi = rng.standard_normal(len(basis))
    psi /= np.linalg.norm(psi)
    return psi
