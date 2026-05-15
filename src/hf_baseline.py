"""
Hartree-Fock baseline for the 1D Hubbard model at half-filling.

System: L=6, N_up=3, N_dn=3, t=1.0, periodic boundary conditions.

Three HF estimates are computed for each U/t:

  1. E0_HF_PM  -- Paramagnetic (uniform) HF.
                  Fills lowest N_up k-states per spin.
                  E_HF_PM = E_TB_total + U * L * (1/4)
                  where E_TB_total = 2 * sum_{k in occ} eps_k = -8.0 for L=6.

  2. E0_HF_AFM -- Antiferromagnetic (symmetry-broken) HF.
                  Staggered mean field: <n_{i,up}> = 1/2 +/- m on sublattices A/B.
                  Solved by minimising over m in [0, 1/2].
                  At small U this reduces to PM (m=0); at large U a finite m lowers energy.
                  Always lower than or equal to PM.

  3. E0_HF_density -- "Density-functional" estimate.
                      Uses the EXACT 1-RDM gamma_GS from data/hubbard_gs.npz to compute
                      the exact kinetic energy, but approximates the interaction energy
                      at the Hartree level: U * sum_i rho_up(i) * rho_dn(i).
                      This overestimates the energy because HF ignores correlations that
                      suppress double occupancy.

Output: data/hubbard_hf.npz  with arrays  U, E0_HF  (= E0_HF_AFM, the variational minimum).
        Additional arrays E0_HF_PM, E0_HF_AFM, E0_HF_density are also saved for reference.
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar

# ── Physical parameters (must match generate_data.py) ──────────────────────
L    = 6
N_up = 3
N_dn = 3
t    = 1.0
U_VALUES = np.linspace(0.0, 12.0, 250)
DATA_DIR = "data"
# ───────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# 1.  Tight-binding k-states (PBC, 1D)
# ---------------------------------------------------------------------------
k_indices   = np.arange(L)
eps_k       = -2.0 * t * np.cos(2.0 * np.pi * k_indices / L)   # shape (L,)
eps_k_sorted = np.sort(eps_k)                                    # ascending

# Kinetic energy per spin at half-filling: fill lowest N_up levels
E_TB_per_spin = eps_k_sorted[:N_up].sum()          # = -4.0 for L=6
E_TB_total    = 2.0 * E_TB_per_spin                # = -8.0 (both spins)

# ---------------------------------------------------------------------------
# 2.  Paramagnetic HF
# ---------------------------------------------------------------------------
# HF decoupling at half-filling (rho_up = rho_dn = 1/2 per site):
#   <n_{i,up} n_{i,dn}> ~ <n_{i,up}><n_{i,dn}> = (1/2)(1/2) = 1/4
#   E_HF_PM = E_TB_total + U * L * (1/4)
def pm_hf_energy(U):
    return E_TB_total + U * L * 0.25


# ---------------------------------------------------------------------------
# 3.  Antiferromagnetic (variational) HF
# ---------------------------------------------------------------------------
# Mean-field Hamiltonian for spin-up:
#   h^{up}_{ii}  = U * <n_{dn,i}>   where <n_{dn,i}> = 1/2 - m * (-1)^i
#   h^{dn}_{ii}  = U * <n_{up,i}>   where <n_{up,i}> = 1/2 + m * (-1)^i
# (sublattice A has extra up density, sublattice B has extra dn density)
# Off-diagonal: standard nearest-neighbour hopping -t.
#
# Total variational energy:
#   E_HF_AFM(m) = [sum of N_up lowest eps_up] + [sum of N_dn lowest eps_dn]
#                 - double-count correction
#
# The MF Hamiltonian counts U*<n_sigma'><n_sigma> for each spin; the actual
# interaction enters once, so we subtract the mean-field interaction once:
#   E_HF_AFM = E_MF_bands - U * sum_i <n_{up,i}><n_{dn,i}>
#            = E_MF_bands - U * L * (1/4 - m^2)
#
# Rearranged: this equals the standard HF total energy
#   E_HF = E_kin + U * sum_i <n_{up,i}><n_{dn,i}>
# which we can verify is equivalent.

def _build_mf_hamiltonians(U, m):
    """Build spin-up and spin-dn mean-field single-particle Hamiltonians."""
    h_up = np.zeros((L, L), dtype=float)
    h_dn = np.zeros((L, L), dtype=float)
    for i in range(L):
        stagger = (-1) ** i
        h_up[i, i] =  U * (0.5 - m * stagger)   # up feels dn density
        h_dn[i, i] =  U * (0.5 + m * stagger)   # dn feels up density
        j = (i + 1) % L
        h_up[i, j] -= t;  h_up[j, i] -= t
        h_dn[i, j] -= t;  h_dn[j, i] -= t
    return h_up, h_dn


def afm_hf_energy_at_m(U, m):
    """AFM HF energy for a fixed staggered magnetisation m."""
    h_up, h_dn = _build_mf_hamiltonians(U, m)
    eps_up = np.sort(np.linalg.eigvalsh(h_up))[:N_up]
    eps_dn = np.sort(np.linalg.eigvalsh(h_dn))[:N_dn]
    E_bands = eps_up.sum() + eps_dn.sum()
    # subtract double-counted mean-field interaction energy
    # sum_i <n_up_i><n_dn_i> = L/2*(1/2+m)(1/2-m) + L/2*(1/2-m)(1/2+m)
    #                         = L*(1/4 - m^2)
    E_dc = U * L * (0.25 - m * m)
    return E_bands - E_dc


def afm_hf_energy(U):
    """Minimise over m in [0, 0.5] and return the lowest variational energy."""
    if U == 0.0:
        return E_TB_total          # no interaction; m is irrelevant
    result = minimize_scalar(
        lambda m: afm_hf_energy_at_m(U, m),
        bounds=(0.0, 0.5),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return float(result.fun)


# ---------------------------------------------------------------------------
# 4.  Density-functional (Hartree) estimate using exact 1-RDM
# ---------------------------------------------------------------------------
# E_density = E_kin[gamma_exact] + U * sum_i rho_up(i) * rho_dn(i)
#
# E_kin is extracted from the exact 1-RDM via:
#   E_kin = -t * sum_sigma sum_<ij> (gamma[sigma,i,j] + gamma[sigma,j,i])
#           (nearest-neighbour bonds only, PBC)
#
# rho_sigma(i) = gamma[sigma*L + i, sigma*L + i]
#
# This estimate is an UPPER BOUND on the true energy only if the exact density
# happened to satisfy stationarity w.r.t. the HF functional.  In practice it
# overestimates because the Hartree term U*rho_up*rho_dn >> true <n_up n_dn>
# at large U (correlations suppress double occupancy).

def compute_ekin_from_rdm(gamma):
    """Extract kinetic energy from a (2L, 2L) 1-RDM."""
    E_kin = 0.0
    for sigma in range(2):
        off = sigma * L
        for i in range(L):
            j = (i + 1) % L
            E_kin += -t * (gamma[off + i, off + j] + gamma[off + j, off + i])
    return E_kin


def compute_edensity(gamma, U):
    """Hartree estimate using exact gamma."""
    E_kin = compute_ekin_from_rdm(gamma)
    rho_up = np.array([gamma[i, i]     for i in range(L)], dtype=float)
    rho_dn = np.array([gamma[L + i, L + i] for i in range(L)], dtype=float)
    E_int  = U * np.sum(rho_up * rho_dn)
    return E_kin + E_int


# ---------------------------------------------------------------------------
# Load exact ground-state data
# ---------------------------------------------------------------------------
gs_path = os.path.join(DATA_DIR, "hubbard_gs.npz")
if not os.path.exists(gs_path):
    raise FileNotFoundError(
        f"Exact ground-state data not found at {gs_path}. "
        "Run generate_data.py first."
    )

gs_data    = np.load(gs_path)
U_exact    = gs_data["U"].astype(float)          # shape (250,)
E0_exact   = gs_data["E0"].astype(float)          # shape (250,)
gamma_exact = gs_data["gamma_gs"].astype(float)   # shape (250, 2L, 2L)

# Verify U grids match (they should, both use np.linspace(0, 12, 250))
if not np.allclose(U_exact, U_VALUES, atol=1e-4):
    print("WARNING: U grid in hubbard_gs.npz differs from U_VALUES in this script.")


# ---------------------------------------------------------------------------
# Compute all HF estimates
# ---------------------------------------------------------------------------
print("Computing HF baselines for 250 U/t values ...")

E0_HF_PM_arr      = np.array([pm_hf_energy(U)        for U in U_VALUES], dtype=np.float64)
E0_HF_AFM_arr     = np.array([afm_hf_energy(U)       for U in U_VALUES], dtype=np.float64)
E0_HF_density_arr = np.array(
    [compute_edensity(gamma_exact[i], float(U_VALUES[i])) for i in range(len(U_VALUES))],
    dtype=np.float64,
)

# Primary HF result: lowest variational energy = min(PM, AFM)
# (AFM <= PM by construction since m=0 recovers PM)
E0_HF_arr = np.minimum(E0_HF_PM_arr, E0_HF_AFM_arr)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
out_path = os.path.join(DATA_DIR, "hubbard_hf.npz")
np.savez(
    out_path,
    U              = U_VALUES.astype(np.float32),
    E0_HF          = E0_HF_arr.astype(np.float32),
    E0_HF_PM       = E0_HF_PM_arr.astype(np.float32),
    E0_HF_AFM      = E0_HF_AFM_arr.astype(np.float32),
    E0_HF_density  = E0_HF_density_arr.astype(np.float32),
)
print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
# Pick a representative subset: U = 0, 1, 2, 3, 4, 6, 8, 10, 12
U_print = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]

print()
print(f"{'U/t':>5}  {'E0_exact':>10}  {'E0_HF_PM':>10}  {'E0_HF_AFM':>11}  "
      f"{'E0_HF_dens':>12}  {'dE_PM':>8}  {'dE_AFM':>8}")
print("-" * 82)
for U_val in U_print:
    idx = int(np.argmin(np.abs(U_VALUES - U_val)))
    U_v    = U_VALUES[idx]
    E_ex   = E0_exact[idx]
    E_pm   = E0_HF_PM_arr[idx]
    E_afm  = E0_HF_AFM_arr[idx]
    E_dens = E0_HF_density_arr[idx]
    dE_pm  = E_pm  - E_ex
    dE_afm = E_afm - E_ex
    print(f"{U_v:5.1f}  {E_ex:10.4f}  {E_pm:10.4f}  {E_afm:11.4f}  "
          f"{E_dens:12.4f}  {dE_pm:+8.4f}  {dE_afm:+8.4f}")
print()
print("dE_PM  = E0_HF_PM  - E0_exact  (positive -> HF overestimates energy)")
print("dE_AFM = E0_HF_AFM - E0_exact  (positive -> HF overestimates energy)")
print()
print("Physics notes:")
print(f"  Tight-binding energies (L={L}, PBC): {eps_k_sorted}")
print(f"  E_TB per spin = {E_TB_per_spin:.4f}   E_TB_total = {E_TB_total:.4f}")
print(f"  PM HF slope dE/dU = L/4 = {L/4:.2f}")
AFM_trans = U_VALUES[np.where(E0_HF_AFM_arr < E0_HF_PM_arr - 1e-6)[0]]
if len(AFM_trans) > 0:
    print(f"  AFM HF onset (E_AFM < E_PM) at U/t ≈ {AFM_trans[0]:.2f}")
print(f"  Saved arrays: U, E0_HF, E0_HF_PM, E0_HF_AFM, E0_HF_density")
print(f"  All arrays shape: ({len(U_VALUES)},)")
