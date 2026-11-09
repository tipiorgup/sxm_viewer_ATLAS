"""
config.py — configuration object for the rigid-body relaxation pass.

All defaults come from src/constants.py (which read from environment
variables), so a run can be tuned without editing source:

    RIGID_RELAX_K_PHI=2.0 RIGID_RELAX_PHI_REF_BETA=-70 python module_B.py ...
"""

from dataclasses import dataclass

from ...constants import (
    RIGID_RELAX_ENABLE,
    RIGID_RELAX_CO_BOND,
    RIGID_RELAX_COC_ANGLE,
    RIGID_RELAX_K_BOND,
    RIGID_RELAX_K_ANGLE,
    RIGID_RELAX_K_PHI,
    RIGID_RELAX_K_PSI,
    RIGID_RELAX_PHI_REF_ALPHA,
    RIGID_RELAX_PHI_REF_BETA,
    RIGID_RELAX_PSI_REF,
    RIGID_RELAX_MAX_SWEEPS,
    RIGID_RELAX_STEP,
    RIGID_RELAX_MAX_ROT,
    RIGID_RELAX_STEP_TRANS,
    RIGID_RELAX_MAX_TRANS,
    RIGID_RELAX_FD_DELTA,
    RIGID_RELAX_FD_DELTA_TRANS,
    RIGID_RELAX_GRAD_TOL,
    RIGID_RELAX_ENERGY_TOL,
)


@dataclass
class RigidRelaxConfig:
    """
    Full configuration for rigid_relax.relax_chain.

    Geometry (spring rest values)
    -----------------------------
    co_bond : float  (Å)
        Glycosidic C-O bond spring rest length (also seeds the free oxygen).
    coc_angle : float  (degrees)
        Glycosidic C-O-C angle spring rest value.

    Spring weights
    --------------
    k_bond : float
        Weight on each C-O bond spring (donor-C···O and O···acceptor-C).
    k_angle : float
        Weight on the C-O-C angle spring.
    k_phi : float
        Weight on the glycosidic φ dihedral restraint (α/β rotation).
    k_psi : float
        Weight on the ψ dihedral restraint (0 disables it).

    Reference dihedrals (keyed by anomeric configuration)
    -----------------------------------------------------
    phi_ref_alpha, phi_ref_beta : float  (degrees)
    psi_ref : float  (degrees)

    Solver
    ------
    max_sweeps : int
    step : float
        Initial rotation step scale (rad per unit gradient).
    max_rot : float  (rad)
        Per-bead per-sweep rotation cap.
    step_trans : float
        Initial translation step scale for free oxygens (Å per unit gradient).
    max_trans : float  (Å)
        Per-free-atom per-sweep translation cap.
    fd_delta : float  (rad)
        Finite-difference angle for the bead torque estimate.
    fd_delta_trans : float  (Å)
        Finite-difference step for the free-oxygen translation gradient.
    grad_tol, energy_tol : float
        Convergence thresholds.
    verbose : bool
    """

    enable: bool = bool(RIGID_RELAX_ENABLE)

    co_bond: float = RIGID_RELAX_CO_BOND
    coc_angle: float = RIGID_RELAX_COC_ANGLE

    k_bond: float = RIGID_RELAX_K_BOND
    k_angle: float = RIGID_RELAX_K_ANGLE
    k_phi: float = RIGID_RELAX_K_PHI
    k_psi: float = RIGID_RELAX_K_PSI

    phi_ref_alpha: float = RIGID_RELAX_PHI_REF_ALPHA
    phi_ref_beta: float = RIGID_RELAX_PHI_REF_BETA
    psi_ref: float = RIGID_RELAX_PSI_REF

    max_sweeps: int = RIGID_RELAX_MAX_SWEEPS
    step: float = RIGID_RELAX_STEP
    max_rot: float = RIGID_RELAX_MAX_ROT
    step_trans: float = RIGID_RELAX_STEP_TRANS
    max_trans: float = RIGID_RELAX_MAX_TRANS
    fd_delta: float = RIGID_RELAX_FD_DELTA
    fd_delta_trans: float = RIGID_RELAX_FD_DELTA_TRANS
    grad_tol: float = RIGID_RELAX_GRAD_TOL
    energy_tol: float = RIGID_RELAX_ENERGY_TOL

    verbose: bool = True

    def phi_ref_for(self, anomeric: str) -> float:
        """Reference φ (degrees) for an 'alpha'/'beta' anomeric label."""
        a = (anomeric or 'beta').strip().lower()
        return self.phi_ref_alpha if a == 'alpha' else self.phi_ref_beta
