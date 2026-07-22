"""
config.py — dataclasses and configuration objects for the MD optimisation.

Separating config from ring_functions.py means pipeline.py can import
OptimizationConfig without pulling in all of the ring-dynamics machinery.

Classes
-------
OptimizationConfig
    Full configuration for optimize_with_slab_and_rings.
RingConstraintConfig
    Parameters for the dual-mode ring constraint (constrained vs free).
RingRotationUnit
    Per-ring state tracked during the MD simulation.
RingIntegrityDetail
    Result of a single ring-integrity check.
RingReferenceGeometry
    Reference bond lengths and angles for a ring, used by integrity checks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from ...constants import (
    DEFAULT_RELAXATION_STEPS,
    DEFAULT_COMPRESSION_STEPS,
    DEFAULT_SLAB_STEP_SIZE,
    DEFAULT_SLAB_STEP_INTERVAL,
    DEFAULT_SLAB_FORCE_SCALE,
    DEFAULT_GRAVITY,
    DEFAULT_STEP_SIZE,
    DEFAULT_TIMESTEP,
    DEFAULT_FRICTION,
    DEFAULT_BOND_TOLERANCE,
    DEFAULT_ANGLE_TOLERANCE,
    DEFAULT_CHECK_RINGS_INTERVAL,
    DEFAULT_CONSTRAINED_RING_COM_LIMIT,
    DEFAULT_FREE_RING_COM_LIMIT,
    DEFAULT_MINIMIZE_INTERVAL,
    DEFAULT_MINIMIZE_ITERATIONS,
    DEFAULT_IMAGE_INTERVAL,
    DEFAULT_PHASE1_KICK_INTERVAL,
    DEFAULT_PHASE1_KICK_AMPLITUDE,
    DEFAULT_MAX_FORCE,
    DEFAULT_MAX_VELOCITY,
    DEFAULT_RING_ROTATION_FRICTION,
    DEFAULT_NORMAL_TOLERANCE_DEG,
    DEFAULT_NORMAL_STIFFNESS,
    CONSTRAINED_MAX_TRANSLATION,
    CONSTRAINED_TRANSLATION_STIFFNESS,
    FREE_MAX_TRANSLATION,
    FREE_TRANSLATION_STIFFNESS,
)


@dataclass
class OptimizationConfig:
    """
    Full configuration for optimize_with_slab_and_rings.

    All parameters have defaults taken from src/constants.py, which in turn
    read from environment variables.  Override any parameter when constructing
    the config object:

        config = OptimizationConfig(compression_steps=2000, gravity=0.1)

    Phase parameters
    ----------------
    relaxation_steps : int
        Phase 1 — number of iterative minimisation update cycles.
    compression_steps : int
        Phase 2 — maximum Langevin MD steps for slab compression.

    Slab parameters
    ---------------
    slab_step_size : float  (Å)
        How far the compression slab rises per interval.
    slab_step_interval : int
        MD steps between each slab position increment.
    slab_force_scale : float
        Overall scale factor for slab forces (floor, ceiling, rising slab).

    Physics
    -------
    gravity : float  (kcal mol⁻¹ Å⁻²)
        Base gravitational acceleration applied to all atoms above z=0.
    step_size : float  (Å)
        Gradient-descent step size (not used in Langevin MD phases).
    timestep : float  (fs)
        Langevin integrator timestep.
    friction : float  (ps⁻¹)
        Langevin friction coefficient.

    Ring monitoring
    ---------------
    ring_tolerance_bond : float  (Å)
    ring_tolerance_angle : float  (degrees)
    check_rings_interval : int
    constrained_ring_com_limit : float  (Å)
    free_ring_com_limit : float  (Å)

    Minimisation
    ------------
    minimize_interval : int
        Phase 2 MD steps between periodic force-field minimisation calls.
    minimize_iterations : int
        Force-field steps per periodic minimisation call.

    Force limits
    ------------
    max_force : float  (kcal mol⁻¹ Å⁻¹)
    max_velocity : float  (Å fs⁻¹)

    Constraints
    -----------
    fixed_atoms : list[int] | None
        Atom indices that are held fixed throughout all phases.
    reference_normals : dict[int, np.ndarray] | None
        Mapping ring_idx → reference normal for constrained rings.

    Ring rotation
    -------------
    enable_ring_rotation : bool
    ring_rotation_friction : float

    Convergence
    -----------
    enable_convergence : bool
    convergence_energy_threshold : float  (kcal mol⁻¹)
    convergence_force_threshold : float  (kcal mol⁻¹ Å⁻¹)
    convergence_rmsd_threshold : float  (Å)
    convergence_window : int

    Output
    ------
    save_images : bool
        Save PNG frames during optimisation (slow).
    image_interval : int
    output_name : str
        Prefix for output files and trajectory.
    """

    # Phase durations
    relaxation_steps: int = DEFAULT_RELAXATION_STEPS
    compression_steps: int = DEFAULT_COMPRESSION_STEPS

    # Slab
    slab_step_size: float = DEFAULT_SLAB_STEP_SIZE
    slab_step_interval: int = DEFAULT_SLAB_STEP_INTERVAL
    slab_force_scale: float = DEFAULT_SLAB_FORCE_SCALE

    # Physics
    gravity: float = DEFAULT_GRAVITY
    step_size: float = DEFAULT_STEP_SIZE
    timestep: float = DEFAULT_TIMESTEP
    friction: float = DEFAULT_FRICTION

    # Ring monitoring
    ring_tolerance_bond: float = DEFAULT_BOND_TOLERANCE
    ring_tolerance_angle: float = DEFAULT_ANGLE_TOLERANCE
    check_rings_interval: int = DEFAULT_CHECK_RINGS_INTERVAL
    constrained_ring_com_limit: float = DEFAULT_CONSTRAINED_RING_COM_LIMIT
    free_ring_com_limit: float = DEFAULT_FREE_RING_COM_LIMIT

    # Minimisation
    minimize_interval: int = DEFAULT_MINIMIZE_INTERVAL
    minimize_iterations: int = DEFAULT_MINIMIZE_ITERATIONS

    # Force limits
    max_force: float = DEFAULT_MAX_FORCE
    max_velocity: float = DEFAULT_MAX_VELOCITY

    # Constraints
    fixed_atoms: Optional[List[int]] = None
    reference_normals: Optional[Dict[int, np.ndarray]] = None

    # Glycosidic trans restraint: list of (i, j, k, l) heavy-atom dihedrals held
    # near 180° during phases 2 & 3 (instead of freezing these atoms in place,
    # which would pin the bond length and prevent a stretched bond from relaxing).
    torsion_constraints: Optional[List[tuple]] = None
    torsion_min_deg: float = 175.0       # window keeps trans but lets phase 3 settle
    torsion_max_deg: float = 185.0
    torsion_force_constant: float = 1.0e4    # stiff enough to dominate MMFF torsion

    # Ring rotation
    enable_ring_rotation: bool = True
    ring_rotation_friction: float = DEFAULT_RING_ROTATION_FRICTION

    # Phase 1 stochastic kicks
    # When True, small random displacements are applied to free atoms every
    # kick_interval updates during phase 1 minimization — same escape mechanism
    # phase 2 uses via velocity reinitialization.  Leave False for LPS runs
    # where the structure is already well-positioned; enable for glycopeptides
    # with large initial torsional clashes between bonded sugar rings.
    enable_phase1_kicks: bool = False
    phase1_kick_interval: int = DEFAULT_PHASE1_KICK_INTERVAL
    phase1_kick_amplitude: float = DEFAULT_PHASE1_KICK_AMPLITUDE

    # Convergence
    enable_convergence: bool = True
    convergence_energy_threshold: float = 0.01
    convergence_force_threshold: float = 0.1
    convergence_rmsd_threshold: float = 0.001
    convergence_window: int = 10

    # Output
    save_images: bool = False
    image_interval: int = DEFAULT_IMAGE_INTERVAL
    output_name: str = "optimization"


@dataclass
class RingConstraintConfig:
    """
    Parameters for the dual-mode ring constraint applied during phase 2 MD.

    Constrained rings (CD mode)
        Ring normal is held close to a reference direction.
        COM translation is tightly restrained.

    Free rings (3D mode)
        Ring can rotate and translate freely; only a soft COM restraint is
        applied to prevent drift too far from the experimental position.
    """

    # Constrained ring parameters
    constrained_friction: float = DEFAULT_RING_ROTATION_FRICTION
    constrained_normal_tolerance: float = DEFAULT_NORMAL_TOLERANCE_DEG
    constrained_normal_stiffness: float = DEFAULT_NORMAL_STIFFNESS
    constrained_max_translation: float = CONSTRAINED_MAX_TRANSLATION
    constrained_translation_stiffness: float = CONSTRAINED_TRANSLATION_STIFFNESS

    # Free ring parameters
    free_max_translation: float = FREE_MAX_TRANSLATION
    free_translation_stiffness: float = FREE_TRANSLATION_STIFFNESS

    # General
    constrain_ring_normal: bool = True
    debug: bool = False


@dataclass
class RingRotationUnit:
    """
    Per-ring state object tracked throughout the MD simulation.

    Created by initialize_ring_rotation_units and updated at each MD step
    by apply_constrained_ring_dynamics / apply_free_ring_dynamics.
    """

    atoms: List[int]
    """Atom indices belonging to this ring."""

    com_initial: np.ndarray
    """Initial centre of mass (experimental STM position)."""

    normal_fixed: np.ndarray
    """Reference ring normal (unit vector); used in constrained mode."""

    angular_velocity: float = 0.0
    """Current angular velocity (rad / fs)."""

    moment_inertia: float = 0.0
    """Moment of inertia around the ring normal (amu·Å²)."""

    ring_id: int = 0
    total_rotation: float = 0.0
    """Cumulative rotation since start of simulation (radians)."""

    is_constrained: bool = False
    """True → constrained (CD) mode; False → free (3D) mode."""

    def to_dict(self) -> dict:
        """Convert to plain dict for compatibility with existing code."""
        return {
            'atoms': self.atoms,
            'com_initial': self.com_initial,
            'normal_fixed': self.normal_fixed,
            'angular_velocity': self.angular_velocity,
            'moment_inertia': self.moment_inertia,
            'ring_id': self.ring_id,
            'total_rotation': self.total_rotation,
            'is_constrained': self.is_constrained,
        }


@dataclass
class RingIntegrityDetail:
    """Result of a single ring-integrity check."""

    ring_id: int
    atoms: List[int]
    max_bond_dev: float
    """Maximum bond-length deviation from reference (Å)."""
    max_angle_dev: float
    """Maximum bond-angle deviation from reference (degrees)."""
    ok: bool
    """True if both deviations are within tolerance."""


@dataclass
class RingReferenceGeometry:
    """
    Reference internal geometry for one ring.

    Populated by get_ring_reference_geometry at the start of optimisation
    and used throughout to detect ring deformation.
    """

    atoms: List[int]
    bonds: List[tuple]
    """List of (atom1_idx, atom2_idx, reference_length_Å)."""
    angles: List[tuple]
    """List of (atom1_idx, atom2_idx, atom3_idx, reference_angle_deg)."""