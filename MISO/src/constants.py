"""
constants.py — numerical defaults for the optimisation pipeline.

Every constant reads from an environment variable first, then falls back to
the hard-coded default.  This lets the job script (module_B_scratch.job)
override values without touching Python source:

    export DEFAULT_GRAVITY=0.055
    python module_B.py ...

Groups
------
Geometry tolerances
    Bond-length and bond-angle thresholds used during ring-integrity checks.

Ring detection
    PYRANOSE_RING_SIZE: expected atom count for a pyranose (6).

Optimisation phases
    Step counts and step sizes for the three optimisation phases:
    constrained minimisation → slab compression MD → free minimisation.

Slab parameters
    Control the rising-slab compression model (phase 2).

Gravity / STM surface potential
    DEFAULT_GRAVITY is the base downward force per unit mass.
    When an STM grid is provided the gravity is spatially modulated by the
    normalised surface height; DEFAULT_GRAVITY still sets the overall scale.

Dynamics (Langevin MD)
    Timestep, friction, velocity and force caps for the MD integrator.

Ring rotation
    Friction and stiffness for the dual-mode ring constraint:
    constrained rings (CD-like, fixed normal) vs free rings (3D motion).

Convergence
    Energy, force and RMSD thresholds checked over a sliding window.
"""

import os


def get_env_float(var_name, default):
    """Return float from environment variable, or *default* if not set."""
    try:
        value = os.environ.get(var_name)
        if value is not None:
            return float(value)
        return default
    except (ValueError, TypeError):
        return default


def get_env_int(var_name, default):
    """Return int from environment variable, or *default* if not set."""
    try:
        value = os.environ.get(var_name)
        if value is not None:
            return int(value)
        return default
    except (ValueError, TypeError):
        return default


# ============================================================================
# Geometry tolerances
# ============================================================================

DEFAULT_BOND_TOLERANCE = get_env_float('DEFAULT_BOND_TOLERANCE', 0.10)
"""Å — maximum allowed deviation in ring bond length before a ring is
considered deformed and Phase 2 reverts it to the last valid geometry.
Overridable via the DEFAULT_BOND_TOLERANCE env var (e.g. in the job script)."""

DEFAULT_ANGLE_TOLERANCE = get_env_float('DEFAULT_ANGLE_TOLERANCE', 10)
"""degrees — maximum allowed deviation in ring bond angle before a ring is
considered deformed. Overridable via the DEFAULT_ANGLE_TOLERANCE env var."""

EPSILON = get_env_float('EPSILON', 1e-8)
"""Small number used to guard against division by zero in vector operations."""

MIN_ATOM_DISTANCE = get_env_float('MIN_ATOM_DISTANCE', 0.5)
"""Å — minimum allowed distance between any two heavy atoms; pairs closer
than this are reported (and optionally fixed) by check_geometry."""

DEFAULT_MIN_SEPARATION = get_env_float('DEFAULT_MIN_SEPARATION', 0.8)
"""Å — target minimum separation used by fix_overlapping_atoms."""

# ============================================================================
# Ring detection
# ============================================================================

PYRANOSE_RING_SIZE = get_env_int('PYRANOSE_RING_SIZE', 6)
"""Number of atoms in a pyranose ring (always 6 for hexose sugars)."""

# ============================================================================
# Optimisation phase parameters
# ============================================================================

DEFAULT_RELAXATION_STEPS = get_env_int('DEFAULT_RELAXATION_STEPS', 50)
"""Phase 1: number of iterative minimisation update cycles
(each cycle = 4 force-field minimisation steps)."""

DEFAULT_COMPRESSION_STEPS = get_env_int('DEFAULT_COMPRESSION_STEPS', 100)
"""Phase 2: maximum MD steps for slab compression.
Early exit occurs if the molecule height drops below the slab ceiling."""

DEFAULT_STEP_SIZE = get_env_float('DEFAULT_STEP_SIZE', 5e-3)
"""Å — position update step size used in custom gradient-descent passes."""

DEFAULT_CHECK_RINGS_INTERVAL = get_env_int('DEFAULT_CHECK_RINGS_INTERVAL', 50)
"""Phase 2 MD steps between ring-integrity checks."""

DEFAULT_MINIMIZE_INTERVAL = get_env_int('DEFAULT_MINIMIZE_INTERVAL', 5)
"""Phase 2 MD steps between periodic force-field minimisation calls."""

DEFAULT_MINIMIZE_ITERATIONS = get_env_int('DEFAULT_MINIMIZE_ITERATIONS', 10)
"""Force-field minimisation steps per periodic minimisation call in phase 2."""

DEFAULT_IMAGE_INTERVAL = get_env_int('DEFAULT_IMAGE_INTERVAL', 50)
"""Steps between trajectory frame writes (and optional PNG saves)."""

DEFAULT_PHASE1_KICK_INTERVAL = get_env_int('DEFAULT_PHASE1_KICK_INTERVAL', 25)
"""Phase 1: number of minimisation update cycles between stochastic kicks.
Only active when --phase1_kicks is passed."""

DEFAULT_PHASE1_KICK_AMPLITUDE = get_env_float('DEFAULT_PHASE1_KICK_AMPLITUDE', 0.05)
"""Å — magnitude of random displacement applied per atom during phase 1 kicks."""

# ============================================================================
# Slab parameters (phase 2 compression)
# ============================================================================

DEFAULT_SLAB_STEP_SIZE = get_env_float('DEFAULT_SLAB_STEP_SIZE', 0.05)
"""Å — how far the compression slab rises each interval."""

DEFAULT_SLAB_STEP_INTERVAL = get_env_int('DEFAULT_SLAB_STEP_INTERVAL', 10)
"""MD steps between each slab position increment."""

DEFAULT_SLAB_FORCE_SCALE = get_env_float('DEFAULT_SLAB_FORCE_SCALE', 50.0)
"""Scale factor applied to all slab forces (floor, ceiling and rising slab)."""

# ============================================================================
# Gravity / STM surface potential
# ============================================================================

DEFAULT_GRAVITY = get_env_float('DEFAULT_GRAVITY', 2.0)
"""Base gravitational acceleration (kcal mol⁻¹ Å⁻²) applied to all atoms
above z = 0.  When molecule_data_dict is supplied the force is spatially
modulated by brightness; when an STM .npz grid is supplied it is further
modulated by the normalised STM height."""

# ============================================================================
# Ring COM constraints
# ============================================================================

DEFAULT_CONSTRAINED_RING_COM_LIMIT = get_env_int('DEFAULT_CONSTRAINED_RING_COM_LIMIT', 50)
"""Maximum allowed COM drift (Å) for rings in constrained (CD) mode before
a restoring force is applied."""

DEFAULT_FREE_RING_COM_LIMIT = get_env_int('DEFAULT_FREE_RING_COM_LIMIT', 50)
"""Maximum allowed COM drift (Å) for rings in free (3D) mode."""

# ============================================================================
# Langevin MD dynamics
# ============================================================================

DEFAULT_TIMESTEP = get_env_float('DEFAULT_TIMESTEP', 0.1)
"""fs — Langevin integrator timestep."""

DEFAULT_FRICTION = get_env_float('DEFAULT_FRICTION', 0.1)
"""Langevin friction coefficient (ps⁻¹)."""

DEFAULT_MAX_FORCE = get_env_float('DEFAULT_MAX_FORCE', 100.0)
"""kcal mol⁻¹ Å⁻¹ — per-atom force magnitude cap applied before velocity
update to prevent atoms from flying out of the box."""

DEFAULT_MAX_VELOCITY = get_env_float('DEFAULT_MAX_VELOCITY', 1.0)
"""Å fs⁻¹ — per-atom velocity magnitude cap."""

DEFAULT_MAX_DISPLACEMENT = get_env_float('DEFAULT_MAX_DISPLACEMENT', 0.1)
"""Å — maximum allowed displacement per MD step (unused currently but
reserved for a future clamp)."""

TEMPERATURE_K = get_env_float('TEMPERATURE_K', 315)
"""K — simulation temperature for the Langevin random force."""

BOLTZMANN_KCAL = get_env_float('BOLTZMANN_KCAL', 0.001987)
"""kcal mol⁻¹ K⁻¹ — Boltzmann constant in kcal units."""

# ============================================================================
# Ring rotation dynamics
# ============================================================================

DEFAULT_RING_ROTATION_FRICTION = get_env_float('DEFAULT_RING_ROTATION_FRICTION', 0.1)
"""Rotational friction for ring angular-velocity damping."""

DEFAULT_NORMAL_TOLERANCE_DEG = get_env_float('DEFAULT_NORMAL_TOLERANCE_DEG', 5.0)
"""degrees — maximum angular deviation of the ring normal from its reference
direction before the normal-restoring torque activates (constrained mode)."""

DEFAULT_NORMAL_STIFFNESS = get_env_float('DEFAULT_NORMAL_STIFFNESS', 100.0)
"""kcal mol⁻¹ rad⁻² — stiffness of the normal-orientation constraint."""

# ============================================================================
# Constrained ring parameters (CD / surface-flat mode)
# ============================================================================

CONSTRAINED_MAX_TRANSLATION = get_env_float('CONSTRAINED_MAX_TRANSLATION', 2.0)
"""Å — maximum COM drift allowed before translation restraint activates
for rings in constrained mode."""

CONSTRAINED_TRANSLATION_STIFFNESS = get_env_float('CONSTRAINED_TRANSLATION_STIFFNESS', 500.0)
"""kcal mol⁻¹ Å⁻² — stiffness of the COM translation restraint."""

# ============================================================================
# Free ring parameters (3D mode)
# ============================================================================

FREE_MAX_TRANSLATION = get_env_float('FREE_MAX_TRANSLATION', 5.0)
"""Å — maximum COM drift for rings in free (3D) mode."""

FREE_TRANSLATION_STIFFNESS = get_env_float('FREE_TRANSLATION_STIFFNESS', 50.0)
"""kcal mol⁻¹ Å⁻² — stiffness of the soft COM restraint in free mode."""

# ============================================================================
# Ring matching (pipeline.extract_initial_ring_coms_from_monomers)
# ============================================================================

RING_MATCHING_DISTANCE_THRESHOLD = get_env_float('RING_MATCHING_DISTANCE_THRESHOLD', 1.0)
"""Å — maximum distance between a detected ring COM and a monomer COM for
them to be considered the same ring during the matching step."""

# ============================================================================
# Convergence criteria (phase 2 and phase 3)
# ============================================================================

CONVERGENCE_ENERGY_THRESHOLD = 0.01
"""kcal mol⁻¹ — energy fluctuation (max - min) over the convergence window
must fall below this for convergence to be declared."""

CONVERGENCE_FORCE_THRESHOLD = 0.1
"""kcal mol⁻¹ Å⁻¹ — maximum per-atom force over the window must fall below
this for convergence to be declared."""

CONVERGENCE_RMSD_THRESHOLD = 0.001
"""Å — maximum per-step RMSD over the window must fall below this."""

CONVERGENCE_WINDOW = 10
"""Number of recent steps checked for all convergence criteria simultaneously."""

# ============================================================================
# Debug output
# ============================================================================

if os.environ.get('DEBUG_CONSTANTS', '0') == '1':
    print("\n" + "=" * 70)
    print("LOADED CONSTANTS")
    print("=" * 70)
    print(f"DEFAULT_RELAXATION_STEPS: {DEFAULT_RELAXATION_STEPS}")
    print(f"DEFAULT_COMPRESSION_STEPS: {DEFAULT_COMPRESSION_STEPS}")
    print(f"DEFAULT_GRAVITY: {DEFAULT_GRAVITY}")
    print(f"DEFAULT_CHECK_RINGS_INTERVAL: {DEFAULT_CHECK_RINGS_INTERVAL}")
    print("=" * 70 + "\n")