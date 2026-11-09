import numpy as np
import time
from rdkit.Chem import AllChem
import os
from rdkit import Chem
from ..geometry.geometry_utils import (
    cap_vectors, get_positions, set_positions,
    get_ring_normal_from_positions, calculate_center_of_mass,
    get_ring_normal_absolute, calculate_moment_of_inertia
)
from .ring_functions import (
    detect_pyranose_rings, get_ring_reference_geometry,
    check_and_update_rings, check_ring_integrity,
    apply_ring_constraints_dual_mode,
)
from .config import OptimizationConfig, RingConstraintConfig, RingRotationUnit
from .utils import (
    check_convergence, calculate_rmsd, save_frame,
    fix_valence_issues, save_molecule, 
)
from ...constants import TEMPERATURE_K, BOLTZMANN_KCAL
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

#Todo: Run a last soft MD after phase 1 and 2 

def setup_force_field(mol):
    """
    Setup force field for molecule.
    
    Returns:
        Tuple of (props, use_mmff: bool)
    """
    props = AllChem.MMFFGetMoleculeProperties(mol)
    use_mmff = props is not None
    
    if use_mmff:
        print("Using MMFF94 force field")
    else:
        print("WARNING: MMFF94 not available, using UFF")
    
    return props, use_mmff

def _apply_torsion_constraints(ff, torsion_constraints):
    """Add 180° trans dihedral restraints to an MMFF force field (no-op if empty).

    Each entry is (i, j, k, l, min_deg, max_deg, force_constant). Used instead of
    freezing the glycosidic atoms in place so the bond can relax to length while
    the dihedral is held trans.
    """
    if not torsion_constraints:
        return
    for (i, j, k, l, dmin, dmax, fk) in torsion_constraints:
        ff.MMFFAddTorsionConstraint(i, j, k, l, False, dmin, dmax, fk)


def get_ff_forces(mol, props, use_mmff, n_atoms, max_force, torsion_constraints=None):
    """
    Get forces from force field.
    
    Args:
        mol: RDKit molecule
        props: MMFF properties
        use_mmff: Whether to use MMFF (vs UFF)
        n_atoms: Number of atoms
        max_force: Maximum force magnitude
    
    Returns:
        Array of forces
    """
    if use_mmff:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return np.zeros((n_atoms, 3))

        _apply_torsion_constraints(ff, torsion_constraints)
        ff.Initialize()
        grad = np.array([ff.CalcGrad()[i] for i in range(n_atoms * 3)])
        forces = -grad.reshape((n_atoms, 3))

        return cap_vectors(forces, max_force)
    else:
        return np.zeros((n_atoms, 3))

def calculate_langevin_random_force(n_atoms, masses, friction, timestep):
    """
    Calculate random force for Langevin dynamics.
    
    Returns:
        Random force array
    """
    return np.random.randn(n_atoms, 3) * np.sqrt(
        2 * friction * TEMPERATURE_K * BOLTZMANN_KCAL * 
        masses[:, np.newaxis] / timestep
    )

def update_velocities_langevin(velocities, forces, masses, friction, 
                               timestep, max_velocity):
    """
    Update velocities using Langevin dynamics.
    
    Args:
        velocities: Current velocities (modified in place)
        forces: Total forces
        masses: Atomic masses
        friction: Friction coefficient
        timestep: Time step
        max_velocity: Maximum velocity cap
    
    Returns:
        Updated velocities
    """
    random_force = calculate_langevin_random_force(
        len(masses), masses, friction, timestep
    )
    
    accel = forces / masses[:, np.newaxis]
    velocities = velocities * (1 - friction * timestep) + \
                 (accel + random_force / masses[:, np.newaxis]) * timestep
    
    return cap_vectors(velocities, max_velocity)

def apply_fixed_constraints(new_positions, velocities, original_positions, 
                           fixed_atoms):
    """
    Apply fixed atom constraints.
    
    Args:
        new_positions: New positions (modified in place)
        velocities: Velocities (modified in place)
        original_positions: Original positions to restore
        fixed_atoms: List of atom indices to fix
    
    Returns:
        Tuple of (new_positions, velocities)
    """
    if fixed_atoms:
        for atom_idx in fixed_atoms:
            new_positions[atom_idx] = original_positions[atom_idx]
            velocities[atom_idx] = 0
    
    return new_positions, velocities

def calculate_gravity_forces_with_multipliers(positions, masses, gravity, 
                                              molecule_data_dict, 
                                              influence_radius=5.0):
    """
    Calculate gravitational forces with spatially-varying multipliers.
    
    Each molecule in molecule_data_dict creates a "gravity field" where the
    'brightness' value determines the local gravity multiplier via: multiplier = max(1.0, brightness + 1.0)
    
    Args:
        positions: Nx3 array of atom positions
        masses: N array of atom masses
        gravity: Base gravity constant
        molecule_data_dict: Dictionary mapping molecule names to data, which includes:
                          - 'COM': center of mass [x, y, z]
                          - 'brightness': float value that determines gravity multiplier
        influence_radius: Distance over which a reference point affects gravity (Å)
                         Default 5.0 Å (slightly larger than pyranose ring)
    
    Returns:
        Nx3 force array
        
    Notes:
        - Multiplier formula: max(1.0, brightness + 1.0)
        - brightness=0 → multiplier=1.0 (normal gravity)
        - brightness=2 → multiplier=3.0 (triple gravity)
        - All multipliers are ≥ 1.0 (gravity can only increase, never decrease)
        - Uses Gaussian falloff: weight = exp(-distance²/(2σ²)) where σ = radius/2.5
        - Distance calculated in XY-plane only (horizontal influence zones)
    """
    n_atoms = len(positions)
    forces = np.zeros((n_atoms, 3))
    
    # Extract reference points from molecule COMs and their brightness values
    reference_points = []
    for mol_name, mol_data in molecule_data_dict.items():
        if 'COM' not in mol_data:
            print(f"  Warning: {mol_name} has no COM, skipping")
            continue
        
        # Get brightness value (default to 0.0 if not present)
        brightness = mol_data.get('brightness', 0.0)
        
        com = np.array(mol_data['COM'])
        reference_points.append({
            'xy': com[:2],  # [x, y] position from COM
            'brightness': brightness,
            'mol_name': mol_name
        })
    
    if not reference_points:
        # No reference points - use uniform gravity
        print("  Warning: No reference points in molecule_data_dict, using uniform gravity")
        for i in range(n_atoms):
            if positions[i, 2] > 0:
                forces[i, 2] -= masses[i] * gravity
        return forces
    
    # Gaussian falloff parameter
    sigma = influence_radius / 2.5
    two_sigma_sq = 2.0 * sigma * sigma
    weight_threshold = 0.01
    
    # Calculate gravity for each atom
    for i in range(n_atoms):
        if positions[i, 2] <= 0:  # Skip if at or below ground plane
            continue
        
        atom_xy = positions[i, :2]  # Get XY position only
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Check influence from each reference point
        for ref in reference_points:
            # Calculate XY distance only (horizontal distance)
            delta_xy = atom_xy - ref['xy']
            dist_xy = np.linalg.norm(delta_xy)
            
            # Check if within influence radius
            if dist_xy < influence_radius:
                # Calculate gravity multiplier from brightness
                # Formula: max(1.0, brightness + 1.0)
                multiplier = max(1.0, ref['brightness'] + 1.0)
                
                # Calculate Gaussian weight
                weight = np.exp(-dist_xy * dist_xy / two_sigma_sq)
                
                # Accumulate weighted multiplier
                if weight > weight_threshold:
                    weighted_sum += multiplier * weight
                    total_weight += weight
        
        # Determine final multiplier for this atom
        if total_weight > weight_threshold:
            final_multiplier = weighted_sum / total_weight
            # Safety clamp (should already be ≥ 1.0 from weighted average)
            final_multiplier = max(1.0, min(final_multiplier, 10.0))
        else:
            # No nearby reference points - use default
            final_multiplier = 1.0
        
        # Apply gravity force with multiplier
        forces[i, 2] -= masses[i] * gravity * final_multiplier
    
    return forces

def calculate_gravity_forces(positions, masses, gravity):
    """
    Calculate gravitational forces.
    
    Returns:
        Force array
    """
    n_atoms = len(positions)
    forces = np.zeros((n_atoms, 3))
    
    for i in range(n_atoms):
        if positions[i, 2] > 0:  # Only apply if above z=0
            forces[i, 2] -= masses[i] * gravity
    
    return forces

def calculate_slab_forces(positions, masses, slab_z, slab_force_scale):
    """
    SANDWICH MODEL: Floor + Ceiling + Rising slab + Gravity
    """
    n_atoms = len(positions)
    forces = np.zeros((n_atoms, 3))
    
    z_floor = 0.0      # Bottom bread
    z_ceiling = 2.0    # Top bread 
    
    for i in range(n_atoms):
        z_atom = positions[i, 2]
        
        # Bottom bread - hard floor
        if z_atom < 1.0:
            floor_force = slab_force_scale * 200 * np.exp(-15 * z_atom)
            forces[i, 2] += floor_force
        
        # Top bread - hard ceiling
        if z_atom > z_ceiling - 0.5:
            ceiling_force = slab_force_scale * 100 * np.exp(10 * (z_atom - z_ceiling))
            forces[i, 2] -= ceiling_force
        
        # The press - rising slab
        distance = z_atom - slab_z
        if distance < 0:
            F_mag = slab_force_scale * 10 * np.exp(-5 * distance)
        elif distance < 1.5:
            F_mag = slab_force_scale * (1.5 - distance)**2
        else:
            F_mag = 0
        forces[i, 2] += F_mag
    
    return forces

def initialize_ring_rotation_units(mol, conf, masses, n_atoms, 
                                   pyranose_rings=None,
                                   reference_normals=None):
    """
    Initialize ring rotation dynamics data structures.
    
    Args:
        mol: RDKit molecule
        conf: Conformer
        masses: Atomic masses array
        n_atoms: Number of atoms
        pyranose_rings: Pre-detected rings (optional)
        reference_normals: Dict mapping ring_idx to normal vectors for constrained rings
    
    Returns:
        List of ring rotation unit dictionaries
    """
    if pyranose_rings is None:
        pyranose_rings = detect_pyranose_rings(mol)
    
    print(f"\nInitializing ring rotation dynamics...")
    print(f"  Found {len(pyranose_rings)} pyranose rings")
    
    ring_rotation_units = []
    positions = get_positions(conf, n_atoms)
    
    for ring_idx, ring_atoms in enumerate(pyranose_rings):
        ring_positions = positions[ring_atoms]
        ring_masses = masses[ring_atoms]
        
        # Calculate initial center of mass
        com = calculate_center_of_mass(ring_positions, ring_masses)
        
        # Determine if this ring is constrained
        is_constrained = (reference_normals is not None and 
                         ring_idx in reference_normals)
        
        if is_constrained:
            # Constrained ring: use reference normal
            normal = reference_normals[ring_idx]
            mode = "CONSTRAINED (CD mode)"
        else:
            # Free ring: calculate current normal
            normal = get_ring_normal_from_positions(positions, ring_atoms)
            mode = "FREE (3D motion)"
        
        # Calculate moment of inertia
        I_zz = calculate_moment_of_inertia(ring_atoms, positions, masses, com)
        
        # Create ring unit
        ring_unit = RingRotationUnit(
            atoms=list(ring_atoms),
            com_initial=com.copy(),
            normal_fixed=normal.copy(),
            angular_velocity=0.0,
            moment_inertia=I_zz,
            ring_id=ring_idx,
            total_rotation=0.0,
            is_constrained=is_constrained
        )
        
        ring_rotation_units.append(ring_unit.to_dict())
        
        print(f"  Ring {ring_idx}: {len(ring_atoms)} atoms - {mode}")
        print(f"    COM: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
        print(f"    Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
        print(f"    I_zz: {I_zz:.3f}")
    
    return ring_rotation_units

def print_final_rotation_state(ring_rotation_units):
    """Print summary of final rotation state."""
    print("\nFinal rotation state:")
    for ring_unit in ring_rotation_units:
        total_deg = np.degrees(ring_unit['total_rotation'])
        mode = "CONSTRAINED" if ring_unit.get('is_constrained', False) else "FREE"
        print(f"  Ring {ring_unit['ring_id']} ({mode}): "
              f"total_rotation={total_deg:.1f}°, "
              f"final_ang_vel={ring_unit['angular_velocity']:.5f} rad/s")
        
def run_compression_phase(mol_copy, conf, n_atoms, masses, props, use_mmff,
                          ring_rotation_units, ring_references,
                          config: OptimizationConfig,
                          last_valid_mol, fixed_atoms, xy_constrained_atoms=[],
                          pyranose_rings=None,
                          initial_ring_coms=None,
                          molecule_data_dict=None, stm_data=None,
                          trajectory_path=None, torsion_constraints=None):
    """
    Phase 2: Compression with rising slab using FULL MD.

    Args:
        xy_constrained_atoms: List of atom indices to constrain in xy (z can move)
    """
    print("\n" + "="*60)
    print("PHASE 2: COMPRESSION WITH MD + RISING SLAB")
    print("="*60)

    phase_start = time.perf_counter()

    t_ff_forces       = 0.0
    t_gravity_forces  = 0.0
    t_slab_forces     = 0.0
    t_velocity_update = 0.0
    t_minimize        = 0.0
    step_times        = []

    velocities = np.random.randn(n_atoms, 3) * 0.01
    print("  Initialized MD velocities")

    positions = get_positions(conf, n_atoms)
    slab_z = np.min(positions[:, 2]) - 2.0
    print(f"  Initial slab position: z = {slab_z:.2f} Å")

    # Ring-integrity safeguard baseline: the geometry leaving Phase 1 is the last
    # KNOWN-GOOD state. (The last_valid_mol passed in is pre-Phase-1 and stale.)
    last_valid_mol = Chem.Mol(mol_copy)
    if ring_references:
        print(f"  Ring safeguard active: {len(ring_references)} rings, "
              f"check every {config.check_rings_interval} steps "
              f"(bond<{config.ring_tolerance_bond:.3f} Å, "
              f"angle<{config.ring_tolerance_angle:.1f}°)")

    initial_xy = None
    if xy_constrained_atoms:
        initial_xy = positions[xy_constrained_atoms, :2].copy()
        print(f"  Constraining xy for {len(xy_constrained_atoms)} atoms")

    if initial_ring_coms is not None and pyranose_rings is not None:
        for ring_idx, ring_atoms in enumerate(pyranose_rings):
            if ring_idx not in initial_ring_coms:
                continue
            com_current = np.mean(positions[ring_atoms], axis=0)
            com_target = initial_ring_coms[ring_idx]
            delta = com_target - com_current
            positions[ring_atoms] += delta
        set_positions(conf, positions, n_atoms)
        print("  Hard reset ring COMs to experimental targets before MD")

    energy_history = []
    force_history  = []
    rmsd_history   = []
    prev_positions = None

    step = 0
    for step in range(config.compression_steps):
        step_start = time.perf_counter()
        positions = get_positions(conf, n_atoms)

        z_min = np.min(positions[:, 2])
        z_max = np.max(positions[:, 2])
        current_height = z_max - z_min

        z_ceiling = 2.0
        compression_tolerance = 0.2

        if current_height <= z_ceiling + compression_tolerance:
            print(f"\n✓ COMPRESSION COMPLETE at step {step}")
            print(f"  Current height: {current_height:.2f} Å")
            print(f"  Target height:  {z_ceiling:.2f} Å")
            print(f"  Molecule is fully compressed!")
            break

        if step % config.minimize_interval == 0:
            t0 = time.perf_counter()
            minimize_with_constraint_no_com(
                mol_copy, props, use_mmff, fixed_atoms,
                n_atoms, config.minimize_iterations,
                torsion_constraints=torsion_constraints)
            t_minimize += time.perf_counter() - t0
            velocities = np.random.randn(n_atoms, 3) * 0.01
            positions = get_positions(conf, n_atoms)

        t0 = time.perf_counter()
        ff_forces = get_ff_forces(mol_copy, props, use_mmff, n_atoms, config.max_force,
                                  torsion_constraints=torsion_constraints)
        t_ff_forces += time.perf_counter() - t0

        t0 = time.perf_counter()
        if stm_data is not None:
            gravity_forces = calculate_stm_surface_forces(
                positions, masses, config.gravity, stm_data
            )
        else:
            gravity_forces = calculate_gravity_forces(
                positions, masses, config.gravity
            )
        t_gravity_forces += time.perf_counter() - t0

        t0 = time.perf_counter()
        slab_forces = calculate_slab_forces(
            positions, masses, slab_z, config.slab_force_scale
        )
        t_slab_forces += time.perf_counter() - t0

        t0 = time.perf_counter()
        total_forces = ff_forces + gravity_forces + slab_forces

        if config.enable_convergence:
            try:
                ff = get_force_field(mol_copy, props, use_mmff)
                energy = ff.CalcEnergy()
                energy_history.append(energy)
            except:
                energy_history.append(0.0)

            max_force = np.max(np.linalg.norm(total_forces, axis=1))
            force_history.append(max_force)

            if prev_positions is not None:
                rmsd = calculate_rmsd(positions, prev_positions)
                rmsd_history.append(rmsd)
            prev_positions = positions.copy()

        velocities = update_velocities_langevin(
            velocities, total_forces, masses, config.friction,
            config.timestep, config.max_velocity
        )

        new_positions = positions + velocities * config.timestep

        if fixed_atoms:
            for atom_idx in fixed_atoms:
                new_positions[atom_idx] = positions[atom_idx]
                velocities[atom_idx] = 0

        if xy_constrained_atoms and initial_xy is not None:
            for i, atom_idx in enumerate(xy_constrained_atoms):
                new_positions[atom_idx, 0] = initial_xy[i, 0]
                new_positions[atom_idx, 1] = initial_xy[i, 1]
                velocities[atom_idx, :2] = 0

        set_positions(conf, new_positions, n_atoms)
        t_velocity_update += time.perf_counter() - t0

        # --- Ring-integrity safeguard ---------------------------------------
        # Every check_rings_interval steps, compare each pyranose ring's internal
        # geometry (bonds/angles) to its as-placed chair reference. If all rings
        # are within tolerance, snapshot the current state as the new "last valid".
        # If any ring has distorted past tolerance, revert to that last-valid
        # geometry and kill the kinetic energy so the slab can't keep crushing it.
        if (ring_references and config.check_rings_interval > 0
                and step > 0 and step % config.check_rings_interval == 0):
            rings_ok, _ = check_ring_integrity(
                mol_copy, ring_references,
                config.ring_tolerance_bond, config.ring_tolerance_angle
            )
            if rings_ok:
                last_valid_mol = Chem.Mol(mol_copy)
            else:
                lv_positions = get_positions(last_valid_mol.GetConformer(), n_atoms)
                set_positions(conf, lv_positions, n_atoms)
                velocities = np.random.randn(n_atoms, 3) * 0.01
                print(f"  ⟲ Step {step}: ring distortion beyond tolerance "
                      f"→ reverted to last valid ring geometry")

        step_times.append(time.perf_counter() - step_start)

        if step % config.slab_step_interval == 0 and step > 0:
            slab_z += config.slab_step_size

        if step % 25 == 0:
            z_lowest = np.min(new_positions[:, 2])
            gap      = z_lowest - slab_z
            avg_vel  = np.mean(np.linalg.norm(velocities, axis=1))
            elapsed  = time.perf_counter() - phase_start
            recent   = step_times[-25:]
            avg_ms   = np.mean(recent) * 1000
            min_ms   = np.min(recent)  * 1000
            max_ms   = np.max(recent)  * 1000
            remaining = config.compression_steps - step
            eta_s     = np.mean(recent) * remaining
            print(f"  Step {step:5d}/{config.compression_steps} | "
                  f"Height={current_height:.2f} Å | "
                  f"Slab z={slab_z:.2f} | Gap={gap:.2f} | "
                  f"Avg vel={avg_vel:.4f} | "
                  f"Step={avg_ms:.1f} ms [{min_ms:.1f}-{max_ms:.1f}] | "
                  f"Elapsed={elapsed:.0f}s | ETA={eta_s:.0f}s")

        if config.enable_convergence and step > config.convergence_window:
            converged, reason = check_convergence(
                energy_history, force_history, rmsd_history, config
            )
            if converged:
                print(f"\n✓ CONVERGED at step {step+1}/{config.compression_steps}")
                print(f"  {reason}")
                break

        if step % config.image_interval == 0:
            if config.save_images:
                save_frame(mol_copy, config.output_name, f"phase2_step_{step:04d}")
            if trajectory_path is not None:
                energy = None
                try:
                    ff = get_force_field(mol_copy, props, use_mmff)
                    energy = ff.CalcEnergy()
                except:
                    pass
                append_frame_to_trajectory(
                    mol_copy, trajectory_path, step, 'compression', energy
                )

    n_steps     = len(step_times)
    total_phase = time.perf_counter() - phase_start
    n_min_calls = max(1, n_steps // max(1, config.minimize_interval))

    gravity_label = 'STM' if stm_data is not None else 'uniform'

    unaccounted = total_phase - (t_ff_forces + t_gravity_forces +
                                 t_slab_forces + t_velocity_update + t_minimize)

    print(f"\n{'='*60}")
    print(f"COMPRESSION PHASE TIMING BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Steps completed:        {n_steps}")
    print(f"  Total time:             {total_phase:.2f} s")
    if step_times:
            print(f"  Avg time/step:          {np.mean(step_times)*1000:.2f} ms")
            print(f"  Min/Max step:           {np.min(step_times)*1000:.2f} / {np.max(step_times)*1000:.2f} ms")
    else:
        print(f"  Avg time/step:          N/A")
        print(f"  Min/Max step:           N/A")
    print(f"  {'─'*50}")
    if total_phase > 0:
        print(f"  FF forces:              {t_ff_forces:.2f} s  ({t_ff_forces/total_phase*100:.1f}%)")
        print(f"  Gravity ({gravity_label:<10}):  {t_gravity_forces:.2f} s  ({t_gravity_forces/total_phase*100:.1f}%)")
        print(f"  Slab forces:            {t_slab_forces:.2f} s  ({t_slab_forces/total_phase*100:.1f}%)")
        print(f"  Velocity/pos update:    {t_velocity_update:.2f} s  ({t_velocity_update/total_phase*100:.1f}%)")
        print(f"  Minimization ({n_min_calls} calls): {t_minimize:.2f} s  ({t_minimize/total_phase*100:.1f}%)")
        print(f"  Other (convergence/IO): {unaccounted:.2f} s  ({unaccounted/total_phase*100:.1f}%)")
    print(f"{'='*60}")

    return slab_z, fixed_atoms, last_valid_mol

def minimize_with_constraint_no_com(mol, props, use_mmff, fixed_atoms,
                              n_atoms, max_iterations, torsion_constraints=None):
    """
    Perform energy minimization with position constraints.
    """
    if use_mmff:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            print(f"ERROR: Force field initialization failed")
            return False

        # Add position constraints for ring atoms
        if fixed_atoms:
            for atom_idx in fixed_atoms:
                if 0 <= atom_idx < n_atoms:
                    ff.MMFFAddPositionConstraint(atom_idx, 0.0, 1e15)

        # Hold glycosidic bonds trans (180°) without pinning their positions
        _apply_torsion_constraints(ff, torsion_constraints)

        conf = mol.GetConformer()

        # Find pairs of atoms that are far apart (>10 Å)
        for i in range(0, n_atoms, 20):  # Sample every 20th atom
            for j in range(i+20, n_atoms, 20):
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)
                dist = pos_i.Distance(pos_j)
                
                if dist > 10.0:  # If atoms are far apart
                    # Add distance constraint: minDist, maxDist, forceConstant
                    ff.MMFFAddDistanceConstraint(i, j, False, dist*0.9, dist*1.1, 1e5)
        
        # print(f"  Added distance restraints to prevent collapse")

        ff.Initialize()
        ff.Minimize(maxIts=max_iterations)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iterations)
    
    return True

def minimize_with_constraints(mol, props, use_mmff, fixed_atoms,
                                            n_atoms, max_iterations,
                                            pyranose_rings, initial_ring_coms,
                                            constrained_ring_com_limit=0.5,
                                            free_ring_com_limit=2.0):
    """
    Iterative minimization with COM constraints enforced at each step.
    """
    print(f"\nStarting iterative COM-constrained minimization...")
    print(f"  Total iterations: {max_iterations}")
    print(f"  Steps per iteration: 10")
    print(f"  Rings to monitor: {len(pyranose_rings)}")
    
    steps_per_iteration = 10
    num_iterations = max_iterations // steps_per_iteration
    
    conf = mol.GetConformer()
    
    for iteration in range(num_iterations):
        # ====================================================================
        # MINIMIZE FOR A FEW STEPS
        # ====================================================================
        if use_mmff:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            if ff is None:
                print("ERROR: Force field initialization failed")
                return False
            
            # Add position constraints for fixed atoms
            if fixed_atoms:
                for atom_idx in fixed_atoms:
                    if 0 <= atom_idx < n_atoms:
                        ff.MMFFAddPositionConstraint(atom_idx, 0.0, 1e10)
            
            ff.Initialize()
            ff.Minimize(maxIts=steps_per_iteration)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=steps_per_iteration)
        
        # ====================================================================
        # APPLY COM CORRECTION
        # ====================================================================
        positions = get_positions(conf, n_atoms)
        
        max_correction = 0.0
        
        for ring_idx, ring_atoms in enumerate(pyranose_rings):
            # Skip if all ring atoms are fixed
            if fixed_atoms and all(atom in fixed_atoms for atom in ring_atoms):
                continue
            
            # Calculate current COM
            ring_pos = positions[ring_atoms]
            com_current = np.mean(ring_pos, axis=0)
            
            # Get target COM
            com_target = initial_ring_coms[ring_idx]
            
            # Calculate drift
            drift = com_current - com_target
            drift_mag = np.linalg.norm(drift)
            
            # Use simple limit for now (can add constrained/free logic later)
            max_drift = constrained_ring_com_limit
            
            # Correct if needed
            if drift_mag > max_drift:
                correction = drift * ((drift_mag - max_drift) / drift_mag)
                positions[ring_atoms] -= correction
                max_correction = max(max_correction, np.linalg.norm(correction))
        
        # Update conformer
        set_positions(conf, positions, n_atoms)
        
        # ====================================================================
        # PERIODIC OUTPUT
        # ====================================================================
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            if use_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                if ff:
                    ff.Initialize()
                    energy = ff.CalcEnergy()
                    print(f"  Iter {iteration:3d}/{num_iterations}: "
                          f"E = {energy:8.2f} kcal/mol | "
                          f"Max correction: {max_correction:.4f} Å")
            else:
                print(f"  Iter {iteration:3d}/{num_iterations}: "
                      f"Max correction: {max_correction:.4f} Å")
    
    print("✓ Iterative minimization complete\n")
    return True

def minimize_with_iterative_com_constraints_single_step(mol, props, use_mmff, 
                                                        fixed_atoms, n_atoms, 
                                                        max_iterations,
                                                        pyranose_rings, 
                                                        initial_ring_coms,
                                                        ring_rotation_units=None,
                                                        config=None):
    """
    Single step of iterative minimization with COM correction.
    Called repeatedly by run_minimization_phase().
    
    Args:
        mol: RDKit molecule
        props: MMFF properties
        use_mmff: Whether to use MMFF
        fixed_atoms: List of fixed atom indices
        n_atoms: Number of atoms
        max_iterations: Steps for this iteration
        pyranose_rings: List of ring atom lists
        initial_ring_coms: Dict of initial COM positions
        ring_rotation_units: Optional ring constraint info
        config: OptimizationConfig object
    
    Returns:
        bool: Success status
    """
    # Get COM limits from config
    if config:
        constrained_limit = getattr(config, 'constrained_ring_com_limit', 0.5)
        free_limit = getattr(config, 'free_ring_com_limit', 2.0)
    else:
        constrained_limit = 0.5
        free_limit = 2.0
    
    conf = mol.GetConformer()
    
    # Perform minimization
    if use_mmff:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return False
        
        # Add position constraints for fixed atoms
        if fixed_atoms:
            for atom_idx in fixed_atoms:
                if 0 <= atom_idx < n_atoms:
                    ff.MMFFAddPositionConstraint(atom_idx, 0.0, 1e10)
        
        # Add distance constraints to prevent collapse
        for i in range(0, n_atoms, 20):
            for j in range(i+20, n_atoms, 20):
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)
                dist = pos_i.Distance(pos_j)
                
                if dist > 10.0:
                    ff.MMFFAddDistanceConstraint(i, j, False, dist*0.9, dist*1.1, 1e5)
        
        ff.Initialize()
        ff.Minimize(maxIts=max_iterations)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iterations)
    
    # Apply COM correction
    positions = get_positions(conf, n_atoms)
    
    for ring_idx, ring_atoms in enumerate(pyranose_rings):
        # Skip if all ring atoms are fixed
        if fixed_atoms and all(atom in fixed_atoms for atom in ring_atoms):
            continue
        
        # Calculate current COM
        ring_pos = positions[ring_atoms]
        com_current = np.mean(ring_pos, axis=0)
        
        # Get target COM
        com_target = initial_ring_coms[ring_idx]
        
        # Calculate drift
        drift = com_current - com_target
        drift_mag = np.linalg.norm(drift)
        
        # Determine limit based on ring type
        if ring_rotation_units:
            is_constrained = ring_rotation_units[ring_idx].get('is_constrained', False)
            max_drift = constrained_limit if is_constrained else free_limit
        else:
            max_drift = constrained_limit
        
        # Correct if needed
        if drift_mag > max_drift:
            correction = drift * ((drift_mag - max_drift) / drift_mag)
            positions[ring_atoms] -= correction
    
    # Update conformer
    set_positions(conf, positions, n_atoms)
    
    return True

def optimize_with_slab_and_rings(mol, config=None, molecule_data_dict=None,
                                  lipid_tail_indices=None, stm_npy_path=None,  # ← path not data
                                  **kwargs):
    """
    Optimized molecular dynamics with slab compression and ring monitoring.
    """
    # Build interpolator ONCE if path provided
    stm_data = None
    if stm_npy_path is not None:             
        stm_data = build_stm_interpolator(stm_npy_path) 
    """
    Optimized molecular dynamics with slab compression and ring monitoring.
    
    This is the main entry point with two operation modes:
    1. Pass an OptimizationConfig object (recommended)
    2. Pass individual parameters as kwargs (legacy support)
    
    Args:
        mol: RDKit molecule
        config: OptimizationConfig object (if None, created from kwargs)
        **kwargs: Individual parameters (for backward compatibility)
    
    Returns:
        Tuple of (optimized_molecule, success: bool)
    
    Example:
        >>> config = OptimizationConfig(
        ...     relaxation_steps=100,
        ...     compression_steps=200,
        ...     save_images=True
        ... )
        >>> opt_mol, success = optimize_with_slab_and_rings(mol, config)
    """
    # Create config from kwargs if not provided
    if config is None:
        config = OptimizationConfig(**kwargs)
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION WITH SLAB AND RING MONITORING")
    print("="*70)
    print(f"Configuration:")
    print(f"  Relaxation steps: {config.relaxation_steps}")
    print(f"  Compression steps: {config.compression_steps}")
    print(f"  Ring rotation: {'enabled' if config.enable_ring_rotation else 'disabled'}")
    print(f"  Save images: {config.save_images}")
    
    # trajectory_path = f"{config.output_name}_trajectory.sdf"
    # if os.path.exists(trajectory_path):
    #     os.remove(trajectory_path)
    # print(f"\n Trajectory will be saved to: {trajectory_path}")
    trajectory_path = None  # intermediate trajectory disabled (None -> appends are no-op)

    # Setup output directory
    if config.save_images:
        os.makedirs(f"{config.output_name}_frames", exist_ok=True)
    
    # Copy and sanitize molecule
    mol_copy = Chem.Mol(mol)
    Chem.GetSymmSSSR(mol_copy)
    
    try:
        Chem.SanitizeMol(mol_copy)
    except Exception as e:
        if "valence" in str(e).lower():
            print(f"Valence issue detected: {e}")
            try:
                mol_copy = fix_valence_issues(mol_copy)
            except:
                print("ERROR: Could not fix valence issues")
                return mol, False
        else:
            raise
    
    # Setup basic parameters
    conf = mol_copy.GetConformer()
    n_atoms = mol_copy.GetNumAtoms()
    masses = np.array([mol_copy.GetAtomWithIdx(i).GetMass() 
                      for i in range(n_atoms)])
    
    # Detect and store pyranose rings
    pyranose_rings = detect_pyranose_rings(mol_copy)
    print(f"\nDetected {len(pyranose_rings)} pyranose rings")
    
    ring_references = []
    for ring_atoms in pyranose_rings:
        ref = get_ring_reference_geometry(mol_copy, ring_atoms)
        ring_references.append(ref)
    
    # ========================================================================
    # EXTRACT INITIAL RING COMs FROM molecule_data_dict
    # ========================================================================
    
    initial_ring_coms = {}
    
    if molecule_data_dict is not None:
        print("\n✓ Extracting initial ring COMs from molecule_data_dict")
        
        conf = mol_copy.GetConformer()
        positions = get_positions(conf, n_atoms)
        
        # Calculate current ring centers
        ring_centers = []
        for ring_atoms in pyranose_rings:
            # Use heavy atoms only
            heavy_atoms = [i for i in ring_atoms 
                          if mol_copy.GetAtomWithIdx(i).GetSymbol() != 'H']
            ring_pos = positions[heavy_atoms]
            center = np.mean(ring_pos, axis=0)
            ring_centers.append(center)
        
        # Match each ring to a monomer in molecule_data_dict
        MATCHING_THRESHOLD = 0.25  # Angstroms
        
        for ring_idx, ring_center in enumerate(ring_centers):
            best_match = None
            min_dist = float('inf')
            
            # Try to match with each monomer
            for mol_name, mol_data in molecule_data_dict.items():
                # Get monomer COM (already at experimental position!)
                monomer_com = np.array(mol_data['COM'])
                
                # Calculate distance
                dist = np.linalg.norm(ring_center - monomer_com)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match = (mol_name, monomer_com)
            
            # Store the matched COM
            if best_match and min_dist < MATCHING_THRESHOLD:
                mol_name, monomer_com = best_match
                initial_ring_coms[ring_idx] = monomer_com
                
                print(f"  Ring {ring_idx}: matched to {mol_name}")
                print(f"    Experimental COM: ({monomer_com[0]:.2f}, {monomer_com[1]:.2f}, {monomer_com[2]:.2f}) Å")
                print(f"    Current ring COM: ({ring_center[0]:.2f}, {ring_center[1]:.2f}, {ring_center[2]:.2f}) Å")
                print(f"    Distance: {min_dist:.3f} Å")
            else:
                # Fallback: use current position
                print(f"  Ring {ring_idx}: No match found (dist={min_dist:.3f} Å), using current COM")
                initial_ring_coms[ring_idx] = ring_center
    else:
        print("\n⚠ molecule_data_dict not provided, calculating COMs from current structure")
        
        # Fallback: calculate from current positions
        conf = mol_copy.GetConformer()
        positions = get_positions(conf, n_atoms)
        
        for ring_idx, ring_atoms in enumerate(pyranose_rings):
            ring_pos = positions[ring_atoms]
            com = np.mean(ring_pos, axis=0)
            initial_ring_coms[ring_idx] = com
    
    print(f"\n✓ Extracted initial COMs for {len(initial_ring_coms)} rings")
    print("\nInitial ring center of mass positions:")
    for ring_idx in sorted(initial_ring_coms.keys()):
        com = initial_ring_coms[ring_idx]
        print(f"  Ring {ring_idx}: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) Å")

    # Setup force field
    props, use_mmff = setup_force_field(mol_copy)
    
    # Initialize ring rotation units
    ring_rotation_units = []
    if config.enable_ring_rotation:
        ring_rotation_units = initialize_ring_rotation_units(
            mol_copy, conf, masses, n_atoms,
            pyranose_rings=pyranose_rings,
            reference_normals=config.reference_normals
        )
    else:
        print("\nRing rotation disabled - using full constraint")
    
    # Track last valid state
    last_valid_mol = Chem.Mol(mol_copy)
    
    
    fixed_atoms = config.fixed_atoms.copy() if config.fixed_atoms else []
    from .utils import save_molecule

    # Glycosidic trans dihedrals: frozen rigid (positions) in phase 1 as an anchor,
    # then held as a 180° torsion restraint in phases 2 & 3 so the bond can relax
    # to length while staying trans. fixed_atoms (linker + PEtN) stay position-
    # frozen in ALL phases; lipids are free.
    torsion_constraints = [
        (i, j, k, l, config.torsion_min_deg, config.torsion_max_deg,
         config.torsion_force_constant)
        for (i, j, k, l) in (config.torsion_constraints or [])
    ]
    glyco_atoms = sorted({a for t in torsion_constraints for a in t[:4]})

    if fixed_atoms:
        print(f"\n  Fixed atoms from config: {len(fixed_atoms)} (frozen all phases)")
    if torsion_constraints:
        print(f"  Trans torsion restraints: {len(torsion_constraints)} "
              f"(phase 1 freezes {len(glyco_atoms)} atoms; phases 2-3 restrain @180°)")

    # PHASE 1: CONSTRAINED MINIMIZATION (glycosidic atoms frozen as trans anchor)
    phase1_fixed = list(dict.fromkeys(fixed_atoms + glyco_atoms))
    t0 = time.perf_counter()
    mol_copy = run_minimization_phase_no_cog(
        mol_copy, props, use_mmff,
        phase1_fixed,
        n_atoms, config,
        trajectory_path=trajectory_path
    )
    t_phase1 = time.perf_counter() - t0

    # save_molecule(mol_copy, f"{config.output_name}_phase1_minimized", file_format='sdf')
    # print(f"  💾 Saved: {config.output_name}_phase1_minimized.sdf")

    # PHASE 2: COMPRESSION

    t0 = time.perf_counter()
    final_slab_z, _, last_valid_mol = run_compression_phase(
        mol_copy, conf, n_atoms, masses, props, use_mmff,
        ring_rotation_units, ring_references,
        config, last_valid_mol, fixed_atoms,
        xy_constrained_atoms=lipid_tail_indices or [],
        molecule_data_dict=molecule_data_dict,
        stm_data=stm_data,
        trajectory_path=trajectory_path,
        torsion_constraints=torsion_constraints
    )
    t_phase2 = time.perf_counter() - t0


    # save_molecule(mol_copy, f"{config.output_name}_phase2_compressed", file_format='sdf')
    # print(f"  💾 Saved: {config.output_name}_phase2_compressed.sdf")

    # PHASE 3: FINAL MINIMIZATION WITH LINKER STILL FROZEN

    t0 = time.perf_counter()
    # Phase 3: linker + PEtN stay frozen (fixed_atoms); glycosidic held trans;
    # everything else (incl. lipids, ring COMs) free to relax the whole structure.
    mol_copy = run_final_minimization_phase(
        mol_copy, props, use_mmff, config,
        fixed_atoms=fixed_atoms,
        trajectory_path=trajectory_path,
        torsion_constraints=torsion_constraints
    )
    t_phase3 = time.perf_counter() - t0

    
    total = t_phase1 + t_phase2 + t_phase3
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 1 (constrained minimization): {t_phase1:8.2f} s  ({t_phase1/total*100:5.1f}%)")
    print(f"  Phase 2 (compression MD):           {t_phase2:8.2f} s  ({t_phase2/total*100:5.1f}%)")
    print(f"  Phase 3 (final minimization):       {t_phase3:8.2f} s  ({t_phase3/total*100:5.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Total:                              {total:8.2f} s")
    print(f"{'='*60}")

    return mol_copy, True

def get_atoms_to_constrain_xy(mol_copy, molecule_data_dict, pyranose_rings):
    """
    Determine which atoms to constrain in xy during compression.
    
    Strategy:
    - If lipids present: constrain last carbon of each lipid chain
    - If no lipids: constrain sugar ring atoms
    
    Returns:
        List of atom indices to constrain in xy
    """
    print("\n" + "="*60)
    print("DETERMINING XY CONSTRAINTS")
    print("="*60)
    
    # Check if any molecule has lipid chains
    has_lipids = False
    for mol_name, mol_data in molecule_data_dict.items():
        if 'lipid_chains' in mol_data and len(mol_data['lipid_chains']) > 0:
            has_lipids = True
            print(f"  Found lipids in {mol_name}: {len(mol_data['lipid_chains'])} chains")
            break
    
    atoms_to_constrain = []
    conf = mol_copy.GetConformer()
    
    if has_lipids:
        print("  Strategy: Lipids detected - constraining lipid tail carbons")
        
        for mol_name, mol_data in molecule_data_dict.items():
            if 'lipid_chains' not in mol_data:
                continue
            
            # Get sugar center position
            coords = np.array(mol_data['absolute_coordinates'])
            sugar_center = np.mean(coords, axis=0)
            
            for lipid_idx, lipid_chain in enumerate(mol_data['lipid_chains']):
                # Find the carbon atom FURTHEST from sugar center
                # (that's the tail end of the lipid)
                
                max_dist_from_sugar = -1
                furthest_carbon = None
                
                for atom_idx in range(mol_copy.GetNumAtoms()):
                    atom = mol_copy.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        pos = conf.GetAtomPosition(atom_idx)
                        pos_arr = np.array([pos.x, pos.y, pos.z])
                        
                        # Distance from sugar center
                        dist_from_sugar = np.linalg.norm(pos_arr - sugar_center[:3])
                        
                        # Check if this carbon is far from sugar (likely lipid tail)
                        if dist_from_sugar > 10.0 and dist_from_sugar > max_dist_from_sugar:  # At least 10 Å from sugar
                            max_dist_from_sugar = dist_from_sugar
                            furthest_carbon = atom_idx
                
                if furthest_carbon is not None:
                    atoms_to_constrain.append(furthest_carbon)
                    print(f"    {mol_name} lipid {lipid_idx}: constraining C{furthest_carbon} (furthest from sugar: {max_dist_from_sugar:.1f} Å)")
                else:
                    print(f"    {mol_name} lipid {lipid_idx}: no carbon found far from sugar")
        
        # Remove duplicates
        atoms_to_constrain = list(set(atoms_to_constrain))
        print(f"\n  ✓ Constraining {len(atoms_to_constrain)} lipid tail carbons")
    
    else:
        print("  Strategy: No lipids - constraining sugar ring atoms")
        
        # Constrain all ring atoms
        for ring_idx, ring_atoms in enumerate(pyranose_rings):
            atoms_to_constrain.extend(ring_atoms)
            print(f"    Ring {ring_idx}: {len(ring_atoms)} atoms")
        
        atoms_to_constrain = list(set(atoms_to_constrain))
        print(f"\n  ✓ Constraining {len(atoms_to_constrain)} ring atoms")
    
    return atoms_to_constrain

def fix_overvalent_carbons(mol):
    """
    Legacy function - Remove hydrogens from carbons with too many bonds.
    Consider using fix_valence_issues() instead.
    """
    mol_rw = Chem.RWMol(mol)
    removed = []
    
    for atom in mol_rw.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetDegree() > 4:
            idx = atom.GetIdx()
            h_neighbors = [n.GetIdx() for n in atom.GetNeighbors() 
                          if n.GetSymbol() == 'H']
            
            if h_neighbors:
                removed.append((idx, h_neighbors[0]))
    
    for carbon_idx, h_idx in sorted(removed, key=lambda x: x[1], reverse=True):
        print(f"Removing H{h_idx} from overvalent C{carbon_idx}")
        mol_rw.RemoveAtom(h_idx)
    
    return mol_rw.GetMol()

def run_final_minimization_phase(mol_copy, props, use_mmff,
                                 config: OptimizationConfig,
                                 fixed_atoms=None,
                                 trajectory_path=None,
                                 torsion_constraints=None):
    """
    Phase 4: Final gentle minimization without constraints.
    fixed_atoms: list of atom indices to pin (e.g. linker ring atoms).
    """
    if fixed_atoms is None:
        fixed_atoms = []
    total_iterations = 200  # Gentle final polish
    steps_per_update = 4
    num_updates = total_iterations // steps_per_update  # 50 updates

    for update in range(num_updates):
        if use_mmff:
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
            if ff:
                for atom_idx in fixed_atoms:
                    if 0 <= atom_idx < mol_copy.GetNumAtoms():
                        ff.MMFFAddPositionConstraint(atom_idx, 0.0, 1e15)
                # Linker/PEtN stay frozen above; glycosidic bonds held trans here
                _apply_torsion_constraints(ff, torsion_constraints)
                ff.Initialize()
                ff.Minimize(maxIts=steps_per_update)  # Only 4 steps at a time
            else:
                print("  ⚠ Could not initialize force field")
                break
        else:
            AllChem.UFFOptimizeMolecule(mol_copy, maxIters=steps_per_update)
        
        if trajectory_path is not None and update % config.image_interval == 0:
            energy = None
            if use_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
                if ff:
                    ff.Initialize()
                    energy = ff.CalcEnergy()
            append_frame_to_trajectory(mol_copy, trajectory_path, update, 'final', energy)
            
        if update % 10 == 0:  # Report every 10 updates
            if use_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
                if ff:
                    energy = ff.CalcEnergy()
                    print(f"  Update {update}/{num_updates}: E = {energy:.2f} kcal/mol")
    
    print("  ✓ Final minimization complete")
    return mol_copy

def get_fixed_atoms_from_constraints_by_position(constraints, molecule_data_dict,
                                                 final_mol, pre_detected_rings=None):
    """
    Identify ring atoms and their constraint status based on position matching.
    
    Args:
        constraints: Dict of constraints per monomer
        molecule_data_dict: Dict mapping monomer names to data
        final_mol: RDKit molecule
        pre_detected_rings: Pre-detected rings (optional)
    
    Returns:
        Tuple of (fixed_atoms, reference_normals, constrained_ring_ids)
    """
    
    if pre_detected_rings is not None:
        all_rings = pre_detected_rings
        print(f"Using {len(all_rings)} pre-detected pyranose rings")
    else:
        all_rings = detect_pyranose_rings(final_mol)
        print(f"Found {len(all_rings)} pyranose rings in final molecule")
    
    print("\nIdentifying rings to fix by position matching...")
    
    conf = final_mol.GetConformer()
    
    # Calculate ring centers
    ring_centers = []
    for ring_atoms in all_rings:
        positions = []
        for atom_idx in ring_atoms:
            pos = conf.GetAtomPosition(atom_idx)
            positions.append(np.array([pos.x, pos.y, pos.z]))
        center = np.mean(positions, axis=0)
        ring_centers.append(center)
    
    fixed_atoms = []
    reference_normals = {}
    constrained_ring_ids = set()
    
    for monomer_name in constraints.keys():
        if monomer_name not in molecule_data_dict:
            print(f"  Warning: {monomer_name} not found")
            continue
        
        mol_data = molecule_data_dict[monomer_name]
        coords = np.array(mol_data['absolute_coordinates'])
        carbon_map = mol_data['carbon_map']
        
        ring_local_indices = carbon_map.get('all_ring_carbons', [])
        ring_oxygen_idx = carbon_map.get('ring_oxygen')
        if ring_oxygen_idx is not None:
            ring_local_indices = list(ring_local_indices) + [ring_oxygen_idx]
        
        if not ring_local_indices:
            print(f"  Warning: No ring atoms for {monomer_name}")
            continue
        
        target_center = np.mean([coords[i] for i in ring_local_indices], axis=0)
        normal_ref = get_ring_normal_absolute(mol_data)
        
        # Find best matching ring
        min_dist = float('inf')
        best_ring_idx = None
        
        for ring_idx, ring_center in enumerate(ring_centers):
            dist = np.linalg.norm(ring_center - target_center)
            if dist < min_dist:
                min_dist = dist
                best_ring_idx = ring_idx
        
        if best_ring_idx is not None and min_dist < RING_MATCHING_DISTANCE_THRESHOLD:
            ring_atoms = all_rings[best_ring_idx]
            fixed_atoms.extend(ring_atoms)
            reference_normals[best_ring_idx] = normal_ref
            constrained_ring_ids.add(best_ring_idx)
            
            print(f"  {monomer_name}: matched to ring {best_ring_idx} "
                  f"(CONSTRAINED - CD mode)")
            print(f"    Ring atoms: {ring_atoms}")
            print(f"    Normal: ({normal_ref[0]:.3f}, {normal_ref[1]:.3f}, "
                  f"{normal_ref[2]:.3f})")
        else:
            print(f"  Warning: Could not match {monomer_name} "
                  f"(min distance: {min_dist:.3f}Å)")
    
    # Print free rings
    print(f"\nFree rings (full 3D motion):")
    for ring_idx in range(len(all_rings)):
        if ring_idx not in constrained_ring_ids:
            print(f"  Ring {ring_idx}: FREE (can rotate/translate in 3D)")
    
    fixed_atoms = list(dict.fromkeys(fixed_atoms))
    print(f"\nTotal constrained ring atoms: {len(fixed_atoms)}")

    print(f"\nApplying flat orientation constraints:")
    flat_count = 0
    
    for monomer_name in constraints.keys():
        if monomer_name not in molecule_data_dict:
            continue
        
        # Check if this monomer has orientation='flat'
        constraint_value = constraints[monomer_name]
        orientation = None
        
        if isinstance(constraint_value, dict):
            orientation = constraint_value.get('orientation', None)
        elif isinstance(constraint_value, str):
            orientation = constraint_value if constraint_value == 'flat' else None
        
        if orientation == 'flat':
            mol_data = molecule_data_dict[monomer_name]
            coords = np.array(mol_data['absolute_coordinates'])
            carbon_map = mol_data['carbon_map']
            
            ring_local_indices = carbon_map.get('all_ring_carbons', [])
            ring_oxygen_idx = carbon_map.get('ring_oxygen')
            if ring_oxygen_idx is not None:
                ring_local_indices = list(ring_local_indices) + [ring_oxygen_idx]
            
            if not ring_local_indices:
                continue
            
            target_center = np.mean([coords[i] for i in ring_local_indices], axis=0)
            
            # Find matching ring
            min_dist = float('inf')
            best_ring_idx = None
            
            for ring_idx, ring_center in enumerate(ring_centers):
                dist = np.linalg.norm(ring_center - target_center)
                if dist < min_dist:
                    min_dist = dist
                    best_ring_idx = ring_idx
            
            if best_ring_idx is not None and min_dist < RING_MATCHING_DISTANCE_THRESHOLD:
                # Force this ring to be flat (normal pointing up in z)
                reference_normals[best_ring_idx] = np.array([0.0, 0.0, 1.0])
                print(f"  {monomer_name} (ring {best_ring_idx}): FORCED FLAT (normal = [0, 0, 1])")
                flat_count += 1
    
    if flat_count > 0:
        print(f"Total rings forced flat: {flat_count}")
    
    return fixed_atoms, reference_normals, constrained_ring_ids

def run_minimization_phase(mol_copy, props, use_mmff, fixed_atoms, 
                           n_atoms, config: OptimizationConfig,
                           pyranose_rings=None,        
                           initial_ring_coms=None,     
                           ring_rotation_units=None,
                           trajectory_path=None):  
    """
    Phase 1: Energy minimization with ring COM constraints.
    Uses iterative minimization with small steps.
    """
    print("\n" + "="*60)
    phase_start = time.perf_counter()
    t_minimize = 0.0
    call_times = []

    print("PHASE 1: CONSTRAINED MINIMIZATION")
    if pyranose_rings and initial_ring_coms:
        print("         WITH COM PRESERVATION")
    print("="*60)
    
    print(f"  Minimizing with {len(fixed_atoms)} constrained atoms...")
    if pyranose_rings and initial_ring_coms:
        print(f"  Preserving COM for {len(pyranose_rings)} pyranose rings...")
    print(f"  Using iterative approach: 4 steps per update")
    
    total_iterations = 1500
    steps_per_update = 4
    num_updates = total_iterations // steps_per_update

    for update in range(num_updates):
        if pyranose_rings and initial_ring_coms:
            t0 = time.perf_counter()
            success = minimize_with_iterative_com_constraints_single_step(
                mol_copy, props, use_mmff, fixed_atoms,
                n_atoms, steps_per_update,
                pyranose_rings, initial_ring_coms,
                ring_rotation_units, config
            )
            call_time = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            success = minimize_with_constraint_no_com(
                mol_copy, props, use_mmff, fixed_atoms,
                n_atoms, max_iterations=steps_per_update
            )
            call_time = time.perf_counter() - t0

        t_minimize += call_time
        call_times.append(call_time)

        if update % 25 == 0 or (trajectory_path is not None and update % config.image_interval == 0):
            energy = None
            if use_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
                if ff:
                    ff.Initialize()
                    energy = ff.CalcEnergy()
            if update % 25 == 0 and energy is not None:
                print(f"  Update {update}/{num_updates}: E = {energy:.2f} kcal/mol")
            if trajectory_path is not None and update % config.image_interval == 0:
                append_frame_to_trajectory(mol_copy, trajectory_path, update, 'minimization', energy)

    print("  ✓ Minimization complete")
    total_phase = time.perf_counter() - phase_start
    n_calls = len(call_times)
    print(f"\n{'='*60}")
    mode = "COM-constrained" if (pyranose_rings and initial_ring_coms) else "plain"
    print(f"  Mode: {mode}")
    print(f"MINIMIZATION PHASE TIMING")
    print(f"{'='*60}")
    print(f"  Total time:       {total_phase:.2f} s")
    print(f"  Minimize calls:   {n_calls}")
    print(f"  Total in calls:   {t_minimize:.2f} s  ({t_minimize/total_phase*100:.1f}%)")
    if n_calls > 0:
        print(f"  Avg/call:         {np.mean(call_times)*1000:.2f} ms")
        print(f"  Min/Max call:     {np.min(call_times)*1000:.2f} / {np.max(call_times)*1000:.2f} ms")
    print(f"{'='*60}")

    return mol_copy

def run_minimization_phase_no_cog(mol_copy, props, use_mmff, fixed_atoms,
                           n_atoms, config: OptimizationConfig,trajectory_path=None):
    """
    Phase 2: Energy minimization with ring constraints (Avogadro style).
    Uses iterative minimization with small steps.
    """
    print("\n" + "="*60)

    phase_start = time.perf_counter()
    t_minimize = 0.0
    call_times = []

    print("PHASE 1: CONSTRAINED MINIMIZATION")
    print("="*60)
    
    print(f"  Minimizing with {len(fixed_atoms)} constrained atoms...")
    print(f"  Using iterative approach: 4 steps per update")
    
    # Iterative minimization like Avogadro (steps per update = 4)
    total_iterations = 1500  # Was 500, now 3x = 1500
    steps_per_update = 4
    num_updates = total_iterations // steps_per_update  # 375 updates
    
    fixed_set = set(fixed_atoms) if fixed_atoms else set()
    conf_p1 = mol_copy.GetConformer()

    for update in range(num_updates):
        t0 = time.perf_counter()
        success = minimize_with_constraint_no_com(
            mol_copy, props, use_mmff, fixed_atoms,
            n_atoms, max_iterations=steps_per_update
        )
        call_time = time.perf_counter() - t0
        t_minimize += call_time
        call_times.append(call_time)

        # Optional stochastic kicks — disabled by default (enable_phase1_kicks=False).
        # Mimics phase 2's velocity reinitialization to help escape torsional local
        # minima (e.g. twisted bonded sugar rings) without disturbing well-built structures.
        if config.enable_phase1_kicks and update > 0 and update % config.phase1_kick_interval == 0:
            for i in range(n_atoms):
                if i not in fixed_set:
                    pos = conf_p1.GetAtomPosition(i)
                    dx, dy, dz = np.random.randn(3) * config.phase1_kick_amplitude
                    conf_p1.SetAtomPosition(i, (pos.x + dx, pos.y + dy, pos.z + dz))

        if trajectory_path is not None and update % config.image_interval == 0:
            energy = None
            if use_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
                if ff:
                    ff.Initialize()
                    energy = ff.CalcEnergy()
            append_frame_to_trajectory(mol_copy, trajectory_path, update, 'minimization', energy)
        
        if update % 25 == 0:  # Report every 25 updates
            if use_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
                if ff:
                    energy = ff.CalcEnergy()
                    print(f"  Update {update}/{num_updates}: E = {energy:.2f} kcal/mol")
    
    print("  ✓ Minimization complete")
    total_phase = time.perf_counter() - phase_start
    n_calls = len(call_times)
    print(f"\n{'='*60}")
    print(f"MINIMIZATION PHASE TIMING")
    print(f"{'='*60}")
    print(f"  Total time:       {total_phase:.2f} s")
    print(f"  Minimize calls:   {n_calls}")
    print(f"  Total in calls:   {t_minimize:.2f} s  ({t_minimize/total_phase*100:.1f}%)")
    if n_calls > 0:
        print(f"  Avg/call:         {np.mean(call_times)*1000:.2f} ms")
        print(f"  Min/Max call:     {np.min(call_times)*1000:.2f} / {np.max(call_times)*1000:.2f} ms")
    print(f"{'='*60}")
    return mol_copy

def build_stm_interpolator(stm_npz_path):
    """
    Load STM surface from NPZ file and build a 2D interpolator.
    Called ONCE before the MD loop.

    Expected NPZ format:
        x: 1D array, shape (W,), x coordinates in Ångströms
        y: 1D array, shape (H,), y coordinates in Ångströms
        z: 2D array, shape (W, H), raw STM heights where z[i,j] = height at x[i], y[j]

    Args:
        stm_npz_path: Path to .npz file

    Returns:
        dict:
            'interpolator': RectBivariateSpline object
            'x_min', 'x_max': scan x boundaries (Å)
            'y_min', 'y_max': scan y boundaries (Å)
    """
    print("\n" + "="*60)
    print("BUILDING STM SURFACE INTERPOLATOR")
    print("="*60)

    # Load - zero manipulation, exactly as saved
    data     = np.load(stm_npz_path)
    x_unique = data['x']    # shape (W,)
    y_unique = data['y']    # shape (H,)
    z_grid   = data['z']    # shape (W, H)

    print(f"  Loaded: {stm_npz_path}")
    print(f"  Grid:   {len(x_unique)} × {len(y_unique)} points")
    print(f"  X range: [{x_unique[0]:.2f}, {x_unique[-1]:.2f}] Å")
    print(f"  Y range: [{y_unique[0]:.2f}, {y_unique[-1]:.2f}] Å")
    print(f"  Z range: [{z_grid.min():.4f}, {z_grid.max():.4f}] (raw units)")

    # Normalize z to [1, 2] — INVERTED mapping:
    #   darkest pixel  (z_min) -> 2.0  (most compression)
    #   brightest pixel (z_max) -> 1.0  (least compression)
    z_min  = z_grid.min()
    z_max  = z_grid.max()
    z_norm = 2.0 - (z_grid - z_min) / (z_max - z_min)
    print(f"  Normalized z to [1, 2] (inverted: dark=2, bright=1)")

    # Gaussian smoothing to suppress noise
    # sigma in pixels derived from grid spacing
    dx           = x_unique[1] - x_unique[0]   # Å/pixel
    sigma_pixels = 2.0 / dx                    # 2.0 Å smoothing
    z_smooth     = gaussian_filter(z_norm, sigma=sigma_pixels)
    print(f"  Gaussian smoothing: σ = 2.0 Å ({sigma_pixels:.1f} pixels)")

    # Build interpolator - x_unique, y_unique already sorted and uniform
    interpolator = RectBivariateSpline(x_unique, y_unique, z_smooth)
    print(f"  ✓ Interpolator ready")

    return {
        'interpolator': interpolator,
        'x_min': float(x_unique[0]),
        'x_max': float(x_unique[-1]),
        'y_min': float(y_unique[0]),
        'y_max': float(y_unique[-1]),
    }

def calculate_stm_surface_forces(positions, masses, gravity, stm_data):
    """
    Calculate downward forces from STM surface potential.
    Called every MD step. Replaces calculate_gravity_forces() when STM data available.

    Force model (vertical only):
        F_z = -m × A × h_norm(x,y) × exp(-z_atom / λ) / λ

    Parameters:
        λ = 1.0 Å  (hardcoded decay length)
        A = gravity × λ  (h_norm ∈ [1,2] carries the multiplier)

    Args:
        positions: Nx3 array of atom positions (Å)
        masses:    N array of atom masses
        gravity:   base gravity constant (from config)
        stm_data:  dict returned by build_stm_interpolator()

    Returns:
        Nx3 force array

    Notes:
        - Atoms below z=0: no force
        - Atoms outside STM scan area: fall back to uniform gravity
        - h_norm in [1, 2] (INVERTED): bright regions = 1× gravity, dark = 2× gravity
        - h_norm clamped to [1, 2] to handle spline overshoot
    """
    LAMBDA_DECAY = 1.0
    A = gravity * LAMBDA_DECAY        # h_norm carries multiplier, base h=1 → 1×g

    n_atoms = len(positions)
    forces  = np.zeros((n_atoms, 3))

    interpolator = stm_data['interpolator']
    x_min        = stm_data['x_min']
    x_max        = stm_data['x_max']
    y_min        = stm_data['y_min']
    y_max        = stm_data['y_max']

    for i in range(n_atoms):
        x_atom = positions[i, 0]
        y_atom = positions[i, 1]
        z_atom = positions[i, 2]

        # Skip atoms at or below ground plane
        if z_atom <= 0:
            continue

        if x_min <= x_atom <= x_max and y_min <= y_atom <= y_max:
            # Inside STM scan: use surface potential
            h_norm = float(interpolator.ev(x_atom, y_atom))
            h_norm = max(1.0, min(2.0, h_norm))    # clamp to [1, 2]

            decay        = np.exp(-z_atom / LAMBDA_DECAY)
            forces[i, 2] -= masses[i] * A * h_norm * decay / LAMBDA_DECAY

        else:
            # Outside STM scan: uniform gravity fallback
            forces[i, 2] -= masses[i] * gravity

    return forces

def append_frame_to_trajectory(mol, trajectory_path, step, phase, energy=None):
    import io
    mol_copy = Chem.RWMol(mol)
    mol_copy.SetProp('step', str(step))
    mol_copy.SetProp('phase', phase)
    if energy is not None:
        mol_copy.SetProp('energy', f'{energy:.4f}')
    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    writer.write(mol_copy)
    writer.flush()
    with open(trajectory_path, 'a') as f:
        f.write(stream.getvalue())
