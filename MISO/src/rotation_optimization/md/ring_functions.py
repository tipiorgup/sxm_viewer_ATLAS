import numpy as np
from rdkit import Chem
from dataclasses import dataclass
from typing import List, Optional, Dict

from ...constants import (
    DEFAULT_BOND_TOLERANCE, DEFAULT_ANGLE_TOLERANCE, EPSILON,
    DEFAULT_RING_ROTATION_FRICTION, DEFAULT_NORMAL_TOLERANCE_DEG,
    DEFAULT_NORMAL_STIFFNESS, CONSTRAINED_MAX_TRANSLATION,
    CONSTRAINED_TRANSLATION_STIFFNESS, FREE_MAX_TRANSLATION,
    FREE_TRANSLATION_STIFFNESS, PYRANOSE_RING_SIZE,
    RING_MATCHING_DISTANCE_THRESHOLD,
)
from ..geometry.geometry_utils import (
    get_positions, set_positions, get_atom_position,
    calculate_distance, calculate_angle,
    get_ring_normal_from_positions, calculate_center_of_mass,
    rodrigues_rotation, calculate_moment_of_inertia,
)
from .config import (
    OptimizationConfig, RingConstraintConfig, RingRotationUnit,
    RingIntegrityDetail, RingReferenceGeometry,
)

def detect_pyranose_rings(mol):
    """
    Detect 6-membered rings containing oxygen (pyranose sugars).
    """
    rings = []
    ring_info = mol.GetRingInfo()
    
    for ring in ring_info.AtomRings():
        # Check if 6-membered
        if len(ring) != PYRANOSE_RING_SIZE:
            continue
        
        # Check if contains oxygen
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        has_oxygen = any(atom.GetSymbol() == 'O' for atom in atoms)
        
        if has_oxygen:
            rings.append(list(ring))
    
    return rings

def get_ring_reference_geometry(mol, ring_atoms):
    """
    Store reference geometry for a ring (internal coordinates only).
    
    Args:
        mol: RDKit molecule
        ring_atoms: List of atom indices in the ring
    
    Returns:
        RingReferenceGeometry object
    """
    conf = mol.GetConformer()
    n_ring = len(ring_atoms)
    
    # Get all bonds in ring
    bonds = []
    for i in range(n_ring):
        atom1 = ring_atoms[i]
        atom2 = ring_atoms[(i + 1) % n_ring]
        
        pos1 = get_atom_position(conf, atom1)
        pos2 = get_atom_position(conf, atom2)
        length = calculate_distance(pos1, pos2)
        
        bonds.append((atom1, atom2, length))
    
    # Get all angles in ring
    angles = []
    for i in range(n_ring):
        atom1 = ring_atoms[i]
        atom2 = ring_atoms[(i + 1) % n_ring]
        atom3 = ring_atoms[(i + 2) % n_ring]
        
        pos1 = get_atom_position(conf, atom1)
        pos2 = get_atom_position(conf, atom2)
        pos3 = get_atom_position(conf, atom3)
        
        angle_deg = calculate_angle(pos1, pos2, pos3)
        angles.append((atom1, atom2, atom3, angle_deg))
    
    return RingReferenceGeometry(
        atoms=ring_atoms,
        bonds=bonds,
        angles=angles
    )

def check_ring_integrity(mol, ring_references, tolerance_bond, tolerance_angle):
    """
    Check if rings have deformed beyond tolerance.
    Only checks internal geometry (bonds, angles).
    
    Args:
        mol: RDKit molecule
        ring_references: List of RingReferenceGeometry objects
        tolerance_bond: Bond length tolerance in Angstroms
        tolerance_angle: Angle tolerance in degrees
    
    Returns:
        Tuple of (all_ok: bool, details: List[RingIntegrityDetail])
    """
    conf = mol.GetConformer()
    all_ok = True
    details = []
    
    for ring_id, ref in enumerate(ring_references):
        ring_atoms = ref.atoms
        
        # Check bonds
        max_bond_dev = 0.0
        for atom1, atom2, ref_length in ref.bonds:
            pos1 = get_atom_position(conf, atom1)
            pos2 = get_atom_position(conf, atom2)
            
            current_length = calculate_distance(pos1, pos2)
            deviation = abs(current_length - ref_length)
            max_bond_dev = max(max_bond_dev, deviation)
        
        # Check angles
        max_angle_dev = 0.0
        for atom1, atom2, atom3, ref_angle in ref.angles:
            pos1 = get_atom_position(conf, atom1)
            pos2 = get_atom_position(conf, atom2)
            pos3 = get_atom_position(conf, atom3)
            
            current_angle = calculate_angle(pos1, pos2, pos3)
            deviation = abs(current_angle - ref_angle)
            max_angle_dev = max(max_angle_dev, deviation)

        ring_ok = (max_bond_dev < tolerance_bond and 
                   max_angle_dev < tolerance_angle)
        
        if not ring_ok:
            all_ok = False
            print(f"  Ring {ring_id} deformed: "
                  f"bond_dev={max_bond_dev:.3f}Å, angle_dev={max_angle_dev:.1f}°")
        
        details.append(RingIntegrityDetail(
            ring_id=ring_id,
            atoms=ring_atoms,
            max_bond_dev=max_bond_dev,
            max_angle_dev=max_angle_dev,
            ok=ring_ok
        ))
    
    return all_ok, details

def get_deformed_ring_atoms(integrity_details):
    """
    Extract atom indices from deformed rings.
    
    Args:
        integrity_details: List of RingIntegrityDetail objects
    
    Returns:
        List of unique atom indices from deformed rings
    """
    deformed_atoms = []
    
    for detail in integrity_details:
        if not detail.ok:
            deformed_atoms.extend(detail.atoms)
            print(f"  Ring {detail.ring_id} deformed - "
                  f"freezing {len(detail.atoms)} atoms")
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(deformed_atoms))

def check_and_update_rings(mol, ring_references, tol_bond, tol_angle, 
                           last_valid, step):
    """
    Check ring integrity and return deformed ring atoms.
    
    Args:
        mol: Current molecule
        ring_references: List of reference geometries
        tol_bond: Bond tolerance
        tol_angle: Angle tolerance
        last_valid: Last valid molecule (unused but kept for compatibility)
        step: Current step number
    
    Returns:
        Tuple of (all_rings_ok: bool, deformed_atoms: list of int)
    """
    rings_ok, details = check_ring_integrity(
        mol, ring_references, tol_bond, tol_angle
    )
    
    if not rings_ok:
        deformed_atoms = get_deformed_ring_atoms(details)
        return False, deformed_atoms
    
    return True, []

def get_ring_substituents(mol, ring_atoms, all_ring_atoms_global):
    """
    Get all atoms that should rotate with the ring.
    Stops at glycosidic linkages (connections to other rings).
    
    Args:
        mol: RDKit molecule
        ring_atoms: Atoms in THIS ring
        all_ring_atoms_global: Set of ALL ring atoms in the molecule
    
    Returns:
        List of atom indices to rotate (includes ring + substituents)
    """
    ring_set = set(ring_atoms)
    to_rotate = set(ring_atoms)
    
    for ring_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(ring_idx)
        
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            
            # Skip if it's another atom in THIS ring
            if neighbor_idx in ring_set:
                continue
            
            # Stop at glycosidic linkages (connections to other rings)
            if neighbor_idx in all_ring_atoms_global:
                continue
            
            # BFS to get all substituents
            queue = [neighbor_idx]
            visited = {neighbor_idx}
            to_rotate.add(neighbor_idx)
            
            while queue:
                current_idx = queue.pop(0)
                current_atom = mol.GetAtomWithIdx(current_idx)
                
                for next_neighbor in current_atom.GetNeighbors():
                    next_idx = next_neighbor.GetIdx()
                    
                    if next_idx in visited or next_idx in all_ring_atoms_global:
                        continue
                    
                    visited.add(next_idx)
                    to_rotate.add(next_idx)
                    queue.append(next_idx)
    
    return list(to_rotate)

def apply_translation_constraint(positions, forces, ring_atoms, ring_masses,
                                 com_current, com_initial, max_translation,
                                 translation_stiffness):
    """
    Apply translation constraint to keep ring near initial position.
    
    Returns:
        Distance of COM from initial position
    """
    displacement = com_current - com_initial
    distance = np.linalg.norm(displacement)
    
    if distance > max_translation:
        excess_displacement = distance - max_translation
        restoring_force_magnitude = translation_stiffness * excess_displacement
        force_direction = -displacement / distance
        
        total_mass = np.sum(ring_masses)
        for i, atom_idx in enumerate(ring_atoms):
            mass_fraction = ring_masses[i] / total_mass
            forces[atom_idx] += force_direction * restoring_force_magnitude * mass_fraction
    
    return distance

def apply_normal_constraint(positions, forces, ring_atoms, ring_masses,
                            com_current, normal_current, normal_ref,
                            tolerance_deg, stiffness):
    """
    Apply constraint to maintain ring normal orientation.
    
    Returns:
        Angle deviation in degrees
    """
    dot_product = np.clip(np.dot(normal_current, normal_ref), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg > tolerance_deg:
        correction_axis = np.cross(normal_current, normal_ref)
        correction_axis_norm = np.linalg.norm(correction_axis)
        
        if correction_axis_norm > EPSILON:
            correction_axis = correction_axis / correction_axis_norm
            torque_correction = stiffness * (angle_deg - tolerance_deg)
            
            for i, atom_idx in enumerate(ring_atoms):
                r_vec = positions[atom_idx] - com_current
                r_perp = np.cross(correction_axis, r_vec)
                r_sq = max(np.dot(r_vec, r_vec), EPSILON)
                
                correction_force = (torque_correction * r_perp / r_sq) * ring_masses[i]
                forces[atom_idx] += correction_force
    
    return angle_deg

def calculate_ring_torque(positions, forces, ring_atoms, com, normal):
    """
    Calculate torque around ring normal through COM.
    
    Returns:
        Torque magnitude
    """
    torque_z = 0.0
    for atom_idx in ring_atoms:
        r_vec = positions[atom_idx] - com
        f_vec = forces[atom_idx]
        cross_prod = np.cross(r_vec, f_vec)
        torque_z += np.dot(cross_prod, normal)
    
    return torque_z

def update_angular_velocity(ring_unit, torque, timestep, friction):
    """
    Update angular velocity with friction damping.
    
    Args:
        ring_unit: Dictionary containing ring rotation data
        torque: Applied torque
        timestep: Time step
        friction: Friction coefficient
    """
    I_zz = ring_unit['moment_inertia']
    angular_accel = torque / I_zz
    
    ring_unit['angular_velocity'] *= (1.0 - friction * timestep)
    ring_unit['angular_velocity'] += angular_accel * timestep

def apply_rotation_to_ring(positions, ring_atoms, com, normal, angle):
    """
    Rotate ring atoms around normal through COM.
    
    Args:
        positions: Array of all atomic positions (modified in place)
        ring_atoms: Indices of atoms to rotate
        com: Center of mass
        normal: Rotation axis (unit vector)
        angle: Rotation angle in radians
    """
    if abs(angle) < EPSILON:
        return
    
    for atom_idx in ring_atoms:
        positions[atom_idx] = rodrigues_rotation(
            positions[atom_idx], normal, angle, com
        )

def apply_constrained_ring_dynamics(positions, forces, ring_unit, masses,
                                    timestep, config, debug=False):
    """
    Apply constrained (CD-mode) ring dynamics.
    
    Ring can rotate around its fixed normal but maintains orientation.
    Translation is tightly constrained.
    """
    ring_atoms = ring_unit['atoms']
    ring_positions = positions[ring_atoms]
    ring_masses = masses[ring_atoms]
    
    com_current = calculate_center_of_mass(ring_positions, ring_masses)
    com_initial = ring_unit['com_initial']
    normal_ref = ring_unit['normal_fixed']
    normal_current = get_ring_normal_from_positions(positions, ring_atoms)
    
    # Apply normal constraint
    angle_dev = apply_normal_constraint(
        positions, forces, ring_atoms, ring_masses,
        com_current, normal_current, normal_ref,
        config.constrained_normal_tolerance,
        config.constrained_normal_stiffness
    )
    
    # Apply translation constraint
    com_drift = apply_translation_constraint(
        positions, forces, ring_atoms, ring_masses,
        com_current, com_initial,
        config.constrained_max_translation,
        config.constrained_translation_stiffness
    )
    
    # Calculate and apply rotation
    torque_z = calculate_ring_torque(positions, forces, ring_atoms, 
                                     com_current, normal_current)
    
    update_angular_velocity(ring_unit, torque_z, timestep, 
                          config.constrained_friction)
    
    d_theta = ring_unit['angular_velocity'] * timestep
    ring_unit['total_rotation'] += d_theta
    
    apply_rotation_to_ring(positions, ring_atoms, com_current, 
                          normal_current, d_theta)
    
    if debug:
        print(f"    Ring {ring_unit['ring_id']} (CONSTRAINED):")
        print(f"      COM drift: {com_drift:.3f} Å")
        print(f"      Normal angle: {angle_dev:.2f}°")
        print(f"      Rotation: {np.degrees(d_theta):.3f}°")

def apply_free_ring_dynamics(positions, forces, ring_unit, masses, config, debug=False):
    """
    Apply free (3D-mode) ring dynamics.
    
    Ring can move and rotate freely with gentle position restraint.
    """
    ring_atoms = ring_unit['atoms']
    ring_positions = positions[ring_atoms]
    ring_masses = masses[ring_atoms]
    
    com_current = calculate_center_of_mass(ring_positions, ring_masses)
    com_initial = ring_unit['com_initial']
    
    # Gentle translation restraint only
    com_drift = apply_translation_constraint(
        positions, forces, ring_atoms, ring_masses,
        com_current, com_initial,
        config.free_max_translation,
        config.free_translation_stiffness
    )
    
    if debug:
        print(f"    Ring {ring_unit['ring_id']} (FREE):")
        print(f"      COM drift: {com_drift:.3f} Å (max: {config.free_max_translation:.1f})")

def apply_ring_constraints_dual_mode(conf, forces, ring_rotation_units,
                                     masses, timestep, n_atoms,
                                     config: RingConstraintConfig):
    """
    Apply two different constraint modes:
    - Constrained rings: CD-like (lock normal, rotate around Z only)
    - Free rings: Full 3D (gentle position restraint only)
    
    Args:
        conf: RDKit conformer
        forces: Force array
        ring_rotation_units: List of ring rotation dictionaries
        masses: Atomic masses
        timestep: Time step
        n_atoms: Number of atoms
        config: RingConstraintConfig object
    """
    if not ring_rotation_units:
        return
    
    positions = get_positions(conf, n_atoms)
    
    for ring_unit in ring_rotation_units:
        is_constrained = ring_unit.get('is_constrained', False)
        
        if is_constrained:
            apply_constrained_ring_dynamics(
                positions, forces, ring_unit, masses,
                timestep, config, config.debug
            )
        else:
            apply_free_ring_dynamics(
                positions, forces, ring_unit, masses,
                config, config.debug
            )
    
    # Update conformer with new positions
    set_positions(conf, positions, n_atoms)

# Legacy function for backward compatibility
def apply_ring_rotation_constraint(conf, forces, ring_rotation_units,
                                  fixed_atoms, masses, timestep, n_atoms,
                                  friction=0.1,
                                  constrain_normal=True,
                                  normal_tolerance_deg=5.0,
                                  normal_stiffness=100.0,
                                  constrain_translation=True,
                                  max_translation=10.0,
                                  translation_stiffness=100.0,
                                  debug=False):
    """
    Legacy interface - converts parameters to config object.
    """
    config = RingConstraintConfig(
        constrained_friction=friction,
        constrained_normal_tolerance=normal_tolerance_deg,
        constrained_normal_stiffness=normal_stiffness,
        constrained_max_translation=max_translation,
        constrained_translation_stiffness=translation_stiffness,
        debug=debug
    )
    
    # Mark all rings as constrained for legacy behavior
    for ring_unit in ring_rotation_units:
        ring_unit['is_constrained'] = True
    
    apply_ring_constraints_dual_mode(
        conf, forces, ring_rotation_units,
        masses, timestep, n_atoms, config
    )
