import numpy as np
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from openbabel import openbabel as ob
from ..geometry.geometry_utils import (
    get_perpendicular_vector,
    VDW_RADII,
)

#TO:DO Implemenet CC bond simple

  
def _compute_ring_normal(mol_data, coords):
    """Return (unit_normal, centroid) of the pyranose ring plane, or (None, None)."""
    carbon_map = mol_data.get('carbon_map', {})
    ring_carbons = carbon_map.get('all_ring_carbons', [])
    ring_o_idx = carbon_map.get('ring_oxygen')

    ring_indices = list(ring_carbons)
    if ring_o_idx is not None:
        ring_indices.append(ring_o_idx)

    valid = [i for i in ring_indices if i < len(coords)]
    if len(valid) < 3:
        return None, None

    ring_pos = np.array([coords[i] for i in valid])
    centroid = ring_pos.mean(axis=0)
    _, _, vh = np.linalg.svd(ring_pos - centroid)
    return vh[-1], centroid  # unit normal (SVD guarantees unit length), centroid


def _scan_oh_by_distance(mol_data, coords, c_pos):
    """LEGACY coordinate-based hydroxyl detection — fallback only.

    Finds the first non-ring oxygen within 1.6 A of ``c_pos`` and that oxygen's
    hydrogen within 1.2 A, returning ``[O_idx, H_idx]`` (H omitted if none).
    Kept as a fallback for molecule_data that predates ``oh_map`` (e.g. old
    .pkl files) or lacks bond-graph connectivity. New data uses the bond graph
    via ``_resolve_oh_to_remove``.
    """
    atoms_to_remove = []
    ring_o_idx = mol_data['carbon_map'].get('ring_oxygen')
    atom_types = mol_data['atom_types']

    for i, atom_type in enumerate(atom_types):
        if i >= len(coords):
            break
        if atom_type == 'O':
            if ring_o_idx is not None and i == ring_o_idx:
                continue
            if np.linalg.norm(coords[i] - c_pos) < 1.6:
                atoms_to_remove.append(i)
                o_pos = coords[i]
                for j, at in enumerate(atom_types):
                    if at == 'H' and np.linalg.norm(coords[j] - o_pos) < 1.2:
                        atoms_to_remove.append(j)
                        break
                break
    return atoms_to_remove


def _resolve_oh_to_remove(mol_data, carbon_name, coords, c_pos):
    """Return [O_idx, H_idx] hydroxyl atoms to remove at ``carbon_name``.

    Prefers the precomputed bond-graph ``oh_map``; falls back to the legacy
    coordinate scan when ``oh_map`` is absent (old data) or has no entry for
    this carbon. This is the safety net that keeps existing inputs working.
    """
    oh_map = mol_data.get('oh_map')
    if oh_map and carbon_name in oh_map:
        o_idx, h_idx = oh_map[carbon_name]
        result = [o_idx]
        if h_idx is not None:
            result.append(h_idx)
            print(f"  oh_map: remove O{o_idx} and H{h_idx} at {carbon_name}")
        else:
            print(f"  oh_map: remove O{o_idx} at {carbon_name} (no explicit H)")
        return result

    print(f"  oh_map miss at {carbon_name} - using legacy distance scan")
    return _scan_oh_by_distance(mol_data, coords, c_pos)


def create_glycosidic_bond(mol1_data, carbon1_name, mol2_data, carbon2_name, anomeric='beta', linkage_name='1-6'):

    """
    Create a glycosidic bond and return info about which OH to remove.
    
    Returns:
        Dictionary with:
        - oxygen_position: position of new bridging oxygen
        - atoms_to_remove: list of (mol_name, atom_indices) for original OH groups
    """
    
    C_O_BOND_LENGTH = 1.43
    C_O_C_ANGLE = 117.0
    
    # Get carbon positions
    carbon1_idx = mol1_data['carbon_map'][carbon1_name]
    carbon2_idx = mol2_data['carbon_map'][carbon2_name]
    
    coords1 = np.array(mol1_data['absolute_coordinates'])
    coords2 = np.array(mol2_data['absolute_coordinates'])
    
    c1_pos = coords1[carbon1_idx]
    c2_pos = coords2[carbon2_idx]
    
    print(f"\n{'='*60}")
    print(f"Creating {anomeric} {linkage_name} glycosidic bond")
    
    # Calculate bridging oxygen position (same as before)
    c1_to_c2 = c2_pos - c1_pos
    c1_c2_distance = np.linalg.norm(c1_to_c2)
    
    angle_rad = np.radians(C_O_C_ANGLE)
    ideal_c_c_distance = np.sqrt(2 * C_O_BOND_LENGTH**2 * (1 - np.cos(angle_rad)))
    
    c1_to_c2_norm = c1_to_c2 / c1_c2_distance
    cos_angle_at_c1 = (C_O_BOND_LENGTH**2 + c1_c2_distance**2 - C_O_BOND_LENGTH**2) / (2 * C_O_BOND_LENGTH * c1_c2_distance)
    cos_angle_at_c1 = np.clip(cos_angle_at_c1, -1.0, 1.0)
    
    projection_dist = C_O_BOND_LENGTH * cos_angle_at_c1
    height = C_O_BOND_LENGTH * np.sqrt(1 - cos_angle_at_c1**2)
    
    # Use the donor ring-plane normal so the oxygen is always placed above/below
    # the ring, never through it.  Fall back to an arbitrary perpendicular only
    # when ring atom data are unavailable or collinear with the bond axis.
    ring_normal, ring_centroid = _compute_ring_normal(mol1_data, coords1)
    perpendicular = None

    if ring_normal is not None:
        # Remove the component parallel to the C1→C2 axis
        proj = np.dot(ring_normal, c1_to_c2_norm)
        candidate = ring_normal - proj * c1_to_c2_norm
        norm_mag = np.linalg.norm(candidate)
        if norm_mag > 0.1:
            perpendicular = candidate / norm_mag
            print(f"  Using ring-normal perpendicular (ring out-of-plane guaranteed)")

    if perpendicular is None:
        # Degenerate: bond axis is nearly parallel to ring normal.
        # Pick a direction that lies IN the ring plane so the displacement
        # stays perpendicular to the ring normal and cannot cross it.
        if ring_normal is not None:
            ref = np.array([1, 0, 0]) if abs(ring_normal[0]) < 0.9 else np.array([0, 1, 0])
            in_plane = np.cross(ring_normal, ref)
            perpendicular = in_plane / np.linalg.norm(in_plane)
            print(f"  Degenerate: bond≈ring_normal — using ring-plane direction")
        else:
            # No ring data at all: last-resort arbitrary perpendicular
            if abs(c1_to_c2_norm[2]) < 0.9:
                perpendicular_base = np.cross(c1_to_c2_norm, np.array([0, 0, 1]))
            else:
                perpendicular_base = np.cross(c1_to_c2_norm, np.array([1, 0, 0]))
            perpendicular = perpendicular_base / np.linalg.norm(perpendicular_base)
            print(f"  Using fallback perpendicular (no ring data)")

    if anomeric == 'alpha':
        perpendicular = -perpendicular

    oxygen_pos = c1_pos + projection_dist * c1_to_c2_norm + height * perpendicular

    # Post-placement verification: regardless of which path chose perpendicular,
    # confirm the oxygen is on the correct side of the ring SVD plane and flip if not.
    if ring_normal is not None and ring_centroid is not None:
        oxygen_side = np.dot(oxygen_pos - ring_centroid, ring_normal)
        want_above = (anomeric == 'beta')
        wrong_side = (want_above and oxygen_side < 0) or (not want_above and oxygen_side > 0)
        if wrong_side:
            perpendicular = -perpendicular
            oxygen_pos = c1_pos + projection_dist * c1_to_c2_norm + height * perpendicular
            print(f"  Post-check: flipped oxygen to correct side for {anomeric} (was {oxygen_side:+.3f})")
        else:
            print(f"  Post-check: oxygen on correct side for {anomeric} (side={oxygen_side:+.3f})")
    
    # Verify geometry
    o_c1_dist = np.linalg.norm(oxygen_pos - c1_pos)
    o_c2_dist = np.linalg.norm(oxygen_pos - c2_pos)
    
    v1 = c1_pos - oxygen_pos
    v2 = c2_pos - oxygen_pos
    cos_c_o_c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    actual_angle = np.degrees(np.arccos(np.clip(cos_c_o_c, -1.0, 1.0)))
    
    print(f"Glycosidic oxygen geometry:")
    print(f"  C1-O distance: {o_c1_dist:.3f} Å")
    print(f"  O-C2 distance: {o_c2_dist:.3f} Å")
    print(f"  C1-O-C2 angle: {actual_angle:.1f}°")
    
    # NEW: Identify original OH groups to remove
    # For C1 (anomeric carbon on donor) - remove its OH
    atoms_to_remove_mol1 = []

    
    if 'KDO' in mol1_data.get('molecule_name', '') and carbon1_name == 'C1':
    
        print("  Special handling for KDO C1 (carboxyl group)")
        
        # For carboxyl, find both oxygens
        carboxyl_oxygens = []
        for i, atom_type in enumerate(mol1_data['atom_types']):
            if atom_type == 'O':
                dist = np.linalg.norm(coords1[i] - c1_pos)
                if dist < 1.6:
                    carboxyl_oxygens.append(i)
        
        # Identify which oxygen is OH (has hydrogen) vs C=O (no hydrogen)
        for o_idx in carboxyl_oxygens:
            o_pos = coords1[o_idx]
            has_hydrogen = False
            h_idx = None
            
            # Check for bonded hydrogen
            for j, at in enumerate(mol1_data['atom_types']):
                if at == 'H':
                    if np.linalg.norm(coords1[j] - o_pos) < 1.2:  # O-H bond
                        has_hydrogen = True
                        h_idx = j
                        break
            
            if has_hydrogen:
                # This is the OH part - remove both O and H
                atoms_to_remove_mol1.append(o_idx)
                atoms_to_remove_mol1.append(h_idx)
                print(f"  Will remove O{o_idx} and H{h_idx} from KDO C1 (OH part of COOH)")
                break  # Only remove the OH, not the C=O
    else:
        # Bond-graph hydroxyl lookup (falls back to legacy distance scan for
        # old data without oh_map). mol1 is the donor at carbon1_name.
        atoms_to_remove_mol1 = _resolve_oh_to_remove(
            mol1_data, carbon1_name, coords1, c1_pos)

    # For C2 (acceptor carbon) - remove its OH (always)
    atoms_to_remove_mol2 = _resolve_oh_to_remove(
        mol2_data, carbon2_name, coords2, c2_pos)

    return {
        'linkage': f"{anomeric} {linkage_name}",
        'oxygen_position': oxygen_pos.tolist(),
        'c1_position': c1_pos.tolist(),
        'c2_position': c2_pos.tolist(),
        'c1_o_distance': float(o_c1_dist),
        'o_c2_distance': float(o_c2_dist),
        'c_o_c_angle': float(actual_angle),
        'c1_c2_distance': float(c1_c2_distance),
        'donor_oh_to_remove': atoms_to_remove_mol2,
        'acceptor_oh_to_remove': atoms_to_remove_mol1
    }

def create_phosphate_bond_from_IK(mol1_data, carbon1_name,
                                   mol2_data, carbon2_name,
                                   phosphate_position_IK,
                                   linkage_name='phosphodiester'):
    """
    Create phosphodiester bond using IK-calculated phosphate position.
    Identifies hydroxyl groups to remove.
    """
    
    # Bond lengths
    C_O_BOND = 1.43
    O_P_BOND = 1.60
    P_O_DOUBLE = 1.48
    P_O_SINGLE = 1.57
    
    # Get carbon positions
    carbon1_idx = mol1_data['carbon_map'][carbon1_name]
    carbon2_idx = mol2_data['carbon_map'][carbon2_name]
    
    coords1 = np.array(mol1_data['absolute_coordinates'])
    coords2 = np.array(mol2_data['absolute_coordinates'])
    
    c1_pos = coords1[carbon1_idx]
    c2_pos = coords2[carbon2_idx]
    p_pos = np.array(phosphate_position_IK)
    
    print(f"\n{'='*60}")
    print(f"Creating {linkage_name} phosphodiester bond (IK-based)")
    print(f"C1 ({carbon1_name}): {c1_pos}")
    print(f"C2 ({carbon2_name}): {c2_pos}")
    print(f"P position (from IK): {p_pos}")
    
    # ========================================================================
    # Calculate O1 (between C1 and P) - placed along C1→P line
    # ========================================================================
    c1_to_p = p_pos - c1_pos
    c1_p_dist = np.linalg.norm(c1_to_p)
    c1_to_p_norm = c1_to_p / c1_p_dist
    
    # Place O1 at 1.43 Å from C1 along C1→P
    o1_pos = c1_pos + c1_to_p_norm * C_O_BOND
    
    # Verify
    o1_c1_dist = np.linalg.norm(o1_pos - c1_pos)
    o1_p_dist = np.linalg.norm(o1_pos - p_pos)
    
    print(f"\nO1 (bridging oxygen C1-P):")
    print(f"  Position: {o1_pos}")
    print(f"  C1-O1: {o1_c1_dist:.3f}Å (target: {C_O_BOND:.3f})")
    print(f"  O1-P: {o1_p_dist:.3f}Å (target: {O_P_BOND:.3f})")
    
    # Calculate C1-O1-P angle
    v1 = c1_pos - o1_pos
    v2 = p_pos - o1_pos
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    c1_o1_p_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    print(f"  C1-O1-P angle: {c1_o1_p_angle:.1f}°")
    
    # ========================================================================
    # Calculate O2 (between P and C2) - placed along P→C2 line
    # ========================================================================
    p_to_c2 = c2_pos - p_pos
    p_c2_dist = np.linalg.norm(p_to_c2)
    p_to_c2_norm = p_to_c2 / p_c2_dist
    
    # Place O2 at 1.60 Å from P along P→C2
    o2_pos = p_pos + p_to_c2_norm * O_P_BOND
    
    # Verify
    o2_p_dist = np.linalg.norm(o2_pos - p_pos)
    o2_c2_dist = np.linalg.norm(o2_pos - c2_pos)
    
    print(f"\nO2 (bridging oxygen P-C2):")
    print(f"  Position: {o2_pos}")
    print(f"  O2-P: {o2_p_dist:.3f}Å (target: {O_P_BOND:.3f})")
    print(f"  O2-C2: {o2_c2_dist:.3f}Å (target: {C_O_BOND:.3f})")
    
    # Calculate P-O2-C2 angle
    v1 = p_pos - o2_pos
    v2 = c2_pos - o2_pos
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    p_o2_c2_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    print(f"  P-O2-C2 angle: {p_o2_c2_angle:.1f}°")
    
    # ========================================================================
    # Calculate O3 (=O) and O4 (-OH) using tetrahedral geometry
    # ========================================================================
    o3_pos, o4_pos = calculate_tetrahedral_oxygens(
        p_pos, o1_pos, o2_pos,
        P_O_DOUBLE, P_O_SINGLE
    )
    
    # Verify
    o3_p_dist = np.linalg.norm(o3_pos - p_pos)
    o4_p_dist = np.linalg.norm(o4_pos - p_pos)
    
    print(f"\nO3 (phosphoryl =O):")
    print(f"  Position: {o3_pos}")
    print(f"  P=O3: {o3_p_dist:.3f}Å (target: {P_O_DOUBLE:.3f})")
    
    print(f"\nO4 (hydroxyl -OH):")
    print(f"  Position: {o4_pos}")
    print(f"  P-O4: {o4_p_dist:.3f}Å (target: {P_O_SINGLE:.3f})")
    
    # Calculate C-C distance
    c1_c2_distance = np.linalg.norm(c2_pos - c1_pos)
    print(f"\nC1-C2 distance: {c1_c2_distance:.3f}Å")
    
    # Quality assessment
    max_bond_error = max(
        abs(o1_c1_dist - C_O_BOND),
        abs(o1_p_dist - O_P_BOND),
        abs(o2_p_dist - O_P_BOND),
        abs(o2_c2_dist - C_O_BOND),
        abs(o3_p_dist - P_O_DOUBLE),
        abs(o4_p_dist - P_O_SINGLE)
    )
    
    if max_bond_error < 0.05:
        quality = "Excellent"
    elif max_bond_error < 0.1:
        quality = "Good"
    else:
        quality = "Acceptable"
    
    print(f"\nQuality: {quality}")
    print(f"  Max bond error: {max_bond_error:.3f}Å")
    print("="*60)
    
    return {
        'linkage': linkage_name,
        'phosphorus_position': p_pos.tolist(),
        'oxygen1_position': o1_pos.tolist(),
        'oxygen2_position': o2_pos.tolist(),
        'oxygen3_position': o3_pos.tolist(),
        'oxygen4_position': o4_pos.tolist(),
        'c1_position': c1_pos.tolist(),
        'c2_position': c2_pos.tolist(),
        'c1_o1_distance': float(o1_c1_dist),
        'o1_p_distance': float(o1_p_dist),
        'p_o2_distance': float(o2_p_dist),
        'o2_c2_distance': float(o2_c2_dist),
        'p_o3_distance': float(o3_p_dist),
        'p_o4_distance': float(o4_p_dist),
        'c1_c2_distance': float(c1_c2_distance),
        'quality': quality,
        'donor_oh_to_remove': [],
        'acceptor_oh_to_remove': []
    }

def find_hydroxyl_oxygen_at_carbon(mol_data, carbon_idx, coords):
    """
    Find OH group on a carbon. Returns [O_idx, H_idx] or [].
    """
    c_pos = coords[carbon_idx]
    atom_types = mol_data['atom_types']
    
    # Get carbon name
    carbon_name = None
    for name, idx in mol_data['carbon_map'].items():
        if idx == carbon_idx:
            carbon_name = name
            break
    print(f"  DEBUG: Looking for OH on carbon_idx={carbon_idx} ({carbon_name})")
    print(f"    Carbon position: {c_pos}")
    
    # Special case: KDO C1 (carboxyl)
    if 'KDO' in mol_data.get('molecule_name', '') and carbon_name == 'C1':
        carboxyl_oxygens = []
        for i in range(len(coords)):
            if atom_types[i] == 'O':
                dist = np.linalg.norm(coords[i] - c_pos)
                if dist < 1.6:
                    carboxyl_oxygens.append(i)
        
        for o_idx in carboxyl_oxygens:
            o_pos = coords[o_idx]
            for j in range(len(coords)):
                if atom_types[j] == 'H':
                    if np.linalg.norm(coords[j] - o_pos) < 1.2:
                        return [o_idx, j]  # Found OH
        return []
    
    # Normal case: Use the working logic
    candidate_oxygens = []
    for i in range(len(coords)):
        if atom_types[i] == 'O':
            dist = np.linalg.norm(coords[i] - c_pos)
            if 1.3 < dist < 1.6:
                candidate_oxygens.append(i)
                print(f"      Found O{i} at distance {dist:.3f}Å")
    
    print(f"    Candidate oxygens: {candidate_oxygens}")
    
    for o_idx in candidate_oxygens:
        o_pos = coords[o_idx]

        print(f"    Checking O{o_idx}...")
        
        # Count bonded atoms
        bonded_atoms = 0
        for j in range(len(coords)):
            if j != o_idx:
                dist = np.linalg.norm(coords[j] - o_pos)
                if dist < 1.6:
                    bonded_atoms += 1

        print(f"      Bonded atoms: {bonded_atoms}")
        
        # Hydroxyl has 2 bonds (C-O and O-H)
        if bonded_atoms == 2:
            # Find the hydrogen
            h_found = False
            for j in range(len(coords)):
                if j != o_idx and atom_types[j] == 'H':
                    dist = np.linalg.norm(coords[j] - o_pos)
                    if dist < 1.1:
                        print(f"      Found H{j} at distance {dist:.3f}Å - MATCH!")
                        return [o_idx, j]
                    elif dist < 1.6:
                        print(f"      Found H{j} at distance {dist:.3f}Å - too far")
            
            if not h_found:
                print(f"      Has 2 bonds but no H found nearby!")
        else:
            print(f"      Wrong bond count (need 2, has {bonded_atoms})")

    print(f"    No valid OH found - returning []")
    return []
    
    # # Find oxygens near this carbon
    # candidate_oxygens = []
    # for i in range(len(coords)):
    #     if atom_types[i] == 'O':
    #         dist = np.linalg.norm(coords[i] - c_pos)
    #         if 1.3 < dist < 1.6:  # C-O bond range
    #             candidate_oxygens.append(i)
    
    # # For each candidate, check if it's a hydroxyl (has an H neighbor)
    # # We can approximate this by checking if the oxygen is "terminal"
    # # (not in a ring, only has 1-2 neighbors)
    
    # for o_idx in candidate_oxygens:
    #     o_pos = coords[o_idx]
        
    #     # Count how many atoms are bonded to this oxygen
    #     bonded_atoms = 0
    #     for j in range(len(coords)):
    #         if j != o_idx:
    #             dist = np.linalg.norm(coords[j] - o_pos)
    #             # Check for bonded atoms (C-O ~1.43Å, O-H ~0.96Å)
    #             if dist < 1.6:
    #                 bonded_atoms += 1
        
    #     # Hydroxyl oxygen should have 2 bonds (C-O and O-H)
    #     # Ring oxygen would have 2 bonds but both to carbons
    #     # We want the one that has a hydrogen
    #     if bonded_atoms == 2:
    #         # Check if one neighbor is hydrogen
    #         has_hydrogen = False
    #         for j in range(len(coords)):
    #             if j != o_idx and atom_types[j] == 'H':
    #                 dist = np.linalg.norm(coords[j] - o_pos)
    #                 if dist < 1.1:  # O-H bond
    #                     has_hydrogen = True
    #                     break
            
    #         if has_hydrogen:
    #             return o_idx
        
    # return None

def solve_phosphate_position_IK(base_carbon, target_carbon,min_height_above=0.5):
    """
    Use inverse kinematics to find optimal phosphate position.
    
    Treats phosphate as a 2-segment arm:
    - Segment 1: base_carbon → P (length 3.03 Å)
    - Segment 2: P → target_carbon (length 3.03 Å)
    
    Args:
        base_carbon: [x,y,z] position of base carbon (e.g., GlcN_1.C1)
        target_carbon: [x,y,z] position of target carbon (e.g., GalA.C1 after rotation)
        
    Returns:
        phosphate_position: [x,y,z] optimal position for phosphorus
        metadata: dict with solution quality
    """
    
    base = np.array(base_carbon)
    target = np.array(target_carbon)
    
    # Ideal segment lengths
    L1_ideal = 3.03  # C-O-P
    L2_ideal = 3.03  # P-O-C
    
    print(f"\n{'='*60}")
    print(f"Solving phosphate position")
    print(f"Base: {base}")
    print(f"Target: {target}")
    
    # Distance between base and target
    D = np.linalg.norm(target - base)

    direction_to_target = (target - base) / D
    
    # Get perpendicular direction - ALWAYS pointing upward
    if abs(direction_to_target[2]) < 0.9:
        perp = np.cross(direction_to_target, np.array([0, 0, 1]))
    else:
        perp = np.cross(direction_to_target, np.array([1, 0, 0]))
    
    perp = perp / np.linalg.norm(perp)
    
    # ALWAYS ensure perpendicular points upward (positive z component)
    if perp[2] < 0:
        perp = -perp
    
    print(f"Perpendicular direction (upward): {perp}")
    
    # ==================================================================
    # CASE 1: UNREACHABLE - Need to stretch bonds
    # ==================================================================
    if D > L1_ideal + L2_ideal:
        shortage = D - (L1_ideal + L2_ideal)
        print(f"⚠ Target UNREACHABLE! Gap: {shortage:.3f} Å")
        print(f"  → Will stretch bonds proportionally")
        
        # Calculate stretch factor
        stretch_factor = D / (L1_ideal + L2_ideal)
        L1 = L1_ideal * stretch_factor
        L2 = L2_ideal * stretch_factor
        
        print(f"  Stretch factor: {stretch_factor:.2f}x")
        print(f"  New L1: {L1:.3f} Å (was {L1_ideal:.3f})")
        print(f"  New L2: {L2:.3f} Å (was {L2_ideal:.3f})")
        
        # Place P at stretched distance along line
        # But add vertical component to ensure it goes OVER
        projection_length = L1
        
        # Add height to ensure phosphate is above both carbons
        min_z = max(base[2], target[2]) + min_height_above
        
        # Calculate P position with upward bias
        # Use 20% of L1 as perpendicular height to arch over
        perpendicular_height = L1 * 0.2
        
        p_position = base + projection_length * direction_to_target + perpendicular_height * perp
        
        # Enforce minimum height
        if p_position[2] < min_z:
            z_adjustment = min_z - p_position[2]
            p_position[2] = min_z
            print(f"  → Adjusted height by {z_adjustment:.3f} Å to meet minimum")
        
        quality = f"⚠ Stretched {stretch_factor:.2f}x (arched over)"
        
    # ==================================================================
    # CASE 2: TOO CLOSE - Compressed
    # ==================================================================
    elif D < abs(L1_ideal - L2_ideal):
        print(f"⚠ Target TOO CLOSE! Overlap: {abs(L1_ideal - L2_ideal) - D:.3f} Å")
        
        # Use ideal lengths but add significant height
        L1 = L1_ideal
        L2 = L2_ideal
        
        # Create a high arc to avoid collision
        projection_length = D / 2
        perpendicular_height = L1 * 0.5  # Much higher arc
        
        p_position = base + projection_length * direction_to_target + perpendicular_height * perp
        
        # Enforce minimum height
        min_z = max(base[2], target[2]) + min_height_above
        if p_position[2] < min_z:
            p_position[2] = min_z
        
        quality = "⚠ Too close - high arc"
        L1 = L1_ideal
        L2 = L2_ideal
        
    # ==================================================================
    # CASE 3: REACHABLE - Optimal IK solution
    # ==================================================================
    else:
        print(f"✓ Target is reachable!")
        
        L1 = L1_ideal
        L2 = L2_ideal
        
        # Using law of cosines to find angle at base
        cos_angle_base = (L1**2 + D**2 - L2**2) / (2 * L1 * D)
        cos_angle_base = np.clip(cos_angle_base, -1.0, 1.0)
        angle_at_base = np.arccos(cos_angle_base)
        
        print(f"Angle at base: {np.degrees(angle_at_base):.1f}°")
        
        # Calculate P position using the angle
        projection_length = L1 * np.cos(angle_at_base)
        perpendicular_height = L1 * np.sin(angle_at_base)
        
        # Place P with upward perpendicular
        p_position = base + projection_length * direction_to_target + perpendicular_height * perp
        
        # Enforce minimum height
        min_z = max(base[2], target[2]) + min_height_above
        if p_position[2] < min_z:
            z_adjustment = min_z - p_position[2]
            p_position[2] = min_z
            print(f"  → Adjusted height by {z_adjustment:.3f} Å to meet minimum")
        
        quality = "✓✓ Optimal IK solution (arched over)"
    
    # ==================================================================
    # Verify solution
    # ==================================================================
    dist_base_p = np.linalg.norm(p_position - base)
    dist_p_target = np.linalg.norm(target - p_position)
    
    print(f"\nSolution:")
    print(f"  Phosphate position: {p_position}")
    print(f"  Base-P distance: {dist_base_p:.3f} Å (target: {L1:.3f})")
    print(f"  P-Target distance: {dist_p_target:.3f} Å (target: {L2:.3f})")
    print(f"  Height above max carbon: {p_position[2] - max(base[2], target[2]):.3f} Å")
    print(f"  Quality: {quality}")
    print("="*60)
    
    return p_position, {
        'base_target_distance': D,
        'base_p_distance': dist_base_p,
        'p_target_distance': dist_p_target,
        'stretched': D > L1_ideal + L2_ideal,
        'stretch_factor': dist_base_p / L1_ideal if D > L1_ideal + L2_ideal else 1.0,
        'height_above_carbons': p_position[2] - max(base[2], target[2]),
        'quality': quality
    }

def calculate_tetrahedral_oxygens(p_pos, o1_pos, o2_pos, p_o3_length, p_o4_length):
    """
    Calculate positions of O3 (=O) and O4 (-OH) to complete tetrahedral geometry around P.
    
    Args:
        p_pos: Phosphorus position
        o1_pos: First bridging oxygen position
        o2_pos: Second bridging oxygen position  
        p_o3_length: P=O bond length (1.48 Å)
        p_o4_length: P-OH bond length (1.57 Å)
        
    Returns:
        (o3_pos, o4_pos) - positions of phosphoryl and hydroxyl oxygens
    """
    p = np.array(p_pos)
    o1 = np.array(o1_pos)
    o2 = np.array(o2_pos)
    
    # Get vectors from P to O1 and O2
    p_o1 = (o1 - p) / np.linalg.norm(o1 - p)
    p_o2 = (o2 - p) / np.linalg.norm(o2 - p)
    
    # Average direction (bisector of O1-P-O2)
    bisector = (p_o1 + p_o2) / 2
    bisector_norm = bisector / np.linalg.norm(bisector)
    
    # For tetrahedral geometry, O3 and O4 should be on opposite sides
    # of the O1-P-O2 plane
    
    # Get a perpendicular to the O1-P-O2 plane
    plane_normal = np.cross(p_o1, p_o2)
    if np.linalg.norm(plane_normal) < 1e-6:
        # O1 and O2 are collinear (unlikely but handle it)
        if abs(p_o1[2]) < 0.9:
            plane_normal = np.cross(p_o1, np.array([0, 0, 1]))
        else:
            plane_normal = np.cross(p_o1, np.array([1, 0, 0]))
    
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # For tetrahedral angle (109.5°), calculate the direction
    # O3 should point roughly opposite to the bisector, but adjusted for tetrahedral angle
    tetrahedral_angle = np.radians(109.5)
    
    # Mix bisector and plane normal to get tetrahedral positions
    # This is approximate - proper tetrahedral would require more careful calculation
    angle_from_bisector = np.radians(70)  # Approximate adjustment
    
    # O3 direction (one side of plane)
    o3_dir = -bisector_norm * np.cos(angle_from_bisector) + plane_normal * np.sin(angle_from_bisector)
    o3_dir = o3_dir / np.linalg.norm(o3_dir)
    o3_pos = p + o3_dir * p_o3_length
    
    # O4 direction (opposite side of plane)
    o4_dir = -bisector_norm * np.cos(angle_from_bisector) - plane_normal * np.sin(angle_from_bisector)
    o4_dir = o4_dir / np.linalg.norm(o4_dir)
    o4_pos = p + o4_dir * p_o4_length
    
    return o3_pos, o4_pos