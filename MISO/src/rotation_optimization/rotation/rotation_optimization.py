"""
rotation_search.py — rotation search algorithms for monomer alignment.

Implements the QUEST attitude estimator, uniform quaternion sampling,
ring/collision clearance checks, and face-aware (normal vs reflected)
solution selection.

No RDKit molecule modification takes place here.  All functions work on
coordinate arrays extracted from molecule_data_dict entries.

Main entry points called by structure/building_chain.py
-------------------------------------------------------
find_rotation_FULL_3D_QUATERNION
    Single-bond or multi-bond 3D rotation search (no orientation constraint).

find_rotation_1D_Z_AXIS
    Z-axis spin search for rings constrained to lie flat on the surface.

find_rotation_QUEST_VARIANTS_TOP_N
    Returns the top-N rotations across QUEST variants and uniform samples;
    used by align_and_position_molecules_TOP_N_POLYMERS.

find_rotation_for_phosphate_linkage
    Torsion search for phosphodiester linkage positioning.

orient_root_molecule
    Align the root (fixed) monomer to a surface orientation constraint.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..geometry.geometry_utils import (
    rotate_molecule_around_com,
    rotate_molecule_z,
    get_ring_normal,
    get_ring_normal_absolute,
    get_ring_normal_from_positions,
    calculate_alignment_rotation,
    rotation_around_axis,
    VDW_RADII,
)


# ============================================================================
# QUEST attitude estimator
# ============================================================================

def solve_wahba_quest(observations, references, weights=None):
    """
    Solve Wahba's problem via the QUEST algorithm.

    Finds the rotation R that minimises:
        Σ w_i || r_i - R · o_i ||²

    Parameters
    ----------
    observations : list[array-like]  shape (3,) each
        Unit vectors in the body (molecule) frame.
    references : list[array-like]  shape (3,) each
        Corresponding unit vectors in the world frame.
    weights : list[float] | None
        Per-pair weights.  Defaults to uniform.

    Returns
    -------
    np.ndarray  shape (3, 3)
        Optimal rotation matrix.
    """
    n = len(observations)
    if weights is None:
        weights = np.ones(n)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    observations = [np.array(v) / np.linalg.norm(v) for v in observations]
    references = [np.array(v) / np.linalg.norm(v) for v in references]

    B = sum(w * np.outer(ref, obs)
            for w, obs, ref in zip(weights, observations, references))

    S = B + B.T
    Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])
    sigma = np.trace(B)

    K = np.block([[sigma,           Z.reshape(1,3)],
                  [Z.reshape(3,1),  S - sigma*np.eye(3)]])

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q = eigenvectors[:, -1]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    return rot.as_matrix()


# ============================================================================
# Geometric primitives
# ============================================================================

def flip_molecule_coordinates(mol_data, com):
    """
    Mirror molecule through the Y = COM[1] plane (reflect face).

    Returns
    -------
    dict  copy of mol_data with flipped absolute_coordinates.
    """
    mol_flipped = mol_data.copy()
    coords = np.array(mol_data['absolute_coordinates'])
    com_arr = np.array(com)
    coords_flipped = coords.copy()
    coords_flipped[:, 1] = 2 * com_arr[1] - coords[:, 1]
    mol_flipped['absolute_coordinates'] = coords_flipped.tolist()
    return mol_flipped


def generate_uniform_rotations(n_samples):
    """
    Generate quasi-uniform rotation matrices via Fibonacci sphere sampling.

    Parameters
    ----------
    n_samples : int

    Returns
    -------
    list[np.ndarray]  each shape (3, 3)
    """
    phi = (1 + np.sqrt(5)) / 2
    rotations = []
    for i in range(n_samples):
        y = 1 - (i / max(n_samples - 1, 1)) * 2
        radius = np.sqrt(max(0.0, 1 - y * y))
        theta = 2 * np.pi * i / phi
        axis = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis /= norm
            angle = np.random.uniform(0, 2 * np.pi)
            rotations.append(R.from_rotvec(angle * axis).as_matrix())
        else:
            rotations.append(np.eye(3))
    return rotations


def get_ring_positions(mol):
    """
    Return world positions of all ring atoms (carbons + ring oxygen).

    Parameters
    ----------
    mol : dict
        molecule_data_dict entry.

    Returns
    -------
    list[np.ndarray]  each shape (3,)
    """
    coords = mol['absolute_coordinates']
    ring_carbons = mol['carbon_map'].get('all_ring_carbons', [])
    positions = [np.array(coords[i]) for i in ring_carbons]
    ring_oxygen = mol['carbon_map'].get('ring_oxygen')
    if ring_oxygen is not None:
        positions.append(np.array(coords[ring_oxygen]))
    return positions


def bond_crosses_ring_plane(bond_start, bond_end, ring_positions, ring_margin=1.2):
    """
    Check whether the bond segment intersects the ring plane inside the ring.

    Parameters
    ----------
    bond_start, bond_end : array-like  shape (3,)
    ring_positions : list[np.ndarray]
    ring_margin : float
        Multiplier applied to ring radius (default 1.2 = 20 % safety margin).

    Returns
    -------
    bool  True → bond crosses ring (reject this rotation).
    """
    bond_start = np.array(bond_start)
    bond_end   = np.array(bond_end)
    ring_pos   = np.array(ring_positions)

    if len(ring_pos) < 3:
        return False

    ring_center = np.mean(ring_pos, axis=0)
    ring_indices = np.arange(len(ring_pos))
    ring_normal = get_ring_normal_from_positions(ring_pos, ring_indices)
    ring_radius = np.max([np.linalg.norm(p - ring_center) for p in ring_pos])

    bond_vec = bond_end - bond_start
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 1e-6:
        return False
    bond_dir = bond_vec / bond_len

    denom = np.dot(ring_normal, bond_dir)
    if abs(denom) < 1e-6:
        return False

    t = np.dot(ring_normal, ring_center - bond_start) / denom
    if not (0 < t < bond_len):
        return False

    intersection = bond_start + t * bond_dir
    return np.linalg.norm(intersection - ring_center) < ring_radius * ring_margin


def calculate_ring_penalty(donor_pos, target_pos, ring_positions,
                           ring_margin=1.2, label="RING", debug=False):
    """
    Continuous ring-plane violation penalty for scoring.

    Returns
    -------
    float  0.0 = no violation; higher = worse.
    """
    donor_pos  = np.array(donor_pos)
    target_pos = np.array(target_pos)
    ring_pos   = np.array(ring_positions)

    bond_vec = target_pos - donor_pos
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 0.1:
        return 0.0
    bond_dir = bond_vec / bond_len

    ring_center = np.mean(ring_pos, axis=0)
    v1 = ring_pos[1] - ring_pos[0]
    v2 = ring_pos[2] - ring_pos[0]
    ring_normal = np.cross(v1, v2)
    norm = np.linalg.norm(ring_normal)
    if norm < 1e-6:
        return 0.0
    ring_normal /= norm
    ring_radius = np.max([np.linalg.norm(p - ring_center) for p in ring_pos])

    penalty = 0.0
    denom = np.dot(ring_normal, bond_dir)
    if abs(denom) > 1e-6:
        t = np.dot(ring_normal, ring_center - donor_pos) / denom
        if 0 < t < bond_len:
            intersection = donor_pos + t * bond_dir
            dist = np.linalg.norm(intersection - ring_center)
            threshold = ring_radius * ring_margin
            if dist < threshold:
                penalty += (threshold - dist) * 100

    return penalty


# ============================================================================
# Clearance / collision scoring
# ============================================================================

def check_ring_clearance(mol, bonds_to_optimize, mode='penalty',
                         ring_margin=1.2, debug=False):
    """
    Check or score ring-plane clearance for all bonds.

    Parameters
    ----------
    mol : dict
        Rotated molecule_data_dict entry.
    bonds_to_optimize : list[dict]
        Each dict must have 'donor_idx', 'target_pos', 'target',
        'target_carbon'.
    mode : 'binary' | 'penalty'
        'binary': returns True if all bonds clear all rings, False otherwise.
        'penalty': returns a float penalty (0 = perfect).
    ring_margin : float
    debug : bool

    Returns
    -------
    bool (binary mode) or float (penalty mode)
    """
    total_penalty = 0.0

    for bond in bonds_to_optimize:
        donor_pos = np.array(mol['absolute_coordinates'][bond['donor_idx']])

        if isinstance(bond.get('target'), dict) and 'carbon_map' in bond['target']:
            target_idx = bond['target']['carbon_map'][bond['target_carbon']]
            target_pos = np.array(bond['target']['absolute_coordinates'][target_idx])
            target_mol = bond['target']
        else:
            target_pos = np.array(bond.get('target_pos', bond['target']))
            target_mol = None

        # Check donor ring
        donor_ring = get_ring_positions(mol)
        if len(donor_ring) >= 3:
            if mode == 'binary':
                if bond_crosses_ring_plane(donor_pos, target_pos, donor_ring, ring_margin):
                    return False
            else:
                total_penalty += calculate_ring_penalty(
                    donor_pos, target_pos, donor_ring, ring_margin, "DONOR", debug)

        # Check target ring
        if target_mol is not None:
            target_ring = get_ring_positions(target_mol)
            if len(target_ring) >= 3:
                if mode == 'binary':
                    if bond_crosses_ring_plane(donor_pos, target_pos, target_ring, ring_margin):
                        return False
                else:
                    total_penalty += calculate_ring_penalty(
                        donor_pos, target_pos, target_ring, ring_margin, "TARGET", debug)

    return True if mode == 'binary' else total_penalty


def detect_clashes(new_mol, existing_chain, clash_factor=0.6, exclude_last_n=0):
    """
    Detect steric clashes between *new_mol* and *existing_chain*.

    Parameters
    ----------
    new_mol : dict
    existing_chain : list[dict]
    clash_factor : float
        Fraction of summed VDW radii used as the clash threshold.
    exclude_last_n : int
        Ignore the last N molecules in *existing_chain*.

    Returns
    -------
    list[dict]  one entry per clashing atom pair.
    """
    clashes = []
    new_coords = np.array(new_mol['absolute_coordinates'])
    new_types  = new_mol['atom_types']
    chain_to_check = existing_chain[:-exclude_last_n] if exclude_last_n > 0 else existing_chain

    for chain_mol in chain_to_check:
        exist_coords = np.array(chain_mol['absolute_coordinates'])
        exist_types  = chain_mol['atom_types']
        for i, (c1, t1) in enumerate(zip(new_coords, new_types)):
            for j, (c2, t2) in enumerate(zip(exist_coords, exist_types)):
                dist = np.linalg.norm(c1 - c2)
                threshold = clash_factor * (VDW_RADII.get(t1, 1.7) + VDW_RADII.get(t2, 1.7))
                if dist < threshold:
                    clashes.append({'new_atom_idx': i, 'exist_atom_idx': j,
                                    'distance': dist, 'threshold': threshold,
                                    'severity': threshold - dist})
    return clashes


def calculate_collision_penalty(mol, chain_list, clash_factor=0.6):
    """
    Continuous collision penalty (sum of exponential clash severities).

    Returns
    -------
    float
    """
    if not chain_list:
        return 0.0
    clashes = detect_clashes(mol, chain_list, clash_factor=clash_factor)
    if not clashes:
        return 0.0
    return sum(np.exp(c['severity'] / c['threshold'] * 5) for c in clashes)


def line_segments_intersect_2D(p1, p2, p3, p4):
    """
    Test whether segments p1→p2 and p3→p4 intersect in 2D (XY projection).

    Returns
    -------
    bool
    """
    p1, p2, p3, p4 = [np.array(p).flatten()[:2] for p in [p1, p2, p3, p4]]
    d1, d2 = p2 - p1, p4 - p3
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-10:
        return False
    dp = p3 - p1
    t = (dp[0]*d2[1] - dp[1]*d2[0]) / cross
    s = (dp[0]*d1[1] - dp[1]*d1[0]) / cross
    return (0 <= t <= 1) and (0 <= s <= 1)


def calculate_bond_intersection_penalty(new_mol, new_carbon_idx, target_pos,
                                        existing_chain, existing_linkages=None,
                                        debug=False):
    """
    2D (XY) bond intersection penalty between the new bond and existing bonds.

    Checks intra-molecular bonds (dist < 2 Å) and inter-molecular linkages.

    Returns
    -------
    float  0.0 = no intersection; 1000.0 per intersection found.
    """
    if not existing_chain:
        return 0.0

    new_donor = np.array(new_mol['absolute_coordinates'][new_carbon_idx])[:2]
    new_target = np.array(target_pos)[:2]
    if np.linalg.norm(new_target - new_donor) < 0.01:
        return 0.0

    penalty = 0.0

    for chain_mol in existing_chain:
        coords = np.array(chain_mol['absolute_coordinates'])
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if np.linalg.norm(coords[i] - coords[j]) < 2.0:
                    p1, p2 = coords[i][:2], coords[j][:2]
                    if np.linalg.norm(p2 - p1) < 0.01:
                        continue
                    if line_segments_intersect_2D(new_donor, new_target, p1, p2):
                        penalty += 1000.0

    if existing_linkages:
        name_map = {m.get('name'): m for m in existing_chain}
        for link in existing_linkages:
            donor_name, donor_c = link[0], link[1]
            acceptor, acceptor_c = link[2], link[3]
            donor_mol = name_map.get(donor_name)
            if not donor_mol:
                continue
            d_idx = donor_mol['carbon_map'].get(donor_c)
            if d_idx is None:
                continue
            d_pos = np.array(donor_mol['absolute_coordinates'][d_idx])[:2]

            if isinstance(acceptor, (list, np.ndarray)):
                a_pos = np.array(acceptor)[:2]
            else:
                a_mol = name_map.get(acceptor)
                if not a_mol:
                    continue
                a_idx = a_mol['carbon_map'].get(acceptor_c)
                if a_idx is None:
                    continue
                a_pos = np.array(a_mol['absolute_coordinates'][a_idx])[:2]

            if np.linalg.norm(a_pos - d_pos) < 0.01:
                continue
            if line_segments_intersect_2D(new_donor, new_target, d_pos, a_pos):
                penalty += 1000.0

    return penalty


def check_multibond_intersections(bonds_to_optimize, rotated_mol):
    """
    Check whether any pair of bonds in *bonds_to_optimize* cross each other
    in 2D projection.

    Returns
    -------
    bool  True → at least one self-intersection found.
    """
    segments = [
        (np.array(rotated_mol['absolute_coordinates'][b['donor_idx']])[:2],
         np.array(b['target_pos'])[:2])
        for b in bonds_to_optimize
    ]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if line_segments_intersect_2D(*segments[i], *segments[j]):
                return True
    return False


# ============================================================================
# Surface orientation scoring
# ============================================================================

_SURFACE_NORMAL = np.array([0.0, 0.0, 1.0])


def calculate_surface_orientation_score(mol_data, target_orientation):
    """
    Score how well the ring normal aligns with the surface normal.

    Parameters
    ----------
    mol_data : dict
    target_orientation : 'flat' | 'perpendicular'

    Returns
    -------
    float  0.0 = perfect; 1.0 = worst.
    """
    n_ring = get_ring_normal_absolute(mol_data)
    dot = np.dot(n_ring, _SURFACE_NORMAL)
    if target_orientation == 'flat':
        return 1.0 - abs(dot)
    if target_orientation == 'perpendicular':
        return abs(dot)
    raise ValueError(f"Unknown orientation: {target_orientation}")


def get_orientation_dot_product(mol_data):
    """Return n_ring · z_surface."""
    return np.dot(get_ring_normal_absolute(mol_data), _SURFACE_NORMAL)


# ============================================================================
# Filter pipeline
# ============================================================================

def apply_all_filters(mol, bonds_to_optimize, existing_chain,
                      existing_linkages, clash_factor=0.6, debug=False):
    """
    Apply all hard geometric filters to a rotation candidate.

    Filters applied in order:
    1. Ring clearance (hard reject).
    2. Bond intersections (hard reject).
    3. VDW collisions (soft — recorded for scoring but not a hard reject).

    Returns
    -------
    dict with keys:
        'passes'            bool — True if passes all hard filters.
        'ring_clearance'    bool
        'bond_intersections' bool
        'details'           dict with 'num_clashes' etc.
    """
    results = {'passes': True, 'ring_clearance': False,
               'bond_intersections': False, 'details': {}}

    passes_rings = check_ring_clearance(mol, bonds_to_optimize, mode='binary')
    results['ring_clearance'] = passes_rings
    if not passes_rings:
        results['passes'] = False
        return results

    if len(bonds_to_optimize) == 1:
        bond = bonds_to_optimize[0]
        inter_penalty = calculate_bond_intersection_penalty(
            mol, bond['donor_idx'], bond['target_pos'],
            existing_chain, existing_linkages, debug=False)
        passes_inter = (inter_penalty == 0)
    else:
        has_self = check_multibond_intersections(bonds_to_optimize, mol)
        bond = bonds_to_optimize[0]
        inter_penalty = calculate_bond_intersection_penalty(
            mol, bond['donor_idx'], bond['target_pos'],
            existing_chain, existing_linkages, debug=False)
        passes_inter = (not has_self) and (inter_penalty == 0)

    results['bond_intersections'] = passes_inter
    if not passes_inter:
        results['passes'] = False
        return results

    clashes = detect_clashes(mol, existing_chain, clash_factor=clash_factor)
    results['details']['num_clashes'] = len(clashes)
    return results


# ============================================================================
# Shared scoring helpers
# ============================================================================

def calculate_average_bond_distance(mol, bonds_to_optimize):
    """Average donor–target distance across all bonds in *bonds_to_optimize*."""
    if not bonds_to_optimize:
        return 0.0
    total = 0.0
    for bond in bonds_to_optimize:
        donor_pos = np.array(mol['absolute_coordinates'][bond['donor_idx']])
        if 'target_pos' in bond:
            target_pos = np.array(bond['target_pos'])
        elif isinstance(bond.get('target'), dict):
            t_idx = bond['target']['carbon_map'][bond['target_carbon']]
            target_pos = np.array(bond['target']['absolute_coordinates'][t_idx])
        else:
            target_pos = np.array(bond['target'])
        total += np.linalg.norm(donor_pos - target_pos)
    return total / len(bonds_to_optimize)


# ============================================================================
# QUEST weight variants
# ============================================================================

def generate_quest_weight_variants(n_bonds, include_normal=False):
    """
    Return a list of (weights_array, description) for systematic QUEST sweeps.

    Parameters
    ----------
    n_bonds : int
    include_normal : bool
        If True, the last observation is a normal-orientation constraint.

    Returns
    -------
    list[tuple[np.ndarray, str]]
    """
    n_obs = n_bonds + (1 if include_normal else 0)
    variants = []

    w = np.ones(n_obs); w[0] = 3.0
    if include_normal: w[-1] = 2.0
    variants.append((w, "primary_bond_3x"))

    variants.append((np.ones(n_obs), "equal_weights"))

    if include_normal:
        w = np.ones(n_obs); w[-1] = 5.0
        variants.append((w, "flatness_5x"))

    for i in range(1, n_bonds):
        w = np.ones(n_obs); w[i] = 3.0
        if include_normal: w[-1] = 2.0
        variants.append((w, f"bond_{i+1}_emphasized"))

    if include_normal:
        w = np.ones(n_obs); w[-1] = 10.0
        variants.append((w, "aggressive_flatness"))

    w = np.ones(n_obs); w[0] = 5.0
    if include_normal: w[-1] = 1.5
    variants.append((w, "primary_bond_5x"))

    return variants


# ============================================================================
# Internal helpers shared by the main search functions
# ============================================================================

def _score_candidate(mol, bonds_to_optimize, existing_chain, clash_factor,
                     target_orientation=None):
    """Compute a composite score for sorting candidates (lower = better)."""
    distance  = calculate_average_bond_distance(mol, bonds_to_optimize)
    collision = calculate_collision_penalty(mol, existing_chain, clash_factor)
    score = distance * 10.0 + collision * 50.0
    if target_orientation:
        score += calculate_surface_orientation_score(mol, target_orientation) * 200.0
    return distance, collision, score


def _test_quest_variants(template, bonds_to_optimize, existing_chain,
                         existing_linkages, com, n_variants, face_label,
                         clash_factor, target_orientation=None):
    """Run QUEST for each weight variant and return passing solutions."""
    observations = []
    references   = []
    for bond in bonds_to_optimize:
        obs = np.array(template['relative_coordinates'][bond['donor_idx']])
        obs = obs / np.linalg.norm(obs)
        observations.append(obs)
        ref = bond['target_pos'] - np.array(com)
        ref = ref / np.linalg.norm(ref)
        references.append(ref)

    if target_orientation:
        mol_normal = get_ring_normal(template)
        mol_normal = mol_normal / np.linalg.norm(mol_normal)
        observations.append(mol_normal)
        references.append(np.array([0.0, 0.0, 1.0]))

    weight_variants = generate_quest_weight_variants(
        len(bonds_to_optimize), include_normal=(target_orientation is not None)
    )[:n_variants]

    passing = []
    for weights, description in weight_variants:
        rot_matrix = solve_wahba_quest(observations, references, weights)
        mol_test = rotate_molecule_around_com(template, rot_matrix)
        mol_test['COM'] = com.tolist()

        filter_result = apply_all_filters(
            mol_test, bonds_to_optimize, existing_chain, existing_linkages,
            clash_factor, debug=False)

        if filter_result['passes']:
            distance, collision, score = _score_candidate(
                mol_test, bonds_to_optimize, existing_chain, clash_factor,
                target_orientation)
            passing.append({
                'mol': mol_test, 'rotation': rot_matrix,
                'distance': distance, 'collision_penalty': collision,
                'score': score, 'method': 'QUEST', 'variant': description,
                'face': face_label,
                'num_clashes': filter_result['details']['num_clashes'],
            })

    return passing


def _test_uniform_samples(template, bonds_to_optimize, existing_chain,
                          existing_linkages, com, n_samples, face_label,
                          clash_factor, target_count, quest_rotation=None,
                          target_orientation=None):
    """
    Test uniform rotation samples; stop early once *target_count* pass.

    If *quest_rotation* is provided (for flat-mode), each sample is composed
    with the QUEST base rotation to preserve flatness.
    """
    rotations = generate_uniform_rotations(n_samples)
    passing   = []

    for idx, rotation in enumerate(rotations):
        if quest_rotation is not None:
            combined = rotation @ quest_rotation
        else:
            combined = rotation

        mol_test = rotate_molecule_around_com(template, combined)
        mol_test['COM'] = com.tolist()

        filter_result = apply_all_filters(
            mol_test, bonds_to_optimize, existing_chain, existing_linkages,
            clash_factor, debug=False)

        if not filter_result['passes']:
            continue

        distance, collision, score = _score_candidate(
            mol_test, bonds_to_optimize, existing_chain, clash_factor,
            target_orientation)

        passing.append({
            'mol': mol_test, 'rotation': combined,
            'distance': distance, 'collision_penalty': collision,
            'score': score, 'method': 'UNIFORM',
            'variant': f'sample_{idx}', 'face': face_label,
            'num_clashes': filter_result['details']['num_clashes'],
        })

        if len(passing) >= target_count:
            break

    return passing


def _finalize_top_n_face_aware(normal_solutions, reflected_solutions, n_top):
    """
    Merge and rank solutions from both faces.

    Interleaves normal and reflected to preserve diversity.
    """
    normal_solutions.sort(key=lambda x: x['score'])
    reflected_solutions.sort(key=lambda x: x['score'])

    has_normal    = len(normal_solutions) > 0
    has_reflected = len(reflected_solutions) > 0

    if has_normal and has_reflected:
        top_n, i, j = [], 0, 0
        while len(top_n) < n_top and (i < len(normal_solutions) or j < len(reflected_solutions)):
            if i < len(normal_solutions):
                top_n.append(normal_solutions[i]); i += 1
            if len(top_n) >= n_top:
                break
            if j < len(reflected_solutions):
                top_n.append(reflected_solutions[j]); j += 1
    elif has_normal:
        top_n = normal_solutions[:n_top]
    else:
        top_n = reflected_solutions[:n_top]

    return top_n


def _count_future_bond_conflicts(candidates, future_bonds):
    """Estimate future bond crossing conflicts for the best candidate."""
    if not candidates or not future_bonds:
        return 0
    test_mol = candidates[0].get('mol') if isinstance(candidates[0], dict) else None
    if test_mol is None:
        return 0

    ring_carbons = test_mol['carbon_map'].get('all_ring_carbons', [])
    if len(ring_carbons) < 3:
        return 0
    ring_center = np.mean([test_mol['absolute_coordinates'][i] for i in ring_carbons],
                          axis=0)[:2]
    conflicts = 0
    for i, fb1 in enumerate(future_bonds):
        c1_idx = test_mol['carbon_map'].get(fb1['my_carbon'])
        if c1_idx is None:
            continue
        c1 = np.array(test_mol['absolute_coordinates'][c1_idx])[:2]
        d1 = c1 - ring_center
        d1 = d1 / (np.linalg.norm(d1) + 1e-10)
        fp1 = c1 + d1 * 5.0
        for j, fb2 in enumerate(future_bonds):
            if i >= j:
                continue
            c2_idx = test_mol['carbon_map'].get(fb2['my_carbon'])
            if c2_idx is None:
                continue
            c2 = np.array(test_mol['absolute_coordinates'][c2_idx])[:2]
            d2 = c2 - ring_center
            d2 = d2 / (np.linalg.norm(d2) + 1e-10)
            fp2 = c2 + d2 * 5.0
            if line_segments_intersect_2D(c1, fp1, c2, fp2):
                conflicts += 1
    return conflicts


# ============================================================================
# Main search functions
# ============================================================================

def find_rotation_QUEST_then_UNIFORM_3D(template, bonds_to_optimize,
                                        existing_chain, fixed_com,
                                        n_samples=500):
    """
    Two-phase 3D rotation search (no surface orientation constraint).

    Phase 1: QUEST for both faces (normal and reflected).
    Phase 2: Uniform sampling if QUEST does not pass all filters.

    Returns
    -------
    tuple (best_mol, best_distance, params_dict)
    """
    com = fixed_com
    template_reflected = flip_molecule_coordinates(template, com)

    def _optimize_face(tmpl, label):
        observations, references, weights = [], [], []
        for i, bond in enumerate(bonds_to_optimize):
            obs = np.array(tmpl['relative_coordinates'][bond['donor_idx']])
            obs = obs / np.linalg.norm(obs)
            observations.append(obs)
            ref = bond['target_pos'] - np.array(com)
            ref = ref / np.linalg.norm(ref)
            references.append(ref)
            weights.append(2.0 if i == 0 else 1.0)

        quest_rot = solve_wahba_quest(observations, references, weights)
        mol_quest = rotate_molecule_around_com(tmpl, quest_rot)
        mol_quest['COM'] = com.tolist()
        quest_dist = calculate_average_bond_distance(mol_quest, bonds_to_optimize)
        quest_passes = check_ring_clearance(mol_quest, bonds_to_optimize, mode='binary')

        if quest_passes:
            return {'mol': mol_quest, 'distance': quest_dist, 'n_valid': 1,
                    'best_score': quest_dist * 10.0, 'quest_mol': mol_quest,
                    'quest_distance': quest_dist,
                    'params': {'method': f'QUEST_{label}', 'face': label}}

        passing = _test_uniform_samples(
            tmpl, bonds_to_optimize, existing_chain, None,
            com, n_samples, label, 0.6, n_samples)

        if passing:
            passing.sort(key=lambda x: x['score'])
            best = passing[0]
            return {'mol': best['mol'], 'distance': best['distance'],
                    'n_valid': len(passing), 'best_score': best['score'],
                    'quest_mol': mol_quest, 'quest_distance': quest_dist,
                    'params': {'method': f'uniform_{label}', 'face': label,
                               'n_valid': len(passing)}}

        return {'mol': mol_quest, 'distance': quest_dist, 'n_valid': 0,
                'best_score': float('inf'), 'quest_mol': mol_quest,
                'quest_distance': quest_dist,
                'params': {'method': f'QUEST_fallback_{label}', 'face': label}}

    res_n = _optimize_face(template, "NORMAL")
    res_r = _optimize_face(template_reflected, "REFLECTED")

    for res in [res_n, res_r]:
        if res['n_valid'] > 0:
            if res_n['n_valid'] > 0 and res_r['n_valid'] > 0:
                winner = res_n if res_n['best_score'] <= res_r['best_score'] else res_r
            else:
                winner = res
            return winner['mol'], winner['distance'], winner['params']

    # both failed — return best QUEST
    if res_n['quest_distance'] <= res_r['quest_distance']:
        return res_n['quest_mol'], res_n['quest_distance'], res_n['params']
    return res_r['quest_mol'], res_r['quest_distance'], res_r['params']


def find_rotation_FULL_3D_QUATERNION(template, mol_carbon, target_mol, target_carbon,
                                     existing_chain, fixed_com, n_samples=500,
                                     additional_bonds=None):
    """
    Single-bond or multi-bond 3D rotation search.

    Builds *bonds_to_optimize* from the explicit carbon names and delegates
    to find_rotation_QUEST_then_UNIFORM_3D.

    Returns
    -------
    tuple (best_mol, best_distance, params_dict)
    """
    mol_c_idx = template['carbon_map'][mol_carbon]

    if isinstance(target_mol, dict) and 'carbon_map' in target_mol:
        t_idx = target_mol['carbon_map'][target_carbon]
        target_pos = np.array(target_mol['absolute_coordinates'][t_idx])
    else:
        target_pos = np.array(target_mol)

    bonds = [{'donor_idx': mol_c_idx, 'donor_name': mol_carbon,
               'target': target_mol, 'target_pos': target_pos,
               'target_carbon': target_carbon}]

    if additional_bonds:
        for ab in additional_bonds:
            ab_idx = template['carbon_map'][ab['donor_carbon']]
            if isinstance(ab['target'], dict) and 'carbon_map' in ab['target']:
                ab_t_idx = ab['target']['carbon_map'][ab['target_carbon']]
                ab_pos = np.array(ab['target']['absolute_coordinates'][ab_t_idx])
            else:
                ab_pos = np.array(ab['target'])
            bonds.append({'donor_idx': ab_idx, 'donor_name': ab['donor_carbon'],
                           'target': ab['target'], 'target_pos': ab_pos,
                           'target_carbon': ab.get('target_carbon')})

    return find_rotation_QUEST_then_UNIFORM_3D(
        template, bonds, existing_chain, fixed_com, n_samples)


def find_rotation_1D_Z_AXIS(template, donor_c, target, acceptor_c,
                             chain_list, fixed_com,
                             target_orientation, n_samples=360,
                             clash_factor=0.6, debug=False,
                             additional_bonds=None, existing_linkages=None,
                             future_bonds_to_this_mol=None):
    """
    Z-axis spin search for surface-flat (or perpendicular) rings.

    Uses QUEST with a normal constraint to establish flatness, then spins
    around the Z-axis while keeping the ring orientation locked.
    Tests both faces (normal and Y-reflected).

    Returns
    -------
    tuple (best_mol, best_distance, params_dict)
    """
    com = fixed_com
    donor_idx = template['carbon_map'][donor_c]

    if isinstance(target, dict) and 'carbon_map' in target:
        t_idx = target['carbon_map'][acceptor_c]
        target_pos = np.array(target['absolute_coordinates'][t_idx])
    else:
        target_pos = np.array(target)

    bonds = [{'donor_idx': donor_idx, 'donor_name': donor_c,
               'target_pos': target_pos, 'target': target,
               'target_carbon': acceptor_c if isinstance(target, dict) else None}]

    if additional_bonds:
        for ab in additional_bonds:
            ab_idx = template['carbon_map'][ab['donor_carbon']]
            if isinstance(ab['target'], dict) and 'carbon_map' in ab['target']:
                ab_t_idx = ab['target']['carbon_map'][ab['target_carbon']]
                ab_pos = np.array(ab['target']['absolute_coordinates'][ab_t_idx])
            else:
                ab_pos = np.array(ab['target'])
            bonds.append({'donor_idx': ab_idx, 'donor_name': ab['donor_carbon'],
                           'target_pos': ab_pos, 'target': ab['target'],
                           'target_carbon': ab.get('target_carbon')})

    molecule_normal = get_ring_normal(template)
    surface_normal  = np.array([0.0, 0.0, 1.0])
    template_reflected = flip_molecule_coordinates(template, com)

    def _optimize_flat_face(tmpl, label):
        observations = []
        references   = []
        for i, bond in enumerate(bonds):
            obs = np.array(tmpl['relative_coordinates'][bond['donor_idx']])
            obs = obs / np.linalg.norm(obs)
            observations.append(obs)
            ref = bond['target_pos'] - np.array(com)
            ref = ref / np.linalg.norm(ref)
            references.append(ref)
        observations.append(molecule_normal / np.linalg.norm(molecule_normal))
        references.append(surface_normal)
        weights = [2.0 if i == 0 else 1.0 for i in range(len(bonds))] + [3.0]

        quest_rot = solve_wahba_quest(observations, references, weights)
        mol_quest = rotate_molecule_around_com(tmpl, quest_rot)
        mol_quest['COM'] = com.tolist()
        quest_dist  = calculate_average_bond_distance(mol_quest, bonds)
        quest_passes = check_ring_clearance(mol_quest, bonds, mode='binary')

        if quest_passes:
            return {'mol': mol_quest, 'distance': quest_dist, 'n_valid': 1,
                    'best_score': quest_dist * 10.0, 'quest_mol': mol_quest,
                    'quest_distance': quest_dist,
                    'params': {'method': f'QUEST_flat_{label}', 'face': label}}

        # Z-spin (preserves flatness established by QUEST)
        passing = _test_uniform_samples(
            tmpl, bonds, chain_list, existing_linkages,
            com, n_samples, label, clash_factor, n_samples,
            quest_rotation=quest_rot,
            target_orientation=target_orientation)

        # Remove candidates with bond intersections
        passing = [p for p in passing
                   if calculate_bond_intersection_penalty(
                       p['mol'], bonds[0]['donor_idx'], bonds[0]['target_pos'],
                       chain_list, existing_linkages) == 0]

        if passing:
            passing.sort(key=lambda x: x['score'])
            best = passing[0]
            return {'mol': best['mol'], 'distance': best['distance'],
                    'n_valid': len(passing), 'best_score': best['score'],
                    'quest_mol': mol_quest, 'quest_distance': quest_dist,
                    'best_candidates': passing[:5],
                    'params': {'method': f'Z_axis_flat_{label}', 'face': label,
                               'n_valid': len(passing)}}

        return {'mol': mol_quest, 'distance': quest_dist, 'n_valid': 0,
                'best_score': float('inf'), 'quest_mol': mol_quest,
                'quest_distance': quest_dist,
                'params': {'method': f'QUEST_fallback_{label}', 'face': label}}

    res_n = _optimize_flat_face(template, "NORMAL")
    res_r = _optimize_flat_face(template_reflected, "REFLECTED")

    # Optional future-bond conflict tiebreaker
    if future_bonds_to_this_mol and (res_n['n_valid'] > 0 or res_r['n_valid'] > 0):
        nc = _count_future_bond_conflicts(
            res_n.get('best_candidates', [res_n]) if res_n['n_valid'] > 0 else [],
            future_bonds_to_this_mol) if res_n['n_valid'] > 0 else float('inf')
        rc = _count_future_bond_conflicts(
            res_r.get('best_candidates', [res_r]) if res_r['n_valid'] > 0 else [],
            future_bonds_to_this_mol) if res_r['n_valid'] > 0 else float('inf')
        if nc < rc and res_n['n_valid'] > 0:
            return res_n['mol'], res_n['distance'], res_n['params']
        if rc < nc and res_r['n_valid'] > 0:
            return res_r['mol'], res_r['distance'], res_r['params']

    if res_n['n_valid'] > 0 and res_r['n_valid'] > 0:
        winner = res_n if res_n['best_score'] <= res_r['best_score'] else res_r
        return winner['mol'], winner['distance'], winner['params']
    for res in [res_n, res_r]:
        if res['n_valid'] > 0:
            return res['mol'], res['distance'], res['params']

    if res_n['quest_distance'] <= res_r['quest_distance']:
        return res_n['quest_mol'], res_n['quest_distance'], res_n['params']
    return res_r['quest_mol'], res_r['quest_distance'], res_r['params']


def find_rotation_QUEST_VARIANTS_TOP_N(template, bonds_to_optimize,
                                       existing_chain, existing_linkages,
                                       fixed_com, n_top=5,
                                       n_quest_variants=10,
                                       max_uniform_samples=500,
                                       clash_factor=0.6):
    """
    Return the top-N rotations using QUEST variants then uniform sampling.

    Used by align_and_position_molecules_TOP_N_POLYMERS to generate diverse
    starting structures.

    Returns
    -------
    tuple (top_solutions, metadata_dict)
        top_solutions is a list of up to n_top solution dicts, each with
        keys 'mol', 'distance', 'score', 'face', 'method'.
    """
    com = fixed_com
    template_reflected = flip_molecule_coordinates(template, com)

    normal_solutions   = _test_quest_variants(template, bonds_to_optimize,
                             existing_chain, existing_linkages, com,
                             n_quest_variants, "NORMAL", clash_factor)
    reflected_solutions = _test_quest_variants(template_reflected, bonds_to_optimize,
                             existing_chain, existing_linkages, com,
                             n_quest_variants, "REFLECTED", clash_factor)

    total_quest = len(normal_solutions) + len(reflected_solutions)

    if total_quest < n_top:
        needed = n_top - total_quest
        if len(normal_solutions) >= len(reflected_solutions):
            normal_solutions.extend(_test_uniform_samples(
                template, bonds_to_optimize, existing_chain, existing_linkages,
                com, max_uniform_samples, "NORMAL", clash_factor, needed))
            still_needed = n_top - len(normal_solutions) - len(reflected_solutions)
            if still_needed > 0:
                reflected_solutions.extend(_test_uniform_samples(
                    template_reflected, bonds_to_optimize, existing_chain,
                    existing_linkages, com, max_uniform_samples, "REFLECTED",
                    clash_factor, still_needed))
        else:
            reflected_solutions.extend(_test_uniform_samples(
                template_reflected, bonds_to_optimize, existing_chain,
                existing_linkages, com, max_uniform_samples, "REFLECTED",
                clash_factor, needed))
            still_needed = n_top - len(normal_solutions) - len(reflected_solutions)
            if still_needed > 0:
                normal_solutions.extend(_test_uniform_samples(
                    template, bonds_to_optimize, existing_chain, existing_linkages,
                    com, max_uniform_samples, "NORMAL", clash_factor, still_needed))

    top_n = _finalize_top_n_face_aware(normal_solutions, reflected_solutions, n_top)

    total = len(normal_solutions) + len(reflected_solutions)
    metadata = {
        'n_solutions': len(top_n),
        'n_total_passing': total,
        'n_from_normal': sum(1 for s in top_n if s['face'] == 'NORMAL'),
        'n_from_reflected': sum(1 for s in top_n if s['face'] == 'REFLECTED'),
        'best_score': top_n[0]['score'] if top_n else None,
    }
    return top_n, metadata


def orient_root_molecule(mol_data, target_orientation, n_samples=360):
    """
    Rotate the root (fixed) molecule to satisfy a surface orientation constraint.

    Uses a coarse Z-axis search followed by scipy local refinement.

    Returns
    -------
    dict  updated mol_data at the best orientation.
    """
    from scipy.optimize import minimize_scalar

    com = np.array(mol_data['COM'])

    def objective(theta_z):
        rotated = rotate_molecule_z(mol_data, theta_z, com)
        return calculate_surface_orientation_score(rotated, target_orientation)

    angles = np.linspace(0, 360, n_samples, endpoint=False)
    candidates = sorted([(objective(a), a) for a in angles], key=lambda x: x[0])[:5]

    best_score, best_angle, best_mol = float('inf'), 0.0, mol_data
    for _, coarse_angle in candidates:
        result = minimize_scalar(objective,
                                 bounds=(coarse_angle - 2, coarse_angle + 2),
                                 method='bounded',
                                 options={'xatol': 0.001})
        if result.fun < best_score:
            best_score = result.fun
            best_angle = result.x
            best_mol   = rotate_molecule_z(mol_data, best_angle, com)

    return best_mol


def find_rotation_for_phosphate_linkage(mol_template, mol_carbon_name,
                                        target_mol, target_carbon_name,
                                        existing_chain,
                                        n_torsion_samples=72):
    """
    Torsion search for phosphodiester linkage positioning.

    Aligns the donor carbon toward the target, then samples torsion angles
    around the approach axis.

    Returns
    -------
    tuple (best_mol, best_distance, metadata_dict)
    """
    mol_c_idx  = mol_template['carbon_map'][mol_carbon_name]
    target_c_idx = target_mol['carbon_map'][target_carbon_name]

    mol_com      = np.array(mol_template['COM'])
    target_c_pos = np.array(target_mol['absolute_coordinates'][target_c_idx])

    donor_vec  = np.array(mol_template['relative_coordinates'][mol_c_idx])
    target_vec = target_c_pos - mol_com
    target_vec_norm = target_vec / np.linalg.norm(target_vec)

    base_rotation = calculate_alignment_rotation(donor_vec, target_vec_norm)
    mol_aligned   = rotate_molecule_around_com(mol_template, base_rotation)

    solutions = []
    for angle in range(0, 360, max(1, 360 // n_torsion_samples)):
        torsion_rot  = rotation_around_axis(target_vec_norm, angle)
        combined_rot = torsion_rot @ base_rotation
        mol_test     = rotate_molecule_around_com(mol_template, combined_rot)
        test_c_pos   = np.array(mol_test['absolute_coordinates'][mol_c_idx])
        distance     = np.linalg.norm(test_c_pos - target_c_pos)
        clashes      = detect_clashes(mol_test, existing_chain, clash_factor=0.6)
        solutions.append({'mol_rotated': mol_test, 'distance': distance,
                          'num_clashes': len(clashes), 'torsion_angle': angle,
                          'rotation_matrix': combined_rot})

    solutions.sort(key=lambda x: (x['num_clashes'], x['distance']))
    best = solutions[0]
    return best['mol_rotated'], best['distance'], {
        'distance': best['distance'],
        'num_clashes': best['num_clashes'],
        'torsion_angle': best['torsion_angle'],
        'all_solutions': solutions[:10],
    }