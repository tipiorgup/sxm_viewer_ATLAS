"""
geometry_utils.py — pure vector and rotation utilities.

All functions are numpy-only.  No RDKit, no OpenBabel, no MD.
Imported by every other module in rotation_optimization.

Groups
------
Molecule rotation
    rotate_molecule_z, rotate_molecule_around_com,
    calculate_alignment_rotation, rotation_around_axis

Ring geometry
    get_ring_normal, get_ring_normal_absolute, get_ring_normal_from_positions,
    calculate_alignment_score, calculate_moment_of_inertia

Conformer position access
    get_positions, set_positions, get_atom_position, get_positions_for_atoms

Vector utilities
    calculate_distance, calculate_angle, calculate_center_of_mass,
    cap_vectors, normalize_vector, get_perpendicular_vector,
    rodrigues_rotation
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...constants import EPSILON


# ============================================================================
# VDW radii (used by optimization_chain for clash detection)
# ============================================================================

VDW_RADII = {
    'C': 1.70,
    'O': 1.52,
    'N': 1.55,
}


# ============================================================================
# Molecule rotation
# ============================================================================

def rotate_molecule_z(mol_data, theta_z, com):
    """
    Rotate molecule around the Z-axis through *com*.

    Parameters
    ----------
    mol_data : dict
        molecule_data_dict entry with 'absolute_coordinates'.
    theta_z : float
        Rotation angle in degrees.
    com : np.ndarray  shape (3,)
        Centre of rotation.

    Returns
    -------
    dict
        Copy of mol_data with updated absolute_coordinates and COM.
    """
    coords = np.array(mol_data['absolute_coordinates'])
    theta_rad = np.radians(theta_z)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    rotation_matrix = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1],
    ])

    rotated_coords = ((rotation_matrix @ (coords - com).T).T + com).tolist()

    rotated_mol = mol_data.copy()
    rotated_mol['absolute_coordinates'] = rotated_coords
    rotated_mol['COM'] = com.tolist()
    return rotated_mol


def rotate_molecule_around_com(molecule_data, rotation_matrix):
    """
    Rotate molecule around its COM; COM position stays fixed.

    Parameters
    ----------
    molecule_data : dict
        molecule_data_dict entry.
    rotation_matrix : np.ndarray  shape (3, 3)
        Orthonormal rotation matrix.

    Returns
    -------
    dict
        Updated molecule_data with new absolute_coordinates, relative_coordinates
        and quaternion.
    """
    com = np.array(molecule_data['COM'])
    coords = np.array(molecule_data['absolute_coordinates'])

    relative_coords = coords - com
    rotated_relative = (rotation_matrix @ relative_coords.T).T
    new_absolute_coords = rotated_relative + com

    old_rot = R.from_quat(np.array(molecule_data['quaternion']))
    new_rot = R.from_matrix(rotation_matrix) * old_rot
    new_quat = new_rot.as_quat()

    updated = molecule_data.copy()
    updated['absolute_coordinates'] = new_absolute_coords.tolist()
    updated['relative_coordinates'] = rotated_relative.tolist()
    updated['quaternion'] = new_quat.tolist()
    return updated


def calculate_alignment_rotation(vector1, vector2):
    """
    Return a rotation matrix that aligns *vector1* onto *vector2*.

    Handles the degenerate cases (already aligned, opposite direction).

    Returns
    -------
    np.ndarray  shape (3, 3)
        Valid rotation matrix (orthonormal, det = +1).
    """
    v1 = np.array(vector1, dtype=float)
    v2 = np.array(vector2, dtype=float)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot = np.dot(v1, v2)

    if dot > 0.999999:
        return np.eye(3)

    if dot < -0.999999:
        perp = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)

    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    w = np.cos(angle / 2)
    x, y, z = axis * np.sin(angle / 2)
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),    2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def rotation_around_axis(axis, angle_degrees):
    """
    Rodrigues rotation matrix for rotation of *angle_degrees* around *axis*.

    Parameters
    ----------
    axis : array-like  shape (3,)
        Rotation axis (will be normalised).
    angle_degrees : float

    Returns
    -------
    np.ndarray  shape (3, 3)
    """
    angle = np.radians(angle_degrees)
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis /= norm
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ============================================================================
# Ring geometry
# ============================================================================

def get_ring_normal(mol_data):
    """
    Unit normal to the pyranose ring plane from *relative_coordinates*.

    Requires 'C1'–'C5' in mol_data['carbon_map'].

    Returns
    -------
    np.ndarray  shape (3,)
    """
    ring_indices = [mol_data['carbon_map'][f'C{i}'] for i in range(1, 6)]
    ring_vectors = np.array([mol_data['relative_coordinates'][i] for i in ring_indices])
    centered = ring_vectors - np.mean(ring_vectors, axis=0)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2]
    return normal / np.linalg.norm(normal)


def get_ring_normal_absolute(mol_data):
    """
    Unit normal to the pyranose ring plane from *absolute_coordinates*.

    Converts to relative coordinates temporarily if needed.

    Returns
    -------
    np.ndarray  shape (3,)
    """
    if 'relative_coordinates' not in mol_data:
        com = np.array(mol_data['COM'])
        abs_coords = np.array(mol_data['absolute_coordinates'])
        mol_data_temp = mol_data.copy()
        mol_data_temp['relative_coordinates'] = (abs_coords - com).tolist()
    else:
        mol_data_temp = mol_data
    return get_ring_normal(mol_data_temp)


def get_ring_normal_from_positions(positions, ring_atoms):
    """
    Unit normal to the ring plane given a full position array and atom indices.

    Parameters
    ----------
    positions : np.ndarray  shape (n_atoms, 3)
    ring_atoms : list[int]

    Returns
    -------
    np.ndarray  shape (3,)
    """
    if len(ring_atoms) < 3:
        return np.array([0.0, 0.0, 1.0])
    ring_positions = positions[ring_atoms]
    centered = ring_positions - np.mean(ring_positions, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        return normal / np.linalg.norm(normal)
    except Exception:
        return np.array([0.0, 0.0, 1.0])


def calculate_alignment_score(mol1, mol2):
    """
    Alignment score between two molecules' ring normals.

    Returns
    -------
    float
        0.0 = perfectly aligned (parallel or antiparallel).
        1.0 = perpendicular.
    """
    return 1.0 - abs(np.dot(get_ring_normal(mol1), get_ring_normal(mol2)))


def calculate_moment_of_inertia(ring_atoms, positions, masses, com):
    """
    Moment of inertia around Z-axis through *com* for the given ring atoms.

    I_zz = Σ m_i * r_i²  where r_i is the XY distance from COM.

    Returns
    -------
    float  (amu·Å²)
    """
    I_zz = 0.0
    for atom_idx in ring_atoms:
        r_xy = positions[atom_idx][:2] - com[:2]
        I_zz += masses[atom_idx] * np.dot(r_xy, r_xy)
    return I_zz


# ============================================================================
# Conformer position access
# ============================================================================

def get_positions(conf, n_atoms):
    """
    Extract atom positions from an RDKit conformer as a numpy array.

    Parameters
    ----------
    conf : rdkit.Chem.Conformer
    n_atoms : int

    Returns
    -------
    np.ndarray  shape (n_atoms, 3)
    """
    return np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(n_atoms)
    ])


def set_positions(conf, positions, n_atoms):
    """
    Write a numpy position array back into an RDKit conformer.

    Parameters
    ----------
    conf : rdkit.Chem.Conformer
    positions : np.ndarray  shape (n_atoms, 3)
    n_atoms : int
    """
    for i in range(n_atoms):
        conf.SetAtomPosition(i, tuple(positions[i]))


def get_atom_position(conf, atom_idx):
    """
    Return the position of a single atom as a numpy array.

    Returns
    -------
    np.ndarray  shape (3,)
    """
    pos = conf.GetAtomPosition(atom_idx)
    return np.array([pos.x, pos.y, pos.z])


def get_positions_for_atoms(conf, atom_indices):
    """
    Return positions for a subset of atoms.

    Returns
    -------
    np.ndarray  shape (len(atom_indices), 3)
    """
    return np.array([get_atom_position(conf, i) for i in atom_indices])


# ============================================================================
# Vector utilities
# ============================================================================

def calculate_distance(pos1, pos2):
    """Euclidean distance between two positions."""
    return np.linalg.norm(np.array(pos2) - np.array(pos1))


def calculate_angle(pos1, pos2, pos3):
    """
    Bond angle at *pos2* formed by pos1–pos2–pos3.

    Returns
    -------
    float  (degrees)
    """
    v1 = np.array(pos1) - np.array(pos2)
    v2 = np.array(pos3) - np.array(pos2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def calculate_center_of_mass(positions, masses):
    """
    Mass-weighted centre of mass.

    Parameters
    ----------
    positions : np.ndarray  shape (n, 3)
    masses : np.ndarray  shape (n,)

    Returns
    -------
    np.ndarray  shape (3,)
    """
    return np.average(positions, axis=0, weights=masses)


def cap_vectors(vectors, max_magnitude):
    """
    Clamp each vector in *vectors* to at most *max_magnitude*.

    Returns
    -------
    np.ndarray  same shape as *vectors*
    """
    magnitudes = np.linalg.norm(vectors, axis=1)
    capped = vectors.copy()
    mask = magnitudes > max_magnitude
    capped[mask] = (vectors[mask].T * (max_magnitude / magnitudes[mask])).T
    return capped


def normalize_vector(vec):
    """
    Return unit vector.  Falls back to [0, 0, 1] for near-zero input.

    Returns
    -------
    np.ndarray  shape (3,)
    """
    norm = np.linalg.norm(vec)
    return vec / norm if norm >= EPSILON else np.array([0.0, 0.0, 1.0])


def get_perpendicular_vector(vec):
    """
    Return an arbitrary unit vector perpendicular to *vec*.

    Returns
    -------
    np.ndarray  shape (3,)
    """
    vec = vec / np.linalg.norm(vec)
    perp = np.cross(vec, [0, 0, 1]) if abs(vec[2]) < 0.9 else np.cross(vec, [1, 0, 0])
    return perp / np.linalg.norm(perp)


def rodrigues_rotation(point, axis, angle, center):
    """
    Rotate *point* around *axis* through *center* by *angle* radians.

    Uses Rodrigues' formula.

    Returns
    -------
    np.ndarray  shape (3,)
    """
    rel_pos = np.array(point) - np.array(center)
    k = np.array(axis)
    rotated = (rel_pos * np.cos(angle)
               + np.cross(k, rel_pos) * np.sin(angle)
               + k * np.dot(k, rel_pos) * (1 - np.cos(angle)))
    return np.array(center) + rotated