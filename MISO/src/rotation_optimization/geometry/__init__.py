"""
geometry — pure vector and rotation utilities.
 
No RDKit, no OpenBabel, no MD, no domain knowledge of sugars or lipids.
This sub-package sits at the bottom of the dependency chain and is imported
by every other sub-package in rotation_optimization.
 
All public names from geometry_utils are re-exported here so callers can use
either of these equivalent forms::
 
    from src.rotation_optimization.geometry import rotate_molecule_around_com
    from src.rotation_optimization.geometry.geometry_utils import rotate_molecule_around_com
 
Public API
----------
 
Molecule rotation
~~~~~~~~~~~~~~~~~
rotate_molecule_z(mol_data, theta_z, com)
    Rotate a molecule_data_dict entry around the Z-axis through *com*.
 
rotate_molecule_around_com(molecule_data, rotation_matrix)
    Rotate around the fixed COM using a 3x3 rotation matrix.
    Updates absolute_coordinates, relative_coordinates and quaternion.
 
calculate_alignment_rotation(vector1, vector2)
    Return the rotation matrix that aligns *vector1* onto *vector2*.
    Handles parallel and antiparallel degenerate cases.
 
rotation_around_axis(axis, angle_degrees)
    Rodrigues rotation matrix for rotation of *angle_degrees* around *axis*.
 
Ring geometry
~~~~~~~~~~~~~
get_ring_normal(mol_data)
    Unit normal to the pyranose ring plane from relative_coordinates.
    Requires 'C1'–'C5' in mol_data['carbon_map'].
 
get_ring_normal_absolute(mol_data)
    Same as get_ring_normal but works from absolute_coordinates.
 
get_ring_normal_from_positions(positions, ring_atoms)
    Unit normal given a full position array and a list of atom indices.
 
calculate_alignment_score(mol1, mol2)
    0.0 = ring normals parallel/antiparallel; 1.0 = perpendicular.
 
calculate_moment_of_inertia(ring_atoms, positions, masses, com)
    I_zz = sum m_i * r_i^2  (XY distance from COM).
 
Conformer position access
~~~~~~~~~~~~~~~~~~~~~~~~~
get_positions(conf, n_atoms) -> np.ndarray shape (n_atoms, 3)
set_positions(conf, positions, n_atoms)
get_atom_position(conf, atom_idx) -> np.ndarray shape (3,)
get_positions_for_atoms(conf, atom_indices) -> np.ndarray shape (n, 3)
 
Vector utilities
~~~~~~~~~~~~~~~~
calculate_distance(pos1, pos2) -> float
calculate_angle(pos1, pos2, pos3) -> float  (degrees)
calculate_center_of_mass(positions, masses) -> np.ndarray shape (3,)
cap_vectors(vectors, max_magnitude) -> np.ndarray
normalize_vector(vec) -> np.ndarray shape (3,)
get_perpendicular_vector(vec) -> np.ndarray shape (3,)
rodrigues_rotation(point, axis, angle, center) -> np.ndarray shape (3,)
 
Constants
~~~~~~~~~
VDW_RADII : dict  {'C': 1.70, 'O': 1.52, 'N': 1.55}
    Used by rotation/rotation_search for clash detection.
"""
