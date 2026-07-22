"""
rotation — QUEST-based rotation search and geometric filtering.
 
No RDKit molecule modification takes place here.  All functions read
coordinate arrays from molecule_data_dict entries and return either a
rotated molecule_data_dict or a scalar score.
 
This sub-package is imported by structure/building_chain.py to align
monomers during chain assembly.
 
Public API
----------
 
Main entry points (called by structure/building_chain.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
find_rotation_FULL_3D_QUATERNION(
        template, mol_carbon, target_mol, target_carbon,
        existing_chain, fixed_com, n_samples, additional_bonds)
    Single-bond or multi-bond 3D rotation search with no surface
    orientation constraint.  Delegates to QUEST + uniform sampling
    with face (normal vs reflected) selection.
 
find_rotation_1D_Z_AXIS(
        template, donor_c, target, acceptor_c,
        chain_list, fixed_com, target_orientation, ...)
    Z-axis spin search for rings constrained to lie flat (or perpendicular)
    on the surface.  Establishes flatness via QUEST, then spins around Z.
    Tests both faces.
 
find_rotation_QUEST_VARIANTS_TOP_N(
        template, bonds_to_optimize, existing_chain, existing_linkages,
        fixed_com, n_top, n_quest_variants, max_uniform_samples, clash_factor)
    Returns the top-N rotations across systematic QUEST weight variants and
    uniform sampling.  Used by align_and_position_molecules_TOP_N_POLYMERS.
 
find_rotation_for_phosphate_linkage(
        mol_template, mol_carbon_name, target_mol, target_carbon_name,
        existing_chain, n_torsion_samples)
    Torsion search for phosphodiester linkage positioning.
 
orient_root_molecule(mol_data, target_orientation, n_samples)
    Rotate the fixed root monomer to satisfy a surface orientation constraint
    ('flat' or 'perpendicular') using coarse Z-search + scipy refinement.
 
Geometric scoring and filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
check_ring_clearance(mol, bonds_to_optimize, mode, ring_margin)
    'binary' mode: True/False hard filter.
    'penalty' mode: float score for ranking.
 
apply_all_filters(mol, bonds_to_optimize, existing_chain,
                  existing_linkages, clash_factor)
    Apply ring clearance + bond intersection checks in sequence.
    Returns a result dict with 'passes' bool and per-filter details.
 
detect_clashes(new_mol, existing_chain, clash_factor, exclude_last_n)
    Return a list of clashing atom pairs based on VDW radii.
 
calculate_collision_penalty(mol, chain_list, clash_factor)
    Continuous collision penalty (sum of exponential clash severities).
 
calculate_surface_orientation_score(mol_data, target_orientation)
    0.0 = perfect alignment with surface normal; 1.0 = worst.
 
get_orientation_dot_product(mol_data)
    n_ring · z_surface scalar.
 
Utilities
~~~~~~~~~
generate_uniform_rotations(n_samples) -> list[np.ndarray shape (3,3)]
    Quasi-uniform rotation matrices via Fibonacci sphere sampling.
 
flip_molecule_coordinates(mol_data, com) -> dict
    Mirror molecule through Y = COM[1] plane to test the reflected face.
 
solve_wahba_quest(observations, references, weights) -> np.ndarray (3,3)
    Core QUEST algorithm: find optimal rotation from vector pairs.
"""
