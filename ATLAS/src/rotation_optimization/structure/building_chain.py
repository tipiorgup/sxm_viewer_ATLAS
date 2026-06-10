import numpy as np
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from openbabel import openbabel as ob
from ..rotation.rotation_optimization import (
    find_rotation_FULL_3D_QUATERNION,
    find_rotation_1D_Z_AXIS,
    find_rotation_QUEST_VARIANTS_TOP_N,
    find_rotation_for_phosphate_linkage,
    orient_root_molecule,
    calculate_surface_orientation_score,
    get_orientation_dot_product,
    calculate_average_bond_distance,
)
from .bonds_creation import (
    create_glycosidic_bond,
    create_phosphate_bond_from_IK,
    find_hydroxyl_oxygen_at_carbon,
    solve_phosphate_position_IK,
)
from ..geometry.geometry_utils import get_ring_normal


"""
Change name of building_lps_VECTOR_PARETO, it does ont reflect its function, update all instances.
When should make a class when building the chain and we should use back propagation or optimization as stochasting gradient descent, so we update the roataion matrix on each
iteration until penalties are decreased.
"""

VDW_RADII = {
    'C': 1.70,  # Angstroms
    'O': 1.52,
    'N': 1.55
}

def align_and_position_molecules(molecule_data_dict, linkage_definitions,
                                orientation_constraints=None, root_mol='GlcN_1', iterations=500):
    """
    Align and position all molecules based on linkage definitions.
    
    Parameters:
    -----------
    molecule_data_dict : dict
        Dictionary of molecule data
    linkage_definitions : list
        List of linkage definitions. Accepts two formats:
        - Short format: [donor, donor_C, acceptor, acceptor_C]
        - Full format: [donor, donor_C, acceptor, acceptor_C, anomeric, name]
    orientation_constraints : dict or None
        Dictionary mapping molecule names to surface orientations:
        {'GlcNAc': 'flat', 'Sia': 'perpendicular'}
        - 'flat': ring lies flat on XY plane (n_ring · z ≈ 1)
        - 'perpendicular': ring stands up from XY plane (n_ring · z ≈ 0)
        Molecules not listed have no orientation constraint.
    root_mol : str
        Name of root molecule
    iterations : int
        Number of optimization iterations
    """
    print("="*70)
    print("ALIGNING AND POSITIONING MOLECULES")
    print("="*70)
    
    # ========================================================================
    # NORMALIZE LINKAGE FORMAT
    # ========================================================================

    normalized_linkages = []
    for link in linkage_definitions:
        # Check if it's a coordinate-based linkage with coordinate at END
        if len(link) == 3 and isinstance(link[2], (list, np.ndarray)):
            # Format: ['Glc', 'C1', [x,y,z]] - coordinate at end
            donor, dc, coord = link
            normalized_linkages.append([donor, dc, coord, None, 'beta', f"{donor}-coord"])
        
        # Check if it's a coordinate-based linkage with coordinate at START
        elif len(link) == 3 and isinstance(link[0], (list, np.ndarray)):
            # Format: [[x,y,z], 'Glc', 'C1'] - coordinate at start
            coord, donor, dc = link
            normalized_linkages.append([donor, dc, coord, None, 'beta', f"{donor}-coord"])
        
        elif len(link) == 4:
            # Short format: add default anomeric and name
            donor, dc, acceptor, ac = link
            anom = 'beta'
            name = f"{donor}-{acceptor}"
            normalized_linkages.append([donor, dc, acceptor, ac, anom, name])
        
        elif len(link) == 6:
            # Full format: use as-is
            normalized_linkages.append(link)
        
        else:
            raise ValueError(f"Invalid linkage format: {link}. Expected 3 (coordinate), 4, or 6 elements.")
    
    linkage_definitions = normalized_linkages
    
    # ========================================================================
    # PARSE ORIENTATION CONSTRAINTS
    # ========================================================================
    if orientation_constraints is None:
        orientation_constraints = {}
    
    print("\nOrientation constraints (surface = XY plane, Z = up):")
    if orientation_constraints:
        for mol, orientation in orientation_constraints.items():
            if orientation not in ['flat', 'perpendicular']:
                raise ValueError(f"Invalid orientation '{orientation}' for {mol}. Use 'flat' or 'perpendicular'")
            print(f"  {mol}: {orientation}")
    else:
        print("  None specified (all molecules can orient freely)")

    
    # ========================================================================
    # SMART ROOT DETECTION
    # ========================================================================
    
    # If user specified a root explicitly, trust them
    if root_mol and root_mol in molecule_data_dict:
        print(f"Using user-specified root: {root_mol}")
    else:
        # Auto-detect root
        
        # Option A: Molecule that bonds to a coordinate
        coordinate_roots = set()
        for link in linkage_definitions:
            acceptor = link[2]
            if isinstance(acceptor, (list, np.ndarray)):
                coordinate_roots.add(link[0])
        
        # Option B: Traditional root (never receives bonds)
        all_donors = {link[0] for link in linkage_definitions}
        all_acceptors = set()
        for link in linkage_definitions:
            acceptor = link[2]
            if isinstance(acceptor, str):
                all_acceptors.add(acceptor)
        traditional_roots = all_donors - all_acceptors
        
        # Prefer coordinate roots, fallback to traditional roots
        if coordinate_roots:
            root_mol = list(coordinate_roots)[0]
            print(f"Auto-detected coordinate-bonded root: {root_mol}")
        elif traditional_roots:
            root_mol = list(traditional_roots)[0]
            print(f"Auto-detected traditional root: {root_mol}")
        else:
            raise ValueError("No root molecule found")
    
    # ========================================================================
    # TOPOLOGICAL SORT
    # ========================================================================
    sorted_linkages = []
    in_chain = {root_mol}
    remaining = list(linkage_definitions)

    while remaining:
        initial_len = len(remaining)
        for link in remaining[:]:
            donor, dc, acceptor, ac, anom, name = link
            # Coordinate targets are always "available"
            # Molecule targets need to be in chain first
            if isinstance(acceptor, (list, np.ndarray)) or acceptor in in_chain:
                sorted_linkages.append(link)
                in_chain.add(donor)
                remaining.remove(link)
        
        if len(remaining) == initial_len:
            print(f"\nWARNING: Could not position all molecules. Remaining: {[l[0] for l in remaining]}")
            break
    
    # ========================================================================
    # BUILD LINKAGE MAP: Collect all bonds per molecule
    # ========================================================================
    molecule_linkage_map = {}
    for donor, dc, acceptor, ac, anom, name in sorted_linkages:
        if donor not in molecule_linkage_map:
            molecule_linkage_map[donor] = []
        molecule_linkage_map[donor].append({
            'donor_carbon': dc,
            'acceptor': acceptor,
            'acceptor_carbon': ac,
            'name': name
        })
    
    # ========================================================================
    # POSITION MOLECULES WITH MULTI-BOND AWARENESS
    # ========================================================================
    chain_dict = {root_mol: molecule_data_dict[root_mol].copy()}
    chain_list = [chain_dict[root_mol]]

    # Apply orientation constraint to root molecule if specified
    root_orientation = orientation_constraints.get(root_mol, None)
    if root_orientation:
        print(f"\nApplying orientation constraint to root molecule ({root_mol}): {root_orientation}")
        
        # Find rotation that satisfies orientation
        best_mol = orient_root_molecule(
            molecule_data_dict[root_mol],
            root_orientation,
            n_samples=iterations
        )
        
        chain_dict[root_mol] = best_mol
        chain_list[0] = best_mol
        
        # Report achievement
        orientation_score = calculate_surface_orientation_score(best_mol, root_orientation)
        dot_z = get_orientation_dot_product(best_mol)
        print(f"  Orientation score: {orientation_score:.4f} (0=perfect)")
        print(f"    n_ring · z = {dot_z:.4f}", end="")
        if root_orientation == 'flat':
            print(f" (target: ±1.0)")
        else:
            print(f" (target: 0.0)")
    
    processed_molecules = {root_mol}
    
    for donor_mol, donor_c, acceptor_mol, acceptor_c, anom, name in sorted_linkages:
        
        # Skip if we've already positioned this molecule
        if donor_mol in processed_molecules:
            continue
        
        print(f"\n{'='*70}")
        
        # Get ALL linkages for this molecule
        all_linkages = molecule_linkage_map.get(donor_mol, [])
        
        # Build linkage info with actual targets (only for already-positioned molecules)
        available_linkages = []
        for link in all_linkages:
            if isinstance(link['acceptor'], (list, np.ndarray)):
                # Coordinate target - always available
                available_linkages.append({
                    'donor_carbon': link['donor_carbon'],
                    'target': link['acceptor'],
                    'target_carbon': None,
                    'name': link['name']
                })
            elif link['acceptor'] in chain_dict:
                # Molecule target - only if already positioned
                available_linkages.append({
                    'donor_carbon': link['donor_carbon'],
                    'target': chain_dict[link['acceptor']],
                    'target_carbon': link['acceptor_carbon'],
                    'name': link['name']
                })
        
        if not available_linkages:
            # Acceptor not positioned yet — use original template position as fallback
            # so chain_dict is always complete and bonding can always proceed.
            print(f"  WARNING: No positioned acceptors for {donor_mol} — using original position as fallback")
            chain_dict[donor_mol] = molecule_data_dict[donor_mol].copy()
            chain_list.append(chain_dict[donor_mol])
            processed_molecules.add(donor_mol)
            continue

        future_bonds_to_this_mol = []

        for future_link in sorted_linkages:
            future_donor = future_link[0]
            future_donor_c = future_link[1]
            future_acceptor = future_link[2]
            future_acceptor_c = future_link[3]

            # If future molecule will bond TO current molecule being positioned
            if future_acceptor == donor_mol and future_donor not in processed_molecules:
                # This is a future bond!
                future_bonds_to_this_mol.append({
                    'future_donor': future_donor,
                    'future_donor_carbon': future_donor_c,
                    'my_carbon': future_acceptor_c,
                    'link_name': future_link[5]
                })

        if future_bonds_to_this_mol:
            print(f" Future bonds detected: {len(future_bonds_to_this_mol)} molecule(s) will bond to {donor_mol}")
            for fb in future_bonds_to_this_mol:
                print(f"    - {fb['future_donor']}.{fb['future_donor_carbon']} will bond to {donor_mol}.{fb['my_carbon']}")
        
        # Print what we're optimizing
        if len(available_linkages) > 1:
            print(f"Positioning {donor_mol} with {len(available_linkages)} simultaneous bond(s):")
            for info in available_linkages:
                print(f"  {info['name']}: {donor_mol}.{info['donor_carbon']}")
        else:
            info = available_linkages[0]
            if info['target_carbon']:
                print(f"{info['name']}: {donor_mol}.{info['donor_carbon']} → {info['target_carbon']}")
            else:
                print(f"{info['name']}: {donor_mol}.{info['donor_carbon']} → coordinate")
        
        template = molecule_data_dict[donor_mol]
        fixed_com = np.array(template['COM'])
        target_orientation = orientation_constraints.get(donor_mol, None)
        
        if target_orientation:
            print(f"  Surface orientation: {target_orientation}")
        
        # Choose optimization method — always fall back to experimental position if
        # the rotation optimizer raises so the molecule is never absent from chain_dict.
        best_mol = None
        try:
            if target_orientation and len(available_linkages) > 1:
                # Multi-bond with orientation constraint
                print(f"  Using 1D Z-axis rotation (optimizing {len(available_linkages)} bonds)")
                primary = available_linkages[0]
                additional = available_linkages[1:]
                additional_bonds = [{'donor_carbon': a['donor_carbon'],
                                     'target': a['target'],
                                     'target_carbon': a['target_carbon']}
                                    for a in additional]
                best_mol, distance, params = find_rotation_1D_Z_AXIS(
                    template, primary['donor_carbon'],
                    primary['target'], primary['target_carbon'],
                    chain_list, fixed_com,
                    target_orientation=target_orientation,
                    n_samples=iterations,
                    additional_bonds=additional_bonds
                )

            elif target_orientation:
                # Single bond with orientation constraint
                print(f"  Using 1D Z-axis rotation (360° search)")
                link_info = available_linkages[0]
                best_mol, distance, params = find_rotation_1D_Z_AXIS(
                    template, link_info['donor_carbon'],
                    link_info['target'], link_info['target_carbon'],
                    chain_list, fixed_com,
                    target_orientation=target_orientation,
                    n_samples=iterations,
                    existing_linkages=sorted_linkages, future_bonds_to_this_mol=future_bonds_to_this_mol
                )

            else:
                # No orientation constraint - use FULL 3D quaternion rotation
                print(f"  Using FULL 3D quaternion rotation")
                if len(available_linkages) > 1:
                    primary = available_linkages[0]
                    additional = available_linkages[1:]
                    additional_bonds = [{'donor_carbon': a['donor_carbon'],
                                         'target': a['target'],
                                         'target_carbon': a['target_carbon']}
                                        for a in additional]
                    best_mol, distance, params = find_rotation_FULL_3D_QUATERNION(
                        template, primary['donor_carbon'],
                        primary['target'], primary['target_carbon'],
                        chain_list, fixed_com,
                        n_samples=iterations,
                        additional_bonds=additional_bonds
                    )
                else:
                    link_info = available_linkages[0]
                    best_mol, distance, params = find_rotation_FULL_3D_QUATERNION(
                        template, link_info['donor_carbon'],
                        link_info['target'], link_info['target_carbon'],
                        chain_list, fixed_com,
                        n_samples=iterations
                    )

        except Exception as e:
            print(f"  WARNING: Rotation optimization failed for {donor_mol}: {e}")
            print(f"           Falling back to experimental position so bond can still be formed")
            best_mol = None

        if best_mol is None:
            print(f"  WARNING: Using experimental position for {donor_mol} (rotation could not be computed)")
            best_mol = molecule_data_dict[donor_mol].copy()

        chain_dict[donor_mol] = best_mol
        chain_list.append(best_mol)
        processed_molecules.add(donor_mol)
        
        # Report final orientation if constrained
        if target_orientation:
            final_score = calculate_surface_orientation_score(best_mol, target_orientation)
            dot_z = get_orientation_dot_product(best_mol)
            
            print(f"  Final orientation score: {final_score:.4f} (0=perfect)")
            print(f"    n_ring · z = {dot_z:.4f}", end="")
            
            if target_orientation == 'flat':
                print(f" (target: ±1.0)")
            else:
                print(f" (target: 0.0)")
        
        print(f"  ✓ Positioned {donor_mol} (C-C distance: {distance:.4f} Å)")
    
    print(f"\n{'='*70}")
    print(f"ALIGNMENT COMPLETE")
    print(f"  Monomers positioned: {len(chain_dict)}")
    if orientation_constraints:
        print(f"  Orientation constraints: {len(orientation_constraints)}")
    print("="*70)
    
    return chain_dict, sorted_linkages

def align_and_position_molecules_TOP_N_POLYMERS(
        molecule_data_dict, linkage_definitions,
        orientation_constraints=None, 
        root_mol='GlcN_1', 
        iterations=500,
        n_polymers=5,
        n_quest_variants=6):
    """
    Align and position molecules, generating N polymer variants.
    
    Strategy: Best polymer + single-monomer variations
      - Polymer 1: All rank1 (best of each monomer)
      - Polymer 2-N: Vary ONE monomer at a time to higher ranks
    
    Parameters:
    -----------
    molecule_data_dict : dict
        Dictionary of molecule data
    linkage_definitions : list
        List of linkage definitions
    orientation_constraints : dict or None
        Dictionary mapping molecule names to surface orientations
    root_mol : str
        Name of root molecule
    iterations : int
        Max uniform samples per face (default: 500)
    n_polymers : int
        Number of polymer variants to generate (default: 5)
    n_quest_variants : int
        Number of QUEST weight variants per face (default: 6)
    
    Returns:
    --------
    polymer_list : list of dicts
        List of chain_dicts, each representing one complete polymer.
        Length = n_polymers (with None for unavailable variants)
        [
            {'GlcN_1': mol_data, 'GalA': mol_data, ...},  # polymer 1
            {'GlcN_1': mol_data, 'GalA': mol_data, ...},  # polymer 2
            ...
        ]
    linkages : list
        Sorted linkage definitions
    """
    
    print("="*70)
    print(f"BUILDING TOP {n_polymers} POLYMER VARIANTS")
    print(f"  Strategy: Best + single-monomer variations")
    print(f"  QUEST variants per face: {n_quest_variants}")
    print("="*70)
    
    # ========================================================================
    # NORMALIZE LINKAGE FORMAT
    # ========================================================================
    normalized_linkages = []
    for link in linkage_definitions:
        if len(link) == 3 and isinstance(link[2], (list, np.ndarray)):
            donor, dc, coord = link
            normalized_linkages.append([donor, dc, coord, None, 'beta', f"{donor}-coord"])
        elif len(link) == 3 and isinstance(link[0], (list, np.ndarray)):
            coord, donor, dc = link
            normalized_linkages.append([donor, dc, coord, None, 'beta', f"{donor}-coord"])
        elif len(link) == 4:
            donor, dc, acceptor, ac = link
            anom = 'beta'
            name = f"{donor}-{acceptor}"
            normalized_linkages.append([donor, dc, acceptor, ac, anom, name])
        elif len(link) == 6:
            normalized_linkages.append(link)
        else:
            raise ValueError(f"Invalid linkage format: {link}")
    
    linkage_definitions = normalized_linkages
    
    # ========================================================================
    # PARSE ORIENTATION CONSTRAINTS
    # ========================================================================
    if orientation_constraints is None:
        orientation_constraints = {}
    
    print("\nOrientation constraints:")
    if orientation_constraints:
        for mol, orientation in orientation_constraints.items():
            if orientation not in ['flat', 'perpendicular']:
                raise ValueError(f"Invalid orientation '{orientation}' for {mol}")
            print(f"  {mol}: {orientation}")
    else:
        print("  None specified")
    
    # ========================================================================
    # ROOT DETECTION
    # ========================================================================
    if root_mol and root_mol in molecule_data_dict:
        print(f"\nUsing user-specified root: {root_mol}")
    else:
        coordinate_roots = set()
        for link in linkage_definitions:
            acceptor = link[2]
            if isinstance(acceptor, (list, np.ndarray)):
                coordinate_roots.add(link[0])
        
        all_donors = {link[0] for link in linkage_definitions}
        all_acceptors = set()
        for link in linkage_definitions:
            acceptor = link[2]
            if isinstance(acceptor, str):
                all_acceptors.add(acceptor)
        traditional_roots = all_donors - all_acceptors
        
        if coordinate_roots:
            root_mol = list(coordinate_roots)[0]
            print(f"\nAuto-detected coordinate-bonded root: {root_mol}")
        elif traditional_roots:
            root_mol = list(traditional_roots)[0]
            print(f"\nAuto-detected traditional root: {root_mol}")
        else:
            raise ValueError("No root molecule found")
    
    # ========================================================================
    # TOPOLOGICAL SORT
    # ========================================================================
    sorted_linkages = []
    in_chain = {root_mol}
    remaining = list(linkage_definitions)
    
    while remaining:
        initial_len = len(remaining)
        for link in remaining[:]:
            donor, dc, acceptor, ac, anom, name = link
            if isinstance(acceptor, (list, np.ndarray)) or acceptor in in_chain:
                sorted_linkages.append(link)
                in_chain.add(donor)
                remaining.remove(link)
        
        if len(remaining) == initial_len:
            print(f"\nWARNING: Could not position all molecules. Remaining: {[l[0] for l in remaining]}")
            break
    
    # ========================================================================
    # BUILD LINKAGE MAP
    # ========================================================================
    molecule_linkage_map = {}
    for donor, dc, acceptor, ac, anom, name in sorted_linkages:
        if donor not in molecule_linkage_map:
            molecule_linkage_map[donor] = []
        molecule_linkage_map[donor].append({
            'donor_carbon': dc,
            'acceptor': acceptor,
            'acceptor_carbon': ac,
            'name': name
        })
    
    # ========================================================================
    # COLLECT TOP N SOLUTIONS PER MOLECULE
    # ========================================================================
    # solutions_dict[mol_name] = [solution1, solution2, ...]
    solutions_dict = {}
    mol_order = [root_mol]  # Track order of processing
    
    # ROOT: Position and orient
    root_mol_data = molecule_data_dict[root_mol].copy()
    root_orientation = orientation_constraints.get(root_mol, None)
    
    if root_orientation:
        print(f"\nApplying orientation constraint to root ({root_mol}): {root_orientation}")
        best_root = orient_root_molecule(
            molecule_data_dict[root_mol],
            root_orientation,
            n_samples=iterations
        )
        root_mol_data = best_root
        
        orientation_score = calculate_surface_orientation_score(best_root, root_orientation)
        dot_z = get_orientation_dot_product(best_root)
        print(f"  Orientation score: {orientation_score:.4f}")
        print(f"    n_ring · z = {dot_z:.4f}")
    
    # Store root (only 1 solution)
    solutions_dict[root_mol] = [root_mol_data]
    processed_molecules = {root_mol}
    
    # POSITION ALL OTHER MOLECULES
    for donor_mol, donor_c, acceptor_mol, acceptor_c, anom, name in sorted_linkages:
        
        if donor_mol in processed_molecules:
            continue
        
        print(f"\n{'='*70}")
        
        mol_order.append(donor_mol)
        
        # Get available linkages
        all_linkages = molecule_linkage_map.get(donor_mol, [])
        available_linkages = []
        
        for link in all_linkages:
            if isinstance(link['acceptor'], (list, np.ndarray)):
                available_linkages.append({
                    'donor_carbon': link['donor_carbon'],
                    'target': link['acceptor'],
                    'target_carbon': None,
                    'name': link['name']
                })
            elif link['acceptor'] in solutions_dict:
                # Use BEST (rank1) solution from target
                target_best = solutions_dict[link['acceptor']][0]
                available_linkages.append({
                    'donor_carbon': link['donor_carbon'],
                    'target': target_best,
                    'target_carbon': link['acceptor_carbon'],
                    'name': link['name']
                })
        
        if not available_linkages:
            # Acceptor not positioned yet — fall back to original template position
            # so solutions_dict is always complete and bonding can always proceed.
            print(f"  WARNING: No positioned acceptors for {donor_mol} — using original position as fallback")
            solutions_dict[donor_mol] = [molecule_data_dict[donor_mol].copy()]
            processed_molecules.add(donor_mol)
            continue

        # Detect future bonds
        future_bonds_to_this_mol = []
        for future_link in sorted_linkages:
            future_donor = future_link[0]
            future_acceptor = future_link[2]
            future_acceptor_c = future_link[3]
            
            if future_acceptor == donor_mol and future_donor not in processed_molecules:
                future_bonds_to_this_mol.append({
                    'future_donor': future_donor,
                    'future_donor_carbon': future_link[1],
                    'my_carbon': future_acceptor_c,
                    'link_name': future_link[5]
                })
        
        # Print info
        if len(available_linkages) > 1:
            print(f"Positioning {donor_mol} with {len(available_linkages)} bonds:")
            for info in available_linkages:
                print(f"  {info['name']}")
        else:
            info = available_linkages[0]
            print(f"{info['name']}: {donor_mol}.{info['donor_carbon']}")
        
        template = molecule_data_dict[donor_mol]
        fixed_com = np.array(template['COM'])
        target_orientation = orientation_constraints.get(donor_mol, None)
        
        if target_orientation:
            print(f"  Surface orientation: {target_orientation}")
        
        # Build bonds_to_optimize
        bonds_to_optimize = []
        current_best_chain = [solutions_dict[m][0] for m in mol_order[:-1]]  # All positioned so far
        
        for link_info in available_linkages:
            if isinstance(link_info['target'], dict):
                target_idx = link_info['target']['carbon_map'][link_info['target_carbon']]
                target_pos = np.array(link_info['target']['absolute_coordinates'][target_idx])
            else:
                target_pos = np.array(link_info['target'])
            
            donor_idx = template['carbon_map'][link_info['donor_carbon']]
            
            bonds_to_optimize.append({
                'donor_idx': donor_idx,
                'donor_name': link_info['donor_carbon'],
                'target_pos': target_pos,
                'target': link_info['target'],
                'target_carbon': link_info.get('target_carbon')
            })
        
        # OPTIMIZE WITH TOP N
        mol_solutions = None
        try:
            if target_orientation:
                print(f"  → Using 1D Z-axis optimizer (orientation constrained)")
                print(f"  [INFO] 1D Z-axis TOP N not yet implemented, using single solution")
                primary = available_linkages[0]
                additional = [{'donor_carbon': a['donor_carbon'],
                               'target': a['target'],
                               'target_carbon': a['target_carbon']}
                              for a in available_linkages[1:]]
                best_mol, distance, params = find_rotation_1D_Z_AXIS(
                    template, primary['donor_carbon'],
                    primary['target'], primary['target_carbon'],
                    current_best_chain, fixed_com,
                    target_orientation=target_orientation,
                    n_samples=iterations,
                    additional_bonds=additional if additional else None,
                    existing_linkages=sorted_linkages,
                    future_bonds_to_this_mol=future_bonds_to_this_mol
                )
                mol_solutions = [best_mol]

            else:
                print(f"  → Using 3D QUEST variants TOP N optimizer")
                top_solutions, metadata = find_rotation_QUEST_VARIANTS_TOP_N(
                    template=template,
                    bonds_to_optimize=bonds_to_optimize,
                    existing_chain=current_best_chain,
                    existing_linkages=sorted_linkages,
                    fixed_com=fixed_com,
                    n_top=n_polymers,
                    n_quest_variants=n_quest_variants,
                    max_uniform_samples=iterations,
                    clash_factor=0.6
                )
                if not top_solutions:
                    print(f"  ✗ WARNING: No valid rotations found for {donor_mol}")
                    mol_solutions = [molecule_data_dict[donor_mol].copy()]
                else:
                    mol_solutions = [sol['mol'] for sol in top_solutions]
                    print(f"  ✓ Found {len(mol_solutions)} valid solutions for {donor_mol}")

        except Exception as e:
            print(f"  WARNING: Rotation optimization failed for {donor_mol}: {e}")
            print(f"           Falling back to experimental position so bond can still be formed")
            mol_solutions = None

        if mol_solutions is None:
            print(f"  WARNING: Using experimental position for {donor_mol}")
            mol_solutions = [molecule_data_dict[donor_mol].copy()]

        # Store solutions for this molecule
        solutions_dict[donor_mol] = mol_solutions
        processed_molecules.add(donor_mol)
    
    # ========================================================================
    # GENERATE N POLYMER VARIANTS
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"GENERATING {n_polymers} POLYMER VARIANTS")
    print(f"{'='*70}")
    
    polymer_list = []
    
    # Polymer 1: ALL rank1 (best)
    polymer_1 = {}
    for mol_name in mol_order:
        polymer_1[mol_name] = solutions_dict[mol_name][0]
    polymer_list.append(polymer_1)
    print(f"  Polymer 1: All rank1 (best) ✓")
    
    # Generate variations (vary one monomer at a time)
    variation_count = 1
    rank = 2  # Start with rank 2
    
    while variation_count < n_polymers:
        made_variation = False
        
        for mol_name in mol_order:
            if mol_name == root_mol:
                continue  # Never vary root
            
            # Check if this molecule has this rank available
            if rank - 1 < len(solutions_dict[mol_name]):
                # Create variation
                polymer_variant = {}
                for m in mol_order:
                    if m == mol_name:
                        polymer_variant[m] = solutions_dict[mol_name][rank - 1]
                    else:
                        polymer_variant[m] = solutions_dict[m][0]  # Use rank1
                
                polymer_list.append(polymer_variant)
                variation_count += 1
                made_variation = True
                
                print(f"  Polymer {variation_count + 1}: Vary {mol_name} to rank{rank} ✓")
                
                if variation_count >= n_polymers - 1:
                    break
        
        # If no more variations at this rank, try next rank
        if not made_variation:
            rank += 1
            
            # Check if ANY molecule has solutions at this rank
            has_any = any(rank - 1 < len(solutions_dict[mol_name]) 
                         for mol_name in mol_order if mol_name != root_mol)
            
            if not has_any:
                # No more variations possible
                break
    
    # Pad with None if needed
    while len(polymer_list) < n_polymers:
        polymer_list.append(None)
        print(f"  Polymer {len(polymer_list)}: None (not enough variations)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"POLYMER GENERATION COMPLETE")
    print(f"  Total polymers: {sum(1 for p in polymer_list if p is not None)}/{n_polymers}")
    print(f"  Monomers per polymer: {len(mol_order)}")
    print(f"{'='*70}")
    
    return polymer_list, sorted_linkages

def create_glycosidic_bonds_in_chain(chain_dict, linkage_definitions):
    """
    Create glycosidic bonds between already-positioned molecules.
    
    Parameters:
    -----------
    chain_dict : dict
        Dictionary of positioned molecules {mol_name: mol_data}
    linkage_definitions : list
        List of linkage definitions (donor, donor_c, acceptor, acceptor_c, anom, name)
    
    Returns:
    --------
    bonds : list
        List of glycosidic bond dictionaries
    """
    print("\n" + "="*70)
    print("CREATING GLYCOSIDIC BONDS")
    print("="*70)
   
    bonds = {}
   
    for donor_mol, donor_c, acceptor_mol, acceptor_c, anom, name in linkage_definitions:
        print(f"\n{name}: {donor_mol}.{donor_c} → {acceptor_mol}.{acceptor_c}")

        # Coordinate-based linkages have a numpy array as acceptor — not a molecule bond
        if not isinstance(acceptor_mol, str):
            print(f"  Skipping coordinate-based linkage {name} (acceptor is a position, not a molecule)")
            continue
        if acceptor_mol not in chain_dict:
            print(f"  WARNING: {acceptor_mol} missing from chain_dict — skipping bond {name}")
            continue
        if donor_mol not in chain_dict:
            print(f"  WARNING: {donor_mol} missing from chain_dict — skipping bond {name}")
            continue

        target = chain_dict[acceptor_mol]
        donor = chain_dict[donor_mol]

        try:
            bond = create_glycosidic_bond(
                target, acceptor_c, donor, donor_c, anom, name
            )
            bonds[name] = bond
        except Exception as e:
            print(f"  WARNING: Bond creation failed for {name}: {e} — skipping")
   
    print(f"\n{'='*70}")
    print(f"GLYCOSIDIC BONDS COMPLETE")
    print(f"  Total bonds: {len(bonds)}")
    print("="*70)
   
    return bonds

def add_hydrogens_with_openbabel(
    mol,
    optimize_geometry=False,
    optimization_steps=500,
    filename=None,
    verbose=False,
    return_as_openbabel=False
):
    """
    Add hydrogens to a molecular structure using Open Babel.
    """
    
    # Validate input
    if mol is None:
        raise ValueError("Input molecule is None")
    
    if mol.GetNumConformers() == 0:
        raise ValueError("Input molecule has no 3D conformer")
    
    # ========================================================================
    # STEP 1: Convert RDKit molecule to Open Babel
    # ========================================================================
    
    # Convert RWMol to Mol if needed
    if isinstance(mol, Chem.RWMol):
        mol = mol.GetMol()

    # ------------------------------------------------------------------
    # Large molecules: skip the OpenBabel MDL round-trip entirely.
    # V2000 molblocks cap at 999 atoms; a protein like RNAse (~1000 heavy
    # atoms, ~2000 with H) overflows the format, RDKit fails to parse Open
    # Babel's output ("bad bond CFG"), and the old code then returned a raw
    # OBMol that crashed fix_overvalent_carbons. RDKit's own AddHs is
    # deterministic and needs no text round-trip. The marker print also lets
    # us confirm at runtime that THIS (fixed) code is the version executing.
    # ------------------------------------------------------------------
    if not return_as_openbabel and mol.GetNumAtoms() > 999:
        print(f"[add_hydrogens_with_openbabel] FIXED PATH: large molecule "
              f"({mol.GetNumAtoms()} atoms) → RDKit AddHs, skipping OpenBabel")
        mol_with_h = Chem.AddHs(mol, addCoords=True)
        try:
            Chem.SanitizeMol(mol_with_h)
        except Exception as sanitize_error:
            print(f"  Warning: sanitize after RDKit AddHs: "
                  f"{str(sanitize_error)[:100]}")
        if filename is not None:
            try:
                out = filename if filename.endswith(('.sdf', '.mol')) else filename + '.sdf'
                writer = Chem.SDWriter(out)
                writer.write(mol_with_h)
                writer.close()
            except Exception as save_error:
                print(f"  Warning: could not save {filename}: {save_error}")
        return mol_with_h

    # V2000 molblocks use fixed 3-char index fields → hard 999-atom limit.
    # Large structures (e.g. RNAse, ~2000 atoms with H) overflow it and produce
    # a corrupt bond block. Use V3000 (no limit) once we exceed the limit; small
    # molecules keep V2000 so existing behavior is unchanged.
    sdf_string = Chem.MolToMolBlock(mol, forceV3000=mol.GetNumAtoms() > 999)

    # Create Open Babel molecule
    ob_mol = ob.OBMol()
    ob_conv = ob.OBConversion()
    ob_conv.SetInFormat("sdf")
    ob_conv.ReadString(ob_mol, sdf_string)
    
    # ========================================================================
    # STEP 2: Add hydrogens with Open Babel
    # ========================================================================
    ob_mol.AddHydrogens()
    
    num_atoms_with_h = ob_mol.NumAtoms()
    num_h_added = num_atoms_with_h - mol.GetNumAtoms()
    
    # ========================================================================
    # STEP 3: Optimize geometry (optional)
    # ========================================================================
    if optimize_geometry:
        if verbose:
            print(f"\nOptimizing geometry ({optimization_steps} steps)...")
        
        # Try MMFF94 first
        ff = ob.OBForceField.FindForceField("MMFF94")
        
        if ff is None:
            if verbose:
                print("MMFF94 not available, trying UFF...")
            ff = ob.OBForceField.FindForceField("UFF")
        
        if ff is not None:
            # Setup the force field
            setup_success = ff.Setup(ob_mol)
            
            if setup_success:
                if verbose:
                    print(f"Using force field: {ff.GetName()}")
                    
                # Get initial energy
                initial_energy = ff.Energy()
                if verbose:
                    print(f"Initial energy: {initial_energy:.2f} kcal/mol")
                
                # Do the optimization with more verbose output
                if verbose:
                    print(f"Running {optimization_steps} optimization steps...")
                
                # Use SteepestDescent first (more stable for bad geometries)
                ff.SteepestDescent(min(100, optimization_steps // 5))
                
                # Then use ConjugateGradients (converges faster)
                ff.ConjugateGradients(optimization_steps)
                
                # Update the molecule coordinates from the force field
                ff.GetCoordinates(ob_mol)
                
                # Get final energy
                final_energy = ff.Energy()
                energy_change = initial_energy - final_energy
                
                if verbose:
                    print(f"Final energy:   {final_energy:.2f} kcal/mol")
                    print(f"Energy change:  {energy_change:.2f} kcal/mol")
                    
                    if abs(energy_change) < 0.1:
                        print("Energy barely changed - structure may already be optimal")
                        print("or optimization didn't work (try more steps)")
                    elif energy_change > 0:
                        print("Optimization successful - energy decreased")
                    else:
                        print("Warning: Energy increased")
            else:
                if verbose:
                    print("Could not setup force field")
        else:
            if verbose:
                print("No force field available")
    
    # ========================================================================
    # STEP 4: Convert back to RDKit (or save directly from Open Babel)
    # ========================================================================
    
    if return_as_openbabel:
        if filename is not None:
          
            out_format = filename.split('.')[-1] if '.' in filename else 'sdf'
            ob_conv.SetOutFormat(out_format)
            ob_conv.WriteFile(ob_mol, filename)
            
            if verbose:
                print("File saved successfully")
        return ob_mol
    
    # Try to convert back to RDKit
    if verbose:
        print("\nConverting back to RDKit format...")
    
    # Convert to SDF string. After AddHydrogens the molecule is larger, so the
    # V2000 999-atom limit bites here too — force Open Babel to emit V3000 (MDL
    # output option "3") for big molecules so RDKit can parse it back.
    ob_conv.SetOutFormat("sdf")
    if ob_mol.NumAtoms() > 999:
        ob_conv.AddOption("3", ob.OBConversion.OUTOPTIONS)
    sdf_with_h = ob_conv.WriteString(ob_mol)
    
    # Try reading into RDKit
    try:
        mol_with_h = Chem.MolFromMolBlock(sdf_with_h, removeHs=False, sanitize=False)
        
        if mol_with_h is None:
            raise ValueError("RDKit returned None when reading molecule")
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol_with_h)
            if verbose:
                print(f"Converted back to RDKit successfully")
        except Exception as sanitize_error:
            if verbose:
                print(f"Warning: Sanitization issue: {str(sanitize_error)[:100]}")
                print("Molecule converted but may have valence issues")
    
    except Exception as e:
        error_msg = str(e)
        print(f"  OpenBabel→RDKit MDL round-trip failed ({error_msg[:120]}); "
              f"falling back to RDKit AddHs")

        # The MDL round-trip is fragile for large molecules (V2000's 999-atom
        # limit corrupts the bond block → "bad bond CFG"). Rather than return a
        # raw OBMol — which callers like fix_overvalent_carbons cannot consume
        # (Chem.RWMol(OBMol) raises) — bypass OpenBabel for the conversion and
        # add hydrogens with RDKit directly. `mol` here is the pre-H RDKit mol
        # with a valid conformer, so addCoords gives reasonable H positions; the
        # geometry is refined later during optimization anyway.
        if not return_as_openbabel:
            try:
                mol_with_h = Chem.AddHs(mol, addCoords=True)
                try:
                    Chem.SanitizeMol(mol_with_h)
                except Exception as sanitize_error:
                    print(f"  Warning: sanitize after RDKit AddHs: "
                          f"{str(sanitize_error)[:100]}")
                # fall through to the shared save/return path below
            except Exception as rdkit_error:
                print(f"  RDKit AddHs fallback also failed "
                      f"({str(rdkit_error)[:120]}); returning OBMol")
                if filename is not None:
                    out_format = filename.split('.')[-1] if '.' in filename else 'sdf'
                    ob_conv.SetOutFormat(out_format)
                    ob_conv.WriteFile(ob_mol, filename)
                return ob_mol
        else:
            # Caller explicitly asked for an OBMol.
            if filename is not None:
                out_format = filename.split('.')[-1] if '.' in filename else 'sdf'
                ob_conv.SetOutFormat(out_format)
                ob_conv.WriteFile(ob_mol, filename)
            return ob_mol
    
   
    # ========================================================================
    # STEP 5: Save to file (optional)
    # ========================================================================
    if filename is not None and not isinstance(mol_with_h, ob.OBMol):
        # Only save here if we haven't already saved (i.e., mol_with_h is RDKit)
        if verbose:
            print(f"\nSaving structure to: {filename}")
        
        try:
            # Determine format from extension
            if filename.endswith('.sdf') or filename.endswith('.mol'):
                writer = Chem.SDWriter(filename)
                writer.write(mol_with_h)
                writer.close()
            elif filename.endswith('.pdb'):
                Chem.MolToPDBFile(mol_with_h, filename)
            else:
                # Default to SDF
                if not filename.endswith('.sdf'):
                    filename += '.sdf'
                writer = Chem.SDWriter(filename)
                writer.write(mol_with_h)
                writer.close()
            
            if verbose:
                print("File saved successfully")
        
        except Exception as e:
            print(f"ERROR saving file: {e}")
            raise
    
    return mol_with_h


