# Standard library
import os
import argparse

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from rdkit import Chem

# Custom modules - Monomer building
from src.monomer_building import monomer_building

from src.rotation_optimization.structure import building_chain
from src.rotation_optimization.structure import petn_building
from src.rotation_optimization.structure import lipid_building
from src.rotation_optimization.structure import saving_lps
from src.rotation_optimization.structure import peptide_building
from src.rotation_optimization.md import force_field_optimization
from src.rotation_optimization.md import ring_functions
from src.rotation_optimization.md import utils
from src.rotation_optimization.md.config import OptimizationConfig
from src import constants


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_circle_data(circle_path):
    """
    Load experimental circle positions and brightness from CSV.
    
    Returns:
        tuple: (circles, brightness)
            - circles: (N, 3) array of [X, Y, Height]
            - brightness: (N,) array of Z (Angstrom) values
    """
    df = pd.read_csv(circle_path)
    circles = df[["X (Angstrom)", "Y (Angstrom)", "Height"]].to_numpy()
    brightness = df["Z (Angstrom)"].to_numpy()
    return circles, brightness

def resolve_coordinates(coord_value, circles):
    """
    Resolve coordinates from either index or direct values.
    
    Args:
        coord_value: Either [index] or [x, y, z]
        circles: numpy array of coordinates from CSV
    
    Returns:
        numpy array of [x, y, z] coordinates
    """
    if len(coord_value) == 1:
        # Single value = index into circles array
        return circles[coord_value[0]]
    else:
        # Multiple values = direct coordinates
        return np.array(coord_value)

# ============================================================================
# CONFORMER GENERATION
# ============================================================================

def generate_conformers(sugars, num_conformers, max_keep):
    """
    Generate conformers for all sugar molecules.
    
    Returns:
        dict: {mol_name: list of conformers}
    """
    print("\n" + "="*70)
    print("CONFORMER GENERATION")
    print("="*70)
    
    conformers = {}
    
    for mol_name, smiles in sugars.items():
        print(f"\nGenerating conformers for {mol_name}...")
        conformers[mol_name] = monomer_building.generate_monomer_conformers(
            smiles, 
            num_conformers=num_conformers, 
            max_keep=max_keep,known_ring_type='chair',center='mass',
            use_cremer_pople=True
        )
        print(f"  ✓ Generated {len(conformers[mol_name])} conformers")
    
    return conformers

def analyze_conformers(conformers):
    """Analyze and plot conformer RMSD matrices."""
    print("\n" + "="*70)
    print("CONFORMER ANALYSIS")
    print("="*70)
    
    for mol_name, mol_conformers in conformers.items():
        print(f"\nAnalyzing {mol_name} conformers...")
        fig, rmsd_matrix = monomer_building.plot_conformer_analysis(mol_conformers)
        fig.suptitle(f'{mol_name.upper()} Conformer Analysis', fontsize=16)
        # plt.show()  # Uncomment to display plots

def extract_monomer_data(conformers):
    """
    Extract rigid monomer data from conformers.
    
    Returns:
        dict: {mol_name: monomer_data}
    """
    print("\n" + "="*70)
    print("EXTRACTING MONOMER DATA")
    print("="*70)
    
    monomer_data = {}
    
    for mol_name, mol_conformers in conformers.items():
        print(f"  Processing {mol_name}...")
        monomer_data[mol_name] = monomer_building.extract_rigid_monomer_data(mol_conformers)
    
    print(f"\n✓ Extracted data for {len(monomer_data)} monomers")
    return monomer_data


# ============================================================================
# MOLECULE POSITIONING
# ============================================================================

def translate_conformers_to_positions(monomer_data, experimental_positions, circles):
    """
    Translate conformers to experimental positions.
    
    Returns:
        tuple: (translated_conformers, carbon_vectors)
    """
    print("\n" + "="*70)
    print("TRANSLATING CONFORMERS TO EXPERIMENTAL POSITIONS")
    print("="*70)
    
    translated_conformers = {}
    carbon_vectors = {}
    
    for sugar, positions in experimental_positions.items():
        mol_data = monomer_data[sugar]
        
        if isinstance(positions, list):
            # Multiple positions for this sugar type
            translated_conformers[sugar] = {}
            carbon_vectors[sugar] = {}
            
            for pos in positions:
                trans = monomer_building.translate_to_experimental_com(
                    mol_data, circles[pos]
                )
                translated_conformers[sugar][pos] = trans
                carbon_vectors[sugar][pos] = monomer_building.construct_com_to_carbon_vectors(trans)
                print(f"  ✓ {sugar} at circle {pos}")
        else:
            # Single position for this sugar type
            trans = monomer_building.translate_to_experimental_com(
                mol_data, circles[positions]
            )
            translated_conformers[sugar] = trans
            carbon_vectors[sugar] = monomer_building.construct_com_to_carbon_vectors(trans)
            print(f"  ✓ {sugar} at circle {positions}")
    
    return translated_conformers, carbon_vectors

def build_molecule_dict(translated_conformers, experimental_positions, 
                        conformers_selection, brightness):
    """
    Build molecule_data_dict with selected conformers and brightness.
    
    Args:
        brightness: numpy array where brightness[i] corresponds to circle position i
    """
    print("\n" + "="*70)
    print(f"BUILDING MOLECULE DICTIONARY ({conformers_selection.upper()} strategy)")
    print("="*70)
    
    molecule_data_dict = {}
    mol_name_to_conformer = {}
    
    for sugar, positions in experimental_positions.items():
        confs = translated_conformers[sugar]
        
        if isinstance(positions, list):
            for pos in positions:
                key = f"{sugar}_{pos}"  # e.g., 'GlcN_1'
                mol_dict = monomer_building.select_conformer(
                    confs[pos], conformers_selection
                )
                mol_dict['brightness'] = brightness[pos]  # NEW: add brightness key
                molecule_data_dict[key] = mol_dict
                mol_name_to_conformer[key] = sugar
                print(f"  ✓ {key} (brightness: {brightness[pos]:.3f})")
        else:
            key = sugar  # e.g., 'GalA'
            mol_dict = monomer_building.select_conformer(
                confs, conformers_selection
            )
            mol_dict['brightness'] = brightness[positions]  # NEW: add brightness key
            molecule_data_dict[key] = mol_dict
            mol_name_to_conformer[key] = sugar
            print(f"  ✓ {key} (brightness: {brightness[positions]:.3f})")
    
    print(f"\n✓ Built molecule dictionary with {len(molecule_data_dict)} molecules")
    return molecule_data_dict, mol_name_to_conformer

def extract_initial_ring_coms_from_monomers(molecule_data_dict, final_no_h, pyranose_rings_no_h):
    """
    Extract initial ring COMs from molecule_data_dict.
    
    Each monomer in molecule_data_dict already has its COM set to the experimental
    position via translate_to_experimental_com(). We just need to match rings
    to monomers.
    
    Args:
        molecule_data_dict: Dict of monomer data with 'COM' and 'absolute_coordinates'
        final_no_h: RDKit molecule without hydrogens
        pyranose_rings_no_h: List of ring atom indices in final_no_h
    
    Returns:
        dict: Mapping ring_idx -> initial COM position (numpy array)
    """
    print("\n" + "="*70)
    print("EXTRACTING INITIAL RING COMs FROM EXPERIMENTAL POSITIONS")
    print("="*70)
    
    conf = final_no_h.GetConformer()
    n_atoms = final_no_h.GetNumAtoms()
    
    # Get positions from final molecule
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        positions[i] = [pos.x, pos.y, pos.z]
    
    # Calculate ring centers in final molecule
    ring_centers = []
    for ring_atoms in pyranose_rings_no_h:
        ring_pos = positions[ring_atoms]
        center = np.mean(ring_pos, axis=0)
        ring_centers.append(center)
    
    # Match rings to monomers using their COMs
    initial_ring_coms = {}
    RING_MATCHING_DISTANCE_THRESHOLD = 2.0  # Å
    
    for ring_idx, ring_center in enumerate(ring_centers):
        best_match = None
        min_dist = float('inf')
        
        # Try each monomer
        for mol_name, mol_data in molecule_data_dict.items():
            # Get monomer COM (already at experimental position!)
            monomer_com = np.array(mol_data['COM'])
            
            # Calculate distance
            dist = np.linalg.norm(ring_center - monomer_com)
            
            if dist < min_dist:
                min_dist = dist
                best_match = (mol_name, monomer_com)
        
        # Store the matched COM
        if best_match and min_dist < RING_MATCHING_DISTANCE_THRESHOLD:
            mol_name, monomer_com = best_match
            initial_ring_coms[ring_idx] = monomer_com
            
            print(f"  Ring {ring_idx}: matched to {mol_name}")
            print(f"    Experimental COM: ({monomer_com[0]:.2f}, {monomer_com[1]:.2f}, {monomer_com[2]:.2f}) Å")
            print(f"    Current ring COM: ({ring_center[0]:.2f}, {ring_center[1]:.2f}, {ring_center[2]:.2f}) Å")
            print(f"    Distance: {min_dist:.3f} Å")
        else:
            # Fall back to current position if no match (shouldn't happen normally)
            print(f"  Ring {ring_idx}: ⚠ No monomer match (dist={min_dist:.3f} Å)")
            print(f"    Using current COM: ({ring_center[0]:.2f}, {ring_center[1]:.2f}, {ring_center[2]:.2f}) Å")
            initial_ring_coms[ring_idx] = ring_center
    
    print(f"\n✓ Extracted initial COMs for {len(initial_ring_coms)} rings")
    return initial_ring_coms

# ============================================================================
# CHAIN BUILDING
# ============================================================================

def build_glycan_chain(molecule_data_dict, config, iterations):
    """
    Align molecules and create glycosidic bonds.
    
    Returns:
        tuple: (chain_dict, bonds_glyco, sorted_linkages)
    """
    print("\n" + "="*70)
    print("BUILDING GLYCAN CHAIN - DEBUG")
    print("="*70)
    
    print(f"\nmolecule_data_dict keys: {list(molecule_data_dict.keys())}")
    print(f"root_mol: {config['root_mol']}")
    
    print(f"\nexperimental_positions:")
    for mol, pos in config['experimental_positions'].items():
        print(f"  {mol}: {pos}")
    
    print(f"\ndirection (raw):")
    for d in config['direction']:
        print(f"  {d}")
    
    linkage_definitions = [tuple(link) for link in config['glycosidic_bonds']]
    rotation_definitions = [tuple(link) for link in config['direction']]
    orientation_constraints = config.get('orientation_constraints', False)

    print(f"\nrotation_definitions (after tuple conversion):")
    for r in rotation_definitions:
        print(f"  {r}")
    
    print(f"\nAligning molecules (iterations: {iterations})...")
    chain_dict, sorted_linkages = building_chain.align_and_position_molecules(
        molecule_data_dict, 
        rotation_definitions,
        root_mol=config['root_mol'],
        iterations=iterations,
        orientation_constraints=None
    )
    
    print("\nCreating glycosidic bonds...")
    bonds_glyco = building_chain.create_glycosidic_bonds_in_chain(
        chain_dict, linkage_definitions
    )
    
    print(f"\n✓ Built chain with {len(chain_dict)} molecules")
    print(f"✓ Created {len(bonds_glyco)} glycosidic bonds")
    
    return chain_dict, bonds_glyco, sorted_linkages

def build_glycan_chain_TOP_N(molecule_data_dict, config, iterations, n_polymers=5, n_quest_variants=6):
    """
    Align molecules and create glycosidic bonds for TOP N polymer variants.
    
    Args:
        molecule_data_dict: Dictionary of molecule data
        config: Configuration dictionary
        iterations: Number of optimization iterations
        n_polymers: Number of polymer variants to generate (default: 5)
        n_quest_variants: Number of QUEST variants per face (default: 6)
    
    Returns:
        tuple: (polymer_list, bonds_glyco_list, sorted_linkages)
            - polymer_list: List of chain_dicts (one per polymer variant)
            - bonds_glyco_list: List of bond lists (one per polymer variant)
            - sorted_linkages: Sorted linkage definitions (same for all)
    """
    print("\n" + "="*70)
    print(f"BUILDING TOP {n_polymers} GLYCAN CHAIN VARIANTS")
    print("="*70)
    
    print(f"\nmolecule_data_dict keys: {list(molecule_data_dict.keys())}")
    print(f"root_mol: {config['root_mol']}")
    
    print(f"\nexperimental_positions:")
    for mol, pos in config['experimental_positions'].items():
        print(f"  {mol}: {pos}")
    
    print(f"\ndirection (raw):")
    for d in config['direction']:
        print(f"  {d}")
    
    linkage_definitions = [tuple(link) for link in config['glycosidic_bonds']]
    rotation_definitions = [tuple(link) for link in config['direction']]
    orientation_constraints = config.get('orientation_constraints', None)
    
    print(f"\nrotation_definitions (after tuple conversion):")
    for r in rotation_definitions:
        print(f"  {r}")
    
    print(f"\nAligning molecules (iterations: {iterations}, n_polymers: {n_polymers})...")
    polymer_list, sorted_linkages = building_chain.align_and_position_molecules_TOP_N_POLYMERS(
        molecule_data_dict, 
        rotation_definitions,
        orientation_constraints=orientation_constraints,
        root_mol=config['root_mol'],
        iterations=iterations,
        n_polymers=n_polymers,
        n_quest_variants=n_quest_variants
    )
    
    print(f"\nCreating glycosidic bonds for {len([p for p in polymer_list if p is not None])} polymers...")
    
    # Create bonds for each polymer variant
    bonds_glyco_list = []
    
    for i, chain_dict in enumerate(polymer_list, 1):
        if chain_dict is None:
            bonds_glyco_list.append(None)
            print(f"  Polymer {i}: None (skipped)")
        else:
            bonds = building_chain.create_glycosidic_bonds_in_chain(
                chain_dict, linkage_definitions
            )
            bonds_glyco_list.append(bonds)
            print(f"  Polymer {i}: {len(bonds)} glycosidic bonds created")
    
    valid_count = sum(1 for p in polymer_list if p is not None)
    print(f"\n✓ Built {valid_count} polymer variants with {len(molecule_data_dict)} molecules each")
    print(f"✓ Created glycosidic bonds for each variant")
    
    return polymer_list, bonds_glyco_list, sorted_linkages


# ============================================================================
# PHOSPHATE LINKAGES
# ============================================================================

def add_phosphate_linkages(config, molecule_data_dict, chain_dict):
    """
    Add phosphate linkages between molecules.
    
    Returns:
        tuple: (phosphate_bonds_with_names, unbonded_monomers)
    """
    phosphate_linkages = config.get('Phosphate', [])
    
    if not phosphate_linkages:
        print("\n" + "="*70)
        print("NO PHOSPHATE LINKAGES")
        print("="*70)
        return [], {}
    
    print("\n" + "="*70)
    print("ADDING PHOSPHATE LINKAGES")
    print("="*70)
    
    chain_list = list(chain_dict.values())
    unbonded_monomers = {}
    phosphate_bonds_with_names = []
    
    # Phase 1: Optimize rotations
    print("\nPhase 1: Optimizing rotations...")
    for linkage in phosphate_linkages:
        mol_to_rotate, carbon_rotate, target_mol, target_carbon = linkage
        
        print(f"\n  Optimizing {mol_to_rotate} for linkage to {target_mol}")
        print(f"    ({mol_to_rotate}.{carbon_rotate} → {target_mol}.{target_carbon})")
        
        rotated_molecule, distance, opt_metadata = building_chain.find_rotation_for_phosphate_linkage(
            molecule_data_dict[mol_to_rotate],
            carbon_rotate,
            chain_dict[target_mol],
            target_carbon,
            chain_list,
            n_torsion_samples=300
        )
        
        molecule_data_dict[mol_to_rotate] = rotated_molecule
        print(f"    ✓ Optimized (distance: {distance:.2f} Å)")
    
    # Phase 2: Create phosphate bonds
    print("\nPhase 2: Creating phosphate bonds...")
    for linkage in phosphate_linkages:
        mol_to_rotate, carbon_rotate, target_mol, target_carbon = linkage
        
        print(f"\n  Creating bond: {target_mol}-P-{mol_to_rotate}")
        
        # Get carbon positions
        target_c_idx = chain_dict[target_mol]['carbon_map'][target_carbon]
        target_c_pos = np.array(chain_dict[target_mol]['absolute_coordinates'][target_c_idx])
        
        if mol_to_rotate in chain_dict:
            print(f"    Using chain coordinates for {mol_to_rotate}")
            rotate_c_idx = chain_dict[mol_to_rotate]['carbon_map'][carbon_rotate]
            rotate_c_pos = np.array(chain_dict[mol_to_rotate]['absolute_coordinates'][rotate_c_idx])
            mol_to_rotate_data = chain_dict[mol_to_rotate]
        else:
            print(f"    Using optimized coordinates for {mol_to_rotate}")
            rotate_c_idx = molecule_data_dict[mol_to_rotate]['carbon_map'][carbon_rotate]
            rotate_c_pos = np.array(molecule_data_dict[mol_to_rotate]['absolute_coordinates'][rotate_c_idx])
            mol_to_rotate_data = molecule_data_dict[mol_to_rotate]
        
        # Solve for phosphate position
        phosphate_position, ik_metadata = building_chain.solve_phosphate_position_IK(
            target_c_pos, rotate_c_pos
        )
        
        # Create bond
        bond_name = f"{target_mol}-P-{mol_to_rotate}"
        phosphate_bond = building_chain.create_phosphate_bond_from_IK(
            chain_dict[target_mol],
            target_carbon,
            mol_to_rotate_data,
            carbon_rotate,
            phosphate_position,
            bond_name
        )
        
        phosphate_bond['carbon1_name'] = target_carbon
        phosphate_bond['carbon2_name'] = carbon_rotate
        
        unbonded_monomers[mol_to_rotate] = mol_to_rotate_data
        phosphate_bonds_with_names.append((target_mol, mol_to_rotate, phosphate_bond))
        
        print(f"    ✓ Bond created: {bond_name}")
    
    print(f"\n✓ Created {len(phosphate_bonds_with_names)} phosphate bonds")
    return phosphate_bonds_with_names, unbonded_monomers

# ============================================================================
# PEtN MODIFICATIONS
# ============================================================================

def add_petn_modifications(config, chain_dict, circles):
    """
    Add phosphoethanolamine (PEtN) modifications.
    
    Returns:
        list: petn_linkages
    """
    petn_configs = config.get('PEtN', [])
    
    if not petn_configs:
        print("\n" + "="*70)
        print("NO PEtN MODIFICATIONS")
        print("="*70)
        return []
    
    print("\n" + "="*70)
    print("ADDING PEtN MODIFICATIONS")
    print("="*70)
    
    petn_linkages = []
    
    for petn_config in petn_configs:
        target_n_position = resolve_coordinates(petn_config['direction'], circles)
        mol_name, carbon_name = petn_config['connection']
        
        print(f"\nCreating PEtN: {mol_name}.{carbon_name}")
        
        # Get carbon position
        carbon_idx = chain_dict[mol_name]['carbon_map'][carbon_name]
        carbon_pos = np.array(chain_dict[mol_name]['absolute_coordinates'][carbon_idx])
        
        print(f"  Start: {carbon_pos}")
        print(f"  Target N: {target_n_position}")
        
        # Create PEtN linkage
        bond_name = f"{mol_name}-{carbon_name}-PEtN"
        petn_linkage = petn_building.create_phosphoethanolamine_compact(
            chain_dict[mol_name],
            carbon_name,
            target_n_position,
            bond_name
        )
        
        petn_linkages.append((mol_name, petn_linkage))
        print(f"  ✓ Created {bond_name}")
    
    print(f"\n✓ Created {len(petn_linkages)} PEtN modifications")
    return petn_linkages

# ============================================================================
# LIPID CHAINS
# ============================================================================

def initialize_lipid_storage(molecule_data_dict):
    """Initialize lipid_chains list in each molecule."""
    for mol_name in molecule_data_dict:
        if 'lipid_chains' not in molecule_data_dict[mol_name]:
            molecule_data_dict[mol_name]['lipid_chains'] = []

def add_lipid_chains(config, molecule_data_dict, circles):
    """
    Add lipid chains (both sugar-anchored and lipid-anchored).

    Returns:
        list: all_lipids
    """
    lipid_linkages = config.get('Lipids', [])

    if not lipid_linkages:
        print("\n" + "="*70)
        print("NO LIPID CHAINS")
        print("="*70)
        return []

    print("\n" + "="*70)
    print("ADDING LIPID CHAINS")
    print("="*70)

    initialize_lipid_storage(molecule_data_dict)
    all_lipids = []

    # PASS 1: Sugar-anchored lipids
    print("\nPASS 1: Building sugar-anchored lipids")
    print("-" * 60)

    sugar_anchored = [l for l in lipid_linkages if l['connection_anchor'] == 'sugar']

    for lipid_def in sugar_anchored:
        mol_name, carbon_name = lipid_def['connection']

        target_position = resolve_coordinates(lipid_def['direction'], circles)
        midpoint_value = lipid_def.get('midpoint', None)
        midpoint_coords = resolve_coordinates(midpoint_value, circles) if midpoint_value else None
        n_carbons = lipid_def.get('n_carbons', None)

        lipid_result = lipid_building.create_lipid_chain(
            mol_data=molecule_data_dict[mol_name],
            carbon_name=carbon_name,
            target_n_position=target_position,
            midpoint=midpoint_coords,
            linkage_type=lipid_def['linkage'],
            connection_anchor='sugar',
            n_carbons=n_carbons
        )

        lipid_result['mol_name'] = mol_name
        lipid_result['carbon_name'] = carbon_name
        lipid_result['connection_anchor'] = 'sugar'

        molecule_data_dict[mol_name]['lipid_chains'].append(lipid_result)
        all_lipids.append(lipid_result)

        n_str = f", n_carbons={n_carbons}" if n_carbons is not None else ""
        print(f"  {mol_name}:{carbon_name}{n_str}")

    # PASS 2: Lipid-anchored lipids
    print("\nPASS 2: Building lipid-anchored lipids")
    print("-" * 60)

    lipid_anchored = [l for l in lipid_linkages if l['connection_anchor'] == 'lipid']

    for lipid_def in lipid_anchored:
        connection_blob_idx = lipid_def.get('connection_blob', None)
        connection_blob_coords = resolve_coordinates(connection_blob_idx, circles) if connection_blob_idx else None

        target_position = resolve_coordinates(lipid_def['direction'], circles)
        midpoint_value = lipid_def.get('midpoint', None)
        midpoint_coords = resolve_coordinates(midpoint_value, circles) if midpoint_value else None
        n_carbons = lipid_def.get('n_carbons', None)

        target_mol = None
        for mol_name, mol_data in molecule_data_dict.items():
            if 'lipid_chains' in mol_data and len(mol_data['lipid_chains']) > 0:
                target_mol = mol_name
                break

        if target_mol is None:
            raise ValueError("No sugar-anchored lipids found for lipid-anchored attachment")

        lipid_result = lipid_building.create_lipid_chain(
            mol_data=molecule_data_dict[target_mol],
            carbon_name=None,
            target_n_position=target_position,
            midpoint=midpoint_coords,
            linkage_type=lipid_def['linkage'],
            connection_anchor='lipid',
            connection_blob=connection_blob_coords,
            n_carbons=n_carbons
        )

        lipid_result['mol_name'] = target_mol
        lipid_result['carbon_name'] = None
        lipid_result['connection_anchor'] = 'lipid'

        molecule_data_dict[target_mol]['lipid_chains'].append(lipid_result)
        all_lipids.append(lipid_result)

        n_str = f", n_carbons={n_carbons}" if n_carbons is not None else ""
        print(f"  Branched lipid on {target_mol}{n_str}")

    print(f"\n  Total lipids: {len(all_lipids)} ({len(sugar_anchored)} sugar + {len(lipid_anchored)} lipid)")
    return all_lipids

def extract_lipid_tail_indices(final_no_h, molecule_data_dict):
    """
    Extract ALL lipid chain carbon indices BEFORE optimization.
    Freezes ALL carbons in each lipid chain.
    These indices will be preserved through optimization.
    
    Returns:
        list: Atom indices of all lipid constraint carbons
    """
    import numpy as np
    
    print("\n" + "="*70)
    print("EXTRACTING LIPID CONSTRAINT INDICES (PRE-OPTIMIZATION)")
    print("FREEZING ALL LIPID CARBONS")
    print("="*70)
    
    lipid_constraint_indices = []
    conf = final_no_h.GetConformer()
    
    # Check if any lipids exist
    has_lipids = False
    for mol_name, mol_data in molecule_data_dict.items():
        if 'lipid_chains' in mol_data and len(mol_data['lipid_chains']) > 0:
            has_lipids = True
            break
    
    if not has_lipids:
        print("\n  No lipids present in structure")
        return []
    
    for mol_name, mol_data in molecule_data_dict.items():
        if 'lipid_chains' not in mol_data or len(mol_data['lipid_chains']) == 0:
            continue
        
        print(f"\n  {mol_name}: {len(mol_data['lipid_chains'])} lipid chain(s)")
        
        for lipid_idx, lipid_chain in enumerate(mol_data['lipid_chains']):
            if 'chain_carbons' not in lipid_chain or len(lipid_chain['chain_carbons']) == 0:
                print(f"    Lipid {lipid_idx}: No chain_carbons found")
                continue
            
            chain_carbons = lipid_chain['chain_carbons']
            chain_length = len(chain_carbons)
            
            print(f"    Lipid {lipid_idx}: {chain_length} carbons in chain")
            
            # Freeze ALL carbons in the chain
            atoms_found = 0
            atoms_not_found = 0
            
            for chain_idx, target_pos in enumerate(chain_carbons):
                target_pos = np.array(target_pos)
                
                min_dist = float('inf')
                best_idx = None
                
                # Find the closest carbon atom to this position
                for atom_idx in range(final_no_h.GetNumAtoms()):
                    atom = final_no_h.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        pos = conf.GetAtomPosition(atom_idx)
                        pos_arr = np.array([pos.x, pos.y, pos.z])
                        dist = np.linalg.norm(pos_arr - target_pos)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = atom_idx
                
                if best_idx is not None and min_dist < 0.5:  # 0.5 Å tolerance
                    if best_idx not in lipid_constraint_indices:  # Avoid duplicates
                        lipid_constraint_indices.append(best_idx)
                    atoms_found += 1
                    print(f"      C{chain_idx:2d}: atom C{best_idx} (dist={min_dist:.3f} Å)")
                else:
                    atoms_not_found += 1
                    print(f"      C{chain_idx:2d}: ⚠ NOT FOUND (min_dist={min_dist:.3f} Å)")
            
            print(f"    → Found {atoms_found}/{chain_length} carbons in this chain")
            if atoms_not_found > 0:
                print(f"    → WARNING: {atoms_not_found} carbons could not be matched")
    
    print(f"\n{'='*70}")
    print(f"TOTAL LIPID CONSTRAINT ATOMS: {len(lipid_constraint_indices)}")
    print(f"{'='*70}")
    print(f"  Indices: {sorted(lipid_constraint_indices)}")
    print(f"{'='*70}")
    
    return lipid_constraint_indices


def extract_petn_nitrogen_indices(final_no_h, petn_linkages):
    """
    Find the terminal N atom of each PEtN chain by position matching.
    Pinning N keeps the chain direction fixed so KDO optimizes around it.
    Heavy-atom indices are stable through AddHs.
    """
    import numpy as np

    if not petn_linkages:
        return []

    conf = final_no_h.GetConformer()
    n_atoms = final_no_h.GetNumAtoms()
    fixed = []

    print("\n" + "="*70)
    print("EXTRACTING PETN NITROGEN INDICES (N freeze)")
    print("="*70)

    for mol_name, petn_data in petn_linkages:
        target = np.array(petn_data['n_position'])
        best_idx, min_dist = None, float('inf')
        for atom_idx in range(n_atoms):
            if final_no_h.GetAtomWithIdx(atom_idx).GetSymbol() != 'N':
                continue
            pos = conf.GetAtomPosition(atom_idx)
            d = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - target)
            if d < min_dist:
                min_dist = d
                best_idx = atom_idx
        if best_idx is not None and min_dist < 0.5:
            fixed.append(best_idx)
            print(f"  {mol_name} PEtN N: atom #{best_idx} (dist={min_dist:.3f} Å)")
        else:
            print(f"  ⚠ {mol_name} PEtN N: NOT FOUND (min={min_dist:.4f} Å)")

    print(f"\n  Total PEtN N atoms frozen: {len(fixed)}")
    print("="*70)
    return fixed


def extract_lipid_tail_indices_3_points(final_no_h, molecule_data_dict):
    """
    Extract lipid chain carbon indices BEFORE optimization.
    Captures FIRST, MIDDLE, and LAST carbons of each lipid chain.
    These indices will be preserved through optimization.
    
    Returns:
        list: Atom indices of lipid constraint carbons (first, middle, last)
    """
    print("\n" + "="*70)
    print("EXTRACTING LIPID CONSTRAINT INDICES (PRE-OPTIMIZATION)")
    print("="*70)
    
    lipid_constraint_indices = []
    conf = final_no_h.GetConformer()
    
    # Check if any lipids exist
    has_lipids = False
    for mol_name, mol_data in molecule_data_dict.items():
        if 'lipid_chains' in mol_data and len(mol_data['lipid_chains']) > 0:
            has_lipids = True
            break
    
    if not has_lipids:
        print("\n  No lipids present in structure")
        return []
    
    for mol_name, mol_data in molecule_data_dict.items():
        if 'lipid_chains' not in mol_data or len(mol_data['lipid_chains']) == 0:
            continue
        
        print(f"\n  {mol_name}: {len(mol_data['lipid_chains'])} lipid chain(s)")
        
        for lipid_idx, lipid_chain in enumerate(mol_data['lipid_chains']):
            if 'chain_carbons' not in lipid_chain or len(lipid_chain['chain_carbons']) == 0:
                print(f"    Lipid {lipid_idx}: No chain_carbons found")
                continue
            
            chain_carbons = lipid_chain['chain_carbons']
            chain_length = len(chain_carbons)
            
            print(f"    Lipid {lipid_idx}: {chain_length} carbons in chain")
            
            # Define which carbons to constrain
            constraint_positions = []
            
            # First carbon (anchor)
            constraint_positions.append(('first', 0, chain_carbons[0]))
            
            # Middle carbon
            if chain_length >= 3:
                middle_idx = chain_length // 2
                constraint_positions.append(('middle', middle_idx, chain_carbons[middle_idx]))
            
            # Last carbon (tail tip)
            constraint_positions.append(('last', chain_length - 1, chain_carbons[-1]))
            
            # Find these carbons in the molecule
            for pos_label, chain_idx, target_pos in constraint_positions:
                target_pos = np.array(target_pos)
                
                min_dist = float('inf')
                best_idx = None
                
                for atom_idx in range(final_no_h.GetNumAtoms()):
                    atom = final_no_h.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        pos = conf.GetAtomPosition(atom_idx)
                        pos_arr = np.array([pos.x, pos.y, pos.z])
                        dist = np.linalg.norm(pos_arr - target_pos)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = atom_idx
                
                if best_idx is not None and min_dist < 0.5:  # 0.5 Å tolerance
                    lipid_constraint_indices.append(best_idx)
                    print(f"      {pos_label.capitalize():6s} (C{chain_idx:2d}): atom C{best_idx} (dist={min_dist:.3f} Å)")
                else:
                    print(f"      {pos_label.capitalize():6s} (C{chain_idx:2d}): ⚠ NOT FOUND (min_dist={min_dist:.3f} Å)")
    
    # Deduplicate (in case chains share atoms, though unlikely)
    lipid_constraint_indices = list(set(lipid_constraint_indices))
    
    print(f"\n{'='*70}")
    print(f"TOTAL LIPID CONSTRAINT ATOMS: {len(lipid_constraint_indices)}")
    print(f"{'='*70}")
    print(f"  Indices: {sorted(lipid_constraint_indices)}")
    print(f"{'='*70}")
    
    return lipid_constraint_indices
# ============================================================================
# PEPTIDE BUILDING
# ============================================================================

def build_peptide(config, circles):
    """
    Build peptide from configuration.
    
    Returns:
        peptide_data or None
    """
    if 'peptide' not in config:
        print("\n" + "="*70)
        print("NO PEPTIDE")
        print("="*70)
        return None
    
    print("\n" + "="*70)
    print("BUILDING PEPTIDE")
    print("="*70)
    
    peptide_sequence = config['peptide']['sequence']
    residue_configs = config['peptide']['residues']
    
    # Parse residue data
    residue_data = []
    for res_config in residue_configs:
        ca_pos_idx = res_config['ca_position']
        func_pos_idx = res_config.get('functional_position')
        
        res_data = {
            'aa': res_config['aa'],
            'ca_position': circles[ca_pos_idx].tolist()
        }
        
        if func_pos_idx is not None:
            res_data['functional_position'] = circles[func_pos_idx].tolist()
        else:
            res_data['functional_position'] = None
        
        residue_data.append(res_data)
        print(f"  {res_config['aa']}: Cα={ca_pos_idx}, Func={func_pos_idx}")

    cyclic = config['peptide'].get('cyclic', False)

    linker_data = None
    if cyclic and 'linker' in config['peptide']:
        linker_cfg = config['peptide']['linker']
        ca_idx = linker_cfg['ca_position']
        func_idx = linker_cfg.get('functional_position')
        linker_data = {
            'ca_position': circles[ca_idx].tolist(),
            'functional_position': circles[func_idx].tolist() if func_idx is not None else None
        }
    
    # Build peptide
    peptide_data = peptide_building.build_peptide_with_rdkit_ca(
        aa_sequence=peptide_sequence,
        residue_data=residue_data,
        cyclic=cyclic
    )   
    
    print(f"\n✓ Peptide built: {peptide_data['biln']}")
    return peptide_data

# ============================================================================
# PEPTIDE LINKAGES
# ============================================================================

def create_peptide_linkages(config, peptide_data, chain_dict):
    """
    Create sugar-peptide linkages.
    
    Returns:
        list: peptide_linkages
    """
    if peptide_data is None or 'peptide_bonds' not in config:
        print("\n" + "="*70)
        print("NO PEPTIDE LINKAGES")
        print("="*70)
        return []
    
    print("\n" + "="*70)
    print("CREATING PEPTIDE LINKAGES")
    print("="*70)
    
    peptide_linkages = []
    
    for bond_def in config['peptide_bonds']:
        
        # Extract bond parameters
        if isinstance(bond_def, dict):
            sugar_mol = bond_def['sugar_mol']
            sugar_carbon = bond_def['sugar_carbon']
            aa_type = bond_def['aa_type']
            residue_idx = bond_def['residue_index']
            glyc_type = bond_def['glycosylation_type']
            anomeric_config = bond_def.get('anomeric_config', 'beta')
            use_spacer = bond_def.get('use_spacer', False)
        
        elif isinstance(bond_def, list):
            sugar_mol, sugar_carbon, aa_type, residue_idx, glyc_type, anomeric_config, use_spacer = bond_def
        
        else:
            print(f"  ⚠ Skipping invalid bond definition: {bond_def}")
            continue
        
        print(f"\n  Creating {glyc_type} bond:")
        print(f"    Sugar: {sugar_mol}:{sugar_carbon}")
        print(f"    Peptide: {aa_type}[{residue_idx}]")
        print(f"    Config: {anomeric_config}, Spacer: {use_spacer}")
        
        # Get sugar mol_data from chain_dict
        if sugar_mol not in chain_dict:
            print(f"    ❌ Error: {sugar_mol} not found in chain_dict")
            continue
        
        sugar_mol_data = chain_dict[sugar_mol]
        
        # Find glycosylation site on peptide
        try:
            attachment_info = peptide_building.find_glycosylation_site(
                peptide_data=peptide_data,
                residue_index=residue_idx
            )
        except Exception as e:
            print(f"    ❌ Error finding glycosylation site: {e}")
            continue
        
        # Create linkage geometry
        target_atom_pos = np.array(attachment_info['position'])
        
        try:
            linkage_data = peptide_building.create_peptide_sugar_linkage(
                mol_data=sugar_mol_data,
                sugar_carbon=sugar_carbon,
                target_atom_pos=target_atom_pos,
                linkage_type=glyc_type,
                anomeric_config=anomeric_config,
                use_spacer=use_spacer
            )
            
            peptide_linkages.append({
                'sugar_mol': sugar_mol,
                'sugar_carbon': sugar_carbon,
                'aa_type': aa_type,
                'residue_index': residue_idx,
                'attachment': attachment_info,
                'linkage': linkage_data
            })
            
            print(f"    ✓ Created {glyc_type} linkage")
            print(f"      Target accuracy: {linkage_data.get('target_accuracy', 'N/A')}")
        
        except Exception as e:
            print(f"    ❌ Error creating linkage: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Created {len(peptide_linkages)} peptide linkages")
    return peptide_linkages

# ============================================================================
# STRUCTURE EXPORT
# ============================================================================

def export_structure(chain_dict, mol_name_to_conformer, bonds_glyco, 
                     linkage_definitions, phosphate_bonds_with_names,
                     petn_linkages, all_lipids, unbonded_monomers,
                     conformers, peptide_data, peptide_linkages,
                     name, conformers_selection):
    """
    Export structure - dispatches to glycopeptide or glycolipid exporter.
    
    Returns:
        RDKit molecule (without hydrogens)
    """
    print("\n" + "="*70)
    print("EXPORTING STRUCTURE")
    print("="*70)
    
    if peptide_data is not None:
        print("\n→ Using GLYCOPEPTIDE exporter")
        
        final_no_h, enforced_atoms = saving_lps.export_complete_glycopeptide(
            chain_dict=chain_dict,
            conformers=conformers,
            mol_name_to_conformer=mol_name_to_conformer,
            peptide_data=peptide_data,
            peptide_linkages=peptide_linkages,
            filename=f"{name}_glycopeptide",
            glycosidic_bonds=bonds_glyco,
            linkage_definitions=linkage_definitions
        )

        print(f"\n✓ Exported glycopeptide: {name}_glycopeptide.sdf")
        print(f"Captured {enforced_atoms} enforced trans bonds before optimization")
        return final_no_h, enforced_atoms
    
    else:
        print("\n→ Using GLYCOLIPID exporter")
        
        final_no_h = saving_lps.export_complete_structure_with_petn_and_lipids(
            chain_dict,
            mol_name_to_conformer=mol_name_to_conformer,
            glycosidic_bonds=bonds_glyco,
            linkage_definitions=linkage_definitions,
            phosphate_bonds_with_names=phosphate_bonds_with_names,
            petn_linkages=petn_linkages,
            lipid_linkages=all_lipids,
            unbonded_monomers=unbonded_monomers,
            conformers=conformers,
            filename=f"{name}_{conformers_selection}"
        )
        
        print(f"\n✓ Exported glycolipid: {name}_{conformers_selection}.sdf")
        print(f"  Atoms (no H): {final_no_h.GetNumAtoms()}")
        return final_no_h

# ============================================================================
# CONSTRAINT MANAGEMENT
# ============================================================================

def _find_linker_ring_atoms(mol):
    """
    Topology-based linker ring detection (phenyl + oxadiazole).
    Works regardless of aromaticity state (no SMARTS).
      - phenyl    : 6-membered ring, all Carbon
      - oxadiazole: 5-membered ring, 2N + 1O + 2C
    Returns (phenyl_indices, oxadiazole_indices) as lists.
    """
    from rdkit import Chem as _Chem
    _Chem.GetSymmSSSR(mol)
    ring_info = mol.GetRingInfo()
    phenyl, oxadiazole = [], []
    for ring in ring_info.AtomRings():
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if len(ring) == 6 and all(s == 'C' for s in syms):
            phenyl = list(ring)
        elif (len(ring) == 5 and
              syms.count('N') == 2 and syms.count('O') == 1 and syms.count('C') == 2):
            oxadiazole = list(ring)
    # No oxadiazole ⇒ no real linker. A linker-less peptide (e.g. a real protein
    # like RNAse) only has Phe/Tyr/Trp 6-C rings here — freezing one pins a side
    # chain during MD. Require the linker's unique oxadiazole fingerprint.
    if not oxadiazole:
        return [], []
    return phenyl, oxadiazole


def gather_fixed_atoms(peptide_data, enforced_atoms, lipid_tail_indices,
                       mol=None):
    """
    Gather fixed atoms based on structure type.
    Similar dispatch pattern to export_structure.

    - Glycopeptide: fix trans bond dihedral atoms from enforced_atoms
                    + linker ring atoms (phenyl + oxadiazole) if mol provided
    - Glycolipid:   fix lipid tail carbons

    Args:
        peptide_data:       Peptide structure data (None if glycolipid)
        enforced_atoms:     List of trans bond constraint dicts from peptide linkages
        lipid_tail_indices: List of lipid tail carbon indices
        mol:                Combined molecule (final_with_h) used to locate linker
                            ring atoms by topology. Pass None to skip linker freeze.

    Returns:
        list: Deduplicated list of atom indices to fix during optimization
    """
    print("\n" + "="*70)
    print("GATHERING FIXED ATOMS FOR OPTIMIZATION")
    print("="*70)

    fixed_atoms = []
    torsion_constraints = []

    if peptide_data is not None:
        # ====================================================================
        # GLYCOPEPTIDE PATH: glycosidic bonds → trans torsion restraints
        # ====================================================================
        print("\n→ Structure type: GLYCOPEPTIDE")
        print("→ Constraint mode: trans torsion restraints (glyco) + linker freeze")
        print("-" * 60)

        for bond_info in enforced_atoms:
            dihedral_atoms = bond_info['dihedral_atoms']
            aa_type = bond_info.get('aa_type', 'Unknown')
            glyc_type = bond_info.get('glycosylation_type', 'Unknown')

            # Hold trans via a 180° dihedral restraint (phases 2-3), NOT by
            # freezing these atoms — freezing both bonded atoms pins the bond
            # length so a stretched bond could never relax.
            torsion_constraints.append(tuple(dihedral_atoms))
            print(f"  {aa_type} ({glyc_type}): trans dihedral {dihedral_atoms} → torsion restraint")

        print(f"\n  ✓ Collected {len(torsion_constraints)} trans torsion restraints")

        # Freeze linker ring atoms so they stay flat (Z=0) during all MD phases
        if mol is not None:
            phenyl, oxadiazole = _find_linker_ring_atoms(mol)
            linker = phenyl + oxadiazole
            if linker:
                fixed_atoms.extend(linker)
                print(f"  Linker rings frozen: {len(linker)} atoms "
                      f"(phenyl={len(phenyl)}, oxadiazole={len(oxadiazole)})")
            else:
                print("  ⚠ Linker rings not found in mol — skipping linker freeze")

    else:
        # ====================================================================
        # GLYCOLIPID PATH: No atoms frozen — lipid tails free to optimize
        # ====================================================================
        print("\n→ Structure type: GLYCOLIPID")
        print("→ Constraint mode: None (lipid tails free)")
        print("-" * 60)
        print("  Lipid tails are free to move during optimization")

    # Deduplicate
    fixed_atoms = list(set(fixed_atoms))

    print("\n" + "="*70)
    print(f"TOTAL FIXED ATOMS: {len(fixed_atoms)}  |  "
          f"TORSION RESTRAINTS: {len(torsion_constraints)}")
    print("="*70)

    return fixed_atoms, torsion_constraints

# ============================================================================
# FORCE FIELD OPTIMIZATION
# ============================================================================

def prepare_structure_for_optimization(final_no_h, name, conformers_selection):
    """
    Add hydrogens and prepare structure for optimization.
    
    Returns:
        tuple: (final_with_h, pyranose_rings_no_h)
    """
    print("\n" + "="*70)
    print("STRUCTURE PREPARATION FOR OPTIMIZATION")
    print("="*70)
    
    # Add hydrogens
    print("\nAdding hydrogens...")
    final_with_h = force_field_optimization.fix_overvalent_carbons(
        building_chain.add_hydrogens_with_openbabel(final_no_h)
    )
    print(f"  ✓ Structure now has {final_with_h.GetNumAtoms()} atoms")
    
    # Detect rings (on molecule WITHOUT H for correct indices)
    print("\nDetecting pyranose rings...")
    Chem.GetSymmSSSR(final_no_h)
    pyranose_rings_no_h = ring_functions.detect_pyranose_rings(final_no_h)
    print(f"  ✓ Found {len(pyranose_rings_no_h)} pyranose rings")
    
    return final_with_h, pyranose_rings_no_h

def setup_optimization_constraints(orientation_constraints, molecule_data_dict, 
                                   final_no_h, pyranose_rings_no_h):
    """
    Setup ring constraints for optimization.
    
    Returns:
        tuple: (fixed_atoms, reference_normals)
    """
    if not orientation_constraints:
        print("\nNo orientation constraints - all rings free to move")
        return None, None
    
    print("\nApplying orientation constraints...")
    fixed_atoms_no_h, reference_normals, constrained_ring_ids = \
        force_field_optimization.get_fixed_atoms_from_constraints_by_position(
            orientation_constraints,
            molecule_data_dict,
            final_no_h,
            pre_detected_rings=pyranose_rings_no_h
        )
    
    # Atom indices preserved when H is added
    fixed_atoms = fixed_atoms_no_h
    print(f"  ✓ Constraining {len(constrained_ring_ids)} rings ({len(fixed_atoms)} atoms)")
    
    return fixed_atoms, reference_normals

def check_and_fix_geometry(final_with_h):
    """Check and fix molecular geometry."""
    print("\nChecking molecular geometry...")
    if not utils.check_geometry(final_with_h):
        print("  ⚠ Overlaps detected - fixing...")
        final_with_h = utils.fix_overlapping_atoms(final_with_h, min_distance=1.0)
        utils.check_geometry(final_with_h)
    else:
        print("  ✓ Geometry looks good")
    
    return final_with_h

def run_optimization(final_with_h, fixed_atoms, reference_normals,
                    name, conformers_selection, molecule_data_dict, lipid_tail_indices,
                    stm_npy_path=None, pyranose_rings_no_h=None, initial_ring_coms=None,
                    enable_phase1_kicks=False, torsion_constraints=None):
    """
    Run force field optimization using constants.
    
    Returns:
        tuple: (optimized_mol, success)
    """
    print("\n" + "="*70)
    print("FORCE FIELD OPTIMIZATION")
    print("="*70)
    
    # Configure optimization using constants from src/constants.py
    opt_config = OptimizationConfig(
        # Phase parameters
        relaxation_steps=constants.DEFAULT_RELAXATION_STEPS,
        compression_steps=constants.DEFAULT_COMPRESSION_STEPS,
        
        # Slab parameters
        slab_step_size=constants.DEFAULT_SLAB_STEP_SIZE,
        slab_step_interval=constants.DEFAULT_SLAB_STEP_INTERVAL,
        slab_force_scale=constants.DEFAULT_SLAB_FORCE_SCALE,
        
        # Physics parameters
        gravity=constants.DEFAULT_GRAVITY,
        step_size=constants.DEFAULT_STEP_SIZE,
        timestep=constants.DEFAULT_TIMESTEP,
        friction=constants.DEFAULT_FRICTION,
        
        # Ring monitoring
        check_rings_interval=constants.DEFAULT_CHECK_RINGS_INTERVAL,
        ring_tolerance_bond=constants.DEFAULT_BOND_TOLERANCE,
        ring_tolerance_angle=constants.DEFAULT_ANGLE_TOLERANCE,
        constrained_ring_com_limit=constants.DEFAULT_CONSTRAINED_RING_COM_LIMIT,
        free_ring_com_limit=constants.DEFAULT_FREE_RING_COM_LIMIT,
        
        # Minimization
        minimize_interval=constants.DEFAULT_MINIMIZE_INTERVAL,
        minimize_iterations=constants.DEFAULT_MINIMIZE_ITERATIONS,
        
        # Force limits
        max_force=constants.DEFAULT_MAX_FORCE,
        max_velocity=constants.DEFAULT_MAX_VELOCITY,
        
        # Constraints
        fixed_atoms=fixed_atoms,
        torsion_constraints=torsion_constraints,
        reference_normals=reference_normals,
        enable_ring_rotation=True,
        ring_rotation_friction=constants.DEFAULT_RING_ROTATION_FRICTION,

        # Phase 1 stochastic kicks — enable for glycopeptides with torsional clashes
        enable_phase1_kicks=enable_phase1_kicks,

        # Convergence
        enable_convergence=True,
        convergence_energy_threshold=0.01,
        convergence_force_threshold=0.1,
        convergence_rmsd_threshold=0.001,
        convergence_window=10,
        
        # Output
        save_images=False,  # Set to True for animation frames
        image_interval=constants.DEFAULT_IMAGE_INTERVAL,
        output_name=f"{name}_{conformers_selection}_optimization"
    )
    
    print("\nConfiguration (from constants):")
    print(f"  Relaxation steps: {opt_config.relaxation_steps}")
    print(f"  Compression steps: {opt_config.compression_steps}")
    print(f"  Gravity: {opt_config.gravity}")
    print(f"  Slab force scale: {opt_config.slab_force_scale}")
    print(f"  Ring rotation: {'enabled' if opt_config.enable_ring_rotation else 'disabled'}")
    if fixed_atoms:
        print(f"  Constrained atoms: {len(fixed_atoms)}")
    else:
        print(f"  Constrained atoms: None (all free)")
    
    # Run optimization
    print("\nStarting optimization...")
    optimized_mol, success = force_field_optimization.optimize_with_slab_and_rings(
        final_with_h,
        opt_config,
        molecule_data_dict=molecule_data_dict, lipid_tail_indices=lipid_tail_indices,stm_npy_path=stm_npy_path,    
        pyranose_rings=pyranose_rings_no_h,
        initial_ring_coms=initial_ring_coms
    )
    
    return optimized_mol, success

def save_optimization_results(final_with_h, optimized_mol, success, 
                              name, conformers_selection):
    """Save pre- and post-optimization structures."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save pre-optimization  (disabled: only the final *_optimized.sdf is kept)
    # utils.save_molecule(
    #     final_with_h,
    #     f"{name}_{conformers_selection}_pre_opt",
    #     file_format='sdf'
    # )

    # Save optimized
    if success:
        print("✓ Optimization completed successfully!")
        filename = f'{name}_{conformers_selection}_optimized'
        utils.save_molecule(optimized_mol, filename, file_format='sdf')
        print(f"\n✓ Saved: {filename}.sdf")
    else:
        print("⚠ Optimization had issues - saving last valid structure")
        filename = f'{name}_{conformers_selection}_partial'
        utils.save_molecule(optimized_mol, filename, file_format='sdf')
        print(f"\n⚠ Saved: {filename}.sdf")
    
    return success