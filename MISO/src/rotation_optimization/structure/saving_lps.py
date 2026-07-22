import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from openbabel import openbabel as ob
from .bonds_creation import (
    find_hydroxyl_oxygen_at_carbon,
    solve_phosphate_position_IK,
    create_phosphate_bond_from_IK,
)
from ..geometry.geometry_utils import calculate_alignment_rotation
from .building_chain import add_hydrogens_with_openbabel

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def adjust_index_after_removals(original_idx, removed_indices):
    """Adjust an index after atoms have been removed."""
    return original_idx - sum(1 for r in removed_indices if r < original_idx)

def find_atom_by_position(conformer, target_pos, start_idx=0, end_idx=None, tolerance=0.01):
    """Find atom index by 3D position within a range."""
    target = np.array(target_pos)
    end = end_idx if end_idx is not None else conformer.GetOwningMol().GetNumAtoms()
    
    for i in range(start_idx, end):
        pos = conformer.GetAtomPosition(i)
        pos_arr = np.array([pos.x, pos.y, pos.z])
        if np.linalg.norm(pos_arr - target) < tolerance:
            return i
    return None

def find_oh_group_near_carbon(mol, conformer, carbon_idx, min_dist=1.3, max_dist=1.6):
    """Find OH group bonded to a specific carbon.

    Uses BOTH distance (1.3-1.6 Å) AND topology: the oxygen must be an actual
    RDKit neighbor of carbon_idx, not just spatially close.
    """
    c_pos = conformer.GetAtomPosition(carbon_idx)
    c_arr = np.array([c_pos.x, c_pos.y, c_pos.z])

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == 'O':
            pos = conformer.GetAtomPosition(i)
            pos_arr = np.array([pos.x, pos.y, pos.z])
            dist = np.linalg.norm(pos_arr - c_arr)

            if min_dist < dist < max_dist:
                neighbor_indices = [n.GetIdx() for n in atom.GetNeighbors()]
                # Must be topologically bonded to the target carbon
                if carbon_idx not in neighbor_indices:
                    continue
                neighbor_symbols = [mol.GetAtomWithIdx(j).GetSymbol() for j in neighbor_indices]
                if neighbor_symbols == ['C'] or set(neighbor_symbols) == {'C', 'H'}:
                    return i
    return None

def find_nh2_group_near_carbon(mol, conformer, carbon_idx, tolerance=1.6):
    """Find NH2 group bonded to a specific carbon, returns [N_idx, H_idx1, H_idx2, ...]."""
    c_pos = conformer.GetAtomPosition(carbon_idx)
    c_arr = np.array([c_pos.x, c_pos.y, c_pos.z])
    
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == 'N':
            pos = conformer.GetAtomPosition(i)
            pos_arr = np.array([pos.x, pos.y, pos.z])
            
            if np.linalg.norm(pos_arr - c_arr) < tolerance:
                # Found N, now find attached H atoms
                atoms_to_remove = [i]
                for j in range(mol.GetNumAtoms()):
                    h_atom = mol.GetAtomWithIdx(j)
                    if h_atom.GetSymbol() == 'H':
                        h_pos = conformer.GetAtomPosition(j)
                        h_arr = np.array([h_pos.x, h_pos.y, h_pos.z])
                        if np.linalg.norm(h_arr - pos_arr) < 1.2:
                            atoms_to_remove.append(j)
                return atoms_to_remove
    return []

# ============================================================================
# MOLECULE COLLECTION
# ============================================================================

def collect_molecules(chain_dict, unbonded_monomers, mol_name_to_conformer, conformers):
    """Collect and prepare all molecules with their coordinates."""
    print("\nCollecting all molecules...")
    processed_mols = []
    atom_offset = 0
    
    all_mol_names = set(chain_dict.keys())
    if unbonded_monomers is not None:
        all_mol_names |= set(unbonded_monomers.keys())
    
    for mol_name in sorted(all_mol_names):
        if mol_name in chain_dict:
            coords = np.array(chain_dict[mol_name]['absolute_coordinates'])
            carbon_map = chain_dict[mol_name]['carbon_map']
        else:
            coords = np.array(unbonded_monomers[mol_name]['absolute_coordinates'])
            carbon_map = unbonded_monomers[mol_name]['carbon_map']
        
        conformer_type = mol_name_to_conformer[mol_name]
        first_conf = list(conformers[conformer_type].keys())[0]
        mol = Chem.Mol(conformers[conformer_type][first_conf]['molecule'])
        
        # Sanity check: coord array must match atom count
        if len(coords) != mol.GetNumAtoms():
            print(f"  WARNING {mol_name}: absolute_coordinates has {len(coords)} entries "
                  f"but molecule has {mol.GetNumAtoms()} atoms — index mismatch!")

        conf = mol.GetConformer()
        for i, coord in enumerate(coords):
            if i >= mol.GetNumAtoms():
                break
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))

        # Diagnostic: print ring atom positions and check ring regularity
        ring_indices = []
        for ci in range(1, 6):
            idx = carbon_map.get(f'C{ci}')
            if idx is not None and idx < len(coords):
                ring_indices.append(idx)
        ro_idx = carbon_map.get('ring_oxygen')
        if ro_idx is not None and ro_idx < len(coords):
            ring_indices.append(ro_idx)

        if ring_indices:
            ring_pos = np.array([coords[i] for i in ring_indices])
            centroid = ring_pos.mean(axis=0)
            dists = np.linalg.norm(ring_pos - centroid, axis=1)
            print(f"  {mol_name}: ring centroid Z={centroid[2]:.2f} Å, "
                  f"radii min={dists.min():.2f} max={dists.max():.2f} Å "
                  f"({'OK' if dists.max() - dists.min() < 0.5 else 'DISTORTED'})")

        processed_mols.append({
            'name': mol_name,
            'mol': mol,
            'atom_offset': atom_offset,
            'carbon_map': carbon_map,
            'num_atoms': mol.GetNumAtoms()
        })

        print(f"  {mol_name}: {mol.GetNumAtoms()} atoms")
        atom_offset += mol.GetNumAtoms()
    
    return processed_mols

def combine_molecules(processed_mols):
    """Combine all molecules into a single molecule."""
    combined = processed_mols[0]['mol']
    for mol_data in processed_mols[1:]:
        combined = Chem.CombineMols(combined, mol_data['mol'])
    rw_mol = Chem.RWMol(combined)
    return Chem.RWMol(combined)

# ============================================================================
# GLYCOSIDIC BONDS
# ============================================================================

def collect_glycosidic_bond_info(glycosidic_bonds, linkage_definitions, processed_mols):
    """
    Collect information about atoms to remove and bonds to add.
    
    Parameters
    ----------
    glycosidic_bonds : dict
        Dictionary mapping bond names to bond data: {bond_name: bond_dict}
    linkage_definitions : list
        List of linkage definitions (donor, donor_c, acceptor, acceptor_c, anom, name)
    """
    atoms_to_remove = []
    bond_info = []
    
    for linkage_def in linkage_definitions:
        donor_name, donor_c, acceptor_name, acceptor_c, anomeric, name = linkage_def

        # Coordinate-based linkages (acceptor is a position array) have no molecule to bond
        if not isinstance(acceptor_name, str):
            print(f"  Skipping coordinate-based linkage {name}")
            continue

        # Bond may have been skipped during creation (e.g. missing molecule) — warn and continue
        if name not in glycosidic_bonds:
            print(f"  WARNING: bond '{name}' was not created — skipping")
            continue

        bond_data = glycosidic_bonds[name]

        donor_data = next((m for m in processed_mols if m['name'] == donor_name), None)
        acceptor_data = next((m for m in processed_mols if m['name'] == acceptor_name), None)

        if donor_data is None:
            print(f"  WARNING: donor '{donor_name}' not found in processed_mols — skipping bond {name}")
            continue
        if acceptor_data is None:
            print(f"  WARNING: acceptor '{acceptor_name}' not found in processed_mols — skipping bond {name}")
            continue
        
        donor_c_local = donor_data['carbon_map'][donor_c]
        acceptor_c_local = acceptor_data['carbon_map'][acceptor_c]
        
        donor_c_global = donor_data['atom_offset'] + donor_c_local
        acceptor_c_global = acceptor_data['atom_offset'] + acceptor_c_local

        print(f"\n  Bond: {name}")
        print(f"    Donor: {donor_name} (offset={donor_data['atom_offset']})")
        print(f"    Acceptor: {acceptor_name} (offset={acceptor_data['atom_offset']})")
        print(f"    Donor OH to remove (local): {bond_data['donor_oh_to_remove']}")
        print(f"    Acceptor OH to remove (local): {bond_data['acceptor_oh_to_remove']}")
        
        # Collect atoms to remove
        for local_idx in bond_data['donor_oh_to_remove']:
            atoms_to_remove.append(donor_data['atom_offset'] + local_idx)
        for local_idx in bond_data['acceptor_oh_to_remove']:
            atoms_to_remove.append(acceptor_data['atom_offset'] + local_idx)
        
        bond_info.append({
            'donor_c': donor_c_global,
            'acceptor_c': acceptor_c_global,
            'oxygen_pos': bond_data['oxygen_position'],
            'name': name
        })
    
    return sorted(set(atoms_to_remove), reverse=True), bond_info

def update_molecule_offsets(processed_mols, atoms_removed):
    """Update atom offsets after removing atoms."""
    current_offset = 0
    for i, mol_data in enumerate(processed_mols):
        old_start = sum(m['mol'].GetNumAtoms() for m in processed_mols[:i])
        old_end = old_start + mol_data['mol'].GetNumAtoms()
        removed_from_mol = len([x for x in atoms_removed if old_start <= x < old_end])
        
        mol_data['atom_offset'] = current_offset
        mol_data['num_atoms'] = mol_data['mol'].GetNumAtoms() - removed_from_mol
        current_offset += mol_data['num_atoms']

def add_glycosidic_bonds(combined_rw, glycosidic_bonds, linkage_definitions, processed_mols):
    """
    Add glycosidic bonds between sugar units.
    
    Parameters
    ----------
    glycosidic_bonds : dict or None
        Dictionary mapping bond names to bond data: {bond_name: bond_dict}
    linkage_definitions : list or None
        List of linkage definitions (donor, donor_c, acceptor, acceptor_c, anom, name)
    """
    if glycosidic_bonds is None or linkage_definitions is None:
        print("\nSkipping glycosidic bonds (not provided)")
        return
    
    print(f"\nAdding {len(linkage_definitions)} glycosidic bonds...")
    
    # Collect removal and bond info
    atoms_to_remove, bond_info = collect_glycosidic_bond_info(
        glycosidic_bonds, linkage_definitions, processed_mols
    )
    
    # Remove atoms
    print(f"Removing {len(atoms_to_remove)} atoms: {atoms_to_remove}")
    for atom_idx in atoms_to_remove:
        combined_rw.RemoveAtom(atom_idx)
    
    # Update offsets
    conf = combined_rw.GetConformer()
    update_molecule_offsets(processed_mols, atoms_to_remove)
    print("\nMolecule offsets AFTER removal:")
    for mol in processed_mols:
        print(f"  {mol['name']}: offset={mol['atom_offset']}, num_atoms={mol['num_atoms']}")
    
    # Add bonds with adjusted indices
    for info in bond_info:
        donor_c_adj = adjust_index_after_removals(info['donor_c'], atoms_to_remove)
        acceptor_c_adj = adjust_index_after_removals(info['acceptor_c'], atoms_to_remove)
        
        print(f"  {info['name']}: C{donor_c_adj} ← O → C{acceptor_c_adj}")
        
        oxygen_idx = combined_rw.AddAtom(Chem.Atom(8))
        conf.SetAtomPosition(oxygen_idx, tuple(info['oxygen_pos']))
        
        combined_rw.AddBond(donor_c_adj, oxygen_idx, Chem.BondType.SINGLE)
        combined_rw.AddBond(oxygen_idx, acceptor_c_adj, Chem.BondType.SINGLE)

# ============================================================================
# PHOSPHATE BONDS
# ============================================================================

def add_phosphate_bonds(combined_rw, phosphate_bonds_with_names, processed_mols):
    """
    Add phosphate linkages by reusing existing OH oxygens on the sugar carbons.
    Only P, O3 (=O), O4 (-OH), and H are inserted as new atoms — no deletions.
    """
    if phosphate_bonds_with_names is None:
        print("Skipping phosphate bonds (not provided)")
        return

    print(f"\nAdding {len(phosphate_bonds_with_names)} phosphate bond(s)...")
    conf = combined_rw.GetConformer()

    for mol1_name, mol2_name, phosphate_bond in phosphate_bonds_with_names:
        mol1_data = next((m for m in processed_mols if m['name'] == mol1_name), None)
        mol2_data = next((m for m in processed_mols if m['name'] == mol2_name), None)

        if mol1_data is None or mol2_data is None:
            print(f"  ⚠ Skipping {mol1_name}-{mol2_name}: molecule not found")
            continue

        carbon1_name = phosphate_bond.get('carbon1_name', 'C1')
        carbon2_name = phosphate_bond.get('carbon2_name', 'C1')

        # Use stored 3D positions to find the correct global atom index.
        # carbon_map offsets become stale after add_glycosidic_bonds removes
        # intra-molecule atoms (e.g. KDO loses C1-OH and C4-OH before C6 is
        # looked up), so position-based lookup is the reliable approach.
        c1_pos_stored = phosphate_bond.get('c1_position')
        c2_pos_stored = phosphate_bond.get('c2_position')

        c1_global = find_atom_by_position(conf, c1_pos_stored, tolerance=0.05)
        c2_global = find_atom_by_position(conf, c2_pos_stored, tolerance=0.05)

        if c1_global is None:
            print(f"  ⚠ Cannot locate {mol1_name}.{carbon1_name} by 3D position — skipping")
            continue
        if c2_global is None:
            print(f"  ⚠ Cannot locate {mol2_name}.{carbon2_name} by 3D position — skipping")
            continue

        print(f"  Located {mol1_name}.{carbon1_name} at global #{c1_global}, "
              f"{mol2_name}.{carbon2_name} at global #{c2_global}")

        # Find the existing OH oxygen on each carbon using RDKit topology
        o1_idx = find_oh_group_near_carbon(combined_rw, conf, c1_global)
        o2_idx = find_oh_group_near_carbon(combined_rw, conf, c2_global)

        if o1_idx is None:
            print(f"  ⚠ No OH found on {mol1_name}.{carbon1_name} (global #{c1_global}) — skipping")
            continue
        if o2_idx is None:
            print(f"  ⚠ No OH found on {mol2_name}.{carbon2_name} (global #{c2_global}) — skipping")
            continue

        print(f"  Reusing O#{o1_idx} ({mol1_name}.{carbon1_name}) and O#{o2_idx} ({mol2_name}.{carbon2_name})")

        # Add only P, O3 (=O), O4 (-OH), H
        p_idx  = combined_rw.AddAtom(Chem.Atom(15))
        o3_idx = combined_rw.AddAtom(Chem.Atom(8))
        o4_idx = combined_rw.AddAtom(Chem.Atom(8))
        h_idx  = combined_rw.AddAtom(Chem.Atom(1))

        conf.SetAtomPosition(p_idx,  tuple(phosphate_bond['phosphorus_position']))
        conf.SetAtomPosition(o3_idx, tuple(phosphate_bond['oxygen3_position']))
        conf.SetAtomPosition(o4_idx, tuple(phosphate_bond['oxygen4_position']))

        o4_p_vec = np.array(phosphate_bond['phosphorus_position']) - np.array(phosphate_bond['oxygen4_position'])
        h_pos = np.array(phosphate_bond['oxygen4_position']) - (o4_p_vec / np.linalg.norm(o4_p_vec)) * 0.96
        conf.SetAtomPosition(h_idx, tuple(h_pos))

        # Bond P to the existing oxygens already attached to the carbons
        combined_rw.AddBond(o1_idx, p_idx,  Chem.BondType.SINGLE)
        combined_rw.AddBond(p_idx,  o2_idx, Chem.BondType.SINGLE)
        combined_rw.AddBond(p_idx,  o3_idx, Chem.BondType.DOUBLE)
        combined_rw.AddBond(p_idx,  o4_idx, Chem.BondType.SINGLE)
        combined_rw.AddBond(o4_idx, h_idx,  Chem.BondType.SINGLE)

        print(f"  ✓ {phosphate_bond['linkage']}")

# ============================================================================
# PEPTIDE BONDS (SUGAR-PEPTIDE)
# ============================================================================

def collect_peptide_bond_info(peptide_linkages, processed_mols, combined_rw, conf):
    """Collect atoms to remove and linkage info for peptide bonds."""
    print(f"\n{'='*60}")
    print(f"INSIDE collect_peptide_bond_info()")
    print(f"  combined_rw has {combined_rw.GetNumAtoms()} atoms")
    print(f"  peptide_linkages: {len(peptide_linkages) if peptide_linkages else 0}")
    print(f"{'='*60}")
    
    atoms_to_remove = []
    peptide_bond_info = []
    if peptide_linkages is None or len(peptide_linkages) == 0:
        return atoms_to_remove, peptide_bond_info
    
    print(f"\nPreparing {len(peptide_linkages)} peptide bond(s)...")
    
    for link in peptide_linkages:
        sugar_mol_name = link['sugar_mol']
        sugar_carbon_name = link['sugar_carbon']
        linkage_data = link['linkage']
        attachment = link['attachment']
        peptide_atom_local = attachment['atom_index']
        
        # Find sugar molecule
        sugar_mol_data = next((m for m in processed_mols if m['name'] == sugar_mol_name), None)
        if sugar_mol_data is None:
            continue
        
        # Get sugar carbon global index
        sugar_c_local = sugar_mol_data['carbon_map'][sugar_carbon_name]
        sugar_c_global = sugar_mol_data['atom_offset'] + sugar_c_local
        
        # Calculate where peptide WILL be added (after current atoms)
        peptide_start_idx = combined_rw.GetNumAtoms()  # This is where peptide will go
        peptide_atom_global = peptide_start_idx + peptide_atom_local
        
        print(f"  {link['aa_type']}[{link['residue_index']}] ← {sugar_mol_name}:{sugar_carbon_name}")
        print(f"    Sugar carbon global index: {sugar_c_global}")
        print(f"    Peptide will start at: {peptide_start_idx}")
        print(f"    Peptide atom global index: {peptide_atom_global} (local: {peptide_atom_local})")
        
        # Find OH group in the combined molecule
        sugar_c_pos = conf.GetAtomPosition(sugar_c_global)
        sugar_c_arr = np.array([sugar_c_pos.x, sugar_c_pos.y, sugar_c_pos.z])
        
        oh_found = []
        
        # Search for O atoms bonded to this carbon
        for i in range(combined_rw.GetNumAtoms()):
            atom = combined_rw.GetAtomWithIdx(i)
            if atom.GetSymbol() == 'O':
                pos = conf.GetAtomPosition(i)
                pos_arr = np.array([pos.x, pos.y, pos.z])
                dist = np.linalg.norm(pos_arr - sugar_c_arr)
                
                if 1.3 < dist < 1.6:  # C-O bond distance
                    neighbors = list(atom.GetNeighbors())
                    neighbor_symbols = [n.GetSymbol() for n in neighbors]
                    
                    print(f"      Found O#{i} at {dist:.3f}Å, neighbors: {neighbor_symbols}")
                    
                    c_count = sum(1 for s in neighbor_symbols if s == 'C')
                    h_count = sum(1 for s in neighbor_symbols if s == 'H')
                    
                    if c_count == 2:
                        print(f"        → This is the ring oxygen (bonded to 2 carbons)")
                    elif c_count == 1 and h_count == 1:
                        oh_found.append(i)
                        for neighbor in neighbors:
                            if neighbor.GetSymbol() == 'H':
                                oh_found.append(neighbor.GetIdx())
                        print(f"        → This is an OH group! Will remove O#{i} and H")
                        break
                    elif c_count == 1 and h_count == 0:
                        oh_found.append(i)
                        print(f"        → This is a bare oxygen (no H)! Will remove O#{i}")
                        break
        
        if oh_found:
            atoms_to_remove.extend(oh_found)
            print(f"    Will remove: {oh_found}")
        else:
            print(f"    ⚠ Warning: No OH found!")
        
        # REMOVE ALL THE DUPLICATE CODE BELOW AND JUST USE:
        peptide_bond_info.append({
            'sugar_mol': sugar_mol_name,
            'sugar_carbon': sugar_c_global,
            'peptide_atom': peptide_atom_global,
            'linkage_data': linkage_data,
            'attachment': link['attachment'],
            'aa_type': link['aa_type'],
            'residue_index': link['residue_index']
        })
    
    return sorted(set(atoms_to_remove), reverse=True), peptide_bond_info

def add_peptide_to_combined(combined_rw, peptide_data, atoms_to_remove):
    """Add peptide atoms to combined molecule and track peptide offset."""
    if peptide_data is None:
        return None
    
    peptide_mol = peptide_data['rdkit_mol']
    
    # Add peptide to combined molecule 
    peptide_start_idx = combined_rw.GetNumAtoms()
    combined_rw = Chem.RWMol(Chem.CombineMols(combined_rw.GetMol(), peptide_mol))
    
    print(f"  Added peptide: {peptide_mol.GetNumAtoms()} atoms starting at index {peptide_start_idx}")
    
    return {
        'start_idx': peptide_start_idx,
        'num_atoms': peptide_mol.GetNumAtoms(),
        'mol': combined_rw
    }

def add_peptide_bonds_to_structure(combined_rw, peptide_bond_info, peptide_offset, atoms_removed, processed_mols):
    """
    Add glycosidic linkages between sugar and peptide with trans configuration enforcement.
    
    Approach:
    1. Create all bonds WITHOUT trans enforcement
    2. After all bonds are created, do ONE sanitization  
    3. Then enforce trans geometry on all bonds at once
    """
    if not peptide_bond_info:
        return
    
    print(f"\nAdding {len(peptide_bond_info)} peptide bond(s) with trans configuration...")
    conf = combined_rw.GetConformer()
    
    # STEP 1: Collect all atoms to remove (OH from sugars + H from peptides)
    all_atoms_to_remove = []
    adjusted_bond_info = []
    
    for info in peptide_bond_info:
        linkage_data = info['linkage_data']
        
        # Calculate indices BEFORE any removal
        sugar_c_global = info['sugar_carbon']
        sugar_c_adj = adjust_index_after_removals(sugar_c_global, atoms_removed)
        
        peptide_atom_original = info['peptide_atom']  # Already has the global index
        peptide_atom_adj = adjust_index_after_removals(peptide_atom_original, atoms_removed)
        
        print(f"\n  Processing {info['aa_type']}[{info['residue_index']}] ← {info['sugar_mol']}")
        print(f"    Sugar C: {sugar_c_adj}, Peptide atom: {peptide_atom_adj}")
        
        # Collect H from peptide nitrogen for N-glycosidic bonds
        if linkage_data['glycosylation_type'] in ['N-glycosidic-direct', 'N-glycosidic']:
            peptide_n_atom = combined_rw.GetAtomWithIdx(peptide_atom_adj)
            
            if peptide_n_atom.GetSymbol() != 'N':
                print(f"    ⚠ Warning: Expected N at index {peptide_atom_adj}, got {peptide_n_atom.GetSymbol()}")
            else:
                h_found = False
                for neighbor in peptide_n_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        all_atoms_to_remove.append(neighbor.GetIdx())
                        print(f"      Peptide H to remove: H#{neighbor.GetIdx()}")
                        h_found = True
                        break
                
                if not h_found:
                    print(f"    ⚠ Warning: No H found on peptide N#{peptide_atom_adj}")
        
        # Store ORIGINAL indices for adjustment later
        adjusted_bond_info.append({
            'sugar_c_original': sugar_c_adj,
            'peptide_atom_original': peptide_atom_adj,
            'linkage_data': linkage_data,
            'aa_type': info['aa_type'],
            'residue_index': info['residue_index'],
            'sugar_mol': info['sugar_mol']
        })
    
    # STEP 2: Remove all atoms at once (in reverse order)
    all_atoms_to_remove = sorted(set(all_atoms_to_remove), reverse=True)
    
    if all_atoms_to_remove:
        print(f"\nRemoving {len(all_atoms_to_remove)} atoms (peptide H only): {all_atoms_to_remove}")
        for atom_idx in all_atoms_to_remove:
            combined_rw.RemoveAtom(atom_idx)
    
    # STEP 3: Create all bonds WITHOUT trans enforcement
    print(f"\nCreating peptide bonds (without trans enforcement)...")
    
    bonds_created = []  # Track bonds for later trans enforcement
    
    for bond_info in adjusted_bond_info:
        linkage_data = bond_info['linkage_data']
        glyc_type = linkage_data['glycosylation_type']
        
        # Adjust indices after H removal
        sugar_c_final = adjust_index_after_removals(bond_info['sugar_c_original'], all_atoms_to_remove)
        peptide_atom_final = adjust_index_after_removals(bond_info['peptide_atom_original'], all_atoms_to_remove)
        
        print(f"  Sugar C: {sugar_c_final}, Peptide atom: {peptide_atom_final}")
        
        # Verify atoms
        sugar_atom = combined_rw.GetAtomWithIdx(sugar_c_final)
        peptide_atom = combined_rw.GetAtomWithIdx(peptide_atom_final)
        print(f"    Bonding: {sugar_atom.GetSymbol()}#{sugar_c_final} to {peptide_atom.GetSymbol()}#{peptide_atom_final}")
        
        if glyc_type == 'N-glycosidic-direct':
            # Direct C-N bond
            combined_rw.AddBond(sugar_c_final, peptide_atom_final, Chem.BondType.SINGLE)
            print(f"  ✓ Direct N-glycosidic: C{sugar_c_final}-N{peptide_atom_final}")
            
            # Track this bond for trans enforcement
            bonds_created.append({
                'type': 'direct_CN',
                'sugar_c': sugar_c_final,
                'peptide_n': peptide_atom_final,
                'aa_type': bond_info['aa_type']
            })
        
        elif glyc_type == 'O-glycosidic':
            # Sugar-C → O → Peptide-O
            o_idx = combined_rw.AddAtom(Chem.Atom(8))
            conf.SetAtomPosition(o_idx, tuple(linkage_data['o_glycosidic_position']))
            
            combined_rw.AddBond(sugar_c_final, o_idx, Chem.BondType.SINGLE)
            combined_rw.AddBond(o_idx, peptide_atom_final, Chem.BondType.SINGLE)
            
            print(f"  ✓ O-glycosidic: C{sugar_c_final}-O{o_idx}-O{peptide_atom_final}")
            
            # Track this bond for trans enforcement
            bonds_created.append({
                'type': 'CO_linkage',
                'sugar_c': sugar_c_final,
                'bridge_o': o_idx,
                'peptide_o': peptide_atom_final,
                'aa_type': bond_info['aa_type']
            })
    
    # STEP 4: Single sanitization after all bonds are created
    print(f"\nPerforming single sanitization...")
    try:
        Chem.SanitizeMol(combined_rw)
        print("  ✓ Sanitization successful")
    except Exception as e:
        print(f"  ⚠ Sanitization warning: {e}")
        # Try partial sanitization
        try:
            Chem.SanitizeMol(combined_rw, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
            print("  ✓ Partial sanitization successful")
        except Exception as e2:
            print(f"  ⚠ Partial sanitization also failed: {e2}")
    
    # STEP 5: Enforce trans geometry on all bonds at once
    print(f"\nEnforcing trans configuration on {len(bonds_created)} bonds...")
    enforced_atoms=enforce_trans_configuration(combined_rw, bonds_created, conf)
    
    return enforced_atoms

def enforce_trans_configuration(mol_rw, bonds_created, conf):
    """
    Enforce trans configuration on glycosidic bonds.
    
    For N-glycosidic bonds: Set dihedral around C-N bond to ~180°
    For O-glycosidic bonds: Set dihedrals around C-O and O-peptide bonds to ~180°
    """
    Chem.GetSymmSSSR(mol_rw)
    for bond_info in bonds_created:
        bond_type = bond_info['type']
        aa_type = bond_info['aa_type']
        
        print(f"\n  Enforcing trans on {aa_type} ({bond_type})...")
        
        if bond_type == 'direct_CN':
            # For direct C-N glycosidic bond
            sugar_c = bond_info['sugar_c']
            peptide_n = bond_info['peptide_n']
            
            success = set_trans_dihedral_CN_bond(mol_rw, conf, sugar_c, peptide_n)
            if success:
                print(f"    ✓ Trans configuration set for C{sugar_c}-N{peptide_n}")
            else:
                print(f"    ⚠ Could not set trans configuration for C{sugar_c}-N{peptide_n}")
                
        elif bond_type == 'CO_linkage':
            # For C-O-peptide linkage
            sugar_c = bond_info['sugar_c']
            bridge_o = bond_info['bridge_o']
            peptide_o = bond_info['peptide_o']
            
            success1 = set_trans_dihedral_CO_bond(mol_rw, conf, sugar_c, bridge_o)
            success2 = set_trans_dihedral_CO_bond(mol_rw, conf, bridge_o, peptide_o)
            
            if success1 and success2:
                print(f"    ✓ Trans configuration set for C{sugar_c}-O{bridge_o}-O{peptide_o}")
            else:
                print(f"    ⚠ Partial success setting trans configuration")

    enforced_atom_sets = []
    
    for bond_info in bonds_created:
        if bond_info['type'] == 'direct_CN':
            sugar_c = bond_info['sugar_c']
            peptide_n = bond_info['peptide_n']
            
            # Get the 4 atoms used for dihedral
            c_atom = mol_rw.GetAtomWithIdx(sugar_c)
            n_atom = mol_rw.GetAtomWithIdx(peptide_n)
            
            c_neighbors = [n.GetIdx() for n in c_atom.GetNeighbors() if n.GetIdx() != peptide_n]
            n_neighbors = [n.GetIdx() for n in n_atom.GetNeighbors() if n.GetIdx() != sugar_c]
            
            if c_neighbors and n_neighbors:
                dihedral_atoms = [c_neighbors[0], sugar_c, peptide_n, n_neighbors[0]]
                
                # Do the trans enforcement
                rdMolTransforms.SetDihedralRad(conf, *dihedral_atoms, np.pi)
                
                # Store the enforced atoms info
                enforced_atom_sets.append({
                    'bond_type': 'direct_CN',
                    'aa_type': bond_info.get('aa_type', 'Unknown'),
                    'dihedral_atoms': dihedral_atoms,
                    'sugar_carbon': sugar_c,
                    'peptide_nitrogen': peptide_n,
                    'target_angle': 180.0,
                    'atom_labels': [f"atom_{i}" for i in dihedral_atoms]
                })
                
                print(f"  ✓ Trans enforced on dihedral: {dihedral_atoms}")
    
    return enforced_atom_sets

def set_trans_dihedral_CN_bond(mol_rw, conf, carbon_idx, nitrogen_idx):
    """
    Set trans configuration around a C-N glycosidic bond.
    Target dihedral angle: ~180° (trans)
    """
    try:
        # Find atoms for dihedral: need 4 atoms in sequence
        # Pattern: [neighbor of C] - C - N - [neighbor of N]
        
        carbon_atom = mol_rw.GetAtomWithIdx(carbon_idx)
        nitrogen_atom = mol_rw.GetAtomWithIdx(nitrogen_idx)
        
        # Get neighbors
        c_neighbors = [n.GetIdx() for n in carbon_atom.GetNeighbors() if n.GetIdx() != nitrogen_idx]
        n_neighbors = [n.GetIdx() for n in nitrogen_atom.GetNeighbors() if n.GetIdx() != carbon_idx]
        
        if len(c_neighbors) == 0 or len(n_neighbors) == 0:
            return False
        
        # Choose first available neighbors for dihedral
        atom1 = c_neighbors[0]  # Neighbor of carbon
        atom2 = carbon_idx      # Carbon
        atom3 = nitrogen_idx    # Nitrogen  
        atom4 = n_neighbors[0]  # Neighbor of nitrogen
        
        # Set dihedral to 180° (trans)
        target_angle = np.pi  # 180° in radians
        rdMolTransforms.SetDihedralRad(conf, atom1, atom2, atom3, atom4, target_angle)
        
        # Verify the dihedral was set
        actual_angle = rdMolTransforms.GetDihedralRad(conf, atom1, atom2, atom3, atom4)
        print(f"      Dihedral {atom1}-{atom2}-{atom3}-{atom4}: {np.degrees(actual_angle):.1f}°")
        
        return True
        
    except Exception as e:
        print(f"      Error setting C-N dihedral: {e}")
        return False

def set_trans_dihedral_CO_bond(mol_rw, conf, atom1_idx, atom2_idx):
    """
    Set trans configuration around a C-O bond.
    """
    try:
        atom1 = mol_rw.GetAtomWithIdx(atom1_idx)
        atom2 = mol_rw.GetAtomWithIdx(atom2_idx)
        
        # Get neighbors for dihedral
        atom1_neighbors = [n.GetIdx() for n in atom1.GetNeighbors() if n.GetIdx() != atom2_idx]
        atom2_neighbors = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != atom1_idx]
        
        if len(atom1_neighbors) == 0 or len(atom2_neighbors) == 0:
            return False
        
        # Create dihedral
        dihedral_atom1 = atom1_neighbors[0]
        dihedral_atom2 = atom1_idx
        dihedral_atom3 = atom2_idx
        dihedral_atom4 = atom2_neighbors[0]
        
        # Set to trans (180°)
        target_angle = np.pi
        rdMolTransforms.SetDihedralRad(conf, dihedral_atom1, dihedral_atom2, dihedral_atom3, dihedral_atom4, target_angle)
        
        actual_angle = rdMolTransforms.GetDihedralRad(conf, dihedral_atom1, dihedral_atom2, dihedral_atom3, dihedral_atom4)
        print(f"      Dihedral {dihedral_atom1}-{dihedral_atom2}-{dihedral_atom3}-{dihedral_atom4}: {np.degrees(actual_angle):.1f}°")
        
        return True
        
    except Exception as e:
        print(f"      Error setting C-O dihedral: {e}")
        return False

def adjust_index_after_removals(original_idx, removed_indices):
    """Helper function to adjust atom indices after atom removal."""
    if not removed_indices:
        return original_idx
    
    # Count how many removed indices are smaller than original_idx
    adjustment = sum(1 for removed_idx in removed_indices if removed_idx < original_idx)
    return original_idx - adjustment

# ============================================================================
# FUNCTIONAL GROUPS (PEtN and Lipids)
# ============================================================================

def collect_petn_removal_info(combined_rw, petn_linkages, processed_mols):
    """Collect atoms to remove for PEtN linkages."""
    conf = combined_rw.GetConformer()
    atoms_to_remove = []
    petn_info = []
    
    if petn_linkages is None:
        return atoms_to_remove, petn_info
    
    print(f"\nPreparing {len(petn_linkages)} PEtN group(s)...")
    
    for mol_name, petn_data in petn_linkages:
        mol_data = next((m for m in processed_mols if m['name'] == mol_name), None)
        if mol_data is None:
            continue
        
        c_sugar_global = find_atom_by_position(conf, petn_data['sugar_carbon'])
        if c_sugar_global is None:
            continue
        
        oh_idx = find_oh_group_near_carbon(combined_rw, conf, c_sugar_global)
        if oh_idx is not None:
            atoms_to_remove.append(oh_idx)
            print(f"  PEtN on {mol_name}: will remove O#{oh_idx}")
        
        petn_info.append({
            'sugar_carbon': c_sugar_global,
            'data': petn_data
        })
    
    return atoms_to_remove, petn_info

def collect_lipid_removal_info(combined_rw, lipid_linkages, processed_mols):
    """Collect atoms to remove for lipid linkages."""
    conf = combined_rw.GetConformer()
    atoms_to_remove = []
    lipid_info = []
    
    if lipid_linkages is None:
        return atoms_to_remove, lipid_info
    
    print(f"\nPreparing {len(lipid_linkages)} lipid tail(s)...")
    
    for lipid_result in lipid_linkages:
        mol_name = lipid_result['mol_name']
        carbon_name = lipid_result['carbon_name']
        linkage_type = lipid_result['linkage']
        connection_anchor = lipid_result.get('connection_anchor', 'sugar')

        is_branched = (connection_anchor == 'lipid') 
        
        # For lipid-anchored lipids, we don't remove any atoms
        # We're branching off an existing carbon, not replacing a functional group
        if is_branched: 
            # if connection_anchor == 'lipid' or carbon_name is None:
            print(f"  Lipid (branched, {linkage_type}): no removal needed")
            lipid_info.append({
                'sugar_carbon': None,  # Will need to find this later
                'linkage_type': linkage_type,
                'data': lipid_result,
                'is_branched': True
            })
            continue
        
        mol_data = next((m for m in processed_mols if m['name'] == mol_name), None)
        if mol_data is None:
            continue
        
        sugar_idx = mol_data['atom_offset'] + mol_data['carbon_map'][carbon_name]
        
        # Handle different linkage types for sugar-anchored lipids
        if linkage_type in ['ester', 'ether']:
            oh_idx = find_oh_group_near_carbon(combined_rw, conf, sugar_idx)
            if oh_idx is not None:
                atoms_to_remove.append(oh_idx)
                print(f"  Lipid ({linkage_type}) on {mol_name}.{carbon_name}: will remove O#{oh_idx}")
            else:
                print(f"  Lipid ({linkage_type}) on {mol_name}.{carbon_name}: no OH found - will add without removal")
        
        elif linkage_type == 'amide':
            # Try to find NH2 group first (for cases where we're replacing an amine)
            nh2_indices = find_nh2_group_near_carbon(combined_rw, conf, sugar_idx)
            
            if nh2_indices:
                # Found NH2 - remove it
                atoms_to_remove.extend(nh2_indices)
                print(f"  Lipid (amide) on {mol_name}.{carbon_name}: will remove NH2 group N#{nh2_indices[0]}")
            else:
                # No NH2 found - try to find and remove OH group instead (like in ceramide sphingosine)
                oh_idx = find_oh_group_near_carbon(combined_rw, conf, sugar_idx)
                if oh_idx is not None:
                    atoms_to_remove.append(oh_idx)
                    print(f"  Lipid (amide) on {mol_name}.{carbon_name}: will remove OH group O#{oh_idx}")
                else:
                    print(f"  Lipid (amide) on {mol_name}.{carbon_name}: no NH2 or OH found - will add without removal")
        
        lipid_info.append({
            'sugar_carbon': sugar_idx,
            'linkage_type': linkage_type,
            'data': lipid_result,
            'is_branched': False
        })
    
    return atoms_to_remove, lipid_info

def add_petn_group(combined_rw, sugar_carbon_idx, petn_data):
    """Add a PEtN (phosphoethanolamine) group to a sugar carbon."""
    conf = combined_rw.GetConformer()
    
    o1_idx = combined_rw.AddAtom(Chem.Atom(8))
    p_idx = combined_rw.AddAtom(Chem.Atom(15))
    o2_idx = combined_rw.AddAtom(Chem.Atom(8))
    o3_idx = combined_rw.AddAtom(Chem.Atom(8))
    o4_idx = combined_rw.AddAtom(Chem.Atom(8))
    c1_idx = combined_rw.AddAtom(Chem.Atom(6))
    c2_idx = combined_rw.AddAtom(Chem.Atom(6))
    n_idx = combined_rw.AddAtom(Chem.Atom(7))
    
    conf.SetAtomPosition(o1_idx, tuple(petn_data['o1_position']))
    conf.SetAtomPosition(p_idx, tuple(petn_data['p_position']))
    conf.SetAtomPosition(o2_idx, tuple(petn_data['o2_position']))
    conf.SetAtomPosition(o3_idx, tuple(petn_data['o3_position']))
    conf.SetAtomPosition(o4_idx, tuple(petn_data['o4_position']))
    conf.SetAtomPosition(c1_idx, tuple(petn_data['c1_position']))
    conf.SetAtomPosition(c2_idx, tuple(petn_data['c2_position']))
    conf.SetAtomPosition(n_idx, tuple(petn_data['n_position']))
    
    combined_rw.GetAtomWithIdx(n_idx).SetFormalCharge(0)
    
    combined_rw.AddBond(sugar_carbon_idx, o1_idx, Chem.BondType.SINGLE)
    combined_rw.AddBond(o1_idx, p_idx, Chem.BondType.SINGLE)
    combined_rw.AddBond(p_idx, o2_idx, Chem.BondType.DOUBLE)
    combined_rw.AddBond(p_idx, o3_idx, Chem.BondType.SINGLE)
    combined_rw.AddBond(p_idx, o4_idx, Chem.BondType.SINGLE)
    combined_rw.AddBond(o4_idx, c1_idx, Chem.BondType.SINGLE)
    combined_rw.AddBond(c1_idx, c2_idx, Chem.BondType.SINGLE)
    combined_rw.AddBond(c2_idx, n_idx, Chem.BondType.SINGLE)

def add_lipid_tail(combined_rw, sugar_carbon_idx, linkage_type, lipid_data, is_branched=False):
    """Add a lipid tail to a sugar carbon or existing lipid."""
    conf = combined_rw.GetConformer()
    prev_idx = None
    carbon_chain_indices = [] 

    # DEBUG: Print what we received
    print(f"\n  DEBUG add_lipid_tail called:")
    print(f"    linkage_type: {linkage_type}")
    print(f"    is_branched: {is_branched}")
    print(f"    lipid_data keys: {lipid_data.keys()}")
    print(f"    has n_link_position: {'n_link_position' in lipid_data}")
    print(f"    has c_carbonyl_position: {'c_carbonyl_position' in lipid_data}")
    print(f"    has o_carbonyl_position: {'o_carbonyl_position' in lipid_data}")
    
    # For branched lipids, find the nearest carbon to anchor position
    if is_branched:
        # Check if we have a specific carbon index stored
        if 'carbon_index' in lipid_data and lipid_data['carbon_index'] is not None:
            # Find the first lipid chain and get the specific carbon
            # We need to find where this carbon is in the combined molecule
            anchor_pos = np.array(lipid_data['sugar_carbon'])
            sugar_carbon_idx = find_nearest_carbon_atom(combined_rw, conf, anchor_pos, max_distance=0.2)
            print(f"  Branching from stored C{lipid_data['carbon_index']+1} at atom #{sugar_carbon_idx}")
        else:
            anchor_pos = np.array(lipid_data['sugar_carbon'])
            sugar_carbon_idx = find_nearest_carbon_atom(combined_rw, conf, anchor_pos)
            print(f"  Branching from carbon atom #{sugar_carbon_idx}")
    
    # Process all linkage types, whether branched or not
    if linkage_type == 'ester':
        # C–O–C(=O)–(CH2)n–CH3
        if 'o_ester_position' in lipid_data:
            o_ester_idx = combined_rw.AddAtom(Chem.Atom('O'))
            c_carbonyl_idx = combined_rw.AddAtom(Chem.Atom('C'))
            o_carbonyl_idx = combined_rw.AddAtom(Chem.Atom('O'))
            
            conf.SetAtomPosition(o_ester_idx, tuple(lipid_data['o_ester_position']))
            conf.SetAtomPosition(c_carbonyl_idx, tuple(lipid_data['c_carbonyl_position']))
            conf.SetAtomPosition(o_carbonyl_idx, tuple(lipid_data['o_carbonyl_position']))
            
            combined_rw.AddBond(sugar_carbon_idx, o_ester_idx, Chem.BondType.SINGLE)
            combined_rw.AddBond(o_ester_idx, c_carbonyl_idx, Chem.BondType.SINGLE)
            combined_rw.AddBond(c_carbonyl_idx, o_carbonyl_idx, Chem.BondType.DOUBLE)
            
            print(f"    Created ester bonds: C#{sugar_carbon_idx}-O#{o_ester_idx}-C#{c_carbonyl_idx}=O#{o_carbonyl_idx}")
            prev_idx = c_carbonyl_idx
    
    elif linkage_type == 'amide':
        # N–C(=O)–(CH2)n–CH3
        if 'n_link_position' in lipid_data:
            n_link_idx = combined_rw.AddAtom(Chem.Atom('N'))
            c_carbonyl_idx = combined_rw.AddAtom(Chem.Atom('C'))
            o_carbonyl_idx = combined_rw.AddAtom(Chem.Atom('O'))
            
            conf.SetAtomPosition(n_link_idx, tuple(lipid_data['n_link_position']))
            conf.SetAtomPosition(c_carbonyl_idx, tuple(lipid_data['c_carbonyl_position']))
            conf.SetAtomPosition(o_carbonyl_idx, tuple(lipid_data['o_carbonyl_position']))
            
            combined_rw.AddBond(sugar_carbon_idx, n_link_idx, Chem.BondType.SINGLE)
            combined_rw.AddBond(n_link_idx, c_carbonyl_idx, Chem.BondType.SINGLE)
            combined_rw.AddBond(c_carbonyl_idx, o_carbonyl_idx, Chem.BondType.DOUBLE)
            
            if is_branched:
                print(f"    Created BRANCHED amide bonds: C#{sugar_carbon_idx}-N#{n_link_idx}-C#{c_carbonyl_idx}=O#{o_carbonyl_idx}")
            else:
                print(f"    Created amide bonds: C#{sugar_carbon_idx}-N#{n_link_idx}-C#{c_carbonyl_idx}=O#{o_carbonyl_idx}")
            prev_idx = c_carbonyl_idx
    
    elif linkage_type == 'ether':
        # C–O–(CH2)n–CH3
        if 'o_ether_position' in lipid_data:
            o_ether_idx = combined_rw.AddAtom(Chem.Atom('O'))
            conf.SetAtomPosition(o_ether_idx, tuple(lipid_data['o_ether_position']))
            combined_rw.AddBond(sugar_carbon_idx, o_ether_idx, Chem.BondType.SINGLE)
            print(f"    Created ether bond: C#{sugar_carbon_idx}-O#{o_ether_idx}")
            prev_idx = o_ether_idx
    
    elif is_branched and 'c_carbonyl_position' in lipid_data:
        # Fallback for branched with only carbonyl
        c_carbonyl_idx = combined_rw.AddAtom(Chem.Atom('C'))
        o_carbonyl_idx = combined_rw.AddAtom(Chem.Atom('O'))
        
        conf.SetAtomPosition(c_carbonyl_idx, tuple(lipid_data['c_carbonyl_position']))
        conf.SetAtomPosition(o_carbonyl_idx, tuple(lipid_data['o_carbonyl_position']))
        
        combined_rw.AddBond(sugar_carbon_idx, c_carbonyl_idx, Chem.BondType.SINGLE)
        combined_rw.AddBond(c_carbonyl_idx, o_carbonyl_idx, Chem.BondType.DOUBLE)
        
        print(f"    Created simple carbonyl: C#{sugar_carbon_idx}-C#{c_carbonyl_idx}=O#{o_carbonyl_idx}")
        prev_idx = c_carbonyl_idx
    
    # Add carbon chain
    if prev_idx is not None and 'chain_carbons' in lipid_data:
        print(f"    DEBUG: Starting carbon chain from prev_idx={prev_idx}")
        for i, c_pos in enumerate(lipid_data['chain_carbons']):
            c_idx = combined_rw.AddAtom(Chem.Atom('C'))
            conf.SetAtomPosition(c_idx, tuple(c_pos))
            print(f"      Adding carbon #{c_idx}, bonding to #{prev_idx}")
            combined_rw.AddBond(prev_idx, c_idx, Chem.BondType.SINGLE)
            carbon_chain_indices.append(c_idx)
            prev_idx = c_idx
        print(f"    Added {len(lipid_data['chain_carbons'])} carbons to chain")
    else:
        print(f"    Warning: Could not add carbon chain")
    
    return carbon_chain_indices

def add_functional_groups(combined_rw, petn_linkages, lipid_linkages, processed_mols):
    """Add PEtN groups and lipid tails."""
    conf = combined_rw.GetConformer()
    
    # Collect all removals
    petn_removals, petn_info = collect_petn_removal_info(combined_rw, petn_linkages, processed_mols)
    lipid_removals, lipid_info = collect_lipid_removal_info(combined_rw, lipid_linkages, processed_mols)
    
    all_removals = sorted(set(petn_removals + lipid_removals), reverse=True)
    
    # Remove atoms
    if all_removals:
        print(f"\nRemoving {len(all_removals)} atoms for functional groups: {all_removals}")
        for idx in all_removals:
            combined_rw.RemoveAtom(idx)
        
        # Adjust indices for PEtN groups
        for info in petn_info:
            if info['sugar_carbon'] is not None: 
                info['sugar_carbon'] = adjust_index_after_removals(info['sugar_carbon'], all_removals)
        
        # Adjust indices for sugar-anchored lipids only
        for info in lipid_info:
            if not info.get('is_branched', False) and info['sugar_carbon'] is not None:
                info['sugar_carbon'] = adjust_index_after_removals(info['sugar_carbon'], all_removals)
        
        # Note: Branched lipids use position-based search, so no index adjustment needed
    
    # Add groups
    print(f"\nAdding {len(petn_info) + len(lipid_info)} functional groups...")
    
    # Add PEtN groups first
    for info in petn_info:
        add_petn_group(combined_rw, info['sugar_carbon'], info['data'])
        print(f"  ✓ {info['data']['linkage']}")
    
    # Pass 1: Adding sugar-anchored lipid tails
    print("\nPass 1: Adding sugar-anchored lipid tails...")
    lipid_carbon_chains = {}  # Track chains

    for info in lipid_info:
        is_branched = info.get('is_branched', False)
        
        if not is_branched:
            sugar_carbon_idx = info['sugar_carbon']
            carbon_indices = add_lipid_tail(combined_rw, sugar_carbon_idx, info['linkage_type'], info['data'], is_branched=False)
            
            # Store the carbon chain indices
            lipid_name = f"{info['data']['mol_name']}:{info['data'].get('carbon_name', 'lipid')}"
            lipid_carbon_chains[lipid_name] = carbon_indices
            
            # DEBUG: Print the chain
            print(f"  ✓ Added lipid tail to {info['data']['mol_name']}:{info['data']['carbon_name']}")
            print(f"    Carbon chain indices: {carbon_indices}")
            if len(carbon_indices) >= 2:
                print(f"    C1 = atom #{carbon_indices[0]}, C2 = atom #{carbon_indices[1]}")

    # Pass 2: Adding branched lipid tails
    print("\nPass 2: Adding branched (lipid-anchored) lipid tails...")
    for info in lipid_info:
        is_branched = info.get('is_branched', False)
        
        if is_branched:
            # Use the SECOND carbon (index 1) of the first lipid chain
            if not lipid_carbon_chains:
                print(f"  ERROR: No lipid chains built in Pass 1!")
                continue
                
            first_lipid_key = list(lipid_carbon_chains.keys())[0]
            first_lipid_carbons = lipid_carbon_chains[first_lipid_key]
            
            if len(first_lipid_carbons) < 2:
                print(f"  ERROR: First lipid has fewer than 2 carbons!")
                continue
            
            c2_idx = first_lipid_carbons[1]  # C2 is second carbon (index 1)
            
            # DEBUG: Verify C2
            conf = combined_rw.GetConformer()
            c2_atom = combined_rw.GetAtomWithIdx(c2_idx)
            c2_pos = conf.GetAtomPosition(c2_idx)
            print(f"  Branching from C2 (atom #{c2_idx}) of {first_lipid_key}")
            print(f"    Atom symbol: {c2_atom.GetSymbol()}")
            print(f"    Position: [{c2_pos.x:.2f}, {c2_pos.y:.2f}, {c2_pos.z:.2f}]")
            print(f"    Current bonds: {[n.GetIdx() for n in c2_atom.GetNeighbors()]}")
            
            add_lipid_tail(combined_rw, c2_idx, info['linkage_type'], info['data'], is_branched=True)
            print(f"  ✓ Added branched lipid tail")
            
            # DEBUG: Check C2 bonds after adding branch
            print(f"    Bonds after branching: {[n.GetIdx() for n in c2_atom.GetNeighbors()]}")
           
def find_nearest_carbon_atom(mol, conf, target_pos, max_distance=0.5):
    """Find nearest carbon atom to target position."""
    min_dist = float('inf')
    nearest_idx = None
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            idx = atom.GetIdx()
            pos = np.array(conf.GetAtomPosition(idx))
            dist = np.linalg.norm(pos - target_pos)
            if dist < min_dist and dist < max_distance:  # ← Add max_distance check
                min_dist = dist
                nearest_idx = idx
    
    if nearest_idx is None:
        print(f"  WARNING: No carbon found within {max_distance}Å of target {target_pos}")
        print(f"  Searching with larger tolerance...")
        # Retry with larger tolerance
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                idx = atom.GetIdx()
                pos = np.array(conf.GetAtomPosition(idx))
                dist = np.linalg.norm(pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
        print(f"  Found carbon at distance {min_dist:.3f}Å")
    
    return nearest_idx

# ============================================================================
# MAIN EXPORT FUNCTION
# ============================================================================

def export_complete_structure_with_petn_and_lipids(
    chain_dict,
    conformers,
    mol_name_to_conformer,
    filename=None,
    glycosidic_bonds=None,
    linkage_definitions=None,
    phosphate_bonds_with_names=None,
    unbonded_monomers=None,
    petn_linkages=None,
    lipid_linkages=None
):
    """
    Export complete LPS structure with:
    - Glycosidic bonds
    - Phosphate bonds
    - PEtN (phosphoethanolamine) groups
    - Lipid tails (amide or ester linkages)
    """
    # Validate dependencies
    if (glycosidic_bonds is None) != (linkage_definitions is None):
        raise ValueError("glycosidic_bonds and linkage_definitions must both be provided or both be None")
    
    if glycosidic_bonds is not None and linkage_definitions is not None:
        for linkage_def in linkage_definitions:
            bond_name = linkage_def[5]
            if bond_name not in glycosidic_bonds:
                print(f"  WARNING: bond '{bond_name}' from linkage_definitions was not created — will be skipped")
    
    if phosphate_bonds_with_names is not None and unbonded_monomers is None:
        raise ValueError("unbonded_monomers required when phosphate_bonds_with_names is provided")
    
    # Step 1: Collect all molecules
    processed_mols = collect_molecules(chain_dict, unbonded_monomers, mol_name_to_conformer, conformers)

    # Freeze ring positions BEFORE any bonding touches the conformer
    ring_snapshot = snapshot_ring_positions(chain_dict)

    # Step 2: Combine molecules
    combined_rw = combine_molecules(processed_mols)

    # Step 3: Add glycosidic bonds
    add_glycosidic_bonds(combined_rw, glycosidic_bonds, linkage_definitions, processed_mols)

    # Step 4: Add phosphate bonds
    add_phosphate_bonds(combined_rw, phosphate_bonds_with_names, processed_mols)

    # Step 5-6: Add PEtN and lipid tails
    add_functional_groups(combined_rw, petn_linkages, lipid_linkages, processed_mols)
    
    # ============================================================
    # Step 7: Finalize and save
    # ============================================================
    
    # BEFORE converting to Mol, explicitly set bond orders
    print("\nSetting all bonds to SINGLE (except specific double bonds)...")
    for bond in combined_rw.GetBonds():
        # Keep P=O and C=O double bonds
        atom1 = combined_rw.GetAtomWithIdx(bond.GetBeginAtomIdx())
        atom2 = combined_rw.GetAtomWithIdx(bond.GetEndAtomIdx())
        
        # Only keep double bonds if they're C=O or P=O
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if not ((atom1.GetSymbol() in ['C', 'P'] and atom2.GetSymbol() == 'O') or
                    (atom2.GetSymbol() in ['C', 'P'] and atom1.GetSymbol() == 'O')):
                bond.SetBondType(Chem.BondType.SINGLE)
        
    # Update property cache with strict=False
    for atom in combined_rw.GetAtoms():
        atom.UpdatePropertyCache(strict=False)

    # Convert to Mol WITHOUT sanitization
    final_mol = combined_rw.GetMol()
    Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)

    # Debug: Check bonds after GetMol
    print("\n" + "="*60)
    print("BOND CHECK immediately after GetMol():")
    print("="*60)
    for idx in [15, 42, 43, 44, 55, 56, 57]:
        if idx < final_mol.GetNumAtoms():
            atom = final_mol.GetAtomWithIdx(idx)
            print(f"Atom #{idx} ({atom.GetSymbol()}): ", end="")
            bonds = [f"{bond.GetOtherAtomIdx(idx)}" for bond in atom.GetBonds()]
            print(f"{len(bonds)} bonds → {bonds}")
    
    # Update property cache again on the final molecule
    for atom in final_mol.GetAtoms():
        atom.UpdatePropertyCache(strict=False)
    
    # Set explicit valence for nitrogen atoms to prevent issues
    for atom in final_mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            atom.SetNoImplicit(True)
    
    restore_ring_positions_from_snapshot(final_mol, ring_snapshot)

    if filename is not None:
        # Built-structure artifacts disabled: only the final *_optimized.sdf is kept.
        # # Write SDF using manual approach to preserve ALL bonds
        # print(f"\nWriting to {filename}.sdf...")
        #
        # # Method 1: Try standard writer first
        # writer = Chem.SDWriter(f"{filename}.sdf")
        # writer.SetKekulize(False)
        #
        # # Write without any modifications
        # try:
        #     writer.write(final_mol)
        #     writer.close()
        #     print(f"  ✓ Standard SDF write successful")
        # except Exception as e:
        #     print(f"  ✗ Standard write failed: {e}")
        #     writer.close()
        #
        # # Method 2: Also write as MOL block (raw format)
        # try:
        #     mol_block = Chem.MolToMolBlock(final_mol, kekulize=False)
        #     with open(f"{filename}_raw.mol", 'w') as f:
        #         f.write(mol_block)
        #     print(f"  ✓ Raw MOL file written to {filename}_raw.mol")
        # except Exception as e:
        #     print(f"  ✗ MOL block write failed: {e}")

        print(f"\nFinal molecule statistics:")
        print(f"  Total atoms: {final_mol.GetNumAtoms()}")
        print(f"  Total bonds: {final_mol.GetNumBonds()}")
    
    return final_mol

def export_complete_glycopeptide(
    chain_dict,
    conformers,
    mol_name_to_conformer,
    peptide_data=None,
    peptide_linkages=None,
    filename=None,
    glycosidic_bonds=None,
    linkage_definitions=None,
    phosphate_bonds_with_names=None,
    unbonded_monomers=None,
    petn_linkages=None,
    lipid_linkages=None
):
    """
    Export complete glycopeptide structure with sugars and peptide.
    """
    
    # Step 1: Collect all molecules (sugars only for now)
    processed_mols = collect_molecules(chain_dict, unbonded_monomers, mol_name_to_conformer, conformers)

    # Freeze ring positions BEFORE any bonding touches the conformer
    ring_snapshot = snapshot_ring_positions(chain_dict)
    # Freeze linker ring positions from the already-positioned peptide
    linker_snapshot = snapshot_peptide_linker_positions(peptide_data)
    ring_snapshot.update(linker_snapshot)

    # Step 2: Combine sugar molecules
    combined_rw = combine_molecules(processed_mols)

    # Step 3: Add glycosidic bonds (sugar-sugar)
    add_glycosidic_bonds(combined_rw, glycosidic_bonds, linkage_definitions, processed_mols)
    
    # Step 4: Add phosphate bonds
    add_phosphate_bonds(combined_rw, phosphate_bonds_with_names, processed_mols)
    
    # Step 5: Add PEtN and lipids
    add_functional_groups(combined_rw, petn_linkages, lipid_linkages, processed_mols)
    
    # ========================================================================
    # NEW: Handle peptide
    # ========================================================================
    
    # Step 6: Collect peptide bond removal info
    peptide_atoms_to_remove, peptide_bond_info = collect_peptide_bond_info(
        peptide_linkages, processed_mols, combined_rw, combined_rw.GetConformer()
    )
    print(f"\nDEBUG after collect_peptide_bond_info():")
    print(f"  peptide_atoms_to_remove: {peptide_atoms_to_remove}")
    print(f"  peptide_bond_info: {peptide_bond_info}")
    
    # Step 7: Remove atoms before adding peptide
    all_atoms_to_remove = sorted(set(peptide_atoms_to_remove), reverse=True)
    if all_atoms_to_remove:
        print(f"\nRemoving {len(all_atoms_to_remove)} atoms for peptide bonds: {all_atoms_to_remove}")
        for idx in all_atoms_to_remove:
            combined_rw.RemoveAtom(idx)

    # Step 8: Add peptide to combined molecule
    peptide_offset = add_peptide_to_combined(combined_rw, peptide_data, all_atoms_to_remove)  # ← Pass removed atoms
    if peptide_offset:
        combined_rw = peptide_offset['mol']
    
    # Step 9: Add peptide bonds (sugar-peptide linkages)
    if peptide_offset:
        enforced_atoms = add_peptide_bonds_to_structure(combined_rw, peptide_bond_info, peptide_offset, all_atoms_to_remove, processed_mols)
    
    # ========================================================================
    # Step 10: Finalize and save
    # ========================================================================
    
    print("\nFinalizing molecule...")
    for bond in combined_rw.GetBonds():
        atom1 = combined_rw.GetAtomWithIdx(bond.GetBeginAtomIdx())
        atom2 = combined_rw.GetAtomWithIdx(bond.GetEndAtomIdx())
        
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if not ((atom1.GetSymbol() in ['C', 'P'] and atom2.GetSymbol() == 'O') or
                    (atom2.GetSymbol() in ['C', 'P'] and atom1.GetSymbol() == 'O')):
                bond.SetBondType(Chem.BondType.SINGLE)
    
    for atom in combined_rw.GetAtoms():
        atom.UpdatePropertyCache(strict=False)
    
    final_mol = combined_rw.GetMol()
    Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
    
    for atom in final_mol.GetAtoms():
        atom.UpdatePropertyCache(strict=False)
        if atom.GetSymbol() == 'N':
            atom.SetNoImplicit(True)
    
    # Peptide Cα alignment runs first (translates residue atoms including linker)
    restore_glycopeptide_positions(final_mol, chain_dict, peptide_data, tolerance=2.0)
    # Sugar + linker rings frozen last — overrides any drift from Cα alignment
    restore_ring_positions_from_snapshot(final_mol, ring_snapshot)
    # Absolute guarantee: linker rings always flat on surface (Z=0)
    enforce_linker_flat(final_mol)
    
    # DEBUG IMMEDIATELY AFTER RESTORATION
    print("\n" + "="*70)
    print("DEBUG: POSITIONS IMMEDIATELY AFTER RESTORATION")
    print("="*70)
    
    conf = final_mol.GetConformer()
    
    # Check sugars
    Chem.GetSymmSSSR(final_mol)
    ring_info = final_mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) == 6:
            oxygen_count = sum(1 for idx in ring if final_mol.GetAtomWithIdx(idx).GetSymbol() == 'O')
            if oxygen_count == 1:
                # FIX: Get positions correctly
                ring_positions = []
                for i in ring:
                    pos = conf.GetAtomPosition(i)
                    ring_positions.append([pos.x, pos.y, pos.z])
                ring_com = np.mean(ring_positions, axis=0)
                print(f"  Ring COM: [{ring_com[0]:.3f}, {ring_com[1]:.3f}, {ring_com[2]:.3f}]")
    
    # Check peptide Cα
    ca_indices = find_peptide_ca_atoms(final_mol)
    for i, ca_idx in enumerate(ca_indices):
        pos = conf.GetAtomPosition(ca_idx)
        print(f"  Cα {i}: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
    
    print("="*70)
    
    # Write file  (disabled: only the final *_optimized.sdf is kept)
    if filename:
        # writer = Chem.SDWriter(f"{filename}.sdf")
        # writer.SetKekulize(False)
        # writer.write(final_mol)
        # writer.close()
        #
        # print(f"\n✓ Saved to {filename}.sdf")
        print(f"  Total atoms: {final_mol.GetNumAtoms()}")
        print(f"  Total bonds: {final_mol.GetNumBonds()}")
    
    return final_mol, enforced_atoms

def snapshot_ring_positions(chain_dict):
    """
    Record exact ring atom positions from chain_dict BEFORE any bonding/export
    operations. Uses carbon_map to identify the ring atoms (C1-C5 + ring_oxygen).

    Returns
    -------
    dict  {mol_name: {atom_idx_in_monomer: [x, y, z]}}
    """
    snapshots = {}
    for mol_name, mol_data in chain_dict.items():
        carbon_map = mol_data.get('carbon_map', {})
        coords = np.array(mol_data['absolute_coordinates'])

        ring_indices = set()
        for ci in range(1, 6):
            idx = carbon_map.get(f'C{ci}')
            if idx is not None and idx < len(coords):
                ring_indices.add(idx)
        ro_idx = carbon_map.get('ring_oxygen')
        if ro_idx is not None and ro_idx < len(coords):
            ring_indices.add(ro_idx)

        snapshots[mol_name] = {idx: coords[idx].tolist() for idx in ring_indices}

    return snapshots


def snapshot_peptide_linker_positions(peptide_data):
    """
    Record exact positions of linker ring atoms (phenyl + oxadiazole) from the
    already-positioned peptide.  Must be called BEFORE any bonding operations
    modify or combine the peptide molecule.

    Returns a dict in the same format as snapshot_ring_positions so it can be
    merged and passed to restore_ring_positions_from_snapshot.

    Returns
    -------
    dict  {'__linker__': {atom_idx: [x, y, z]}}   or {} if no linker found
    """
    from rdkit.Chem import MolFromSmarts as _smarts

    mol = peptide_data.get('rdkit_mol') if peptide_data else None
    if mol is None or mol.GetNumConformers() == 0:
        return {}

    phenyl_match    = mol.GetSubstructMatch(_smarts('c1ccccc1'))
    oxadiazole_match = mol.GetSubstructMatch(_smarts('c1nnco1'))

    # The oxadiazole is the linker's unique fingerprint. Phe/Tyr/Trp aromatic
    # side chains also match c1ccccc1, so without this gate a linker-less
    # peptide snapshots a Phe ring here and later _restore_linker_by_smarts
    # teleports a (different) aromatic ring onto it. Require the oxadiazole.
    if not oxadiazole_match:
        return {}

    linker_indices  = list(phenyl_match) + list(oxadiazole_match)
    if not linker_indices:
        return {}

    conf = mol.GetConformer()
    snapshot = {idx: list(conf.GetAtomPosition(idx)) for idx in linker_indices}
    print(f"  [linker snapshot] captured {len(snapshot)} ring atoms "
          f"(phenyl={len(phenyl_match)}, oxadiazole={len(oxadiazole_match)})")
    return {'__linker__': snapshot}


def _find_linker_ring_atoms(mol):
    """
    Find phenyl (6-membered all-carbon) and oxadiazole (5-membered, 2N+1O+2C)
    ring atoms by ring topology — no reliance on aromaticity perception.

    Safe to call after SANITIZE_NONE or bond-order normalisation that may have
    cleared aromatic flags.

    Returns (phenyl_indices, oxadiazole_indices) as lists of atom indices.
    """
    Chem.GetSymmSSSR(mol)
    ring_info = mol.GetRingInfo()
    phenyl, oxadiazole = [], []

    for ring in ring_info.AtomRings():
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if len(ring) == 6 and all(s == 'C' for s in syms):
            phenyl = list(ring)
        elif (len(ring) == 5 and
              syms.count('N') == 2 and syms.count('O') == 1 and syms.count('C') == 2):
            oxadiazole = list(ring)

    # No oxadiazole ⇒ no real linker. Any 6-C ring found here is a Phe/Tyr/Trp
    # side chain; returning it causes restore/flatten/freeze to grab side chains.
    if not oxadiazole:
        return [], []

    return phenyl, oxadiazole


def _restore_linker_by_smarts(final_mol, linker_snapshot):
    """
    Restore linker ring (phenyl + oxadiazole) positions by matching ring atoms
    in the final molecule to their pre-bonding snapshot positions.

    Uses ring topology (size + heteroatom count) instead of SMARTS so this works
    even when aromaticity flags are absent (SANITIZE_NONE path).

    Matching: greedy nearest-neighbour within the topology-identified ring set.
    Because the linker moves as a rigid body, relative intra-ring distances are
    preserved and greedy assignment is reliable.

    Returns number of atoms restored.
    """
    conf = final_mol.GetConformer()

    phenyl, oxadiazole = _find_linker_ring_atoms(final_mol)
    final_indices = phenyl + oxadiazole

    if not final_indices:
        print("  [linker] WARNING: phenyl/oxadiazole rings not found by topology — skipping restore")
        return 0

    snapshot_positions = list(linker_snapshot.values())

    if len(final_indices) != len(snapshot_positions):
        print(f"  [linker] WARNING: topology found {len(final_indices)} atoms but snapshot "
              f"has {len(snapshot_positions)} — skipping")
        return 0

    # Greedy nearest-neighbour: for each snapshot position find the closest
    # unmatched atom from the topology-identified linker atom set.
    used_final = set()
    assignments = []

    for snap_pos in snapshot_positions:
        snap = np.array(snap_pos)
        best_fi, best_dist = None, float('inf')
        for fi in final_indices:
            if fi in used_final:
                continue
            p = conf.GetAtomPosition(fi)
            d = np.linalg.norm(np.array([p.x, p.y, p.z]) - snap)
            if d < best_dist:
                best_dist, best_fi = d, fi
        if best_fi is not None:
            assignments.append((best_fi, snap_pos))
            used_final.add(best_fi)

    n_restored = 0
    for fi, pos in assignments:
        p = conf.GetAtomPosition(fi)
        if np.linalg.norm(np.array([p.x, p.y, p.z]) - np.array(pos)) > 1e-6:
            conf.SetAtomPosition(fi, (float(pos[0]), float(pos[1]), float(pos[2])))
        n_restored += 1

    print(f"  [linker] topology restore: {n_restored} ring atoms → snapshot positions "
          f"(phenyl={len(phenyl)}, oxadiazole={len(oxadiazole)})")
    return n_restored


def enforce_linker_flat(final_mol):
    """
    Force all linker ring atoms (phenyl + oxadiazole) to Z = 0.

    Uses ring topology matching (not SMARTS) so this works even when the
    molecule has gone through SANITIZE_NONE and aromatic flags may be absent.
    Called last in the export pipeline as an absolute guarantee of flatness.
    """
    conf = final_mol.GetConformer()
    phenyl, oxadiazole = _find_linker_ring_atoms(final_mol)
    all_linker = phenyl + oxadiazole

    n_forced = 0
    for idx in all_linker:
        p = conf.GetAtomPosition(idx)
        if abs(p.z) > 1e-6:
            conf.SetAtomPosition(idx, (p.x, p.y, 0.0))
            n_forced += 1

    if all_linker:
        print(f"  [linker] Z=0 enforced on {len(all_linker)} ring atoms "
              f"(phenyl={len(phenyl)}, oxadiazole={len(oxadiazole)}, "
              f"{n_forced} had non-zero Z)")
    else:
        print("  [linker] WARNING: no phenyl/oxadiazole rings found — Z=0 enforcement skipped")


def restore_ring_positions_from_snapshot(final_mol, snapshots, sugar_tolerance=0.5):
    """
    After bonding/export, restore ring atom positions from pre-bonding snapshot.

    Sugar rings  — position-based matching with tight tolerance (they do not move).
    Linker rings — SMARTS matching on final mol (reliable regardless of displacement;
                   SetDihedralRad and Cα alignment can move the linker 10–30 Å).
    """
    print("\n" + "="*70)
    print("FREEZING RING POSITIONS (snapshot restore)")
    print("="*70)

    conf = final_mol.GetConformer()
    n_atoms = final_mol.GetNumAtoms()

    current_positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        p = conf.GetAtomPosition(i)
        current_positions[i] = [p.x, p.y, p.z]

    total_restored = 0
    total_drifted  = 0

    for mol_name, ring_atom_map in snapshots.items():
        if mol_name == '__linker__':
            n = _restore_linker_by_smarts(final_mol, ring_atom_map)
            total_restored += n
            total_drifted  += n
            continue

        # Sugar rings: position-based with tight tolerance
        for _, stored_pos in ring_atom_map.items():
            stored = np.array(stored_pos)
            dists = np.linalg.norm(current_positions - stored, axis=1)
            nearest_idx  = int(np.argmin(dists))
            nearest_dist = dists[nearest_idx]

            if nearest_dist > sugar_tolerance:
                print(f"  WARNING ({mol_name}): snapshot atom not found within "
                      f"{sugar_tolerance} Å (nearest={nearest_dist:.3f} Å) — skipping")
                continue

            if nearest_dist > 1e-6:
                conf.SetAtomPosition(nearest_idx, tuple(stored_pos))
                current_positions[nearest_idx] = stored
                total_drifted += 1

            total_restored += 1

    print(f"  Ring atoms found:   {total_restored}")
    print(f"  Ring atoms drifted: {total_drifted}  (corrected)")
    print("="*70)


def restore_ring_coms_after_export(final_mol, molecule_data_dict, tolerance=2.0):
    """Kept for backward compatibility — no longer modifies positions."""
    print("\n[restore_ring_coms_after_export] skipped — use snapshot_ring_positions "
          "/ restore_ring_positions_from_snapshot instead.")
    pass

def restore_glycopeptide_positions(final_mol, molecule_data_dict, peptide_data, tolerance=2.0):
    """
    Restore glycopeptide positions - sugars AND peptide residues.
    """
    print("\n" + "="*70)
    print("RESTORING GLYCOPEPTIDE TO EXPERIMENTAL POSITIONS")
    print("="*70)

    # Must be called before any is_sugar_ring_carbon() / find_residue_atoms() checks
    # so that GetRingInfo() returns correct data and the BFS stops at sugar boundaries.
    Chem.GetSymmSSSR(final_mol)

    conf = final_mol.GetConformer()
    
    # ========================================================================
    # PART 1: Sugar rings — already frozen by snapshot before bonding; skip.
    # ========================================================================
    print(f"\nSugar ring positions frozen via snapshot — no correction needed.")
    
    # ========================================================================
    # PART 2: Restore peptide residues (Cα + rotation)
    # ========================================================================
    print(f"\n{'='*70}")
    print("RESTORING PEPTIDE RESIDUES (Cα + Functional Group Orientation)")
    print(f"{'='*70}")
    
    exp_residues = peptide_data.get('experimental_data', [])
    sequence = peptide_data.get('sequence', [])
    residue_info = peptide_data.get('residue_info', [])
    
    print(f"Peptide sequence: {'-'.join(sequence)}")
    
    # Find Cα atoms IN ORDER (they are built sequentially)
    ca_indices = find_peptide_ca_atoms(final_mol)
    print(f"Found {len(ca_indices)} Cα atoms: {ca_indices}")
    
    if len(ca_indices) != len(exp_residues):
        print(f"  WARNING: Expected {len(exp_residues)} residues, found {len(ca_indices)} Ca atoms!")
        return
    
    # DEBUG: Print what we're matching
    print(f"\nMatching Cα atoms by sequential order:")
    for i in range(len(ca_indices)):
        ca_idx = ca_indices[i]
        pos = conf.GetAtomPosition(ca_idx)
        exp_pos = exp_residues[i]['ca_position']
        print(f"  Ca #{ca_idx} (built as {sequence[i]}) → Experimental {i} ({exp_residues[i]['aa']}) at {exp_pos}")
    
    # Match by ORDER, not proximity (peptide built sequentially)
    # Cα index i in molecule corresponds to residue i in sequence
    matched_pairs = [(i, i) for i in range(len(ca_indices))]
    
    # Restore each residue
    for curr_idx, exp_idx in matched_pairs:
        ca_idx = ca_indices[curr_idx]
        residue_aa = exp_residues[exp_idx]['aa']
        
        # Get current and target Cα positions
        pos = conf.GetAtomPosition(ca_idx)
        current_ca = np.array([pos.x, pos.y, pos.z])
        target_ca = np.array(exp_residues[exp_idx]['ca_position'])
        
        print(f"\n  Residue {exp_idx} ({residue_aa}):")
        print(f"    Current Cα: [{current_ca[0]:.3f}, {current_ca[1]:.3f}, {current_ca[2]:.3f}]")
        print(f"    Target Cα:  [{target_ca[0]:.3f}, {target_ca[1]:.3f}, {target_ca[2]:.3f}]")
        
        # Find all atoms in this residue
        residue_atoms = find_residue_atoms(final_mol, ca_idx)
        
        # STEP 1: Translate so Cα is at target position
        translation = target_ca - current_ca
        
        for atom_idx in residue_atoms:
            pos = conf.GetAtomPosition(atom_idx)
            pos_arr = np.array([pos.x, pos.y, pos.z])
            new_pos = pos_arr + translation
            conf.SetAtomPosition(atom_idx, tuple(new_pos))
        
        print(f"    Translation: {np.linalg.norm(translation):.3f} Å")
        
        # STEP 2: Rotate side chain toward functional_position using centroid
        # (func_idx from residue_info is stale after mol combination — use centroid instead)
        target_func = exp_residues[exp_idx].get('functional_position')

        if target_func is not None:
            ca_xy = target_ca[:2]
            target_xy = np.array(target_func[:2])

            side_atoms = _find_side_chain_atoms(final_mol, ca_idx)
            if side_atoms:
                sc_positions = np.array([[conf.GetAtomPosition(i).x,
                                          conf.GetAtomPosition(i).y] for i in side_atoms])
                centroid_xy = sc_positions.mean(axis=0)
                cv = centroid_xy - ca_xy
                tv = target_xy  - ca_xy
                if np.linalg.norm(cv) > 0.1 and np.linalg.norm(tv) > 0.1:
                    cv /= np.linalg.norm(cv); tv /= np.linalg.norm(tv)
                    angle = np.arctan2(np.cross(cv, tv), np.dot(cv, tv))
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    rot2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    for atom_idx in side_atoms:
                        pos = conf.GetAtomPosition(atom_idx)
                        rel = np.array([pos.x, pos.y]) - ca_xy
                        new_xy = rot2d @ rel + ca_xy
                        conf.SetAtomPosition(atom_idx, (new_xy[0], new_xy[1], 0.0))
                    print(f"    Side chain rotated {np.degrees(angle):.1f}° → functional_position")
        
        # Flatten to Z=0
        for atom_idx in residue_atoms:
            pos = conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(atom_idx, [pos.x, pos.y, 0.0])
        
        print(f"    ✓ Restored {len(residue_atoms)} atoms")
    
    print("\n" + "="*70)
    print("✓ Glycopeptide restoration complete")
    print("="*70)

def _find_side_chain_atoms(mol, ca_idx):
    """
    BFS from ca_idx into the side chain only.
    Stops at backbone N, backbone C=O, sugar ring carbons, and H atoms.
    Requires ring info to be populated (call Chem.GetSymmSSSR before using).
    """
    ca_atom = mol.GetAtomWithIdx(ca_idx)
    backbone_n, backbone_c = None, None
    for nb in ca_atom.GetNeighbors():
        if nb.GetSymbol() == 'N':
            backbone_n = nb.GetIdx()
        elif nb.GetSymbol() == 'C':
            for nn in nb.GetNeighbors():
                if nn.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(nb.GetIdx(), nn.GetIdx())
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        backbone_c = nb.GetIdx()
                        break

    blocked = {ca_idx}
    if backbone_n is not None:
        blocked.add(backbone_n)
    if backbone_c is not None:
        blocked.add(backbone_c)

    roots = [n.GetIdx() for n in ca_atom.GetNeighbors()
             if n.GetIdx() not in blocked and n.GetSymbol() != 'H']
    if not roots:
        return []

    visited = {ca_idx} | set(roots)
    queue, side_atoms = list(roots), list(roots)
    while queue:
        cur = queue.pop(0)
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            nidx = n.GetIdx()
            if nidx not in visited and mol.GetAtomWithIdx(nidx).GetSymbol() != 'H':
                if is_sugar_ring_carbon(mol, nidx):
                    continue
                visited.add(nidx)
                queue.append(nidx)
                side_atoms.append(nidx)
    return side_atoms


def find_residue_atoms(mol, ca_idx):
    """
    Find all atoms belonging to a single amino acid residue.
    
    IMPORTANT: Stops at glycosidic bonds - does NOT include bonded sugars!
    """
    visited = set()
    queue = [ca_idx]
    residue_atoms = []
    
    ca_atom = mol.GetAtomWithIdx(ca_idx)
    
    # Find backbone atoms
    backbone_c = None  # C=O (next residue)
    backbone_n = None  # N-H (previous residue)
    
    for neighbor in ca_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            # Check if it's carbonyl C
            for nn in neighbor.GetNeighbors():
                if nn.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), nn.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        backbone_c = neighbor.GetIdx()
                        break
        elif neighbor.GetSymbol() == 'N':
            backbone_n = neighbor.GetIdx()
    
    # BFS from Cα
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        
        visited.add(current)
        residue_atoms.append(current)
        
        atom = mol.GetAtomWithIdx(current)
        
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            
            if neighbor_idx in visited:
                continue
            
            # Don't cross peptide bonds to next residue
            if current == backbone_c and neighbor.GetSymbol() == 'N':
                continue
            
            # Don't cross peptide bonds to previous residue
            if current == backbone_n and neighbor.GetSymbol() == 'C':
                is_carbonyl = False
                for nn in neighbor.GetNeighbors():
                    if nn.GetSymbol() == 'O':
                        bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), nn.GetIdx())
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            is_carbonyl = True
                            break
                if is_carbonyl:
                    continue
            
            # NEW: Don't cross glycosidic bonds (N or O bonded to ring carbon)
            if neighbor.GetSymbol() == 'C':
                # Check if this carbon is part of a sugar ring
                if is_sugar_ring_carbon(mol, neighbor_idx):
                    print(f"      Stopping at glycosidic bond: atom {current} → sugar C{neighbor_idx}")
                    continue
            
            queue.append(neighbor_idx)
    
    return residue_atoms

def is_sugar_ring_carbon(mol, atom_idx):
    """Check if an atom is part of a pyranose (6-membered sugar) ring."""
    atom = mol.GetAtomWithIdx(atom_idx)
    
    if atom.GetSymbol() != 'C':
        return False
    
    # Check if atom is in a 6-membered ring with one oxygen
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if atom_idx in ring and len(ring) == 6:
            oxygen_count = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'O')
            if oxygen_count == 1:
                return True
    
    return False

def find_peptide_ca_atoms(mol):
    """Find Cα atoms: carbons bonded to a backbone N and a carbonyl C.

    The backbone-Cα signature is being bonded to an amide N *and* to a
    carbonyl carbon (C=O). Chirality is NOT required: glycine's Cα carries
    two hydrogens and is not a chiral centre, so a chirality gate silently
    drops every glycine and misaligns the sequential residue restoration
    downstream. The N + carbonyl pattern alone is specific to backbone Cα
    (side-chain amide/guanidinium carbons fail one of the two tests).
    """
    ca_indices = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue

        neighbors = list(atom.GetNeighbors())
        has_n = any(n.GetSymbol() == 'N' for n in neighbors)
        has_carbonyl = False

        for n in neighbors:
            if n.GetSymbol() == 'C':
                for nn in n.GetNeighbors():
                    if nn.GetSymbol() == 'O':
                        bond = mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx())
                        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                            has_carbonyl = True
                            break
            if has_carbonyl:
                break

        if has_n and has_carbonyl:
            ca_indices.append(atom.GetIdx())
    return ca_indices

def find_peptide_atom_range(mol, ca_indices):
    """Find all atoms connected to peptide (BFS from Cα atoms)"""
    if not ca_indices:
        return []
    
    visited = set()
    queue = list(ca_indices)
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        
        visited.add(current)
        atom = mol.GetAtomWithIdx(current)
        
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                queue.append(neighbor_idx)
    
    return sorted(list(visited))

def match_ca_positions(current_positions, exp_positions, tolerance=5.0):
    """Match current Cα positions to experimental positions by proximity"""
    matched_pairs = []
    used_exp = set()
    
    for i, curr_pos in enumerate(current_positions):
        best_match = None
        min_dist = tolerance
        
        for j, exp_pos in enumerate(exp_positions):
            if j in used_exp:
                continue
            
            dist = np.linalg.norm(curr_pos - exp_pos)
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        if best_match is not None:
            matched_pairs.append((i, best_match))
            used_exp.add(best_match)
    
    return matched_pairs