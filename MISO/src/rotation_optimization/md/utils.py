import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ...constants import (
    MIN_ATOM_DISTANCE, DEFAULT_MIN_SEPARATION, EPSILON,
)

def check_convergence(energy_history, force_history, rmsd_history, config):
    """
    Check if optimization has converged.
    
    Args:
        energy_history: List of recent energies
        force_history: List of recent max forces
        rmsd_history: List of recent RMSD values
        config: OptimizationConfig
    
    Returns:
        Tuple of (converged: bool, reason: str)
    """
    if not config.enable_convergence:
        return False, "Convergence checking disabled"
    
    window = config.convergence_window
    
    # Need enough history
    if len(energy_history) < window:
        return False, "Not enough history"
    
    # Check energy convergence
    recent_energies = energy_history[-window:]
    energy_change = max(recent_energies) - min(recent_energies)
    energy_converged = energy_change < config.convergence_energy_threshold
    
    # Check force convergence
    recent_forces = force_history[-window:]
    max_recent_force = max(recent_forces)
    force_converged = max_recent_force < config.convergence_force_threshold
    
    # Check RMSD convergence
    if len(rmsd_history) >= window:
        recent_rmsd = rmsd_history[-window:]
        max_rmsd = max(recent_rmsd)
        rmsd_converged = max_rmsd < config.convergence_rmsd_threshold
    else:
        rmsd_converged = False
    
    # All criteria must be met
    if energy_converged and force_converged and rmsd_converged:
        reason = (f"Energy Δ={energy_change:.4f}, "
                 f"Max Force={max_recent_force:.4f}, "
                 f"Max RMSD={max_rmsd:.6f}")
        return True, reason
    
    return False, "Not converged"

def calculate_rmsd(positions1, positions2):
    """Calculate RMSD between two sets of positions."""
    diff = positions1 - positions2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def save_frame(mol, output_name, frame_name):
    """Save visualization frame as image."""
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol, size=(800, 800))
        img.save(f"{output_name}_frames/{frame_name}.png")
    except Exception as e:
        print(f"Warning: Could not save frame {frame_name}: {e}")

def check_geometry(mol):
    """
    Check molecule geometry for overlapping atoms.
    
    Returns:
        bool: True if geometry is acceptable
    """
    conf = mol.GetConformer()
    
    print("\n=== GEOMETRY CHECK ===")
    
    overlaps = 0
    for i in range(mol.GetNumAtoms()):
        pos_i = conf.GetAtomPosition(i)
        pi = np.array([pos_i.x, pos_i.y, pos_i.z])
        
        for j in range(i+1, mol.GetNumAtoms()):
            pos_j = conf.GetAtomPosition(j)
            pj = np.array([pos_j.x, pos_j.y, pos_j.z])
            
            dist = np.linalg.norm(pi - pj)
            if dist < MIN_ATOM_DISTANCE:
                overlaps += 1
                if overlaps <= 5:  # Print first 5
                    print(f"  WARNING: Atoms {i}-{j} very close: {dist:.3f}Å")
    
    if overlaps > 0:
        print(f"  Total overlaps: {overlaps}")
        print("  → Structure needs fixing before optimization!")
    else:
        print("  ✓ No obvious overlaps")
    
    print("="*70)
    return overlaps == 0

def fix_overlapping_atoms(mol, min_distance=DEFAULT_MIN_SEPARATION):
    """
    Fix overlapping atoms by pushing them apart.
    
    Args:
        mol: RDKit molecule
        min_distance: Minimum allowed distance between atoms
    
    Returns:
        Fixed molecule
    """
    print(f"\n{'='*70}")
    print("FIXING OVERLAPPING ATOMS")
    print("="*70)
    
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()
    
    # Find all overlapping pairs
    overlaps = []
    for i in range(mol_copy.GetNumAtoms()):
        pos_i = conf.GetAtomPosition(i)
        pi = np.array([pos_i.x, pos_i.y, pos_i.z])
        atom_i = mol_copy.GetAtomWithIdx(i)
        
        for j in range(i+1, mol_copy.GetNumAtoms()):
            pos_j = conf.GetAtomPosition(j)
            pj = np.array([pos_j.x, pos_j.y, pos_j.z])
            
            dist = np.linalg.norm(pi - pj)
            if dist < min_distance:
                atom_j = mol_copy.GetAtomWithIdx(j)
                overlaps.append({
                    'i': i, 
                    'j': j, 
                    'dist': dist,
                    'atom_i': atom_i.GetSymbol(),
                    'atom_j': atom_j.GetSymbol()
                })
    
    print(f"Found {len(overlaps)} overlapping pairs:")
    for overlap in overlaps:
        print(f"  Atoms {overlap['i']}({overlap['atom_i']}) - "
              f"{overlap['j']}({overlap['atom_j']}): {overlap['dist']:.3f}Å")
    
    # Fix by pushing atoms apart
    fixed = 0
    for overlap in overlaps:
        i, j = overlap['i'], overlap['j']
        
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        pi = np.array([pos_i.x, pos_i.y, pos_i.z])
        pj = np.array([pos_j.x, pos_j.y, pos_j.z])
        
        vec = pj - pi
        current_dist = np.linalg.norm(vec)
        
        if current_dist < 0.01:  # Atoms at same position
            vec = np.random.randn(3)
        
        vec_norm = vec / np.linalg.norm(vec)
        midpoint = (pi + pj) / 2
        
        new_pi = midpoint - vec_norm * (min_distance / 2)
        new_pj = midpoint + vec_norm * (min_distance / 2)
        
        conf.SetAtomPosition(i, tuple(new_pi))
        conf.SetAtomPosition(j, tuple(new_pj))
        fixed += 1
    
    print(f"✓ Fixed {fixed} overlapping pairs")
    print("="*70)
    
    return mol_copy

def fix_valence_issues(mol):
    """
    Remove extra hydrogens from atoms exceeding valence.
    
    Returns:
        Fixed molecule
    """
    mol_rw = Chem.RWMol(mol)
    
    # Collect atoms to fix
    problems = []
    for atom in mol_rw.GetAtoms():
        idx = atom.GetIdx()
        expected_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
        
        # Count actual bonds
        total_valence = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
        
        if total_valence > max(expected_valence):
            h_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'H')
            excess = int(total_valence - max(expected_valence))
            problems.append((idx, atom.GetSymbol(), h_count, excess))
    
    if not problems:
        return mol
    
    print(f"\nValence problems detected:")
    for idx, symbol, h_count, excess in problems:
        print(f"  Atom {idx} ({symbol}): has {h_count} H, needs to remove {excess}")
    
    # Remove hydrogens in reverse order to avoid index shifts
    atoms_to_remove = []
    for idx, symbol, h_count, excess in problems:
        atom = mol_rw.GetAtomWithIdx(idx)
        h_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
        
        for h_idx in h_neighbors[:excess]:
            atoms_to_remove.append(h_idx)
    
    for h_idx in sorted(set(atoms_to_remove), reverse=True):
        mol_rw.RemoveAtom(h_idx)
    
    fixed_mol = mol_rw.GetMol()
    
    try:
        Chem.SanitizeMol(fixed_mol)
        print("Valence fixed successfully")
        return fixed_mol
    except Exception as e:
        print(f"Could not fix valence: {e}")
        raise

def save_molecule(mol, filename, file_format='sdf'):
    """
    Save molecule to file.
    
    Args:
        mol: RDKit molecule
        filename: Output filename (without extension)
        file_format: File format ('sdf', 'mol2', 'pdb', 'xyz')
    
    Returns:
        Path to saved file
    """
    file_format = file_format.lower()
    
    if file_format == 'sdf':
        filepath = f"{filename}.sdf"
        writer = Chem.SDWriter(filepath)
        writer.write(mol)
        writer.close()
        print(f"✓ Saved to {filepath}")
        
    elif file_format == 'mol2':
        filepath = f"{filename}.mol2"
        Chem.MolToMol2File(mol, filepath)
        print(f"✓ Saved to {filepath}")
        
    elif file_format == 'pdb':
        filepath = f"{filename}.pdb"
        Chem.MolToPDBFile(mol, filepath)
        print(f"✓ Saved to {filepath}")
        
    elif file_format == 'xyz':
        filepath = f"{filename}.xyz"
        Chem.MolToXYZFile(mol, filepath)
        print(f"✓ Saved to {filepath}")
        
    else:
        raise ValueError(f"Unsupported format: {file_format}. "
                        f"Use 'sdf', 'mol2', 'pdb', or 'xyz'")
    
    return filepath