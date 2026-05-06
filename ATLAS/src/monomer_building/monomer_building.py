# Standard library
import os
import sys
import math
import copy
import pickle
from collections import defaultdict
import random

# Third-party numerical and plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# RDKit core
import rdkit
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem, 
    Descriptors, 
    Draw, 
    rdMolDescriptors,
    rdMolAlign,
    ChemicalFeatures,
    rdDepictor,
    rdDistGeom
)
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole

# IPython display
from IPython.display import Image

# RDKit configuration
IPythonConsole.ipython_3d = True
print(f"RDKit version: {rdkit.__version__}")

# Initialize RDKit feature factory for H-bond detection
fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def view_mol(smiles):
    mol_kdo = Chem.MolFromSmiles(smiles)
    mol_kdo = Chem.AddHs(mol_kdo)


    AllChem.EmbedMolecule(mol_kdo)
    AllChem.MMFFOptimizeMolecule(mol_kdo)

    # Create 3D view manually
    view = py3Dmol.view(width=500, height=400)
    sdf = Chem.MolToMolBlock(mol_kdo)
    view.addModel(sdf, 'sdf')
    view.setStyle({'stick': {'radius': 0.1}, 'sphere': {'radius': 0.3}})
    view.zoomTo()
    view.show()

def identify_sugar_carbons(mol):
    """Identify ALL carbons in pyranose ring (C1-C5) plus C6 if present."""
    ring_info = mol.GetRingInfo()
    carbon_map = {}
    
    for ring in ring_info.AtomRings():
        if len(ring) == 6:  # Pyranose ring
            # Find ring oxygen
            ring_oxygen_idx = None
            for idx in ring:
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'O':
                    ring_oxygen_idx = idx
                    break
            
            if ring_oxygen_idx is None:
                continue
            
            ring_carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'C']
            ring_oxygen = mol.GetAtomWithIdx(ring_oxygen_idx)
            
            # Find C1 and C5 (both bonded to ring oxygen)
            carbons_bonded_to_ring_o = []
            for neighbor in ring_oxygen.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    carbons_bonded_to_ring_o.append(neighbor.GetIdx())
            
            if len(carbons_bonded_to_ring_o) != 2:
                continue
                
            # Determine which is C1 (has more oxygen neighbors)
            c1_candidate = carbons_bonded_to_ring_o[0]
            c5_candidate = carbons_bonded_to_ring_o[1]
            
            c1_atom = mol.GetAtomWithIdx(c1_candidate)
            c5_atom = mol.GetAtomWithIdx(c5_candidate)
            
            c1_oxygens = sum(1 for n in c1_atom.GetNeighbors() if n.GetSymbol() == 'O')
            c5_oxygens = sum(1 for n in c5_atom.GetNeighbors() if n.GetSymbol() == 'O')
            
            if c1_oxygens >= c5_oxygens:
                carbon_map['C1'] = c1_candidate
                carbon_map['C5'] = c5_candidate
            else:
                carbon_map['C1'] = c5_candidate
                carbon_map['C5'] = c1_candidate
            
            # Store ring oxygen
            carbon_map['ring_oxygen'] = ring_oxygen_idx
            
            # Traverse ring to find C2, C3, C4
            current_idx = carbon_map['C1']
            visited = {current_idx, ring_oxygen_idx}
            carbon_number = 1
            
            while carbon_number < 5:
                current_atom = mol.GetAtomWithIdx(current_idx)
                
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    
                    if neighbor_idx in ring and neighbor_idx not in visited:
                        if neighbor.GetSymbol() == 'C':
                            carbon_number += 1
                            carbon_map[f'C{carbon_number}'] = neighbor_idx
                            visited.add(neighbor_idx)
                            current_idx = neighbor_idx
                            break
                        elif neighbor.GetSymbol() == 'O':  # Hit ring oxygen
                            visited.add(neighbor_idx)
                            for o_neighbor in neighbor.GetNeighbors():
                                if o_neighbor.GetIdx() in ring and o_neighbor.GetIdx() not in visited:
                                    carbon_number += 1
                                    carbon_map[f'C{carbon_number}'] = o_neighbor.GetIdx()
                                    visited.add(o_neighbor.GetIdx())
                                    current_idx = o_neighbor.GetIdx()
                                    break
                            break
                else:
                    break
            
            # Find C6 (exocyclic carbon bonded to C5, outside ring)
            # print(f"\n  Looking for C6 (exocyclic carbon bonded to C5)...")
            # print(f"  Ring atoms: {ring}")

            # Find exocyclic chain (C6, C7, ...) starting from C5
            if 'C5' in carbon_map:
                current_chain_idx = carbon_map['C5']
                visited_chain = set(ring)
                carbon_number = 5

                while True:
                    current_atom = mol.GetAtomWithIdx(current_chain_idx)
                    next_carbon = None

                    for neighbor in current_atom.GetNeighbors():
                        n_idx = neighbor.GetIdx()
                        if neighbor.GetSymbol() == 'C' and n_idx not in visited_chain:
                            next_carbon = n_idx
                            break

                    if next_carbon is None:
                        break

                    carbon_number += 1
                    carbon_map[f'C{carbon_number}'] = next_carbon
                    visited_chain.add(next_carbon)
                    current_chain_idx = next_carbon
            
            # if 'C5' in carbon_map:
            #     c5_idx = carbon_map['C5']
            #     c5_atom = mol.GetAtomWithIdx(c5_idx)
            #     neighbors = [(n.GetIdx(), n.GetSymbol()) for n in c5_atom.GetNeighbors()]
            #     print(f"  C5 (idx {c5_idx}) neighbors: {neighbors}")
                
            #     for neighbor in c5_atom.GetNeighbors():
            #         n_idx = neighbor.GetIdx()
            #         print(f"    Checking neighbor {n_idx}: symbol={neighbor.GetSymbol()}, in_ring={n_idx in ring}")
                    
            #         if neighbor.GetSymbol() == 'C' and n_idx not in ring:
            #             carbon_map['C6'] = n_idx
            #             print(f"    ✓ Found C6 at index {n_idx}")
            #             break
            #     else:
            #         print(f"    ✗ No exocyclic carbon found on C5")
            
            carbon_map['all_ring_carbons'] = ring_carbons
            break
    
    return carbon_map

def identify_sugar_carbons_v0(mol):
    """Identify ALL carbons in pyranose ring (C1-C5) plus C6 if present."""
    ring_info = mol.GetRingInfo()
    carbon_map = {}
    
    for ring in ring_info.AtomRings():
        if len(ring) == 6:  # Pyranose ring
            # Find ring oxygen
            ring_oxygen_idx = None
            for idx in ring:
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'O':
                    ring_oxygen_idx = idx
                    break
            
            if ring_oxygen_idx is None:
                continue
            
            ring_carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'C']
            ring_oxygen = mol.GetAtomWithIdx(ring_oxygen_idx)
            
            # Find C1 and C5 (both bonded to ring oxygen)
            carbons_bonded_to_ring_o = []
            for neighbor in ring_oxygen.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    carbons_bonded_to_ring_o.append(neighbor.GetIdx())
            
            if len(carbons_bonded_to_ring_o) != 2:
                continue
                
            # Determine which is C1 (has more oxygen neighbors - the anomeric carbon)
            c1_candidate = carbons_bonded_to_ring_o[0]
            c5_candidate = carbons_bonded_to_ring_o[1]
            
            c1_atom = mol.GetAtomWithIdx(c1_candidate)
            c5_atom = mol.GetAtomWithIdx(c5_candidate)
            
            c1_oxygens = sum(1 for n in c1_atom.GetNeighbors() if n.GetSymbol() == 'O')
            c5_oxygens = sum(1 for n in c5_atom.GetNeighbors() if n.GetSymbol() == 'O')
            
            if c1_oxygens >= c5_oxygens:
                carbon_map['C1'] = c1_candidate
                carbon_map['C5'] = c5_candidate
            else:
                carbon_map['C1'] = c5_candidate
                carbon_map['C5'] = c1_candidate
            
            # Now traverse the ring from C1 to number C2, C3, C4
            # Start at C1 and walk around the ring
            current_idx = carbon_map['C1']
            visited = {current_idx, ring_oxygen_idx}
            carbon_number = 1
            
            # Find path from C1 to C5 (one direction around ring)
            path = [current_idx]
            
            while carbon_number < 5:
                current_atom = mol.GetAtomWithIdx(current_idx)
                
                # Find next carbon in ring (not visited yet)
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    
                    if neighbor_idx in ring and neighbor_idx not in visited:
                        if neighbor.GetSymbol() == 'C':
                            carbon_number += 1
                            carbon_map[f'C{carbon_number}'] = neighbor_idx
                            visited.add(neighbor_idx)
                            current_idx = neighbor_idx
                            path.append(neighbor_idx)
                            break
                        elif neighbor.GetSymbol() == 'O':  # Hit ring oxygen, jump to C5
                            visited.add(neighbor_idx)
                            # Next carbon should be C5
                            for o_neighbor in neighbor.GetNeighbors():
                                if o_neighbor.GetIdx() in ring and o_neighbor.GetIdx() not in visited:
                                    carbon_number += 1
                                    carbon_map[f'C{carbon_number}'] = o_neighbor.GetIdx()
                                    visited.add(o_neighbor.GetIdx())
                                    current_idx = o_neighbor.GetIdx()
                                    path.append(o_neighbor.GetIdx())
                                    break
                            break
                else:
                    break  # No more neighbors found
            
            # Find C6 (exocyclic carbon bonded to C5, outside ring)
            if 'C5' in carbon_map:
                c5_atom = mol.GetAtomWithIdx(carbon_map['C5'])
                for neighbor in c5_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() not in ring:
                        carbon_map['C6'] = neighbor.GetIdx()
                        break
            
            carbon_map['all_ring_carbons'] = ring_carbons
            break
    
    return carbon_map
# ============================================================================
# CREMER-POPLE PUCKERING ANALYSIS FUNCTIONS
# ============================================================================

def calculate_cremer_pople_parameters(ring_coords):
    """
    Corrected Cremer-Pople puckering parameters for a 6-membered ring.
    Based on the original Cremer & Pople (1975) formulation.
    
    Args:
        ring_coords: numpy array of shape (6, 3) with ring atom coordinates
        
    Returns:
        Dictionary with:
            - Q: puckering amplitude (Å)
            - theta: polar angle (degrees, 0-180)
            - phi: pseudorotation angle (degrees, 0-360)
            - puckering_type: classification (chair, boat, twist, envelope, etc.)
    """
    
    if ring_coords.shape != (6, 3):
        raise ValueError("Ring coordinates must be (6, 3)")
    
    N = 6  # Number of ring atoms
    
    # Center the ring
    center = np.mean(ring_coords, axis=0)
    centered_positions = ring_coords - center
    
    # Calculate the best-fit plane through all atoms
    # Use SVD to find the normal vector
    U, S, Vt = np.linalg.svd(centered_positions, full_matrices=False)
    normal = Vt[-1]  # Normal vector (smallest singular value)
    
    # Calculate z-coordinates as displacements from the plane
    z_j = np.array([np.dot(pos, normal) for pos in centered_positions])
    
    # Cremer-Pople puckering coordinates
    # For 6-membered rings: calculate q_m for m = 2, 3, 4, 5
    q_values = {}
    
    for m in range(2, N):  # m = 2, 3, 4, 5
        q_m_real = 0
        q_m_imag = 0
        
        for j in range(N):  # j = 0, 1, 2, 3, 4, 5
            phi_jm = 2 * np.pi * j * m / N
            q_m_real += z_j[j] * np.cos(phi_jm)
            q_m_imag += z_j[j] * np.sin(phi_jm)
        
        # Apply normalization factor
        normalization = np.sqrt(2.0 / N)
        q_values[m] = {
            'real': q_m_real * normalization,
            'imag': q_m_imag * normalization,
            'magnitude': np.sqrt(q_m_real**2 + q_m_imag**2) * normalization
        }
    
    # Extract individual q values
    q2 = q_values[2]['real']  # q2 is always real for symmetry
    q3 = q_values[3]['real']  # q3 is always real for symmetry
    q4 = q_values[4]['real']  # q4 is always real for symmetry  
    q5 = q_values[5]['real']  # q5 is always real for symmetry
    
    # Alternative approach - calculate using the magnitudes
    q2_mag = q_values[2]['magnitude']
    q3_mag = q_values[3]['magnitude'] 
    q4_mag = q_values[4]['magnitude']
    q5_mag = q_values[5]['magnitude']
    
    # Total puckering amplitude Q
    Q = np.sqrt(q2_mag**2 + q3_mag**2 + q4_mag**2 + q5_mag**2)
    
    # Calculate theta and phi
    if Q > 1e-6:
        # Theta: polar angle (0 to π)
        cos_theta = q3_mag / Q
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        
        # For chair conformations, we expect theta ≈ 0° or 180°
        # If q3 is negative, theta should be > 90°
        if q3 < 0:
            theta_deg = 180 - theta_deg
            
        # Phi: pseudorotation angle (0 to 2π) 
        # Use q2 and q4 to determine phi
        if abs(q2) > 1e-6 or abs(q4) > 1e-6:
            phi_rad = np.arctan2(q4, q2)
            if phi_rad < 0:
                phi_rad += 2 * np.pi
            phi_deg = np.degrees(phi_rad)
        else:
            phi_deg = 0
            
    else:
        theta_deg = 0
        phi_deg = 0
    
    # Classify based on Cremer-Pople parameters
    puckering_type = classify_puckering(Q, theta_deg, phi_deg)
    
    return {
        'Q': Q,
        'theta': theta_deg,
        'phi': phi_deg,
        'puckering_type': puckering_type,
        'q2': q2,
        'q3': q3,
        'q4': q4,
        'q5': q5,
        'debug_info': {
            'z_displacements': z_j.tolist(),
            'q_values': q_values
        }
    }

def classify_puckering(Q, theta_deg, phi_deg):
    """
    Classify 6-membered ring conformation based on Cremer-Pople parameters.
    
    Args:
        Q: Total puckering amplitude (Å)
        theta_deg: Theta angle in degrees (0-180)
        phi_deg: Phi angle in degrees (0-360)
    
    Returns:
        str: Conformation classification
    """
    
    if Q < 0.1:
        return "planar"
    
    # Chair conformations: theta ~ 0° or 180°
    if theta_deg < 15:
        # 4C1-like chair
        return "chair_4C1"
    elif theta_deg > 165:
        # 1C4-like chair  
        return "chair_1C4"
    
    # Boat conformations: theta ~ 90°
    elif 75 <= theta_deg <= 105:
        # Use phi to distinguish boat types
        phi_normalized = phi_deg % 180  # Consider symmetry
        
        if phi_normalized < 30 or phi_normalized > 150:
            return "boat"
        elif 60 <= phi_normalized <= 120:
            return "skew_boat"
        else:
            return "boat_intermediate"
    
    # Half-chair conformations: theta ~ 50° or 130°
    elif (35 <= theta_deg <= 65) or (115 <= theta_deg <= 145):
        return "half_chair"
    
    # Envelope conformations: theta ~ 25° or 155°
    elif (15 <= theta_deg <= 35) or (145 <= theta_deg <= 165):
        # Further classify envelope by phi
        if 0 <= phi_deg < 60 or 300 <= phi_deg < 360:
            return "envelope_1"
        elif 60 <= phi_deg < 120:
            return "envelope_2" 
        elif 120 <= phi_deg < 180:
            return "envelope_3"
        elif 180 <= phi_deg < 240:
            return "envelope_4"
        elif 240 <= phi_deg < 300:
            return "envelope_5"
        else:
            return "envelope"
    
    else:
        # Twist conformations or other intermediates
        return f"twist_intermediate"

def analyze_conformer_puckering_cremer_pople(mol, conf_id):
    """
    Analyze single conformer's puckering using corrected Cremer-Pople method.
    
    Args:
        mol: RDKit molecule
        conf_id: conformer ID
        
    Returns:
        Dictionary with Cremer-Pople parameters and classification
    """
    
    conf = mol.GetConformer(conf_id)
    ring_info = mol.GetRingInfo()
    
    # Find the first 6-membered ring (should be the pyranose)
    for ring in ring_info.AtomRings():
        if len(ring) == 6:
            ring_coords = []
            for atom_idx in ring:
                pos = conf.GetAtomPosition(atom_idx)
                ring_coords.append([pos.x, pos.y, pos.z])
            
            ring_coords = np.array(ring_coords)
            
            try:
                result = calculate_cremer_pople_parameters(ring_coords)
                print(f"  Conformer {conf_id}: Q={result['Q']:.3f}, θ={result['theta']:.1f}°, φ={result['phi']:.1f}° → {result['puckering_type']}")
                return result
            except Exception as e:
                print(f"  Error calculating Cremer-Pople for conformer {conf_id}: {e}")
                return None
    
    print(f"  No 6-membered ring found in conformer {conf_id}")
    return None

# ============================================================================
# MAIN CONFORMER GENERATION FUNCTION
# ============================================================================

def generate_monomer_conformers(smiles, num_conformers=25, max_keep=15,
                                rmsd_threshold=0.5, known_ring_type=None,
                                center='mass', use_cremer_pople=True):
    """
    Generate different conformations for molecule monomer with Cremer-Pople analysis.
   
    Args:
        smiles: SMILES string
        num_conformers: Number of initial conformers to generate
        max_keep: Maximum number of conformers to keep
        rmsd_threshold: RMSD threshold for filtering similar conformers
        known_ring_type: Optional filter for ring type (e.g., 'chair', 'boat', 'twist', 'envelope')
                        If provided, ONLY conformers of this type will be kept
        center: 'mass' for center of mass, or 'geometric' for geometric center
        use_cremer_pople: If True, use Cremer-Pople analysis; if False, use simple heuristic
       
    Returns:
        Dictionary of {conformer_name: conformer_data} with puckering analysis
    
    Usage examples:
        # Generate all types with Cremer-Pople
        all_confs = generate_monomer_conformers(smiles)
        
        # Generate only chair conformers
        chairs = generate_monomer_conformers(smiles, known_ring_type='chair')
        
        # Generate only boats
        boats = generate_monomer_conformers(smiles, known_ring_type='boat')
        
        # Generate only twists
        twists = generate_monomer_conformers(smiles, known_ring_type='twist')
    """
   
    mol = Chem.MolFromSmiles(smiles)
    carbon_map = identify_sugar_carbons(mol) 
    if mol is None:
        print("Failed to parse SMILES")
        return []
   
    # Generate multiple conformations
    try:
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers,
                                randomSeed=42, clearConfs=True,
                                useExpTorsionAnglePrefs=True,
                                useBasicKnowledge=True,
                                enforceChirality=True,
                                useRandomCoords=False,
                                numZeroFail=2,
                                pruneRmsThresh=0.5,
                                maxAttempts=1000)
    except Exception as e:
        print(f"Conformer generation failed: {e}")
        return []
   
    print(f"Generated {mol.GetNumConformers()} initial conformers")
   
    # Optimize conformers
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    except:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500)
        except:
            print("Warning: Conformer optimization failed")

    # Evaluate and filter conformers
    conformers = []
    energies = []
    puckering_data = []
   
    for conf_id in range(mol.GetNumConformers()):
        try:
            # Calculate energy
            energy = None
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
                if ff is not None:
                    energy = ff.CalcEnergy()
            
            if energy is None:
                try:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    if ff is not None:
                        energy = ff.CalcEnergy()
                except:
                    pass
            
            if energy is None:
                if conf_id == 0:
                    energy = 0.0
                else:
                    try:
                        rmsd = AllChem.GetConformerRMS(mol, 0, conf_id)
                        energy = rmsd
                    except:
                        energy = float(conf_id)
            
            conf = mol.GetConformer(conf_id)
            
            # Analyze puckering
            if use_cremer_pople:
                # Use Cremer-Pople analysis
                puck_params = analyze_conformer_puckering_cremer_pople(mol, conf_id)
                
                if puck_params is None:
                    continue
                
                puckering_descriptor = puck_params['puckering_type']
                Q = puck_params['Q']
                theta = puck_params['theta']
                phi = puck_params['phi']
                
                # Filter by known_ring_type if specified
                if known_ring_type is not None:
                    # Extract base type (e.g., 'chair' from 'chair_4C1')
                    base_type = puckering_descriptor.split('_')[0]
                    if base_type.lower() != known_ring_type.lower():
                        continue
                
                # Create conformer name with Cremer-Pople parameters
                conf_name = f"{puckering_descriptor}_Q{Q:.3f}_theta{theta:.1f}_phi{phi:.1f}_E{energy:.1f}"
                puckering_amplitude = Q
                
            else:
                # Use simple heuristic (original method)
                ring_info = mol.GetRingInfo()
                puckering_descriptor = "unknown"
                puckering_amplitude = 0.0
                
                for ring in ring_info.AtomRings():
                    if len(ring) == 6:  # Pyranose ring
                        ring_coords = []
                        for atom_idx in ring:
                            pos = conf.GetAtomPosition(atom_idx)
                            ring_coords.append([pos.x, pos.y, pos.z])
                        
                        ring_coords = np.array(ring_coords)
                        centroid = np.mean(ring_coords, axis=0)
                        coords_centered = ring_coords - centroid
                        
                        v1 = coords_centered[1] - coords_centered[0]
                        v2 = coords_centered[2] - coords_centered[0]
                        normal = np.cross(v1, v2)
                        normal = normal / np.linalg.norm(normal)
                        
                        deviations = []
                        for coord in coords_centered:
                            deviation = abs(np.dot(coord, normal))
                            deviations.append(deviation)
                        
                        puckering_amplitude = np.std(deviations)
                        max_deviation = max(deviations)
                        
                        if max_deviation < 0.1:
                            puckering_descriptor = "planar"
                        elif max_deviation < 0.4:
                            sorted_devs = sorted(deviations)
                            if sorted_devs[-1] > 2.5 * sorted_devs[2]:
                                puckering_descriptor = "envelope"
                            else:
                                puckering_descriptor = "chair"
                        elif max_deviation < 0.7:
                            puckering_descriptor = "boat"
                        else:
                            puckering_descriptor = "twist"
                        break
                
                # Filter by known_ring_type if specified
                if known_ring_type is not None:
                    if puckering_descriptor.lower() != known_ring_type.lower():
                        continue
                
                # Create conformer name
                conf_name = f"{puckering_descriptor}_{conf_id}_P{puckering_amplitude:.2f}_E{energy:.1f}"
                puck_params = None
            
            conf_mol = copy.deepcopy(mol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
            conf_mol.SetProp("_Name", conf_name)
            
            conformers.append(conf_mol)
            energies.append(energy)
            puckering_data.append((puckering_descriptor, puckering_amplitude, puck_params))
            
        except Exception as e:
            print(f"Error processing conformer {conf_id}: {e}")
            continue
   
    if energies:
        # Filter by RMSD
        sorted_quads = sorted(zip(energies, conformers, puckering_data))
        filtered_conformers = []
        filtered_energies = []
        filtered_puckering = []

        for i, (energy, mol, puck_info) in enumerate(sorted_quads):
            is_duplicate = False
            
            for j, accepted_mol in enumerate(filtered_conformers):
                try:
                    rmsd = rdMolAlign.GetBestRMS(mol, accepted_mol)
                    if rmsd < rmsd_threshold:
                        is_duplicate = True
                        original_name = mol.GetProp("_Name")
                        puck_type = original_name.split('_')[0]
                        print(f"  Removing {puck_type} conformer: RMSD {rmsd:.2f} Å (similar to rank {j+1})")
                        break
                except Exception as e:
                    print(f"  Warning: RMSD calculation failed: {e}")
                    continue
            
            if not is_duplicate:
                filtered_conformers.append(mol)
                filtered_energies.append(energy)
                filtered_puckering.append(puck_info)
                if len(filtered_conformers) >= max_keep:
                    break
        
        print(f"Kept {len(filtered_conformers)} unique conformers out of {len(sorted_quads)} total")

        best_conformers = {}
        
        for rank, (energy, mol, puck_info) in enumerate(zip(filtered_energies, filtered_conformers, filtered_puckering)):
            puckering_descriptor, puckering_amplitude, puck_params = puck_info
            
            if use_cremer_pople and puck_params is not None:
                Q = puck_params['Q']
                theta = puck_params['theta']
                phi = puck_params['phi']
                final_name = f"{puckering_descriptor}_rank{rank+1}_Q{Q:.3f}_theta{theta:.1f}_phi{phi:.1f}_E{energy:.1f}"
            else:
                final_name = f"{puckering_descriptor}_rank{rank+1}_P{puckering_amplitude:.2f}_E{energy:.1f}"
            
            mol.SetProp("_Name", final_name)

            atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
            
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            coords = np.array(coords)
            masses = []
            for atom in mol.GetAtoms():
                masses.append(atom.GetMass())
            
            masses = np.array(masses)
            
            if center=='mass':
                com = np.average(coords, weights=masses, axis=0)
            else:
                com = np.mean(coords, axis=0)

            conf_data = {
                'rotation_center': com.tolist(),
                'COM': com.tolist(),
                'coordinates': coords.tolist(),
                'masses': masses.tolist(),
                'atom_types': atom_types,
                'carbon_map': carbon_map,
                'molecule': mol,
                'puckering_type': puckering_descriptor,
                'ring_puckering': float(puckering_amplitude),
                'energy': energy
            }
            
            # Add Cremer-Pople specific data if available
            if use_cremer_pople and puck_params is not None:
                conf_data.update({
                    'Q': float(puck_params['Q']),
                    'theta': float(puck_params['theta']),
                    'phi': float(puck_params['phi']),
                    'cremer_pople_data': puck_params
                })
            
            best_conformers[final_name] = conf_data
        
        print(f"\nSuccessfully processed {len(best_conformers)} conformers")
        print("Conformer Results:")
        print("="*80)
        for name, data in best_conformers.items():
            if use_cremer_pople and 'Q' in data:
                print(f"{name}")
                print(f"  Cremer-Pople: Q={data['Q']:.4f} Å, θ={data['theta']:.1f}°, φ={data['phi']:.1f}°")
                print(f"  Energy={data['energy']:.2f} kcal/mol")
            else:
                print(f"{name}")
                print(f"  Puckering amplitude={data['ring_puckering']:.4f}, Energy={data['energy']:.2f}")
        
        return best_conformers
    
    print("No valid conformers found")
    return {}

def save_monomer_solution(solution, filename, conf_name):
    coordinates = solution['coordinates']
    atom_types = solution['atom_types']
    
    with open(filename, 'w') as f:
        f.write(f"{len(coordinates)}\n")
        f.write(f"Monomer {conf_name} - Error: {solution['total_error']:.3f}Å\n")
        
        for coord, atom_type in zip(coordinates, atom_types):
            f.write(f"{atom_type:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
    
    print(f"✓ Saved monomer: {filename}")

def plot_conformer_analysis(conformer_data):
    """
    Plot conformer energies and structural differences (RMSD)
    """
    conformer_names = list(conformer_data.keys())
    energies = []
    molecules = []
    
    for name in conformer_names:
        # Extract energy from name
        energy_str = name.split('_E')[1]
        energies.append(float(energy_str))
        molecules.append(conformer_data[name]['molecule'])
    
    # Calculate RMSD matrix between all conformers
    n_conformers = len(molecules)
    rmsd_matrix = np.zeros((n_conformers, n_conformers))
    
    for i in range(n_conformers):
        for j in range(i+1, n_conformers):
            try:
                rmsd = rdMolAlign.GetBestRMS(molecules[i], molecules[j])
                rmsd_matrix[i][j] = rmsd
                rmsd_matrix[j][i] = rmsd
            except:
                rmsd_matrix[i][j] = 0
                rmsd_matrix[j][i] = 0
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    #RMSD heatmap
    im = ax.imshow(rmsd_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Conformer Index')
    ax.set_ylabel('Conformer Index')
    ax.set_title('RMSD Matrix (Å)')
    plt.colorbar(im, ax=ax)
    
    # Print analysis
    energy_range = max(energies) - min(energies)
    max_rmsd = np.max(rmsd_matrix)
    mean_rmsd = np.mean(rmsd_matrix[rmsd_matrix > 0])
    
    print(f"\nConformer Analysis:")
    print(f"Number of conformers: {len(energies)}")
    print(f"Energy range: {min(energies):.1f} to {max(energies):.1f} ({energy_range:.1f})")
    print(f"RMSD range: 0.0 to {max_rmsd:.1f} Å")
    print(f"Mean RMSD between conformers: {mean_rmsd:.1f} Å")
    
    # Identify most different conformers
    most_different_idx = np.unravel_index(np.argmax(rmsd_matrix), rmsd_matrix.shape)
    print(f"Most different pair: Conformer {most_different_idx[0]+1} vs {most_different_idx[1]+1} (RMSD: {rmsd_matrix[most_different_idx]:.1f} Å)")
    
    return fig, rmsd_matrix

def extract_rigid_monomer_data(conformers):
    """Extract whole dimer as rigid body with glycosidic bond as rotation center."""
    monomer_data = {}
   
    for conf_name, conf_data in conformers.items():
        mol = conf_data['molecule']  # Get the RDKit molecule
        
        if mol.GetNumConformers() == 0:
            continue
           
        conf = mol.GetConformer()
       
        # Get ALL atom positions and types (complete dimer)
        positions = []
        atom_types = []
       
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            atom_types.append(atom.GetSymbol())
       
        positions = np.array(positions)
       
        # Find COM rotation center, it is already stored on each conformer as 'COM'
        COM = np.array(conf_data['COM'])
       
        # Calculate coordinates relative to glycosidic center
        relative_coords = positions - COM
       
        # Calculate molecular orientation quaternion from glycosidic center
        cov_matrix = np.cov(relative_coords.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
       
        # Sort by eigenvalues (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
       
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
       
        # Convert to quaternion
        rotation = R.from_matrix(eigenvectors.T)
        quaternion = rotation.as_quat()  # [x, y, z, w]
   
       
        monomer_data[conf_name] = {
            'COM': COM.tolist(),
            'relative_coordinates': relative_coords.tolist(),
            'atom_types': atom_types,
            'quaternion': quaternion.tolist(),
            'carbon_map': conf_data.get('carbon_map', {})
        }
   
    return monomer_data

def save_translated_conformers(translated_data, base_filename, folder_name="translated_conformers"):
    """Save translated conformers to XYZ files in a specified folder."""
    
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    for conf_name, data in translated_data.items():
        coordinates = data['absolute_coordinates']
        atom_types = data['atom_types']
        com = data['COM']
        
        # Generate filename with folder path
        filename = os.path.join(folder_name, f"{base_filename}_{conf_name}.xyz")
        
        with open(filename, 'w') as f:
            f.write(f"{len(coordinates)}\n")
            f.write(f"Translated {conf_name} - COM: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]\n")
            
            for coord, atom_type in zip(coordinates, atom_types):
                f.write(f"{atom_type:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
        
        print(f"Saved translated conformer: {filename}")

def translate_to_experimental_com(rigid_body_data, experimental_com):
    """
    Translate rigid body from calculated COM to experimental COM position.
    
    Args:
        rigid_body_data: Dictionary from extract_rigid_monomer_data()
        experimental_com: Target COM position [x, y, z]
        
    Returns:
        Updated rigid body data with new COM and absolute coordinates
    """
    experimental_com = np.array(experimental_com)
    translated_data = {}
    
    for conf_name, data in rigid_body_data.items():
        current_com = np.array(data['COM'])
        relative_coords = np.array(data['relative_coordinates'])
        
        # Translation vector from current to experimental COM
        translation_vector = experimental_com - current_com
        
        # Calculate new absolute coordinates
        new_absolute_coords = relative_coords + experimental_com
        
        translated_data[conf_name] = {
            'COM': experimental_com.tolist(),
            'carbon_map': data.get('carbon_map', {}),
            'relative_coordinates': data['relative_coordinates'],  # Stay the same
            'absolute_coordinates': new_absolute_coords.tolist(),  # New positions
            'atom_types': data['atom_types'],
            'quaternion': data['quaternion'],  # Orientation unchanged
            'translation_vector': translation_vector.tolist()  # For reference
        }
    
    return translated_data

def save_translated_conformers_pickle(translated_conformers, molecule_name, folder="translated_pickles"):
    """Save translated conformers dictionary to pickle file."""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{molecule_name}_translated_conformers.pkl")
    
    with open(filename, 'wb') as f:
        pickle.dump(translated_conformers, f)
    
    print(f"Saved {molecule_name} translated conformers to {filename}")

def construct_com_to_carbon_vectors(translated_data):
    """Construct vectors from COM to ALL identified carbons (C1-C6)."""
    carbon_vectors_data = {}
   
    for conf_name, data in translated_data.items():
        com = np.array(data['COM'])
        absolute_coords = np.array(data['absolute_coordinates'])
        carbon_map = data['carbon_map']
       
        # Extract ALL numbered carbons
        identified_carbons = {}
        
        # Check for C1 through C6
        for i in range(1, 7):
            carbon_name = f'C{i}'
            if carbon_name in carbon_map:
                idx = carbon_map[carbon_name]
                coord = absolute_coords[idx]
                vector = coord - com
               
                identified_carbons[carbon_name] = {
                    'index': idx,
                    'coordinate': coord.tolist(),
                    'vector_from_com': vector.tolist(),
                    'distance_from_com': float(np.linalg.norm(vector))
                }
       
        carbon_vectors_data[conf_name] = {
            'COM': data['COM'],
            'carbons': identified_carbons  # C1, C2, C3, C4, C5, and C6 (if present)
        }
   
    return carbon_vectors_data

def visualize_complete_structure_with_vectors(translated_data, carbon_vectors, conf_name, mol):
    """
    Visualize complete molecular structure (C and O atoms with bonds) and COM-to-carbon vectors.
    
    Args:
        translated_data: Dictionary from translate_to_experimental_com()
        carbon_vectors: Dictionary from construct_com_to_carbon_vectors()
        conf_name: Name of conformer to visualize
        mol: RDKit molecule object to get bond information
    """
    # Get data
    mol_data = translated_data[conf_name]
    vector_data = carbon_vectors[conf_name]
    
    com = np.array(mol_data['COM'])
    all_coords = np.array(mol_data['absolute_coordinates'])
    atom_types = mol_data['atom_types']
    carbons = vector_data['carbons']
    
    # Get bonds from RDKit molecule
    bonds = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        bonds.append((idx1, idx2))
    
    # Create 3D plot
    fig = plt.figure(figsize=(18, 8))
    
    # Left plot: Complete structure with C and O atoms (no H)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Color map for atoms
    atom_colors = {'C': 'black', 'O': 'red', 'H': 'lightgray', 'N': 'blue'}
    atom_sizes = {'C': 120, 'O': 100, 'H': 30, 'N': 100}
    
    # Draw all bonds first
    for bond_idx1, bond_idx2 in bonds:
        atom1_type = atom_types[bond_idx1]
        atom2_type = atom_types[bond_idx2]
        
        # Skip H bonds for clarity, but draw all C-C, C-O, O-O bonds
        if atom1_type != 'H' and atom2_type != 'H':
            coord1 = all_coords[bond_idx1]
            coord2 = all_coords[bond_idx2]
            ax1.plot([coord1[0], coord2[0]], 
                    [coord1[1], coord2[1]], 
                    [coord1[2], coord2[2]], 
                    'gray', linewidth=2, alpha=0.5)
    
    # Plot all non-hydrogen atoms
    for i, (coord, atom_type) in enumerate(zip(all_coords, atom_types)):
        if atom_type != 'H':  # Skip hydrogens for clarity
            ax1.scatter(*coord, c=atom_colors.get(atom_type, 'gray'), 
                       s=atom_sizes.get(atom_type, 80), alpha=0.8,
                       edgecolors='black', linewidths=1)
            
            # Label atoms
            if atom_type == 'O':
                ax1.text(coord[0], coord[1], coord[2], 
                        f'  O', fontsize=9, color='darkred', alpha=0.7)
    
    # Plot COM
    ax1.scatter(*com, c='gold', s=300, marker='*', 
               label='COM', edgecolors='black', linewidths=2, zorder=100)
    
    # Plot vectors from COM to each carbon
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    for i, (carbon_name, c_data) in enumerate(sorted(carbons.items())):
        carbon_coord = np.array(c_data['coordinate'])
        vector = np.array(c_data['vector_from_com'])
        
        # Draw arrow from COM to carbon
        ax1.quiver(com[0], com[1], com[2],
                  vector[0], vector[1], vector[2],
                  color=colors[i % len(colors)], 
                  arrow_length_ratio=0.15, linewidth=3,
                  label=f'{carbon_name}', alpha=0.9)
        
        # Label the carbon
        ax1.text(carbon_coord[0], carbon_coord[1], carbon_coord[2], 
                f'  {carbon_name}', fontsize=12, fontweight='bold',
                color=colors[i % len(colors)])
    
    ax1.set_xlabel('X (Å)', fontsize=11)
    ax1.set_ylabel('Y (Å)', fontsize=11)
    ax1.set_zlabel('Z (Å)', fontsize=11)
    ax1.set_title(f'Rhamnose {conf_name}\nComplete Structure (C & O) with Vectors', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=20, azim=45)
    
    # Right plot: Different viewing angle
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Draw all bonds
    for bond_idx1, bond_idx2 in bonds:
        atom1_type = atom_types[bond_idx1]
        atom2_type = atom_types[bond_idx2]
        
        if atom1_type != 'H' and atom2_type != 'H':
            coord1 = all_coords[bond_idx1]
            coord2 = all_coords[bond_idx2]
            ax2.plot([coord1[0], coord2[0]], 
                    [coord1[1], coord2[1]], 
                    [coord1[2], coord2[2]], 
                    'gray', linewidth=2, alpha=0.5)
    
    # Plot all non-hydrogen atoms
    for i, (coord, atom_type) in enumerate(zip(all_coords, atom_types)):
        if atom_type != 'H':
            ax2.scatter(*coord, c=atom_colors.get(atom_type, 'gray'), 
                       s=atom_sizes.get(atom_type, 80), alpha=0.8,
                       edgecolors='black', linewidths=1)
            
            if atom_type == 'O':
                ax2.text(coord[0], coord[1], coord[2], 
                        f'  O', fontsize=9, color='darkred', alpha=0.7)
    
    # Plot COM
    ax2.scatter(*com, c='gold', s=300, marker='*', 
               label='COM', edgecolors='black', linewidths=2, zorder=100)
    
    # Plot vectors
    for i, (carbon_name, c_data) in enumerate(sorted(carbons.items())):
        carbon_coord = np.array(c_data['coordinate'])
        vector = np.array(c_data['vector_from_com'])
        distance = c_data['distance_from_com']
        
        ax2.quiver(com[0], com[1], com[2],
                  vector[0], vector[1], vector[2],
                  color=colors[i % len(colors)], 
                  arrow_length_ratio=0.15, linewidth=3,
                  label=f'{carbon_name} ({distance:.2f} Å)', alpha=0.9)
        
        ax2.text(carbon_coord[0], carbon_coord[1], carbon_coord[2], 
                f'  {carbon_name}', fontsize=12, fontweight='bold',
                color=colors[i % len(colors)])
    
    ax2.set_xlabel('X (Å)', fontsize=11)
    ax2.set_ylabel('Y (Å)', fontsize=11)
    ax2.set_zlabel('Z (Å)', fontsize=11)
    ax2.set_title(f'Rhamnose Structure - Rotated View\n(Black=C, Red=O)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.view_init(elev=60, azim=135)  # Different angle
    
    plt.tight_layout()
    plt.show()
    
    # Print vector information
    print(f"\n{conf_name} - Carbon Vector Analysis:")
    print("="*60)
    print(f"COM Position: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
    print()
    for carbon_name in sorted(carbons.keys()):
        c_data = carbons[carbon_name]
        print(f"{carbon_name}:")
        print(f"  Index: {c_data['index']}")
        print(f"  Coordinate: [{c_data['coordinate'][0]:.3f}, {c_data['coordinate'][1]:.3f}, {c_data['coordinate'][2]:.3f}]")
        print(f"  Distance from COM: {c_data['distance_from_com']:.3f} Å")
        print(f"  Vector: [{c_data['vector_from_com'][0]:.3f}, "
              f"{c_data['vector_from_com'][1]:.3f}, "
              f"{c_data['vector_from_com'][2]:.3f}]")
        print()

def select_conformer(translated_conformers_dict, strategy='lowest_energy'):
    """Select a conformer based on strategy."""
    conformer_keys = list(translated_conformers_dict.keys())
    
    if strategy == 'lowest_energy':
        return translated_conformers_dict[conformer_keys[0]]
    elif strategy == 'random':
        # Safely select random index within available conformers
        random_idx = random.randint(0, len(conformer_keys) - 1)
        return translated_conformers_dict[conformer_keys[random_idx]]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")