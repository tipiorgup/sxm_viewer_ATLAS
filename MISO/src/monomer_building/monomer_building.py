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
# from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole

# IPython display
# from IPython.display import Image

# RDKit configuration
# IPythonConsole.ipython_3d = True
print(f"RDKit version: {rdkit.__version__}")

# Initialize RDKit feature factory for H-bond detection
fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def view_mol(smiles):
    mol_kdo = Chem.MolFromSmiles(smiles)
    mol_kdo = Chem.AddHs(mol_kdo)
    AllChem.EmbedMolecule(mol_kdo)
    AllChem.MMFFOptimizeMolecule(mol_kdo)
    view = py3Dmol.view(width=500, height=400)
    sdf = Chem.MolToMolBlock(mol_kdo)
    view.addModel(sdf, 'sdf')
    view.setStyle({'stick': {'radius': 0.1}, 'sphere': {'radius': 0.3}})
    view.zoomTo()
    view.show()


##To do: First identifify 6-member rings. If they are all carbons it goes to identify_aromatic_ring_atoms, if there is an oxugen it goe to idneitgy_ dugar_atoms and if O>1 we need to behave as 
# identify aromagic_ring_carbos and use substituent as C1.

def identify_aromatic_ring_atoms(mol):
    """Identify atoms in a 6-membered all-carbon aromatic ring (e.g. benzene, anisole).

    Labels ring atoms C1-C6 so that C1 is the ring atom with the most
    heteroatom substituents (e.g. the C-OMe carbon in anisole).  This makes
    C1 the natural bonding anchor and keeps the returned dict compatible with
    the same ``carbon_map`` keys used by the rest of the pipeline.

    Returns:
        dict with keys C1-C6, all_ring_carbons, ring_type='aromatic', or
        an empty dict if no suitable ring is found.

    To-do:  Carbon 1 is te one with the most subsituents and between susbtituents
    If they are 2 it does not matter 
    Methoxy carbon we never bond, is free soul

    Numbering for dioxan:

    """
    ring_info = mol.GetRingInfo()
    ring_map = {}

    for ring in ring_info.AtomRings():
        if len(ring) != 6:
            continue

        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        # Must be an all-carbon ring
        if not all(a.GetSymbol() == 'C' for a in atoms):
            continue
        # At least one atom must be flagged aromatic
        if not any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue

        ring_set = set(ring)

        def _heteroatom_substituents(idx):
            return sum(
                1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() not in ring_set and n.GetSymbol() not in ('C', 'H')
            )

        def _total_substituents(idx):
            return sum(
                1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() not in ring_set
            )

        # C1 = ring atom with most heteroatom substituents; break ties by total subs
        sorted_ring = sorted(
            ring,
            key=lambda idx: (_heteroatom_substituents(idx), _total_substituents(idx)),
            reverse=True
        )
        start_idx = sorted_ring[0]

        # Walk ring from start_idx to produce a consistent ordering
        ordered = [start_idx]
        visited = {start_idx}
        current = start_idx
        while len(ordered) < 6:
            moved = False
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in ring_set and n_idx not in visited:
                    ordered.append(n_idx)
                    visited.add(n_idx)
                    current = n_idx
                    moved = True
                    break
            if not moved:
                break

        if len(ordered) != 6:
            continue

        for i, idx in enumerate(ordered, 1):
            ring_map[f'C{i}'] = idx

        ring_map['all_ring_carbons'] = list(ring)
        ring_map['ring_type'] = 'aromatic'
        # No ring_oxygen for aromatic rings; downstream code uses .get('ring_oxygen')
        break

    return ring_map


def identify_sugar_carbons(mol):
    """Identify ALL carbons in pyranose ring (C1-C5) plus C6 if present."""
    ring_info = mol.GetRingInfo()
    carbon_map = {}

    for ring in ring_info.AtomRings():
        if len(ring) == 6:
            ring_oxygen_idx = None
            for idx in ring:
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'O':
                    ring_oxygen_idx = idx
                    break

            if ring_oxygen_idx is None:
                continue

            ring_carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'C']
            ring_oxygen = mol.GetAtomWithIdx(ring_oxygen_idx)

            carbons_bonded_to_ring_o = []
            for neighbor in ring_oxygen.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    carbons_bonded_to_ring_o.append(neighbor.GetIdx())

            if len(carbons_bonded_to_ring_o) != 2:
                continue

            c1_candidate = carbons_bonded_to_ring_o[0]
            c5_candidate = carbons_bonded_to_ring_o[1]

            # Explicit override: atom map number :1 in the SMILES marks the anomeric carbon.
            # Needed for sugars like KDO where the oxygen-count heuristic is ambiguous.
            explicit_c1 = next(
                (idx for idx in carbons_bonded_to_ring_o
                 if mol.GetAtomWithIdx(idx).GetAtomMapNum() == 1),
                None
            )

            if explicit_c1 is not None:
                carbon_map['C1'] = explicit_c1
                carbon_map['C5'] = next(i for i in carbons_bonded_to_ring_o if i != explicit_c1)
            else:
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

            carbon_map['ring_oxygen'] = ring_oxygen_idx

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
                        elif neighbor.GetSymbol() == 'O':
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

            carbon_map['all_ring_carbons'] = ring_carbons
            break

    return carbon_map


# ============================================================================
# ANOMERIC CONFIGURATION (ALPHA/BETA) CLASSIFICATION
# ============================================================================

def find_anomeric_oxygen(mol, carbon_map):
    """
    Find the exocyclic oxygen bonded to C1 (the anomeric oxygen).

    This is the non-ring oxygen directly bonded to C1. In a free sugar this is
    the anomeric OH; in a glycoside it is the glycosidic oxygen.

    Args:
        mol: RDKit molecule (flat, topology only)
        carbon_map: output of identify_sugar_carbons()

    Returns:
        int: atom index of the anomeric oxygen, or None if not found
    """
    if 'C1' not in carbon_map:
        return None

    c1_idx = carbon_map['C1']
    ring_oxygen_idx = carbon_map.get('ring_oxygen')
    ring_atoms = set(carbon_map.get('all_ring_carbons', []))
    if ring_oxygen_idx is not None:
        ring_atoms.add(ring_oxygen_idx)

    c1_atom = mol.GetAtomWithIdx(c1_idx)
    anomeric_oxygen = None
    candidate_oxygens = []

    for neighbor in c1_atom.GetNeighbors():
        n_idx = neighbor.GetIdx()
        # Must be oxygen, must not be the ring oxygen, must not be in the ring
        if neighbor.GetSymbol() == 'O' and n_idx != ring_oxygen_idx and n_idx not in ring_atoms:
            candidate_oxygens.append(n_idx)

    if len(candidate_oxygens) == 1:
        anomeric_oxygen = candidate_oxygens[0]
    elif len(candidate_oxygens) > 1:
        # Prefer the oxygen with fewest heavy-atom neighbors (free OH over ether/ester)
        anomeric_oxygen = min(
            candidate_oxygens,
            key=lambda idx: sum(
                1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetSymbol() != 'H'
            )
        )

    return anomeric_oxygen

def compute_oh_map(mol, carbon_map):
    """Bond-graph hydroxyl lookup: carbon name -> (oxygen_idx, hydrogen_idx).

    For each mapped ring carbon ('C1', 'C2', ...) this finds the exocyclic
    hydroxyl oxygen bonded to it (using RDKit connectivity, not coordinates)
    and that oxygen's hydrogen. Ring oxygens and carbonyl (C=O double-bond)
    oxygens are excluded; when a carbon has several candidate oxygens the one
    with the fewest heavy neighbours (a free OH over an ether/ester) is chosen.

    Returns:
        dict mapping carbon name -> (oxygen_idx, hydrogen_idx). hydrogen_idx is
        None when no explicit H is present. Carbons with no exocyclic OH are
        omitted. Returns {} when carbon_map has no usable carbon entries.

    These indices are stable across the rigid-body pipeline (atom order is
    preserved by extract_rigid_monomer_data and translate_to_experimental_com),
    so the same map is valid against 'absolute_coordinates' downstream.
    """
    oh_map = {}
    ring_oxygen_idx = carbon_map.get('ring_oxygen')
    ring_atoms = set(carbon_map.get('all_ring_carbons', []))
    if ring_oxygen_idx is not None:
        ring_atoms.add(ring_oxygen_idx)

    for key, c_idx in carbon_map.items():
        # Only real carbon entries like 'C1', 'C5' — skip 'ring_oxygen',
        # 'all_ring_carbons', 'ring_type', etc.
        if not (key.startswith('C') and key[1:].isdigit()):
            continue

        c_atom = mol.GetAtomWithIdx(c_idx)
        best = None  # (n_heavy_neighbours, o_idx, h_idx)

        for neighbor in c_atom.GetNeighbors():
            if neighbor.GetSymbol() != 'O':
                continue
            o_idx = neighbor.GetIdx()
            if o_idx == ring_oxygen_idx or o_idx in ring_atoms:
                continue

            bond = mol.GetBondBetweenAtoms(c_idx, o_idx)
            if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                continue  # carbonyl oxygen, not a hydroxyl

            h_idx = None
            n_heavy = 0
            for o_nbr in neighbor.GetNeighbors():
                if o_nbr.GetSymbol() == 'H':
                    if h_idx is None:
                        h_idx = o_nbr.GetIdx()
                else:
                    n_heavy += 1

            candidate = (n_heavy, o_idx, h_idx)
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is not None:
            oh_map[key] = (best[1], best[2])

    return oh_map

def classify_anomeric_configuration(mol, conf_id, carbon_map, ring_normal, ring_centroid):
    """
    Classify the anomeric configuration as alpha or beta from 3D coordinates.

    Convention (Haworth):
        The ring normal is oriented to point toward the same side as C5
        (i.e. toward the CH2OH group). This is the reference "up" direction.
        beta  -> anomeric OH is on the same side as C5 (dot product > 0)
        alpha -> anomeric OH is on the opposite side      (dot product < 0)

    Args:
        mol: RDKit molecule
        conf_id: conformer ID
        carbon_map: output of identify_sugar_carbons()
        ring_normal: unit normal vector of ring plane (numpy array, shape (3,))
                     as returned by calculate_cremer_pople_parameters (before orientation fix)
        ring_centroid: centroid of ring atoms (numpy array, shape (3,))

    Returns:
        dict with keys:
            'anomer': 'alpha', 'beta', or 'unknown'
            'dot_product': raw dot product value (float)
            'anomeric_oxygen_idx': atom index of the anomeric oxygen (int or None)
    """
    result = {
        'anomer': 'unknown',
        'dot_product': None,
        'anomeric_oxygen_idx': None
    }

        # Aromatic monomers have no anomeric centre
    if carbon_map.get('ring_type') == 'aromatic':
        result['anomer'] = 'not_applicable'
        return result
    
    if 'C1' not in carbon_map or 'C5' not in carbon_map:
        return result

    anomeric_o_idx = find_anomeric_oxygen(mol, carbon_map)
    if anomeric_o_idx is None:
        return result

    result['anomeric_oxygen_idx'] = anomeric_o_idx

    conf = mol.GetConformer(conf_id)

    c1_idx = carbon_map['C1']
    c5_idx = carbon_map['C5']

    c1_pos = np.array(conf.GetAtomPosition(c1_idx))
    c5_pos = np.array(conf.GetAtomPosition(c5_idx))
    anom_o_pos = np.array(conf.GetAtomPosition(anomeric_o_idx))

    # FIX Issue 4: orient the ring normal consistently toward C5
    # Vector from ring centroid to C5
    centroid_to_c5 = c5_pos - ring_centroid
    oriented_normal = ring_normal.copy()
    if np.dot(oriented_normal, centroid_to_c5) < 0:
        oriented_normal = -oriented_normal

    # Vector from C1 to anomeric oxygen
    c1_to_anom_o = anom_o_pos - c1_pos

    dot = float(np.dot(c1_to_anom_o, oriented_normal))
    result['dot_product'] = dot

    # Positive -> anomeric O on C5 side -> beta
    # Negative -> anomeric O on opposite side -> alpha
    result['anomer'] = 'beta' if dot > 0 else 'alpha'

    return result


# ============================================================================
# CREMER-POPLE PUCKERING ANALYSIS FUNCTIONS
# ============================================================================

def calculate_cremer_pople_parameters(ring_coords):
    """
    Cremer-Pople puckering parameters for a 6-membered ring.
    Based on Cremer & Pople (1975).

    Args:
        ring_coords: numpy array of shape (6, 3) with ring atom coordinates

    Returns:
        Dictionary with Q, theta, phi, puckering_type, ring_normal, ring_centroid
        (ring_normal and ring_centroid are new additions needed for alpha/beta)
    """
    if ring_coords.shape != (6, 3):
        raise ValueError("Ring coordinates must be (6, 3)")

    N = 6

    center = np.mean(ring_coords, axis=0)
    centered_positions = ring_coords - center

    # SVD to find best-fit plane normal
    U, S, Vt = np.linalg.svd(centered_positions, full_matrices=False)
    normal = Vt[-1]  # unit normal (arbitrary sign)

    z_j = np.array([np.dot(pos, normal) for pos in centered_positions])

    q_values = {}
    for m in range(2, N):
        q_m_real = 0
        q_m_imag = 0
        for j in range(N):
            phi_jm = 2 * np.pi * j * m / N
            q_m_real += z_j[j] * np.cos(phi_jm)
            q_m_imag += z_j[j] * np.sin(phi_jm)
        normalization = np.sqrt(2.0 / N)
        q_values[m] = {
            'real': q_m_real * normalization,
            'imag': q_m_imag * normalization,
            'magnitude': np.sqrt(q_m_real**2 + q_m_imag**2) * normalization
        }

    q2 = q_values[2]['real']
    q3 = q_values[3]['real']
    q4 = q_values[4]['real']
    q5 = q_values[5]['real']

    q2_mag = q_values[2]['magnitude']
    q3_mag = q_values[3]['magnitude']
    q4_mag = q_values[4]['magnitude']
    q5_mag = q_values[5]['magnitude']

    Q = np.sqrt(q2_mag**2 + q3_mag**2 + q4_mag**2 + q5_mag**2)

    if Q > 1e-6:
        cos_theta = q3_mag / Q
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        if q3 < 0:
            theta_deg = 180 - theta_deg
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
        # FIX Issue 2: expose ring_normal and ring_centroid for alpha/beta use
        'ring_normal': normal,       # raw SVD normal, sign not yet fixed
        'ring_centroid': center,     # centroid of ring atoms in original coordinates
        'debug_info': {
            'z_displacements': z_j.tolist(),
            'q_values': q_values
        }
    }

def classify_puckering(Q, theta_deg, phi_deg):
    """
    Classify 6-membered ring conformation based on Cremer-Pople parameters.
    """
    if Q < 0.1:
        return "planar"

    if theta_deg < 15:
        return "chair_4C1"
    elif theta_deg > 165:
        return "chair_1C4"
    elif 75 <= theta_deg <= 105:
        phi_normalized = phi_deg % 180
        if phi_normalized < 30 or phi_normalized > 150:
            return "boat"
        elif 60 <= phi_normalized <= 120:
            return "skew_boat"
        else:
            return "boat_intermediate"
    elif (35 <= theta_deg <= 65) or (115 <= theta_deg <= 145):
        return "half_chair"
    elif (15 <= theta_deg <= 35) or (145 <= theta_deg <= 165):
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
        return "twist_intermediate"

def analyze_conformer_puckering_cremer_pople(mol, conf_id, carbon_map=None):
    """
    Analyze single conformer puckering using Cremer-Pople, and classify
    anomeric configuration if carbon_map is provided.

    FIX Issue 5: carbon_map is now accepted as an argument.

    Args:
        mol: RDKit molecule
        conf_id: conformer ID
        carbon_map: output of identify_sugar_carbons() (optional)
                    if provided, alpha/beta classification is performed

    Returns:
        Dictionary with Cremer-Pople parameters, classification, and anomer
    """
    conf = mol.GetConformer(conf_id)
    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if len(ring) == 6:
            ring_coords = []
            for atom_idx in ring:
                pos = conf.GetAtomPosition(atom_idx)
                ring_coords.append([pos.x, pos.y, pos.z])

            ring_coords = np.array(ring_coords)

            try:
                result = calculate_cremer_pople_parameters(ring_coords)

                # FIX Issue 5: perform alpha/beta classification when carbon_map available
                if carbon_map is not None:
                    anomer_data = classify_anomeric_configuration(
                        mol,
                        conf_id,
                        carbon_map,
                        result['ring_normal'],
                        result['ring_centroid']
                    )
                    result['anomer'] = anomer_data['anomer']
                    result['anomer_dot_product'] = anomer_data['dot_product']
                    result['anomeric_oxygen_idx'] = anomer_data['anomeric_oxygen_idx']
                else:
                    result['anomer'] = 'unknown'
                    result['anomer_dot_product'] = None
                    result['anomeric_oxygen_idx'] = None

                print(
                    f"  Conformer {conf_id}: Q={result['Q']:.3f}, "
                    f"theta={result['theta']:.1f}, phi={result['phi']:.1f} "
                    f"-> {result['puckering_type']} | anomer={result['anomer']}"
                )
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
                                known_anomer=None,
                                center='mass', use_cremer_pople=True):
    """
    Generate different conformations for molecule monomer with Cremer-Pople analysis
    and anomeric (alpha/beta) classification.

    Args:
        smiles: SMILES string
        num_conformers: Number of initial conformers to generate
        max_keep: Maximum number of conformers to keep
        rmsd_threshold: RMSD threshold for filtering similar conformers
        known_ring_type: Optional filter for ring type ('chair', 'boat', 'twist', 'envelope')
        known_anomer: Optional filter for anomeric configuration ('alpha' or 'beta')
        center: 'mass' for center of mass, or 'geometric' for geometric center
        use_cremer_pople: If True, use Cremer-Pople analysis; if False, simple heuristic

    Returns:
        Dictionary of {conformer_name: conformer_data} with puckering and anomer analysis
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Failed to parse SMILES")
        return []

    # FIX Issue 1: carbon_map is computed on flat mol (topology), atom indices
    # are stable across conformer generation so this is correct.
    carbon_map = identify_sugar_carbons(mol)
    if not carbon_map:
        carbon_map = identify_aromatic_ring_atoms(mol)
    monomer_type = carbon_map.get('ring_type', 'sugar')

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

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    except:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500)
        except:
            print("Warning: Conformer optimization failed")

    conformers = []
    energies = []
    puckering_data = []

    for conf_id in range(mol.GetNumConformers()):
        try:
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

            if use_cremer_pople:
                # FIX Issue 5: pass carbon_map into puckering analysis
                puck_params = analyze_conformer_puckering_cremer_pople(mol, conf_id, carbon_map)

                if puck_params is None:
                    continue

                puckering_descriptor = puck_params['puckering_type']
                Q = puck_params['Q']
                theta = puck_params['theta']
                phi = puck_params['phi']
                anomer = puck_params['anomer']

                # Aromatic rings are always planar; override label for clarity
                if monomer_type == 'aromatic':
                    anomer = 'not_applicable'
                    puckering_descriptor = 'planar'

                if known_ring_type is not None:
                    base_type = puckering_descriptor.split('_')[0]
                    if base_type.lower() != known_ring_type.lower():
                        continue

                # known_anomer filter is meaningless for aromatic monomers
                if known_anomer is not None and monomer_type != 'aromatic':
                    if anomer.lower() != known_anomer.lower():
                        continue

                # Anomeric label included in conformer name
                conf_name = (
                    f"{puckering_descriptor}_{anomer}"
                    f"_Q{Q:.3f}_theta{theta:.1f}_phi{phi:.1f}_E{energy:.1f}"
                )
                puckering_amplitude = Q

            else:
                ring_info = mol.GetRingInfo()
                puckering_descriptor = "unknown"
                puckering_amplitude = 0.0
                anomer = 'unknown'

                for ring in ring_info.AtomRings():
                    if len(ring) == 6:
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

                if known_ring_type is not None:
                    if puckering_descriptor.lower() != known_ring_type.lower():
                        continue

                conf_name = f"{puckering_descriptor}_{conf_id}_P{puckering_amplitude:.2f}_E{energy:.1f}"
                puck_params = None

            conf_mol = copy.deepcopy(mol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
            conf_mol.SetProp("_Name", conf_name)

            conformers.append(conf_mol)
            energies.append(energy)
            puckering_data.append((puckering_descriptor, puckering_amplitude, puck_params, anomer))

        except Exception as e:
            print(f"Error processing conformer {conf_id}: {e}")
            continue

    if energies:
        sorted_quads = sorted(zip(energies, conformers, puckering_data))
        filtered_conformers = []
        filtered_energies = []
        filtered_puckering = []

        for i, (energy, mol_conf, puck_info) in enumerate(sorted_quads):
            is_duplicate = False

            for j, accepted_mol in enumerate(filtered_conformers):
                try:
                    rmsd = rdMolAlign.GetBestRMS(mol_conf, accepted_mol)
                    if rmsd < rmsd_threshold:
                        is_duplicate = True
                        original_name = mol_conf.GetProp("_Name")
                        puck_type = original_name.split('_')[0]
                        print(f"  Removing {puck_type} conformer: RMSD {rmsd:.2f} (similar to rank {j+1})")
                        break
                except Exception as e:
                    print(f"  Warning: RMSD calculation failed: {e}")
                    continue

            if not is_duplicate:
                filtered_conformers.append(mol_conf)
                filtered_energies.append(energy)
                filtered_puckering.append(puck_info)
                if len(filtered_conformers) >= max_keep:
                    break

        print(f"Kept {len(filtered_conformers)} unique conformers out of {len(sorted_quads)} total")

        best_conformers = {}

        for rank, (energy, mol_conf, puck_info) in enumerate(
            zip(filtered_energies, filtered_conformers, filtered_puckering)
        ):
            puckering_descriptor, puckering_amplitude, puck_params, anomer = puck_info

            if use_cremer_pople and puck_params is not None:
                Q = puck_params['Q']
                theta = puck_params['theta']
                phi = puck_params['phi']
                final_name = (
                    f"{puckering_descriptor}_{anomer}"
                    f"_rank{rank+1}_Q{Q:.3f}_theta{theta:.1f}_phi{phi:.1f}_E{energy:.1f}"
                )
            else:
                final_name = f"{puckering_descriptor}_rank{rank+1}_P{puckering_amplitude:.2f}_E{energy:.1f}"

            mol_conf.SetProp("_Name", final_name)

            atom_types = [mol_conf.GetAtomWithIdx(i).GetSymbol() for i in range(mol_conf.GetNumAtoms())]

            conf = mol_conf.GetConformer()
            coords = []
            for i in range(mol_conf.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])

            coords = np.array(coords)
            masses = np.array([atom.GetMass() for atom in mol_conf.GetAtoms()])

            if center == 'mass':
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
                'molecule': mol_conf,
                'puckering_type': puckering_descriptor,
                'ring_puckering': float(puckering_amplitude),
                'energy': energy,
                'anomer': anomer,
                'monomer_type': monomer_type
            }

            if use_cremer_pople and puck_params is not None:
                conf_data.update({
                    'Q': float(puck_params['Q']),
                    'theta': float(puck_params['theta']),
                    'phi': float(puck_params['phi']),
                    'cremer_pople_data': puck_params,
                    'anomer_dot_product': puck_params.get('anomer_dot_product'),
                    'anomeric_oxygen_idx': puck_params.get('anomeric_oxygen_idx'),
                })

            best_conformers[final_name] = conf_data

        print(f"\nSuccessfully processed {len(best_conformers)} conformers")
        print("Conformer Results:")
        print("=" * 80)
        for name, data in best_conformers.items():
            if use_cremer_pople and 'Q' in data:
                print(f"{name}")
                print(f"  Cremer-Pople: Q={data['Q']:.4f}, theta={data['theta']:.1f}, phi={data['phi']:.1f}")
                print(f"  Anomer: {data['anomer']}")
                print(f"  Energy={data['energy']:.2f} kcal/mol")
            else:
                print(f"{name}")
                print(f"  Puckering amplitude={data['ring_puckering']:.4f}, Energy={data['energy']:.2f}")

        return best_conformers

    print("No valid conformers found")
    return {}


# ============================================================================
# ALL DOWNSTREAM FUNCTIONS UNCHANGED
# ============================================================================

def save_monomer_solution(solution, filename, conf_name):
    coordinates = solution['coordinates']
    atom_types = solution['atom_types']

    with open(filename, 'w') as f:
        f.write(f"{len(coordinates)}\n")
        f.write(f"Monomer {conf_name} - Error: {solution['total_error']:.3f}\n")
        for coord, atom_type in zip(coordinates, atom_types):
            f.write(f"{atom_type:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

    print(f"Saved monomer: {filename}")


def plot_conformer_analysis(conformer_data):
    conformer_names = list(conformer_data.keys())
    energies = []
    molecules = []

    for name in conformer_names:
        energy_str = name.split('_E')[1]
        energies.append(float(energy_str))
        molecules.append(conformer_data[name]['molecule'])

    n_conformers = len(molecules)
    rmsd_matrix = np.zeros((n_conformers, n_conformers))

    for i in range(n_conformers):
        for j in range(i + 1, n_conformers):
            try:
                rmsd = rdMolAlign.GetBestRMS(molecules[i], molecules[j])
                rmsd_matrix[i][j] = rmsd
                rmsd_matrix[j][i] = rmsd
            except:
                rmsd_matrix[i][j] = 0
                rmsd_matrix[j][i] = 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(rmsd_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Conformer Index')
    ax.set_ylabel('Conformer Index')
    ax.set_title('RMSD Matrix')
    plt.colorbar(im, ax=ax)

    energy_range = max(energies) - min(energies)
    max_rmsd = np.max(rmsd_matrix)
    mean_rmsd = np.mean(rmsd_matrix[rmsd_matrix > 0])

    print(f"\nConformer Analysis:")
    print(f"Number of conformers: {len(energies)}")
    print(f"Energy range: {min(energies):.1f} to {max(energies):.1f} ({energy_range:.1f})")
    print(f"RMSD range: 0.0 to {max_rmsd:.1f}")
    print(f"Mean RMSD between conformers: {mean_rmsd:.1f}")

    most_different_idx = np.unravel_index(np.argmax(rmsd_matrix), rmsd_matrix.shape)
    print(f"Most different pair: Conformer {most_different_idx[0]+1} vs {most_different_idx[1]+1} "
          f"(RMSD: {rmsd_matrix[most_different_idx]:.1f})")

    return fig, rmsd_matrix


def extract_rigid_monomer_data(conformers):
    monomer_data = {}

    for conf_name, conf_data in conformers.items():
        mol = conf_data['molecule']

        if mol.GetNumConformers() == 0:
            continue

        conf = mol.GetConformer()
        positions = []
        atom_types = []

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            atom_types.append(atom.GetSymbol())

        positions = np.array(positions)
        COM = np.array(conf_data['COM'])
        relative_coords = positions - COM

        cov_matrix = np.cov(relative_coords.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1

        rotation = R.from_matrix(eigenvectors.T)
        quaternion = rotation.as_quat()

        monomer_data[conf_name] = {
            'COM': COM.tolist(),
            'relative_coordinates': relative_coords.tolist(),
            'atom_types': atom_types,
            'quaternion': quaternion.tolist(),
            'carbon_map': conf_data.get('carbon_map', {}),
            'anomer': conf_data.get('anomer', 'unknown'),
            # Bond-graph hydroxyl lookup computed while the RDKit mol is still
            # in hand; consumed by bonds_creation instead of distance-guessing.
            'oh_map': compute_oh_map(mol, conf_data.get('carbon_map', {})),
            'anomeric_oxygen_idx': conf_data.get('anomeric_oxygen_idx'),
        }

    return monomer_data


def save_translated_conformers(translated_data, base_filename, folder_name="translated_conformers"):
    os.makedirs(folder_name, exist_ok=True)

    for conf_name, data in translated_data.items():
        coordinates = data['absolute_coordinates']
        atom_types = data['atom_types']
        com = data['COM']

        filename = os.path.join(folder_name, f"{base_filename}_{conf_name}.xyz")

        with open(filename, 'w') as f:
            f.write(f"{len(coordinates)}\n")
            f.write(f"Translated {conf_name} - COM: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]\n")
            for coord, atom_type in zip(coordinates, atom_types):
                f.write(f"{atom_type:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

        print(f"Saved translated conformer: {filename}")


def translate_to_experimental_com(rigid_body_data, experimental_com):
    experimental_com = np.array(experimental_com)
    translated_data = {}

    for conf_name, data in rigid_body_data.items():
        current_com = np.array(data['COM'])
        relative_coords = np.array(data['relative_coordinates'])
        translation_vector = experimental_com - current_com
        new_absolute_coords = relative_coords + experimental_com

        translated_data[conf_name] = {
            'COM': experimental_com.tolist(),
            'carbon_map': data.get('carbon_map', {}),
            'relative_coordinates': data['relative_coordinates'],
            'absolute_coordinates': new_absolute_coords.tolist(),
            'atom_types': data['atom_types'],
            'quaternion': data['quaternion'],
            'translation_vector': translation_vector.tolist(),
            'anomer': data.get('anomer', 'unknown'),
            'oh_map': data.get('oh_map', {}),
            'anomeric_oxygen_idx': data.get('anomeric_oxygen_idx'),
        }

    return translated_data


def save_translated_conformers_pickle(translated_conformers, molecule_name, folder="translated_pickles"):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{molecule_name}_translated_conformers.pkl")

    with open(filename, 'wb') as f:
        pickle.dump(translated_conformers, f)

    print(f"Saved {molecule_name} translated conformers to {filename}")


def construct_com_to_carbon_vectors(translated_data):
    carbon_vectors_data = {}

    for conf_name, data in translated_data.items():
        com = np.array(data['COM'])
        absolute_coords = np.array(data['absolute_coordinates'])
        carbon_map = data['carbon_map']

        identified_carbons = {}
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
            'carbons': identified_carbons
        }

    return carbon_vectors_data


def visualize_complete_structure_with_vectors(translated_data, carbon_vectors, conf_name, mol):
    mol_data = translated_data[conf_name]
    vector_data = carbon_vectors[conf_name]

    com = np.array(mol_data['COM'])
    all_coords = np.array(mol_data['absolute_coordinates'])
    atom_types = mol_data['atom_types']
    carbons = vector_data['carbons']

    bonds = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        bonds.append((idx1, idx2))

    fig = plt.figure(figsize=(18, 8))

    atom_colors = {'C': 'black', 'O': 'red', 'H': 'lightgray', 'N': 'blue'}
    atom_sizes = {'C': 120, 'O': 100, 'H': 30, 'N': 100}
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for ax_idx, (elev, azim, title_suffix) in enumerate([
        (20, 45, 'Complete Structure (C & O) with Vectors'),
        (60, 135, 'Rotated View (Black=C, Red=O)')
    ]):
        ax = fig.add_subplot(1, 2, ax_idx + 1, projection='3d')

        for bond_idx1, bond_idx2 in bonds:
            if atom_types[bond_idx1] != 'H' and atom_types[bond_idx2] != 'H':
                coord1 = all_coords[bond_idx1]
                coord2 = all_coords[bond_idx2]
                ax.plot([coord1[0], coord2[0]],
                        [coord1[1], coord2[1]],
                        [coord1[2], coord2[2]],
                        'gray', linewidth=2, alpha=0.5)

        for i, (coord, atom_type) in enumerate(zip(all_coords, atom_types)):
            if atom_type != 'H':
                ax.scatter(*coord, c=atom_colors.get(atom_type, 'gray'),
                           s=atom_sizes.get(atom_type, 80), alpha=0.8,
                           edgecolors='black', linewidths=1)
                if atom_type == 'O':
                    ax.text(coord[0], coord[1], coord[2], '  O', fontsize=9, color='darkred', alpha=0.7)

        ax.scatter(*com, c='gold', s=300, marker='*',
                   label='COM', edgecolors='black', linewidths=2, zorder=100)

        for i, (carbon_name, c_data) in enumerate(sorted(carbons.items())):
            carbon_coord = np.array(c_data['coordinate'])
            vector = np.array(c_data['vector_from_com'])
            ax.quiver(com[0], com[1], com[2],
                      vector[0], vector[1], vector[2],
                      color=colors[i % len(colors)],
                      arrow_length_ratio=0.15, linewidth=3,
                      label=f'{carbon_name}', alpha=0.9)
            ax.text(carbon_coord[0], carbon_coord[1], carbon_coord[2],
                    f'  {carbon_name}', fontsize=12, fontweight='bold',
                    color=colors[i % len(colors)])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{conf_name}\n{title_suffix}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()

    print(f"\n{conf_name} - Carbon Vector Analysis:")
    print("=" * 60)
    print(f"COM Position: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
    print()
    for carbon_name in sorted(carbons.keys()):
        c_data = carbons[carbon_name]
        print(f"{carbon_name}:")
        print(f"  Index: {c_data['index']}")
        print(f"  Coordinate: [{c_data['coordinate'][0]:.3f}, {c_data['coordinate'][1]:.3f}, {c_data['coordinate'][2]:.3f}]")
        print(f"  Distance from COM: {c_data['distance_from_com']:.3f}")
        print(f"  Vector: [{c_data['vector_from_com'][0]:.3f}, "
              f"{c_data['vector_from_com'][1]:.3f}, "
              f"{c_data['vector_from_com'][2]:.3f}]")
        print()


def select_conformer(translated_conformers_dict, strategy='lowest_energy'):
    conformer_keys = list(translated_conformers_dict.keys())

    if strategy == 'lowest_energy':
        return translated_conformers_dict[conformer_keys[0]]
    elif strategy == 'random':
        random_idx = random.randint(0, len(conformer_keys) - 1)
        return translated_conformers_dict[conformer_keys[random_idx]]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")