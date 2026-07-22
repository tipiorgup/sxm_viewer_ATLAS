from rdkit import Chem
from .bonds_creation import find_hydroxyl_oxygen_at_carbon
from ..geometry.geometry_utils import (
    get_perpendicular_vector,
    get_positions,
    set_positions,
    get_positions_for_atoms,
    calculate_alignment_rotation,
    get_ring_normal_from_positions,
)
import numpy as np
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmarts

#To do: build extra alognemnt for rings or big fucntional groups with blob additional info
#Check glycosidic bond 

CYCLIC_PEPTIDE_SMILES = {
    'Pro-Gly-Leu-Asp-Thr-Linker': 'CC[C@H]1c2nnc(o2)-c2cccc(c2)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)CNC(=O)[C@@H]2CCCN21',
}

CA_SMARTS = {
    'Pro':    '[C@@H]1CCCN1',           # chiral C in pyrrolidine
    'Gly': '[CH2](-N)(-C)',       # CH2 between N and C, non-ring
    'Leu':    '[C@@H](CC(C)C)',         # chiral C bearing isobutyl
    'Asp':    '[C@@H](CC(=O)O)',        # chiral C bearing CH2COOH
    'Thr':    '[C@@H]([C@@H](C)O)',     # chiral C bearing CHOH-CH3
    'Linker': '[C@@H](CC)c1nnco1',     # chiral C bearing ethyl and oxadiazole
}

AA_FUNCTIONAL_GROUPS = {
    # Non-polar, no functional group
    'Ala': None, 'Val': None, 'Leu': None, 'Ile': None, 'Met': None, 'Gly': None,
    
    # Rings (track centroid)
    'Pro': 'ring_centroid',
    'Phe': 'ring_centroid',
    'Trp': 'ring_centroid',
    'His': 'ring_centroid',
    'Linker': 'ring_centroid',
    
    # Hydroxyl groups
    'Ser': 'hydroxyl_o',
    'Thr': 'hydroxyl_o',
    'Tyr': 'hydroxyl_o',
    
    # Thiol
    'Cys': 'thiol_s',
    
    # Amides
    'Asn': 'amide_n',
    'Gln': 'amide_n',
    
    # Carboxyl
    'Asp': 'carboxyl_c',
    'Glu': 'carboxyl_c',
    
    # Basic groups
    'Lys': 'amine_n',
    'Arg': 'guanidinium_c',
}

# ============================================================================
# CONSTANTS - Bond Lengths (Å)
# ============================================================================

BOND_LENGTHS = {
    'C_C': 1.54,      # C-C single bond
    'C_N': 1.47,      # C-N single bond
    'C_O': 1.43,      # C-O single bond
    'O_C': 1.43,      # O-C single bond (same as C-O)
    'O_P': 1.60,      # O-P single bond
    'P_O_SINGLE': 1.57,  # P-O single bond
    'P_O_DOUBLE': 1.48,  # P=O double bond
    'C_O_DOUBLE': 1.23,  # C=O double bond (carbonyl)
}

# Angle constants (degrees)
ANGLES = {
    'TETRAHEDRAL': 109.47,
    'TRIGONAL': 120.0,
    'BETA_BEND': 5,      # β-anomeric slight bend
    'ALPHA_BEND': 23,    # α-anomeric axial offset
    'PETN_BEND_1': 10,
    'PETN_BEND_2': 15,
    'N_ACETYL_ANGLE': 110,
    'TRANS_DIHEDRAL': 180,
}

# ============================================================================
# AMINO ACID GLYCOSYLATION SITES
# ============================================================================

AA_GLYCOSYLATION_INFO = {
    'Asn': {
        'codes': ['N', 'Asn', 'ASN'],
        'glycosylation_type': 'N-glycosidic',
        'atom_name': 'N_delta',
        'requires_spacer': True,
        'h_to_remove': 1,
        'description': 'Side chain amide nitrogen'
    },
    'Ser': {
        'codes': ['S', 'Ser', 'SER'],
        'glycosylation_type': 'O-glycosidic',
        'atom_name': 'O_gamma',
        'requires_spacer': False,
        'h_to_remove': 1,
        'description': 'Hydroxyl oxygen'
    },
    'Thr': {
        'codes': ['T', 'Thr', 'THR'],
        'glycosylation_type': 'O-glycosidic',
        'atom_name': 'O_gamma1',
        'requires_spacer': False,
        'h_to_remove': 1,
        'description': 'Hydroxyl oxygen'
    },
    'Tyr': {
        'codes': ['Y', 'Tyr', 'TYR'],
        'glycosylation_type': 'O-glycosidic',
        'atom_name': 'O_eta',
        'requires_spacer': False,
        'h_to_remove': 1,
        'description': 'Phenolic hydroxyl'
    },
    'Trp': {
        'codes': ['W', 'Trp', 'TRP'],
        'glycosylation_type': 'C-glycosidic',
        'atom_name': 'C2_indole',
        'requires_spacer': False,
        'h_to_remove': 1,
        'description': 'Indole C2 carbon'
    }
}

# Reverse lookup: code -> canonical name
AA_CODE_TO_NAME = {}
for canonical_name, info in AA_GLYCOSYLATION_INFO.items():
    for code in info['codes']:
        AA_CODE_TO_NAME[code] = canonical_name

def build_peptide_with_rdkit_ca(aa_sequence, residue_data, cyclic=False, linker_data=None):
    """
    Build peptide with RDKit positioned at Cα + functional group coordinates.
    
    Parameters:
    - aa_sequence: 'N-F-A' or ['Asn', 'Phe', 'Ala']
    - residue_data: List of dicts with keys:
        {
            'aa': 'Asn',
            'ca_position': [x, y, z],
            'functional_position': [x, y, z] or None
        }
    """
    
    if isinstance(aa_sequence, str):
        aa_map = {'N': 'Asn', 'F': 'Phe', 'A': 'Ala', 'S': 'Ser', 'T': 'Thr', 
                  'Y': 'Tyr', 'W': 'Trp', 'V': 'Val', 'L': 'Leu', 'I': 'Ile',
                  'M': 'Met', 'P': 'Pro', 'G': 'Gly', 'C': 'Cys', 'D': 'Asp',
                  'E': 'Glu', 'K': 'Lys', 'R': 'Arg', 'H': 'His', 'Q': 'Gln'}
        aa_sequence = [aa_map.get(code, code) for code in aa_sequence.split('-')]
    
    print(f"\n{'='*60}")
    print(f"Building peptide with RDKit: {'-'.join(aa_sequence)}")
    print(f"Using {len(residue_data)} residues with Cα + functional positions")

    # For large linear chains the per-residue whole-peptide embed (below) gets
    # expensive and RDKit's embedder silently fails (returns -1, no conformer).
    # Switch to a cheap per-residue placement strategy: never embed the full
    # molecule, just place each residue's individually-embedded geometry at its
    # experimental Cα. Small peptides keep the existing (good-geometry) path.
    PER_RESIDUE_THRESHOLD = 40
    big_chain = (not cyclic) and len(aa_sequence) > PER_RESIDUE_THRESHOLD
    if big_chain:
        print(f"  Large chain ({len(aa_sequence)} residues) → "
              f"per-residue placement (skipping whole-peptide embed)")


    sequence_key = '-'.join(aa_sequence)
    if cyclic and sequence_key in CYCLIC_PEPTIDE_SMILES:
        peptide = Chem.MolFromSmiles(CYCLIC_PEPTIDE_SMILES[sequence_key])
        peptide = Chem.AddHs(peptide)
        AllChem.EmbedMolecule(peptide, randomSeed=np.random.randint(0, 100000), useMacrocycleTorsions=True)
        AllChem.MMFFOptimizeMolecule(peptide, maxIters=2000)
        residue_info = find_all_ca_in_macrocycle(peptide, aa_sequence)
        # Identify linker ring indices before alignment — indices are invariant
        phenyl = peptide.GetSubstructMatch(MolFromSmarts('c1ccccc1'))
        oxadiazole = peptide.GetSubstructMatch(MolFromSmarts('c1nnco1'))
        linker_ring_indices = list(phenyl) + list(oxadiazole)
        position_cyclic_peptide(peptide, residue_info, residue_data)
        conf = peptide.GetConformer()
        atom_positions = [[*conf.GetAtomPosition(i)] for i in range(peptide.GetNumAtoms())]

        return {
            'rdkit_mol': peptide,
            'sequence': aa_sequence,
            'biln': '-'.join([aa[0] for aa in aa_sequence]),
            'atom_positions': atom_positions,
            'conformer': conf,
            'residue_info': residue_info,
            'experimental_data': residue_data,
            'linker_ring_indices': linker_ring_indices,
        }
    
    aa_smiles = {
        'Ala': 'N[C@@H](C)C(=O)O',
        'Val': 'N[C@@H](C(C)C)C(=O)O',
        'Leu': 'N[C@@H](CC(C)C)C(=O)O',
        'Ile': 'N[C@@H]([C@H](CC)C)C(=O)O',
        'Met': 'N[C@@H](CCSC)C(=O)O',
        'Phe': 'N[C@@H](Cc1ccccc1)C(=O)O',
        'Trp': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
        'Pro': 'N1[C@@H](CCC1)C(=O)O',
        'Ser': 'N[C@@H](CO)C(=O)O',
        'Thr': 'N[C@@H]([C@H](O)C)C(=O)O',
        'Tyr': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
        'Cys': 'N[C@@H](CS)C(=O)O',
        'Asn': 'N[C@@H](CC(=O)N)C(=O)O',
        'Gln': 'N[C@@H](CCC(=O)N)C(=O)O',
        'Asp': 'N[C@@H](CC(=O)O)C(=O)O',
        'Glu': 'N[C@@H](CCC(=O)O)C(=O)O',
        'Lys': 'N[C@@H](CCCCN)C(=O)O',
        'Arg': 'N[C@@H](CCCNC(=N)N)C(=O)O',
        'His': 'N[C@@H](Cc1c[nH]cn1)C(=O)O',
        'Gly': 'NCC(=O)O'
    }
    
    # Build peptide by connecting amino acids
    peptide = None
    residue_info = []  # Store {ca_idx, functional_idx, functional_type}
    
    for i, (aa, res_data) in enumerate(zip(aa_sequence, residue_data)):
        aa_mol = Chem.MolFromSmiles(aa_smiles[aa])
        aa_mol = Chem.AddHs(aa_mol)
        AllChem.EmbedMolecule(aa_mol, randomSeed=42)

        # Tag every atom with its residue index so the per-residue placement
        # path can recover membership after CombineMols/RemoveAtom shuffles
        # indices. Properties survive editing; harmless for the small path.
        for atom in aa_mol.GetAtoms():
            atom.SetIntProp('residue_id', i)

        # Find Cα
        ca_idx = find_ca_in_aa(aa_mol, aa)
        
        # Find functional group atom/centroid
        functional_idx = None
        functional_type = AA_FUNCTIONAL_GROUPS.get(aa)
        
        if functional_type == 'ring_centroid':
            functional_idx = 'centroid'  # Will calculate after positioning
        elif functional_type:
            functional_idx = find_functional_group_atom(aa_mol, aa, functional_type)
        
        if i == 0:
            peptide = aa_mol
            residue_info.append({
                'ca_idx': ca_idx,
                'functional_idx': functional_idx,
                'functional_type': functional_type,
                'aa': aa
            })
        else:
            # Connect peptides (same as before)
            c_term_c, c_term_oh = find_c_terminus(peptide)
            n_term_n, n_term_h = find_n_terminus(aa_mol, ca_idx)
            
            combined = Chem.CombineMols(peptide, aa_mol)
            editable = Chem.RWMol(combined)
            
            peptide_offset = peptide.GetNumAtoms()
            n_term_n_global = peptide_offset + n_term_n
            n_term_h_global = [peptide_offset + h for h in n_term_h]
            
            atoms_to_remove = sorted(c_term_oh + n_term_h_global, reverse=True)
            for idx in atoms_to_remove:
                editable.RemoveAtom(idx)
            
            c_term_c_adj = c_term_c - sum(1 for x in atoms_to_remove if x < c_term_c)
            n_term_n_adj = n_term_n_global - sum(1 for x in atoms_to_remove if x < n_term_n_global)
            
            editable.AddBond(c_term_c_adj, n_term_n_adj, Chem.BondType.SINGLE)
            
            peptide = editable.GetMol()
            Chem.SanitizeMol(peptide)
            if not big_chain:
                # Small path (unchanged): re-embed the whole growing peptide so
                # Kabsch positioning has distinct, globally-consistent Cα coords.
                peptide = Chem.AddHs(peptide)
                AllChem.EmbedMolecule(peptide, randomSeed=42)
            # Big path: skip the whole-peptide embed. The combined conformer
            # carries each residue's individually-embedded coordinates through
            # CombineMols; per-residue placement re-positions them afterwards.

            # Adjust indices
            if functional_idx != 'centroid' and functional_idx is not None:
                functional_idx_adj = peptide_offset + functional_idx - sum(1 for x in atoms_to_remove if x < peptide_offset + functional_idx)
            else:
                functional_idx_adj = functional_idx
            
            ca_idx_adj = peptide_offset + ca_idx - sum(1 for x in atoms_to_remove if x < peptide_offset + ca_idx)
            
            residue_info.append({
                'ca_idx': ca_idx_adj,
                'functional_idx': functional_idx_adj,
                'functional_type': functional_type,
                'aa': aa
            })
        
        print(f"  Added {aa}, total atoms: {peptide.GetNumAtoms()}")
    
    if cyclic:
        peptide, residue_info = close_peptide_ring(peptide, residue_info)
        peptide = embed_cyclic_molecule(peptide, residue_info, residue_data)
        position_cyclic_peptide(peptide, residue_info, residue_data)
    elif big_chain:
        position_peptide_per_residue(
            peptide, residue_info, residue_data
        )
    else:
        position_peptide_at_experimental_positions(
            peptide, residue_info, residue_data
        )
    
    # Extract positions
    conf = peptide.GetConformer()

    z_values = []
    
    for i in range(peptide.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        z_values.append(pos.z)
    
    z_min = min(z_values)
    z_max = max(z_values)
    z_range = z_max - z_min
    z_mean = np.mean(z_values)
    
    print(f"  Number of atoms: {peptide.GetNumAtoms()}")
    print(f"  Z range: [{z_min:.6f}, {z_max:.6f}] Å")
    print(f"  Z spread: {z_range:.6f} Å")
    print(f"  Z mean: {z_mean:.6f} Å")
    
    if z_range < 0.001:
        print(f"  Peptide is FLAT (spread < 0.001 Å)")
    elif z_range < 0.01:
        print(f"  Peptide has small height variation (< 0.01 Å)")
    else:
        print(f"  WARNING: Peptide has significant height variation!")
        print(f"     This may cause issues during structure building.")

    atom_positions = []
    for i in range(peptide.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_positions.append([pos.x, pos.y, pos.z])
    
    print(f"✓ Peptide built with {peptide.GetNumAtoms()} atoms")
    print(f"  SMILES: {Chem.MolToSmiles(Chem.RemoveHs(peptide))}")
    print("="*60)
    
    return {
        'rdkit_mol': peptide,
        'sequence': aa_sequence,
        'biln': '-'.join([aa[0] for aa in aa_sequence]),
        'atom_positions': atom_positions,
        'conformer': conf,
        'residue_info': residue_info,
        'experimental_data': residue_data
    }

def find_all_ca_in_macrocycle(mol, aa_sequence):
    residue_info = []
    for aa in aa_sequence:
        pattern = MolFromSmarts(CA_SMARTS[aa])
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            raise ValueError(f"Could not find Ca for {aa} in macrocycle")
        ca_idx = matches[0][0]
        residue_info.append({
            'ca_idx': ca_idx,
            'aa': aa,
            'functional_idx': None,
            'functional_type': AA_FUNCTIONAL_GROUPS.get(aa)
        })
    return residue_info

def find_ca_in_aa(mol, aa):
    """Find Cα atom in amino acid."""
    if aa == 'Gly':
        # Glycine: C between N and C(=O)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if 'N' in neighbors and neighbors.count('C') == 1:
                    return atom.GetIdx()
    if aa == 'Linker':
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and \
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                # neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                # if 'C' in neighbors and 'O' in neighbors:
                return atom.GetIdx()
        raise ValueError("Could not find Ca in Linker")

    else:
        # Chiral carbon
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                return atom.GetIdx()
    raise ValueError(f"Could not find Cα in {aa}")

def find_functional_group_atom(mol, aa, functional_type):
    """Find specific functional group atom."""
    
    if functional_type == 'hydroxyl_o':
        # Find O-H
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if 'H' in neighbors and 'C' in neighbors:
                    return atom.GetIdx()
    
    elif functional_type == 'amide_n':
        # Find N in -C(=O)-NH2
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                neighbors = list(atom.GetNeighbors())
                for n in neighbors:
                    if n.GetSymbol() == 'C':
                        c_neighbors = list(n.GetNeighbors())
                        has_double_o = any(
                            nn.GetSymbol() == 'O' and 
                            mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                            for nn in c_neighbors
                        )
                        if has_double_o:
                            h_count = sum(1 for nn in neighbors if nn.GetSymbol() == 'H')
                            if h_count >= 1:  # Side chain amide
                                return atom.GetIdx()
    
    elif functional_type == 'thiol_s':
        # Find S-H
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S':
                return atom.GetIdx()
    
    elif functional_type == 'carboxyl_c':
        # Find side chain -COOH carbon
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                neighbors = list(atom.GetNeighbors())
                o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
                if o_count == 2:
                    # Check it's not backbone (backbone has N neighbor)
                    has_n = any(n.GetSymbol() == 'N' for n in neighbors)
                    if not has_n:
                        return atom.GetIdx()
    
    elif functional_type == 'amine_n':
        # Find terminal -NH2 (Lys)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if neighbors.count('H') >= 2 and 'C' in neighbors:
                    return atom.GetIdx()
    
    return None

def _collect_hydroxyl(mol, carboxyl_c):
    """Return [O(H) idx, H idx] for the single-bonded -OH on a carboxyl carbon."""
    c_term_oh = []
    for n in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
        if n.GetSymbol() == 'O':
            bond = mol.GetBondBetweenAtoms(carboxyl_c, n.GetIdx())
            if bond.GetBondType() == Chem.BondType.SINGLE:
                c_term_oh.append(n.GetIdx())
                for nn in n.GetNeighbors():
                    if nn.GetSymbol() == 'H':
                        c_term_oh.append(nn.GetIdx())
    return c_term_oh

def find_c_terminus(mol):
    """Find the BACKBONE C-terminus carboxyl C(=O)OH.

    Must NOT match side-chain carboxyls (Asp/Glu), which are also C(=O)OH. The
    backbone C-terminus sits on an N–Cα–C(=O)OH motif: its carboxyl carbon is
    bonded to a Cα that is itself bonded to a backbone N. Asp/Glu side-chain
    carboxyls fail this (their preceding carbon is a CH2, not bonded to N), so
    this disambiguates them. Returns (c_term_c, c_term_oh).
    """
    # N–Cα–C(=O)–OH ; atoms: 0=N 1=Cα 2=carboxyl C 3==O 4=OH
    patt = MolFromSmarts('[NX3,NX4][CX4][CX3](=[OX1])[OX2H1]')
    matches = mol.GetSubstructMatches(patt)
    if matches:
        if len(matches) > 1:
            print(f"  ⚠ find_c_terminus: {len(matches)} backbone-COOH motifs "
                  f"found; using the first")
        c_term_c = matches[0][2]
        return c_term_c, _collect_hydroxyl(mol, c_term_c)

    # Fallback: any free carboxyl (e.g. unusual terminus) — old behavior.
    patt = MolFromSmarts('[CX3](=[OX1])[OX2H1]')
    matches = mol.GetSubstructMatches(patt)
    if matches:
        c_term_c = matches[0][0]
        return c_term_c, _collect_hydroxyl(mol, c_term_c)
    raise ValueError("Could not find C-terminus")

def find_n_terminus(mol, ca_idx):
    """Find the BACKBONE N-terminus: the amine N bonded to the Cα.

    Using Cα adjacency avoids picking side-chain nitrogens (Lys/Arg/Asn/Gln/
    His/Trp), which also carry H's. Works for Pro (ring N bonded to Cα) too.
    Returns (n_idx, [h_idx] or []).
    """
    for n in mol.GetAtomWithIdx(ca_idx).GetNeighbors():
        if n.GetSymbol() == 'N':
            h_neighbors = [x.GetIdx() for x in n.GetNeighbors() if x.GetSymbol() == 'H']
            return n.GetIdx(), ([h_neighbors[0]] if h_neighbors else [])
    raise ValueError("Could not find N-terminus")

def calculate_ring_centroid(mol, conf):
    """Calculate centroid of aromatic/cyclic rings."""
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return None
    
    # Get largest ring
    largest_ring = max(rings, key=len)
    
    positions = []
    for atom_idx in largest_ring:
        pos = conf.GetAtomPosition(atom_idx)
        positions.append([pos.x, pos.y, pos.z])
    
    centroid = np.mean(positions, axis=0)
    return centroid

def _get_side_chain_atoms_linear(mol, ca_idx):
    """
    BFS from ca_idx into the side chain only.
    Stops at backbone N and backbone C=O so only true side-chain atoms are returned.
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
    queue = list(roots)
    side_atoms = list(roots)
    while queue:
        cur = queue.pop(0)
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            nidx = n.GetIdx()
            if nidx not in visited:
                visited.add(nidx)
                queue.append(nidx)
                side_atoms.append(nidx)
    return side_atoms


def _rotate_side_chain_toward(conf, side_atoms, ca_xy, target_xy):
    """
    Rotate side_atoms in XY around ca_xy so their centroid points toward target_xy.
    Returns the rotation angle applied (degrees), or 0 if skipped.
    """
    if not side_atoms:
        return 0.0
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y] for i in side_atoms])
    centroid_xy = positions.mean(axis=0)
    current_vec = centroid_xy - ca_xy
    target_vec  = target_xy  - ca_xy
    if np.linalg.norm(current_vec) < 0.1 or np.linalg.norm(target_vec) < 0.1:
        return 0.0
    cv = current_vec / np.linalg.norm(current_vec)
    tv = target_vec  / np.linalg.norm(target_vec)
    angle = np.arctan2(np.cross(cv, tv), np.dot(cv, tv))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    for atom_idx in side_atoms:
        pos = conf.GetAtomPosition(atom_idx)
        rel = np.array([pos.x, pos.y]) - ca_xy
        new_xy = rot2d @ rel + ca_xy
        conf.SetAtomPosition(atom_idx, (new_xy[0], new_xy[1], 0.0))
    return float(np.degrees(angle))


def position_peptide_at_experimental_positions(peptide, residue_info, residue_data):
    """
    Position peptide backbone via global Kabsch alignment on Cα positions,
    then orient each side chain toward its experimental functional_position.
    """
    conf = peptide.GetConformer()
    n_atoms = peptide.GetNumAtoms()

    ca_indices = [ri['ca_idx'] for ri in residue_info]
    current_ca = np.array([[*conf.GetAtomPosition(i)] for i in ca_indices], dtype=float)
    target_ca  = np.array([rd['ca_position'] for rd in residue_data], dtype=float)

    # ── Global Kabsch in XY ──────────────────────────────────────────────────
    cur_xy = current_ca.copy(); cur_xy[:, 2] = 0.0
    tgt_xy = target_ca.copy();  tgt_xy[:, 2] = 0.0

    cur_center = cur_xy.mean(axis=0)
    tgt_center = tgt_xy.mean(axis=0)
    H = (cur_xy - cur_center).T @ (tgt_xy - tgt_center)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    rot = Vt.T @ np.diag([1, 1, d]) @ U.T

    all_pos = np.array([[*conf.GetAtomPosition(i)] for i in range(n_atoms)], dtype=float)
    all_rot = (rot @ (all_pos - cur_center).T).T + tgt_center
    all_rot[:, 2] = 0.0   # flatten to surface

    for i in range(n_atoms):
        conf.SetAtomPosition(i, all_rot[i].tolist())

    print(f"\nGlobal Kabsch alignment applied ({len(ca_indices)} Cα).")

    _orient_all_side_chains(peptide, conf, residue_info, residue_data)


def _orient_all_side_chains(peptide, conf, residue_info, residue_data):
    """Rotate each residue's side chain in XY toward its functional_position."""
    print(f"Orienting side chains:")
    for res_idx, (res_info, res_data) in enumerate(zip(residue_info, residue_data)):
        func_pos = res_data.get('functional_position')
        if func_pos is None:
            print(f"  Residue {res_idx} ({res_info['aa']}): no functional_position, skipped")
            continue

        ca_idx = res_info['ca_idx']
        pos = conf.GetAtomPosition(ca_idx)
        ca_xy = np.array([pos.x, pos.y])
        target_xy = np.array(func_pos[:2])

        side_atoms = _get_side_chain_atoms_linear(peptide, ca_idx)
        deg = _rotate_side_chain_toward(conf, side_atoms, ca_xy, target_xy)
        print(f"  Residue {res_idx} ({res_info['aa']}): side chain rotated {deg:.1f}°")


def position_peptide_per_residue(peptide, residue_info, residue_data):
    """
    Cheap placement for large linear chains (no global embed, no global Kabsch).

    Each residue keeps the internal geometry from its individual EmbedMolecule
    (carried through CombineMols). We translate every residue rigidly so its Cα
    lands on the experimental position, flatten to the surface (z=0), then orient
    side chains. This never embeds the full molecule, so it is cheap and cannot
    hit the silent embedder failure that occurs on large peptides.
    """
    from collections import defaultdict

    if peptide.GetNumConformers() == 0:
        raise ValueError("position_peptide_per_residue: peptide has no conformer "
                         "(per-AA embeds did not propagate through CombineMols)")
    conf = peptide.GetConformer()

    # Recover residue membership from the residue_id tag set during the build.
    members = defaultdict(list)
    for atom in peptide.GetAtoms():
        if atom.HasProp('residue_id'):
            members[atom.GetIntProp('residue_id')].append(atom.GetIdx())

    print(f"\nPer-residue placement ({len(residue_info)} residues):")
    for res_idx, (res_info, res_data) in enumerate(zip(residue_info, residue_data)):
        ca_idx = res_info['ca_idx']
        ca_pos = np.array([*conf.GetAtomPosition(ca_idx)], dtype=float)
        target = np.array(res_data['ca_position'], dtype=float)
        shift = target - ca_pos
        for aid in members.get(res_idx, [ca_idx]):
            p = np.array([*conf.GetAtomPosition(aid)], dtype=float) + shift
            p[2] = 0.0   # flatten to surface
            conf.SetAtomPosition(aid, p.tolist())

    _orient_all_side_chains(peptide, conf, residue_info, residue_data)


def find_glycosylation_site(peptide_data, residue_index=0, aa_type=None):
    romol = peptide_data['rdkit_mol']
    conf = romol.GetConformer()
    
    aa_sequence = peptide_data['sequence']  # This is ['Asn', 'Phe', 'Ala']
    
    # Get AA at target position
    actual_aa = aa_sequence[residue_index]
    
    if aa_type and actual_aa != aa_type:
        raise ValueError(f"Expected {aa_type} at position {residue_index}, got {actual_aa}")
    
    # Check if AA is supported
    if actual_aa not in AA_GLYCOSYLATION_INFO:
        supported = ', '.join(AA_GLYCOSYLATION_INFO.keys())
        raise ValueError(
            f"Amino acid '{actual_aa}' does not support standard glycosylation. "
            f"Supported AAs: {supported}"
        )
    
    aa_info = AA_GLYCOSYLATION_INFO[actual_aa]
    
    print(f"\nFinding glycosylation site on residue {residue_index}: {actual_aa}")
    print(f"  Type: {aa_info['glycosylation_type']}")
    print(f"  Target atom: {aa_info['atom_name']} ({aa_info['description']})")
    
    finder_map = {
        'Asn': find_asn_site,
        'Ser': find_ser_site,
        'Thr': find_thr_site,
        'Tyr': find_tyr_site,
        'Trp': find_trp_site
    }
    
    finder_func = finder_map[actual_aa]
    return finder_func(romol, residue_index, aa_sequence, conf, actual_aa, aa_info)

def find_asn_site(romol, residue_index, aa_sequence, conf, canonical_name, aa_info):
    """Asparagine: Find Nδ in side chain amide."""
    atom_idx = find_asn_n_delta(romol, residue_index, aa_sequence)['atom_index']
    pos = conf.GetAtomPosition(atom_idx)
    
    print(f"  Found {aa_info['atom_name']} at atom index {atom_idx}")
    print(f"  Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
    
    return {
        'residue_index': residue_index,
        'aa_type': canonical_name,
        'glycosylation_type': aa_info['glycosylation_type'],
        'atom_name': aa_info['atom_name'],
        'atom_index': atom_idx,
        'position': [pos.x, pos.y, pos.z],
        'requires_spacer': aa_info['requires_spacer'],
        'h_to_remove': aa_info['h_to_remove']
    }

def find_ser_site(romol, residue_index, seq, conf, canonical_name, aa_info):
    """Serine: Find Oγ hydroxyl."""
    atom_idx = find_ser_o_gamma(romol, residue_index, seq)['atom_index']
    pos = conf.GetAtomPosition(atom_idx)
    
    print(f"  Found {aa_info['atom_name']} at atom index {atom_idx}")
    print(f"  Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
    
    return {
        'residue_index': residue_index,
        'aa_type': canonical_name,
        'glycosylation_type': aa_info['glycosylation_type'],
        'atom_name': aa_info['atom_name'],
        'atom_index': atom_idx,
        'position': [pos.x, pos.y, pos.z],
        'requires_spacer': aa_info['requires_spacer'],
        'h_to_remove': aa_info['h_to_remove']
    }

def find_thr_site(romol, residue_index, seq, conf, canonical_name, aa_info):
    """Threonine: Find Oγ1 hydroxyl."""
    atom_idx = find_thr_o_gamma(romol, residue_index, seq)['atom_index']
    pos = conf.GetAtomPosition(atom_idx)
    
    print(f"  Found {aa_info['atom_name']} at atom index {atom_idx}")
    print(f"  Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
    
    return {
        'residue_index': residue_index,
        'aa_type': canonical_name,
        'glycosylation_type': aa_info['glycosylation_type'],
        'atom_name': aa_info['atom_name'],
        'atom_index': atom_idx,
        'position': [pos.x, pos.y, pos.z],
        'requires_spacer': aa_info['requires_spacer'],
        'h_to_remove': aa_info['h_to_remove']
    }

def find_tyr_site(romol, residue_index, seq, conf, canonical_name, aa_info):
    """Tyrosine: Find Oη phenolic hydroxyl (restricted to residue_index)."""
    allowed = _residue_atom_indices(romol, residue_index)
    for atom in romol.GetAtoms():
        if atom.GetSymbol() == 'O':
            if allowed is not None and atom.GetIdx() not in allowed:
                continue
            neighbors = list(atom.GetNeighbors())
            has_aromatic_c = any(
                n.GetSymbol() == 'C' and n.GetIsAromatic() for n in neighbors
            )
            has_h = any(n.GetSymbol() == 'H' for n in neighbors)
            
            if has_aromatic_c and has_h:
                atom_idx = atom.GetIdx()
                pos = conf.GetAtomPosition(atom_idx)
                
                print(f"  Found {aa_info['atom_name']} at atom index {atom_idx}")
                print(f"  Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
                
                return {
                    'residue_index': residue_index,
                    'aa_type': canonical_name,
                    'glycosylation_type': aa_info['glycosylation_type'],
                    'atom_name': aa_info['atom_name'],
                    'atom_index': atom_idx,
                    'position': [pos.x, pos.y, pos.z],
                    'requires_spacer': aa_info['requires_spacer'],
                    'h_to_remove': aa_info['h_to_remove']
                }
    
    raise ValueError(f"Could not find {aa_info['atom_name']} in Tyr at residue {residue_index}")

def find_trp_site(romol, residue_index, seq, conf, canonical_name, aa_info):
    """Tryptophan: Find C2 in indole ring (restricted to residue_index)."""
    allowed = _residue_atom_indices(romol, residue_index)
    # Find indole nitrogen in 5-membered ring
    indole_n = None
    for atom in romol.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
            if allowed is not None and atom.GetIdx() not in allowed:
                continue
            rings = atom.GetOwningMol().GetRingInfo()
            for ring in rings.AtomRings():
                if atom.GetIdx() in ring and len(ring) == 5:
                    indole_n = atom
                    break
            if indole_n:
                break
    
    if indole_n is None:
        raise ValueError(f"Could not find indole nitrogen in Trp at residue {residue_index}")
    
    # Find C2: aromatic carbon neighbor
    for neighbor in indole_n.GetNeighbors():
        if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
            atom_idx = neighbor.GetIdx()
            pos = conf.GetAtomPosition(atom_idx)
            
            print(f"  Found {aa_info['atom_name']} at atom index {atom_idx}")
            print(f"  Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
            
            return {
                'residue_index': residue_index,
                'aa_type': canonical_name,
                'glycosylation_type': aa_info['glycosylation_type'],
                'atom_name': aa_info['atom_name'],
                'atom_index': atom_idx,
                'position': [pos.x, pos.y, pos.z],
                'requires_spacer': aa_info['requires_spacer'],
                'h_to_remove': aa_info['h_to_remove']
            }
    
    raise ValueError(f"Could not find {aa_info['atom_name']} in Trp at residue {residue_index}")

# ============================================================================
# LINKAGE BUILDERS 
# ============================================================================

def create_o_glycosidic_compact(mol_data, sugar_carbon, target_o_gamma, 
                                anomeric_config='beta', linkage_name='O-Glyc'):
    """Create TRANS O-glycosidic linkage."""
    
    carbon_idx = mol_data['carbon_map'][sugar_carbon]
    coords = np.array(mol_data['absolute_coordinates'])
    c1_pos = coords[carbon_idx]
    target_og = np.array(target_o_gamma)
    
    total_distance = np.linalg.norm(target_og - c1_pos)
    main_direction = (target_og - c1_pos) / total_distance
    
    # Get perpendicular vectors
    perp1 = get_perpendicular_vector(main_direction)
    perp2 = np.cross(main_direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # TRANS placement: oxygen on opposite side
    # Use 180° dihedral (opposite to main direction)
    trans_direction = -main_direction  # 180° rotation
    
    o_glycosidic = c1_pos + trans_direction * BOND_LENGTHS['C_O']
    
    remaining_distance = np.linalg.norm(target_og - o_glycosidic)
    
    oh_to_remove = find_hydroxyl_oxygen_at_carbon(mol_data, carbon_idx, coords)
    
    return {
        'linkage': linkage_name,
        'glycosylation_type': 'O-glycosidic',
        'anomeric_config': anomeric_config,
        'dihedral': 'TRANS (180°)',  # ADD THIS
        'sugar_carbon': c1_pos.tolist(),
        'o_glycosidic_position': o_glycosidic.tolist(),
        'target_o_gamma': target_og.tolist(),
        'bridge_distance': remaining_distance,
        'total_distance': total_distance,
        'oh_to_remove': oh_to_remove if oh_to_remove else [],
        'h_to_remove_peptide': ['Ser/Thr/Tyr-Oγ-H']
    }

def create_n_glycosidic_compact(mol_data, sugar_carbon, target_n_delta,
                                anomeric_config='beta', linkage_name='N-Glyc'):
    """Create TRANS N-glycosidic linkage with GlcNAc spacer."""
    
    carbon_idx = mol_data['carbon_map'][sugar_carbon]
    coords = np.array(mol_data['absolute_coordinates'])
    c1_sugar = coords[carbon_idx]
    target_n = np.array(target_n_delta)
    
    total_distance = np.linalg.norm(target_n - c1_sugar)
    main_direction = (target_n - c1_sugar) / total_distance
    
    # Get perpendicular vectors
    perp1 = get_perpendicular_vector(main_direction)
    perp2 = np.cross(main_direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # TRANS configuration: first oxygen placed opposite to target
    trans_direction = -main_direction
    
    # 1. β-Glycosidic oxygen (TRANS to target)
    o1_pos = c1_sugar + trans_direction * BOND_LENGTHS['C_O']
    
    # 2. GlcNAc C1 (continue trans)
    glcnac_c1 = o1_pos + trans_direction * BOND_LENGTHS['O_C']
    
    # 3. GlcNAc C2
    glcnac_c2 = glcnac_c1 + trans_direction * BOND_LENGTHS['C_C']
    
    # 4. Amide carbonyl C
    amide_c = glcnac_c2 + trans_direction * BOND_LENGTHS['C_C']
    
    # 5. Final N (TRANS to spacer chain)
    n_final = amide_c + trans_direction * BOND_LENGTHS['C_N']
    
    achieved_distance = np.linalg.norm(n_final - target_n)
    
    # N-acetyl branch (perpendicular to trans chain)
    acetyl_angle = np.radians(ANGLES['N_ACETYL_ANGLE'])
    n_acetyl_dir = trans_direction * np.cos(acetyl_angle) + perp1 * np.sin(acetyl_angle)
    n_acetyl = glcnac_c2 + n_acetyl_dir * BOND_LENGTHS['C_N']
    
    acetyl_c = n_acetyl + n_acetyl_dir * BOND_LENGTHS['C_N']
    acetyl_o = acetyl_c + perp2 * BOND_LENGTHS['C_O_DOUBLE']
    acetyl_ch3 = acetyl_c + n_acetyl_dir * BOND_LENGTHS['C_C']
    
    # Amide carbonyl O (perpendicular to main chain)
    amide_o = amide_c + perp2 * BOND_LENGTHS['C_O_DOUBLE']
    
    oh_to_remove = find_hydroxyl_oxygen_at_carbon(mol_data, carbon_idx, coords)
    
    return {
        'linkage': linkage_name,
        'glycosylation_type': 'N-glycosidic',
        'anomeric_config': anomeric_config,
        'dihedral': 'TRANS (180°)', 
        'sugar_carbon': c1_sugar.tolist(),
        'o1_beta_position': o1_pos.tolist(),
        'glcnac_c1_position': glcnac_c1.tolist(),
        'glcnac_c2_position': glcnac_c2.tolist(),
        'n_acetyl_n_position': n_acetyl.tolist(),
        'acetyl_carbonyl_c': acetyl_c.tolist(),
        'acetyl_carbonyl_o': acetyl_o.tolist(),
        'acetyl_ch3_position': acetyl_ch3.tolist(),
        'amide_carbonyl_c': amide_c.tolist(),
        'amide_carbonyl_o': amide_o.tolist(),
        'n_final_position': n_final.tolist(),
        'target_n_delta': target_n.tolist(),
        'target_accuracy': achieved_distance,
        'total_distance': total_distance,
        'will_require_md': achieved_distance > 1.0,
        'oh_to_remove': oh_to_remove if oh_to_remove else [],
        'h_to_remove_peptide': ['Asn-Nδ-H1']
    }

def create_n_glycosidic_direct(mol_data, sugar_carbon, target_n_delta,
                                anomeric_config='beta', linkage_name='N-Glyc-Direct'):
    """Create DIRECT N-glycosidic linkage: Sugar-C1 directly to Peptide-N (no oxygen!)"""
    
    carbon_idx = mol_data['carbon_map'][sugar_carbon]
    coords = np.array(mol_data['absolute_coordinates'])
    c1_sugar = coords[carbon_idx]
    target_n = np.array(target_n_delta)
    
    total_distance = np.linalg.norm(target_n - c1_sugar)
    main_direction = (target_n - c1_sugar) / total_distance
    
    print(f"\n  Direct N-glycosidic linkage:")
    print(f"    C1 sugar: {c1_sugar}")
    print(f"    Target N: {target_n}")
    print(f"    Distance: {total_distance:.3f} Å")
    
    # N-glycosidic is DIRECT C-N bond (no bridging oxygen!)
    # Just report the positions - the bond will be C1 → N
    
    achieved_distance = total_distance
    
    if achieved_distance > 2.0:  # C-N bond is ~1.47Å
        print(f"    ⚠ Distance {achieved_distance:.2f} Å is large")
        print(f"    → MD optimization will compress this")
    else:
        print(f"    ✓ Distance is {achieved_distance:.3f} Å")
    
    # Find OH to remove from anomeric carbon
    oh_to_remove = find_hydroxyl_oxygen_at_carbon(mol_data, carbon_idx, coords)
    
    return {
        'linkage': linkage_name,
        'glycosylation_type': 'N-glycosidic-direct',
        'anomeric_config': anomeric_config,
        'sugar_carbon': c1_sugar.tolist(),
        'target_n_delta': target_n.tolist(),
        'target_accuracy': achieved_distance,
        'total_distance': total_distance,
        'will_require_md': achieved_distance > 2.0,
        'oh_to_remove': oh_to_remove if oh_to_remove else [],
        'h_to_remove_peptide': ['Asn-Nδ-H1']
    }

def _residue_atom_indices(romol, residue_index):
    """Atom indices tagged with residue_id == residue_index.

    Returns None when the molecule carries no residue_id tags at all (older
    embed paths / reloaded mols) so callers can fall back to an unrestricted
    scan. The tag is set per atom during build_peptide_with_rdkit_ca, so for a
    normally-built peptide this confines the functional-group search to the
    requested residue — without it the finders return the first matching atom
    in the whole chain (wrong residue on any multi-site peptide).
    """
    ids = set()
    tagged = False
    for atom in romol.GetAtoms():
        if atom.HasProp('residue_id'):
            tagged = True
            if atom.GetIntProp('residue_id') == residue_index:
                ids.add(atom.GetIdx())
    return ids if tagged else None


def find_asn_n_delta(romol, residue_index, aa_sequence):  # Changed parameter
    """Find Nδ atom in Asparagine side chain (restricted to residue_index)."""
    allowed = _residue_atom_indices(romol, residue_index)

    for atom in romol.GetAtoms():
        if atom.GetSymbol() == 'N':
            if allowed is not None and atom.GetIdx() not in allowed:
                continue
            neighbors = list(atom.GetNeighbors())

            for neighbor in neighbors:
                if neighbor.GetSymbol() == 'C':
                    c_neighbors = list(neighbor.GetNeighbors())
                    
                    has_carbonyl = False
                    for c_neighbor in c_neighbors:
                        if c_neighbor.GetSymbol() == 'O':
                            bond = romol.GetBondBetweenAtoms(
                                neighbor.GetIdx(), 
                                c_neighbor.GetIdx()
                            )
                            if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                                has_carbonyl = True
                                break
                    
                    if has_carbonyl:
                        c_count = sum(1 for n in neighbors if n.GetSymbol() == 'C')
                        if c_count == 1:
                            return {
                                'atom_name': 'N_delta',
                                'atom_index': atom.GetIdx()
                            }
    
    raise ValueError(f"Could not find Nδ in Asn at residue {residue_index}")

def find_ser_o_gamma(romol, residue_index, seq):
    """Find Oγ atom in Serine side chain (restricted to residue_index)."""
    allowed = _residue_atom_indices(romol, residue_index)
    for atom in romol.GetAtoms():
        if atom.GetSymbol() == 'O':
            if allowed is not None and atom.GetIdx() not in allowed:
                continue
            neighbors = list(atom.GetNeighbors())

            if len(neighbors) == 2:
                has_c = any(n.GetSymbol() == 'C' for n in neighbors)
                has_h = any(n.GetSymbol() == 'H' for n in neighbors)
                
                if has_c and has_h:
                    for neighbor in neighbors:
                        if neighbor.GetSymbol() == 'C':
                            bond = romol.GetBondBetweenAtoms(
                                atom.GetIdx(), 
                                neighbor.GetIdx()
                            )
                            if bond and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                                c_neighbors = [n for n in neighbors if n.GetSymbol() == 'C']
                                if len(c_neighbors) == 1:
                                    return {
                                        'atom_name': 'O_gamma',
                                        'atom_index': atom.GetIdx()
                                    }
    
    raise ValueError(f"Could not find Oγ in Ser at residue {residue_index}")

def find_thr_o_gamma(romol, residue_index, seq):
    """Find Oγ1 atom in Threonine side chain."""
    return find_ser_o_gamma(romol, residue_index, seq)

def create_peptide_sugar_linkage(mol_data, sugar_carbon, target_atom_pos, 
                                 linkage_type, anomeric_config, use_spacer, enforce_trans=True):
    """
    Create glycosidic linkage (dispatcher function).
    
    Calls appropriate builder based on linkage_type.
    """
    
    if linkage_type == 'O-glycosidic':
        return create_o_glycosidic_compact(
            mol_data=mol_data,
            sugar_carbon=sugar_carbon,
            target_o_gamma=target_atom_pos,
            anomeric_config=anomeric_config,
            linkage_name='O-Glyc',
            enforce_trans=enforce_trans
        )
    
    elif linkage_type == 'N-glycosidic':
        if use_spacer:
            return create_n_glycosidic_compact(                mol_data=mol_data,
                sugar_carbon=sugar_carbon,
                target_n_delta=target_atom_pos,
                anomeric_config=anomeric_config,
                linkage_name='N-Glyc-Direct'
                )
        else:
            return create_n_glycosidic_direct(  
                mol_data=mol_data,
                sugar_carbon=sugar_carbon,
                target_n_delta=target_atom_pos,
                anomeric_config=anomeric_config,
                linkage_name='N-Glyc-Direct'
            )
    
    elif linkage_type == 'C-glycosidic':
        # For Trp C-mannosylation
        raise NotImplementedError(
            "C-glycosidic linkage not yet implemented"
        )
    
    else:
        raise ValueError(f"Unsupported linkage type: {linkage_type}")
    
def close_peptide_ring(peptide, residue_info):
    """
    Close a linear peptide chain into a macrocycle by forming
    the final amide bond between the C-terminus of the last
    residue and the N-terminus of the first residue.

    Parameters:
    - peptide: RDKit Mol, the linear chain with Hs added
    - residue_info: list of dicts with 'ca_idx' per residue

    Returns:
    - cyclic RDKit Mol (sanitized, with Hs)
    """

    # find free termini on the linear chain
    c_term_c, c_term_oh = find_c_terminus(peptide)

    # N-terminus of residue 0: pass its ca_idx so find_n_terminus
    # excludes the alpha carbon nitrogen in Pro
    ca_idx_res0 = residue_info[0]['ca_idx']
    n_term_n, n_term_h = find_n_terminus(peptide, ca_idx_res0)

    # build editable mol and remove condensation atoms
    editable = Chem.RWMol(peptide)

    atoms_to_remove = sorted(c_term_oh + n_term_h, reverse=True)
    for idx in atoms_to_remove:
        editable.RemoveAtom(idx)

    # adjust indices after atom removal
    c_term_c_adj = c_term_c - sum(
        1 for x in atoms_to_remove if x < c_term_c
    )
    n_term_n_adj = n_term_n - sum(
        1 for x in atoms_to_remove if x < n_term_n
    )

    # form the closing amide bond
    editable.AddBond(c_term_c_adj, n_term_n_adj, Chem.BondType.SINGLE)

    cyclic = editable.GetMol()
    Chem.SanitizeMol(cyclic)

    # update ca indices in residue_info after atom removal
    for res in residue_info:
        res['ca_idx'] = res['ca_idx'] - sum(
            1 for x in atoms_to_remove if x < res['ca_idx']
        )

    return cyclic, residue_info

def embed_cyclic_molecule(mol, residue_info, residue_data):
    """
    Embed a cyclic molecule using the experimental Ca positions
    as coordinate seeds via coordMap.
 
    coordMap maps atom_idx -> (x, y, z) for each Ca, biasing
    the distance geometry toward the experimental geometry.
 
    Falls back to random coords if constrained embedding fails.
    """
    mol = Chem.AddHs(mol)
 
    # build coordMap from Ca positions
    coord_map = {}
    for res_info, res_data in zip(residue_info, residue_data):
        ca_idx = res_info['ca_idx']
        pos = res_data['ca_position']
        coord_map[ca_idx] = pos
 
    # convert coordMap to boost dict format expected by RDKit
    coord_map_boost = {}
    for idx, pos in coord_map.items():
        from rdkit.Geometry import rdGeometry
        pt = rdGeometry.Point3D(pos[0], pos[1], pos[2])
        coord_map_boost[idx] = pt
 
    result = AllChem.EmbedMolecule(
        mol,
        useRandomCoords=True,
        randomSeed=42,
        maxAttempts=1000,
        coordMap=coord_map_boost,
        useMacrocycleTorsions=True,
    )
 
    if result == -1:
        print("  Constrained embedding failed, trying without coordMap")
        result = AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            randomSeed=42,
            maxAttempts=1000,
            useMacrocycleTorsions=True,
        )
 
    if result == -1:
        raise RuntimeError("Embedding failed for cyclic molecule")
 
    # relax geometry while keeping Ca atoms near target
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
 
    print(f"  Embedded: {mol.GetNumAtoms()} atoms")
    return mol


def _get_side_chain_atoms_cyclic(mol, ca_idx, backbone_ring_set):
    """BFS from ca_idx into the side chain; does not cross the macrocycle backbone."""
    roots = [n.GetIdx() for n in mol.GetAtomWithIdx(ca_idx).GetNeighbors()
             if n.GetIdx() not in backbone_ring_set]
    if not roots:
        return []
    visited = {ca_idx} | set(roots)
    queue, side_atoms = list(roots), list(roots)
    while queue:
        cur = queue.pop(0)
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            nidx = n.GetIdx()
            if nidx not in visited and nidx not in backbone_ring_set:
                visited.add(nidx)
                queue.append(nidx)
                side_atoms.append(nidx)
    return side_atoms


def _orient_side_chains_cyclic(mol, all_positions, residue_info, residue_data):
    """
    Rotate each residue's side chain in XY (around its Cα) so the side-chain
    centroid points toward the experimental functional_position.
    Operates on all_positions numpy array in-place.
    """
    from rdkit import Chem as _Chem
    _Chem.GetSymmSSSR(mol)
    all_rings = list(mol.GetRingInfo().AtomRings())
    if not all_rings:
        return
    backbone_ring = set(max(all_rings, key=len))

    print("  Orienting side chains (cyclic):")
    for res_info, res_data in zip(residue_info, residue_data):
        func_pos = res_data.get('functional_position')
        if func_pos is None:
            continue

        ca_idx = res_info['ca_idx']
        ca_xy = all_positions[ca_idx, :2]
        target_xy = np.array(func_pos[:2])

        side_atoms = _get_side_chain_atoms_cyclic(mol, ca_idx, backbone_ring)
        if not side_atoms:
            print(f"    {res_info['aa']}: no side-chain atoms, skipped")
            continue

        centroid_xy = all_positions[side_atoms, :2].mean(axis=0)
        cv = centroid_xy - ca_xy
        tv = target_xy  - ca_xy
        if np.linalg.norm(cv) < 0.1 or np.linalg.norm(tv) < 0.1:
            print(f"    {res_info['aa']}: vector too small, skipped")
            continue

        cv /= np.linalg.norm(cv); tv /= np.linalg.norm(tv)
        angle = np.arctan2(np.cross(cv, tv), np.dot(cv, tv))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        for atom_idx in side_atoms:
            rel = all_positions[atom_idx, :2] - ca_xy
            all_positions[atom_idx, :2] = rot2d @ rel + ca_xy

        print(f"    {res_info['aa']}: side chain rotated {np.degrees(angle):.1f}°")


def _flatten_ring_to_surface(ring_indices, all_positions, target_z):
    """
    Rigid-body rotate a ring so its plane is parallel to the surface (XY),
    then translate it to target_z.  Preserves all internal bond lengths and
    angles — treats the ring as a rigid body throughout.
    """
    ring_pos = all_positions[ring_indices]
    centroid = ring_pos.mean(axis=0)

    normal = get_ring_normal_from_positions(all_positions, ring_indices)
    rot_matrix = calculate_alignment_rotation(normal, [0.0, 0.0, 1.0])

    rotated = (rot_matrix @ (ring_pos - centroid).T).T + centroid
    rotated[:, 2] += target_z - rotated[:, 2].mean()
    return rotated


def position_cyclic_peptide(peptide, residue_info, residue_data, linker_data=None):
    """
    Align cyclic peptide to experimental Ca positions using a global
    Kabsch superposition in XY only (Z ignored), then flatten all
    atoms to Z=0 for STM/surface compatibility.
    Uses geometry_utils functions to stay DRY.
    """

    conf = peptide.GetConformer()
    n_atoms = peptide.GetNumAtoms()

    ca_indices = [res['ca_idx'] for res in residue_info]
    target_ca = np.array([res_data['ca_position'] for res_data in residue_data])

    if linker_data is not None:
        all_pos_temp = get_positions(peptide.GetConformer(), peptide.GetNumAtoms())
        linker_pos = np.array(linker_data['ca_position'])
        linker_anchor_idx = int(np.argmin(
            np.linalg.norm(all_pos_temp - linker_pos, axis=1)
        ))
        ca_indices.append(linker_anchor_idx)
        target_ca = np.vstack([target_ca, linker_pos])
        print(f"  Linker anchor: atom {linker_anchor_idx}")

    all_positions = get_positions(conf, n_atoms)
    current_ca = get_positions_for_atoms(conf, ca_indices)

    # XY only: zero out Z before alignment
    current_ca_xy = current_ca.copy()
    current_ca_xy[:, 2] = 0.0
    target_ca_xy = target_ca.copy()
    target_ca_xy[:, 2] = 0.0

    current_center = current_ca_xy.mean(axis=0)
    target_center = target_ca_xy.mean(axis=0)

    current_centered = current_ca_xy - current_center
    target_centered = target_ca_xy - target_center

    # Kabsch SVD
    H = current_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    rot = Vt.T @ np.diag([1, 1, d]) @ U.T

    # apply to all atoms and flatten Z to 0
    all_centered = all_positions - current_center
    all_rotated = (rot @ all_centered.T).T + target_center
    all_rotated[:, 2] = all_positions[:, 2]
    phenyl_match = peptide.GetSubstructMatch(MolFromSmarts('c1ccccc1'))
    oxadiazole_match = peptide.GetSubstructMatch(MolFromSmarts('c1nnco1'))
    linker_ring_atoms = list(phenyl_match) + list(oxadiazole_match)

    if linker_ring_atoms:
        ring_z = np.mean([all_rotated[i, 2] for i in linker_ring_atoms])

        # Treat phenyl + oxadiazole as one rigid body so both share the same plane
        flat = _flatten_ring_to_surface(linker_ring_atoms, all_rotated, ring_z)
        for j, atom_idx in enumerate(linker_ring_atoms):
            all_rotated[atom_idx] = flat[j]
        print(f"  Linker (phenyl + oxadiazole) flattened as single rigid body to Z={ring_z:.3f} A")

    # Orient each side chain toward its experimental functional_position
    _orient_side_chains_cyclic(peptide, all_rotated, residue_info, residue_data)

    set_positions(conf, all_rotated, n_atoms)

    final_ca = get_positions_for_atoms(conf, ca_indices)
    ca_rmsd = np.sqrt(np.mean(np.sum(
        (final_ca[:, :2] - target_ca_xy[:, :2])**2, axis=1
    )))
    print(f"  Ca XY RMSD after Kabsch alignment: {ca_rmsd:.4f} A")
    print(f"  All atoms flattened to Z=0")

def create_glycopeptide_from_blobs(
    mol_data,
    aa_sequence,
    experimental_ca,
    sugar_carbon='C1',
    target_residue=0,
    linkage_type='O-glycosidic',
    anomeric_config='beta',
    use_spacer=True
):
    """
    Create glycopeptide using experimental COG positions.
    
    This is the MAIN function you'll use.
    
    Parameters:
    - mol_data: Sugar structure data
    - aa_sequence: "N-Y-T" or ['N', 'Y', 'T']
    - experimental_ca: [[x1,y1,z1], [x2,y2,z2], ...]
    - sugar_carbon: Anomeric carbon (usually 'C1')
    - target_residue: Which AA connects to sugar (0-indexed)
    - linkage_type: 'N-glycosidic' or 'O-glycosidic'
    - anomeric_config: 'beta' or 'alpha'
    - use_spacer: Add GlcNAc for N-glycosidic
    
    Returns:
    - glycopeptide_data: Complete structure
    """
    
    print(f"\n{'='*70}")
    print(f"CREATING GLYCOPEPTIDE FROM EXPERIMENTAL COG POSITIONS")
    print(f"{'='*70}")
    
    # Step 1: Build peptide with pyPept at experimental positions
    peptide_data = build_peptide_with_rdkit_ca(
        aa_sequence=aa_sequence,
        experimental_ca=experimental_ca
    )
    
    # Step 2: Find glycosylation site
    attachment_info = find_glycosylation_site(
        peptide_data=peptide_data,
        residue_index=target_residue
    )
    
    # Validate linkage type
    if linkage_type != attachment_info['glycosylation_type']:
        print(f"⚠ Warning: Requested {linkage_type} but {attachment_info['aa_type']} "
              f"typically uses {attachment_info['glycosylation_type']}")
    
    # Step 3: Create linkage geometry
    target_atom_pos = np.array(attachment_info['position'])
    
    linkage_data = create_peptide_sugar_linkage(
        mol_data=mol_data,
        sugar_carbon=sugar_carbon,
        target_atom_pos=target_atom_pos,
        linkage_type=linkage_type,
        anomeric_config=anomeric_config,
        use_spacer=use_spacer
    )
    
    # Step 4: Assemble result
    glycopeptide = {
        'peptide': peptide_data,
        'attachment': attachment_info,
        'linkage': linkage_data,
        'sugar': {
            'carbon': sugar_carbon,
            'position': linkage_data['sugar_carbon']
        },
        'modifications': {
            'oh_to_remove': linkage_data['oh_to_remove'],
            'h_to_remove': linkage_data.get('h_to_remove_peptide', []),
            'peptide_atom_to_modify': attachment_info['atom_index']
        }
    }
    
    print(f"\n{'='*70}")
    print(f"✓ GLYCOPEPTIDE CREATED SUCCESSFULLY")
    print(f"{'='*70}\n")
    
    return glycopeptide