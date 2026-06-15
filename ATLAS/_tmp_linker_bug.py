"""Reproduce the Phe-ring teleport: linker machinery latching onto Phe benzene
rings in a linker-less peptide. No full pipeline needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from src.rotation_optimization.structure import saving_lps as sl

# A linear peptide with TWO Phe rings, far apart, NO oxadiazole (no linker).
# Phe-Gly-Gly-Gly-Phe
smi = 'N[C@@H](Cc1ccccc1)C(=O)NCC(=O)NCC(=O)NCC(=O)N[C@@H](Cc1ccccc1)C(=O)O'
mol = Chem.MolFromSmiles(smi)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

# Spread the two rings far apart so a teleport is obvious (stretch along x).
conf = mol.GetConformer()
peptide_data = {'rdkit_mol': mol}

# 1) What does the linker SNAPSHOT grab on a no-linker peptide?
snap = sl.snapshot_peptide_linker_positions(peptide_data)
print("snapshot keys:", list(snap.keys()))
if '__linker__' in snap:
    idxs = list(snap['__linker__'].keys())
    print("  snapshot ring atom idxs:", idxs)
    com = np.mean([snap['__linker__'][i] for i in idxs], axis=0)
    print("  snapshot ring COM:", np.round(com, 2))

# 2) What does _find_linker_ring_atoms grab on the SAME mol (the 'final' mol)?
phenyl, oxa = sl._find_linker_ring_atoms(mol)
print("\n_find_linker_ring_atoms -> phenyl:", phenyl, " oxadiazole:", oxa)
if phenyl:
    com2 = np.mean([list(conf.GetAtomPosition(i)) for i in phenyl], axis=0)
    print("  final phenyl COM:", np.round(com2, 2))

# 3) Show the teleport: record a ring atom, run the snapshot restore, compare.
before = {i: list(conf.GetAtomPosition(i)) for i in (phenyl or [])}
sl.restore_ring_positions_from_snapshot(mol, snap)
print("\nPer-atom movement caused by snapshot restore:")
for i in (phenyl or []):
    after = np.array(list(conf.GetAtomPosition(i)))
    d = np.linalg.norm(after - np.array(before[i]))
    print(f"  atom {i}: moved {d:.2f} A")
