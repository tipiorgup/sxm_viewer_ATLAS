"""Backmap (fine-grain) an optimized CG model to all-atom.

Inverse of :mod:`cg_export`.  After the coarse-grained beads have been
optimized (flattened / relaxed), this restores the *exact* atomistic
conformation of every monomer onto the new bead positions.

Method: per residue, a single rigid-body transform (rotation + translation,
found by the Kabsch algorithm) maps that residue's pre-optimization bead
centroids onto its optimized bead positions.  The transform is applied to every
atom of the residue.  Because it is rigid, the monomer's internal geometry -
the chair conformer built in ``monomer_building`` - is preserved exactly.

The atomistic template is the assembled ``final_no_h`` molecule: it already
carries the exact chair conformers *and* the full bond graph, so the bonding is
kept for free - only coordinates change, never the topology.

Caveats
-------
* Inter-monomer junctions (e.g. glycosidic bonds) are slightly strained, because
  each monomer is fit independently.  A short junction-only relaxation can fix
  this afterwards if needed.
* A residue with fewer than 3 beads has no unique rotation; it falls back to a
  translation that matches the bead centroid, keeping its template orientation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ANGSTROM_TO_NM = 0.1
NM_TO_ANGSTROM = 10.0


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple:
    """Rigid transform mapping P onto Q. Returns (R, t) with ``R @ p + t ~= q``.

    P, Q are (N, 3) corresponding point sets.  Pure rotation (proper, det=+1)
    plus translation; no scaling or reflection.
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def read_gro_positions(path: str) -> np.ndarray:
    """Read bead positions (nm) from a .gro file, in file (bead-index) order."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1].strip())
    pos = np.zeros((n, 3))
    for i in range(n):
        line = lines[2 + i]
        # GROMACS fixed columns: positions are the 3 floats after atom number,
        # each 8 chars wide starting at column 20.
        pos[i, 0] = float(line[20:28])
        pos[i, 1] = float(line[28:36])
        pos[i, 2] = float(line[36:44])
    return pos


def _residue_atom_indices(mol: Chem.Mol, heavy_globals: List[int]) -> List[int]:
    """Heavy atoms plus their attached hydrogens."""
    idxs = set(heavy_globals)
    for gi in heavy_globals:
        for nb in mol.GetAtomWithIdx(gi).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                idxs.add(nb.GetIdx())
    return sorted(idxs)


def backmap(
    template_mol: Chem.Mol,
    cg_model: dict,
    target_bead_pos_nm: np.ndarray,
) -> Chem.Mol:
    """Return a copy of ``template_mol`` moved onto the optimized bead positions.

    Parameters
    ----------
    template_mol :
        Assembled all-atom molecule with a conformer (the exact chairs + bonds).
        Coordinates in Angstrom.
    cg_model :
        Output of :func:`cg_export.build_cg_model` (beads carry ``index``,
        ``atom_globals``, ``pos`` (nm), ``resname``, ``resnum``).
    target_bead_pos_nm :
        (n_beads, 3) optimized bead positions in nm, indexed by bead ``index``.
    """
    beads = cg_model['beads']
    if len(target_bead_pos_nm) != len(beads):
        raise ValueError(
            f"target has {len(target_bead_pos_nm)} beads but model has {len(beads)}")

    out = Chem.Mol(template_mol)
    conf_in = template_mol.GetConformer()
    conf_out = out.GetConformer()

    # Group beads by residue.
    by_res: Dict[tuple, List[dict]] = defaultdict(list)
    for b in beads:
        by_res[(b['resname'], b['resnum'])].append(b)

    for (resname, resnum), res_beads in by_res.items():
        # Source = pre-opt bead centroids (Angstrom); target = optimized (Angstrom).
        src = np.array([b['pos'] for b in res_beads]) * NM_TO_ANGSTROM
        dst = np.array([target_bead_pos_nm[b['index']] for b in res_beads]) * NM_TO_ANGSTROM

        if len(res_beads) >= 3:
            R, t = kabsch(src, dst)
        else:
            # No unique rotation: translate to match centroid, keep orientation.
            R = np.eye(3)
            t = dst.mean(axis=0) - src.mean(axis=0)
            print(f"  [backmap] {resname}_{resnum}: {len(res_beads)} bead(s) - "
                  f"translation only (orientation kept)")

        heavy = [gi for b in res_beads for gi in b['atom_globals']]
        for gi in _residue_atom_indices(template_mol, heavy):
            p = conf_in.GetAtomPosition(gi)
            new = R @ np.array([p.x, p.y, p.z]) + t
            conf_out.SetAtomPosition(gi, (float(new[0]), float(new[1]), float(new[2])))

    return out


def backmap_from_gro(
    template_mol: Chem.Mol,
    cg_model: dict,
    optimized_gro: str,
) -> Chem.Mol:
    """Convenience: read optimized bead positions from a .gro, then backmap."""
    target = read_gro_positions(optimized_gro)
    return backmap(template_mol, cg_model, target)


def write_backmapped(mol: Chem.Mol, stem: str, out_dir: str = ".") -> dict:
    """Write the backmapped all-atom structure to PDB (+ SDF)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdb = out / f"{stem}_backmapped.pdb"
    sdf = out / f"{stem}_backmapped.sdf"

    Chem.MolToPDBFile(mol, str(pdb), flavor=8)
    try:
        writer = Chem.SDWriter(str(sdf))
        writer.SetKekulize(False)
        writer.write(mol)
        writer.close()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [backmap] SDF write skipped ({exc})")
    print(f"  [backmap] Wrote {pdb}")
    return {'pdb': pdb, 'sdf': sdf}
