"""Coarse-grained (Martini) export of an assembled multi-unit structure.

Given the final assembled RDKit molecule (with a 3D conformer) and a residue
map (``unit_key -> {resname, resnum, chain, atom_indices}`` as produced by
``pdb_export.build_residue_index``), this writes two debugging models:

  * ``*_cg_unbonded`` : every bead is a free point (empty ``[bonds]``).
  * ``*_cg_bonded``   : intra-unit and cross-unit bonds, with a constant
                        force constant and ``b0`` measured from the structure.

The number and type of beads for each unit come straight from
``martini_mapper`` (via ``monomer_building.martini_beads.map_smiles_to_beads``),
so sugars, peptide residues, lipid tails, PEtN and phosphate are all handled by
the same path: each residue's sub-structure is coarse-grained on its own, then
bonds are read off the *assembled* bond graph — any bond whose two heavy atoms
fall in different beads becomes a CG bond, which captures glycosidic, peptide,
lipid, PEtN and phosphate linkages without per-type special casing.

Bead position is the mass-weighted centroid (heavy atoms + attached hydrogens)
of the bead's atoms at the assembled coordinates.  Coordinates are converted
from Angstrom (pipeline units) to nanometre (GROMACS units).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from src.monomer_building.martini_beads import map_smiles_to_beads
from src.martini_mapper.utils import clean_smiles, safe_mol_from_smiles
from src.rotation_optimization.structure import cg_published_map

# LPS grown-on groups martini_mapper has no bead rule for (phosphate/PEtN) or
# that map more cleanly as standard Martini 3 (lipid tail).  Dispatched by
# pdb_export unit resname to the dedicated structural handlers.
_STRUCTURAL_HANDLERS = {
    "PO4": cg_published_map.map_phosphate_residue,
    "PEN": cg_published_map.map_petn_residue,
    "LIP": cg_published_map.map_lipid_residue,
}


def _bead_mass(bead_type: str) -> float:
    """Martini 3 mass by size prefix: T=36, S=54, regular=72."""
    if bead_type.startswith("T"):
        return 36.0
    if bead_type.startswith("S"):
        return 54.0
    return 72.0

# Constant harmonic force constant for the debug topology (kJ/mol/nm^2).
# Intra-residue bonds (both beads on the same unit) keep the stiff default,
# inter-residue bonds (glycosidic, phosphate, PEtN, lipid-anchor linkages,
# identified by resnum mismatch, which is exactly the yaml-defined
# connectivity, since no other mechanism ever bonds two units together)
# get the softer INTER_RESIDUE_BOND_K instead.
DEFAULT_BOND_K = 20000
INTER_RESIDUE_BOND_K = 2000
LIP_BOND_K = 125  # lipid tail bonds (intra-chain and anchor-to-sugar), soft
                  # regardless of intra/inter classification, so the tails
                  # stay floppy rather than rigid
ANGSTROM_TO_NM = 0.1


@lru_cache(maxsize=None)
def _mapper_beads(smiles: str) -> dict:
    """Coarse-grain a SMILES to beads, cached per unique SMILES.

    A typical structure repeats the same monomer many times (e.g. dozens of
    identical ``Glc`` units), so caching by SMILES avoids re-running the mapper
    for every copy.  The bead ``atom_idx`` are geometry-independent, so the
    cached result is reused across all residues with the same SMILES; only the
    per-residue substructure match (cheap) is redone.
    """
    return map_smiles_to_beads("res", smiles)


# ---------------------------------------------------------------------------
# Per-residue coarse-graining
# ---------------------------------------------------------------------------

def _heavy_globals(mol: Chem.Mol, atom_indices: List[int]) -> List[int]:
    """Heavy (non-hydrogen) atom indices from a residue's atom list."""
    return [i for i in atom_indices
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]


def _build_residue_submol(
    mol: Chem.Mol, heavy_globals: List[int]
) -> Tuple[Optional[Chem.Mol], Dict[int, int]]:
    """Build a heavy-atom-only sub-molecule for one residue.

    Returns ``(submol, sub_to_global)`` where ``sub_to_global`` maps the
    submol's atom index to the assembled molecule's atom index.  Cross-unit
    bonds are simply not copied (their valence is filled by implicit H on
    sanitisation), so the submol is the residue in isolation.
    """
    rw = Chem.RWMol()
    global_to_sub: Dict[int, int] = {}
    for gi in heavy_globals:
        src = mol.GetAtomWithIdx(gi)
        atom = Chem.Atom(src.GetAtomicNum())
        atom.SetFormalCharge(src.GetFormalCharge())
        global_to_sub[gi] = rw.AddAtom(atom)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in global_to_sub and j in global_to_sub:
            rw.AddBond(global_to_sub[i], global_to_sub[j], bond.GetBondType())

    submol = rw.GetMol()
    try:
        Chem.SanitizeMol(submol)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ! CG: could not sanitise residue submol ({exc}); skipping")
        return None, {}

    sub_to_global = {sub: glob for glob, sub in global_to_sub.items()}
    return submol, sub_to_global


def _map_residue_beads(
    mol: Chem.Mol, unit_key: str, meta: dict
) -> Optional[List[dict]]:
    """Coarse-grain one residue into beads carrying *assembled* atom indices.

    Each returned bead dict has: ``bead_type``, ``mass``, and
    ``atom_globals`` (heavy-atom indices in the assembled molecule).
    Returns ``None`` if the residue could not be mapped.
    """
    heavy_globals = _heavy_globals(mol, meta['atom_indices'])
    if not heavy_globals:
        return None

    # Structural handlers for LPS groups martini_mapper can't map (PO4/PEN/LIP).
    handler = _STRUCTURAL_HANDLERS.get(meta['resname'].strip().upper())
    if handler is not None:
        struct_beads = handler(mol, mol.GetConformer(), heavy_globals)
        if struct_beads is None:
            print(f"  ! CG: structural handler failed on '{unit_key}'; skipping unit")
            return None
        return [{
            'bead_type': b['bead_type'],
            'mass': _bead_mass(b['bead_type']),
            'atom_globals': b['atom_globals'],
        } for b in struct_beads]

    submol, sub_to_global = _build_residue_submol(mol, heavy_globals)
    if submol is None:
        return None

    smiles = Chem.MolToSmiles(submol)
    try:
        bead_info = _mapper_beads(smiles)
    except Exception as exc:
        print(f"  ! CG: martini_mapper failed on '{unit_key}' "
              f"(SMILES {smiles}): {exc}; skipping unit")
        return None

    # Reconstruct the mapper's molecule exactly as it indexed atoms, so the
    # bead ``atom_idx`` line up, then match it onto our residue submol.
    mapper_mol = safe_mol_from_smiles(clean_smiles(smiles))
    match = submol.GetSubstructMatch(mapper_mol)  # match[k] = submol idx for mapper atom k
    if not match or len(match) != mapper_mol.GetNumAtoms():
        print(f"  ! CG: could not align beads to atoms for '{unit_key}' "
              f"(SMILES {smiles}); skipping unit")
        return None

    beads = []
    for b in bead_info['beads']:
        try:
            atom_globals = [sub_to_global[match[k]] for k in b['atom_idx']]
        except (KeyError, IndexError):
            print(f"  ! CG: bead {b['index']} of '{unit_key}' references an "
                  f"unmatched atom; skipping unit")
            return None
        beads.append({
            'bead_type': b['bead_type'],
            'mass': float(b['mass']),
            'atom_globals': atom_globals,
        })
    return beads


# ---------------------------------------------------------------------------
# Assembly of the global bead table + bonds
# ---------------------------------------------------------------------------

def _bead_position(mol: Chem.Mol, conf, atom_globals: List[int]) -> np.ndarray:
    """Mass-weighted centroid (nm) over the bead's heavy atoms + attached H."""
    idxs = list(atom_globals)
    for gi in atom_globals:
        for nb in mol.GetAtomWithIdx(gi).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                idxs.append(nb.GetIdx())

    total_mass = 0.0
    acc = np.zeros(3)
    for gi in idxs:
        m = mol.GetAtomWithIdx(gi).GetMass()
        p = conf.GetAtomPosition(gi)
        acc += m * np.array([p.x, p.y, p.z])
        total_mass += m
    centroid_ang = acc / total_mass if total_mass > 0 else acc
    return centroid_ang * ANGSTROM_TO_NM


def build_cg_model(mol: Chem.Mol, residue_index: Dict[str, dict]) -> dict:
    """Coarse-grain the whole assembled molecule.

    Returns a dict with:
      ``beads`` : list of {index, bead_type, mass, resname, resnum, pos(nm)}
      ``bonds`` : list of (i, j, b0_nm) for the bonded model (0-based bead idx)
      ``uncovered`` : list of unit keys that could not be mapped
      ``unit_beads`` : {unit_key: [bead_index, ...]}, built directly from the
        unique residue_index key while iterating, never from resnum/resname
        (resnum is NOT globally unique in this pipeline's numbering, e.g. a
        KDO unit and a GAL unit can share the same resnum, so any downstream
        code needing "all beads of this specific unit" should use this dict,
        not resnum equality)
    """
    conf = mol.GetConformer()

    beads: List[dict] = []
    atom_to_bead: Dict[int, int] = {}
    uncovered: List[str] = []
    unit_beads: Dict[str, List[int]] = {}

    for unit_key, meta in residue_index.items():
        unit_beads_data = _map_residue_beads(mol, unit_key, meta)
        if unit_beads_data is None:
            uncovered.append(unit_key)
            continue
        unit_beads[unit_key] = []
        for ub in unit_beads_data:
            bead_index = len(beads)
            for gi in ub['atom_globals']:
                atom_to_bead[gi] = bead_index
            beads.append({
                'index': bead_index,
                'bead_type': ub['bead_type'],
                'mass': ub['mass'],
                'resname': meta['resname'],
                'resnum': meta['resnum'],
                'unit_key': unit_key,
                'atom_globals': list(ub['atom_globals']),  # heavy atoms, for backmapping
                'pos': _bead_position(mol, conf, ub['atom_globals']),
            })
            unit_beads[unit_key].append(bead_index)

    # Bonds: any assembled bond whose heavy endpoints fall in different beads.
    bond_pairs = set()
    skipped_bonds = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1 or mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        bi, bj = atom_to_bead.get(i), atom_to_bead.get(j)
        if bi is None or bj is None:
            skipped_bonds += 1
            continue
        if bi != bj:
            bond_pairs.add((min(bi, bj), max(bi, bj)))

    bonds = []
    for i, j in sorted(bond_pairs):
        b0 = float(np.linalg.norm(beads[i]['pos'] - beads[j]['pos']))
        # unit_key, not resnum, since resnum is not globally unique here
        # (e.g. a KDO unit and a GAL unit can share the same resnum).
        inter_residue = beads[i]['unit_key'] != beads[j]['unit_key']
        bonds.append((i, j, b0, inter_residue))

    if uncovered:
        print(f"  ! CG: {len(uncovered)} unit(s) not coarse-grained: "
              f"{', '.join(uncovered)}")
    if skipped_bonds:
        print(f"  ! CG: {skipped_bonds} bond(s) skipped (touch an uncovered unit)")

    return {'beads': beads, 'bonds': bonds, 'uncovered': uncovered, 'unit_beads': unit_beads}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

SUGAR_RESNAMES = {"GAL", "GLC", "GLA", "RHM", "KDO"}


def build_ring_anchors(model: dict, sugar_resnames: set = SUGAR_RESNAMES) -> dict:
    """For every sugar unit, finds the existing bead whose own position is
    closest to that unit's own centroid (mean bead position), using the
    model's own unit_beads/pos directly, no resnum-based lookup, so the
    resnum collision (a KDO unit and a GAL unit can share the same resnum)
    cannot affect this.

    Returns {unit_key: {"resname":.., "resnum":.., "anchor_bead":..}}.
    """
    beads_by_index = {b['index']: b for b in model['beads']}
    anchors = {}

    for unit_key, bead_indices in model['unit_beads'].items():
        sample = beads_by_index[bead_indices[0]]
        if sample['resname'].strip().upper() not in sugar_resnames:
            continue

        positions = [beads_by_index[i]['pos'] for i in bead_indices]
        centroid = np.mean(positions, axis=0)

        best_idx, best_dist = None, None
        for i in bead_indices:
            d2 = float(np.sum((beads_by_index[i]['pos'] - centroid) ** 2))
            if best_dist is None or d2 < best_dist:
                best_dist = d2
                best_idx = i

        anchors[unit_key] = {
            "resname": sample['resname'],
            "resnum": sample['resnum'],
            "anchor_bead": best_idx,
        }

    return anchors


def write_ring_posre_file(anchors: dict, path: Path, fc_xy: float = 5000) -> Path:
    """Writes a posre.itp restraining only the ring-anchor beads: fcx=fcy
    rigid (in-plane), fcz=0 (free), so each ring can still rotate in 3D
    about that point while its own translation is suppressed. No other
    bead gets any restraint."""
    lines = ["[ position_restraints ]", "; ai   funct   fcx    fcy    fcz"]
    for unit_key in sorted(anchors, key=lambda k: anchors[k]["anchor_bead"]):
        a = anchors[unit_key]
        bead_1based = a["anchor_bead"] + 1
        lines.append(
            f"{bead_1based} 1 {fc_xy:.0f} {fc_xy:.0f} 0  "
            f"; ring anchor: {unit_key} ({a['resname']}{a['resnum']})"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"  [ring-posre] Wrote {path} ({len(anchors)} ring anchors)")
    return path


def _bead_atom_name(index: int) -> str:
    """GROMACS atom name (<=5 chars) for a bead."""
    return f"B{index + 1}"[:5]


def write_gro(model: dict, name: str, path: Path, box=(10.0, 10.0, 10.0)) -> Path:
    beads = model['beads']
    lines = [f"{name} CG", f"{len(beads)}"]
    for b in beads:
        x, y, z = b['pos']
        lines.append(
            f"{b['resnum']:5d}{b['resname'][:5]:<5s}{_bead_atom_name(b['index']):>5s}"
            f"{b['index'] + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  [cg] Wrote {path} ({len(beads)} beads)")
    return path


def write_itp(model: dict, name: str, path: Path, *, bonded: bool) -> Path:
    beads = model['beads']
    lines = [
        "[ moleculetype ]",
        "; name  nrexcl",
        f"{name[:10]:10s} 1",
        "",
        "[ atoms ]",
        ";  nr   type  resnr  resid  atom  cgnr   charge     mass",
    ]
    for b in beads:
        lines.append(
            f"{b['index'] + 1:5d} {b['bead_type'][:6]:6s} {b['resnum']:5d} "
            f"{b['resname'][:5]:5s} {_bead_atom_name(b['index']):5s} "
            f"{b['index'] + 1:5d} {0.0:8.3f} {b['mass']:8.3f}"
        )
    lines.append("")
    lines.append("[ bonds ]")
    if bonded:
        lines.append(";  ai    aj  funct       b0(nm)        k(kJ/mol/nm^2)")
        for i, j, b0, inter_residue in model['bonds']:
            if beads[i]['resname'].strip().upper() == 'LIP' or beads[j]['resname'].strip().upper() == 'LIP':
                k = LIP_BOND_K
            else:
                k = INTER_RESIDUE_BOND_K if inter_residue else DEFAULT_BOND_K
            line = f"{i + 1:5d} {j + 1:5d} {1:5d} {b0:12.5f} {k:12d}"
            if inter_residue:
                ri, rj = beads[i]['resname'].strip(), beads[j]['resname'].strip()
                rni, rnj = beads[i]['resnum'], beads[j]['resnum']
                line += f"  ; inter-residue: {ri}{rni} - {rj}{rnj}"
            lines.append(line)
    else:
        lines.append("; (unbonded debug model - no bonds)")
    lines.append("")
    lines.append("#ifdef POSRES")
    lines.append('#include "posre.itp"')
    lines.append("#endif")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    kind = "bonded" if bonded else "unbonded"
    print(f"  [cg] Wrote {path} ({kind}, {len(model['bonds']) if bonded else 0} bonds)")
    return path


def build_carbon_guide(
    model: dict,
    residue_index: Dict[str, dict],
    mol: Chem.Mol,
) -> dict:
    """Maps each bead back to the atom name(s) (C1, C2, O5, ...) RDKit's
    AtomPDBResidueInfo already carries on each atom, per pdb_export.py's
    assign_sugar_residues/_set_atom_info. No index arithmetic, no
    carbon_map, no molecule_data_dict needed: the name is read directly off
    the atom object, which survives every atom removal/reindexing done
    during assembly, since RDKit keeps MonomerInfo attached to the atom
    itself rather than to its index.

    Structural units (PO4/PEN/LIP) get element+counter names from
    finalize_residue_info, not sugar carbon names, those are reported as-is.

    Returns {unit_key: {"resname":.., "resnum":.., "beads": [
        {"bead_index":.., "bead_type":.., "atom_names": [...]}
    ]}}
    """
    guide: dict = {}
    for unit_key, meta in residue_index.items():
        guide[unit_key] = {
            "resname": meta.get("resname"),
            "resnum": meta.get("resnum"),
            "beads": [],
        }

    for bead in model['beads']:
        owning_unit = bead.get("unit_key")
        if owning_unit is None or owning_unit not in guide:
            continue

        names = []
        for gi in bead["atom_globals"]:
            atom = mol.GetAtomWithIdx(gi)
            info = atom.GetMonomerInfo()
            if info is not None:
                names.append(info.GetName().strip())
        guide[owning_unit]["beads"].append({
            "bead_index": bead["index"],
            "bead_type": bead["bead_type"],
            "atom_names": sorted(set(names)) if names else None,
        })

    return guide


def write_carbon_guide(guide: dict, path: Path) -> Path:
    import json
    path.write_text(json.dumps(guide, indent=2))
    print(f"  [cg] Wrote {path} (bead-to-carbon guide)")
    return path


def export_cg_models(
    mol: Chem.Mol,
    residue_index: Dict[str, dict],
    name: str,
    out_dir: str = ".",
    write_guide: bool = True,
    write_ring_posre: bool = True,
) -> dict:
    """Build and write both debug CG models.

    ``name`` is the SDF stem (e.g. ``<prefix>_p1_lowest_energy_optimized``);
    the two models append ``_cg_unbonded`` / ``_cg_bonded``.

    When ``write_guide`` is True (default), also writes
    ``<name>_bead_carbon_guide.json``, mapping each bead back to the atom
    name(s) (C1, O5, ...) already carried on ``mol``'s atoms via
    AtomPDBResidueInfo (see build_carbon_guide's docstring).

    When ``write_ring_posre`` is True (default), also writes
    ``<name>_ring_posre.itp``, restraining only one bead per sugar ring (the
    bead closest to that ring's own centroid) rigid in-plane (fcz=0, free),
    so the ring can still rotate in 3D about that point (see
    build_ring_anchors's docstring).
    """
    if not residue_index:
        print("  ! CG: no residue map available; skipping CG export")
        return {'model': None, 'paths': {}}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_cg_model(mol, residue_index)

    paths = {}
    # Coordinates are identical for both models.
    write_gro(model, name, out / f"{name}_cg_unbonded.gro")
    paths['unbonded_gro'] = out / f"{name}_cg_unbonded.gro"
    paths['unbonded_itp'] = write_itp(model, f"{name}_ub", out / f"{name}_cg_unbonded.itp", bonded=False)

    write_gro(model, name, out / f"{name}_cg_bonded.gro")
    paths['bonded_gro'] = out / f"{name}_cg_bonded.gro"
    paths['bonded_itp'] = write_itp(model, name, out / f"{name}_cg_bonded.itp", bonded=True)

    if write_guide:
        guide = build_carbon_guide(model, residue_index, mol)
        paths['carbon_guide'] = write_carbon_guide(
            guide, out / f"{name}_bead_carbon_guide.json"
        )

    if write_ring_posre:
        anchors = build_ring_anchors(model)
        paths['ring_posre'] = write_ring_posre_file(
            anchors, out / f"{name}_ring_posre.itp"
        )

    return {'model': model, 'paths': paths}