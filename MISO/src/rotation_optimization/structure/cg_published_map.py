"""Published Martini 3 atom->bead mappings, keyed on chemical atom names.

We adopt the *topology* from polyply (bonded params, charges, constraints,
virtual sites) and only need to place beads consistently with it.  This module
holds the atom-name -> bead assignment per residue so we can compute bead
positions from our own (experimentally positioned, post-QUEST) coordinates.

Sugars: the canonical Martini 3 carbohydrate mapping (Grunewald et al., JCTC
2022, 10.1021/acs.jctc.2c00757).  Rule summary: bead A starts at the anomeric
carbon C1 and proceeds counterclockwise; bead C always holds the ring oxygen
O5; 4:1 grouping (2 carbons + their hydroxyls); a massless virtual site TC4
sits at the ring centre (in the topology it is the COG of A, B, C).

Atom names are the chemically meaningful names assigned by ``pdb_export``
(C1..C6, ring O5, hydroxyls O1..O6).  Missing atoms (e.g. the anomeric O1 when
C1 is in a glycosidic link) are simply skipped for that bead's centroid.

Bead *types* here are informational only — the authoritative types come from the
polyply .itp.  What matters for coordinates is the bead count, order, and which
atoms each bead averages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

ANGSTROM_TO_NM = 0.1

# Per-residue mapping: ordered list of (bead_label, bead_type, [atom_names]).
# Order MUST match the residue's [ atoms ] order in the polyply .itp.
# 'vsite' entries are built from other beads (COG), not from atoms.
PUBLISHED_MAPPINGS: Dict[str, dict] = {
    # D-glucopyranose (dextran/cellulose monomer).
    "GLC": {
        "beads": [
            ("A", "SP4r", ["C1", "O1", "C2", "O2"]),
            ("B", "SP1r", ["C3", "O3", "C4", "O4"]),
            ("C", "SN6r", ["C5", "C6", "O5", "O6"]),
        ],
        "vsites": [
            ("VS", "TC4", ["A", "B", "C"]),  # COG of the ring beads
        ],
    },
}

# Common PDB resname aliases -> our mapping key.
RESNAME_ALIASES = {
    "DEX": "GLC", "BGC": "GLC", "GLC": "GLC", "GLU": "GLC",  # glucose variants
}


def resolve_resname(resname: str) -> Optional[str]:
    key = RESNAME_ALIASES.get(resname.upper(), resname.upper())
    return key if key in PUBLISHED_MAPPINGS else None


def _centroid_nm(mol, conf, globals_: List[int]) -> np.ndarray:
    """Mass-weighted centroid (nm) over the given atom indices (+ attached H)."""
    idxs = list(globals_)
    for gi in globals_:
        for nb in mol.GetAtomWithIdx(gi).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                idxs.append(nb.GetIdx())
    tot = 0.0
    acc = np.zeros(3)
    for gi in idxs:
        m = mol.GetAtomWithIdx(gi).GetMass()
        p = conf.GetAtomPosition(gi)
        acc += m * np.array([p.x, p.y, p.z])
        tot += m
    return (acc / tot if tot else acc) * ANGSTROM_TO_NM


def map_residue_published(
    resname: str,
    atom_name_to_global: Dict[str, int],
    mol,
    conf,
) -> Optional[List[dict]]:
    """Map one residue to beads in polyply order using the published scheme.

    ``atom_name_to_global`` maps this residue's chemical atom name -> assembled
    atom index.  Returns an ordered list of bead dicts (real beads then vsites),
    each: {label, bead_type, atom_globals, pos(nm), is_vsite, vsite_from}.
    Returns None if the residue is not in the published registry.
    """
    key = resolve_resname(resname)
    if key is None:
        return None
    spec = PUBLISHED_MAPPINGS[key]

    beads: List[dict] = []
    by_label: Dict[str, dict] = {}
    for label, btype, names in spec["beads"]:
        globals_ = [atom_name_to_global[n] for n in names if n in atom_name_to_global]
        if not globals_:
            print(f"  ! published-map: bead {label} of {resname} has no atoms; skipping unit")
            return None
        bead = {
            "label": label, "bead_type": btype, "atom_globals": globals_,
            "pos": _centroid_nm(mol, conf, globals_), "is_vsite": False,
            "vsite_from": None,
        }
        beads.append(bead)
        by_label[label] = bead

    for label, btype, from_labels in spec.get("vsites", []):
        src = [by_label[l]["pos"] for l in from_labels if l in by_label]
        pos = np.mean(src, axis=0) if src else np.zeros(3)  # placeholder; gmx rebuilds
        beads.append({
            "label": label, "bead_type": btype, "atom_globals": [],
            "pos": pos, "is_vsite": True, "vsite_from": list(from_labels),
        })

    return beads


# ---------------------------------------------------------------------------
# Amino acids: structural (graph-based) mapping in our own coordinate frame.
#
# Our peptide atoms are NOT named with CHARMM conventions, so we identify
# backbone vs sidechain structurally (via the Ca signature) rather than by name.
# BB = backbone (N, Ca, C, O); the sidechain is split into the number of SC
# beads the published Martini 3 protein model uses for that residue.  Exact
# per-atom SC partition is refined later; count + rough position is what the
# polyply topology (fixed params) needs.
# ---------------------------------------------------------------------------

# Number of side-chain beads per residue in the Martini 3 protein model
# (from vermouth martini3001 maps: ALA/ASN=1 SC, PHE=3, ...).
AA_SIDECHAIN_BEADS = {
    "GLY": 0, "ALA": 1, "CYS": 1, "SER": 1, "THR": 1, "VAL": 1, "LEU": 1,
    "ILE": 1, "MET": 1, "PRO": 1, "ASN": 1, "ASP": 1, "GLN": 1, "GLU": 1,
    "PHE": 3, "TYR": 3, "HIS": 3, "TRP": 4, "ARG": 2, "LYS": 2,
}


def find_ca_atoms(mol) -> List[int]:
    """Backbone Ca: a carbon bonded to an amide N and to a carbonyl C (=O)."""
    cas = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        nbrs = list(atom.GetNeighbors())
        has_n = any(n.GetSymbol() == "N" for n in nbrs)
        has_carbonyl = any(
            n.GetSymbol() == "C" and any(
                nn.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx()).GetBondTypeAsDouble() == 2.0
                for nn in n.GetNeighbors())
            for n in nbrs)
        if has_n and has_carbonyl:
            cas.append(atom.GetIdx())
    return cas


def _backbone_atoms(mol, ca_idx: int) -> tuple:
    """Return (N, Ca, C, O) backbone atom indices for a Ca (Nones dropped)."""
    ca = mol.GetAtomWithIdx(ca_idx)
    n_idx = c_idx = o_idx = None
    for nb in ca.GetNeighbors():
        if nb.GetSymbol() == "N" and n_idx is None:
            n_idx = nb.GetIdx()
        elif nb.GetSymbol() == "C":
            for nn in nb.GetNeighbors():
                b = mol.GetBondBetweenAtoms(nb.GetIdx(), nn.GetIdx())
                if nn.GetSymbol() == "O" and b.GetBondTypeAsDouble() == 2.0:
                    c_idx, o_idx = nb.GetIdx(), nn.GetIdx()
                    break
    return tuple(i for i in (n_idx, ca_idx, c_idx, o_idx) if i is not None)


def _sidechain_atoms(mol, ca_idx: int, backbone: set) -> List[int]:
    """Heavy side-chain atoms: BFS from Ca, not crossing backbone or sugar rings."""
    ca = mol.GetAtomWithIdx(ca_idx)
    roots = [n.GetIdx() for n in ca.GetNeighbors()
             if n.GetIdx() not in backbone and n.GetSymbol() != "H"]
    visited = set(backbone) | {ca_idx} | set(roots)
    queue, side = list(roots), list(roots)
    while queue:
        cur = queue.pop(0)
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = n.GetIdx()
            if ni in visited or n.GetSymbol() == "H":
                continue
            visited.add(ni)
            queue.append(ni)
            side.append(ni)
    return side


def _split_atoms_geometric(mol, conf, atom_globals: List[int], k: int) -> List[List[int]]:
    """Split atoms into k groups along their principal axis (deterministic).

    Used for multi-bead side chains (e.g. PHE ring -> 3 beads).  Exact published
    partition is a later refinement; this yields k reasonable groups.
    """
    if k <= 1 or len(atom_globals) <= 1:
        return [atom_globals]
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in atom_globals])
    centered = pts - pts.mean(axis=0)
    # principal axis
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]
    order = np.argsort(proj)
    groups: List[List[int]] = [[] for _ in range(k)]
    for rank, idx in enumerate(order):
        groups[min(rank * k // len(order), k - 1)].append(atom_globals[idx])
    return [g for g in groups if g]


def map_amino_acid_residue(mol, conf, ca_idx: int, resname: str) -> Optional[List[dict]]:
    """Map one amino-acid residue to BB + SC beads in Martini order.

    Returns ordered bead dicts {label, bead_type, atom_globals, pos, is_vsite}.
    ``bead_type`` here is a placeholder (real types come from the polyply .itp).
    """
    resname = resname.upper()
    if resname not in AA_SIDECHAIN_BEADS:
        return None
    backbone = _backbone_atoms(mol, ca_idx)
    if not backbone:
        return None

    beads = [{
        "label": "BB", "bead_type": "BB", "atom_globals": list(backbone),
        "pos": _centroid_nm(mol, conf, list(backbone)), "is_vsite": False,
        "vsite_from": None,
    }]

    n_sc = AA_SIDECHAIN_BEADS[resname]
    if n_sc >= 1:
        side = _sidechain_atoms(mol, ca_idx, set(backbone))
        if side:
            for k, grp in enumerate(_split_atoms_geometric(mol, conf, side, n_sc), 1):
                beads.append({
                    "label": f"SC{k}", "bead_type": f"SC{k}", "atom_globals": grp,
                    "pos": _centroid_nm(mol, conf, grp), "is_vsite": False,
                    "vsite_from": None,
                })
    return beads


# ---------------------------------------------------------------------------
# LPS grown-on groups: lipid tail, phosphate, PEtN (standard Martini 3).
# Data-driven: dispatched by pdb_export unit type (LIP / PO4 / PEN).  Coordinate
# mapping is structural (these fragments have generic atom names).
# ---------------------------------------------------------------------------

def _order_carbon_chain(mol, carbons: List[int]) -> List[int]:
    """Order chain carbons from one endpoint (endpoint = 1 in-chain neighbour)."""
    cset = set(carbons)
    adj = {c: [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()
               if n.GetIdx() in cset] for c in carbons}
    ends = [c for c in carbons if len(adj[c]) == 1]
    start = ends[0] if ends else (carbons[0] if carbons else None)
    if start is None:
        return []
    order, prev, cur = [start], None, start
    while len(order) < len(carbons):
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def map_lipid_residue(mol, conf, atom_globals: List[int],
                      carbons_per_bead: int = 4) -> Optional[List[dict]]:
    """Alkyl tail -> C1 apolar beads, ~4 carbons per bead (Martini 3)."""
    carbons = [i for i in atom_globals if mol.GetAtomWithIdx(i).GetSymbol() == "C"]
    ordered = _order_carbon_chain(mol, carbons)
    if not ordered:
        return None
    beads = []
    for k in range(0, len(ordered), carbons_per_bead):
        grp = ordered[k:k + carbons_per_bead]
        beads.append({
            "label": f"C{k // carbons_per_bead + 1}", "bead_type": "C1",
            "atom_globals": grp, "pos": _centroid_nm(mol, conf, grp),
            "is_vsite": False, "vsite_from": None,
        })
    return beads


def map_phosphate_residue(mol, conf, atom_globals: List[int]) -> Optional[List[dict]]:
    """Phosphate group -> one charged PO4 bead (Q5, -1) (Martini 3)."""
    if not atom_globals:
        return None
    return [{
        "label": "PO4", "bead_type": "Q5", "atom_globals": list(atom_globals),
        "pos": _centroid_nm(mol, conf, list(atom_globals)),
        "is_vsite": False, "vsite_from": None,
    }]


def map_petn_residue(mol, conf, atom_globals: List[int]) -> Optional[List[dict]]:
    """Phosphoethanolamine -> PO4 bead (Q5,-1) + amine bead (Q4,+1) (Martini 3 PE)."""
    if not atom_globals:
        return None
    aset = set(atom_globals)
    phos = set()
    for i in atom_globals:
        if mol.GetAtomWithIdx(i).GetSymbol() == "P":
            phos.add(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetSymbol() == "O" and nb.GetIdx() in aset:
                    phos.add(nb.GetIdx())
    amine = [i for i in atom_globals if i not in phos]  # ethanolamine C-C-N
    beads = [{
        "label": "PO4", "bead_type": "Q5", "atom_globals": sorted(phos),
        "pos": _centroid_nm(mol, conf, sorted(phos)),
        "is_vsite": False, "vsite_from": None,
    }]
    if amine:
        beads.append({
            "label": "NH3", "bead_type": "Q4", "atom_globals": amine,
            "pos": _centroid_nm(mol, conf, amine),
            "is_vsite": False, "vsite_from": None,
        })
    return beads


def atom_name_to_global_by_residue(mol) -> Dict[tuple, Dict[str, int]]:
    """Group atoms by residue and index them by their MonomerInfo atom name.

    Returns {(resname, resnum): {atom_name: global_idx}} using the names
    assigned by pdb_export.  Only heavy atoms are indexed (H are picked up via
    bond neighbours during centroid calculation).
    """
    out: Dict[tuple, Dict[str, int]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        info = atom.GetMonomerInfo()
        if info is None:
            continue
        key = (info.GetResidueName().strip(), info.GetResidueNumber())
        out.setdefault(key, {})[info.GetName().strip()] = atom.GetIdx()
    return out


# ---------------------------------------------------------------------------
# Full-structure CG bead table + .gro emitter
#
# INVARIANT: coordinates are ALWAYS our own (each bead = centroid of our
# positioned atoms).  Polyply supplies only the topology/order.  This emitter
# walks residues in a caller-supplied order (the SAME order handed to polyply
# gen_itp), so the .gro particle order matches polyply's .itp exactly.
# ---------------------------------------------------------------------------

def build_cg_beads(mol, residue_specs: List[dict]) -> List[dict]:
    """Map an ordered list of residues to a flat, ordered bead table.

    ``residue_specs`` is an ordered list; each item is one of:
      {'kind': 'sugar', 'resname': str, 'resnum': int}
      {'kind': 'aa',    'resname': str, 'resnum': int, 'ca_idx': int}
    The order MUST match the sequence given to polyply gen_itp.

    Returns a flat list of bead dicts with a running ``index`` (0-based),
    ``resname``, ``resnum``, ``label``, ``pos`` (nm), ``is_vsite``,
    ``atom_globals``.  Coordinates come solely from ``mol``'s conformer.
    """
    conf = mol.GetConformer()
    by_res = atom_name_to_global_by_residue(mol)

    beads: List[dict] = []
    for spec in residue_specs:
        kind = spec["kind"]
        resname, resnum = spec["resname"], spec["resnum"]

        if kind == "sugar":
            name2glob = by_res.get((resname, resnum), {})
            res_beads = map_residue_published(resname, name2glob, mol, conf)
        elif kind == "aa":
            res_beads = map_amino_acid_residue(mol, conf, spec["ca_idx"], resname)
        elif kind == "lipid":
            res_beads = map_lipid_residue(mol, conf, spec["atom_globals"])
        elif kind == "phosphate":
            res_beads = map_phosphate_residue(mol, conf, spec["atom_globals"])
        elif kind == "petn":
            res_beads = map_petn_residue(mol, conf, spec["atom_globals"])
        else:
            raise ValueError(f"unknown residue kind: {kind!r}")

        if res_beads is None:
            print(f"  ! cg-map: could not map {resname}_{resnum} ({kind}); "
                  f"BEAD COUNT WILL NOT MATCH the .itp — aborting emit")
            return []

        for b in res_beads:
            beads.append({
                "index": len(beads),
                "resname": resname, "resnum": resnum,
                "label": b["label"], "bead_type": b["bead_type"],
                "pos": b["pos"], "is_vsite": b["is_vsite"],
                "atom_globals": b["atom_globals"],
            })
    return beads


def write_cg_gro(beads: List[dict], name: str, path: str,
                 box=(10.0, 10.0, 10.0)) -> str:
    """Write a GROMACS .gro for the CG bead table (positions already in nm)."""
    from pathlib import Path
    lines = [f"{name} CG (experimental coords)", f"{len(beads)}"]
    for b in beads:
        x, y, z = b["pos"]
        atom = b["label"][:5]
        lines.append(
            f"{b['resnum']:5d}{b['resname'][:5]:<5s}{atom:>5s}"
            f"{b['index'] + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"  [cg-map] wrote {path} ({len(beads)} beads)")
    return path
