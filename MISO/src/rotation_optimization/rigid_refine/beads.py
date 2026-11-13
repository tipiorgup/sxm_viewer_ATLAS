"""
beads.py: rigid bead bookkeeping for the rigid body REFINE.

A bead is one sugar unit of the exported molecule: all heavy atoms of its
residue (as named by pdb_export) plus the hydrogens attached to them.  The
bridging oxygens are not part of any bead; pdb_export assigns them to the
acceptor residue for PDB naming, but here they are free particles.

Every bead carries

    pivot      the point the bead rotates about.  This is the template COM,
               i.e. the experimental STM position, taken from chain_dict when
               available.  It is a body fixed point, so restraining it in xy
               restrains exactly the experimental position.
    body       atom coordinates relative to the pivot, in the body frame.
    face       +1 when the oriented ring normal points up, -1 when down,
               measured at the start (the face the builder assembled).

Nothing in here moves atoms.  The functions at the end compare a bead's
current heavy atoms with its template after superposition, which is the
acceptance test of the whole rigid REFINE: that number must stay at
numerical zero.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from ..structure import pdb_export

RING_ORDER = ('O5', 'C1', 'C2', 'C3', 'C4', 'C5')


@dataclass
class Bead:
    name: str                      # chain_dict unit name ('Glc_0') or residue key
    residue: str                   # residue key as in build_residue_index ('GLC_1')
    atoms: List[int]               # all atoms (heavy + H), molecule indices
    heavy: List[int]               # heavy atoms only
    ring: Optional[List[int]]      # [O5, C1, C2, C3, C4, C5] or None
    pivot: np.ndarray              # (3,) rotation centre, template COM
    body: np.ndarray               # (n, 3) atom coords relative to pivot at start
    masses: np.ndarray             # (n,)
    inertia: float                 # sum m |r|^2 (isotropic scale for steps)
    face: int = 0                  # +1 / -1 / 0 at start
    brightness: float = 0.0
    atom_names: Dict[int, str] = field(default_factory=dict)


@dataclass
class BeadSet:
    beads: List[Bead]
    free_atoms: List[int]          # bridging O, non sugar atoms, their H
    fixed_atoms: List[int]
    bridging_oxygens: List[int]
    n_atoms: int

    def bead_of_atom(self):
        out = {}
        for k, b in enumerate(self.beads):
            for i in b.atoms:
                out[i] = k
        return out


# ---------------------------------------------------------------------------
# Residue naming helpers
# ---------------------------------------------------------------------------

def unit_to_residue_keys(unit_names: Sequence[str], resname_overrides=None) -> Dict[str, str]:
    """chain_dict unit name -> residue key used by pdb_export.build_residue_index.

    Mirrors assign_sugar_residues: base name -> 3 letter resname, number + 1.
    """
    resnames = pdb_export.build_resname_map(list(unit_names), resname_overrides)
    keys = {}
    for position, name in enumerate(unit_names):
        base, number = pdb_export._base_and_number(name, position)
        keys[name] = f"{resnames[base]}_{number + 1}"
    return keys


def atom_name(atom) -> str:
    info = atom.GetMonomerInfo()
    return info.GetName().strip() if info is not None else ''


def residue_key(atom) -> Optional[str]:
    info = atom.GetMonomerInfo()
    if info is None:
        return None
    return f"{info.GetResidueName().strip()}_{info.GetResidueNumber()}"


def find_bridging_oxygens(mol) -> List[int]:
    """Oxygens bonded to heavy atoms of two different residues."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O':
            continue
        keys = {residue_key(n) for n in atom.GetNeighbors() if n.GetSymbol() != 'H'}
        keys.discard(None)
        if len(keys) >= 2:
            out.append(atom.GetIdx())
    return out


# ---------------------------------------------------------------------------
# Bead construction
# ---------------------------------------------------------------------------

def oriented_normal(coords: np.ndarray) -> Optional[np.ndarray]:
    """Unit normal from the sum of cross products of consecutive centred ring
    positions.  Same convention as rotation_optimization.oriented_ring_normal,
    so face labels agree with the builder."""
    centred = coords - coords.mean(axis=0)
    n = np.zeros(3)
    for i in range(len(centred)):
        n += np.cross(centred[i], centred[(i + 1) % len(centred)])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else None


def build_beads(mol, residue_index, chain_dict=None, fixed_atoms=(),
                sugar_chain=pdb_export.SUGAR_CHAIN, verbose=True) -> BeadSet:
    """
    Group the atoms of *mol* (with hydrogens and residue info) into rigid
    sugar beads, free atoms and fixed atoms.

    residue_index : dict from pdb_export.build_residue_index(mol) on this
        molecule (hydrogens included).
    chain_dict : optional, the builder's unit dicts.  Gives each bead its
        experimental COM as pivot and its brightness; without it the pivot is
        the heavy atom mass centre of the kept atoms.
    """
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
    mass = np.array([a.GetMass() for a in mol.GetAtoms()])

    bridging = set(find_bridging_oxygens(mol))
    fixed = set(int(i) for i in fixed_atoms)

    key_to_unit = {}
    if chain_dict:
        for unit, key in unit_to_residue_keys(list(chain_dict.keys())).items():
            key_to_unit[key] = unit

    beads = []
    in_bead = set()
    for key, entry in residue_index.items():
        if entry.get('chain') != sugar_chain:
            continue
        atoms = [i for i in entry['atom_indices'] if i not in bridging and i not in fixed]
        if not atoms:
            continue
        heavy = [i for i in atoms if mol.GetAtomWithIdx(i).GetSymbol() != 'H']
        names = {i: atom_name(mol.GetAtomWithIdx(i)) for i in atoms}
        by_name = {v: k for k, v in names.items() if v}
        ring = [by_name[n] for n in RING_ORDER] if all(n in by_name for n in RING_ORDER) else None

        unit = key_to_unit.get(key, key)
        brightness = 0.0
        if chain_dict and unit in chain_dict and chain_dict[unit].get('COM') is not None:
            pivot = np.asarray(chain_dict[unit]['COM'], dtype=float)
            brightness = float(chain_dict[unit].get('brightness', 0.0) or 0.0)
        else:
            w = mass[heavy]
            pivot = (pos[heavy] * w[:, None]).sum(axis=0) / w.sum()

        body = pos[atoms] - pivot
        m = mass[atoms]
        face = 0
        if ring is not None:
            n = oriented_normal(pos[ring])
            if n is not None and abs(n[2]) > 1e-6:
                face = 1 if n[2] > 0 else -1
        beads.append(Bead(name=unit, residue=key, atoms=atoms, heavy=heavy, ring=ring,
                          pivot=pivot.copy(), body=body, masses=m,
                          inertia=float((m * (body ** 2).sum(axis=1)).sum()),
                          face=face, brightness=brightness, atom_names=names))
        in_bead.update(atoms)

    beads.sort(key=lambda b: b.residue)
    free = [i for i in range(n_atoms) if i not in in_bead and i not in fixed]
    bs = BeadSet(beads=beads, free_atoms=free, fixed_atoms=sorted(fixed),
                 bridging_oxygens=sorted(bridging), n_atoms=n_atoms)
    if verbose:
        print(f"  Beads: {len(beads)} sugar units "
              f"({sum(len(b.atoms) for b in beads)} atoms), "
              f"free atoms: {len(free)} (bridging O: {len(bridging)}), "
              f"fixed atoms: {len(fixed)}")
        for b in beads:
            print(f"    {b.name:<10} {b.residue:<8} atoms={len(b.atoms):3d} "
                  f"heavy={len(b.heavy):2d} ring={'yes' if b.ring else 'no '} "
                  f"face={'+' if b.face > 0 else '-' if b.face < 0 else '?'} "
                  f"pivot=({b.pivot[0]:.2f}, {b.pivot[1]:.2f}, {b.pivot[2]:.2f})")
    return bs


# ---------------------------------------------------------------------------
# Linkages (for the phi restraint and the stretch report)
# ---------------------------------------------------------------------------

def find_linkages(mol, beadset: BeadSet, linkage_definitions=None, verbose=True):
    """
    One entry per bridging oxygen:
        o, cd (donor carbon, named C1 or the anomeric side), ca (acceptor
        carbon), o5 (donor ring oxygen), donor/acceptor bead indices,
        anomer ('alpha'/'beta'/None from linkage_definitions, matched by
        unit names), name.
    """
    atom_bead = beadset.bead_of_atom()
    anomer_by_pair = {}
    for link in (linkage_definitions or []):
        if len(link) >= 5:
            anomer_by_pair[(link[0], link[2])] = (str(link[4]).lower(),
                                                 link[5] if len(link) > 5 else f"{link[0]}-{link[2]}")
    out = []
    for o in beadset.bridging_oxygens:
        atom = mol.GetAtomWithIdx(o)
        carbons = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
        if len(carbons) != 2:
            continue
        names = [atom_name(mol.GetAtomWithIdx(c)) for c in carbons]
        if names[0] == 'C1' and names[1] != 'C1':
            cd, ca = carbons
        elif names[1] == 'C1' and names[0] != 'C1':
            ca, cd = carbons
        else:
            cd, ca = carbons
        bd, ba = atom_bead.get(cd), atom_bead.get(ca)
        if bd is None or ba is None:
            continue
        donor = beadset.beads[bd]
        o5 = next((i for i in donor.atoms if donor.atom_names.get(i) == 'O5'), None)
        if o5 is None and donor.ring:
            o5 = donor.ring[0]
        anomer, lname = anomer_by_pair.get((donor.name, beadset.beads[ba].name), (None, None))
        out.append({'o': o, 'cd': cd, 'ca': ca, 'o5': o5, 'donor': bd, 'acceptor': ba,
                    'anomer': anomer,
                    'name': lname or f"{donor.name}.{names[0] if cd == carbons[0] else names[1]}"
                                     f"->{beadset.beads[ba].name}"})
    if verbose:
        print(f"  Linkages: {len(out)}")
        for l in out:
            print(f"    {l['name']:<28} donor={beadset.beads[l['donor']].name} "
                  f"acceptor={beadset.beads[l['acceptor']].name} anomer={l['anomer']}")
    return out


# ---------------------------------------------------------------------------
# Rigidity and template checks
# ---------------------------------------------------------------------------

def kabsch_rmsd(a: np.ndarray, b: np.ndarray):
    """RMSD between point sets a and b (same order) after optimal superposition.
    Returns (rmsd, max_deviation, det) with det the sign of the rotation (a
    mirror image gives det -1 before the correction, reported for diagnosis)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    A, B = a - ca, b - cb
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    Rm = Vt.T @ D @ U.T
    A2 = A @ Rm.T
    dev = np.linalg.norm(A2 - B, axis=1)
    return float(np.sqrt((dev ** 2).mean())), float(dev.max()), float(d)


def bead_rigidity(positions_now: np.ndarray, beadset: BeadSet, positions_ref: np.ndarray):
    """Per bead RMSD (all atoms) between two coordinate sets after superposition.
    Zero means the bead moved as a rigid body."""
    rows = []
    for b in beadset.beads:
        rmsd, mx, det = kabsch_rmsd(positions_now[b.atoms], positions_ref[b.atoms])
        rows.append({'bead': b.name, 'rmsd': rmsd, 'max': mx, 'mirror': det < 0})
    return rows


def _match_template(mol, heavy, pos, template_xyz, template_types, tol=0.6):
    """Greedy nearest neighbour match of a bead's heavy atoms to template atoms of
    the same element.  Returns (bead_idx_list, template_idx_list) for matched pairs."""
    used = set()
    pairs = []
    for i in heavy:
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        best, best_d = None, tol
        for t, ts in enumerate(template_types):
            if ts != sym or t in used:
                continue
            d = np.linalg.norm(pos[i] - template_xyz[t])
            if d < best_d:
                best, best_d = t, d
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def template_deviation(mol, beadset: BeadSet, chain_dict, verbose=True):
    """
    Compare each bead's heavy atoms with its template (chain_dict absolute
    coordinates) after superposition.  Matching is by element and proximity,
    so it works whatever atom removals happened at export.  Reports RMSD, the
    largest atom deviation, whether the best superposition needed a mirror
    (an inverted unit), and how many template atoms were matched.
    """
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    rows = []
    for b in beadset.beads:
        md = chain_dict.get(b.name) if chain_dict else None
        if md is None:
            rows.append({'bead': b.name, 'rmsd': None})
            continue
        txyz = np.asarray(md['absolute_coordinates'], float)
        ttypes = list(md['atom_types'])
        pairs = _match_template(mol, b.heavy, pos, txyz, ttypes)
        if len(pairs) < 4:
            rows.append({'bead': b.name, 'rmsd': None, 'matched': len(pairs)})
            continue
        a = pos[[p[0] for p in pairs]]
        t = txyz[[p[1] for p in pairs]]
        rmsd, mx, det = kabsch_rmsd(a, t)
        rows.append({'bead': b.name, 'rmsd': rmsd, 'max': mx, 'mirror': det < 0,
                     'matched': len(pairs), 'template_atoms': len(ttypes),
                     'unmatched_heavy': len(b.heavy) - len(pairs)})
    if verbose:
        print("  Template deviation (heavy atoms, after superposition):")
        for r in rows:
            if r.get('rmsd') is None:
                print(f"    {r['bead']:<10} no template match")
            else:
                print(f"    {r['bead']:<10} rmsd={r['rmsd']:.3f} A  max={r['max']:.3f} A  "
                      f"matched {r['matched']}/{r['template_atoms']}"
                      f"{'  MIRROR' if r['mirror'] else ''}")
    return rows


def ring_pucker(pos: np.ndarray, ring: Sequence[int]):
    """Cremer Pople Q, theta (deg) for a six membered ring given in order
    O5, C1 .. C5.  The Fourier index runs over the standard sequence
    C1, C2, C3, C4, C5, O5, so theta near 0 is 4C1 and near 180 is 1C4."""
    ring = list(ring)
    coords = pos[ring[1:] + ring[:1]]
    centre = coords.mean(axis=0)
    rc = coords - centre
    n = oriented_normal(coords)
    if n is None:
        return 0.0, 0.0
    z = rc @ n
    N = 6
    j = np.arange(N)
    q2c = np.sqrt(2.0 / N) * (z * np.cos(2 * np.pi * 2 * j / N)).sum()
    q2s = -np.sqrt(2.0 / N) * (z * np.sin(2 * np.pi * 2 * j / N)).sum()
    q3 = np.sqrt(1.0 / N) * (z * np.cos(np.pi * j)).sum()
    q2 = np.hypot(q2c, q2s)
    Q = np.sqrt(q2 ** 2 + q3 ** 2)
    theta = np.degrees(np.arctan2(q2, q3)) if Q > 1e-9 else 0.0
    return float(Q), float(theta)
