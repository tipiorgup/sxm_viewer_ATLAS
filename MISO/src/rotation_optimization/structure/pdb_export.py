"""
Residue-aware PDB export.

The SDF exporters merge every monomer into one flat molecule with renumbered
atoms.  PDB, unlike SDF, can carry the unit structure: each sugar becomes its
own residue with a name, a number, and chemically meaningful atom names
(C1..C6, O5, O1..O6).

This matters beyond readability.  Residue-based coarse-graining
(martinize2/vermouth) maps a structure residue by residue, so a PDB with
correctly named units is the input format the Martini stage needs, and the
unit -> chain/resnum map is what lets beads be traced back to the monomer they
came from when the templates are fitted back on.

Assignment happens in two passes, because the exporter mutates the molecule:

  assign_sugar_residues()  -- call straight after combine_molecules(), while
      carbon_map's local indices are still valid.  The glycosidic condensation
      later *removes* the leaving hydroxyls, which shifts every local index
      (see the stale-offset warning in saving_lps.add_phosphate_bonds).  RDKit
      keeps MonomerInfo attached to the atom object, so info assigned now
      survives those removals for free.

  finalize_residue_info()  -- call just before returning the molecule.  Picks
      up everything that did not exist during the first pass: the bridging
      oxygens created by the condensation, plus phosphate, PEtN, lipid tails
      and peptide grown on afterwards.
"""

import json
from collections import deque

from rdkit import Chem

# Sugars on chain A; anything grown onto them (lipid, PEtN, phosphate, peptide)
# on chain X, so a CG mapper can tell them apart.
SUGAR_CHAIN = 'A'
OTHER_CHAIN = 'X'


def _base_and_number(unit_name, fallback_number):
    """'Glc_0' -> ('Glc', 0).  'GalA' -> ('GalA', fallback_number)."""
    if '_' in unit_name:
        base, _, suffix = unit_name.rpartition('_')
        if suffix.isdigit():
            return base, int(suffix)
    return unit_name, fallback_number


def build_resname_map(unit_names, overrides=None):
    """Map each sugar type to a 3-character PDB residue name.

    PDB's resName field is three columns, so 'GlcN' and 'GlcA' both want to
    become 'GLC'.  Silent truncation would merge two chemically different
    sugars into one residue type, so collisions are resolved and reported.
    `overrides` (YAML `pdb_residue_names:`) always wins.
    """
    overrides = {k: v.upper()[:3] for k, v in (overrides or {}).items()}

    bases = []
    for name in unit_names:
        base, _ = _base_and_number(name, 0)
        if base not in bases:
            bases.append(base)

    resnames = {}
    used = set(overrides.values())
    for base in bases:
        if base in overrides:
            resnames[base] = overrides[base]
            continue

        candidate = base.upper()[:3].ljust(3, 'X')
        if candidate in used:
            stem = candidate[:2]
            for filler in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                alt = stem + filler
                if alt not in used:
                    print(f"  ! PDB resname collision: '{base}' wants '{candidate}', "
                          f"already taken; using '{alt}'. Pin it explicitly with "
                          f"`pdb_residue_names: {{{base}: XXX}}` in the YAML.")
                    candidate = alt
                    break
            else:
                raise ValueError(f"no free 3-char PDB resname left for '{base}'")

        used.add(candidate)
        resnames[base] = candidate

    return resnames


def _set_atom_info(atom, name, resname, resnum, chain, hetero):
    info = Chem.AtomPDBResidueInfo()
    # PDB atom names occupy 4 columns; single-letter elements conventionally
    # start in column 14, hence the leading space.
    info.SetName(f" {name:<3}"[:4])
    info.SetResidueName(resname)
    info.SetResidueNumber(int(resnum))
    info.SetChainId(chain)
    info.SetIsHeteroAtom(hetero)
    info.SetOccupancy(1.0)
    info.SetTempFactor(0.0)
    atom.SetMonomerInfo(info)


def _sugar_atom_names(mol, carbon_map, local_indices):
    """Chemically meaningful atom names for one monomer.

    Ring carbons come from carbon_map, the ring oxygen is O5 (pyranose
    convention), and each hydroxyl oxygen is named after the carbon it hangs
    off (C3 -> O3) by walking the bond graph rather than trusting a
    precomputed index.  Leftovers get element+counter.  Names are unique within
    the residue, which PDB requires.
    """
    names = {}
    ring_oxygen = None

    for key, idx in (carbon_map or {}).items():
        if not isinstance(idx, int) or idx not in local_indices:
            continue
        if key == 'ring_oxygen':
            names[idx] = 'O5'
            ring_oxygen = idx
        elif key.startswith('C') and key[1:].isdigit():
            names[idx] = key

    # Hydroxyl / exocyclic oxygens, named after their carbon.
    for idx in [i for i in list(names) if names[i].startswith('C')]:
        carbon_number = names[idx][1:]
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            n_idx = neighbor.GetIdx()
            if (neighbor.GetSymbol() == 'O' and n_idx != ring_oxygen
                    and n_idx in local_indices and n_idx not in names):
                names[n_idx] = f"O{carbon_number}"
                break

    taken = set(names.values())
    counters = {}
    for idx in local_indices:
        if idx in names:
            continue
        symbol = mol.GetAtomWithIdx(idx).GetSymbol().upper()
        while True:
            counters[symbol] = counters.get(symbol, 0) + 1
            candidate = f"{symbol}{counters[symbol]}"
            if candidate not in taken:
                break
        names[idx] = candidate
        taken.add(candidate)

    return names


def assign_sugar_residues(mol, processed_mols, resname_overrides=None):
    """First pass: name the sugar units.

    MUST be called before the glycosidic condensation removes atoms, while
    carbon_map's local indices still line up with atom_offset.
    """
    unit_names = [entry['name'] for entry in processed_mols]
    resnames = build_resname_map(unit_names, resname_overrides)

    for position, entry in enumerate(processed_mols):
        offset = entry['atom_offset']
        count = entry['num_atoms']
        base, number = _base_and_number(entry['name'], position)
        resname = resnames[base]
        resnum = number + 1  # PDB residue numbers are 1-based

        local_indices = list(range(offset, offset + count))
        if local_indices and local_indices[-1] >= mol.GetNumAtoms():
            print(f"  ! {entry['name']}: atom range {local_indices[0]}..{local_indices[-1]} "
                  f"exceeds molecule ({mol.GetNumAtoms()} atoms); skipping")
            continue

        carbon_map = {}
        for key, value in (entry.get('carbon_map') or {}).items():
            if isinstance(value, int):
                carbon_map[key] = value + offset

        names = _sugar_atom_names(mol, carbon_map, local_indices)
        for idx in local_indices:
            _set_atom_info(mol.GetAtomWithIdx(idx), names[idx],
                           resname, resnum, SUGAR_CHAIN, hetero=False)

    print(f"  ✓ Named {len(unit_names)} sugar residues: "
          f"{', '.join(f'{n}->{resnames[_base_and_number(n, 0)[0]]}' for n in unit_names[:6])}"
          f"{' ...' if len(unit_names) > 6 else ''}")


def _classify_fragment(mol, indices):
    """Guess a residue name for a non-sugar fragment from its composition."""
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in indices]
    if 'P' in symbols and 'N' in symbols:
        return 'PEN'                      # phosphoethanolamine
    if 'P' in symbols:
        return 'PO4'
    if symbols.count('C') >= 8:
        return 'LIP'                      # lipid tail
    if 'N' in symbols:
        return 'PEP'
    return 'LIG'


def _assign_bridging_oxygens(mol):
    """The condensation creates a new O bonded to two sugars; name it.

    By convention the glycosidic oxygen belongs to the acceptor and is named
    after the acceptor carbon (a 1->4 linkage gives O4).  The donor side is
    identifiable because its carbon is the anomeric C1.
    """
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O' or atom.GetMonomerInfo() is not None:
            continue

        carbons = [n for n in atom.GetNeighbors()
                   if n.GetSymbol() == 'C' and n.GetMonomerInfo() is not None]
        if len(carbons) != 2:
            continue

        names = [c.GetMonomerInfo().GetName().strip() for c in carbons]
        # The acceptor is whichever side is *not* the anomeric carbon.
        if names[0] == 'C1' and names[1] != 'C1':
            acceptor = carbons[1]
        elif names[1] == 'C1' and names[0] != 'C1':
            acceptor = carbons[0]
        else:
            acceptor = carbons[0]   # 1->1 linkage or unnamed: pick a side

        parent = acceptor.GetMonomerInfo()
        carbon_number = parent.GetName().strip()[1:]
        _set_atom_info(atom, f"O{carbon_number}" if carbon_number else 'O',
                       parent.GetResidueName(), parent.GetResidueNumber(),
                       parent.GetChainId(), hetero=False)
        count += 1

    if count:
        print(f"  ✓ Assigned {count} bridging oxygen(s) to their acceptor residue")


def _unassigned_fragments(mol):
    """Connected components among atoms that still have no residue.

    Lipid tails and PEtN are bonded *into* the sugars, so GetMolFrags would
    swallow them into one fragment.  Walking the bond graph but refusing to
    cross into an already-named atom isolates each grown-on group as its own
    residue -- a lipid tail must not end up as part of a sugar residue.
    """
    fragments = []
    seen = set()

    for atom in mol.GetAtoms():
        start = atom.GetIdx()
        if atom.GetMonomerInfo() is not None or start in seen:
            continue

        component = []
        queue = deque([start])
        seen.add(start)
        while queue:
            idx = queue.popleft()
            component.append(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                n_idx = neighbor.GetIdx()
                if (neighbor.GetMonomerInfo() is not None
                        or n_idx in seen):
                    continue
                seen.add(n_idx)
                queue.append(n_idx)

        fragments.append(sorted(component))

    return fragments


def finalize_residue_info(mol):
    """Second pass: name everything created after assign_sugar_residues().

    Returns the unit -> residue map, built by reading the atoms back, so it
    always reflects final atom indices rather than the exporter's bookkeeping.
    """
    _assign_bridging_oxygens(mol)

    existing = [a.GetMonomerInfo().GetResidueNumber()
                for a in mol.GetAtoms() if a.GetMonomerInfo() is not None]
    next_resnum = (max(existing) if existing else 0) + 1

    for fragment in _unassigned_fragments(mol):
        resname = _classify_fragment(mol, fragment)
        counters = {}
        for idx in fragment:
            symbol = mol.GetAtomWithIdx(idx).GetSymbol().upper()
            counters[symbol] = counters.get(symbol, 0) + 1
            _set_atom_info(mol.GetAtomWithIdx(idx), f"{symbol}{counters[symbol]}",
                           resname, next_resnum, OTHER_CHAIN, hetero=True)
        print(f"  ✓ {resname} {next_resnum}: {len(fragment)} atoms")
        next_resnum += 1

    return build_residue_index(mol)


def build_residue_index(mol):
    """unit key -> {resname, resnum, chain, atom_indices}, read off the atoms."""
    index = {}
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info is None:
            continue
        key = f"{info.GetResidueName().strip()}_{info.GetResidueNumber()}"
        entry = index.setdefault(key, {
            'resname': info.GetResidueName().strip(),
            'resnum': info.GetResidueNumber(),
            'chain': info.GetChainId(),
            'atom_indices': [],
        })
        entry['atom_indices'].append(atom.GetIdx())
    return index


def propagate_residue_info_to_h(mol_no_h, mol_with_h):
    """Carry residue info across the openbabel hydrogen-addition step.

    Open Babel round-trips through a file format, which drops the residue info,
    but it preserves heavy-atom order and appends hydrogens at the end.  Heavy
    atoms are copied across by index; each hydrogen inherits the residue of the
    heavy atom it is bonded to.
    """
    n_heavy = mol_no_h.GetNumAtoms()
    if mol_with_h.GetNumAtoms() < n_heavy:
        print("  ! H addition returned fewer atoms than the heavy-atom structure; "
              "skipping residue propagation")
        return mol_with_h

    for idx in range(n_heavy):
        source = mol_no_h.GetAtomWithIdx(idx)
        target = mol_with_h.GetAtomWithIdx(idx)
        if source.GetSymbol() != target.GetSymbol():
            print(f"  ! atom {idx} changed element ({source.GetSymbol()} -> "
                  f"{target.GetSymbol()}) during H addition; heavy-atom order is "
                  f"not preserved, residue info would be misaligned — aborting "
                  f"propagation")
            return mol_with_h
        info = source.GetMonomerInfo()
        if info is not None:
            target.SetMonomerInfo(info)

    h_counters = {}
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() != 'H' or atom.GetMonomerInfo() is not None:
            continue
        heavy = next((n for n in atom.GetNeighbors() if n.GetSymbol() != 'H'), None)
        if heavy is None or heavy.GetMonomerInfo() is None:
            continue
        parent = heavy.GetMonomerInfo()
        key = heavy.GetIdx()
        h_counters[key] = h_counters.get(key, 0) + 1
        _set_atom_info(atom, f"H{parent.GetName().strip()}{h_counters[key]}"[:4],
                       parent.GetResidueName(), parent.GetResidueNumber(),
                       parent.GetChainId(), parent.GetIsHeteroAtom())

    return mol_with_h


def write_residue_pdb(mol, filename, residue_index=None):
    """Write a PDB with per-unit residues, plus a JSON sidecar of the mapping.

    The sidecar is not decoration: after coarse-graining, each bead has to be
    traced back to the monomer it came from so the right template can be fitted
    onto it.  chain+resnum is that link.
    """
    path = filename if filename.endswith('.pdb') else f"{filename}.pdb"

    # flavor=8: one CONECT record per bond instead of repeating records to encode
    # bond order.  These are non-standard residues, so CONECT is what tells a
    # reader where the bonds actually are.
    Chem.MolToPDBFile(mol, path, flavor=8)
    print(f"  ✓ Wrote {path}")

    if residue_index is None:
        residue_index = build_residue_index(mol)

    json_path = path[:-4] + '_residues.json'
    with open(json_path, 'w') as handle:
        json.dump(residue_index, handle, indent=2)
    print(f"  ✓ Wrote {json_path} ({len(residue_index)} residues)")

    return path
