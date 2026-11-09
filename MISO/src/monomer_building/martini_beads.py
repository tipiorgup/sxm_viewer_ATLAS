"""Thin adapter over the vendored Martini_mapper (``src/martini_mapper``).

Only the *mapping* path is used: no xtb coordinate generation and no file
output, so the sole dependencies are rdkit + numpy (already required).  The
mapper assigns heavy atoms to Martini beads; its atom indices are the
heavy-atom SMILES order — the same order RDKit uses *before* ``AddHs`` — so
they line up directly with the monomer template's heavy atoms.

The returned dict is meant to be stored on the conformer/monomer dict so the
chain can later be coarse-grained by taking, per bead, the (mass-weighted)
centroid of its ``atom_idx`` at the assembled coordinates.
"""

from src.martini_mapper.main import run_mapping
from src.martini_mapper.outputs import export_bead_mapping


def map_smiles_to_beads(name, smiles):
    """Coarse-grain one monomer SMILES into Martini beads.

    Parameters
    ----------
    name : str
        Compound name (used only for the mapper's bookkeeping).
    smiles : str
        Monomer SMILES.  Must be the same SMILES used to build the conformer
        so the heavy-atom indices align.

    Returns
    -------
    dict
        ``{'beads': [...], 'bonds': [...]}`` where each bead is
        ``{'index', 'bead_type', 'atom_idx', 'elements', 'mass',
        'num_connections', 'ref_pos'}`` and each bond is
        ``(bead_i, bead_j, order)`` in bead indices.  ``ref_pos`` is the
        mapper's own bead coordinate (nm) — informational only; real positions
        come from the assembled atom coordinates.
    """
    beads_raw, mapping = run_mapping(
        name, smiles, run_xtb=False, write_files=False)
    beads_data, connections_data = export_bead_mapping(
        beads_raw, mapping, smiles, name, write_file=False)

    beads = [
        {
            'index': int(b['index']),
            'bead_type': str(b['bead_type']),
            'atom_idx': [int(i) for i in b['composed_of_index']],
            'elements': [str(e) for e in b['composed_of_element']],
            'mass': float(b['mass']),
            'num_connections': int(b['num_connections']),
            'ref_pos': [float(b['x']), float(b['y']), float(b['z'])],
        }
        for b in beads_data
    ]
    bonds = [
        (int(c['index']), int(c['connection_index']), float(c['bond_type']))
        for c in connections_data
    ]
    return {'beads': beads, 'bonds': bonds}
