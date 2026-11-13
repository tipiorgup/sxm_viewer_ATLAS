"""
face_screening.py: which faces each sugar can present, given the STM layout.

A face state fixes, for every ring, which face points up (+z).  In a
branched glycan most states are geometrically impossible: a sugar with
several linkages has to point C1 at its parent and C3/C4/C6 at its children
with ONE rigid orientation, and seen from above the order of those carbons
around the ring reverses when the sugar is turned over.  At a branch point
only one face usually fits the layout.

screen_faces tests every sugar on its own (a "star" test): over many
orientations that present a given face, is there one where every linkage
carbon of this sugar is within reach of its partner?  Reach of a partner =
distance of the partner's linkage carbon from the partner COM + one ideal
C...C bridge.  It needs only the templates and the COMs, costs milliseconds,
and is a necessary condition: a face it rejects cannot be built.

Measured on a synthetic branched tree of 15 glucose units (C4 chain, C3 and
C6 branches): it fixed the face of 4 sugars and left 2048 of 32768 states.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rotation_optimization import ring_atom_order, oriented_ring_normal

IDEAL_BRIDGE_CC = 2.44   # C...C across a C-O-C bridge: 1.43 A bonds, 117 degrees


def _linkages_by_sugar(linkages, molecule_data_dict):
    """{sugar: [(own carbon, partner name or None, partner position, partner carbon)]}."""
    out = {}
    for link in linkages:
        donor, dc, acceptor, ac = link[0], link[1], link[2], link[3]
        if donor not in molecule_data_dict:
            continue
        if isinstance(acceptor, str):
            if acceptor not in molecule_data_dict:
                continue
            out.setdefault(donor, []).append((dc, acceptor, None, ac))
            out.setdefault(acceptor, []).append((ac, donor, None, dc))
        else:
            out.setdefault(donor, []).append((dc, None, np.asarray(acceptor, float)[:3], None))
    return out


def carbon_reach(mol_data, carbon):
    """Distance of a named carbon from the molecule's COM (0 if unknown)."""
    cmap = mol_data.get('carbon_map', {})
    if carbon not in cmap:
        return 0.0
    xyz = np.asarray(mol_data['absolute_coordinates'], dtype=float)
    return float(np.linalg.norm(xyz[cmap[carbon]] - np.asarray(mol_data['COM'], float)))


def screen_faces(molecule_data_dict, linkages, n_samples=4000, min_up=0.3,
                 tolerance=0.3, bridge=IDEAL_BRIDGE_CC, seed=0):
    """
    Faces each sugar can present while reaching all its linkage partners.

    Parameters
    ----------
    linkages : normalised linkages [donor, donor_C, acceptor, acceptor_C, ...]
        (acceptor may be a coordinate).
    min_up : float
        A face counts as presented when n_z * face > min_up (n = oriented
        ring normal), i.e. the ring is tilted less than about 72 degrees.
    tolerance : float (A)
        Slack allowed on top of the reach before a face is rejected.

    Returns
    -------
    dict  {sugar: {'valid': [faces], 'slack': {+1: s, -1: s},
                   'n_linkages': k, 'note': str}}
        slack = smallest achievable "worst linkage excess" for that face
        (<= 0 means all linkages reachable at once).  Sugars without a ring
        or without linkages get both faces.  If NEITHER face passes, both are
        kept and note='neither face fits': that points at the layout or the
        linkage hypothesis rather than at the face.
    """
    rots = R.random(n_samples, random_state=seed)
    by_sugar = _linkages_by_sugar(linkages, molecule_data_dict)
    result = {}

    for name, data in molecule_data_dict.items():
        links = by_sugar.get(name, [])
        n0 = oriented_ring_normal(data) if ring_atom_order(data) is not None else None
        if n0 is None or not links:
            result[name] = {'valid': [1, -1], 'slack': {1: 0.0, -1: 0.0},
                            'n_linkages': len(links), 'note': 'no ring' if n0 is None else 'no linkage'}
            continue

        com = np.asarray(data['COM'], dtype=float)
        rel = np.asarray(data['absolute_coordinates'], dtype=float) - com
        nz = rots.apply(n0)[:, 2]
        slack = {}
        for face in (1, -1):
            keep = rots[nz * face > min_up]
            worst = np.full(len(keep), -np.inf)
            for own, partner, ppos, pc in links:
                if own not in data['carbon_map']:
                    continue
                X = keep.apply(rel[data['carbon_map'][own]]) + com
                if partner is not None:
                    target = np.asarray(molecule_data_dict[partner]['COM'], float)
                    allowed = carbon_reach(molecule_data_dict[partner], pc) + bridge
                else:
                    target, allowed = ppos, bridge
                worst = np.maximum(worst, np.linalg.norm(X - target, axis=1) - allowed)
            slack[face] = float(worst.min()) if len(worst) else np.inf

        valid = [f for f in (1, -1) if slack[f] <= tolerance]
        note = ''
        if not valid:
            valid, note = [1, -1], 'neither face fits'
        elif len(valid) == 1:
            note = 'face fixed by layout'
        result[name] = {'valid': valid, 'slack': slack, 'n_linkages': len(links), 'note': note}
    return result


def count_states(mol_order, allowed):
    """Number of face states allowed by a {sugar: [faces]} map."""
    return int(np.prod([len(allowed.get(m, [1, -1])) for m in mol_order], dtype=object))
