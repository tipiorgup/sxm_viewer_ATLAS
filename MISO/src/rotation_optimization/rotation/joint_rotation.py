"""
joint_rotation.py: close all glycosidic linkages at once by rotating every
sugar about its fixed experimental COM.

The sequential builder places one sugar at a time and never revisits it, so
in a branched glycan a sugar is fixed before its children are placed and
their linkages often cannot close (4 to 7 of 14 open on a 15 sugar test even
in geometrically valid face states).  This pass treats every sugar as a rigid
body with a pinned COM and three rotational degrees of freedom, and
minimises, over all sugars together,

    E = k_link   * sum_linkages  (|C_donor - C_acceptor| - d0)^2
      + k_bridge * sum_linkages  |O_donor - O_acceptor|^2
      + k_face   * sum_sugars    max(0, margin - face * n_z)^2
      + k_clash  * sum_pairs     max(0, limit - |x_i - x_j|)^2

O_donor and O_acceptor are the two hydroxyl oxygens the bridge replaces.
Bringing them together puts both carbons in a tetrahedral arrangement
around the future bridging O; closing C...C alone can leave the bridge
pointing where a hydrogen belongs, which REFINE then resolves by inverting
the carbon (seen on a 1->6 linkage of a 15 sugar glycan).

d0 is the ideal C...C distance across a C-O-C bridge.  Rings cannot deform
(rigid templates) and COMs cannot move, so puckers and experimental positions
are kept exactly.  The face term keeps the requested face state; when a state
is geometrically impossible the optimiser has to give up a face or leave a
linkage open, and the state is reported as rejected.

Gradients are analytic: the torque on a sugar is sum (x_i - COM) x dE/dx_i,
plus (dE/dn_z) (n x z) for the face term.  Descent is steepest descent on
the rotation vectors with backtracking, each step re-based on the current
orientation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rotation_optimization import ring_atom_order, oriented_ring_normal, face_sign
from ..geometry.geometry_utils import rotate_molecule_around_com

IDEAL_BRIDGE_CC = 2.44          # 1.43 A C-O bonds at 117 degrees
CLOSABLE_CC = 2 * 1.43          # beyond this no C-O-C bridge exists


def _leaving_oxygen_index(mol_data, carbon_name):
    """Index of the hydroxyl O on carbon_name that the bridge replaces, or None."""
    cmap = mol_data.get('carbon_map', {})
    if carbon_name not in cmap:
        return None
    oh_map = mol_data.get('oh_map')
    if oh_map and carbon_name in oh_map and oh_map[carbon_name][0] is not None:
        return int(oh_map[carbon_name][0])
    types = mol_data.get('atom_types')
    if types is None:
        return None
    xyz = np.asarray(mol_data['absolute_coordinates'], dtype=float)
    c = xyz[cmap[carbon_name]]
    best, best_d = None, 1.6
    for i, t in enumerate(types):
        if t != 'O' or i == cmap.get('ring_oxygen'):
            continue
        d = np.linalg.norm(xyz[i] - c)
        if 0.5 < d < best_d:
            best, best_d = i, d
    return best


def joint_rigid_rotation(chain_dict, linkages, face_state=None,
                         d0=IDEAL_BRIDGE_CC, k_link=10.0, k_bridge=0.5, k_face=20.0, k_clash=1.0,
                         face_margin=0.3, clash_limit=2.9, bonded_clash_limit=2.5,
                         max_iter=4000, tol=1e-7, max_step=0.1,
                         max_gap=3.0, min_gap=2.1, verbose=True):
    """
    Rotate every sugar in chain_dict about its COM to close the linkages.

    Parameters
    ----------
    chain_dict : {name: mol_data}   positioned monomers (not modified)
    linkages   : normalised linkages [donor, donor_C, acceptor, acceptor_C, ...]
                 Molecule to molecule linkages are closed; coordinate linkages
                 are only prevented from opening further than they are now.
    face_state : {name: +1 or -1} or None.  None disables the face term.
    max_gap    : linkage C...C above which the result is rejected (A).  2.86
                 is the strict limit for a 1.43 A bridge; REFINE closes a
                 little more.
    min_gap    : linkage C...C below which the result is rejected (A); the
                 bridge angle would be far too acute.

    Returns
    -------
    (new_chain_dict, report)
        report keys: 'valid', 'reason', 'gaps' {linkage name: A},
        'worst_gap', 'faces_kept', 'min_nonbonded' (A), 'energy',
        'energy_start', 'n_iter'
    """
    names = [n for n in chain_dict if ring_atom_order(chain_dict[n]) is not None
             or any(n in (l[0], l[2]) for l in linkages)]
    index = {n: k for k, n in enumerate(names)}
    coms = [np.asarray(chain_dict[n]['COM'], dtype=float) for n in names]
    rel = [np.asarray(chain_dict[n]['absolute_coordinates'], dtype=float) - coms[k]
           for k, n in enumerate(names)]
    normals = []
    for n in names:
        nv = oriented_ring_normal(chain_dict[n]) if ring_atom_order(chain_dict[n]) is not None else None
        normals.append(nv)

    # linkages
    mol_links, coord_links = [], []
    leaving = {k: set() for k in range(len(names))}
    for link in linkages:
        donor, dc, acceptor, ac = link[0], link[1], link[2], link[3]
        lname = link[5] if len(link) > 5 else f"{donor}-{acceptor}"
        if donor not in index:
            continue
        kd = index[donor]
        if dc not in chain_dict[donor]['carbon_map']:
            continue
        id_ = chain_dict[donor]['carbon_map'][dc]
        lo_d = _leaving_oxygen_index(chain_dict[donor], dc)
        if lo_d is not None:
            leaving[kd].add(lo_d)
        if isinstance(acceptor, str):
            if acceptor not in index or ac not in chain_dict[acceptor]['carbon_map']:
                continue
            ka = index[acceptor]
            ia = chain_dict[acceptor]['carbon_map'][ac]
            lo_a = _leaving_oxygen_index(chain_dict[acceptor], ac)
            if lo_a is not None:
                leaving[ka].add(lo_a)
            mol_links.append((lname, kd, id_, ka, ia, lo_d, lo_a))
        else:
            target = np.asarray(acceptor, dtype=float)[:3]
            d_init = float(np.linalg.norm(rel[kd][id_] + coms[kd] - target))
            coord_links.append((lname, kd, id_, target, d_init))

    # clash atoms: heavy atoms that stay after bridging
    heavy = []
    for k, n in enumerate(names):
        types = chain_dict[n].get('atom_types')
        if types is None:
            idx = np.arange(len(rel[k]))
        else:
            idx = np.array([i for i, t in enumerate(types) if t != 'H' and i not in leaving[k]])
        heavy.append(idx)

    bonded = {}
    for lname, kd, id_, ka, ia, _, _ in mol_links:
        bonded.setdefault(frozenset((kd, ka)), []).append((kd, id_, ka, ia))
    extent = [np.linalg.norm(r, axis=1).max() for r in rel]
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if np.linalg.norm(coms[a] - coms[b]) < extent[a] + extent[b] + clash_limit:
                key = frozenset((a, b))
                lim = bonded_clash_limit if key in bonded else clash_limit
                mask = np.ones((len(heavy[a]), len(heavy[b])), dtype=bool)
                for kd, id_, ka, ia in bonded.get(key, []):
                    i_a, i_b = (id_, ia) if kd == a else (ia, id_)
                    ma = np.where(heavy[a] == i_a)[0]
                    mb = np.where(heavy[b] == i_b)[0]
                    if len(ma) and len(mb):
                        mask[ma[0], mb[0]] = False
                pairs.append((a, b, lim, mask))

    faces = [None if face_state is None else face_state.get(n) for n in names]

    def evaluate(rots):
        pos = [rel[k] @ rots[k].T + coms[k] for k in range(len(names))]
        grad = [np.zeros_like(p) for p in pos]
        e = 0.0
        face_torque = [np.zeros(3) for _ in names]
        for lname, kd, id_, ka, ia, lo_d, lo_a in mol_links:
            v = pos[kd][id_] - pos[ka][ia]
            d = max(np.linalg.norm(v), 1e-9)
            # two sided: too short is as wrong as too long (the clash term
            # skips this carbon pair, so nothing else would stop it)
            e += k_link * (d - d0) ** 2
            g = 2 * k_link * (d - d0) * v / d
            grad[kd][id_] += g
            grad[ka][ia] -= g
            if k_bridge and lo_d is not None and lo_a is not None:
                w = pos[kd][lo_d] - pos[ka][lo_a]
                e += k_bridge * float(w @ w)
                grad[kd][lo_d] += 2 * k_bridge * w
                grad[ka][lo_a] -= 2 * k_bridge * w
        for lname, kd, id_, target, d_init in coord_links:
            v = pos[kd][id_] - target
            d = np.linalg.norm(v)
            if d > d_init:
                e += k_link * (d - d_init) ** 2
                grad[kd][id_] += 2 * k_link * (d - d_init) * v / max(d, 1e-9)
        for k in range(len(names)):
            if faces[k] in (1, -1) and normals[k] is not None:
                n = rots[k] @ normals[k]
                short = face_margin - faces[k] * n[2]
                if short > 0:
                    e += k_face * short ** 2
                    dE_dnz = -2 * k_face * short * faces[k]
                    face_torque[k] += dE_dnz * np.cross(n, [0.0, 0.0, 1.0])
        for a, b, lim, mask in pairs:
            A, B = pos[a][heavy[a]], pos[b][heavy[b]]
            diff = A[:, None, :] - B[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            over = (d < lim) & mask
            if over.any():
                s = np.where(over, lim - d, 0.0)
                e += k_clash * np.sum(s ** 2)
                coef = np.where(over, -2 * k_clash * s / np.maximum(d, 1e-9), 0.0)
                g = coef[:, :, None] * diff
                np.add.at(grad[a], heavy[a], g.sum(axis=1))
                np.add.at(grad[b], heavy[b], -g.sum(axis=0))
        torques = [np.sum(np.cross(pos[k] - coms[k], grad[k]), axis=0) + face_torque[k]
                   for k in range(len(names))]
        return e, torques, pos

    rots = [np.eye(3) for _ in names]
    e, torques, pos = evaluate(rots)
    e_start = e
    alpha, it = 0.05, 0
    for it in range(1, max_iter + 1):
        steps = []
        for t in torques:
            s = -alpha * t
            n_s = np.linalg.norm(s)
            if n_s > max_step:
                s *= max_step / n_s
            steps.append(s)
        trial = [R.from_rotvec(s).as_matrix() @ rk for s, rk in zip(steps, rots)]
        e_new, t_new, p_new = evaluate(trial)
        if e_new < e:
            improvement = e - e_new
            rots, e, torques, pos = trial, e_new, t_new, p_new
            alpha = min(alpha * 1.2, 10.0)
            if improvement < tol:
                break
        else:
            alpha *= 0.5
            if alpha < 1e-10:
                break
        if e < 1e-10:
            break

    # results
    new_chain = dict(chain_dict)
    for k, n in enumerate(names):
        md = dict(chain_dict[n])
        md['COM'] = coms[k].tolist()
        md.setdefault('quaternion', [0.0, 0.0, 0.0, 1.0])
        md = rotate_molecule_around_com(md, rots[k])
        md['COM'] = coms[k].tolist()
        new_chain[n] = md

    gaps = {lname: float(np.linalg.norm(pos[kd][id_] - pos[ka][ia]))
            for lname, kd, id_, ka, ia, _, _ in mol_links}
    worst = max(gaps.values()) if gaps else 0.0
    shortest = min(gaps.values()) if gaps else d0
    faces_kept = True
    if face_state is not None:
        for n in names:
            want = face_state.get(n)
            if want in (1, -1) and normals[index[n]] is not None:
                if face_sign(new_chain[n], 0.0) != want:
                    faces_kept = False
    min_nb = np.inf
    for a, b, lim, mask in pairs:
        if frozenset((a, b)) in bonded:
            continue
        d = np.linalg.norm(pos[a][heavy[a]][:, None] - pos[b][heavy[b]][None], axis=2)
        min_nb = min(min_nb, float(d.min()))

    reasons = []
    if worst > max_gap:
        reasons.append(f"linkage open ({worst:.2f} A > {max_gap:.2f} A)")
    if shortest < min_gap:
        reasons.append(f"linkage collapsed ({shortest:.2f} A < {min_gap:.2f} A)")
    if not faces_kept:
        reasons.append("face state could not be kept")
    report = {'valid': not reasons, 'reason': '; '.join(reasons) or 'ok',
              'gaps': gaps, 'worst_gap': worst, 'shortest_gap': shortest,
              'faces_kept': faces_kept,
              'min_nonbonded': min_nb, 'energy': e, 'energy_start': e_start,
              'n_iter': it}
    if verbose:
        print(f"\nJOINT RIGID ROTATION ({len(names)} sugars, {len(mol_links)} linkages, "
              f"COMs fixed, rings rigid)")
        print(f"  energy {e_start:.3f} -> {e:.3f} in {it} steps")
        nb = f"{min_nb:.2f} A" if np.isfinite(min_nb) else "n/a"
        print(f"  linkage C...C {shortest:.2f} to {worst:.2f} A, faces kept: {faces_kept}, "
              f"closest non bonded heavy atoms {nb}")
        print(f"  state {'ACCEPTED' if report['valid'] else 'REJECTED: ' + report['reason']}")
    return new_chain, report
