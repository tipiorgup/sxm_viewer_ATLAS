"""
rigid_relax.py — bonded-only spring relaxation of a glycan chain.

The pass repairs deformed structures *before* the slab-compression MD.  Each
sugar monomer is a RIGID bead: its centre of mass is pinned and only its
orientation (a rotation about that COM) may change.  Each glycosidic bridging
oxygen is a FREE particle (3 translational DOF).  Covalency is corrected by
HARMONIC SPRINGS — the energy is bonded-only:

    E = Σ_linkage  k_bond  · (|Cd − O| − co_bond)²      (donor  C-O bond)
                 + k_bond  · (|O − Ca| − co_bond)²      (acceptor C-O bond)
                 + k_angle · (∠Cd-O-Ca − coc_angle)²    (C-O-C angle)
                 + k_phi   · (φ − φ_ref(α/β))²          (glycosidic dihedral)
                 [+ k_psi  · (ψ − ψ_ref)²]              (optional)

where, for each glycosidic linkage,
    Cd, Ca      donor / acceptor linkage carbons (ring atoms of two beads),
    O           the free bridging oxygen (seeded near the ideal position),
    φ           dihedral(ring_O_d, Cd, O, Ca) — the α/β-driven rotation,
    ψ           dihedral(Cd, O, Ca, neighbour_a)  (optional).

There are no van der Waals / electrostatic / gravity / slab terms.  The solver
rotates the rigid beads (about their pinned COMs) and translates the free
oxygens down these springs.  Because the sugars are rigid the pyranose rings
cannot deform.

The driver operates on ``chain_dict`` (mol_name → mol_data with
'absolute_coordinates', 'COM', 'carbon_map', 'quaternion'), mutates the bead
coordinates/quaternions in place, and returns (chain_dict, report).

Everything here is numpy/scipy only (no RDKit), so the core is unit-testable
without the full chemistry stack.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..geometry.geometry_utils import rotate_molecule_around_com
from .config import RigidRelaxConfig


# ============================================================================
# Geometry primitives (pure numpy)
# ============================================================================

def _dihedral(p0, p1, p2, p3):
    """Signed dihedral angle p0-p1-p2-p3 in degrees, range (-180, 180]."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    # project b0 and b2 onto plane perpendicular to b1
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def _angle_deg(p1, vertex, p2):
    """Angle p1-vertex-p2 in degrees."""
    v1 = np.asarray(p1, dtype=float) - np.asarray(vertex, dtype=float)
    v2 = np.asarray(p2, dtype=float) - np.asarray(vertex, dtype=float)
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _ring_normal(ring_coords):
    """Unit normal to a set of (near-planar) ring atom coordinates via SVD."""
    ring_coords = np.asarray(ring_coords, dtype=float)
    centroid = ring_coords.mean(axis=0)
    _, _, vh = np.linalg.svd(ring_coords - centroid)
    n = vh[-1]
    return n / (np.linalg.norm(n) + 1e-12)


def _place_bridging_oxygen(c1, c2, ring_normal, ring_centroid,
                           anomeric, co_bond, coc_angle):
    """
    Deterministic bridging-oxygen position for a glycosidic C1-O-C2 linkage.

    Ports the placement rule from structure/bonds_creation.create_glycosidic_bond:
    O sits at ``co_bond`` from each carbon (when reachable), displaced off the
    C1→C2 axis along the donor ring-normal perpendicular; the side is set by the
    anomeric label (β above the ring plane, α below), with a post-check flip.
    """
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)

    c1_to_c2 = c2 - c1
    d = np.linalg.norm(c1_to_c2)
    if d < 1e-6:
        return (c1 + c2) / 2.0
    axis = c1_to_c2 / d

    # isosceles apex: |O-c1| = |O-c2| = co_bond  (feasible when d <= 2*co_bond)
    cos_at_c1 = np.clip(d / (2.0 * co_bond), -1.0, 1.0)
    projection = co_bond * cos_at_c1
    height = co_bond * np.sqrt(max(0.0, 1.0 - cos_at_c1 ** 2))

    # perpendicular to the bond axis, taken from the donor ring normal
    proj = np.dot(ring_normal, axis)
    perp = ring_normal - proj * axis
    if np.linalg.norm(perp) > 0.1:
        perp = perp / np.linalg.norm(perp)
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(ring_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(ring_normal, ref)
        perp = perp / (np.linalg.norm(perp) + 1e-12)

    if str(anomeric).lower() == 'alpha':
        perp = -perp

    o_pos = c1 + projection * axis + height * perp

    # side post-check: β wants O above the donor ring plane, α below
    if ring_centroid is not None:
        side = np.dot(o_pos - ring_centroid, ring_normal)
        want_above = (str(anomeric).lower() != 'alpha')
        wrong = (want_above and side < 0) or (not want_above and side > 0)
        if wrong:
            perp = -perp
            o_pos = c1 + projection * axis + height * perp

    return o_pos


def ideal_cc_distance(co_bond, coc_angle):
    """Ideal donor-C···acceptor-C separation for a C-O-C bridge (Å)."""
    a = np.radians(coc_angle)
    return float(np.sqrt(2.0 * co_bond ** 2 * (1.0 - np.cos(a))))


def _rotate(coords, com, rotvec):
    """Rotate coords about *com* by the rotation vector *rotvec* (radians)."""
    mat = R.from_rotvec(rotvec).as_matrix()
    return (mat @ (coords - com).T).T + com


def _ang_diff_deg(a, b):
    """Signed smallest difference a-b wrapped to (-180, 180], in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


# ============================================================================
# SDF snapshot (pure Python — keeps this module RDKit-free)
# ============================================================================

def _write_sdf_snapshot(chain_dict, path, title="rigid_relax",
                        mol_bonds=None, linkages=None, free_pos=None):
    """
    Write the bead atoms in *chain_dict* to a V2000 SDF file, with bonds.

    chain_dict carries coordinates and element symbols but no connectivity --
    the bonds live in the RDKit conformer templates, which this module
    deliberately does not import.  So the caller passes them in:

    mol_bonds : dict | None
        mol_name -> [(local_i, local_j, order), ...] from the monomer's
        template.  Without it the file has no bond block and viewers show a
        bare point cloud.
    linkages, free_pos : optional
        When both are given, each glycosidic bridging oxygen is written as a
        real atom bonded to its donor and acceptor carbons -- which is the
        whole point of the snapshot, since it shows directly whether the
        relaxation closed the linkage.

    Returns (n_atoms, n_bonds).
    """
    atoms = []      # (element, x, y, z)
    bonds = []      # (global_i, global_j, order)  -- 0-based here
    offsets = {}

    for name in sorted(chain_dict.keys()):
        md = chain_dict[name]
        coords = md.get('absolute_coordinates')
        types = md.get('atom_types')
        if coords is None or types is None:
            continue
        offsets[name] = len(atoms)
        for sym, xyz in zip(types, coords):
            atoms.append((str(sym), float(xyz[0]), float(xyz[1]), float(xyz[2])))

        for i, j, order in (mol_bonds or {}).get(name, []):
            # Guard against a template whose atom count drifted from the
            # coordinate array: a stale index would corrupt the bond block.
            if i < len(coords) and j < len(coords):
                bonds.append((offsets[name] + i, offsets[name] + j, order))

    # Bridging oxygens: real atoms, bonded to the linkage carbons.
    if linkages and free_pos:
        for link in linkages:
            o_xyz = free_pos.get(link['name'])
            donor, acceptor = link['donor'], link['acceptor']
            if o_xyz is None or donor not in offsets or acceptor not in offsets:
                continue
            o_idx = len(atoms)
            atoms.append(('O', float(o_xyz[0]), float(o_xyz[1]), float(o_xyz[2])))
            bonds.append((offsets[donor] + link['cd_idx'], o_idx, 1))
            bonds.append((o_idx, offsets[acceptor] + link['ca_idx'], 1))

    # V2000 counts and bond blocks use fixed 3-character fields.
    if len(atoms) > 999 or len(bonds) > 999:
        print(f"[rigid_relax] {len(atoms)} atoms / {len(bonds)} bonds exceeds the "
              f"V2000 999 limit; snapshot '{path}' will be malformed")

    lines = [title, "  rigid_relax", ""]
    lines.append(f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000")
    for sym, x, y, z in atoms:
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
    for i, j, order in bonds:
        lines.append(f"{i + 1:3d}{j + 1:3d}{int(order):3d}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(atoms), len(bonds)


def _seed_free_oxygens(linkages, coords_by_name, cfg):
    """Deterministic bridging-oxygen position for each linkage, from the current
    bead coordinates.  Used both to seed the relaxation and to draw the linkage
    in the 'before' snapshot."""
    free = {}
    for link in linkages:
        donor, acceptor = link['donor'], link['acceptor']
        if donor not in coords_by_name or acceptor not in coords_by_name:
            continue
        cd = coords_by_name[donor][link['cd_idx']]
        ca = coords_by_name[acceptor][link['ca_idx']]
        ring = coords_by_name[donor][link['donor_ring_idx']]
        free[link['name']] = _place_bridging_oxygen(
            cd, ca, _ring_normal(ring), ring.mean(axis=0),
            link['anomeric'], cfg.co_bond, cfg.coc_angle)
    return free


# ============================================================================
# Linkage extraction
# ============================================================================

def _ring_indices(carbon_map):
    """All ring-atom indices for a monomer (ring carbons + ring oxygen)."""
    idx = list(carbon_map.get('all_ring_carbons', []))
    ring_o = carbon_map.get('ring_oxygen')
    if ring_o is not None:
        idx = idx + [ring_o]
    return idx


def _donor_ref_index(carbon_map, donor_carbon):
    """Reference atom on the donor for φ: the ring oxygen (O5) if present,
    else an in-ring neighbour of the donor carbon."""
    ring_o = carbon_map.get('ring_oxygen')
    if ring_o is not None:
        return ring_o
    return _neighbor_index(carbon_map, donor_carbon)


def _neighbor_index(carbon_map, carbon_name):
    """An in-ring neighbour atom index of *carbon_name* (for ψ / fallback φ)."""
    if carbon_name and carbon_name.startswith('C') and carbon_name[1:].isdigit():
        n = int(carbon_name[1:])
        for cand in (f'C{n-1}', f'C{n+1}'):
            if cand in carbon_map and isinstance(carbon_map[cand], int):
                return carbon_map[cand]
    ring = _ring_indices(carbon_map)
    self_idx = carbon_map.get(carbon_name)
    for i in ring:
        if i != self_idx:
            return i
    return self_idx


def build_linkages(chain_dict, glycosidic_bonds, cfg):
    """
    Turn the input ``glycosidic_bonds`` into resolved linkage records.

    Each ``glycosidic_bonds`` entry is expected in the full form
    ``[donor, donor_C, acceptor, acceptor_C, anomeric, name]``.  Entries whose
    donor/acceptor are not beads in chain_dict, or whose carbons are unknown,
    are skipped (with the reason recorded).

    Returns (linkages, skipped) where each linkage is a dict of atom indices,
    the anomeric label, and reference angles.
    """
    linkages = []
    skipped = []

    for entry in glycosidic_bonds:
        entry = list(entry)
        if len(entry) < 4:
            skipped.append((entry, 'too few fields'))
            continue
        donor, donor_c, acceptor, acceptor_c = entry[0], entry[1], entry[2], entry[3]
        anomeric = entry[4] if len(entry) > 4 else 'beta'
        name = entry[5] if len(entry) > 5 else f'{donor}-{acceptor}'

        if donor not in chain_dict or acceptor not in chain_dict:
            skipped.append((name, 'donor/acceptor not a bead'))
            continue
        if not isinstance(acceptor_c, str):
            skipped.append((name, 'acceptor carbon is a coordinate, not a bead atom'))
            continue

        cmd = chain_dict[donor]['carbon_map']
        cma = chain_dict[acceptor]['carbon_map']
        if donor_c not in cmd or acceptor_c not in cma:
            skipped.append((name, 'carbon name not in carbon_map'))
            continue

        linkages.append({
            'name': name,
            'anomeric': anomeric,
            'donor': donor,
            'acceptor': acceptor,
            'cd_idx': cmd[donor_c],
            'ca_idx': cma[acceptor_c],
            'donor_ref_idx': _donor_ref_index(cmd, donor_c),
            'donor_ring_idx': _ring_indices(cmd),
            'acc_neighbor_idx': _neighbor_index(cma, acceptor_c),
            'phi_ref': cfg.phi_ref_for(anomeric),
            'psi_ref': cfg.psi_ref,
        })

    return linkages, skipped


# ============================================================================
# Energy
# ============================================================================

def linkage_terms(link, state, free_pos, cfg):
    """
    Bonded-only spring energy of a single glycosidic linkage.

    The bridging oxygen is a FREE particle at ``free_pos[link['name']]``.
    Covalency is corrected by harmonic springs:

        E =  k_bond · (|Cd − O| − co_bond)²           (donor C-O bond)
          +  k_bond · (|O − Ca| − co_bond)²           (acceptor C-O bond)
          +  k_angle · (∠Cd-O-Ca − coc_angle)²        (C-O-C angle)
          +  k_phi  · (φ − φ_ref)²                     (glycosidic dihedral)
          [+ k_psi  · (ψ − ψ_ref)²]                    (optional)

    Angle/dihedral deviations are converted to radians before squaring.

    Returns a dict with the energy and the diagnostics reported per linkage.
    """
    dcoords = state[link['donor']]
    acoords = state[link['acceptor']]

    cd = dcoords[link['cd_idx']]
    ca = acoords[link['ca_idx']]
    o5 = dcoords[link['donor_ref_idx']]
    o = free_pos[link['name']]

    r1 = float(np.linalg.norm(cd - o))          # donor C-O bond
    r2 = float(np.linalg.norm(o - ca))          # acceptor C-O bond
    ang = _angle_deg(cd, o, ca)                 # C-O-C angle
    phi = _dihedral(o5, cd, o, ca)              # glycosidic dihedral
    dist = float(np.linalg.norm(cd - ca))       # Cd···Ca (diagnostic only)

    e = cfg.k_bond * (r1 - cfg.co_bond) ** 2
    e += cfg.k_bond * (r2 - cfg.co_bond) ** 2
    e += cfg.k_angle * np.radians(ang - cfg.coc_angle) ** 2
    e += cfg.k_phi * np.radians(_ang_diff_deg(phi, link['phi_ref'])) ** 2

    if cfg.k_psi > 0.0:
        neigh = acoords[link['acc_neighbor_idx']]
        psi = _dihedral(cd, o, ca, neigh)
        e += cfg.k_psi * np.radians(_ang_diff_deg(psi, link['psi_ref'])) ** 2

    return {'e': e, 'phi': phi, 'r1': r1, 'r2': r2, 'angle': ang, 'dist': dist}


def total_energy(linkages, state, free_pos, cfg):
    """Sum of all linkage spring energies for the current state."""
    return sum(linkage_terms(link, state, free_pos, cfg)['e'] for link in linkages)


# ============================================================================
# Driver
# ============================================================================

def relax_chain(chain_dict, glycosidic_bonds, config=None, phosphate_bonds=None,
                output_name=None, mol_bonds=None):
    """
    Bonded-only spring relaxation of a glycan chain: rigid sugar beads (COM
    pinned, rotation only) connected by FREE bridging oxygens, corrected by
    harmonic bond + angle + dihedral springs (see ``linkage_terms``).

    Parameters
    ----------
    chain_dict : dict
        mol_name -> mol_data (mutated in place: 'absolute_coordinates',
        'quaternion', 'relative_coordinates' updated for beads that moved).
    glycosidic_bonds : list
        The input ``config['glycosidic_bonds']`` entries
        (``[donor, donor_C, acceptor, acceptor_C, anomeric, name]``).
    config : RigidRelaxConfig | None
        Defaults from src/constants.py when None.
    phosphate_bonds : optional
        Reserved for phosphodiester linkages (not yet relaxed here); accepted so
        callers can pass it without error.
    output_name : str | None
        When given, write SDF snapshots of the structure before and after the
        relaxation to ``<output_name>_rigid_relax_before.sdf`` and
        ``<output_name>_rigid_relax_after.sdf``.
    mol_bonds : dict | None
        mol_name -> [(local_i, local_j, order), ...], the monomer's own bonds
        from its RDKit template.  chain_dict has no connectivity of its own, so
        without this the snapshots are point clouds with no bonds.  The bridging
        oxygens are drawn regardless, so the linkages are always visible.

    Returns
    -------
    (chain_dict, report) where report is a dict with per-linkage before/after
    φ, bond lengths, angle and Cd···Ca distance, convergence info, any
    skipped/flagged linkages, and the snapshot paths written.

    Notes
    -----
    The free oxygen positions are internal to this pass (used to compute the
    spring forces that drive the bead rotations); the exporter re-creates the
    bridging oxygen deterministically from the corrected linkage carbons.
    """
    cfg = config or RigidRelaxConfig()

    linkages, skipped = build_linkages(chain_dict, glycosidic_bonds, cfg)

    report = {
        'ran': False,
        'n_linkages': len(linkages),
        'skipped': skipped,
        'sweeps': 0,
        'energy_initial': None,
        'energy_final': None,
        'linkage_before': {},
        'linkage_after': {},
        'flagged': [],
        'snapshots': {},
    }

    if not cfg.enable:
        if cfg.verbose:
            print("\n[rigid_relax] disabled (RIGID_RELAX_ENABLE=0) — skipping")
        return chain_dict, report
    if not linkages:
        if cfg.verbose:
            print("\n[rigid_relax] no resolvable glycosidic linkages — nothing to do")
        return chain_dict, report

    # "Before" snapshot (only on the running path). The bridging oxygens do not
    # exist yet, so seed them from the current coordinates to draw the linkages.
    if output_name is not None:
        before_path = f"{output_name}_rigid_relax_before.sdf"
        initial_coords = {
            name: np.array(chain_dict[name]['absolute_coordinates'], dtype=float)
            for name in chain_dict if 'absolute_coordinates' in chain_dict[name]
        }
        n, nb = _write_sdf_snapshot(
            chain_dict, before_path, title=f"{output_name} before",
            mol_bonds=mol_bonds, linkages=linkages,
            free_pos=_seed_free_oxygens(linkages, initial_coords, cfg))
        report['snapshots']['before'] = before_path
        if cfg.verbose:
            print(f"\n[rigid_relax] saved before snapshot: {before_path} "
                  f"({n} atoms, {nb} bonds)")
            if not mol_bonds:
                print("[rigid_relax]   ! no mol_bonds passed — only the glycosidic "
                      "bonds are drawn; pass mol_bonds for full connectivity")

    # Bead working state: current coordinates, pinned COMs, cumulative rotation.
    state, com, rtot, moving = {}, {}, {}, set()
    for link in linkages:
        for name in (link['donor'], link['acceptor']):
            if name in state:
                continue
            coords = np.array(chain_dict[name]['absolute_coordinates'], dtype=float)
            state[name] = coords
            c = chain_dict[name].get('COM')
            com[name] = np.array(c, dtype=float) if c is not None else coords.mean(axis=0)
            rtot[name] = np.eye(3)
            moving.add(name)
    moving = sorted(moving)

    # Free bridging-oxygen particles, one per linkage, seeded at the ideal
    # deterministic position (a good starting guess for the springs to refine).
    free_pos = _seed_free_oxygens(linkages, state, cfg)

    if cfg.verbose:
        print("\n" + "=" * 60)
        print("RIGID-BODY RELAXATION (bonded-only springs)")
        print("=" * 60)
        print(f"  Beads (rigid, COM pinned):     {len(moving)}")
        print(f"  Free bridging oxygens:         {len(free_pos)}")
        print(f"  Glycosidic linkages:           {len(linkages)}")
        if skipped:
            print(f"  Skipped linkages:              {len(skipped)}")
        print(f"  Springs: C-O={cfg.co_bond:.2f} Å (k={cfg.k_bond}), "
              f"C-O-C={cfg.coc_angle:.0f}° (k={cfg.k_angle}), "
              f"φ (k={cfg.k_phi}), ψ (k={cfg.k_psi})")

    def _record(dst):
        for link in linkages:
            t = linkage_terms(link, state, free_pos, cfg)
            dst[link['name']] = {
                'phi': t['phi'], 'phi_ref': link['phi_ref'],
                'phi_dev': _ang_diff_deg(t['phi'], link['phi_ref']),
                'r1': t['r1'], 'r2': t['r2'], 'angle': t['angle'], 'dist': t['dist'],
            }

    _record(report['linkage_before'])

    E = total_energy(linkages, state, free_pos, cfg)
    report['energy_initial'] = E
    scale = 1.0
    d_rot = cfg.fd_delta
    d_tr = cfg.fd_delta_trans
    axes = np.eye(3)

    sweep = 0
    for sweep in range(cfg.max_sweeps):
        gmax = 0.0

        # Bead rotation gradients (numerical torque about the pinned COM).
        bead_grad = {}
        for name in moving:
            g = np.zeros(3)
            base = state[name]
            for k in range(3):
                state[name] = _rotate(base, com[name], axes[k] * d_rot)
                e_plus = total_energy(linkages, state, free_pos, cfg)
                state[name] = _rotate(base, com[name], -axes[k] * d_rot)
                e_minus = total_energy(linkages, state, free_pos, cfg)
                g[k] = (e_plus - e_minus) / (2.0 * d_rot)
            state[name] = base
            bead_grad[name] = g
            gmax = max(gmax, np.linalg.norm(g))

        # Free-oxygen translation gradients.
        free_grad = {}
        for oid, base in free_pos.items():
            g = np.zeros(3)
            for k in range(3):
                free_pos[oid] = base + axes[k] * d_tr
                e_plus = total_energy(linkages, state, free_pos, cfg)
                free_pos[oid] = base - axes[k] * d_tr
                e_minus = total_energy(linkages, state, free_pos, cfg)
                g[k] = (e_plus - e_minus) / (2.0 * d_tr)
            free_pos[oid] = base
            free_grad[oid] = g
            gmax = max(gmax, np.linalg.norm(g))

        if gmax < cfg.grad_tol:
            if cfg.verbose:
                print(f"  Converged (|grad|={gmax:.2e}) at sweep {sweep}")
            break

        # Trial step: rotate beads and translate free oxygens down the gradient.
        trial, trial_rot = {}, {}
        for name in moving:
            v = -(cfg.step * scale) * bead_grad[name]
            n = np.linalg.norm(v)
            if n > cfg.max_rot:
                v = v * (cfg.max_rot / n)
            trial[name] = _rotate(state[name], com[name], v)
            trial_rot[name] = R.from_rotvec(v).as_matrix()

        trial_free = {}
        for oid in free_pos:
            d = -(cfg.step_trans * scale) * free_grad[oid]
            n = np.linalg.norm(d)
            if n > cfg.max_trans:
                d = d * (cfg.max_trans / n)
            trial_free[oid] = free_pos[oid] + d

        E_new = total_energy(linkages, trial, trial_free, cfg)

        if E_new < E:
            for name in moving:
                state[name] = trial[name]
                rtot[name] = trial_rot[name] @ rtot[name]
            free_pos = trial_free
            improved = E - E_new
            E = E_new
            scale = min(scale * 1.1, 4.0)
            if improved < cfg.energy_tol:
                if cfg.verbose:
                    print(f"  Converged (ΔE={improved:.2e}) at sweep {sweep}")
                break
        else:
            scale *= 0.5
            if scale < 1e-6:
                if cfg.verbose:
                    print(f"  Step underflow at sweep {sweep} — stopping")
                break

    report['ran'] = True
    report['sweeps'] = sweep + 1
    report['energy_final'] = E

    # Write corrected orientations back into chain_dict (updates quaternion +
    # relative_coordinates via the shared rigid-rotation helper; COM unchanged).
    for name in moving:
        md = chain_dict[name]
        if 'quaternion' not in md:
            md['quaternion'] = [0.0, 0.0, 0.0, 1.0]
        updated = rotate_molecule_around_com(md, rtot[name])
        md['absolute_coordinates'] = updated['absolute_coordinates']
        md['relative_coordinates'] = updated['relative_coordinates']
        md['quaternion'] = updated['quaternion']

    # "After" snapshot (corrected orientations written back above). free_pos holds
    # the relaxed bridging oxygens, so the linkages show their final geometry.
    if output_name is not None:
        after_path = f"{output_name}_rigid_relax_after.sdf"
        n, nb = _write_sdf_snapshot(
            chain_dict, after_path, title=f"{output_name} after",
            mol_bonds=mol_bonds, linkages=linkages, free_pos=free_pos)
        report['snapshots']['after'] = after_path
        if cfg.verbose:
            print(f"[rigid_relax] saved after snapshot:  {after_path} "
                  f"({n} atoms, {nb} bonds)")

    # Record "after" + flag linkages whose C-O bonds stay stretched (the beads'
    # COMs are too far apart to close the bond by rotation alone).
    _record(report['linkage_after'])
    for link in linkages:
        a = report['linkage_after'][link['name']]
        bond_dev = max(abs(a['r1'] - cfg.co_bond), abs(a['r2'] - cfg.co_bond))
        if bond_dev > 0.3:
            report['flagged'].append((link['name'], a['r1'], a['r2']))

    if cfg.verbose:
        print(f"  Sweeps: {report['sweeps']}   "
              f"E: {report['energy_initial']:.4f} → {report['energy_final']:.4f}")
        for link in linkages:
            b = report['linkage_before'][link['name']]
            a = report['linkage_after'][link['name']]
            print(f"  {link['name']:>22} [{link['anomeric']}]: "
                  f"φ {b['phi']:+7.1f}°→{a['phi']:+7.1f}° "
                  f"(ref {a['phi_ref']:+.0f}°) | "
                  f"C-O {a['r1']:.2f}/{a['r2']:.2f} Å | "
                  f"C-O-C {a['angle']:.0f}°")
        for nm, r1, r2 in report['flagged']:
            print(f"  ⚠ {nm}: C-O bonds {r1:.2f}/{r2:.2f} Å (rest {cfg.co_bond:.2f}) "
                  f"— COMs too far to close by rotation alone; left for the MD")
        print("=" * 60)

    return chain_dict, report
