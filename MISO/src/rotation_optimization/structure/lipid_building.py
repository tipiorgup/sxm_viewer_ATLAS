import numpy as np
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from ..geometry.geometry_utils import get_perpendicular_vector, normalize_vector
from ...IK.fabrikSolver import FabrikSolver3D

#To-do: Add lipid anchor_position into ring (get_anchor_postion)


def build_zigzag_chain(start_pos, direction, n_carbons, C_C_BOND=1.54):
    """
    Build a carbon chain with zigzag (trans) conformation

    Parameters:
    - start_pos: starting position (numpy array)
    - direction: main direction vector (normalized)
    - n_carbons: number of carbons in chain
    - C_C_BOND: C-C bond length

    Returns:
    - List of carbon positions
    """

    TETRAHEDRAL_HALF_ANGLE = np.radians(109.47 / 2)

    if abs(direction[2]) < 0.9:
        perp = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp = np.cross(direction, np.array([1, 0, 0]))
    perp = perp / np.linalg.norm(perp)

    chain_carbons = []
    current_pos = start_pos

    for i in range(n_carbons):
        zigzag_sign = (-1) ** i

        forward = direction * np.cos(TETRAHEDRAL_HALF_ANGLE) * C_C_BOND
        sideways = perp * zigzag_sign * np.sin(TETRAHEDRAL_HALF_ANGLE) * C_C_BOND

        step = forward + sideways
        current_pos = current_pos + step
        chain_carbons.append(current_pos.copy())

    return chain_carbons

def add_ch2_hydrogens(c_pos, prev_pos, next_pos, bond_length):
    """Add 2 hydrogens to a CH2 group using proper tetrahedral geometry."""
    back_vec = (prev_pos - c_pos) / np.linalg.norm(prev_pos - c_pos)

    if next_pos is not None:
        forward_vec = (next_pos - c_pos) / np.linalg.norm(next_pos - c_pos)
        bisector = back_vec - forward_vec
        if np.linalg.norm(bisector) > 0.001:
            bisector = bisector / np.linalg.norm(bisector)
        else:
            bisector = get_perpendicular_vector(back_vec)
    else:
        bisector = get_perpendicular_vector(back_vec)

    h1_pos = c_pos + bisector * bond_length
    h2_pos = c_pos - bisector * bond_length

    return [h1_pos, h2_pos]

def add_methyl_hydrogens(c_pos, prev_pos, bond_length):
    """Add 3 hydrogens to a terminal CH3 group."""
    back_dir = (prev_pos - c_pos) / np.linalg.norm(prev_pos - c_pos)

    perp1 = get_perpendicular_vector(back_dir)
    perp2 = np.cross(back_dir, perp1)

    h1_pos = c_pos - back_dir * bond_length
    h2_pos = c_pos + (perp1 * np.cos(0) + perp2 * np.sin(0)) * bond_length * 0.942
    h3_pos = c_pos + (perp1 * np.cos(2*np.pi/3) + perp2 * np.sin(2*np.pi/3)) * bond_length * 0.942

    return [h1_pos, h2_pos, h3_pos]

def get_anchor_position(connection_anchor, mol_data, carbon_name, connection_blob,
                       target_position=None, offset_from_ring=3.5):
    """
    Now accepts target_position to intelligently offset towards it.
    """
    if connection_anchor == "sugar":
        if carbon_name is None:
            raise ValueError("Sugar attachment requires a carbon name")

        carbon_idx = mol_data['carbon_map'][carbon_name]
        coords = np.array(mol_data['absolute_coordinates'])
        carbon_pos = coords[carbon_idx]
        ring_center = np.array(mol_data['COM'])

        if target_position is not None:
            target_pos = np.array(target_position)
            towards_target = target_pos - carbon_pos
            towards_target = towards_target / np.linalg.norm(towards_target)

            anchor_pos = carbon_pos + towards_target * offset_from_ring

            print(f"  Carbon position: {carbon_pos}")
            print(f"  Ring center (COM): {ring_center}")
            print(f"  Target position: {target_pos}")
            print(f"  Offset anchor (towards target): {anchor_pos}")
        else:
            outward_direction = carbon_pos - ring_center
            outward_direction = outward_direction / np.linalg.norm(outward_direction)
            anchor_pos = carbon_pos + outward_direction * offset_from_ring
            print(f"  Offset anchor (radial): {anchor_pos}")

        return anchor_pos, carbon_idx, ring_center

    elif connection_anchor == "lipid":
        if 'lipid_chains' not in mol_data or len(mol_data['lipid_chains']) == 0:
            raise ValueError("No existing lipid chains found to anchor to")

        if connection_blob is None:
            first_lipid = mol_data['lipid_chains'][0]
            if len(first_lipid['chain_carbons']) < 2:
                raise ValueError("Lipid chain has fewer than 2 carbons")

            c2_pos = np.array(first_lipid['chain_carbons'][1])
            anchor_pos = c2_pos

            print(f"Branching from C2: {anchor_pos}")
            return anchor_pos, 1, None

        else:
            connection_blob = np.array(connection_blob)
            min_distance = float('inf')
            nearest_carbon_pos = None

            for lipid in mol_data['lipid_chains']:
                for carbon_pos in lipid['chain_carbons']:
                    carbon_pos = np.array(carbon_pos)
                    distance = np.linalg.norm(carbon_pos - connection_blob)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_carbon_pos = carbon_pos

            if nearest_carbon_pos is None:
                raise ValueError("Could not find nearest carbon")

            anchor_pos = nearest_carbon_pos

            print(f"Found nearest carbon at distance: {min_distance:.3f} A")
            print(f"Using anchor: {anchor_pos}")
            return anchor_pos, None, None

def build_linkage_headgroup(start_pos, direction, linkage_type,
                           is_branched=False,
                           C_N_BOND=1.47, C_C_BOND=1.54, C_O_BOND=1.43, C_O_DOUBLE=1.23):
    """
    Build the linkage head group (ester, amide, ether, or branched).

    For branched lipids (lipid-anchored), always creates a carbonyl regardless of linkage_type.
    """
    headgroup_data = {}
    current_pos = start_pos.copy()

    if is_branched:
        print(f"    Creating BRANCHED lipid with {linkage_type} linkage")

        if linkage_type == "amide":
            n_link = current_pos + direction * C_N_BOND
            c_carbonyl = n_link + direction * C_C_BOND
            o_carbonyl = c_carbonyl + np.array([0, 0, 1]) * C_O_DOUBLE

            headgroup_data.update({
                "n_link_position": n_link.tolist(),
                "c_carbonyl_position": c_carbonyl.tolist(),
                "o_carbonyl_position": o_carbonyl.tolist(),
                "branched": True
            })
            current_pos = c_carbonyl
        else:
            c_carbonyl = current_pos + direction * C_C_BOND
            o_carbonyl = c_carbonyl + np.array([0, 0, 1]) * C_O_DOUBLE

            headgroup_data.update({
                "c_carbonyl_position": c_carbonyl.tolist(),
                "o_carbonyl_position": o_carbonyl.tolist(),
                "branched": True
            })
            current_pos = c_carbonyl

        return headgroup_data, current_pos

    if linkage_type == "ester":
        o_ester = current_pos + direction * C_O_BOND
        c_carbonyl = o_ester + direction * C_O_BOND
        o_carbonyl = c_carbonyl + np.array([0, 0, 1]) * C_O_DOUBLE
        headgroup_data.update({
            "o_ester_position": o_ester.tolist(),
            "c_carbonyl_position": c_carbonyl.tolist(),
            "o_carbonyl_position": o_carbonyl.tolist(),
        })
        current_pos = c_carbonyl

    elif linkage_type == "amide":
        n_link = current_pos + direction * C_N_BOND
        c_carbonyl = n_link + direction * C_C_BOND
        o_carbonyl = c_carbonyl + np.array([0, 0, 1]) * C_O_DOUBLE
        headgroup_data.update({
            "n_link_position": n_link.tolist(),
            "c_carbonyl_position": c_carbonyl.tolist(),
            "o_carbonyl_position": o_carbonyl.tolist(),
        })
        current_pos = c_carbonyl

    elif linkage_type == "ether":
        o_ether = current_pos + direction * C_O_BOND
        headgroup_data["o_ether_position"] = o_ether.tolist()
        current_pos = o_ether

    else:
        raise ValueError(f"Unsupported linkage type: {linkage_type}")

    return headgroup_data, current_pos

def build_lipid_backbone(start_pos, target_pos, midpoint=None, C_C_BOND=1.54, use_fabrik=True, n_carbons=None):
    """
    Build the carbon backbone from start to target, optionally through midpoint.

    Parameters:
    - start_pos: starting position (numpy array)
    - target_pos: target position (numpy array)
    - midpoint: optional waypoint (numpy array)
    - C_C_BOND: C-C bond length
    - use_fabrik: whether to apply FABRIK refinement for exact targeting
    - n_carbons: if given, override the distance-based carbon count estimate

    Returns:
    - backbone_data: dict with chain_carbons, n_carbons, distance_from_target
    """
    target_pos = np.array(target_pos)

    if midpoint is not None:
        print("\nBuilding two-segment chain through waypoint...")
        return _build_two_segment_chain(start_pos, target_pos, midpoint, C_C_BOND, use_fabrik, n_carbons)
    else:
        print("\nBuilding single-segment chain toward target...")
        return _build_single_segment_chain(start_pos, target_pos, C_C_BOND, use_fabrik, n_carbons)

def _build_single_segment_chain(start_pos, target_pos, C_C_BOND, use_fabrik=True, n_carbons=None):
    """Build a single-segment chain from start to target."""
    vector_to_target = target_pos - start_pos
    distance = np.linalg.norm(vector_to_target)
    unit_direction = vector_to_target / distance

    if n_carbons is not None:
        n_carbons = max(2, n_carbons)
        print(f"  Segment distance: {distance:.2f} A")
        print(f"  Using fixed carbon count: {n_carbons}")
    else:
        n_carbons = max(2, int(round(distance / C_C_BOND)))
        print(f"  Segment distance: {distance:.2f} A")
        print(f"  Estimated carbons: {n_carbons}")

    chain_carbons = build_zigzag_chain(start_pos, unit_direction, n_carbons, C_C_BOND)

    final_pos = chain_carbons[-1]
    dist_from_target = np.linalg.norm(final_pos - target_pos)

    print(f"  Initial final carbon: {final_pos}")
    print(f"  Initial distance from target: {dist_from_target:.3f} A")

    if use_fabrik and dist_from_target > 0.5:
        print(f"  Applying FABRIK refinement...")
        chain_carbons = refine_chain_with_fabrik(
            chain_carbons,
            target_pos,
            start_pos,
            C_C_BOND=C_C_BOND
        )
        final_pos = chain_carbons[-1]
        dist_from_target = np.linalg.norm(final_pos - target_pos)
        print(f"  Refined distance from target: {dist_from_target:.3f} A")

    backbone_data = {
        "chain_carbons": [c.tolist() for c in chain_carbons],
        "n_carbons": n_carbons,
        "distance_from_target": dist_from_target
    }

    for i, c_pos in enumerate(chain_carbons):
        backbone_data[f"c{i+1}_position"] = c_pos.tolist()

    return backbone_data

def _build_two_segment_chain(start_pos, target_pos, waypoint, C_C_BOND, use_fabrik=True, n_carbons=None):
    """Build a two-segment chain from start through waypoint to target."""
    waypoint = np.array(waypoint)

    seg1_vector = waypoint - start_pos
    seg1_distance = np.linalg.norm(seg1_vector)
    seg1_direction = seg1_vector / seg1_distance

    seg2_vector = target_pos - waypoint
    seg2_distance = np.linalg.norm(seg2_vector)
    seg2_direction = seg2_vector / seg2_distance

    if n_carbons is not None:
        n_carbons = max(4, n_carbons)
        total_distance = seg1_distance + seg2_distance
        ratio = seg1_distance / total_distance
        seg1_carbons = max(2, round(n_carbons * ratio))
        seg2_carbons = max(2, n_carbons - seg1_carbons)
        print(f"  Using fixed carbon count: {n_carbons} (split {seg1_carbons}/{seg2_carbons})")
    else:
        seg1_carbons = max(2, int(round(seg1_distance / C_C_BOND)))
        seg2_carbons = max(2, int(round(seg2_distance / C_C_BOND)))
        print(f"  Estimated carbons: {seg1_carbons + seg2_carbons}")

    total_carbons = seg1_carbons + seg2_carbons

    print(f"  Segment 1: {seg1_carbons} carbons over {seg1_distance:.2f} A")
    print(f"  Segment 2: {seg2_carbons} carbons over {seg2_distance:.2f} A")
    print(f"  Total: {total_carbons} carbons")

    seg1_chain = build_zigzag_chain(start_pos, seg1_direction, seg1_carbons, C_C_BOND)
    chain_carbons = seg1_chain.copy()

    print(f"  Built segment 1, ended at: {seg1_chain[-1]}")

    seg2_chain = build_zigzag_chain(seg1_chain[-1], seg2_direction, seg2_carbons, C_C_BOND)
    chain_carbons.extend(seg2_chain)

    print(f"  Built segment 2, ended at: {seg2_chain[-1]}")

    final_pos = chain_carbons[-1]
    dist_from_target = np.linalg.norm(final_pos - target_pos)

    print(f"  Initial distance from target: {dist_from_target:.3f} A")

    if use_fabrik and dist_from_target > 0.5:
        print(f"  Applying FABRIK refinement to full chain...")
        chain_carbons = refine_chain_with_fabrik(
            chain_carbons,
            target_pos,
            start_pos,
            C_C_BOND=C_C_BOND
        )
        final_pos = chain_carbons[-1]
        dist_from_target = np.linalg.norm(final_pos - target_pos)
        print(f"  Refined distance from target: {dist_from_target:.3f} A")

    backbone_data = {
        "chain_carbons": [c.tolist() for c in chain_carbons],
        "n_carbons": total_carbons,
        "distance_from_target": dist_from_target
    }

    for i, c_pos in enumerate(chain_carbons):
        backbone_data[f"c{i+1}_position"] = c_pos.tolist()

    return backbone_data

def refine_chain_with_fabrik(chain_carbons, target_pos, anchor_pos, C_C_BOND=1.54, margin=0.1, max_iterations=100):
    """
    Refine an existing carbon chain to reach exact target using FABRIK.

    Parameters:
    - chain_carbons: list of numpy arrays with initial carbon positions
    - target_pos: target position to reach
    - anchor_pos: fixed anchor position (base of chain)
    - C_C_BOND: bond length to maintain
    - margin: acceptable distance from target
    - max_iterations: maximum FABRIK iterations

    Returns:
    - refined_carbons: list of refined positions
    """

    solver = FabrikSolver3D(
        baseX=anchor_pos[0],
        baseY=anchor_pos[1],
        baseZ=anchor_pos[2],
        marginOfError=margin
    )

    for i in range(len(chain_carbons)):
        if i == 0:
            direction = chain_carbons[i] - anchor_pos
        else:
            direction = chain_carbons[i] - chain_carbons[i-1]

        length = np.linalg.norm(direction)
        zAngle = np.degrees(np.arctan2(direction[1], direction[0]))
        yAngle = np.degrees(np.arcsin(direction[2] / (length + 1e-10)))

        solver.addSegment(C_C_BOND, zAngle, yAngle)

    if not solver.isReachable(target_pos[0], target_pos[1], target_pos[2]):
        print(f"  Warning: Target not reachable by FABRIK (distance too far)")
        return chain_carbons

    iterations = 0
    while not solver.inMarginOfError(target_pos[0], target_pos[1], target_pos[2]) and iterations < max_iterations:
        solver.iterate(target_pos[0], target_pos[1], target_pos[2])
        iterations += 1

    refined_carbons = []
    for seg in solver.segments:
        refined_carbons.append(seg.point.copy())

    final_distance = np.linalg.norm(refined_carbons[-1] - target_pos)
    print(f"FABRIK refinement: {iterations} iterations, final distance: {final_distance:.3f} A")

    return refined_carbons


# ============================================================================
# Chain on the drawn path (MISO builder)
# ============================================================================

ZIGZAG_ANGLE_DEG = 112.0          # natural C-C-C angle of an all trans chain
ANGLE_MIN, ANGLE_MAX = 100.0, 124.0   # zig zag angles a defined count may use


def _flat(v):
    v = np.array(v, dtype=float)
    v[2] = 0.0
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _zigzag_step(d, n, C_C_BOND):
    """Advance per carbon of an ideal zig zag (bonds C_C_BOND, equal angles)
    whose n-th carbon is at distance d from the start atom; None if impossible.
    Even n end on the zig zag axis (d = n a); odd n end one zig off it
    (d^2 = (n a)^2 + h^2 with h^2 = b^2 - a^2)."""
    b = C_C_BOND
    if n % 2 == 0:
        a = d / n
    else:
        if d <= b:
            return None
        a = np.sqrt((d * d - b * b) / (n * n - 1)) if n > 1 else None
    return a if a is not None and 0 < a < b else None


def _zigzag_angle(a, C_C_BOND):
    return float(np.degrees(2 * np.arcsin(a / C_C_BOND)))


def _segment_count(d, wanted, C_C_BOND):
    """Carbons on one segment.  Auto (wanted None): the count whose zig zag
    angle is closest to ZIGZAG_ANGLE_DEG.  Defined: the count closest to
    *wanted* whose angle lies in [ANGLE_MIN, ANGLE_MAX] (carbons are added or
    removed); auto when none does."""
    a_nat = C_C_BOND * np.sin(np.radians(ZIGZAG_ANGLE_DEG / 2))
    centre = max(2, int(round(d / a_nat)))
    cand = []
    for n in range(max(2, centre - 3), centre + 4):
        a = _zigzag_step(d, n, C_C_BOND)
        if a is not None:
            cand.append((n, _zigzag_angle(a, C_C_BOND)))
    if not cand:
        return centre
    auto = min(cand, key=lambda t: abs(t[1] - ZIGZAG_ANGLE_DEG))[0]
    if wanted is None:
        return auto
    ok = [n for n, ang in cand if ANGLE_MIN <= ang <= ANGLE_MAX]
    if not ok:
        return auto
    return min(ok, key=lambda n: (abs(n - wanted), abs(dict(cand)[n] - ZIGZAG_ANGLE_DEG)))


def build_chain_on_path(start_pos, points, n_requested=None, C_C_BOND=1.54, z_surface=None):
    """
    Carbon chain lying flat along the drawn path.

    start_pos : the atom the chain grows from (carbonyl C, ether O or branch C).
    points    : [tip] or [midpoint, tip]; only x and y are used.
    n_requested : carbons asked for (None: auto).
    z_surface : height the chain lies at (None: the height of start_pos).

    The path is a straight segment from start_pos to each point in turn.  The
    points sit at z_surface, so the first segment descends from the linkage
    to the surface and the rest lies flat.  On each segment the
    carbons form a natural zig zag in the surface plane: bonds C_C_BOND, angle
    close to ZIGZAG_ANGLE_DEG.  The advance per carbon is the segment length
    divided by its carbon count, and a small tilt of the zig zag across the
    segment puts the last carbon exactly on the point.

    Returns the backbone dict used by the exporter (chain_carbons, n_carbons)
    plus n_requested, mid_carbon_index, end_carbon_index, mid_target and
    end_target (the index and point of the carbons REFINE pins).
    """
    start_pos = np.asarray(start_pos, dtype=float)
    z0 = float(start_pos[2]) if z_surface is None else float(z_surface)
    pts = [np.array([p[0], p[1], z0], dtype=float) for p in points]
    lengths, prev = [], start_pos
    for p in pts:
        lengths.append(float(np.linalg.norm(p - prev)))
        prev = p
    total = sum(lengths)
    wanted = ([max(2, int(round(n_requested * d / total))) for d in lengths]
              if n_requested else [None] * len(lengths))
    counts = [_segment_count(d, w, C_C_BOND) for d, w in zip(lengths, wanted)]

    zhat = np.array([0.0, 0.0, 1.0])

    def _segment(prev, p, d, n, side):
        """Ideal zig zag from prev to p (n carbons), on the given side of the
        line, turned in its plane so its last carbon lies on p.  The plane holds
        the segment and the horizontal normal to it, so a descending segment
        keeps the zig zag across the surface."""
        u = (p - prev) / np.linalg.norm(p - prev)
        w = np.cross(zhat, _flat(p - prev)) * side
        a = _zigzag_step(d, n, C_C_BOND) or (d / n)
        h = np.sqrt(max(C_C_BOND ** 2 - a ** 2, 0.0))
        local = [(k * a, h * (k % 2)) for k in range(1, n + 1)]
        ex, ey = local[-1]
        t = -np.arctan2(ey, ex)
        ct, st_ = np.cos(t), np.sin(t)
        out = []
        for x, y in local:
            xr, yr = ct * x - st_ * y, st_ * x + ct * y
            out.append(prev + u * xr + w * yr)
        out[-1] = p.copy()
        return out

    def _angle(a_, m_, b_):
        v1, v2 = a_ - m_, b_ - m_
        return float(np.degrees(np.arccos(np.clip(
            v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))

    chain, ends, prev = [], [], start_pos
    for k_seg, (p, d, n) in enumerate(zip(pts, lengths, counts)):
        if k_seg == 0:
            seg = _segment(prev, p, d, n, +1)
        else:
            # zig zag side of the next segment: the one whose first bond makes
            # the angle at the bend carbon closest to the natural angle
            before = chain[-2] if len(chain) >= 2 else start_pos
            options = [_segment(prev, p, d, n, sd) for sd in (+1, -1)]
            seg = min(options, key=lambda o: abs(_angle(before, prev, o[0]) - ZIGZAG_ANGLE_DEG))
        chain.extend(seg)
        ends.append(len(chain) - 1)
        prev = p

    if len(pts) == 2:
        # interior angle of the drawn path at the midpoint (180 = straight)
        interior = _angle(start_pos, pts[0], pts[1])
        if interior < 70.0:
            print(f"  ! lipid: the drawn path folds back at the midpoint (interior angle "
                  f"{interior:.0f} deg); a chain cannot turn that sharply, so the carbons "
                  f"around the midpoint will be strained")

    used = sum(counts)
    note = ''
    if n_requested and used != int(n_requested):
        note = (f" ({'added' if used > n_requested else 'removed'} "
                f"{abs(used - int(n_requested))} to fit the path)")
    print(f"  Lipid chain: requested {n_requested if n_requested else 'auto'}, used {used}"
          f"{note}; path " + " + ".join(f"{d:.2f}" for d in lengths) + " A, carbons "
          + "/".join(str(n) for n in counts))

    data = {
        "chain_carbons": [c.tolist() for c in chain],
        "n_carbons": used,
        "n_requested": int(n_requested) if n_requested else None,
        "distance_from_target": 0.0,
        "mid_carbon_index": ends[0] if len(pts) == 2 else None,
        "end_carbon_index": ends[-1],
        "mid_target": pts[0].tolist() if len(pts) == 2 else None,
        "end_target": pts[-1].tolist(),
    }
    for i, c_pos in enumerate(chain):
        data[f"c{i+1}_position"] = c_pos.tolist()
    return data


def replaced_atom_position(mol_data, carbon_name, prefer):
    """Position of the sugar's own exocyclic N or O on *carbon_name* (the group
    the lipid linkage replaces), in the sugar's current orientation.  *prefer*
    is 'N' or 'O'; the other element is the fallback.  None if neither."""
    cmap = mol_data['carbon_map']
    coords = np.asarray(mol_data['absolute_coordinates'], dtype=float)
    types = mol_data.get('atom_types') or []
    c = coords[cmap[carbon_name]]
    ring_o = cmap.get('ring_oxygen')
    best = {}
    for i, t in enumerate(types):
        if t not in ('N', 'O') or i == ring_o:
            continue
        d = float(np.linalg.norm(coords[i] - c))
        if 1.2 < d < 1.65 and (t not in best or d < best[t][0]):
            best[t] = (d, coords[i])
    for t in (prefer, 'O' if prefer == 'N' else 'N'):
        if t in best:
            return best[t][1].copy()
    return None


def build_headgroup_at_sugar(carbon_pos, link_pos, first_point, linkage_type,
                             C_N_BOND=1.47, C_O_BOND=1.43, AMIDE_CN=1.33, ESTER_CO=1.34,
                             C_O_DOUBLE=1.23):
    """Linkage atoms of a sugar attached lipid, lying flat.

    The linkage atom (amide N, ester or ether O) sits where the sugar's own N or
    O was (link_pos), so the bond to the sugar and the stereo at that carbon are
    the template's.  The carbonyl carbon follows towards the first path point,
    level with the linkage atom, and its O lies in the surface plane.
    Returns (headgroup_data, position the chain grows from)."""
    carbon_pos = np.asarray(carbon_pos, dtype=float)
    if link_pos is None:
        bond = C_N_BOND if linkage_type == 'amide' else C_O_BOND
        link_pos = carbon_pos + _flat(np.asarray(first_point) - carbon_pos) * bond
    link_pos = np.asarray(link_pos, dtype=float)
    d1 = _flat(np.asarray(first_point) - link_pos)
    w = np.cross(np.array([0.0, 0.0, 1.0]), d1)
    if linkage_type == 'amide':
        c = link_pos + d1 * AMIDE_CN
        return ({"n_link_position": link_pos.tolist(), "c_carbonyl_position": c.tolist(),
                 "o_carbonyl_position": (c + w * C_O_DOUBLE).tolist()}, c)
    if linkage_type == 'ester':
        c = link_pos + d1 * ESTER_CO
        return ({"o_ester_position": link_pos.tolist(), "c_carbonyl_position": c.tolist(),
                 "o_carbonyl_position": (c + w * C_O_DOUBLE).tolist()}, c)
    if linkage_type == 'ether':
        return {"o_ether_position": link_pos.tolist()}, link_pos
    raise ValueError(f"Unsupported linkage type: {linkage_type}")


def create_lipid_chain(connection_anchor, target_n_position, mol_data=None,
                       carbon_name=None, connection_blob=None, midpoint=None,
                       linkage_type="amide", use_fabrik=False, n_carbons=None):
    """
    Build a lipid chain lying flat along the drawn path: from the sugar carbon
    (or, for a branch, the nearest carbon of the parent chain) through the
    midpoint, if given, to the tip (target_n_position).  See
    build_chain_on_path; the chain passes exactly through the midpoint and
    ends exactly on the tip.  n_carbons: carbons asked for (None: auto); it is
    adjusted, with a log line, when it cannot fit the path.  use_fabrik is
    accepted for compatibility and not used.
    """

    C_C_BOND = 1.54
    C_N_BOND = 1.47
    C_O_BOND = 1.43
    C_O_DOUBLE = 1.23

    target_n = np.array(target_n_position)
    waypoint = np.array(midpoint) if midpoint is not None else None

    offset_target = waypoint if waypoint is not None else target_n

    first_point = waypoint if waypoint is not None else target_n
    path = [waypoint, target_n] if waypoint is not None else [target_n]
    is_branched = (connection_anchor == "lipid")

    print(f"\n{'='*60}")
    if not is_branched:
        # Sugar attached: the linkage atom takes the place of the sugar's own N or O.
        if carbon_name is None:
            raise ValueError("Sugar attachment requires a carbon name")
        carbon_idx = mol_data['carbon_map'][carbon_name]
        anchor_pos = np.asarray(mol_data['absolute_coordinates'][carbon_idx], dtype=float)
        ring_center = np.asarray(mol_data['COM'], dtype=float)
        link_pos = replaced_atom_position(mol_data, carbon_name,
                                          'N' if linkage_type == 'amide' else 'O')
        print(f"Creating {linkage_type} lipid chain on {carbon_name}")
        print(f"  Sugar carbon: {anchor_pos}, linkage atom "
              + ("at the sugar's own N/O" if link_pos is not None else "placed towards the path"))
        headgroup_data, current_pos = build_headgroup_at_sugar(
            anchor_pos, link_pos, first_point, linkage_type)
    else:
        anchor_pos, carbon_idx, ring_center = get_anchor_position(
            connection_anchor, mol_data, carbon_name, connection_blob,
            target_position=first_point)
        print(f"Creating BRANCHED lipid chain from {anchor_pos}")
        # Same flat linkage as on a sugar, from the parent chain carbon: ester O
        # (or amide N) towards the path, then C(=O) with its O in the surface
        # plane.  build_linkage_headgroup's branched form had no ester O, which
        # the exporter needs, so branched ester lipids were dropped.
        headgroup_data, current_pos = build_headgroup_at_sugar(
            anchor_pos, None, first_point, linkage_type)
        headgroup_data["branched"] = True

    # Surface height: a sugar attached lipid lies at the height of the sugar's
    # lowest heavy atom (where the sugar touches the surface); a branch stays
    # level with its parent chain.
    z_surface = None
    if not is_branched:
        coords = np.asarray(mol_data['absolute_coordinates'], dtype=float)
        types = mol_data.get('atom_types') or []
        heavy = [i for i, t in enumerate(types) if t != 'H'] or list(range(len(coords)))
        z_surface = float(coords[heavy, 2].min())
        print(f"  Chain lies at z {z_surface:.2f} (sugar contact height), "
              f"linkage at z {float(current_pos[2]):.2f}")
    backbone_data = build_chain_on_path(current_pos, path, n_requested=n_carbons,
                                        C_C_BOND=C_C_BOND, z_surface=z_surface)

    print("=" * 60)

    result = {
        "linkage": linkage_type,
        "sugar_carbon": anchor_pos.tolist(),
        "carbon_index": carbon_idx,
        "target": target_n.tolist(),
        "is_branched": is_branched,
        **headgroup_data,
        **backbone_data
    }

    print(f"\n  DEBUG create_lipid_chain returning:")
    print(f"    linkage: {result['linkage']}")
    print(f"    has n_link_position: {'n_link_position' in result}")
    print(f"    has c_carbonyl_position: {'c_carbonyl_position' in result}")
    if 'n_link_position' in result:
        print(f"    n_link_position: {result['n_link_position']}")

    return result