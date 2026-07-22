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

def create_lipid_chain(connection_anchor, target_n_position, mol_data=None,
                       carbon_name=None, connection_blob=None, midpoint=None,
                       linkage_type="amide", use_fabrik=False, n_carbons=None):
    """
    Build a lipid chain from a sugar carbon towards a target position.

    Parameters:
    - n_carbons: if given, use exactly this many carbons in the backbone
                 instead of estimating from distance. FABRIK still runs
                 to reconcile geometry.
    """

    C_C_BOND = 1.54
    C_N_BOND = 1.47
    C_O_BOND = 1.43
    C_O_DOUBLE = 1.23

    target_n = np.array(target_n_position)
    waypoint = np.array(midpoint) if midpoint is not None else None

    offset_target = waypoint if waypoint is not None else target_n

    anchor_pos, carbon_idx, ring_center = get_anchor_position(
        connection_anchor, mol_data, carbon_name, connection_blob,
        target_position=offset_target
    )
    is_branched = (connection_anchor == "lipid")

    if waypoint is not None:
        initial_direction = (waypoint - anchor_pos)
    else:
        initial_direction = (target_n - anchor_pos)

    initial_direction = initial_direction / np.linalg.norm(initial_direction)

    print(f"\n{'='*60}")
    print(f"Creating {'BRANCHED' if is_branched else linkage_type} lipid chain")
    print(f"Anchor position (offset): {anchor_pos}")
    if ring_center is not None:
        print(f"Ring center (COM): {ring_center}")
    print(f"Initial direction (towards target): {initial_direction}")
    if n_carbons is not None:
        print(f"Fixed carbon count: {n_carbons}")

    headgroup_data, current_pos = build_linkage_headgroup(
        anchor_pos, initial_direction, linkage_type,
        is_branched=is_branched,
        C_N_BOND=C_N_BOND,
        C_C_BOND=C_C_BOND,
        C_O_BOND=C_O_BOND,
        C_O_DOUBLE=C_O_DOUBLE
    )

    backbone_data = build_lipid_backbone(current_pos, target_n, waypoint, C_C_BOND, use_fabrik, n_carbons)

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