import numpy as np
from ..geometry.geometry_utils import get_perpendicular_vector
from .bonds_creation import find_hydroxyl_oxygen_at_carbon

def create_phosphoethanolamine_compact(mol_data, carbon_name, target_n_position, 
                                       linkage_name='PEtN'):
    """
    Create compact phosphoethanolamine.
    C-O-P with branches, then O-C1-C2-N in proper sequence.
    """
    
    # Bond lengths
    C_O_BOND = 1.43
    O_P_BOND = 1.60
    P_O_DOUBLE = 1.48
    P_O_SINGLE = 1.57
    O_C_BOND = 1.43
    C_C_BOND = 1.54
    C_N_BOND = 1.47
    
    # Get starting carbon position
    carbon_idx = mol_data['carbon_map'][carbon_name]
    coords = np.array(mol_data['absolute_coordinates'])
    c_sugar = coords[carbon_idx]
    target_n = np.array(target_n_position)
  
    print(f"\n{'='*60}")
    print(f"Creating compact {linkage_name} on {carbon_name}")
    print(f"Sugar carbon: {c_sugar}")
    print(f"Target N: {target_n}")
    
    # Calculate distances
    total_distance = np.linalg.norm(target_n - c_sugar)
    print(f"Distance to target: {total_distance:.3f}Å")
    
    # Main direction: sugar → target
    main_direction = (target_n - c_sugar) / total_distance
    
    # Get perpendicular directions
    if abs(main_direction[2]) < 0.9:
        perp1 = np.cross(main_direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(main_direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    
    perp2 = np.cross(main_direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # ========================================================================
    # Build the main chain: C-O1-P-O4-C1-C2-N
    # ========================================================================
    
    # O1: Slightly off main axis
    bend_angle_1 = np.radians(10)
    o1_dir = main_direction * np.cos(bend_angle_1) + perp1 * np.sin(bend_angle_1)
    o1_pos = c_sugar + o1_dir * C_O_BOND
    
    # P: Continue toward target
    o1_to_target = target_n - o1_pos
    o1_to_target = o1_to_target / np.linalg.norm(o1_to_target)
    p_pos = o1_pos + o1_to_target * O_P_BOND
    
    # O2 (=O) and O3 (O⁻): Perpendicular branches from P
    o2_pos = p_pos + perp1 * P_O_DOUBLE
    o3_pos = p_pos - perp1 * P_O_SINGLE
    
    # O4: Continue along main path
    p_to_target = target_n - p_pos
    p_to_target = p_to_target / np.linalg.norm(p_to_target)
    o4_pos = p_pos + p_to_target * P_O_SINGLE
    
    # C1: Bend slightly
    o4_to_target = target_n - o4_pos
    o4_to_target = o4_to_target / np.linalg.norm(o4_to_target)
    bend_angle_2 = np.radians(15)
    c1_dir = o4_to_target * np.cos(bend_angle_2) + perp2 * np.sin(bend_angle_2)
    c1_pos = o4_pos + c1_dir * O_C_BOND
    
    # C2: Continue toward target with opposite bend
    c1_to_target = target_n - c1_pos
    c1_to_target = c1_to_target / np.linalg.norm(c1_to_target)
    bend_angle_3 = np.radians(15)
    c2_dir = c1_to_target * np.cos(bend_angle_3) - perp2 * np.sin(bend_angle_3)
    c2_pos = c1_pos + c2_dir * C_C_BOND
    
    # N: Final segment to target
    c2_to_target = target_n - c2_pos
    distance_remaining = np.linalg.norm(c2_to_target)
    
    if distance_remaining > 0.01:
        n_dir = c2_to_target / distance_remaining
        # Adjust C-N bond length to hit target exactly if close
        if distance_remaining < C_N_BOND * 1.5:
            n_pos = c2_pos + n_dir * distance_remaining
        else:
            n_pos = c2_pos + n_dir * C_N_BOND
    else:
        n_pos = c2_pos + main_direction * C_N_BOND
    
    # Verification
    achieved_distance = np.linalg.norm(n_pos - target_n)
    
    print(f"\nChain positions:")
    print(f"  O1: {o1_pos}")
    print(f"  P:  {p_pos}")
    print(f"  O2 (=O): {o2_pos}")
    print(f"  O3 (O⁻): {o3_pos}")
    print(f"  O4: {o4_pos}")
    print(f"  C1: {c1_pos}")
    print(f"  C2: {c2_pos}")
    print(f"  N:  {n_pos}")
    print(f"\nTarget accuracy: {achieved_distance:.3f}Å")
    
    
    print(f"\n✓ Compact PEtN created!")
    print("="*60)

    carbon_idx = mol_data['carbon_map'][carbon_name]
    coords = np.array(mol_data['absolute_coordinates'])
    
    oh_to_remove = []
    sugar_oh = find_hydroxyl_oxygen_at_carbon(mol_data, carbon_idx, coords)
    
    if sugar_oh is not None:
        # oh_to_remove.append(sugar_oh)
        oh_to_remove.extend(sugar_oh)
        print(f"\n  Will remove OH at O{sugar_oh} (bonded to {carbon_name})")
    
    return {
        'linkage': linkage_name,
        'sugar_carbon': c_sugar.tolist(),
        'o1_position': o1_pos.tolist(),
        'p_position': p_pos.tolist(),
        'o2_position': o2_pos.tolist(),
        'o3_position': o3_pos.tolist(),
        'o4_position': o4_pos.tolist(),
        'c1_position': c1_pos.tolist(),
        'c2_position': c2_pos.tolist(),
        'n_position': n_pos.tolist(),
        'oh_to_remove': oh_to_remove
    }