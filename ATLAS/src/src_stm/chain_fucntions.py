import numpy as np
import random
import cv2 as cv
from typing import List, Tuple, Dict, Optional

class NoValidPathException(Exception):
    pass

def validate_2d_bond_path(point1_idx, point2_idx, valid_circles, original_img, std_threshold=30):
    """
    Validate if 2D path between two circles goes through dark zones
    
    Parameters:
    - point1_idx, point2_idx: indices in valid_circles
    - valid_circles: list of [x, y, radius] for each circle
    - original_img: 2D image to check path on
    - std_threshold: standard deviation threshold for dark path detection
    
    Returns:
    - is_valid: bool, True if path is acceptable
    - path_stats: dict with path analysis info
    """
    # Get 2D coordinates and convert to int to avoid uint16 overflow
    x1, y1 = int(valid_circles[point1_idx][0]), int(valid_circles[point1_idx][1])
    x2, y2 = int(valid_circles[point2_idx][0]), int(valid_circles[point2_idx][1])
    
    # Convert to grayscale if needed
    if len(original_img.shape) == 3:
        gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
    else:
        gray_img = original_img
    
    # Get all pixels along the line path
    line_points = []
    
    # Use Bresenham's line algorithm (cv2.line coordinates)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        # Check bounds and sample pixel
        if 0 <= x < gray_img.shape[1] and 0 <= y < gray_img.shape[0]:
            line_points.append(gray_img[y, x])
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    # Analyze path statistics
    if len(line_points) < 2:
        return False, {'reason': 'path too short', 'length': len(line_points)}
    
    line_intensities = np.array(line_points)
    path_mean = np.mean(line_intensities)
    path_std = np.std(line_intensities)
    path_min = np.min(line_intensities)
    path_max = np.max(line_intensities)
    
    # Check for dark zones (high standard deviation indicates valleys)
    is_valid = path_std <= std_threshold
    
    path_stats = {
        'mean': path_mean,
        'std': path_std,
        'min': path_min,
        'max': path_max,
        'length': len(line_points),
        'threshold': std_threshold,
        'reason': 'dark_path' if not is_valid else 'valid'
    }
    
    return is_valid, path_stats

def get_circle_intensity(circle_idx, valid_circles, original_img, region_size=3):
    """Get average intensity in region around circle center"""
    x, y = int(valid_circles[circle_idx][0]), int(valid_circles[circle_idx][1])
    
    if len(original_img.shape) == 3:
        gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
    else:
        gray_img = original_img
    
    # Define region around center
    y_min = max(0, y - region_size//2)
    y_max = min(gray_img.shape[0], y + region_size//2 + 1)
    x_min = max(0, x - region_size//2)
    x_max = min(gray_img.shape[1], x + region_size//2 + 1)
    
    return np.mean(gray_img[y_min:y_max, x_min:x_max])

def find_valid_neighbors_with_intensity(current_point, used_points, points, valid_circles, 
                                       original_img, max_bond_distance, intensity_threshold=20, 
                                       std_threshold=30):
    """
    Find valid neighbors prioritizing similar intensity and valid 2D paths
    
    Parameters:
    - current_point: index of current point
    - used_points: set of already used point indices
    - points: 3D coordinates list
    - valid_circles: circles data for 2D validation
    - original_img: image for path validation
    - max_bond_distance: max 3D distance
    - intensity_threshold: max intensity difference for similar intensity
    - std_threshold: std dev threshold for dark path detection
    """
    distances = calculate_distance_matrix(points)
    current_intensity = get_circle_intensity(current_point, valid_circles, original_img)
    
    valid_neighbors = []
    rejected_bonds = []
    
    for i in range(len(points)):
        if i in used_points:
            continue
            
        # Check 3D distance first
        distance_3d = distances[current_point][i]
        if distance_3d > max_bond_distance:
            rejected_bonds.append({
                'point': i,
                'reason': 'distance_too_far',
                '3d_distance': distance_3d,
                'max_allowed': max_bond_distance
            })
            continue
        
        # Check intensity similarity
        neighbor_intensity = get_circle_intensity(i, valid_circles, original_img)
        intensity_diff = abs(current_intensity - neighbor_intensity)
        
        if intensity_diff > intensity_threshold:
            rejected_bonds.append({
                'point': i,
                'reason': 'intensity_mismatch',
                'current_intensity': current_intensity,
                'neighbor_intensity': neighbor_intensity,
                'difference': intensity_diff,
                'threshold': intensity_threshold
            })
            continue
        
        # Check 2D path validity
        path_valid, path_stats = validate_2d_bond_path(current_point, i, valid_circles, 
                                                      original_img, std_threshold)
        
        if not path_valid:
            rejected_bonds.append({
                'point': i,
                'reason': 'dark_path',
                'path_stats': path_stats
            })
            continue
        
        # Valid neighbor found
        valid_neighbors.append({
            'index': i,
            '3d_distance': distance_3d,
            'intensity_diff': intensity_diff,
            'path_stats': path_stats
        })
    
    return valid_neighbors, rejected_bonds

def try_extend_chain_end_enhanced(chain, end_idx, used_points, points, valid_circles, 
                                 original_img, max_bond_distance, direction, 
                                 intensity_threshold=20, std_threshold=30):
    """Enhanced chain extension with 2D path validation and intensity priority"""
    current_point = chain[end_idx]
    
    valid_neighbors, rejected_bonds = find_valid_neighbors_with_intensity(
        current_point, used_points, points, valid_circles, original_img,
        max_bond_distance, intensity_threshold, std_threshold
    )
    
    if not valid_neighbors:
        return False, rejected_bonds
    
    # Sort by intensity similarity first, then by distance
    valid_neighbors.sort(key=lambda x: (x['intensity_diff'], x['3d_distance']))
    
    # Weighted selection favoring similar intensities
    weights = []
    for neighbor in valid_neighbors:
        # Higher weight for more similar intensities and closer distances
        intensity_weight = 1.0 / (neighbor['intensity_diff'] + 1.0)  # +1 to avoid division by zero
        distance_weight = 1.0 / (neighbor['3d_distance'] ** 2)
        combined_weight = intensity_weight * 2 + distance_weight  # Prioritize intensity
        weights.append(combined_weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        
        # Select using weighted random choice
        selected_neighbor = np.random.choice(valid_neighbors, p=weights)
        selected_idx = selected_neighbor['index']
        
        # Add to chain
        if direction == 'left':
            chain.insert(0, selected_idx)
        else:  # direction == 'right'
            chain.append(selected_idx)
        
        used_points.add(selected_idx)
        return True, []
    
    return False, rejected_bonds

def grow_chain_from_center_enhanced(points, center_idx, valid_circles, original_img, 
                                   max_bond_distance, intensity_threshold=20, 
                                   std_threshold=30):
    """Enhanced chain growth with 2D validation and intensity priority"""
    chain = [center_idx]
    used_points = {center_idx}
    unbonded_points = []
    all_rejected_bonds = []
    
    extend_left = True
    consecutive_failures = 0
    max_consecutive_failures = 20  # Increased for more attempts
    
    while len(chain) < len(points):
        if extend_left:
            success, rejected = try_extend_chain_end_enhanced(
                chain, 0, used_points, points, valid_circles, original_img,
                max_bond_distance, 'left', intensity_threshold, std_threshold
            )
        else:
            success, rejected = try_extend_chain_end_enhanced(
                chain, len(chain)-1, used_points, points, valid_circles, original_img,
                max_bond_distance, 'right', intensity_threshold, std_threshold
            )
        
        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            all_rejected_bonds.extend(rejected)
            
        extend_left = not extend_left
        
        if consecutive_failures >= max_consecutive_failures:
            # Find unbonded points
            for i in range(len(points)):
                if i not in used_points:
                    unbonded_points.append(i)
            break
    
    return chain, unbonded_points, all_rejected_bonds

def generate_chain_networks_enhanced(points, valid_circles, original_img, max_bond_distance, 
                                    num_simulations=1000, num_best=10, failure_threshold=0.8, 
                                    intensity_threshold=20, std_threshold=30, seed=None):
    """
    Enhanced chain generation with 2D path validation and intensity-based bonding
    
    Parameters:
    - points: List of (x, y, z) coordinates  
    - valid_circles: List of [x, y, radius] for 2D validation
    - original_img: Image for path validation
    - max_bond_distance: Maximum 3D bond distance
    - intensity_threshold: Max intensity difference for similar intensity bonding
    - std_threshold: Standard deviation threshold for dark path detection
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"Starting enhanced chain generation:")
    print(f"  Points: {len(points)}")
    print(f"  Max 3D distance: {max_bond_distance}")
    print(f"  Intensity threshold: {intensity_threshold}")
    print(f"  Path std threshold: {std_threshold}")
    
    center_point = find_most_centered_point(points)
    print(f"  Center point: {center_point}")
    
    successful_chains = []
    failed_attempts = 0
    total_unbonded = []
    total_rejected_bonds = []
    
    for sim in range(num_simulations):
        try:
            chain, unbonded_points, rejected_bonds = grow_chain_from_center_enhanced(
                points, center_point, valid_circles, original_img, max_bond_distance,
                intensity_threshold, std_threshold
            )
            
            total_unbonded.extend(unbonded_points)
            total_rejected_bonds.extend(rejected_bonds)
            
            if len(unbonded_points) == 0:  # All points bonded
                total_length = calculate_chain_length(chain, points)
                bonds, bond_distances = convert_chain_to_bonds(chain, points)
                
                successful_chains.append({
                    'chain': chain,
                    'total_length': total_length,
                    'bonds': bonds,
                    'bond_distances': bond_distances,
                    'simulation_id': sim,
                    'unbonded_points': [],
                    'rejected_bonds': rejected_bonds
                })
            else:
                # Partial solution
                if len(chain) > len(points) * 0.7:  # Accept if > 70% bonded
                    total_length = calculate_chain_length(chain, points)
                    bonds, bond_distances = convert_chain_to_bonds(chain, points)
                    
                    successful_chains.append({
                        'chain': chain,
                        'total_length': total_length,
                        'bonds': bonds,
                        'bond_distances': bond_distances,
                        'simulation_id': sim,
                        'unbonded_points': unbonded_points,
                        'rejected_bonds': rejected_bonds
                    })
                else:
                    failed_attempts += 1
                    
        except NoValidPathException:
            failed_attempts += 1
    
    # Print results
    failure_rate = failed_attempts / num_simulations
    print(f"\\nResults: {len(successful_chains)} successful, {failed_attempts} failed")
    print(f"Failure rate: {failure_rate:.2%}")
    
    # Report unbonded points
    if total_unbonded:
        unique_unbonded = list(set(total_unbonded))
        print(f"\\nUnbonded points across all simulations: {unique_unbonded}")
        print(f"Points that couldn't be bonded due to:")
        
        # Analyze rejection reasons
        rejection_reasons = {}
        for bond in total_rejected_bonds:
            reason = bond['reason']
            if reason not in rejection_reasons:
                rejection_reasons[reason] = []
            rejection_reasons[reason].append(bond['point'])
        
        for reason, points_list in rejection_reasons.items():
            unique_points = list(set(points_list))
            print(f"  {reason}: points {unique_points}")
    
    if failure_rate > failure_threshold and len(successful_chains) == 0:
        print(f"\\nNOT POSSIBLE: Constraints too restrictive")
        return None
    
    if len(successful_chains) == 0:
        print("\\nNOT POSSIBLE: No valid chains found")
        return None
    
    # Sort by completeness first (fewer unbonded points), then by length
    successful_chains.sort(key=lambda x: (len(x['unbonded_points']), x['total_length']))
    
    print(f"\\nTop {min(num_best, len(successful_chains))} solutions:")
    for i, chain in enumerate(successful_chains[:num_best]):
        unbonded_count = len(chain['unbonded_points'])
        bonded_count = len(chain['chain'])
        print(f"  Solution {i}: Length={chain['total_length']:.3f}, Bonded={bonded_count}, Unbonded={unbonded_count}")
        if unbonded_count > 0:
            print(f"    Unbonded points: {chain['unbonded_points']}")
    
    return successful_chains[:num_best]

# Keep the original helper functions
def calculate_distance_matrix(points):
    """Calculate distance matrix between all pairs of points"""
    n_points = len(points)
    distances = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                p1 = points[i]
                p2 = points[j]
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
                distances[i][j] = distance
    
    return distances

def find_most_centered_point(points):
    """Find point with minimum maximum distance to all others"""
    n_points = len(points)
    distances = calculate_distance_matrix(points)
    
    max_distances = []
    for i in range(n_points):
        max_dist_from_i = np.max(distances[i])
        max_distances.append(max_dist_from_i)
    
    center_idx = np.argmin(max_distances)
    return center_idx

def calculate_chain_length(chain, points):
    """Calculate total length of the chain"""
    total_length = 0.0
    for i in range(len(chain) - 1):
        p1 = points[chain[i]]
        p2 = points[chain[i + 1]]
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        total_length += distance
    return total_length

def convert_chain_to_bonds(chain, points):
    """Convert chain to bonds matrix and distances dictionary"""
    n_points = len(points)
    bonds = np.zeros((n_points, n_points), dtype=bool)
    bond_distances = {}
    
    for i in range(len(chain) - 1):
        point1 = chain[i]
        point2 = chain[i + 1]
        
        bonds[point1][point2] = True
        bonds[point2][point1] = True
        
        p1 = points[point1]
        p2 = points[point2]
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        bond_distances[(min(point1, point2), max(point1, point2))] = distance
    
    return bonds, bond_distances