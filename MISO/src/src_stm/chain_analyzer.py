import numpy as np
import cv2 as cv
from typing import List, Dict, Optional
from .chain_functions import generate_chain_networks_enhanced 
from .utils import calculate_circle_radii_in_pixels

class ChainAnalyzer:
    def __init__(self):
        # Chain parameters
        self.num_simulations = 1000
        self.intensity_threshold = 100
        self.std_threshold = 30

    def _calculate_connection_profiles(self, chain, valid_circles, img, pixelsize, num_sample_points=50):
        """Calculate intensity profiles for all connections in the chain"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        else:
            gray_img = img.copy()
        
        connection_profiles = []
        
        for i in range(len(chain) - 1):
            point1_idx = chain[i]
            point2_idx = chain[i + 1]
            
            # Get circle coordinates
            x1, y1 = valid_circles[point1_idx][0], valid_circles[point1_idx][1]
            x2, y2 = valid_circles[point2_idx][0], valid_circles[point2_idx][1]
            
            # Calculate total distance in nm
            total_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2) * pixelsize
            
            # Generate sample points along the connection
            x_coords = np.linspace(x1, x2, num_sample_points)
            y_coords = np.linspace(y1, y2, num_sample_points)
            distances = np.linspace(0, total_distance, num_sample_points)
            
            # Sample intensities using bilinear interpolation
            intensities = []
            for x, y in zip(x_coords, y_coords):
                x_int, y_int = int(x), int(y)
                if 0 <= x_int < gray_img.shape[1]-1 and 0 <= y_int < gray_img.shape[0]-1:
                    dx, dy = x - x_int, y - y_int
                    intensity = (gray_img[y_int, x_int] * (1-dx) * (1-dy) +
                            gray_img[y_int, x_int+1] * dx * (1-dy) +
                            gray_img[y_int+1, x_int] * (1-dx) * dy +
                            gray_img[y_int+1, x_int+1] * dx * dy)
                else:
                    intensity = gray_img[max(0, min(gray_img.shape[0]-1, int(y))),
                                    max(0, min(gray_img.shape[1]-1, int(x)))]
                intensities.append(intensity)
            
            # Convert to numpy array and normalize
            intensities = np.array(intensities)
            normalized_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            
            # Store the profile data
            profile = {
                'start_circle': point1_idx,
                'end_circle': point2_idx,
                'distances': distances,  # in nm
                'raw_intensities': intensities,
                'normalized_intensities': normalized_intensities,
                'total_distance': total_distance
            }
            connection_profiles.append(profile)
        
        return connection_profiles
    
    def analyze_chains(self, points_3d: List, valid_circles: List,
                    processed_data: Dict, sample_name: str,
                    max_bond_distance_angstrom: float = 15.0) -> Optional[List]:
        """Generate chain networks from detected points"""
    
        # Get pixel size and convert to Angstroms
        pixelsize = processed_data[sample_name]['Pixelsize'][0]
        pixelsize_angstrom = pixelsize * 10  # nm to Angstrom
    
        # Convert points from pixels to Angstroms
        points_3d_angstrom = [(x * pixelsize_angstrom, y * pixelsize_angstrom, z) 
                            for x, y, z in points_3d]
    
        # Convert valid_circles from pixels to Angstroms (x, y positions only)
        valid_circles_angstrom = [[x * pixelsize_angstrom, y * pixelsize_angstrom, r] 
                                for x, y, r in valid_circles]
    
        # Run chain generation (now everything in Angstrom space)
        solutions = generate_chain_networks_enhanced(
            points=points_3d_angstrom,
            valid_circles=valid_circles_angstrom,
            original_img=processed_data[sample_name]['result_black_bg'],
            max_bond_distance=max_bond_distance_angstrom,  # Keep as 15.0 Å
            pixelsize_angstrom=pixelsize_angstrom,  # Pass the conversion factor
            intensity_threshold=self.intensity_threshold,
            std_threshold=self.std_threshold
        )
    
        if solutions:
            # Add connection profiles to each solution
            for solution in solutions:
                solution['connection_profiles'] = self._calculate_connection_profiles(
                    solution['chain'], valid_circles_angstrom,  # Use Angstrom version
                    processed_data[sample_name]['result_black_bg'],
                    pixelsize  # Pass original pixelsize for any remaining pixel operations
                )
        
            return self._get_unique_solutions(solutions)
        return None
    
    def create_final_analysis(self, best_solution: Dict, valid_circles: List,
                            processed_data: Dict, sample_name: str) -> Dict:
        """Create final analysis dictionary with all results"""
        chain = best_solution['chain']
        pixelsize = processed_data[sample_name]['Pixelsize'][0]
        
        # Extract coordinates
        chain_coords_nm = self._extract_chain_coordinates(
            chain, valid_circles, processed_data[sample_name]['original_img'], pixelsize
        )
        
        return {
            'sample_name': sample_name,
            'total_circles_detected': len(valid_circles),
            'circles_in_chain': len(chain),
            'backbone_chain_3d_nm': np.array(chain_coords_nm),
            'total_length_nm': best_solution['total_length'] * pixelsize,
            'pixelsize_nm_per_pixel': pixelsize,
            'connection_profiles': best_solution['connection_profiles'],
            'best_solution': best_solution  # Keep original solution data
        }
    
    
    def _get_unique_solutions(self, solutions):
        """Check how many solutions are actually unique (including reverse direction check)"""
        if not solutions:
            return []
    
        unique_solutions = []
        seen_chains = set()
        duplicate_count = 0
        reverse_count = 0
    
        for i, sol in enumerate(solutions):
            chain = sol['chain']
            chain_tuple = tuple(chain)
            reverse_tuple = tuple(reversed(chain))
        
            # Check if we've seen this chain or its reverse
            if chain_tuple not in seen_chains and reverse_tuple not in seen_chains:
                unique_solutions.append(sol)
                seen_chains.add(chain_tuple)
                print(f"Solution {i}: unique chain {chain[:3] if len(chain) >= 3 else chain}...{chain[-3:] if len(chain) >= 3 else []}")
            else:
                if chain_tuple in seen_chains:
                    duplicate_count += 1
                    print(f"Solution {i}: exact duplicate of existing chain")
                else:  # reverse_tuple in seen_chains
                    reverse_count += 1
                    print(f"Solution {i}: reverse direction of existing chain")
    
        print(f"Total solutions: {len(solutions)}")
        print(f"Unique solutions: {len(unique_solutions)}")
        print(f"Exact duplicates: {duplicate_count}")
        print(f"Reverse duplicates: {reverse_count}")
    
        return unique_solutions
    
    def _extract_chain_coordinates(self, chain, valid_circles, img_array, pixelsize):
        """Extract 3D coordinates in nm for the chain"""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray_img = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        else:
            gray_img = img_array
        
        chain_coords_nm = []
        for point_idx in chain:
            x_pix, y_pix = valid_circles[point_idx][0], valid_circles[point_idx][1]
            
            # Calculate z (height)
            region_size = 3
            y_min = max(0, int(y_pix) - region_size//2)
            y_max = min(gray_img.shape[0], int(y_pix) + region_size//2 + 1)
            x_min = max(0, int(x_pix) - region_size//2)
            x_max = min(gray_img.shape[1], int(x_pix) + region_size//2 + 1)
            z_pix = np.mean(gray_img[y_min:y_max, x_min:x_max])
            
            # Convert to nm
            x_nm = float(x_pix) * pixelsize
            y_nm = float(y_pix) * pixelsize
            z_nm = float(z_pix) * pixelsize
            
            chain_coords_nm.append([x_nm, y_nm, z_nm])
        
        return chain_coords_nm  # Just return the list, convert to array in calling method