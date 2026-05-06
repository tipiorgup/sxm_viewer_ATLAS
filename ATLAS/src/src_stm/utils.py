import json 
import numpy as np

# Load the JSON configuration file
def load_paths(config_file='paths.json'):
    """Load path configuration from JSON file"""
    with open(config_file, 'r') as f:
        paths = json.load(f)
    return paths

def deep_update(dict1, dict2):
    """Update dictionaries"""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_update(dict1[key], value)
        else:
            dict1[key] = value

def calculate_circle_radii_in_pixels(pixelsize, distance_angstrom=3.0):
    # Use average pixel size (they're nearly identical anyway)
    avg_pixelsize = np.mean(pixelsize)  # nm/pixel
    
    # Convert pixelsize to Angstrom/pixel for consistent units
    avg_pixelsize_angstrom = avg_pixelsize * 10  # Angstrom/pixel
    
    # Convert to pixels: (Angstroms) / (Angstrom/pixel) = pixels
    min_radius_px = int(np.round(distance_angstrom / avg_pixelsize_angstrom))
    
    return min_radius_px