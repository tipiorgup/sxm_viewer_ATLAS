"""
Utility functions for STM analysis
Contains vector operations, image processing utilities, and helper functions
"""
import numpy as np
from typing import Union, Tuple

# Vector calculation functions
def length(v1: np.ndarray) -> float:
    """Calculate vector length/magnitude"""
    return np.linalg.norm(v1)

def dist(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate distance between two vectors"""
    return length(v2 - v1)

def uvec(v1: np.ndarray) -> np.ndarray:
    """Return unit vector (normalized)"""
    return v1 / length(v1)

def nor(v1: np.ndarray) -> np.ndarray:
    """Normalize vector (alias for uvec for backwards compatibility)"""
    return v1 / np.linalg.norm(v1)

def angle(v1: np.ndarray, v2: np.ndarray, allpos: bool = True) -> float:
    """
    Calculate angle between vectors in degrees
    
    Args:
        v1, v2: Input vectors
        allpos: If True, return positive angle (0-360°), else (-180° to 180°)
    
    Returns:
        Angle in degrees. Positive = Clockwise
    """
    a = np.degrees(np.arccos(np.dot(uvec(v1), uvec(v2))))
    if allpos:
        return (a + 360) % 360
    else:
        return a

def rot(v1: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate 2D vector by angle (counter-clockwise)
    
    Args:
        v1: Input vector [x, y]
        angle_deg: Rotation angle in degrees
    
    Returns:
        Rotated vector
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    v2 = np.zeros(v1.shape)
    v2[0] = v1[0] * cos_a - v1[1] * sin_a
    v2[1] = v1[0] * sin_a + v1[1] * cos_a
    return v2

# Image processing utilities
def nlz(image: np.ndarray) -> np.ndarray:
    """Normalize image to range [0, 1]"""
    image_normalized = image - image.min()
    return image_normalized / image_normalized.max()

# String parsing utility
def parse(s: str, delim: str = ' ', ty: str = 'str') -> list:
    """
    Parse string with delimiter and convert to specified type
    
    Args:
        s: Input string
        delim: Delimiter character
        ty: Type to convert to ('str', 'flo', 'int')
    
    Returns:
        List of parsed and converted values
    """
    temp = []
    s = s.lstrip()
    
    # Clean up multiple spaces
    while s.find('  ') > 0:
        s = s.replace('  ', ' ')
    
    while s:
        if delim in s:
            token = s[0:s.find(delim)].strip()
            s = s[s.find(delim)+1:].lstrip()
        else:
            token = s
            s = ''
        
        # Convert based on type
        if ty == 'flo':
            temp.append(float(token))
        elif ty == 'int':
            temp.append(int(token))
        elif ty == 'str':
            temp.append(token)
        else:
            raise ValueError(f"Unknown type: {ty}")
    
    return temp

# Additional utilities that might be useful
def calculate_entropy(image: np.ndarray) -> float:
    """Calculate Shannon entropy of image"""
    histogram, _ = np.histogram(image, bins=256)
    histogram = histogram[histogram > 0]  # Remove zero bins
    return -np.sum((histogram / histogram.sum()) * np.log2(histogram / histogram.sum()))

def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Deep update dictionary (recursively merge)"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict