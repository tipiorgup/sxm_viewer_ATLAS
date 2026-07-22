import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any

class CircleDetector:
    
    def detect_circles(self, processed_data: Dict, sample_name: str, blob_radius_angstrom: float = 3.0,
                       intensity_threshold: int = None, param1=300,param2=5) -> Tuple[List, np.ndarray]:
        """Detect circles and return valid ones with visualization"""
        data = processed_data[sample_name]
        pixelsize = data['Pixelsize']
        
        # Calculate radius range
        base_radius = self._calculate_radius_pixels(pixelsize, blob_radius_angstrom)
        radius_range = [base_radius, base_radius + self._calculate_radius_pixels(pixelsize, 0.5)]
        
        # Detect circles
        img = data['result_black_bg']
        circles = cv.HoughCircles(
            img, cv.HOUGH_GRADIENT, 1, radius_range[0]*2,
            param1=param1, param2=param2,
            minRadius=radius_range[0], maxRadius=radius_range[1]
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            valid_circles = self._filter_valid_circles(circles, img, intensity_threshold)
            return valid_circles, self._create_visualization(img, valid_circles,sample_name)
        
        return [], None
    
    def extract_3d_points(self, valid_circles: List, original_img: np.ndarray) -> List:
        """Extract 3D coordinates from detected circles"""
        if len(original_img.shape) == 3:
            gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
        else:
            gray_img = original_img
        
        points_3d = []
        for circle in valid_circles:
            x, y = int(circle[0]), int(circle[1])
            z = self._get_region_intensity(gray_img, x, y)
            points_3d.append((x, y, z))
        
        return points_3d
    
    def _calculate_radius_pixels(self, pixelsize, distance_angstrom):
        avg_pixelsize_angstrom = np.mean(pixelsize) * 10
        return int(np.round(distance_angstrom / avg_pixelsize_angstrom))
    
    def _filter_valid_circles(self, circles, img, intensity_threshold):
        return [circle for circle in circles[0,:] 
                if (0 <= circle[1] < img.shape[0] and 0 <= circle[0] < img.shape[1] 
                    and img[circle[1], circle[0]] > intensity_threshold)]
    
    def _get_region_intensity(self, img, x, y, region_size=3):
        y_min = max(0, y - region_size//2)
        y_max = min(img.shape[0], y + region_size//2 + 1)
        x_min = max(0, x - region_size//2)
        x_max = min(img.shape[1], x + region_size//2 + 1)
        return np.mean(img[y_min:y_max, x_min:x_max])
    
    def _create_visualization(self, img, valid_circles, sample_name):
        original_img = img

        plt.figure(figsize=(10, 10))
        plt.imshow(original_img, cmap='viridis')
        plt.colorbar(label='Height (Angstrom)')

        # Overlay circle positions
        for i, circle in enumerate(valid_circles):
            x, y, r = circle[0], circle[1], circle[2]
            circle_plot = plt.Circle((x, y), r, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(circle_plot)
            plt.text(x, y-r-5, str(i), color='yellow', ha='center', fontsize=10, weight='bold')

        plt.title(f'Detected Circles on Original Image - {sample_name}')
        plt.axis('off')
        plt.show()