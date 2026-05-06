"""
SXM file loading and processing utilities
Handles Nanonis SXM file format with calibration and plane correction

@author: Anggara
"""
import os
import numpy as np
import scipy.ndimage as snd
import scipy.linalg
import nanonispy2 as nap
from typing import Optional, Dict, Any, List

# Fix numpy compatibility
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

class SXMLoader:
    """Class to handle SXM file loading with calibration parameters"""
    
    def __init__(self):
        # CALIB MAY 2021 ONWARDS (REF: MUC1Tn Folder --> 210519_Cu100_056.sxm)
        self.calibration = {
            'Xscale': 0.861735094047142000,
            'XYcrosstalk': 0.000476218259800334,
            'YXcrosstalk': -0.01765821454416330,
            'Yscale': 0.867834394953107000,
            'Zscale': 1.000000000000000000,
            'Psi': 6.800000000000000000
        }
    
    def load_sxm_files_from_directory(self, directory: str = None, 
                                     verbose: bool = False) -> Dict[str, Any]:
        """
        Load all SXM files from a directory
        
        Args:
            directory: Directory path (uses current directory if None)
            verbose: Print loading progress
            
        Returns:
            Dictionary with filename (no extension) as key, processed data as value
        """
        if directory is None:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = os.listdir(directory)
        sxm_files = [f for f in files if f.endswith(".sxm")]
        
        if not sxm_files:
            print(f"No SXM files found in directory: {directory}")
            return {}
        
        if verbose:
            print(f"Found {len(sxm_files)} SXM files in {directory}")
        
        sxm_data = {}
        successful_loads = 0
        
        for file in sxm_files:
            file_path = os.path.join(directory, file)
            filename_without_ext = os.path.splitext(file)[0]
            
            data = self.load_sxm_file(file_path, verbose=verbose)
            if data is not None:
                sxm_data[filename_without_ext] = data
                successful_loads += 1
            else:
                print(f"Failed to load: {file}")
        
        if verbose:
            print(f"Successfully loaded {successful_loads}/{len(sxm_files)} files")
        
        return sxm_data
    
    def load_sxm_file(self, file_path: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load single SXM file
        
        Args:
            file_path: Path to SXM file
            verbose: Print loading status
            
        Returns:
            Dictionary containing processed SXM data or None if failed
        """
        try:
            sxm_data = self._read_sxm(file_path)
            if sxm_data is not None:
                if verbose:
                    print(f"Loaded: {os.path.basename(file_path)}")
                return sxm_data
            else:
                if verbose:
                    print(f"Failed: {os.path.basename(file_path)}")
                return None
        except Exception as e:
            if verbose:
                print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            return None
    
    def get_sxm_file_list(self, directory: str = None) -> List[str]:
        """
        Get list of SXM files in directory
        
        Args:
            directory: Directory to search (current directory if None)
            
        Returns:
            List of SXM file paths
        """
        if directory is None:
            directory = os.getcwd()
        
        files = os.listdir(directory)
        return [os.path.join(directory, f) for f in files if f.endswith(".sxm")]
    
    def update_calibration(self, calibration_dict: Dict[str, float]) -> None:
        """Update calibration parameters"""
        self.calibration.update(calibration_dict)
    
    def _read_sxm(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Internal method to read and process SXM file"""
        try:
            dat = nap.read.Scan(file_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f'File access error for {file_path}: {e}')
            return None
        except Exception as e:
            print(f'Cannot parse SXM file {file_path}: {e}')
            return None
        
        # Process the raw data
        processed_data = self._process_sxm_data(dat)
        
        return {
            'originalimg': processed_data['Zcorr'],
            'img': processed_data['Zcorr'],
            'Zmask': processed_data['Zmask'],
            'Pixelsize': processed_data['Pixelsize'],
            'header': dat.header,
            'raw_data': dat,
            'acq_time': dat.header['acq_time'],
            'bias': dat.header['bias']
        }
    
    def _process_sxm_data(self, dat: Any) -> Dict[str, Any]:
        """Process SXM data with calibration and plane correction"""
        
        # Raw Z data - convert to Angstrom
        Z = dat.signals['Z']['forward'] * 1e10
        
        # Handle NaN values by cropping
        if True in np.isnan(Z):
            Z = self._crop_nan_regions(Z)
        
        # Correct for scan direction
        if dat.header['scan_dir'] == 'down':
            Z = np.flipud(Z)
        
        # Apply plane corrections
        Z = self._apply_plane_correction(Z)
        
        # Apply calibration
        Zcorr = self._apply_calibration(Z)
        
        # Create mask
        Zmask = self._create_mask(Zcorr)
        
        # Calculate pixel size
        Pixelsize = self._calculate_pixelsize(dat)
        
        # Handle non-square pixels if necessary
        Zcorr, Zmask, Pixelsize = self._handle_non_square_pixels(
            Zcorr, Zmask, Pixelsize, dat
        )
        
        return {
            'Zcorr': Zcorr,
            'Zmask': Zmask,
            'Pixelsize': Pixelsize
        }
    
    def _crop_nan_regions(self, Z: np.ndarray) -> np.ndarray:
        """Crop image to remove NaN regions"""
        rows = np.any(~np.isnan(Z), axis=1)
        cols = np.any(~np.isnan(Z), axis=0)
        
        row_start, row_end = np.where(rows)[0][[0, -2]]
        col_start, col_end = np.where(cols)[0][[0, -1]]
        
        return Z[row_start:row_end+1, col_start:col_end+1]
    
    def _apply_plane_correction(self, Z: np.ndarray) -> np.ndarray:
        """Apply rough and precise plane correction"""
        # Rough plane correction
        Ztest = Z - self._plane2(Z)
        Ztest = Ztest - Ztest.min()
        
        # Precise plane correction
        lim = 0
        count = 0
        Zcheck = Ztest
        
        while abs(np.median(Zcheck)) > 0.005:
            upp_lim = lim + 0.2
            low_lim = lim
            
            mask = np.logical_and(
                Ztest < (upp_lim * Ztest.max()),
                Ztest > (low_lim * Ztest.max())
            )
            
            Zcheck = Z - self._plane2(Z, mask=mask)
            lim += 0.005
            count += 1
            
            if count > 100:
                print('Precise deplaning fails - using rough planing')
                Zcheck = Ztest
                break
        
        return Zcheck
    
    def _apply_calibration(self, Z: np.ndarray) -> np.ndarray:
        """Apply calibration matrix to correct image distortion"""
        calib = np.array([
            [self.calibration['Xscale'], self.calibration['XYcrosstalk'], 0],
            [self.calibration['YXcrosstalk'], self.calibration['Yscale'], 0],
            [0, 0, 1]
        ])
        
        return scipy.ndimage.affine_transform(
            Z, np.linalg.inv(calib), mode='constant', cval=0
        )
    
    def _create_mask(self, Zcorr: np.ndarray) -> np.ndarray:
        """Create binary mask for valid data regions"""
        Zmask = np.ones(Zcorr.shape)
        Zmask[Zcorr == 0] = False
        Zmask[Zcorr != 0] = True
        return snd.binary_erosion(Zmask, iterations=5)
    
    def _calculate_pixelsize(self, dat: Any) -> np.ndarray:
        """Calculate pixel size in nm/pixel"""
        return dat.header['scan_range'] * 1e9 / dat.header['scan_pixels']
    
    def _handle_non_square_pixels(self, Zcorr: np.ndarray, Zmask: np.ndarray, 
                                 Pixelsize: np.ndarray, dat: Any) -> tuple:
        """Handle non-square pixels by resizing"""
        if Pixelsize[0] - Pixelsize[1] > 1e-2:
            print(f'WARNING - PIXELS ARE NON-SQUARE: {Pixelsize}')
            
            original_shape = Zcorr.shape
            print(f'OLD IMAGE SHAPE: {original_shape} pixels')
            
            aspect_ratio = original_shape[1] / original_shape[0]
            
            # Calculate zoom factors
            if aspect_ratio > 1:
                zoom_factors = (aspect_ratio, 1)
            else:
                zoom_factors = (1, 1/aspect_ratio)
            
            # Resize arrays
            Zcorr = snd.zoom(Zcorr, zoom_factors)
            Zmask = snd.zoom(Zmask, zoom_factors)
            
            print(f'NEW IMAGE SHAPE: {Zcorr.shape} pixels')
            
            # Recalculate pixel size using the method consistently
            Pixelsize = 1e9 * dat.header['scan_range'] / np.array(Zcorr.shape)
            print(f'NEW PIXELSIZE: {Pixelsize}')
        
        return Zcorr, Zmask, Pixelsize
    
    def _plane2(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Corrects the plane of a 2D numpy array by subtracting the best-fit plane
        
        Args:
            image: 2D numpy array to correct
            mask: Optional mask to determine which pixels to use for fitting
            
        Returns:
            Best-fit plane as 2D array
        """
        height, width = image.shape
        y, x = np.mgrid[:height, :width]
        
        if mask is not None:
            if mask.shape != image.shape:
                raise ValueError("Mask must have the same shape as the input image")
            valid_pixels = mask != 0
            x = x[valid_pixels]
            y = y[valid_pixels]
            z = image[valid_pixels]
        else:
            x = x.flatten()
            y = y.flatten()
            z = image.flatten()
        
        A = np.column_stack((x, y, np.ones_like(x)))
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        a, b, c = coeffs
        
        y, x = np.mgrid[:height, :width]
        return a*x + b*y + c

# Convenience functions for backwards compatibility
def load_sxm_file(file_path: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Load single SXM file"""
    loader = SXMLoader()
    return loader.load_sxm_file(file_path, verbose)

def load_sxm_files(directory: str = None, verbose: bool = False) -> Dict[str, Any]:
    """Load all SXM files from directory"""
    loader = SXMLoader()
    return loader.load_sxm_files_from_directory(directory, verbose)