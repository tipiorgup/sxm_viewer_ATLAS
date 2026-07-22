"""
SXM Image Filtering Functions 

Extracted from PyQt GUI version for web interface use

@author: Anggara
"""

import numpy as np
from scipy.signal.windows import blackmanharris
from skimage.filters import unsharp_mask
from scipy.ndimage import gaussian_filter as blur

def apply_fft_filter(image, filter_type='blackman', **kwargs):
    """
    Apply FFT filtering to image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    filter_type : str
        'blackman', 'gaussian', or 'exponential'
    **kwargs : dict
        sigma : float (for gaussian, default=16)
        decay_rate : float (for exponential, default=0.06)
    
    Returns:
    --------
    numpy.ndarray : Filtered image
    """
    
    # Convert to FFT
    Z_fft = np.fft.fftshift(np.fft.fft2(image))
    
    if filter_type == 'blackman':
        # Blackman-Harris window
        window_1d = blackmanharris(image.shape[0])
        window_2d = np.outer(window_1d, blackmanharris(image.shape[1]))
        window = window_2d
        
    elif filter_type == 'gaussian':
        # Gaussian window
        rows, cols = image.shape
        sigma = kwargs.get('sigma', 16.0)
        if sigma < 1e-3: 
            sigma = 1e-3
            
        x = np.arange(rows) - rows // 2
        y = np.arange(cols) - cols // 2
        X, Y = np.meshgrid(x, y, indexing="ij")
        distance = np.sqrt(X**2 + Y**2)
        window = np.exp(-distance**2 / (2 * sigma**2))
        
    elif filter_type == 'exponential':
        # Exponential window
        rows, cols = image.shape
        decay_rate = kwargs.get('decay_rate', 0.06)
        
        x = np.arange(rows) - rows // 2
        y = np.arange(cols) - cols // 2
        X, Y = np.meshgrid(x, y, indexing="ij")
        distance = np.sqrt(X**2 + Y**2)
        window = np.exp(-decay_rate * distance)
    
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    
    # Apply window and convert back
    Z_fft_windowed = Z_fft * window
    Z_processed = np.fft.ifft2(np.fft.ifftshift(Z_fft_windowed))
    Z_processed = np.abs(Z_processed)
    
    # Normalize
    if (Z_processed.max() - Z_processed.min()) < 1e-3:
        Z_processed = Z_processed - Z_processed.min()
    else:
        Z_processed = (Z_processed - Z_processed.min()) / (Z_processed.max() - Z_processed.min())
    
    return Z_processed

def apply_unsharp_mask(image, radius=5.0, amount=20.0):
    """
    Apply unsharp masking to enhance image details
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    radius : float
        Unsharp mask radius (default=5.0)
    amount : float
        Unsharp mask amount (default=20.0)
    
    Returns:
    --------
    numpy.ndarray : Enhanced image
    """
    return unsharp_mask(image, radius=radius, amount=amount)

def invert_image(image):
    """
    Invert image contrast
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray : Inverted image
    """
    Z = -image
    return Z

def apply_cosine_transform(image, mask=None, blur_sigma=1.0):
    """
    Apply cosine transform based on image gradients
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    mask : numpy.ndarray, optional
        Binary mask for the image
    blur_sigma : float
        Gaussian blur sigma for gradient calculation (default=1.0)
    
    Returns:
    --------
    numpy.ndarray : Cosine-transformed image
    """
    if mask is None:
        mask = np.ones_like(image)
    
    # Calculate gradients
    gy, gx = np.gradient(blur(image, blur_sigma))
    slope_magnitude = np.sqrt(gx**2 + gy**2)
    angles = np.arctan(slope_magnitude * mask)
    
    return np.cos(angles)

def generate_contour_levels(image, num_levels=4):
    """
    Generate contour level values for image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    num_levels : int
        Number of contour levels (default=4)
    
    Returns:
    --------
    list : Contour level values
    """
    return np.linspace(0.5, image.max(), num_levels).tolist()

def process_sxm_image(image, mask=None, filters=None):
    """
    Apply comprehensive image processing pipeline
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input SXM image
    mask : numpy.ndarray, optional
        Binary mask for the image
    filters : dict
        Filter configuration:
        {
            'fft_filter': {'type': 'gaussian', 'sigma': 16.0},
            'unsharp_mask': {'radius': 5.0, 'amount': 20.0},
            'invert': True,
            'cosine': True,
            'blur_sigma': 1.0
        }
    
    Returns:
    --------
    dict : Processing results
        {
            'processed_image': numpy.ndarray,
            'contour_levels': list,
            'processing_steps': list
        }
    """
    
    if filters is None:
        filters = {}
    
    Z = image.copy()
    processing_steps = []
    
    # FFT Filtering
    if 'fft_filter' in filters:
        fft_config = filters['fft_filter']
        filter_type = fft_config.get('type', 'blackman')
        
        if filter_type == 'gaussian':
            Z = apply_fft_filter(Z, 'gaussian', sigma=fft_config.get('sigma', 16.0))
        elif filter_type == 'exponential':
            Z = apply_fft_filter(Z, 'exponential', decay_rate=fft_config.get('decay_rate', 0.06))
        elif filter_type == 'blackman':
            Z = apply_fft_filter(Z, 'blackman')
        
        processing_steps.append(f"FFT Filter: {filter_type}")
    
    # Unsharp Masking
    if 'unsharp_mask' in filters:
        umask_config = filters['unsharp_mask']
        Z = apply_unsharp_mask(
            Z, 
            radius=umask_config.get('radius', 5.0),
            amount=umask_config.get('amount', 20.0)
        )
        processing_steps.append("Unsharp Masking")
    
    # Image Inversion
    if filters.get('invert', False):
        Z = invert_image(Z)
        processing_steps.append("Image Inversion")
    
    # Cosine Transform
    if filters.get('cosine', False):
        blur_sigma = filters.get('blur_sigma', 1.0)
        Z = apply_cosine_transform(Z, mask, blur_sigma)
        processing_steps.append("Cosine Transform")
    
    # Generate contour levels
    contour_levels = generate_contour_levels(Z, filters.get('contour_levels', 4))
    
    return {
        'processed_image': Z,
        'contour_levels': contour_levels,
        'processing_steps': processing_steps
    }

# Validation functions for parameter ranges
def validate_fft_gaussian_sigma(sigma):
    """Validate and clamp FFT Gaussian sigma parameter"""
    return np.clip(float(sigma), 0, 100)

def validate_fft_exp_decay(decay_rate):
    """Validate and clamp FFT Exponential decay rate parameter"""
    return np.clip(float(decay_rate), 1e-3, 1)

def validate_umask_radius(radius):
    """Validate and clamp Unsharp Mask radius parameter"""
    return np.clip(float(radius), 0, 50)

def validate_umask_amount(amount):
    """Validate and clamp Unsharp Mask amount parameter"""
    return np.clip(float(amount), 0, 50)

def display_sxm_image(image, title="SXM Image", colormap='viridis', pixel_size=None):
    """
    Display SXM image using matplotlib
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image data to display
    title : str
        Plot title
    colormap : str
        Matplotlib colormap name
    pixel_size : tuple or None
        (x_size, y_size) in nm/pixel for axis scaling
    
    Returns:
    --------
    matplotlib.figure.Figure : Figure object for Streamlit
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(image, cmap=colormap, origin='lower')
    ax.set_title(title)
    
    if pixel_size is not None:
        # Convert pixel coordinates to nm
        height, width = image.shape
        x_extent = width * pixel_size[0]
        y_extent = height * pixel_size[1]
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        im.set_extent([0, x_extent, 0, y_extent])
    else:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
    
    plt.colorbar(im, ax=ax, label="Height (Å)")
    return fig