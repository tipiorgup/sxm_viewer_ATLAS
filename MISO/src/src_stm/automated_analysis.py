# Core libraries
import os
import numpy as np
import pandas as pd
import cv2

# Matplotlib
import matplotlib.pyplot as plt

# Scikit-image
from skimage.io import imread
from skimage.util import img_as_ubyte

# Scipy
from scipy.stats import entropy

def calculate_entropy_from_array(image_array, bins=256):
    """Calculate entropy directly from numpy array"""
    # Calculate histogram
    hist, _ = np.histogram(image_array.ravel(), bins=bins)
    
    # Calculate probability distribution (remove zeros to avoid log(0))
    prob_dist = hist[hist > 0] / hist.sum()
    
    # Calculate entropy (base 2 for bits)
    image_entropy = entropy(prob_dist, base=2)
    
    return image_entropy

def extract_features_to_dataframe(data_dict, target_size=(128, 128), include_image_pixels=False):
    """
    Extract image features and return as pandas DataFrame
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing image data and metadata
    target_size : tuple
        Target size for image resizing (height, width)
    include_image_pixels : bool
        Whether to include flattened image pixels as features (can be very large)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with samples as rows and features as columns
    """
    
    names = list(data_dict.keys())
    all_data = []
    
    for name in names:
        # Process image
        img = data_dict[name]['img']
        if len(img.shape) == 2:  # Grayscale
            resized = cv2.resize(img, target_size)
            gray = img.copy()
        else:  # Color
            resized = cv2.resize(img, target_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Start building feature dictionary for this sample
        feature_dict = {'sample_name': name}
        
        # Original metadata features
        feature_dict['bias'] = data_dict[name]['bias']
        feature_dict['acq_time'] = data_dict[name]['acq_time']
        
        # Intensity features
        feature_dict['max_intensity'] = np.max(img)
        feature_dict['min_intensity'] = np.min(img)
        feature_dict['intensity_range'] = feature_dict['max_intensity'] - feature_dict['min_intensity']
        feature_dict['mean_intensity'] = np.mean(img)
        feature_dict['std_intensity'] = np.std(img)
        feature_dict['intensity_skewness'] = calculate_skewness(img.flatten())
        feature_dict['intensity_kurtosis'] = calculate_kurtosis(img.flatten())

        #Entropy
        feature_dict['entropy']=calculate_entropy_from_array(img)
        
        # Edge features
        feature_dict['edge_density'] = calculate_edge_density_from_array(img)
        
        # Object counting features
        feature_dict['num_contours'] = count_objects_contours(gray)
        
        # Texture features
        feature_dict['contrast'] = calculate_contrast(gray)
        feature_dict['homogeneity'] = calculate_homogeneity(gray)
        
        # Shape/structure features
        feature_dict['laplacian_variance'] = calculate_sharpness(gray)
        feature_dict['gradient_magnitude'] = calculate_gradient_strength(gray)
        
        # Frequency domain features
        freq_high, freq_low = calculate_frequency_features(gray)
        feature_dict['freq_energy_high'] = freq_high
        feature_dict['freq_energy_low'] = freq_low
        feature_dict['freq_ratio_high_low'] = freq_high / (freq_low + 1e-8)  # Avoid division by zero

        
        # Image shape features
        feature_dict['image_height'] = img.shape[0]
        feature_dict['image_width'] = img.shape[1]
        feature_dict['image_aspect_ratio'] = img.shape[1] / img.shape[0]
        feature_dict['image_area'] = img.shape[0] * img.shape[1]
        
        # Add flattened image pixels if requested (warning: can be very large!)
        if include_image_pixels:
            flattened_pixels = resized.flatten()
            for i, pixel_val in enumerate(flattened_pixels):
                feature_dict[f'pixel_{i}'] = pixel_val
        
        all_data.append(feature_dict)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Set sample_name as index
    df.set_index('sample_name', inplace=True)
    
    print(f"\nDataFrame created with shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    return df

def calculate_skewness(data):
    """Calculate skewness of data"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0
    return np.mean(((data - mean_val) / std_val) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis of data"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0
    return np.mean(((data - mean_val) / std_val) ** 4) - 3

def count_objects_contours(gray_img, min_area=50):
    """Count objects using contour detection"""
    # Convert to uint8 if needed
    if gray_img.dtype != np.uint8:
        # Normalize to 0-255 range and convert to uint8
        gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return len(valid_contours)

def calculate_edge_density_from_array(img_array, threshold1=50, threshold2=150):
    """Calculate edge density from numpy array instead of file path"""
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()
        
        # Convert to uint8 if needed
        if gray.dtype != np.uint8:
            gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Calculate edge density (percentage of edge pixels)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return edge_density
    except:
        return 0

def calculate_sharpness(gray_img):
    """Calculate sharpness using Laplacian variance"""
    # Convert to appropriate type for Laplacian
    if gray_img.dtype != np.uint8:
        gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
    
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return laplacian.var()

def calculate_gradient_strength(gray_img):
    """Calculate average gradient magnitude"""
    # Convert to appropriate type for Sobel
    if gray_img.dtype != np.uint8:
        gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
    
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(gradient_magnitude)

def calculate_contrast(gray_img):
    """Calculate local contrast using standard deviation of pixel intensities"""
    return np.std(gray_img) / np.mean(gray_img) if np.mean(gray_img) > 0 else 0

def calculate_homogeneity(gray_img):
    """Calculate homogeneity (inverse of contrast)"""
    # Using local binary patterns or simple variance measure
    kernel = np.ones((5,5), np.float32) / 25
    smooth = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
    variance = np.var(gray_img - smooth)
    return 1 / (1 + variance)

def calculate_sharpness(gray_img):
    """Calculate sharpness using Laplacian variance"""
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return laplacian.var()

def calculate_gradient_strength(gray_img):
    """Calculate average gradient magnitude"""
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(gradient_magnitude)

def calculate_frequency_features(gray_img):
    """Calculate frequency domain features using FFT"""
    # Apply FFT
    fft = np.fft.fft2(gray_img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Get center coordinates
    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2
    
    # Create frequency masks
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # High frequency (edges, details)
    high_freq_mask = distance > min(h, w) * 0.1
    high_freq_energy = np.sum(magnitude[high_freq_mask])
    
    # Low frequency (general structure)
    low_freq_mask = distance <= min(h, w) * 0.1
    low_freq_energy = np.sum(magnitude[low_freq_mask])
    
    return high_freq_energy, low_freq_energy

def analyze_image_characteristics(additional_features, names):
    """
    Analyze image characteristics and recommend filters
    
    Features order (14 total):
    [bias, entropy, acq_time, intensity_range, mean_intensity, std_intensity,
     edge_density, num_contours, contrast, homogeneity, laplacian_variance, 
     gradient_magnitude, freq_energy_high, freq_energy_low]
    """
    
    # Create DataFrame for easier analysis
    feature_names = [
        'bias', 'entropy', 'acq_time', 'intensity_range', 'mean_intensity', 
        'std_intensity', 'edge_density', 'num_contours', 'contrast', 
        'homogeneity', 'laplacian_variance', 'gradient_magnitude', 
        'freq_energy_high', 'freq_energy_low'
    ]
    
    df = pd.DataFrame(additional_features, columns=feature_names, index=names)
    
    # Calculate percentiles for classification
    percentiles = {}
    for col in feature_names:
        percentiles[col] = {
            'low': np.percentile(df[col], 33),
            'high': np.percentile(df[col], 67)
        }
    
    return df, percentiles

def recommend_filters(image_name, features_row, percentiles):
    """
    Recommend filters based on image characteristics
    """
    recommendations = []
    
    # Extract key metrics
    edge_density = features_row['edge_density']
    contrast = features_row['contrast']
    laplacian_variance = features_row['laplacian_variance']
    freq_energy_high = features_row['freq_energy_high']
    freq_energy_low = features_row['freq_energy_low']
    homogeneity = features_row['homogeneity']
    num_contours = features_row['num_contours']
    std_intensity = features_row['std_intensity']
    
    # Noise reduction filters
    if std_intensity > percentiles['std_intensity']['high'] or contrast < percentiles['contrast']['low']:
        recommendations.append({
            'filter': 'Gaussian',
            'reason': 'High noise/low contrast - smooth noise while preserving structure',
            'priority': 'High',
            'params': 'sigma=1-2'
        })
        
        recommendations.append({
            'filter': 'Blackman Harris',
            'reason': 'Advanced noise reduction with better frequency response',
            'priority': 'Medium',
            'params': 'window_size=5-7'
        })
    
    # Edge enhancement filters
    if edge_density < percentiles['edge_density']['low'] or laplacian_variance < percentiles['laplacian_variance']['low']:
        recommendations.append({
            'filter': 'Unsharp Mask',
            'reason': 'Low edge content - enhance edges and fine details',
            'priority': 'High',
            'params': 'radius=1-3, amount=0.5-1.5'
        })
    
    # Frequency domain processing
    freq_ratio = freq_energy_high / (freq_energy_low + 1e-10)  # Avoid division by zero
    
    if freq_ratio < percentiles['freq_energy_high']['low'] / (percentiles['freq_energy_low']['high'] + 1e-10):
        recommendations.append({
            'filter': 'FFT High-pass',
            'reason': 'Low high-frequency content - enhance details and edges',
            'priority': 'Medium',
            'params': 'cutoff=0.1-0.3'
        })
    elif freq_ratio > percentiles['freq_energy_high']['high'] / (percentiles['freq_energy_low']['low'] + 1e-10):
        recommendations.append({
            'filter': 'FFT Low-pass',
            'reason': 'High noise in high frequencies - smooth while preserving main features',
            'priority': 'Medium',
            'params': 'cutoff=0.7-0.9'
        })
    
    # Contrast enhancement
    if contrast < percentiles['contrast']['low']:
        recommendations.append({
            'filter': 'Cosine',
            'reason': 'Low contrast - enhance dynamic range with smooth transition',
            'priority': 'Medium',
            'params': 'alpha=0.5-1.0'
        })
    
    # Special cases
    if homogeneity > percentiles['homogeneity']['high']:
        recommendations.append({
            'filter': 'Unsharp Mask',
            'reason': 'Very uniform image - add local contrast enhancement',
            'priority': 'High',
            'params': 'radius=2-4, amount=1.0-2.0'
        })
    
    if num_contours < percentiles['num_contours']['low']:
        recommendations.append({
            'filter': 'FFT High-pass + Unsharp Mask',
            'reason': 'Few objects detected - enhance object boundaries',
            'priority': 'High',
            'params': 'Combined processing'
        })
    
    # Always consider inverse if other enhancements don't work
    recommendations.append({
        'filter': 'Inverse',
        'reason': 'Alternative perspective - may reveal hidden features',
        'priority': 'Low',
        'params': 'Simple inversion'
    })
    
    return recommendations

def create_filter_strategy(df, percentiles):
    """
    Create comprehensive filter strategy for all images
    """
    strategy = {}
    
    for image_name in df.index:
        features_row = df.loc[image_name]
        recommendations = recommend_filters(image_name, features_row, percentiles)
        
        # Sort by priority
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        recommendations.sort(key=lambda x: priority_order[x['priority']], reverse=True)
        
        strategy[image_name] = recommendations
    
    return strategy

def get_processing_pipeline(image_name, strategy):
    """
    Get optimal processing pipeline for a specific image
    """
    recommendations = strategy[image_name]
    high_priority = [r for r in recommendations if r['priority'] == 'High']
    
    if len(high_priority) == 0:
        # Use medium priority if no high priority
        medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
        return medium_priority[:2] if medium_priority else recommendations[:1]
    
    return high_priority

def analyze_clusters(feature_frame, clusters):
    """
    Analyze image characteristics by cluster and recommend filters per cluster
    """
    feature_keys = [
        'bias', 'entropy', 'acq_time',
        'intensity_range', 'mean_intensity', 'std_intensity',
        'edge_density', 'num_contours', 'contrast', 'homogeneity',
        'laplacian_variance', 'gradient_magnitude',
        'freq_energy_high', 'freq_energy_low'
    ]
    
    df = feature_frame[feature_keys].copy()
    df['cluster'] = clusters
    
    # Calculate overall percentiles for reference
    
    percentiles = {}
    for col in feature_keys:
        percentiles[col] = {
            'low': np.percentile(df[col], 33),
            'high': np.percentile(df[col], 67)
        }
    
    # Calculate cluster statistics
    cluster_stats = {}
    unique_clusters = sorted(df['cluster'].unique())
    
    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Calculate mean, std, and other stats for each feature
        stats = {}
        for feature in feature_keys:
            stats[feature] = {
                'mean': cluster_data[feature].mean(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max(),
                'count': len(cluster_data)
            }
        
        cluster_stats[cluster_id] = stats
    
    return df, cluster_stats, percentiles, unique_clusters

def create_cluster_filter_configs(cluster_id, cluster_stats, global_percentiles):
    """
    Create practical filter configuration dictionary for a cluster
    """
    stats = cluster_stats[cluster_id]
    
    # Extract key metrics (using mean values for cluster)
    edge_density = stats['edge_density']['mean']
    contrast = stats['contrast']['mean']
    laplacian_variance = stats['laplacian_variance']['mean']
    freq_energy_high = stats['freq_energy_high']['mean']
    freq_energy_low = stats['freq_energy_low']['mean']
    homogeneity = stats['homogeneity']['mean']
    num_contours = stats['num_contours']['mean']
    std_intensity = stats['std_intensity']['mean']
    
    # Initialize filter config
    filter_config = {
        'fft_filter': {'type': None, 'sigma': 16.0, 'decay_rate': 0.06},
        'unsharp_mask': {'radius': 5.0, 'amount': 20.0},
        'invert': False,
        'cosine': False,
        'blur_sigma': 1.0
    }
    
    # Determine FFT filter type and parameters
    freq_ratio = freq_energy_high / (freq_energy_low + 1e-10)
    global_freq_ratio_median = np.median([
        global_percentiles['freq_energy_high']['low'] / (global_percentiles['freq_energy_low']['high'] + 1e-10),
        global_percentiles['freq_energy_high']['high'] / (global_percentiles['freq_energy_low']['low'] + 1e-10)
    ])
    
    # High noise or low contrast -> Gaussian smoothing
    if std_intensity > global_percentiles['std_intensity']['high'] or contrast < global_percentiles['contrast']['low']:
        filter_config['fft_filter']['type'] = 'gaussian'
        # Higher sigma for noisier images
        if std_intensity > global_percentiles['std_intensity']['high'] * 1.5:
            filter_config['fft_filter']['sigma'] = 20.0
        else:
            filter_config['fft_filter']['sigma'] = 16.0
    
    # Low high-frequency content -> Exponential for edge enhancement
    elif freq_ratio < global_freq_ratio_median * 0.8:
        filter_config['fft_filter']['type'] = 'exponential'
        filter_config['fft_filter']['decay_rate'] = 0.04  # More aggressive enhancement
    
    # Moderate processing -> Blackman
    elif freq_ratio > global_freq_ratio_median * 1.2:
        filter_config['fft_filter']['type'] = 'blackman'
        filter_config['fft_filter']['sigma'] = 12.0
    
    else:
        # Default mild processing
        filter_config['fft_filter']['type'] = 'gaussian'
        filter_config['fft_filter']['sigma'] = 14.0
    
    # Configure Unsharp Mask
    if edge_density < global_percentiles['edge_density']['low']:
        # Low edge density - more aggressive unsharp
        filter_config['unsharp_mask']['radius'] = 6.0
        filter_config['unsharp_mask']['amount'] = 25.0
    elif laplacian_variance < global_percentiles['laplacian_variance']['low']:
        # Low sharpness - moderate unsharp
        filter_config['unsharp_mask']['radius'] = 5.0
        filter_config['unsharp_mask']['amount'] = 20.0
    else:
        # Default mild unsharp
        filter_config['unsharp_mask']['radius'] = 4.0
        filter_config['unsharp_mask']['amount'] = 15.0
    
    # Configure Cosine
    if contrast < global_percentiles['contrast']['low']:
        filter_config['cosine'] = True
        # Adjust blur sigma based on image characteristics
        if homogeneity > global_percentiles['homogeneity']['high']:
            filter_config['blur_sigma'] = 2.0  # More smoothing for uniform images
        else:
            filter_config['blur_sigma'] = 1.0
    
    # Configure Invert
    # Only recommend invert for very specific cases
    if (contrast < global_percentiles['contrast']['low'] * 0.7 and 
        edge_density < global_percentiles['edge_density']['low'] * 0.8):
        filter_config['invert'] = True
    
    return filter_config

def generate_all_cluster_configs(cluster_stats, global_percentiles):
    """
    Generate filter configurations for all clusters
    """
    cluster_configs = {}
    
    for cluster_id in cluster_stats.keys():
        config = create_cluster_filter_configs(cluster_id, cluster_stats, global_percentiles)
        cluster_configs[f'cluster_{cluster_id}_filters'] = config
    
    return cluster_configs

def run_cluster_config_analysis(feature_frame,  clusters):

    # Analyze by clusters
    df, cluster_stats, percentiles, unique_clusters = analyze_clusters(feature_frame, clusters)
    
    # Generate filter configurations
    cluster_configs = generate_all_cluster_configs(cluster_stats, percentiles)
    
    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id]
        config_name = f'cluster_{cluster_id}_filters'
        config = cluster_configs[config_name]
        
        # Reasoning
        print(f"Reasoning:")
        fft_type = config['fft_filter']['type']
        if fft_type == 'gaussian':
            print(f"   FFT Gaussian: Noise reduction/smoothing")
        elif fft_type == 'exponential':
            print(f"   FFT Exponential: Edge enhancement")
        elif fft_type == 'blackman':
            print(f"   FFT Blackman: Balanced filtering")
        
        if config['unsharp_mask']['amount'] > 20:
            print(f"   Strong Unsharp: Low edge content detected")
        elif config['unsharp_mask']['amount'] > 15:
            print(f"   Moderate Unsharp: Standard enhancement")
        
        if config['cosine']:
            print(f"   Cosine: Low contrast enhancement")
        
        if config['invert']:
            print(f"   Invert: Very low contrast/edge content")
    
    return cluster_configs, df, cluster_stats

def print_all_configs(cluster_configs):
    """
    Print all configurations in a copy-paste ready format
    """
    print("\n" + "="*60)
    print(" COPY-PASTE READY CONFIGURATIONS")
    print("="*60)
    
    for config_name, config in cluster_configs.items():
        print(f"\n{config_name} = {{")
        print(f"    'fft_filter': {config['fft_filter']},")
        print(f"    'unsharp_mask': {config['unsharp_mask']},")
        print(f"    'invert': {config['invert']},")
        print(f"    'cosine': {config['cosine']},")
        print(f"    'blur_sigma': {config['blur_sigma']}")
        print("}")

def run_cluster_filter_analysis(additional_features, names, clusters):
    """
    Complete cluster-based analysis and recommendation system
    """
    # Analyze by clusters
    df, cluster_stats, percentiles, unique_clusters = analyze_clusters(feature_frame, clusters)
    
    print(" CLUSTER-BASED FILTER RECOMMENDATIONS")
    print("="*60)
    
    cluster_strategies = {}
    
    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id]
        recommendations = recommend_cluster_filters(cluster_id, cluster_stats, percentiles)
        
        # Sort by priority
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        recommendations.sort(key=lambda x: priority_order[x['priority']], reverse=True)
        
        cluster_strategies[cluster_id] = recommendations
        
        # Print cluster analysis
        print(f"\n CLUSTER {cluster_id} ({cluster_stats[cluster_id]['bias']['count']} images)")
        print("=" * 40)
        
        # Key statistics
        stats = cluster_stats[cluster_id]
        print(f" Key Characteristics:")
        print(f"   Edge Density: {stats['edge_density']['mean']:.3f} ± {stats['edge_density']['std']:.3f}")
        print(f"   Contrast: {stats['contrast']['mean']:.3f} ± {stats['contrast']['std']:.3f}")
        print(f"   Sharpness: {stats['laplacian_variance']['mean']:.1f} ± {stats['laplacian_variance']['std']:.1f}")
        print(f"   Objects: {stats['num_contours']['mean']:.1f} ± {stats['num_contours']['std']:.1f}")
        print(f"   Homogeneity: {stats['homogeneity']['mean']:.3f} ± {stats['homogeneity']['std']:.3f}")
        
        # Filter recommendations
        print(f"\n🎯 Recommended Filters:")
        for i, rec in enumerate(recommendations[:3]):
            priority_emoji = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
            print(f"   {priority_emoji} {rec['filter']}")
            print(f"      Reason: {rec['reason']}")
            print(f"      Params: {rec['params']}")
        
        # Sample images from cluster
        sample_images = cluster_data.index[:3].tolist()
        print(f"\n Sample Images: {', '.join(sample_images)}")
    
    return df, cluster_strategies, cluster_stats, percentiles

def apply_recommended_filters(image_name, strategy, image_data):
    """
    Apply the recommended filters to an image
    """
    pipeline = get_processing_pipeline(image_name, strategy)
    
    print(f"\n Processing pipeline for {image_name}:")
    for step in pipeline:
        print(f"  → {step['filter']}: {step['reason']}")
    
    # Here you would implement the actual filter applications
    # processed_image = apply_gaussian(image_data) # etc.
    
    return pipeline

def get_cluster_processing_pipeline(cluster_id, cluster_strategies):
    """
    Get optimal processing pipeline for a specific cluster
    """
    recommendations = cluster_strategies[cluster_id]
    high_priority = [r for r in recommendations if r['priority'] == 'High']
    
    if len(high_priority) == 0:
        medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
        return medium_priority[:2] if medium_priority else recommendations[:1]
    
    return high_priority

def calculate_circle_radii_in_pixels(pixelsize):
    """
    Convert physical radii to pixel units
    
    Parameters:
    -----------
    pixelsize : array-like
        Pixel size in nm/pixel [x, y]
    
    Returns:
    --------
    min_radius_px, max_radius_px : int
        Minimum and maximum radii in pixels
    """
    # Physical dimensions in nm
    min_radius_physical = 0.12  # 1.2 Angstroms = 0.12 nm
    max_radius_physical = 0.2   # 2.0 Angstroms = 0.2 nm
    
    # Use average pixel size (they're nearly identical anyway)
    avg_pixelsize = np.mean(pixelsize)  # nm/pixel
    
    # Convert to pixels
    min_radius_px = int(np.round(min_radius_physical / avg_pixelsize))
    max_radius_px = int(np.round(max_radius_physical / avg_pixelsize))
    
    # print(f"Physical radii: {min_radius_physical:.2f} - {max_radius_physical:.2f} nm")
    # print(f"Pixel size: {avg_pixelsize:.6f} nm/pixel")
    # print(f"Radii in pixels: {min_radius_px} - {max_radius_px} pixels")
    
    return min_radius_px, max_radius_px

def remove_background_outside_edges(img, gauss=False, sensitivity='medium'):
    """
    Remove background outside the main image area using Otsu thresholding
   
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (can be grayscale or color)
    gauss : bool
        Whether to apply Gaussian smoothing to the mask edges
    sensitivity : str
        'low', 'medium', 'high' - controls how aggressive the background removal is
   
    Returns:
    --------
    result : dict
        Dictionary containing processed images and mask
    """
   
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = img.copy()
        is_color = False
   
    # CRITICAL: Convert to uint8 for OpenCV operations
    if gray.dtype != np.uint8:
        # Normalize to 0-255 range and convert to uint8
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
   
    # Get Otsu threshold but modify it based on sensitivity
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adjust threshold based on sensitivity
    if sensitivity == 'low':
        # Less aggressive - lower threshold to keep more image area
        adjusted_thresh = otsu_thresh * 0.7
        print(f"Low sensitivity: Using {adjusted_thresh:.1f} instead of Otsu {otsu_thresh:.1f}")
    elif sensitivity == 'medium':
        # Moderate - slightly lower than Otsu
        adjusted_thresh = otsu_thresh * 0.85
        print(f"Medium sensitivity: Using {adjusted_thresh:.1f} instead of Otsu {otsu_thresh:.1f}")
    else:  # high
        # Use original Otsu threshold
        adjusted_thresh = otsu_thresh
        print(f"High sensitivity: Using Otsu threshold {otsu_thresh:.1f}")
    
    # Apply the adjusted threshold
    _, mask = cv2.threshold(gray, adjusted_thresh, 255, cv2.THRESH_BINARY)
   
    # Clean up the mask with morphological operations - make less aggressive for low sensitivity
    if sensitivity == 'low':
        kernel = np.ones((3,3), np.uint8)  # Smaller kernel
    else:
        kernel = np.ones((5,5), np.uint8)  # Standard kernel
   
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   
    # Find the largest contour (main image area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
       
        # Create a clean mask from the largest contour
        clean_mask = np.zeros_like(mask)
        cv2.fillPoly(clean_mask, [largest_contour], 255)
       
        # Optional: smooth the edges
        if gauss:
            clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)
            _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)
       
    else:
        clean_mask = mask
   
    # Apply the mask to the original image
    if img.dtype != np.uint8:
        img_uint8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_uint8 = img.copy()
   
    if is_color:
        result_img = img_uint8.copy()
        result_img[clean_mask == 0] = [0, 0, 0]
        result_img_white = img_uint8.copy()
        result_img_white[clean_mask == 0] = [255, 255, 255]
    else:
        result_img = img_uint8.copy()
        result_img[clean_mask == 0] = 0
        result_img_white = img_uint8.copy()
        result_img_white[clean_mask == 0] = 255
   
    # Return results
    results = {
        'mask': clean_mask,
        'result_black_bg': result_img,
        'result_white_bg': result_img_white,
        'contour': largest_contour if contours else None,
        'threshold_used': adjusted_thresh,
        'otsu_threshold': otsu_thresh
    }
   
    return results

def get_distance_transform(img, mask, name):
    """
    Simple distance transform using existing mask
    
    Parameters:
    -----------
    img : numpy.ndarray
        Original image (for reference)
    mask : numpy.ndarray
        Binary mask (255 = object, 0 = background)
    name : str
        Image name for display
    
    Returns:
    --------
    numpy.ndarray : Distance transform array
    """
   
    # Apply distance transform directly to your existing mask
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    return dist_transform

def extract_backbone_from_distance_transform(img, distance_transform, name, method='ridge'):
    """
    Extract backbone directly from distance transform
    
    Parameters:
    -----------
    img : numpy.ndarray
        Original image (for visualization)
    distance_transform : numpy.ndarray
        The distance transform array we calculated
    name : str
        Image name
    method : str
        'ridge' or 'peaks'
    
    Returns:
    --------
    dict : Contains backbone coordinates and visualization
    """
    
    # Convert image to uint8 for visualization
    if img.dtype != np.uint8:
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img.copy()
    
    backbone_points = []
    
    if method == 'ridge':
        # Method 1: Follow the ridge (highest values) of distance transform
        # Find pixels that are local maxima along ridges
        
        # Smooth slightly to avoid noise
        dist_smooth = cv2.GaussianBlur(distance_transform, (3, 3), 0)
        
        # Find ridge points using morphological operations
        kernel_h = np.ones((1, 7))  # Horizontal kernel
        kernel_v = np.ones((7, 1))  # Vertical kernel
        
        # Find local maxima in horizontal and vertical directions
        max_h = cv2.dilate(dist_smooth, kernel_h)
        max_v = cv2.dilate(dist_smooth, kernel_v)
        
        # Ridge points are local maxima in at least one direction
        ridge_mask = ((dist_smooth == max_h) | (dist_smooth == max_v)) & (dist_smooth > 0.3 * distance_transform.max())
        
        # Get ridge coordinates
        ridge_coords = np.where(ridge_mask)
        backbone_points = list(zip(ridge_coords[1], ridge_coords[0]))  # (x, y) format
        
        print(f"   Ridge method: {len(backbone_points)} backbone points")
    
    else:  # peaks method
        # Method 2: Connect the peaks (local maxima) of distance transform
        
        # Find local maxima
        kernel = np.ones((15, 15))
        local_max = cv2.dilate(distance_transform, kernel)
        peaks_mask = (distance_transform == local_max) & (distance_transform > 0.4 * distance_transform.max())
        
        # Get peak coordinates
        peak_coords = np.where(peaks_mask)
        peaks = list(zip(peak_coords[1], peak_coords[0]))  # (x, y) format
        
        # Sort peaks to create a connected backbone
        if len(peaks) > 1:
            # Sort by x-coordinate (left to right)
            peaks_sorted = sorted(peaks, key=lambda p: p[0])
            backbone_points = peaks_sorted
        else:
            backbone_points = peaks
        
        print(f"   Peaks method: {len(backbone_points)} backbone points")
    
    # Create visualization
    result_img = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
    
    # Draw the distance transform as background (semi-transparent)
    dist_colored = cv2.applyColorMap((distance_transform / distance_transform.max() * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    result_img = cv2.addWeighted(result_img, 0.7, dist_colored, 0.3, 0)
    
    # Draw backbone
    if len(backbone_points) > 1:
        if method == 'ridge':
            # For ridge: draw all points
            for x, y in backbone_points:
                if 0 <= x < result_img.shape[1] and 0 <= y < result_img.shape[0]:
                    cv2.circle(result_img, (x, y), 2, (0, 255, 0), -1)  # Green dots
        else:
            # For peaks: draw connected lines
            for i in range(len(backbone_points) - 1):
                pt1 = backbone_points[i]
                pt2 = backbone_points[i + 1]
                cv2.line(result_img, pt1, pt2, (0, 255, 0), 3)  # Green line
            
            # Mark individual peaks
            for x, y in backbone_points:
                cv2.circle(result_img, (x, y), 5, (255, 255, 0), 2)  # Yellow circles
        
        # Mark start and end
        if len(backbone_points) > 0:
            start_point = backbone_points[0]
            end_point = backbone_points[-1]
            cv2.circle(result_img, start_point, 8, (255, 0, 0), -1)  # Blue start
            cv2.circle(result_img, end_point, 8, (0, 0, 255), -1)    # Red end
    
    # Calculate backbone length
    backbone_length = 0
    if len(backbone_points) > 1:
        for i in range(1, len(backbone_points)):
            x1, y1 = backbone_points[i-1]
            x2, y2 = backbone_points[i]
            backbone_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    results = {
        'name': name,
        'method': method,
        'backbone_points': backbone_points,
        'backbone_length': backbone_length,
        'num_points': len(backbone_points),
        'visualization': result_img,
        'distance_transform_used': distance_transform
    }
    
    return results

def analyze_backbone_measurements(backbone_points, name, pixelsize):
    """
    Calculate different backbone measurements in both pixels and nanometers
    
    Parameters:
    -----------
    backbone_points : list
        List of (x, y) coordinates
    name : str
        Image name
    pixelsize : array-like
        [x_nm_per_pixel, y_nm_per_pixel]
    """
    if len(backbone_points) < 2:
        return {
            'name': name,
            'backbone_length_px': 0,
            'backbone_length_nm': 0,
            'end_to_end_distance_px': 0,
            'end_to_end_distance_nm': 0,
            'contour_ratio': 0,
            'bounding_diagonal_px': 0,
            'bounding_diagonal_nm': 0,
            'num_points': len(backbone_points),
            'start_point': None,
            'end_point': None,
            'pixelsize': pixelsize
        }
    
    # Average pixel size for distance calculations
    avg_pixelsize = np.mean(pixelsize)  # nm per pixel
    
    # 1. Backbone Length (total path length)
    backbone_length_px = 0
    for i in range(1, len(backbone_points)):
        x1, y1 = backbone_points[i-1]
        x2, y2 = backbone_points[i]
        segment_length_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        backbone_length_px += segment_length_px
    
    backbone_length_nm = backbone_length_px * avg_pixelsize
    
    # 2. End-to-End Distance (straight line)
    start_point = backbone_points[0]
    end_point = backbone_points[-1] 
    end_to_end_distance_px = np.sqrt((end_point[0] - start_point[0])**2 + 
                                    (end_point[1] - start_point[1])**2)
    end_to_end_distance_nm = end_to_end_distance_px * avg_pixelsize
    
    # 3. Contour Length Ratio (dimensionless)
    contour_ratio = backbone_length_px / end_to_end_distance_px if end_to_end_distance_px > 0 else 0
    
    # 4. Bounding Box Diagonal
    points_array = np.array(backbone_points)
    min_x, min_y = points_array.min(axis=0)
    max_x, max_y = points_array.max(axis=0)
    bounding_diagonal_px = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    bounding_diagonal_nm = bounding_diagonal_px * avg_pixelsize
    
    results = {
        'name': name,
        'backbone_length_px': backbone_length_px,           # Total path length in pixels
        'backbone_length_nm': backbone_length_nm,           # Total path length in nm
        'end_to_end_distance_px': end_to_end_distance_px,   # Straight line distance in pixels
        'end_to_end_distance_nm': end_to_end_distance_nm,   # Straight line distance in nm
        'contour_ratio': contour_ratio,                     # How "curvy" it is (dimensionless)
        'bounding_diagonal_px': bounding_diagonal_px,       # Bounding box in pixels
        'bounding_diagonal_nm': bounding_diagonal_nm,       # Bounding box in nm
        'num_points': len(backbone_points),
        'start_point': start_point,
        'end_point': end_point,
        'pixelsize': pixelsize,
        'avg_pixelsize_nm_per_px': avg_pixelsize
    }
    
    return results


