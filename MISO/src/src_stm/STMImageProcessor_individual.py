import numpy as np
from .automated_analysis import extract_features_to_dataframe, run_cluster_config_analysis, remove_background_outside_edges, get_distance_transform
from .filters import process_sxm_image
from .utils import *

class STMImageProcessor:
    def __init__(self):
        self.target_size = (128, 128)
        self.n_clusters = 1  # You're using 1 cluster
    
    def process_all_images(self, sxm_data):
        """Process all images through the complete pipeline"""
        
        # 1. Extract features
        feature_frame = self._extract_features(sxm_data)
        
        # 2. Process with cluster configs
        processed_data = self._process_with_cluster_configs(sxm_data, feature_frame)
        
        # 3. Remove backgrounds
        processed_data = self._remove_backgrounds(processed_data)
        
        # 4. Calculate distance transforms
        processed_data = self._calculate_distance_transforms(processed_data)
        
        return processed_data
    
    def _extract_features(self, sxm_data):
        """Extract numerical features for clustering"""
        return extract_features_to_dataframe(
            sxm_data, 
            target_size=self.target_size, 
            include_image_pixels=False
        )
    
    def _process_with_cluster_configs(self, sxm_data, feature_frame):
        """Apply cluster-specific processing"""
        # Your existing cluster processing logic
        cluster_configs, df, cluster_stats = run_cluster_config_analysis(
            feature_frame, [0] * len(feature_frame)
        )
        
        cluster_processed = {}
        sample_name = list(sxm_data.keys())[0]  # Single sample for now
        
        cluster_config = cluster_configs[f'cluster_0_filters']
        image = sxm_data[sample_name]['img']
        zmask = sxm_data[sample_name]['Zmask']
        
        try:
            result = process_sxm_image(image, zmask, cluster_config)
            processed_image = result['processed_image']
            
            try:
                image_entropy = calculate_entropy(processed_image)
            except:
                image_entropy = sxm_data[sample_name].get('entropy', 0)
            
            cluster_processed[sample_name] = {
                'original_img': image,
                'processed_img': processed_image,
                'Zmask': zmask,
                'Pixelsize': sxm_data[sample_name]['Pixelsize'],
                'entropy': image_entropy,
                'header': sxm_data[sample_name]['header'],
                'filters_applied': cluster_config.copy(),
                'processing_result': result
            }
            
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            # Fallback logic here
            
        return cluster_processed
    
    def _remove_backgrounds(self, processed_data):
        """Remove backgrounds using Otsu thresholding"""
        otsus = {}
        for name, data in processed_data.items():
            img = data['processed_img']
            otsus[name] = remove_background_outside_edges(img, gauss=False, sensitivity='low')
        
        deep_update(processed_data, otsus)
        return processed_data
    
    def _calculate_distance_transforms(self, processed_data):
        """Calculate distance transforms for shape analysis"""
        for name, data in processed_data.items():
            img = data['result_black_bg']
            mask = data['mask']
            distance_array = get_distance_transform(img, mask, name)
            
            data.update({
                'distance_transform': distance_array,
                'max_distance': distance_array.max(),
                'mean_distance': distance_array[distance_array > 0].mean() if distance_array[distance_array > 0].size > 0 else 0,
                'image_shape': img.shape,
                'non_zero_pixels': np.sum(distance_array > 0)
            })
        
        return processed_data