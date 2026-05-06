import os
from PIL import Image, ImageFilter
import numpy as np
import cv2 as cv
import sklearn
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

def analyze_circle_clusters(circles, eps=50, min_samples=2):
    """
    Analyze clusters of circle centers
    
    Args:
        circles: Output from cv.HoughCircles
        eps: Maximum distance between two samples for clustering (pixels)
        min_samples: Minimum number of samples in a cluster
    
    Returns:
        cluster_info: Dictionary with cluster analysis
    """
    
    if circles is None:
        print("No circles to analyze")
        return None
    
    # Extract circle centers
    centers = circles[0, :, :2]  # x, y coordinates only
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # Number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')
    
    # Analyze each cluster
    cluster_info = {}
    
    for cluster_id in set(labels):
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_name = f'cluster_{cluster_id}'
            
        # Get points in this cluster
        cluster_mask = labels == cluster_id
        cluster_centers = centers[cluster_mask]
        cluster_circles = circles[0, cluster_mask, :]
        
        # Calculate cluster statistics
        cluster_size = len(cluster_centers)
        centroid = np.mean(cluster_centers, axis=0)
        
        # Calculate density (circles per unit area)
        if cluster_size > 1:
            # Find bounding box of cluster
            min_x, min_y = np.min(cluster_centers, axis=0)
            max_x, max_y = np.max(cluster_centers, axis=0)
            area = (max_x - min_x) * (max_y - min_y)
            density = cluster_size / max(area, 1)  # Avoid division by zero
        else:
            density = 0
        
        cluster_info[cluster_name] = {
            'size': cluster_size,
            'centroid': centroid,
            'density': density,
            'centers': cluster_centers,
            'circles': cluster_circles,
            'bbox': (min_x, min_y, max_x, max_y) if cluster_size > 1 else None
        }
        
        print(f'Cluster {cluster_name}: {cluster_size} circles, '
              f'centroid at ({centroid[0]:.1f}, {centroid[1]:.1f}), '
              f'density: {density:.4f}')
    
    return cluster_info, labels

def visualize_clusters_clean(img, circles, labels, cluster_info):
    """
    Clean visualization with ordered cluster numbering
    """
    # Create color image for visualization
    if len(img.shape) == 3:
        vis_img = img.copy()
    else:
        vis_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    # Define colors for different clusters
    colors = [
        (255, 100, 100),  # Light Red
        (100, 255, 100),  # Light Green
        (100, 100, 255),  # Light Blue
        (255, 255, 100),  # Light Cyan
        (255, 100, 255),  # Light Magenta
        (100, 255, 255),  # Light Yellow
        (200, 100, 200),  # Light Purple
        (255, 200, 100),  # Light Orange
        (100, 200, 100),  # Light Dark Green
        (200, 200, 100),  # Light Olive
    ]
    
    # Create ordered mapping of clusters by size
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['size'], reverse=True)
    cluster_id_to_ordered = {}
    for ordered_num, (original_cluster_id, info) in enumerate(sorted_clusters, 1):
        cluster_id_to_ordered[original_cluster_id] = ordered_num
    
    centers = circles[0, :, :2]
    
    # Draw circles using ordered color scheme
    for i, (center, label) in enumerate(zip(centers, labels)):
        x, y = int(center[0]), int(center[1])
        radius = int(circles[0, i, 2])
        
        if label == -1:  # Noise points
            color = (128, 128, 128)  # Gray
        else:
            # Use ordered number for consistent colors
            ordered_num = cluster_id_to_ordered.get(label, label)
            color = colors[(ordered_num-1) % len(colors)]
        
        # Draw circle
        cv.circle(vis_img, (x, y), radius, color, 2)
        # Draw center point
        cv.circle(vis_img, (x, y), 4, color, -1)
    
    # Draw cluster centroids using ordered colors
    for original_cluster_id, info in cluster_info.items():
        centroid = info['centroid']
        ordered_num = cluster_id_to_ordered.get(original_cluster_id, original_cluster_id)
        color = colors[(ordered_num-1) % len(colors)]
        
        # Draw large cross for centroid
        cx, cy = int(centroid[0]), int(centroid[1])
        cv.line(vis_img, (cx-15, cy), (cx+15, cy), color, 4)
        cv.line(vis_img, (cx, cy-15), (cx, cy+15), color, 4)
    
    return vis_img

def create_cluster_legend(cluster_info, labels, circles, img_height=600, img_width=400):
    """
    Create a separate legend image showing cluster information with ordered numbering
    """
    # Create white background for legend
    legend_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Define colors (same as visualization)
    colors = [
        (255, 100, 100),  # Light Red
        (100, 255, 100),  # Light Green
        (100, 100, 255),  # Light Blue
        (255, 255, 100),  # Light Cyan
        (255, 100, 255),  # Light Magenta
        (100, 255, 255),  # Light Yellow
        (200, 100, 200),  # Light Purple
        (255, 200, 100),  # Light Orange
        (100, 200, 100),  # Light Dark Green
        (200, 200, 100),  # Light Olive
    ]
    
    # Title
    cv.putText(legend_img, "CLUSTER ANALYSIS", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv.line(legend_img, (10, 40), (img_width-10, 40), (0, 0, 0), 2)
    
    y_pos = 70
    line_height = 50
    
    # Sort clusters by size for better readability and create ordered numbering
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['size'], reverse=True)
    
    # Create mapping from original cluster_id to ordered number (1, 2, 3, ...)
    cluster_id_to_ordered = {}
    for ordered_num, (original_cluster_id, info) in enumerate(sorted_clusters, 1):
        cluster_id_to_ordered[original_cluster_id] = ordered_num
    
    for ordered_num, (original_cluster_id, info) in enumerate(sorted_clusters, 1):
        color = colors[(ordered_num-1) % len(colors)]  # Use ordered number for consistent colors
        
        # Draw color box
        cv.rectangle(legend_img, (15, y_pos-15), (45, y_pos+15), color, -1)
        cv.rectangle(legend_img, (15, y_pos-15), (45, y_pos+15), (0, 0, 0), 2)
        
        # Add cluster information text with ordered numbering
        text = f"Cluster {ordered_num}"
        cv.putText(legend_img, text, (55, y_pos-5), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Size info
        size_text = f"Size: {info['size']} circles"
        cv.putText(legend_img, size_text, (55, y_pos+15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Density info
        density_text = f"Density: {info['density']:.4f}"
        cv.putText(legend_img, density_text, (220, y_pos+15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        y_pos += line_height
        
        # Check if we're running out of space
        if y_pos > img_height - 50:
            cv.putText(legend_img, "...", (55, y_pos), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            break
    
    # Store the mapping for use in visualization
    global cluster_id_mapping
    cluster_id_mapping = cluster_id_to_ordered
    
    # Add noise information if present
    n_noise = list(labels).count(-1)
    if n_noise > 0:
        y_pos += 20
        cv.rectangle(legend_img, (15, y_pos-15), (45, y_pos+15), (128, 128, 128), -1)
        cv.rectangle(legend_img, (15, y_pos-15), (45, y_pos+15), (0, 0, 0), 2)
        
        cv.putText(legend_img, "Noise Points", (55, y_pos-5), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv.putText(legend_img, f"Count: {n_noise}", (55, y_pos+15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add summary statistics
    y_pos += 60
    cv.line(legend_img, (10, y_pos), (img_width-10, y_pos), (0, 0, 0), 1)
    y_pos += 30
    
    total_circles = len(circles[0]) if circles is not None else 0
    cv.putText(legend_img, "SUMMARY", (10, y_pos), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += 25
    cv.putText(legend_img, f"Total Circles: {total_circles}", (15, y_pos), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += 20
    cv.putText(legend_img, f"Clusters: {len(cluster_info)}", (15, y_pos), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += 20
    cv.putText(legend_img, f"Noise Points: {n_noise}", (15, y_pos), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return legend_img

def display_clusters_with_legend(img, circles, cluster_info, labels):
    """
    Display clusters with clean visualization and separate legend
    """
    # Create clean cluster visualization
    cluster_vis = visualize_clusters_clean(img, circles, labels, cluster_info)
    
    # Create legend
    legend = create_cluster_legend(cluster_info, labels,circles)
    
    # Resize images to same height for side-by-side display
    target_height = 600
    
    # Resize cluster visualization
    h, w = cluster_vis.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    cluster_resized = cv.resize(cluster_vis, (new_width, target_height))
    
    # Resize legend to same height
    legend_resized = cv.resize(legend, (400, target_height))
    
    # Combine images side by side
    combined = np.hstack([cluster_resized, legend_resized])
    
    return combined, cluster_resized, legend_resized

def visualize_cluster_centroids(img, cluster_info, radius, color=(0, 255, 255), thickness=3):
    """
    Visualize cluster centroids as single circles
    
    Args:
        img: Original image or threshold image
        cluster_info: Dictionary from analyze_circle_clusters function
        radius: Radius for centroid circles (default: 30)
        color: Color for centroid circles (default: yellow (0, 255, 255))
        thickness: Line thickness for circles (default: 3)
    
    Returns:
        img_with_centroids: Image with centroid circles drawn
    """
    
    # Create a copy for visualization
    img_cv = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img_centroids = img_cv.copy()
    if len(img_centroids.shape) == 2:  # If grayscale
        img_centroids = cv.cvtColor(img_centroids, cv.COLOR_GRAY2BGR)
    
    if cluster_info is None:
        print("No cluster info provided")
        return img_centroids
    
    print(f"Drawing centroids for {len(cluster_info)} clusters")
    
    # Draw a circle at each cluster centroid
    for cluster_name, cluster_data in cluster_info.items():
        centroid = cluster_data['centroid']
        cx, cy = int(centroid[0]), int(centroid[1])
        cluster_size = cluster_data['size']
        
        # Draw the centroid circle
        cv.circle(img_centroids, (cx, cy), radius, color, thickness)
        
        # Add a small center dot
        cv.circle(img_centroids, (cx, cy), 3, (0, 0, 255), -1)  # Red filled dot
        
        # Add cluster label
        cv.putText(img_centroids, f"{cluster_name} ({cluster_size})", 
                   (cx + radius + 5, cy), cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        print(f"{cluster_name}: centroid at ({cx}, {cy}), {cluster_size} circles")
    
    return img_centroids

def show_centroid_visualization(img, cluster_info,radius=150):
    """
    Complete function to create and display centroid visualization
    """
    # Create the visualization
    img_with_centroids = visualize_cluster_centroids(img, cluster_info,radius)

    # Show the result with matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(cv.cvtColor(img_with_centroids, cv.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title('Cluster Centroids')
    plt.axis('off')
    plt.show()
    
    # Also show with matplotlib for better quality
    # plt.figure(figsize=(10, 10))
    # img_rgb = cv.cvtColor(img_with_centroids, cv.COLOR_BGR2RGB)
    # plt.imshow(img_rgb)
    # plt.title('Cluster Centroids - One Circle per Cluster')
    # plt.axis('off')
    # plt.show()
    
    return img_with_centroids

def get_chain_line_array(cluster_info, segment_length=50):
    """
    Create and return the chain line as an array of points
    
    Args:
        cluster_info: Dictionary from analyze_circle_clusters function
        segment_length: Length of each chain segment (pixels)
    
    Returns:
        chain_line_array: Array of all points forming the chain line
    """
    
    if cluster_info is None or len(cluster_info) < 2:
        print("Need at least 2 clusters to create a chain")
        return None
    
    # Extract centroids
    centroids = []
    cluster_names = []
    for cluster_name, cluster_data in cluster_info.items():
        centroids.append(cluster_data['centroid'])
        cluster_names.append(cluster_name)
    
    centroids = np.array(centroids)
    print(f"Creating chain line from {len(centroids)} centroids")
    
    # Find optimal chain path (same as in connect_centroids_chain)
    chain_path = find_chain_path(centroids)
    
    # Create the complete chain line as array of points
    chain_line_points = []
    
    # Add first centroid
    chain_line_points.append(centroids[chain_path[0]])
    
    # Create chain segments between consecutive centroids
    for i in range(len(chain_path) - 1):
        start_idx = chain_path[i]
        end_idx = chain_path[i + 1]
        
        start_point = centroids[start_idx]
        end_point = centroids[end_idx]
        
        # Create chain segments between these two points
        chain_segments = create_chain_segments(start_point, end_point, segment_length)
        
        # Add all intermediate points (skip first to avoid duplication)
        for point in chain_segments[1:]:
            chain_line_points.append(point)
    
    # Convert to numpy array
    chain_line_array = np.array(chain_line_points)
    
    print(f"Chain line saved with {len(chain_line_array)} points")
    return chain_line_array

def visualize_chain_connection_with_array(img, cluster_info, segment_length=50):
    """
    Complete function to create chain connection and return both image and line array
    """
    # Create the chain visualization
    img_with_chain = connect_centroids_chain(img, cluster_info, segment_length)
    
    # Get the chain line as array
    chain_line_array = get_chain_line_array(cluster_info, segment_length)
    
    # Show with matplotlib
    plt.figure(figsize=(12, 12))
    img_rgb = cv.cvtColor(img_with_chain, cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Centroids Connected by Freely Jointed Chain')
    plt.axis('off')
    plt.show()
    
    return img_with_chain, chain_line_array

def filter_clusters_by_chain_distance(mini_cluster_info, chain_line_array, max_distance):
    """
    Check which clusters are over distance x from the chain line
    
    Args:
        mini_cluster_info: Dictionary with cluster info (e.g., mini_cluster_info['cluster_0']['centroid'])
        chain_line_array: Array of points forming the chain line
        max_distance: Maximum allowed distance from chain line
    
    Returns:
        valid_clusters: Dictionary of clusters within distance
        invalid_clusters: Dictionary of clusters over distance
        distance_results: Dictionary with distance info for each cluster
    """
    
    if chain_line_array is None or len(chain_line_array) < 2:
        print("Invalid chain line array")
        return mini_cluster_info, {}, {}
    
    valid_clusters = {}
    invalid_clusters = {}
    distance_results = {}
    
    print(f"Checking {len(mini_cluster_info)} clusters against chain line (max distance: {max_distance})")
    print("-" * 60)
    
    for cluster_name, cluster_data in mini_cluster_info.items():
        centroid = cluster_data['centroid']
        
        # Calculate distance to chain line
        distance = distance_point_to_chain_line(centroid, chain_line_array)
        
        # Store distance info
        distance_results[cluster_name] = {
            'distance': distance,
            'centroid': centroid,
            'valid': distance <= max_distance
        }
        
        # Categorize cluster
        if distance <= max_distance:
            valid_clusters[cluster_name] = cluster_data
            print(f"✓ {cluster_name}: distance {distance:.1f} <= {max_distance} (VALID)")
        else:
            invalid_clusters[cluster_name] = cluster_data
            print(f"✗ {cluster_name}: distance {distance:.1f} > {max_distance} (INVALID - TOO FAR)")
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Valid clusters (≤{max_distance}px): {len(valid_clusters)}")
    print(f"  Invalid clusters (>{max_distance}px): {len(invalid_clusters)}")
    
    return valid_clusters, invalid_clusters, distance_results

def show_cluster_distances(mini_cluster_info, chain_line_array, max_distance):
    """
    Simple function to just show which clusters are over distance x
    """
    
    print(f"Clusters over {max_distance}px from chain line:")
    print("=" * 50)
    
    over_distance_clusters = []
    
    for cluster_name, cluster_data in mini_cluster_info.items():
        centroid = cluster_data['centroid']
        distance = distance_point_to_chain_line(centroid, chain_line_array)
        
        if distance > max_distance:
            over_distance_clusters.append(cluster_name)
            print(f"{cluster_name}: distance = {distance:.1f}px (OVER LIMIT)")
    
    if not over_distance_clusters:
        print("No clusters are over the distance limit!")
    else:
        print(f"\nTotal clusters over {max_distance}px: {len(over_distance_clusters)}")
        print(f"Cluster names: {over_distance_clusters}")
    
    return over_distance_clusters

def visualize_dual_cluster_centroids(img, cluster_info1, cluster_info2, radius1=150, radius2=75, 
                                    color1=(0, 255, 255), color2=(255, 0, 255), thickness=3):
    """
    Visualize centroids from two different dictionaries with different radii
    
    Args:
        img: Original image
        cluster_info1: First dictionary with centroids
        cluster_info2: Second dictionary with centroids
        radius1: Radius for first dictionary circles (default: 150)
        radius2: Radius for second dictionary circles (default: 75)
        color1: Color for first dictionary circles (default: yellow (0, 255, 255))
        color2: Color for second dictionary circles (default: magenta (255, 0, 255))
        thickness: Line thickness for circles (default: 3)
    
    Returns:
        img_with_centroids: Image with both sets of centroids drawn
    """
    
    # Create a copy for visualization
    img_cv = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img_centroids = img_cv.copy()
    if len(img_centroids.shape) == 2:  # If grayscale
        img_centroids = cv.cvtColor(img_centroids, cv.COLOR_GRAY2BGR)
    
    # Draw first dictionary centroids
    if cluster_info1 is not None:
        
        for cluster_name, cluster_data in cluster_info1.items():
            centroid = cluster_data['centroid']
            cx, cy = int(centroid[0]), int(centroid[1])
            cluster_size = cluster_data['size']
            
            # Draw the centroid circle
            cv.circle(img_centroids, (cx, cy), radius1, color1, thickness)
            
            # Add a small center dot
            cv.circle(img_centroids, (cx, cy), 3, (0, 0, 255), -1)  # Red filled dot
            
            # Add cluster label
            cv.putText(img_centroids, f"{cluster_name} ",
                       (cx + radius1 + 5, cy), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
            

    
    # Draw second dictionary centroids
    if cluster_info2 is not None:
        print(f"Drawing centroids for {len(cluster_info2)} clusters from second dictionary (radius: {radius2})")
        
        for cluster_name, cluster_data in cluster_info2.items():
            centroid = cluster_data['centroid']
            cx, cy = int(centroid[0]), int(centroid[1])
            cluster_size = cluster_data['size']
            
            # Draw the centroid circle
            cv.circle(img_centroids, (cx, cy), radius2, color2, thickness)
            
            # Add a small center dot
            cv.circle(img_centroids, (cx, cy), 3, (0, 255, 0), -1)  # Green filled dot
            
            # Add cluster label (offset to avoid overlap)
            cv.putText(img_centroids,f"{cluster_name}",
                       (cx + radius2 + 5, cy + 15), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
            
    
    return img_centroids

def show_dual_centroid_visualization(img, cluster_info1, cluster_info2, radius1=150, radius2=75):
    """
    Complete function to create and display dual centroid visualization
    
    Args:
        img: Original image
        cluster_info1: First dictionary (e.g., big circles)
        cluster_info2: Second dictionary (e.g., mini circles)
        radius1: Radius for first dictionary
        radius2: Radius for second dictionary
    """
    # Create the visualization
    img_with_centroids = visualize_dual_cluster_centroids(
        img, cluster_info1, cluster_info2, radius1, radius2
    )
    
    # Show the result with matplotlib
    plt.figure(figsize=(15, 12))
    plt.imshow(cv.cvtColor(img_with_centroids, cv.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title(f'Dual Cluster Centroids (Yellow: r={radius1}, Magenta: r={radius2})')
    plt.axis('off')
    plt.show()
    
    return img_with_centroids
