import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2

def display_plot(created_image, title, raw_image=False):
    """
    Display or save a plot of the created image.
    
    Parameters:
    - created_image: 2D array of the image data
    - title: Title for the plot (ignored if raw_image=True)
    - raw_image: Boolean flag
        - True: Save raw image without colorbar, title, or axes
        - False: Display formatted plot with colorbar, title, and axes
    
    Returns:
    - fig: matplotlib figure object
    """
    if raw_image:
        # Create figure with no axes, titles, or colorbars
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(created_image, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
    else:
        # Original formatted plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(created_image, cmap='viridis', origin='lower')
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label="Height (Å)")
    
    return fig

def save_plots_dicts(dictionary, cluster_dict, clus, save_path, image_key='img', name_file='which', n_cols=10,save_dict=False):
    """
    Plot all images in a cluster, save to file and save dictionary as pickle
   
    Parameters:
    - dictionary: dictionary containing image data
    - cluster_dict: dictionary mapping cluster numbers to image names
    - clus: cluster number to visualize
    - save_path: path where cluster folders are located
    - image_key: key to access image data in dictionary (default 'img')
    - name_file: prefix for saved files (default 'which')
    - n_cols: number of columns in the grid (default 10)
    
    Returns:
    - tuple: (plot_file_path, pickle_file_path) or None if failed
    """
    # Early validation
    if not dictionary or clus not in cluster_dict:
        print(f"No data or cluster {clus} not found")
        return None
    
    # Setup paths and ensure directory exists
    cluster_folder = os.path.join(save_path, f'cluster_{clus}')
    os.makedirs(cluster_folder, exist_ok=True)
    
    # Calculate grid dimensions
    n_files = len(cluster_dict[clus])
    n_rows = int(np.ceil(n_files / n_cols))
   
    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
   
    # Handle axes dimension edge cases
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_files == 1:
        axes = np.array([[axes]])
   
    # Plot each image in the cluster
    for i, name in enumerate(cluster_dict[clus]):
        row = i // n_cols
        col = i % n_cols
       
        # Extract image data - more flexible approach
        try:
            if isinstance(dictionary[name], dict):
                image = dictionary[name][image_key]
            else:
                # Handle case where dictionary[name] is directly the image
                image = dictionary[name]
        except (KeyError, TypeError):
            print(f"Warning: Could not extract image for {name} using key '{image_key}'")
            # Create empty subplot for missing data
            axes[row, col].text(0.5, 0.5, f'No data\n{name}', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
            continue
            
        # Plot the image
        im = axes[row, col].imshow(image, cmap='viridis')
        axes[row, col].set_title(f'{name}', fontsize=8)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
   
    # Hide empty subplots
    for i in range(n_files, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
   
    # Configure layout and title
    plt.tight_layout()
    plt.suptitle(f'Cluster {clus} - {n_files} Images', fontsize=16, y=1.02)
   
    # Save plot
    plot_file = os.path.join(cluster_folder, f'{name_file}_{clus}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save dictionary as pickle
    if save_dict:
        pickle_file = os.path.join(cluster_folder, f'{name_file}_{clus}.pkl')
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(dictionary, f)
            print(f"Saved cluster {clus} dictionary to: {pickle_file}")
            print(f"Saved cluster {clus} plot to: {plot_file}")
            return plot_file, pickle_file
        except Exception as e:
            print(f"Error saving pickle file: {e}")
            print(f"Saved cluster {clus} plot to: {plot_file}")
            return plot_file, None
    else:
        print(f"Saved cluster {clus} plot to: {plot_file}")
        return plot_file, None
    
def save_ridge_plots_dicts(dictionary, cluster_dict, clus, save_path, 
                          ridge_key='ridge', name_file='ridge', n_cols=10):
    # Early validation
    if not dictionary or clus not in cluster_dict:
        print(f"No data or cluster {clus} not found")
        return None
    
    # Setup paths and ensure directory exists
    cluster_folder = os.path.join(save_path, f'cluster_{clus}')
    os.makedirs(cluster_folder, exist_ok=True)
    
    # Calculate grid dimensions
    n_files = len(cluster_dict[clus])
    n_rows = int(np.ceil(n_files / n_cols))

    # Setup paths and ensure directory exists
    cluster_folder = os.path.join(save_path, f'cluster_{clus}')
    os.makedirs(cluster_folder, exist_ok=True)
   
    # Calculate grid dimensions
    n_files = len(cluster_dict[clus])
    n_rows = int(np.ceil(n_files / n_cols))
   
    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
   
    # Handle axes dimension edge cases
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_files == 1:
        axes = np.array([[axes]])
   
    for i, name in enumerate(cluster_dict[clus]):
        row = i // n_cols
        col = i % n_cols
       
        try:
            backbone_result = dictionary[name]['backbone_ridge']
            measurements = dictionary[name]['measurements']
            
            # Plot the image
            backbone_rgb = cv2.cvtColor(backbone_result['visualization'], cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(backbone_rgb)
            
            # Title with key measurements in NANOMETERS
            title = f'{name}\n'
            title += f'Length: {measurements["backbone_length_nm"]:.1f} nm\n'
            title += f'E2E: {measurements["end_to_end_distance_nm"]:.1f} nm\n'
            title += f'Ratio: {measurements["contour_ratio"]:.2f}'
            
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis('off')
            
        except (KeyError, TypeError):
            print(f"Warning: Could not extract ridge for {name}")
            # Create empty subplot for missing data
            axes[row, col].text(0.5, 0.5, f'No data\n{name}',
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
            continue
   
    # Hide empty subplots
    for i in range(n_files, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Configure layout and title
    plt.tight_layout()
    plt.suptitle(f'Cluster {clus} - {n_files} Images', fontsize=16, y=1.02)
    
    # Save plot
    plot_file = os.path.join(cluster_folder, f'{name_file}_{clus}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved cluster {clus} plot to: {plot_file}")
    return plot_file