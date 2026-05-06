import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import cv2 as cv
from scipy.ndimage import rotate

def plot_chain_solutions(solutions, points, max_bond_distance=None, pixelsize=None):
    """
    Plot comprehensive analysis of chain solutions
    - Bond length histogram for best solution
    - 2D overlay of all chain paths in different colors
    - 3D visualization of best solution
    - Solution comparison statistics
    """
    
    if not solutions:
        print("No solutions to plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Convert points to arrays for easier plotting
    points_array = np.array(points)
    x_pos = points_array[:, 0]
    y_pos = points_array[:, 1]
    z_pos = points_array[:, 2]
    
    # Plot 1: Bond length distribution for best solution
    ax1 = fig.add_subplot(2, 3, 1)
    
    best_solution = solutions[0]
    distances_list = list(best_solution['bond_distances'].values())
    
    if pixelsize:
        distances_nm = [dist * pixelsize for dist in distances_list]
        ax1.hist(distances_nm, bins=min(15, len(distances_nm)), alpha=0.7, 
                color='blue', edgecolor='black')
        ax1.axvline(np.mean(distances_nm), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances_nm):.3f} nm')
        if max_bond_distance:
            ax1.axvline(max_bond_distance * pixelsize, color='orange', linestyle='--', 
                       label=f'Max allowed: {max_bond_distance * pixelsize:.2f} nm')
        ax1.set_xlabel('Bond Distance (nm)')
        ax1.set_title('Bond Length Distribution (Best Solution)')
    else:
        ax1.hist(distances_list, bins=min(15, len(distances_list)), alpha=0.7, 
                color='blue', edgecolor='black')
        ax1.axvline(np.mean(distances_list), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances_list):.3f}')
        if max_bond_distance:
            ax1.axvline(max_bond_distance, color='orange', linestyle='--', 
                       label=f'Max allowed: {max_bond_distance}')
        ax1.set_xlabel('Bond Distance (pixels)')
        ax1.set_title('Bond Length Distribution (Best Solution)')
    
    ax1.set_ylabel('Number of Bonds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2D overlay of all chain paths
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Plot points first
    scatter = ax2.scatter(x_pos, y_pos, c=z_pos, cmap='viridis', s=100, alpha=0.8, 
                         edgecolors='black', linewidth=1, zorder=3)
    
    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        ax2.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Get colors for different solutions
    colors = plt.cm.tab10(np.linspace(0, 1, len(solutions)))
    
    # Plot each chain path
    for i, (solution, color) in enumerate(zip(solutions, colors)):
        chain = solution['chain']
        
        # Draw chain as connected line segments
        chain_x = [x_pos[idx] for idx in chain]
        chain_y = [y_pos[idx] for idx in chain]
        
        ax2.plot(chain_x, chain_y, color=color, linewidth=2, alpha=0.7, 
                label=f'Solution {i} (L={solution["total_length"]:.1f})', zorder=2)
        
        # Mark endpoints
        ax2.scatter([chain_x[0], chain_x[-1]], [chain_y[0], chain_y[-1]], 
                   c=[color, color], s=200, marker='s', edgecolors='black', 
                   linewidth=2, zorder=4)
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'All Chain Solutions Overlay\\n({len(solutions)} solutions)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for z-values
    plt.colorbar(scatter, ax=ax2, label='Z Position (Intensity)', shrink=0.8)
    
    # Plot 3: 3D visualization of best solution
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Plot all points
    ax3.scatter(x_pos, y_pos, z_pos, c=z_pos, cmap='viridis', s=100, alpha=0.6, 
               edgecolors='black', linewidth=1)
    
    # Plot best chain in 3D
    best_chain = best_solution['chain']
    chain_x = [x_pos[idx] for idx in best_chain]
    chain_y = [y_pos[idx] for idx in best_chain]
    chain_z = [z_pos[idx] for idx in best_chain]
    
    ax3.plot(chain_x, chain_y, chain_z, color='red', linewidth=3, alpha=0.8)
    
    # Mark endpoints
    ax3.scatter([chain_x[0], chain_x[-1]], [chain_y[0], chain_y[-1]], [chain_z[0], chain_z[-1]], 
               c=['red', 'red'], s=300, marker='s', edgecolors='black', linewidth=2)
    
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_zlabel('Z Position (Intensity)')
    ax3.set_title(f'Best Chain Solution (3D)\\nLength: {best_solution["total_length"]:.3f}')
    
    # Plot 4: Solution length comparison
    ax4 = fig.add_subplot(2, 3, 4)
    
    solution_lengths = [sol['total_length'] for sol in solutions]
    solution_ids = list(range(len(solutions)))
    
    bars = ax4.bar(solution_ids, solution_lengths, alpha=0.7, color='skyblue', edgecolor='black')
    bars[0].set_color('red')  # Highlight best solution
    bars[0].set_alpha(1.0)
    
    ax4.set_xlabel('Solution ID')
    if pixelsize:
        # Convert to nm if pixelsize available
        solution_lengths_nm = [length * pixelsize for length in solution_lengths]
        ax4_twin = ax4.twinx()
        ax4_twin.bar(solution_ids, solution_lengths_nm, alpha=0, width=0)  # Invisible bars for scale
        ax4_twin.set_ylabel('Total Chain Length (nm)')
        ax4.set_ylabel('Total Chain Length (pixels)')
    else:
        ax4.set_ylabel('Total Chain Length')
    
    ax4.set_title('Solution Length Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Chain statistics summary
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    # Calculate statistics
    best_distances = list(best_solution['bond_distances'].values())
    if pixelsize:
        best_distances = [d * pixelsize for d in best_distances]
        unit = "nm"
        total_length = best_solution['total_length'] * pixelsize
    else:
        unit = "pixels"
        total_length = best_solution['total_length']
    
    stats_text = f"""
CHAIN NETWORK STATISTICS

Best Solution (ID: 0):
• Total Length: {total_length:.3f} {unit}
• Number of Bonds: {len(best_distances)}
• Average Bond: {np.mean(best_distances):.3f} {unit}
• Min Bond: {np.min(best_distances):.3f} {unit}
• Max Bond: {np.max(best_distances):.3f} {unit}
• Std Dev: {np.std(best_distances):.3f} {unit}

Chain Properties:
• Start Point: {best_chain[0]}
• End Point: {best_chain[-1]}
• Points Connected: {len(best_chain)}

All Solutions:
• Generated: {len(solutions)}
• Length Range: {min(solution_lengths):.1f} - {max(solution_lengths):.1f}
• Improvement: {((max(solution_lengths) - min(solution_lengths)) / min(solution_lengths) * 100):.1f}%
"""
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Plot 6: Bond length vs position along chain
    ax6 = fig.add_subplot(2, 3, 6)
    
    bond_positions = list(range(len(best_distances)))
    if pixelsize:
        ax6.plot(bond_positions, best_distances, 'o-', linewidth=2, markersize=6, color='blue')
        ax6.set_ylabel(f'Bond Length ({unit})')
    else:
        ax6.plot(bond_positions, best_distances, 'o-', linewidth=2, markersize=6, color='blue')
        ax6.set_ylabel(f'Bond Length ({unit})')
    
    ax6.axhline(np.mean(best_distances), color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {np.mean(best_distances):.3f} {unit}')
    
    ax6.set_xlabel('Bond Position in Chain')
    ax6.set_title('Bond Lengths Along Chain Path')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_chain_on_image(solution, points, img, valid_circles, method_name="Chain Network"):
    """
    Plot the selected chain solution on the original image
    """
    
    # Create a copy of the image
    img_with_chain = img.copy()
    
    # Draw the chain path
    chain = solution['chain']
    
    # Draw bonds as lines
    for i in range(len(chain) - 1):
        point1_idx = chain[i]
        point2_idx = chain[i + 1]
        
        # Get pixel coordinates from valid_circles
        x1, y1 = valid_circles[point1_idx][0], valid_circles[point1_idx][1]
        x2, y2 = valid_circles[point2_idx][0], valid_circles[point2_idx][1]
        
        # Draw line
        cv.line(img_with_chain, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green lines
    
    # Mark endpoints
    start_idx = chain[0]
    end_idx = chain[-1]
    start_x, start_y = valid_circles[start_idx][0], valid_circles[start_idx][1]
    end_x, end_y = valid_circles[end_idx][0], valid_circles[end_idx][1]
    
    # Draw endpoint markers
    cv.circle(img_with_chain, (start_x, start_y), 8, (255, 0, 0), -1)  # Red start
    cv.circle(img_with_chain, (end_x, end_y), 8, (0, 0, 255), -1)     # Blue end
    
    # Add labels
    cv.putText(img_with_chain, 'START', (start_x - 30, start_y - 15), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv.putText(img_with_chain, 'END', (end_x - 20, end_y - 15), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display
    plt.figure(figsize=(12, 12))
    plt.imshow(np.fliplr(rotate(img_with_chain, 0)))
    plt.title(f'{method_name} - Chain Path\\nLength: {solution["total_length"]:.3f}, Bonds: {len(solution["bond_distances"])}')
    plt.axis('off')
    plt.show()
    
    return img_with_chain

# Example usage functions
def analyze_and_plot_chains(solutions, points, valid_circles=None, img=None, 
                           max_bond_distance=None, pixelsize=None):
    """
    Complete analysis and plotting of chain solutions
    """
    
    if solutions is None:
        print("No solutions to analyze")
        return
    
    # Plot comprehensive analysis
    plot_chain_solutions(solutions, points, "Linear Chain Networks", 
                        max_bond_distance, pixelsize)
    
    # Plot best solution on image if available
    if valid_circles is not None and img is not None:
        best_solution = solutions[0]
        plot_chain_on_image(best_solution, points, img, valid_circles, 
                           "Best Chain Solution")
    
    return solutions[0]  # Return best solution