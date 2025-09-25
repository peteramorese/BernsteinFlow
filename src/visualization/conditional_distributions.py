"""
Visualization utilities for conditional distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_conditional_distributions_slices(pdf_funcs, u_range=(0.1, 0.9), up_range=(0.1, 0.9), 
                                        n_slices=4, resolution=100, save_path=None, show_plot=False):
    """
    Plot conditional distributions p(up | u) as multiple slices for fixed u values.
    Shows both model and true distributions overlaid for comparison.
    
    Parameters:
    -----------
    pdf_funcs : list of callable
        List of two functions, each with signature f(up, u) returning density values
    u_range : tuple
        Range for u values (u_min, u_max)
    up_range : tuple  
        Range for up values (up_min, up_max)
    n_slices : int
        Number of slices to show (number of fixed u values)
    resolution : int
        Number of points along up axis for each slice
    save_path : str, optional
        Path to save the figure. If None, uses default name with timestamp
    show_plot : bool
        Whether to display the plot
    """
    # Create fixed u values for slices
    u_slice_vals = np.linspace(u_range[0], u_range[1], n_slices)
    up_vals = np.linspace(up_range[0], up_range[1], resolution)
    
    # Create figure with subplots arranged in a grid
    n_cols = 2
    n_rows = (n_slices + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, u_fixed in enumerate(u_slice_vals):
        ax = axes_flat[i]
        
        # Evaluate both PDFs for this fixed u value
        model_densities = []
        true_densities = []
        
        for up_val in up_vals:
            model_densities.append(pdf_funcs[0](up_val, u_fixed))
            true_densities.append(pdf_funcs[1](up_val, u_fixed))
        
        # Plot both distributions on the same subplot
        ax.plot(up_vals, model_densities, 'b-', linewidth=2, label='Model', alpha=0.8)
        ax.plot(up_vals, true_densities, 'r--', linewidth=2, label='True', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('up')
        ax.set_ylabel('p(up | u)')
        ax.set_title(f'p(up | u = {u_fixed:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits across all subplots for better comparison
        all_densities = model_densities + true_densities
        y_max = max(all_densities) * 1.1
        ax.set_ylim(0, y_max)
    
    # Hide any unused subplots
    for i in range(n_slices, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"conditional_distributions_slices_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory
    
    return fig, axes


def plot_conditional_distributions_2D(pdf_funcs, u_range=(0.1, 0.9), up_range=(0.1, 0.9), resolution=50, 
                                    save_path=None, show_plot=False):
    """
    Plot conditional distributions p(up | u) as 2D heatmaps.
    
    Parameters:
    -----------
    pdf_funcs : list of callable
        List of two functions, each with signature f(up, u) returning density values
    u_range : tuple
        Range for u values (u_min, u_max)
    up_range : tuple  
        Range for up values (up_min, up_max)
    resolution : int
        Number of points along each axis
    save_path : str, optional
        Path to save the figure. If None, uses default name with timestamp
    show_plot : bool
        Whether to display the plot
    """
    # Create coordinate grids
    u_vals = np.linspace(u_range[0], u_range[1], resolution)
    up_vals = np.linspace(up_range[0], up_range[1], resolution)
    U, UP = np.meshgrid(u_vals, up_vals, indexing='ij')
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    titles = ['Model Conditional Distribution', 'True Conditional Distribution']
    
    for i, pdf_func in enumerate(pdf_funcs):
        # Evaluate PDF over the grid
        density = np.zeros_like(U)
        for j in range(resolution):
            for k in range(resolution):
                u_val = u_vals[j]
                up_val = up_vals[k]
                density[j, k] = pdf_func(up_val, u_val)
        
        # Plot as heatmap
        im = axes[i].imshow(density, extent=[up_range[0], up_range[1], u_range[0], u_range[1]], 
                          aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_xlabel('up')
        axes[i].set_ylabel('u')
        axes[i].set_title(titles[i])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Density')
    
    plt.tight_layout()
    
    # Save the figure
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"conditional_distributions_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory
    
    return fig, axes
