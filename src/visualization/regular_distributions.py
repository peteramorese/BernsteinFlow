"""
Visualization utilities for regular 1D distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_regular_distributions_1D(pdf_funcs, x_range=(0.1, 0.9), resolution=100, 
                                 save_path=None, show_plot=False, labels=None):
    """
    Plot regular 1D distributions p(x) for comparison between model and true distributions.
    
    Parameters:
    -----------
    pdf_funcs : list of callable
        List of functions, each with signature f(x) returning density values
    x_range : tuple
        Range for x values (x_min, x_max)
    resolution : int
        Number of points along x axis for plotting
    save_path : str, optional
        Path to save the figure. If None, uses default name with timestamp
    show_plot : bool
        Whether to display the plot
    labels : list of str, optional
        Labels for each distribution. If None, uses default labels
    """
    # Create x values for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Default labels if not provided
    if labels is None:
        labels = [f'Distribution {i+1}' for i in range(len(pdf_funcs))]
    
    # Colors for different distributions
    colors = ['b-', 'r--', 'g:', 'm-', 'c-', 'y-']
    
    # Plot each distribution
    for i, (pdf_func, label) in enumerate(zip(pdf_funcs, labels)):
        # Evaluate PDF
        densities = []
        for x_val in x_vals:
            densities.append(pdf_func(x_val))
        
        # Plot the distribution
        color_style = colors[i % len(colors)]
        ax.plot(x_vals, densities, color_style, linewidth=2, label=label, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.set_title('Regular 1D Distributions Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    all_densities = []
    for pdf_func in pdf_funcs:
        for x_val in x_vals:
            all_densities.append(pdf_func(x_val))
    y_max = max(all_densities) * 1.1
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    
    # Save the figure
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"regular_distributions_1D_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory
    
    return fig, ax


def plot_regular_distributions_slices(pdf_funcs, x_range=(0.1, 0.9), n_slices=4, 
                                     resolution=100, save_path=None, show_plot=False, labels=None):
    """
    Plot regular 1D distributions p(x) as multiple subplots for better comparison.
    Useful when you have many distributions to compare.
    
    Parameters:
    -----------
    pdf_funcs : list of callable
        List of functions, each with signature f(x) returning density values
    x_range : tuple
        Range for x values (x_min, x_max)
    n_slices : int
        Number of subplots to show (if more than n_slices distributions, they'll be grouped)
    resolution : int
        Number of points along x axis for each plot
    save_path : str, optional
        Path to save the figure. If None, uses default name with timestamp
    show_plot : bool
        Whether to display the plot
    labels : list of str, optional
        Labels for each distribution. If None, uses default labels
    """
    n_funcs = len(pdf_funcs)
    
    # Create figure with subplots arranged in a grid
    n_cols = 2
    n_rows = (n_slices + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Default labels if not provided
    if labels is None:
        labels = [f'Distribution {i+1}' for i in range(n_funcs)]
    
    # Colors for different distributions
    colors = ['b-', 'r--', 'g:', 'm-', 'c-', 'y-']
    
    # Create x values for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    
    # Plot each distribution in its own subplot
    for i in range(min(n_slices, n_funcs)):
        ax = axes_flat[i]
        
        # Evaluate and plot the PDF
        densities = []
        for x_val in x_vals:
            densities.append(pdf_funcs[i](x_val))
        
        color_style = colors[i % len(colors)]
        ax.plot(x_vals, densities, color_style, linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('x')
        ax.set_ylabel('p(x)')
        ax.set_title(f'{labels[i]}')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        y_max = max(densities) * 1.1
        ax.set_ylim(0, y_max)
    
    # Hide any unused subplots
    for i in range(min(n_slices, n_funcs), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"regular_distributions_slices_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory
    
    return fig, axes
