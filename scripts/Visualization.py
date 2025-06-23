from bernstein_flow.DistributionTransform import GaussianDistTransform

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import numpy as np
import torch

def create_interactive_transformer_plot(model, dim):
    """
    Create an interactive plot of transformer functions and their derivatives.
    
    Args:
        model: Your model with transformer() and transformer_deriv() methods
        dim: Number of dimensions
    """
    
    # Create figure and subplots
    fig, axes = plt.subplots(dim, 2, figsize=(12, 3*dim))
    if dim == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for consistency
    
    # Adjust layout to make room for sliders
    plt.subplots_adjust(bottom=0.3, right=0.95)
    
    # Create slider axes - one slider for each dimension
    slider_height = 0.03
    slider_spacing = 0.04
    slider_axes = []
    sliders = []
    
    for i in range(dim):
        ax_slider = plt.axes([0.1, 0.05 + i * slider_spacing, 0.8, slider_height])
        slider = widgets.Slider(
            ax_slider, 
            f'Dim {i}', 
            0.0, 1.0, 
            valinit=0.5,
            valfmt='%.3f'
        )
        slider_axes.append(ax_slider)
        sliders.append(slider)
    
    # Store initial values
    xi_vals = torch.linspace(0, 1, 100, requires_grad=False)
    lines_tf = []
    lines_deriv = []
    
    # Initialize plots
    for i in range(dim):
        line_tf, = axes[i, 0].plot([], [], 'b-', linewidth=2)
        line_deriv, = axes[i, 1].plot([], [], 'r-', linewidth=2)
        lines_tf.append(line_tf)
        lines_deriv.append(line_deriv)
        
        axes[i, 0].set_ylabel(f"TF {i}")
        axes[i, 0].set_xlabel(f"x_{i}")
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].set_ylabel(f"dTF/dx {i}")
        axes[i, 1].set_xlabel(f"x_{i}")
        axes[i, 1].grid(True, alpha=0.3)
    
    def update_plots(val=None):
        """Update all plots when slider values change"""
        # Get current slider values
        slider_values = [slider.val for slider in sliders]
        
        for i in range(dim):
            # Create x_vals with current slider values
            x_vals = torch.ones(100, dim, requires_grad=False)
            for j in range(dim):
                x_vals[:, j] = slider_values[j]
            
            # Set the varying dimension
            x_vals[:, i] = xi_vals
            
            # Calculate transformer values and derivatives
            tf_vals = model.transformer(x_vals, i)
            tf_deriv_vals = model.transformer_deriv(x_vals, i)
            
            # Update line data
            lines_tf[i].set_data(xi_vals.numpy(), tf_vals.detach().numpy())
            lines_deriv[i].set_data(xi_vals.numpy(), tf_deriv_vals.detach().numpy())
            
            # Auto-scale y-axis
            axes[i, 0].relim()
            axes[i, 0].autoscale_view()
            axes[i, 1].relim()
            axes[i, 1].autoscale_view()
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    for slider in sliders:
        slider.on_changed(update_plots)
    
    # Initial plot update
    update_plots()
    
    # Add title
    fig.suptitle('Interactive Transformer Functions and Derivatives', fontsize=14)
    
    # Add instruction text
    fig.text(0.5, 0.02, 'Use sliders to change the fixed values of other dimensions', 
             ha='center', fontsize=10, style='italic')
    
    plt.show()
    
    return fig, axes, sliders

def create_interactive_traj_data_plot(trajectory_data):
    """
    Plots an interactive scatter plot of 2D state distributions across time steps.

    Parameters:
    -----------
    trajectory_data : list of np.ndarray
        A list of length k, where each element is a (p x 2) array.
        Each array contains p samples from the 2D state distribution at a specific time step.
    """
    k = len(trajectory_data)
    if k == 0:
        raise ValueError("trajectory_data must be a non-empty list")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    # Set global axis limits
    all_data = np.vstack(trajectory_data)
    x_min, x_max = np.min(all_data[:, 0]), np.max(all_data[:, 0])
    y_min, y_max = np.min(all_data[:, 1]), np.max(all_data[:, 1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("State dimension 1")
    ax.set_ylabel("State dimension 2")
    ax.set_title("State Distribution at Timestep 0")

    # Initial scatter
    scatter = ax.scatter(trajectory_data[0][:, 0], trajectory_data[0][:, 1], alpha=0.6)

    # Slider axis and widget
    slider_ax = plt.axes([0.15, 0.1, 0.7, 0.05])  # [left, bottom, width, height]
    timestep_slider = widgets.Slider(
        ax=slider_ax,
        label='Timestep',
        valmin=0,
        valmax=k - 1,
        valinit=0,
        valstep=1,
        color='steelblue'
    )

    # Update function
    def update(val):
        t = int(timestep_slider.val)
        scatter.set_offsets(trajectory_data[t])
        ax.set_title(f"p(x{t})")
        fig.canvas.draw_idle()

    timestep_slider.on_changed(update)

    plt.show()

    return fig, ax

def evaluate_u_density_on_grid(model, resolution=100, device=None):
    """
    Evaluate a trained model on a 2D grid over [0, 1]^2.

    Args:
        model: A trained PyTorch model that maps 2D inputs to scalar densities.
        resolution: Number of points along each axis (default: 100).
        device: Device to run the model on. If None, uses model's device.

    Returns:
        U0, U1`: Meshgrid coordinates (numpy arrays)
        Z: Density values on the grid (numpy array of shape [resolution, resolution])
    """
    if device is None:
        device = next(model.parameters()).device

    # Create 2D grid over [0, 1] x [0, 1]
    u0 = np.linspace(0, 1, resolution)
    u1 = np.linspace(0, 1, resolution)
    U0, U1 = np.meshgrid(u0, u1)

    # Flatten grid and convert to tensor
    grid_points = np.stack([U0.ravel(), U1.ravel()], axis=-1)  # shape: (resolution^2, 2)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # Evaluate model
    with torch.no_grad():
        model.eval()
        densities = model(grid_tensor).cpu().numpy()  # shape: (resolution^2,)

    Z = densities.reshape(resolution, resolution)
    return U0, U1, Z

def evaluate_x_density_on_grid(model, gdt: GaussianDistTransform, bounds : list, resolution=100, device=None):
    """
    Evaluate a trained model on a 2D grid over R^n

    Args:
        model: A trained PyTorch model that maps 2D inputs to scalar densities.
        resolution: Number of points along each axis (default: 100).
        device: Device to run the model on. If None, uses model's device.

    Returns:
        X0, X1: Meshgrid coordinates (numpy arrays)
        Z: Density values on the grid (numpy array of shape [resolution, resolution])
    """
    if device is None:
        device = next(model.parameters()).device

    # Create 2D grid over [0, 1] x [0, 1]
    x0 = np.linspace(bounds[0], bounds[1], resolution)
    x1 = np.linspace(bounds[2], bounds[3], resolution)
    X0, X1 = np.meshgrid(x0, x1)

    # Flatten grid and convert to tensor
    x_grid_points = np.stack([X0.ravel(), X1.ravel()], axis=-1)  # shape: (resolution^2, 2)
    x_grid_tensor = torch.tensor(x_grid_points, dtype=torch.float32, device=device)

    x_grid_tensor.dtype

    # Evaluate model
    def u_density(u : np.ndarray):
        u = torch.from_numpy(u)
        u = u.to(dtype=x_grid_tensor.dtype)
        with torch.no_grad():
            model.eval()
            densities = model(u).cpu().numpy()  # shape: (resolution^2,)
            return densities
        
    x_densities = gdt.x_density(x_grid_tensor, u_density)

    Z = x_densities.reshape(resolution, resolution)
    return X0, X1, Z

def plot_density_surface(ax, X0, X1, Z):
    # Create surface plot
    surf = ax.plot_surface(X0, X1, Z, cmap='viridis', linewidth=0, antialiased=True)
    return ax

def plot_density(ax : plt.Axes, X0, X1, Z):
    ax.contourf(X0, X1, Z, levels=50, cmap='viridis')
