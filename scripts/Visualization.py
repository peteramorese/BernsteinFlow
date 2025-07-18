from bernstein_flow.DistributionTransform import GaussianDistTransform

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import numpy as np
import torch

def interactive_transformer_plot(model, dim, cond_dim = 0, dtype = torch.float32):
    """
    Create an interactive plot of transformer functions and their derivatives.
    
    Args:
        model: Model with transformer() and transformer_deriv() methods
        dim: Number of dimensions
        cond_dim: Number of conditional dimensions
    """
    
    n_sliders = dim + cond_dim
    
    # Create figure and subplots
    fig, axes = plt.subplots(dim, 2, figsize=(12, 3*(dim)))
    if dim == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for consistency
    
    # Adjust layout to make room for sliders
    plt.subplots_adjust(bottom=0.1 * n_sliders, right=0.95)
    
    # Create slider axes - one slider for each dimension
    slider_height = 0.03
    slider_spacing = 0.04
    slider_axes = []
    sliders = []
    
    for i in range(cond_dim):
        ax_slider = plt.axes([0.1, 0.02 + (n_sliders-1-i) * slider_spacing, 0.8, slider_height])
        #ax_slider = plt.axes([0.1, 0.05 + i * slider_spacing, 0.8, slider_height])
        slider = widgets.Slider(
            ax_slider, 
            f'Cond. Dim {i}', 
            0.0, 1.0, 
            valinit=0.5,
            valfmt='%.3f'
        )
        slider_axes.append(ax_slider)
        sliders.append(slider)
    
    for i in range(dim):
        ax_slider = plt.axes([0.1, 0.02 + (dim-1-i) * slider_spacing, 0.8, slider_height])
        #ax_slider = plt.axes([0.1, 0.05 + i * slider_spacing, 0.8, slider_height])
        slider = widgets.Slider(
            ax_slider, 
            f'Dim {i}', 
            0.0, 1.0, 
            valinit=0.5,
            valfmt='%.3f'
        )
        slider_axes.append(ax_slider)
        sliders.append(slider)

    print("number of sliders: ", len(slider_axes), len(sliders))

    # Store initial values
    n_xi_vals = 100
    xi_vals = torch.linspace(0, 1, n_xi_vals, requires_grad=False, dtype=dtype)
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
            x_vals = torch.ones(n_xi_vals, cond_dim + dim, requires_grad=False, dtype=dtype)
            for j in range(cond_dim + dim):
                x_vals[:, j] = slider_values[j]
            
            # Set the varying dimension
            x_vals[:, cond_dim + i] = xi_vals
            
            # Calculate transformer values and derivatives
            tf_vals = model.transformer(x_vals, i, layer_i=0)
            tf_deriv_vals = model.transformer_deriv(x_vals, i, layer_i=0)
            
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

def state_distribution_plot_2D(trajectory_data, pdf_func=None, interactive=True, bounds=None):
    """
    Plots a (interactive) scatter plot of 2D state distributions across time steps.
    
    If pdf_func is provided, adds a second subplot to visualize a 2D density.

    Parameters:
    -----------
    trajectory_data : list of np.ndarray
        A list of length k, where each element is a (p x 2) array.
        Each array contains p samples from the 2D state distribution at a specific time step.

    pdf_func : Optional[Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]]]
        A function that takes a timestep index `k` and returns (X, Y, Z) for a 2D density plot,
        where X and Y are meshgrids and Z is the density evaluated over them.
    """
    k = len(trajectory_data)
    if k == 0:
        raise ValueError("trajectory_data must be a non-empty list")

    all_data = np.vstack(trajectory_data)
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        #print("xmin: ", x_min, " xmax: ", x_max)
    else:
        x_min, x_max = np.min(all_data[:, 0]), np.max(all_data[:, 0])
        y_min, y_max = np.min(all_data[:, 1]), np.max(all_data[:, 1])
        #print("xmin: ", x_min, " xmax: ", x_max)

    if not interactive:
        if pdf_func is not None:
            fig, axes = plt.subplots(2, k, figsize=(4*k, 8), squeeze=False)
            scatter_axes = axes[0]
            pdf_axes = axes[1]
        else:
            fig, scatter_axes = plt.subplots(1, k, figsize=(4*k, 4), squeeze=False)
            scatter_axes = scatter_axes[0]
            pdf_axes = None

        for t in range(k):
            ax = scatter_axes[t]
            ax.scatter(trajectory_data[t][:, 0], trajectory_data[t][:, 1], alpha=0.6)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("State dim 1")
            ax.set_ylabel("State dim 2")
            ax.set_title(f"Timestep {t}")

            if pdf_func is not None:
                ax_pdf = pdf_axes[t]
                X, Y, Z = pdf_func(t)
                ax_pdf.contourf(X, Y, Z, levels=50, cmap='viridis')
                ax_pdf.set_xlim(x_min, x_max)
                ax_pdf.set_ylim(y_min, y_max)
                ax_pdf.set_xlabel("x1")
                ax_pdf.set_ylabel("x2")
                ax_pdf.set_title(f"PDF {t}")

        plt.tight_layout()
        #plt.show()
        return fig, axes

    if pdf_func is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_scatter, ax_pdf = axes
    else:
        fig, ax_scatter = plt.subplots(figsize=(6, 6))
        ax_pdf = None

    plt.subplots_adjust(bottom=0.25)

    # Set global axis limits
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(y_min, y_max)
    ax_scatter.set_xlabel("State dimension 1")
    ax_scatter.set_ylabel("State dimension 2")
    ax_scatter.set_title("State Distribution at Timestep 0")

    # Initial scatter
    scatter = ax_scatter.scatter(trajectory_data[0][:, 0], trajectory_data[0][:, 1], alpha=0.6)

    # Initial density plot if pdf_func is provided
    if pdf_func is not None:
        X, Y, Z = pdf_func(0)
        pdf_plot = ax_pdf.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax_pdf.set_xlim(x_min, x_max)
        ax_pdf.set_ylim(y_min, y_max)
        ax_pdf.set_title("PDF at Timestep 0")
        ax_pdf.set_xlabel("x1")
        ax_pdf.set_ylabel("x2")

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

    def update(val):
        t = int(timestep_slider.val)

        # Update scatter plot
        scatter.set_offsets(trajectory_data[t])
        ax_scatter.set_title(f"State Distribution at Timestep {t}")

        # Update PDF plot
        if pdf_func is not None:
            for c in ax_pdf.collections:
                c.remove()
            X, Y, Z = pdf_func(t)
            ax_pdf.contourf(X, Y, Z, levels=50, cmap='viridis')
            ax_pdf.set_title(f"PDF at Timestep {t}")

        fig.canvas.draw_idle()

    timestep_slider.on_changed(update)
    plt.show()

def interactive_state_distribution_plot_1D(trajectory_data, pdf_func=None, bins=30):
    """
    Plots an interactive histogram of 1D state distributions across time steps.
    
    If pdf_func is provided, adds a second subplot to visualize the 1D density.

    Parameters:
    -----------
    trajectory_data : list of np.ndarray
        A list of length k, where each element is a (p,) or (p,1) array of 1D samples.

    pdf_func : Optional[Callable[[int], Tuple[np.ndarray, np.ndarray]]]
        A function that takes a timestep index `k` and returns (X, Y),
        where X is a 1D array of positions and Y is the corresponding PDF values.

    bins : int
        Number of histogram bins.
    """
    k = len(trajectory_data)
    if k == 0:
        raise ValueError("trajectory_data must be a non-empty list")

    # Ensure all data is flat 1D
    trajectory_data = [x.flatten() for x in trajectory_data]

    # Set up figure and subplots
    if pdf_func is not None:
        fig, (ax_hist, ax_pdf) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax_hist = plt.subplots(figsize=(6, 5))
        ax_pdf = None

    plt.subplots_adjust(bottom=0.25)

    # Global x-axis limits
    all_data = np.concatenate(trajectory_data)
    x_min, x_max = np.min(all_data), np.max(all_data)

    # Initial histogram
    hist_vals, bins_, patches = ax_hist.hist(
        trajectory_data[0], bins=bins, range=(x_min, x_max), alpha=0.7, edgecolor='black'
    )
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.set_title("State Histogram at Timestep 0")
    ax_hist.set_xlabel("State value")
    ax_hist.set_ylabel("Frequency")

    # Initial PDF line plot
    if pdf_func is not None:
        X, Y = pdf_func(0)
        pdf_line, = ax_pdf.plot(X, Y, color='purple', lw=2, zorder=2)
        ax_pdf.set_xlim(x_min, x_max)
        ax_pdf.set_ylim(0, max(Y) * 1.1)
        ax_pdf.set_title("PDF at Timestep 0")
        ax_pdf.set_xlabel("x")
        ax_pdf.set_ylabel("Density")

    # Slider
    slider_ax = plt.axes([0.15, 0.1, 0.7, 0.05])
    timestep_slider = widgets.Slider(
        ax=slider_ax,
        label='Timestep',
        valmin=0,
        valmax=k - 1,
        valinit=0,
        valstep=1,
        color='steelblue'
    )

    def update(val):
        t = int(timestep_slider.val)

        # Update histogram
        ax_hist.cla()
        ax_hist.hist(trajectory_data[t], bins=bins, range=(x_min, x_max),
                     alpha=0.7, edgecolor='black', density=True)
        ax_hist.set_xlim(x_min, x_max)
        ax_hist.set_title(f"State Histogram at Timestep {t}")
        ax_hist.set_xlabel("x")
        ax_hist.set_ylabel("Frequency")

        # Update PDF
        if pdf_func is not None:
            X, Y = pdf_func(t)
            pdf_line.set_data(X, Y)
            ax_pdf.set_ylim(0, max(Y) * 1.1)
            ax_pdf.set_title(f"PDF at Timestep {t}")
            ax_pdf.relim()
            ax_pdf.autoscale_view()

        fig.canvas.draw_idle()

    timestep_slider.on_changed(update)
    plt.show()


def plot_density_2D_surface(ax, X0, X1, Z):
    # Create surface plot
    surf = ax.plot_surface(X0, X1, Z, cmap='viridis', linewidth=0, antialiased=True)
    return ax

def plot_density_1D(ax : plt.Axes, X, Z, alpha = 0.2):
    ax.plot(X, Z, color='purple')
    ax.fill_between(X, Z, alpha=alpha, color='purple')

def plot_density_2D(ax : plt.Axes, X0, X1, Z):
    ax.contourf(X0, X1, Z, levels=50, cmap='viridis')
    ax.contour(X0, X1, Z, levels=10, colors='white', linewidths=0.8)

def plot_data_1D(ax : plt.Axes, data : np.ndarray, bins=10):
    assert data.shape[1] == 1
    ax.hist(data, bins=bins, density=True)

def plot_data_2D(ax : plt.Axes, data : np.ndarray):
    assert data.shape[1] == 2
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
