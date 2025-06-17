import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

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
