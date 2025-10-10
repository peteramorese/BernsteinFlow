from bernstein_flow.DistributionTransform import GaussianDistTransform
#from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
from sos_form.SumBetaModel import SumBetaSOSModel, SumBetaMarginalSOSModel
from sos_form.PowerFunctionModel import PowerFunctionSOSModel
from sos_form.SignomialModel import SignomialSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, mc_auc

from .Systems import VanDerPol, Pendulum, sample_trajectories, sample_io_pairs
from .Visualization import interactive_transformer_plot, state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import os


DTYPE = torch.float64

def plot_beliefs_pdfs_2d(beliefs, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50, save_path=None, show_plot=True):
    """
    Plot all 2D PDFs from beliefs list in a grid of subplots using contour plots.
    
    Args:
        beliefs: List of models where each model's forward method returns the PDF
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        resolution: Number of points per axis for evaluation
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_beliefs = len(beliefs)
    
    # Calculate grid layout (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_beliefs)))
    n_rows = int(np.ceil(n_beliefs / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_beliefs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    # Create grid for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten for model evaluation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    for i, belief in enumerate(beliefs):
        if i >= len(axes):
            break
            
        # Evaluate the PDF for this belief
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid_points).float()
            pdf_vals = belief(grid_tensor).numpy()
        
        # Reshape back to grid
        Z = pdf_vals.reshape(X.shape)
        
        # Create contour plot
        contour = axes[i].contour(X, Y, Z, levels=10, colors='blue', alpha=0.7)
        axes[i].contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
        axes[i].set_title(f'Belief {i}')
        axes[i].set_xlabel('u1')
        axes[i].set_ylabel('u2')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(x_range)
        axes[i].set_ylim(y_range)
    
    # Hide unused subplots
    for i in range(n_beliefs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_mc_particles_scatter_2d(u_traj_data, x_range=(0.1, 0.9), y_range=(0.1, 0.9), save_path=None, show_plot=True):
    """
    Plot 2D MC particles from u_traj_data as scatter plots in a grid of subplots.
    
    Args:
        u_traj_data: List of arrays where each array contains 2D particle data for a timestep
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_timesteps = len(u_traj_data)
    
    # Calculate grid layout (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_timesteps)))
    n_rows = int(np.ceil(n_timesteps / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_timesteps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for i, particles in enumerate(u_traj_data):
        if i >= len(axes):
            break
            
        # Filter particles within the specified ranges
        mask = ((particles[:, 0] >= x_range[0]) & (particles[:, 0] <= x_range[1]) & 
                (particles[:, 1] >= y_range[0]) & (particles[:, 1] <= y_range[1]))
        filtered_particles = particles[mask]
        
        if len(filtered_particles) > 0:
            # Create scatter plot
            axes[i].scatter(filtered_particles[:, 0], filtered_particles[:, 1], 
                           alpha=0.6, s=10, color='orange', edgecolors='black', linewidth=0.5)
            axes[i].set_title(f'MC Particles t={i}')
            axes[i].set_xlabel('u1')
            axes[i].set_ylabel('u2')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(x_range)
            axes[i].set_ylim(y_range)
        else:
            axes[i].set_title(f'MC Particles t={i} (No data)')
            axes[i].set_xlabel('u1')
            axes[i].set_ylabel('u2')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(x_range)
            axes[i].set_ylim(y_range)
    
    # Hide unused subplots
    for i in range(n_timesteps, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MC particles plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_beliefs_with_marginals(beliefs, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50, save_path=None, show_plot=True):
    """
    Plot each belief showing the full 2D distribution and its 1D marginals.
    
    Args:
        beliefs: List of models where each model's forward method returns the PDF
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        resolution: Number of points per axis for evaluation
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_beliefs = len(beliefs)
    
    # Create a figure with subplots for each belief
    # Each belief gets 3 subplots: 2D contour, x1 marginal, x2 marginal
    fig, axes = plt.subplots(n_beliefs, 3, figsize=(15, 5*n_beliefs))
    if n_beliefs == 1:
        axes = axes.reshape(1, -1)
    
    # Create grid for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten for model evaluation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    for i, belief in enumerate(beliefs):
        # Evaluate the PDF for this belief
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid_points).float()
            pdf_vals = belief(grid_tensor).numpy()
        
        # Reshape back to grid
        Z = pdf_vals.reshape(X.shape)
        
        # Plot 2D contour
        contour = axes[i, 0].contour(X, Y, Z, levels=10, colors='blue', alpha=0.7)
        axes[i, 0].contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
        axes[i, 0].set_title(f'Belief {i} - Full 2D Distribution')
        axes[i, 0].set_xlabel('u1')
        axes[i, 0].set_ylabel('u2')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(x_range)
        axes[i, 0].set_ylim(y_range)
        
        # Compute and plot x1 marginal (integrate over x2)
        x1_marginal = np.trapz(Z, y_vals, axis=0)
        axes[i, 1].plot(x_vals, x1_marginal, 'b-', linewidth=2)
        axes[i, 1].fill_between(x_vals, x1_marginal, alpha=0.3, color='blue')
        axes[i, 1].set_title(f'Belief {i} - x1 Marginal')
        axes[i, 1].set_xlabel('u1')
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(x_range)
        
        # Compute and plot x2 marginal (integrate over x1)
        x2_marginal = np.trapz(Z, x_vals, axis=1)
        axes[i, 2].plot(y_vals, x2_marginal, 'r-', linewidth=2)
        axes[i, 2].fill_between(y_vals, x2_marginal, alpha=0.3, color='red')
        axes[i, 2].set_title(f'Belief {i} - x2 Marginal')
        axes[i, 2].set_xlabel('u2')
        axes[i, 2].set_ylabel('Density')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_xlim(y_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Beliefs with marginals plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_marginalized_beliefs(beliefs, x1_marginals, x2_marginals, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50, save_path=None, show_plot=True):
    """
    Plot the marginalized beliefs (1D distributions) alongside the original beliefs.
    
    Args:
        beliefs: List of original 2D models
        x1_marginals: List of x1 marginalized models
        x2_marginals: List of x2 marginalized models
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        resolution: Number of points per axis for evaluation
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_beliefs = len(beliefs)
    
    # Create a figure with subplots for each belief
    # Each belief gets 4 subplots: 2D contour, x1 marginal (computed), x1 marginal (exact), x2 marginal (exact)
    fig, axes = plt.subplots(n_beliefs, 4, figsize=(20, 5*n_beliefs))
    if n_beliefs == 1:
        axes = axes.reshape(1, -1)
    
    # Create grid for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten for model evaluation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    for i, (belief, x1_marg, x2_marg) in enumerate(zip(beliefs, x1_marginals, x2_marginals)):
        # Evaluate the PDF for this belief
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid_points).float()
            pdf_vals = belief(grid_tensor).numpy()
        
        # Reshape back to grid
        Z = pdf_vals.reshape(X.shape)
        
        # Plot 2D contour
        contour = axes[i, 0].contour(X, Y, Z, levels=10, colors='blue', alpha=0.7)
        axes[i, 0].contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
        axes[i, 0].set_title(f'Belief {i} - Full 2D Distribution')
        axes[i, 0].set_xlabel('u1')
        axes[i, 0].set_ylabel('u2')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(x_range)
        axes[i, 0].set_ylim(y_range)
        
        # Compute x1 marginal by integration (for comparison)
        x1_marginal_computed = np.trapz(Z, y_vals, axis=0)
        axes[i, 1].plot(x_vals, x1_marginal_computed, 'b-', linewidth=2, label='Computed')
        axes[i, 1].fill_between(x_vals, x1_marginal_computed, alpha=0.3, color='blue')
        axes[i, 1].set_title(f'Belief {i} - x1 Marginal (Computed)')
        axes[i, 1].set_xlabel('u1')
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(x_range)
        
        # Plot exact x1 marginal
        with torch.no_grad():
            # For marginalized models, input is now just the 1D variable z
            x1_tensor = torch.from_numpy(x_vals.reshape(-1, 1)).float()
            x1_marginal_exact = x1_marg(x1_tensor).numpy().flatten()
        axes[i, 2].plot(x_vals, x1_marginal_exact, 'g-', linewidth=2, label='Exact')
        axes[i, 2].fill_between(x_vals, x1_marginal_exact, alpha=0.3, color='green')
        axes[i, 2].set_title(f'Belief {i} - x1 Marginal (Exact)')
        axes[i, 2].set_xlabel('u1')
        axes[i, 2].set_ylabel('Density')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_xlim(x_range)
        
        # Plot exact x2 marginal
        with torch.no_grad():
            # For marginalized models, input is now just the 1D variable z
            x2_tensor = torch.from_numpy(y_vals.reshape(-1, 1)).float()
            x2_marginal_exact = x2_marg(x2_tensor).numpy().flatten()
        axes[i, 3].plot(y_vals, x2_marginal_exact, 'r-', linewidth=2, label='Exact')
        axes[i, 3].fill_between(y_vals, x2_marginal_exact, alpha=0.3, color='red')
        axes[i, 3].set_title(f'Belief {i} - x2 Marginal (Exact)')
        axes[i, 3].set_xlabel('u2')
        axes[i, 3].set_ylabel('Density')
        axes[i, 3].grid(True, alpha=0.3)
        axes[i, 3].set_xlim(y_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Marginalized beliefs plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_marginal_comparison(beliefs, x1_marginals, x2_marginals, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=100, save_path=None, show_plot=True):
    """
    Plot comparison between computed marginals (via integration) and exact marginals (via marginalize()).
    
    Args:
        beliefs: List of original 2D models
        x1_marginals: List of x1 marginalized models
        x2_marginals: List of x2 marginalized models
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        resolution: Number of points per axis for evaluation
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_beliefs = len(beliefs)
    
    # Create a figure with subplots for each belief
    # Each belief gets 2 subplots: x1 comparison, x2 comparison
    fig, axes = plt.subplots(n_beliefs, 2, figsize=(12, 5*n_beliefs))
    if n_beliefs == 1:
        axes = axes.reshape(1, -1)
    
    # Create grid for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten for model evaluation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    for i, (belief, x1_marg, x2_marg) in enumerate(zip(beliefs, x1_marginals, x2_marginals)):
        # Evaluate the PDF for this belief
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid_points).float()
            pdf_vals = belief(grid_tensor).numpy()
        
        # Reshape back to grid
        Z = pdf_vals.reshape(X.shape)
        
        # Compute x1 marginal by integration
        x1_marginal_computed = np.trapz(Z, y_vals, axis=0)
        
        # Get exact x1 marginal
        with torch.no_grad():
            # For marginalized models, input is now just the 1D variable z
            x1_tensor = torch.from_numpy(x_vals.reshape(-1, 1)).float()
            x1_marginal_exact = x1_marg(x1_tensor).numpy().flatten()
        
        # Plot x1 comparison
        axes[i, 0].plot(x_vals, x1_marginal_computed, 'b-', linewidth=2, label='Computed (integration)', alpha=0.7)
        axes[i, 0].plot(x_vals, x1_marginal_exact, 'g--', linewidth=2, label='Exact (marginalize)', alpha=0.9)
        axes[i, 0].set_title(f'Belief {i} - x1 Marginal Comparison')
        axes[i, 0].set_xlabel('u1')
        axes[i, 0].set_ylabel('Density')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(x_range)
        
        # Compute x2 marginal by integration
        x2_marginal_computed = np.trapz(Z, x_vals, axis=1)
        
        # Get exact x2 marginal
        with torch.no_grad():
            # For marginalized models, input is now just the 1D variable z
            x2_tensor = torch.from_numpy(y_vals.reshape(-1, 1)).float()
            x2_marginal_exact = x2_marg(x2_tensor).numpy().flatten()
        
        # Plot x2 comparison
        axes[i, 1].plot(y_vals, x2_marginal_computed, 'b-', linewidth=2, label='Computed (integration)', alpha=0.7)
        axes[i, 1].plot(y_vals, x2_marginal_exact, 'r--', linewidth=2, label='Exact (marginalize)', alpha=0.9)
        axes[i, 1].set_title(f'Belief {i} - x2 Marginal Comparison')
        axes[i, 1].set_xlabel('u2')
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(y_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Marginal comparison plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":

    # System model
    #system = Pendulum(dt=0.05, length=1.0, damp=1.1, covariance=0.005 * np.eye(2))
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 1000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 1000

    # Time horizon
    training_timesteps = 3
    timesteps = training_timesteps

    init_mode_means = [np.array([0.5, 0.5]), np.array([-0.5, -0.5])]
    def init_state_sampler():
        mode = np.random.randint(0, 2)
        return multivariate_normal.rvs(mean=init_mode_means[mode], cov = np.diag([0.2, 0.2]))

    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-5.0, -5.0], region_uppers=[5.0, 5.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[2.2, 2.2])
    #gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[0.2, 0.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data 
    Up_io_data = np.hstack([gdt.X_to_U(io_data[:, :dim]), gdt.X_to_U(io_data[:, dim:])])

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=1024, shuffle=True)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=512, shuffle=True)

    ## Create initial state and transition models

    print("Using GPU: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    n = 4
    n_terms = 7
    #transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, min_alpha_beta=0.1, max_alpha_beta=40.0, mu=0.1, min_Q_eigval=1e-8)
    transition_model = SumBetaSOSModel(dy=dim, dx=dim, n=n, n_terms=n_terms, min_alpha_beta=0.1, max_alpha_beta=40.0, mu=0.1, min_Q_eigval=1e-8, regularization_weight=1e-5)

    print("Training transition model...")
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=5)

    transition_model.to(device=torch.device("cpu"))
    print("Done training transition model \n")

    #init_state_model = BetaSOSModel(dy=dim, dx=0, n=n, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.1, max_alpha_beta=25.0, mu=0.1, min_Q_eigval=1e-8)
    init_state_model = SumBetaSOSModel(dy=dim, dx=0, n=n, n_terms=n_terms, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.1, max_alpha_beta=40.0, mu=0.1, min_Q_eigval=1e-8, regularization_weight=1e-4)

    #init_state_model = BetaSOSModel(dy=dim, dx=0, n=n, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.1, max_alpha_beta=25.0, mu=0.1, min_Q_eigval=1e-8)

    print("Training init state model...")
    init_state_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    optimize(init_state_model, U0_dataloader, trans_optimizer, epochs=50)

    init_state_model.to(device=torch.device("cpu"))
    print("Done training init state model \n")

    beliefs = [init_state_model]
    for i in range(timesteps):
        #beliefs.append(transition_model.propagate(beliefs[i]))
        beliefs.append(transition_model.propagate(beliefs[i], n_terms=n_terms))

    print("\n")
    for i, belief in enumerate(beliefs):
        with torch.no_grad():
            auc = mc_auc(2, lambda u : belief(torch.from_numpy(u)).numpy(), n_samples=10000)
            print(f"Belief {i} auc: ", auc)

    # Create marginalized beliefs for testing
    print("\nCreating marginalized beliefs...")
    x1_marginals = []
    x2_marginals = []
    
    for i, belief in enumerate(beliefs):
        print(f"Marginalizing belief {i}...")
        # Create x1 marginal (integrate out x2, keep x1)
        x1_marginal = belief.marginalize([1])  # Integrate out dimension 1 (x2)
        #x1_marginal = belief.marginalize([1], n_terms=n_terms)  # Integrate out dimension 1 (x2)
        x1_marginals.append(x1_marginal)
        
        # Create x2 marginal (integrate out x1, keep x2)  
        x2_marginal = belief.marginalize([0])  # Integrate out dimension 0 (x1)
        x2_marginals.append(x2_marginal)
    
    print("Done creating marginalized beliefs\n")

    # Validate marginalization by comparing computed vs exact marginals
    print("Validating marginalization...")
    for i, (belief, x1_marg, x2_marg) in enumerate(zip(beliefs, x1_marginals, x2_marginals)):
        # Test a few points to verify the marginals are correct
        # For marginalized models, input is now just the 1D variable z
        test_points_x1 = torch.tensor([[0.3], [0.5], [0.7]], dtype=torch.float64)
        test_points_x2 = torch.tensor([[0.3], [0.5], [0.7]], dtype=torch.float64)
        
        with torch.no_grad():
            x1_marg_vals = x1_marg(test_points_x1).numpy()
            x2_marg_vals = x2_marg(test_points_x2).numpy()
            
        print(f"Belief {i} - x1 marginal at test points: {x1_marg_vals.flatten()}")
        print(f"Belief {i} - x2 marginal at test points: {x2_marg_vals.flatten()}")
    print("Validation complete\n")

    # Plot all beliefs in a grid using contour plots
    plot_beliefs_pdfs_2d(beliefs, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50, 
                        save_path="figures/beliefs_evolution_2d.png", show_plot=True)
    
    # Plot beliefs with computed marginals
    plot_beliefs_with_marginals(beliefs, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50,
                               save_path="figures/beliefs_with_marginals.png", show_plot=True)
    
    # Plot marginalized beliefs (exact vs computed)
    plot_marginalized_beliefs(beliefs, x1_marginals, x2_marginals, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50,
                             save_path="figures/marginalized_beliefs_comparison.png", show_plot=True)
    
    # Plot detailed comparison between computed and exact marginals
    plot_marginal_comparison(beliefs, x1_marginals, x2_marginals, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50,
                            save_path="figures/marginal_comparison_detailed.png", show_plot=True)
    
    # Plot MC particles as scatter plots for comparison
    plot_mc_particles_scatter_2d(u_traj_data, x_range=(0.1, 0.9), y_range=(0.1, 0.9),
                                save_path="figures/mc_particles_scatter_2d.png", show_plot=True)

