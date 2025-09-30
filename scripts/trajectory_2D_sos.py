from bernstein_flow.DistributionTransform import GaussianDistTransform
#from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
from sos_form.SumBetaModel import SumBetaSOSModel
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

if __name__ == "__main__":

    # System model
    #system = Pendulum(dt=0.05, length=1.0, damp=1.1, covariance=0.005 * np.eye(2))
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 2000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 1000

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-5.0, -5.0], region_uppers=[5.0, 5.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[4.2, 4.2])
    #gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[0.2, 0.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data 
    Up_io_data = np.hstack([gdt.X_to_U(io_data[:, :dim]), gdt.X_to_U(io_data[:, dim:])])

    #use_gpu = torch.cuda.is_available()
    use_gpu = True
    print("Using GPU: ", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
    #device = torch.device("cpu")
    print("device: ", device)

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=256, shuffle=True, pin_memory=use_gpu)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=512, shuffle=True, pin_memory=use_gpu)

    ## Create initial state and transition models


    n = 8
    n_terms = 10
    #transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, min_alpha_beta=0.1, max_alpha_beta=60.0, mu=0.1, min_Q_eigval=1e-8)
    transition_model = SumBetaSOSModel(dy=dim, dx=dim, n=n, n_terms=n_terms, min_alpha_beta=0.1, max_alpha_beta=60.0, mu=0.1, min_Q_eigval=1e-8)

    print("Training transition model...")
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=300)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=300)

    transition_model.to(device=torch.device("cpu"))
    print("Done training transition model \n")

    init_state_model = SumBetaSOSModel(dy=dim, dx=0, n=n, n_terms=n_terms, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.1, max_alpha_beta=40.0, mu=0.1, min_Q_eigval=1e-8)

    print("Training init state model...")
    init_state_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    optimize(init_state_model, U0_dataloader, trans_optimizer, epochs=500)
    trans_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-4)
    optimize(init_state_model, U0_dataloader, trans_optimizer, epochs=500)

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

    # Plot all beliefs in a grid using contour plots
    plot_beliefs_pdfs_2d(beliefs, x_range=(0.1, 0.9), y_range=(0.1, 0.9), resolution=50, 
                        save_path="figures/beliefs_evolution_2d.png", show_plot=True)
    
    # Plot MC particles as scatter plots for comparison
    plot_mc_particles_scatter_2d(u_traj_data, x_range=(0.1, 0.9), y_range=(0.1, 0.9),
                                save_path="figures/mc_particles_scatter_2d.png", show_plot=True)

