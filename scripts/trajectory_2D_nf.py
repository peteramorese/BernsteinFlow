from bernstein_flow.NormalizingFlow import ConditionalNormalizingFlow, optimize
from bernstein_flow.Tools import create_transition_data_matrix
from bernstein_flow.Propagate import propagate_nf

from .Systems import VanDerPol, Pendulum, sample_trajectories, sample_io_pairs

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import os


DTYPE = torch.float32  # nflows typically uses float32

def plot_particles_comparison_2d(predicted_particles_list, true_particles_list, 
                                  x_range=(-5.0, 5.0), y_range=(-5.0, 5.0), 
                                  save_path=None, show_plot=True):
    """
    Plot predicted and true particles side by side for comparison.
    
    Args:
        predicted_particles_list: List of arrays where each array contains predicted 2D particle data for a timestep
        true_particles_list: List of arrays where each array contains true 2D particle data for a timestep
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_timesteps = len(predicted_particles_list)
    assert len(true_particles_list) == n_timesteps, "Predicted and true particle lists must have same length"
    
    # Calculate grid layout (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_timesteps)))
    n_rows = int(np.ceil(n_timesteps / n_cols))
    
    # Create subplots - 2 columns per timestep (predicted and true)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6*n_cols, 4*n_rows))
    if n_timesteps == 1:
        axes = axes.reshape(1, -1) if axes.ndim > 1 else [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for i in range(n_timesteps):
        if i >= n_timesteps:
            break
        
        # Predicted particles (left subplot)
        pred_ax_idx = i * 2
        true_ax_idx = i * 2 + 1
        
        if pred_ax_idx < len(axes):
            # Filter particles within the specified ranges
            pred_particles = predicted_particles_list[i]
            mask_pred = ((pred_particles[:, 0] >= x_range[0]) & (pred_particles[:, 0] <= x_range[1]) & 
                        (pred_particles[:, 1] >= y_range[0]) & (pred_particles[:, 1] <= y_range[1]))
            filtered_pred = pred_particles[mask_pred]
            
            if len(filtered_pred) > 0:
                axes[pred_ax_idx].scatter(filtered_pred[:, 0], filtered_pred[:, 1], 
                                         alpha=0.6, s=10, color='blue', edgecolors='black', linewidth=0.5)
            axes[pred_ax_idx].set_title(f'Predicted t={i}')
            axes[pred_ax_idx].set_xlabel('x1')
            axes[pred_ax_idx].set_ylabel('x2')
            axes[pred_ax_idx].grid(True, alpha=0.3)
            axes[pred_ax_idx].set_xlim(x_range)
            axes[pred_ax_idx].set_ylim(y_range)
        
        # True particles (right subplot)
        if true_ax_idx < len(axes):
            true_particles = true_particles_list[i]
            mask_true = ((true_particles[:, 0] >= x_range[0]) & (true_particles[:, 0] <= x_range[1]) & 
                        (true_particles[:, 1] >= y_range[0]) & (true_particles[:, 1] <= y_range[1]))
            filtered_true = true_particles[mask_true]
            
            if len(filtered_true) > 0:
                axes[true_ax_idx].scatter(filtered_true[:, 0], filtered_true[:, 1], 
                                         alpha=0.6, s=10, color='orange', edgecolors='black', linewidth=0.5)
            axes[true_ax_idx].set_title(f'True t={i}')
            axes[true_ax_idx].set_xlabel('x1')
            axes[true_ax_idx].set_ylabel('x2')
            axes[true_ax_idx].grid(True, alpha=0.3)
            axes[true_ax_idx].set_xlim(x_range)
            axes[true_ax_idx].set_ylim(y_range)
    
    # Hide unused subplots
    for i in range(n_timesteps * 2, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
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
    n_epochs_init = 200
    n_epochs_tran = 20

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    # Number of particles for belief representation
    n_particles = 1000

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov=np.diag([0.2, 0.2]))

    # Sample trajectory data
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Create the data matrices for training (in original state space, no GDT)
    X0_data = traj_data[0]  # Initial state data
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])  # Transition data: [x, x']

    # For NF training, we need [x, x'] where x is conditioning and x' is target
    # Xp_data is already in format [x, x'], so we can use it directly
    # But we need to make sure the order is [x, x'] for p(x' | x)
    # create_transition_data_matrix returns [x_k, x_{k+1}], which is what we need!

    # Verify data format: Xp_data should be [x_k, x_{k+1}] = [x, x']
    print(f"Xp_data shape: {Xp_data.shape}")
    print(f"Xp_data sample (first row): x={Xp_data[0, :dim]}, x'={Xp_data[0, dim:]}")
    print(f"Xp_data stats - x: mean={Xp_data[:, :dim].mean(axis=0)}, std={Xp_data[:, :dim].std(axis=0)}")
    print(f"Xp_data stats - x': mean={Xp_data[:, dim:].mean(axis=0)}, std={Xp_data[:, dim:].std(axis=0)}")

    # Normalize data for better NF training (NFs work better with normalized data)
    # Store normalization stats for later use
    x_mean = Xp_data[:, :dim].mean(axis=0, keepdims=True)
    x_std = Xp_data[:, :dim].std(axis=0, keepdims=True) + 1e-8
    xp_mean = Xp_data[:, dim:].mean(axis=0, keepdims=True)
    xp_std = Xp_data[:, dim:].std(axis=0, keepdims=True) + 1e-8
    
    Xp_data_normalized = np.zeros_like(Xp_data)
    Xp_data_normalized[:, :dim] = (Xp_data[:, :dim] - x_mean) / x_std
    Xp_data_normalized[:, dim:] = (Xp_data[:, dim:] - xp_mean) / xp_std
    
    print(f"Normalized Xp_data stats - x: mean={Xp_data_normalized[:, :dim].mean(axis=0)}, std={Xp_data_normalized[:, :dim].std(axis=0)}")
    print(f"Normalized Xp_data stats - x': mean={Xp_data_normalized[:, dim:].mean(axis=0)}, std={Xp_data_normalized[:, dim:].std(axis=0)}")

    # Setup device
    use_gpu = torch.cuda.is_available()
    print("Using GPU: ", use_gpu)
    device = torch.device("cuda" if use_gpu else "cpu")
    print("device: ", device)

    # Create data loader for transition data: [x, x'] for p(x' | x)
    # Use normalized data for training
    Xp_data_torch = torch.tensor(Xp_data_normalized, dtype=DTYPE)
    Xp_dataset = TensorDataset(Xp_data_torch)
    Xp_dataloader = DataLoader(Xp_dataset, batch_size=512, shuffle=True, pin_memory=use_gpu)

    # Create transition model (conditional normalizing flow)
    print("Creating transition model...")
    device_str = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    # Increase model capacity for better learning
    transition_model = ConditionalNormalizingFlow(
        dim_x=dim,  # conditioning variable dimension
        dim_y=dim,  # target variable dimension
        num_layers=8,  # More layers for better expressiveness
        hidden_features=128,  # Larger hidden dimension
        device=device_str,
    )

    print("Training transition model...")
    transition_model.to(device)
    # Update device string to match actual device
    transition_model.device = device_str
    
    # Use learning rate schedule
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trans_optimizer, mode='min', factor=0.5, patience=20)
    
    # Train using optimize method
    optimize(transition_model, Xp_dataloader, trans_optimizer, epochs=n_epochs_tran, 
             print_interval=5, scheduler=scheduler)
    
    print("Done training transition model \n")

    # For initial state, we'll sample particles from the initial state distribution
    # (same distribution used to generate training data)
    print("Initializing belief particles from initial state distribution...")
    initial_particles = np.array([init_state_sampler() for _ in range(n_particles)])

    # Propagate beliefs forward using particle sets
    print("Propagating beliefs...")
    beliefs_particles = [initial_particles.copy()]
    print("initial belief particles shape = ", beliefs_particles[0].shape)
    
    for i in range(timesteps):
        current_particles = beliefs_particles[-1]
        
        # Propagate using propagate_nf function
        next_particles = propagate_nf(
            current_particles, 
            transition_model, 
            n_added_samples=1,
            x_mean=x_mean,
            x_std=x_std,
            xp_mean=xp_mean,
            xp_std=xp_std,
            dtype=DTYPE,
            device=device
        )
        
        beliefs_particles.append(next_particles)
        print(f"Propagated belief {i+1}/{timesteps}, shape = {next_particles.shape}")    



    print("\nDone propagating beliefs\n")

    # Prepare true trajectory data for comparison
    # Propagate the same initial particles through the true system
    print("Generating true system trajectories for comparison...")
    true_particles_list = [initial_particles.copy()]
    current_true_particles = initial_particles.copy()
    
    for t in range(timesteps):
        # Propagate each particle through the true system
        next_true_particles = np.array([system(particle) for particle in current_true_particles])
        true_particles_list.append(next_true_particles)
        current_true_particles = next_true_particles

    # Plot comparison
    print("Creating comparison plots...")
    os.makedirs("figures", exist_ok=True)
    plot_particles_comparison_2d(
        beliefs_particles, 
        true_particles_list,
        x_range=(-5.0, 5.0), 
        y_range=(-5.0, 5.0),
        save_path="figures/nf_particles_comparison_2d.png", 
        show_plot=True
    )

    print("Done!")
