from bernstein_flow.DistributionTransform import GaussianDistTransform
#from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
from sos_form.SumBetaModel import SumBetaSOSModel
from sos_form.PowerFunctionModel import PowerFunctionSOSModel
from sos_form.SignomialModel import SignomialSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, mc_auc

from .Systems import CubicMap, sample_trajectories, sample_io_pairs
from .Visualization import interactive_state_distribution_plot_1D, transition_distribution_plot, plot_density_1D, plot_data_1D, plot_data_2D, plot_density_2D_surface, plot_density_2D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

DTYPE = torch.float64

def plot_beliefs_pdfs(beliefs, x_range=(0.1, 0.9), resolution=100, save_path=None, show_plot=True):
    """
    Plot all PDFs from beliefs list in a single row of subplots.
    
    Args:
        beliefs: List of models where each model's forward method returns the PDF
        x_range: Tuple of (min, max) for x-axis range
        resolution: Number of points to evaluate the PDF
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_beliefs = len(beliefs)
    
    # Create subplots in a single row
    fig, axes = plt.subplots(1, n_beliefs, figsize=(4*n_beliefs, 4))
    if n_beliefs == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    # Create x values for evaluation
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    
    for i, belief in enumerate(beliefs):
        # Evaluate the PDF for this belief
        with torch.no_grad():
            # Convert x_vals to tensor format expected by the model
            x_tensor = torch.from_numpy(x_vals).reshape(-1, 1).float()
            pdf_vals = belief(x_tensor).numpy()
        
        # Plot the PDF
        axes[i].plot(x_vals, pdf_vals, 'b-', linewidth=2)
        axes[i].set_title(f'Belief {i}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)
        
        # Set consistent y-axis limits across all subplots
        if i == 0:
            max_density = np.max(pdf_vals)
        else:
            max_density = max(max_density, np.max(pdf_vals))
    
    # Set consistent y-axis limits
    for ax in axes:
        ax.set_ylim(0, max_density * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_mc_particles_histograms(u_traj_data, x_range=(0.1, 0.9), bins=50, save_path=None, show_plot=True):
    """
    Plot histograms of MC particles from u_traj_data in a single row of subplots.
    
    Args:
        u_traj_data: List of arrays where each array contains particle data for a timestep
        x_range: Tuple of (min, max) for x-axis range
        bins: Number of bins for histogram
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    n_timesteps = len(u_traj_data)
    
    # Create subplots in a single row
    fig, axes = plt.subplots(1, n_timesteps, figsize=(4*n_timesteps, 4))
    if n_timesteps == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    # Find global max count for consistent y-axis scaling
    max_count = 0
    for i, particles in enumerate(u_traj_data):
        # Filter particles within x_range
        mask = (particles >= x_range[0]) & (particles <= x_range[1])
        filtered_particles = particles[mask]
        
        if len(filtered_particles) > 0:
            counts, _ = np.histogram(filtered_particles, bins=bins, range=x_range)
            max_count = max(max_count, np.max(counts))
    
    for i, particles in enumerate(u_traj_data):
        # Filter particles within x_range
        mask = (particles >= x_range[0]) & (particles <= x_range[1])
        filtered_particles = particles[mask]
        
        if len(filtered_particles) > 0:
            # Create histogram
            axes[i].hist(filtered_particles, bins=bins, range=x_range, 
                        alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
            axes[i].set_title(f'MC Particles t={i}')
            axes[i].set_xlabel('u')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)
            
            # Set consistent y-axis limits
            axes[i].set_ylim(0, max_count * 1.1)
        else:
            axes[i].set_title(f'MC Particles t={i} (No data)')
            axes[i].set_xlabel('u')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
    
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
    system = CubicMap(dt=0.1, alpha=0.8, variance=0.5)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 1000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 1000

    # Time horizon
    training_timesteps = 5
    timesteps = training_timesteps

    init_mode_means = [2.0, -2.0]
    def init_state_sampler():
        mode = np.random.randint(0, 2)
        #return float(mode) * norm.rvs(loc=np.array([1.0]), scale = 1.2) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.0]), scale = 1.2)
        #return float(mode) * norm.rvs(loc=np.array([init_mode_means[mode]]), scale = 0.5) + (1.0 - float(mode)) * norm.rvs(loc=np.array([init_mode_means[mode]]), scale = 0.5)
        return norm.rvs(loc=np.array([init_mode_means[0]]), scale = 0.5) #+ (1.0 - float(mode)) * norm.rvs(loc=np.array([init_mode_means[mode]]), scale = 0.5)


    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-2.0], region_uppers=[2.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    #interactive_state_distribution_plot_1D(traj_data, bins=60)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[3.5])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    #interactive_state_distribution_plot_1D(u_traj_data)

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    #Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data 
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, dim:]), gdt.X_to_U(Xp_data[:, :dim])])  # Transition kernel data 
    #Up_io_data = np.hstack([gdt.X_to_U(io_data[:, :dim]), gdt.X_to_U(io_data[:, dim:])])
    #Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data 

    #plt.scatter(Up_io_data[:, 0], Up_io_data[:, 1], s=1)
    #plt.show()

    use_gpu = True
    print("GPU available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print("device: ", device)
    print("Using GPU: ", use_gpu)

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=32, shuffle=True, pin_memory=use_gpu)

    #Up_data_torch = torch.tensor(Up_io_data, dtype=DTYPE)
    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=512, shuffle=True, pin_memory=use_gpu)
    Up_dataloader_refine = DataLoader(Up_dataset, batch_size=2048, shuffle=True, pin_memory=use_gpu)

    ## Create initial state and transition models



    n = 4
    n_terms = 5
    transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, min_alpha_beta=0.1, max_alpha_beta=50.0, mu=0.05, min_Q_eigval=1e-8)
    #transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, m=m, min_alpha_beta=0.1, sigma_init=10.0, sigma_max=500, eta=0.8)
    #transition_model = PowerFunctionSOSModel(dy=dim, dx=dim, n=n, m=m, min_exp=0.0, sigma_init=10.0, sigma_max=500, max_exp=30.0)
    #transition_model = SignomialSOSModel(dy=dim, dx=dim, n=n, m=m, n_terms=5, min_exp=0.0, sigma_init=10.0, sigma_max=300, max_exp=30.0)
    #transition_model = SumBetaSOSModel(dy=dim, dx=dim, n=n, n_terms=n_terms, min_alpha_beta=0.1, max_alpha_beta=50.0, mu=0.01, min_Q_eigval=1e-8)



    print("Training transition model...")
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=100)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
    optimize(transition_model, Up_dataloader_refine, trans_optimizer, epochs=50)

    transition_model.to(device=torch.device("cpu"))
    print("Done training transition model \n")

    init_state_model = BetaSOSModel(dy=dim, dx=0, n=n, conditional=False,reference_factor_model=transition_model, min_alpha_beta=0.1, max_alpha_beta=50.0, mu=0.05, min_Q_eigval=1e-8)
    #init_state_model = SumBetaSOSModel(dy=dim, dx=0, n=n, n_terms=n_terms, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.1, max_alpha_beta=50.0, mu=0.01, min_Q_eigval=1e-8)

    print("Training init state model...")
    init_state_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    optimize(init_state_model, U0_dataloader, trans_optimizer, epochs=100)

    init_state_model.to(device=torch.device("cpu"))
    print("Done training init state model \n")







    with torch.no_grad():
        uls = torch.linspace(0.1, 0.9, 10)
        for u in uls:
            auc = mc_auc(1, lambda up : transition_model(torch.hstack((torch.from_numpy(up), u*torch.ones_like(torch.from_numpy(up))))).numpy(), n_samples=10000)
            print("auc: ", auc)

            ts_llh_auc = mc_auc(1, lambda up : gdt.u_density(up, lambda xp : system.transition_likelihood(gdt.u_to_x(u)* np.ones_like(xp), xp)))
            print("True auc: ", ts_llh_auc)

    beliefs = [init_state_model]
    for i in range(timesteps):
        #beliefs.append(transition_model.propagate(beliefs[i]))
        beliefs.append(transition_model.propagate(beliefs[i], n_terms=n_terms))
    
    print("\n")
    for i, belief in enumerate(beliefs):
        with torch.no_grad():
            auc = mc_auc(1, lambda u : belief(torch.from_numpy(u)).numpy(), n_samples=10000)
            print(f"Belief {i} auc: ", auc)

    def np_model_density(y, x):
        # Handle scalar inputs by converting to numpy arrays
        if np.isscalar(y):
            y = np.array([y])
        if np.isscalar(x):
            x = np.array([x])
        
        yx = torch.vstack((torch.from_numpy(y), torch.from_numpy(x))).t()
        with torch.no_grad():
            return transition_model(yx).numpy()

    pdf_funcs = [np_model_density, lambda up, u : gdt.u_density(up, lambda xp : system.transition_likelihood(gdt.u_to_x(u)* np.ones_like(xp), xp))]


    # Import the visualization functions
    from src.visualization import plot_conditional_distributions_2D, plot_conditional_distributions_slices
    from src.visualization.regular_distributions import plot_regular_distributions_1D, plot_regular_distributions_slices

    # Create the visualization
    plot_conditional_distributions_2D(pdf_funcs, u_range=(0.2, 0.8), up_range=(0.01, 0.99), resolution=50, 
                                    save_path="figures/conditional_distributions_comparison_1d.png", show_plot=False)
    
    # Create the new slice visualization
    plot_conditional_distributions_slices(pdf_funcs, u_range=(0.2, 0.8), up_range=(0.01, 0.99), 
                                        n_slices=20, resolution=100, 
                                        save_path="figures/conditional_distributions_slices_1d.png", show_plot=True)

    # Create regular distribution functions for initial state model
    def np_init_model_density(u):
        # Handle scalar inputs by converting to numpy arrays
        if np.isscalar(u):
            u = np.array([u])
        
        u_tensor = torch.from_numpy(u).reshape(-1, 1)
        with torch.no_grad():
            return init_state_model(u_tensor).numpy()
    
    def np_true_init_x_density(x):
        from scipy.stats import norm
        #density1 = 0.5 * norm.pdf(x, loc=init_mode_means[0], scale=0.5)
        #density2 = 0.5 * norm.pdf(x, loc=init_mode_means[1], scale=0.5)
        #return density1 + density2
        density1 = norm.pdf(x, loc=init_mode_means[0], scale=0.5)
        return density1

    np_true_init_u_density = lambda u : gdt.u_density(u, np_true_init_x_density)

    # Create regular distribution visualization
    regular_pdf_funcs = [np_true_init_u_density, np_init_model_density, ]
    
    # Plot regular distributions comparison
    plot_regular_distributions_1D(regular_pdf_funcs, x_range=(0.1, 0.9), resolution=100,
                                 save_path="figures/regular_distributions_1d.png", show_plot=False,
                                 labels=['Model Initial State', 'True Initial State'])
    
    # Plot regular distributions as slices
    plot_regular_distributions_slices(regular_pdf_funcs, x_range=(0.1, 0.9), n_slices=2, resolution=100,
                                     save_path="figures/regular_distributions_slices_1d.png", show_plot=False,
                                     labels=['Model Initial State', 'True Initial State'])

    #transition_distribution_plot(pdf_funcs, dim=dim)
    
    # Plot all beliefs in a single row
    plot_beliefs_pdfs(beliefs, x_range=(0.1, 0.9), resolution=100, 
                     save_path="figures/beliefs_evolution_1d.png", show_plot=True)
    
    # Plot MC particles as histograms for comparison
    plot_mc_particles_histograms(u_traj_data, x_range=(0.1, 0.9), bins=50,
                               save_path="figures/mc_particles_histograms_1d.png", show_plot=True)