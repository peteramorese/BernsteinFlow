from bernstein_flow.DistributionTransform import GaussianDistTransform
#from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
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

if __name__ == "__main__":

    # System model
    system = CubicMap(dt=0.01, alpha=0.5, variance=0.5)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 10

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 1000

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        mode = np.random.randint(0, 2)
        #return float(mode) * norm.rvs(loc=np.array([1.0]), scale = 1.2) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.0]), scale = 1.2)
        return float(mode) * norm.rvs(loc=np.array([1.5]), scale = 0.5) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.5]), scale = 0.5)


    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-2.0], region_uppers=[2.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    #interactive_state_distribution_plot_1D(traj_data, bins=60)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[0.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    #interactive_state_distribution_plot_1D(u_traj_data)

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data 
    Up_io_data = np.hstack([gdt.X_to_U(io_data[:, :dim]), gdt.X_to_U(io_data[:, dim:])])
    #Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data 

    #plt.scatter(Up_io_data[:, 0], Up_io_data[:, 1], s=1)
    #plt.show()

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=32, shuffle=True)

    #Up_data_torch = torch.tensor(Up_io_data, dtype=DTYPE)
    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=256, shuffle=True)
    Up_dataloader_refine = DataLoader(Up_dataset, batch_size=2048, shuffle=True)

    ## Create initial state and transition models

    n = 2
    transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, min_alpha_beta=0.1)
    #transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, m=m, min_alpha_beta=0.1, sigma_init=10.0, sigma_max=500, eta=0.8)
    #transition_model = PowerFunctionSOSModel(dy=dim, dx=dim, n=n, m=m, min_exp=0.0, sigma_init=10.0, sigma_max=500, max_exp=30.0)
    #transition_model = SignomialSOSModel(dy=dim, dx=dim, n=n, m=m, n_terms=5, min_exp=0.0, sigma_init=10.0, sigma_max=300, max_exp=30.0)



    print("Training transition model...")
    transition_model.to(DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-1)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=700)

    print("Done training transition model \n")

    Q, R = transition_model.get_QR_matrices()
    #print("R: \n", R)
    print("R evals: ", torch.linalg.eigvals(R))
    #print("Q: \n", Q)
    print("Q evals: ", torch.linalg.eigvals(Q))


    with torch.no_grad():
        uls = torch.linspace(0.1, 0.9, 10)
        for u in uls:
            auc = mc_auc(1, lambda up : transition_model(torch.hstack((torch.from_numpy(up), u*torch.ones_like(torch.from_numpy(up))))).numpy(), n_samples=10000)
            print("auc: ", auc)

            ts_llh_auc = mc_auc(1, lambda up : gdt.u_density(up, lambda xp : system.transition_likelihood(gdt.u_to_x(u)* np.ones_like(xp), xp)))
            print("True auc: ", ts_llh_auc)

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


    # Import the visualization function
    from src.visualization import plot_conditional_distributions_2D, plot_conditional_distributions_slices

    # Create the visualization
    plot_conditional_distributions_2D(pdf_funcs, u_range=(0.2, 0.8), up_range=(0.01, 0.99), resolution=50, 
                                    save_path="figures/conditional_distributions_comparison.png", show_plot=False)
    
    # Create the new slice visualization
    plot_conditional_distributions_slices(pdf_funcs, u_range=(0.2, 0.8), up_range=(0.01, 0.99), 
                                        n_slices=10, resolution=100, 
                                        save_path="figures/conditional_distributions_slices.png", show_plot=False)

    #transition_distribution_plot(pdf_funcs, dim=dim)