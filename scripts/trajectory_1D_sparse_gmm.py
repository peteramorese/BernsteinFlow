from bernstein_flow.SparseModel import ConditionalGMM, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, mc_auc
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct
from bernstein_flow.Propagate import propagate_bfm

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
    n_traj = 1000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 100

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        mode = np.random.randint(0, 2)
        #return float(mode) * norm.rvs(loc=np.array([1.0]), scale = 1.2) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.0]), scale = 1.2)
        return float(mode) * norm.rvs(loc=np.array([1.5]), scale = 0.5) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.5]), scale = 0.5)


    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-10.0], region_uppers=[10.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    #interactive_state_distribution_plot_1D(traj_data, bins=60)

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    Xp_io_data = np.hstack([io_data[:, :dim], io_data[:, dim:]])


    ## Create data loader
    #U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    #U0_dataset = TensorDataset(U0_data_torch)
    #U0_dataloader = DataLoader(U0_dataset, batch_size=32, shuffle=True)

    #Up_data_torch = torch.tensor(Up_io_data, dtype=DTYPE)
    Xp_data_torch = torch.tensor(Xp_data, dtype=DTYPE)
    Xp_dataset = TensorDataset(Xp_data_torch)
    Xp_dataloader = DataLoader(Xp_dataset, batch_size=128, shuffle=True)

    ## Create initial state and transition models
    #n_components = 200
    #max_degree = 60
    #init_state_model = BetaMixtureModel(dim, n_components, max_degree)

    nv = 100
    min_var_sigma = 0.05
    min_erf_sigma = 0.05
    ns = 1
    nt= 5
    transition_model = ConditionalGMM(d=dim, 
                                    dc=dim, 
                                    nv=nv, 
                                    ns=ns,
                                    nt=nt,
                                    min_var_sigma=min_var_sigma,
                                    min_erf_sigma=min_erf_sigma)



    ## Train the models
    #init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    #print("Training initial state model...")
    #optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init)
    #print("Done training initial state model \n")

    print("Training transition model...")
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    optimize(transition_model, Xp_dataloader, trans_optimizer, epochs=n_epochs_tran)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-3)
    optimize(transition_model, Xp_dataloader, trans_optimizer, epochs=n_epochs_tran)
    print("Done training transition model \n")

    x_bounds = [-4.0, 4.0]
    #with torch.no_grad():
    #    uls = torch.linspace(-4.0, 4.0, 10)
    #    for u in uls:
    #        auc = mc_auc(1, lambda up : transition_model(torch.hstack((torch.from_numpy(up), u*torch.ones_like(torch.from_numpy(up))))).numpy(), n_samples=10000)
    #        print("auc: ", auc)

    #        ts_llh_auc = mc_auc(1, lambda up : gdt.u_density(up, lambda xp : system.transition_likelihood(gdt.u_to_x(u)* np.ones_like(xp), xp)))
    #        print("True auc: ", ts_llh_auc)

    def np_model_density(y : np.ndarray, x : np.ndarray):
        yx = torch.vstack((torch.from_numpy(y), torch.from_numpy(x))).t()
        with torch.no_grad():
            return transition_model(yx).numpy()

    pdf_funcs = [np_model_density, lambda xp, x : system.transition_likelihood(x, xp)]

    transition_distribution_plot(pdf_funcs, dim=dim, x_range=x_bounds, y_range=x_bounds)