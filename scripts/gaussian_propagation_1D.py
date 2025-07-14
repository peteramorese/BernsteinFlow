from bernstein_flow.GPGMM import fit_gmm, fit_gp, compute_mean_jacobian
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Propagate import propagate_gpgmm_ekf

from .Systems import CubicMap, sample_trajectories, sample_io_pairs
from .Visualization import interactive_transformer_plot, interactive_state_distribution_plot_1D, plot_density_1D, plot_data_1D, plot_data_2D, plot_density_2D_surface, plot_density_2D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from scipy.integrate import quad
import torch

if __name__ == "__main__":

    # System model
    system = CubicMap(dt=0.01, alpha=0.5, variance=0.5)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 500

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
    X_data, Xp_data = create_transition_data_matrix(traj_data[:training_timesteps], separate=True)

    print("Fitting gmm...")
    init_state_model = fit_gmm(X0_data, n_components=10)
    print("Fitting gp...")
    transition_model = fit_gp(X_data, Xp_data)

    # Visualize init state model
    def viz_init_state_model(ax : plt.Axes):
        ax.hist(X0_data, bins=30, density=True)

        X0 = np.linspace(-5, 5, 100).reshape(-1, 1)
        densities = init_state_model.density(X0)
        
        ax.plot(X0, densities)
    
    # Visualize the transition model
    def viz_transition_model(axes : list[plt.Axes]):
        x_slices = np.linspace(-5, 5, len(axes))

        for ax, x_slice in zip(axes, x_slices):
            ax.set_title(f"p(x' | x = {x_slice:.2f})")
            Xp = np.linspace(-5, 5, 100).reshape(-1, 1)

            true_p_xp = system.transition_likelihood(x_slice * np.ones_like(Xp), Xp)
            ax.plot(Xp, true_p_xp)

            mean, std = transition_model.predict(torch.tensor([x_slice], dtype=torch.float32))

            ax.plot(Xp, norm.pdf(Xp, mean, std))
    
    fig = plt.figure()
    ax = fig.gca()
    viz_init_state_model(ax)

    fig, axes = plt.subplots(1,7)
    viz_transition_model(axes)

    gmms = [init_state_model]
    for k in range(1, timesteps):
        next_gmm = propagate_gpgmm_ekf(gmms[k-1], transition_model)
        gmms.append(next_gmm)

    def pdf_plotter(k : int):
        X = np.linspace(-5, 5, 100)

        densities = gmms[k].density(X)
        return X, densities

    interactive_state_distribution_plot_1D(traj_data, pdf_plotter, bins=60)

    plt.show()
        
    