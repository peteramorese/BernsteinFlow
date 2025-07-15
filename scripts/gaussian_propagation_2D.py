from bernstein_flow.GPGMM import fit_gmm, fit_gp, compute_mean_jacobian
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Propagate import propagate_gpgmm_ekf

from .Systems import VanDerPol, Pendulum, sample_trajectories, sample_io_pairs
from .Visualization import interactive_transformer_plot, interactive_state_distribution_plot_2D, plot_density_1D, plot_data_1D, plot_data_2D, plot_density_2D_surface, plot_density_2D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.integrate import quad
import torch

if __name__ == "__main__":

    # System model
    #system = Pendulum(dt=0.15, length=1.0, damp=5.1, covariance=0.005 * np.eye(2))
    system = VanDerPol(dt=0.3, mu=1.5, covariance=0.005 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 300

    # Time horizon
    training_timesteps = 8
    timesteps = training_timesteps

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.1, 0.1]), cov = np.diag([0.10, 0.01]))

    #io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-10.0], region_uppers=[10.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    interactive_state_distribution_plot_2D(traj_data)

    # Create the data matrices for training
    X0_data = traj_data[0]
    X_data, Xp_data = create_transition_data_matrix(traj_data[:training_timesteps], separate=True)

    print("Fitting gmm...")
    print("X0 data size: ", X0_data.shape)
    init_state_model = fit_gmm(X0_data, n_components=10)
    print("Fitting gp...")
    print("X data size: ", X_data.shape)
    transition_model = fit_gp(X_data, Xp_data)

    ## Visualize init state model
    #def viz_init_state_model(ax : plt.Axes):
    #    ax.hist(X0_data, bins=30, density=True)

    #    X0 = np.linspace(-5, 5, 100).reshape(-1, 1)
    #    densities = init_state_model.density(X0)
    #    
    #    ax.plot(X0, densities)
    
    ## Visualize the transition model
    #def viz_transition_model(axes : list[plt.Axes]):
    #    x_slices = np.linspace(-5, 5, len(axes))

    #    for ax, x_slice in zip(axes, x_slices):
    #        ax.set_title(f"p(x' | x = {x_slice:.2f})")
    #        Xp = np.linspace(-5, 5, 100).reshape(-1, 1)

    #        true_p_xp = system.transition_likelihood(x_slice * np.ones_like(Xp), Xp)
    #        ax.plot(Xp, true_p_xp)

    #        mean, std = transition_model.predict(torch.tensor([x_slice], dtype=torch.float32))

    #        ax.plot(Xp, norm.pdf(Xp, mean, std))
    
    #fig = plt.figure()
    #ax = fig.gca()
    #viz_init_state_model(ax)

    #fig, axes = plt.subplots(1,7)
    #viz_transition_model(axes)

    gmms = [init_state_model]
    for k in range(1, timesteps):
        print("Propagating distribution: ", k)
        next_gmm = propagate_gpgmm_ekf(gmms[k-1], transition_model)
        gmms.append(next_gmm)

    plot_bounds = [-3.0, 3.0, -3.0, 3.0]
    def pdf_plotter(k : int):
        return grid_eval(lambda x : gmms[k].density(x), plot_bounds)

    interactive_state_distribution_plot_2D(traj_data, pdf_plotter)

    plt.show()
        
    