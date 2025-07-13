from bernstein_flow.GPGMM import fit_gmm, fit_gp
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn

from .Systems import CubicMap, sample_trajectories, sample_io_pairs
from .Visualization import interactive_transformer_plot, interactive_state_distribution_plot_1D, plot_density_1D, plot_data_1D, plot_data_2D, plot_density_2D_surface, plot_density_2D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from scipy.integrate import quad

if __name__ == "__main__":

    # System model
    system = CubicMap(dt=0.01, alpha=0.5, variance=0.5)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 2000

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        mode = np.random.randint(0, 2)
        #return float(mode) * norm.rvs(loc=np.array([1.0]), scale = 1.2) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.0]), scale = 1.2)
        return float(mode) * norm.rvs(loc=np.array([1.5]), scale = 0.5) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.5]), scale = 0.5)

    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-10.0], region_uppers=[10.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    interactive_state_distribution_plot_1D(traj_data, bins=60)

    # Create the data matrices for training
    X0_data = traj_data[0]
    X_data, Xp_data = create_transition_data_matrix(traj_data[:training_timesteps], separate=True)

    init_state_model = fit_gmm(X0_data)
    transtion_model = fit_gp(X_data, Xp_data)