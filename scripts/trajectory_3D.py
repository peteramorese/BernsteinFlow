from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct
from bernstein_flow.Propagate import propagate_bfm

from .Systems import DisturbedDubinsCar, sample_trajectories
from .Visualization import interactive_transformer_plot, state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm, multivariate_normal, beta


DTYPE = torch.float64

if __name__ == "__main__":

    # System model
    #system = Pendulum(dt=0.05, length=1.0, damp=1.1, covariance=0.005 * np.eye(2))
    system = DisturbedDubinsCar(dt=0.5, track_heading_function=lambda x : np.sin(1.2*x), noise_magnitude=0.3)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 2000

    # Number of training epochs
    n_epochs_init = 500
    n_epochs_tran = 50

    # Time horizon
    training_timesteps = 20
    timesteps = 20

    def init_state_sampler():
        xy_init = multivariate_normal.rvs(mean=np.array([0.0, 0.0]), cov = np.diag([0.2, 0.2]))
        theta_init = beta.rvs(a=16, b=16, loc=-0.5, scale=1)
        return np.append(xy_init, theta_init)

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    x_bounds = [-1.0, 6.0, -1.0, 3.0]
    #x_bounds = [0.0, 10.0, 0.0, 10.0]

    traj_data_xy_marginal = [timestep_data[:, :2] for timestep_data in traj_data]
    state_distribution_plot_2D(traj_data_xy_marginal, interactive=True, bounds=x_bounds)
    plt.show()
