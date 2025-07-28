from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct, marginal
from bernstein_flow.Propagate import propagate_bfm

from .Systems import DisturbedDubinsCar, sample_trajectories
from .Visualization import interactive_transformer_plot, state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal, beta


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

    x_bounds = [-1.0, 8.0, -5.0, 5.0]
    #x_bounds = [0.0, 10.0, 0.0, 10.0]

    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[2.2, 2.2, 2.2])
    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]

    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(u_traj_data[:training_timesteps])

    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data

    u_traj_data_xy_marginal = [timestep_data[:, :2] for timestep_data in u_traj_data]
    state_distribution_plot_2D(u_traj_data_xy_marginal, interactive=True, bounds=[0.0, 1.0, 0.0, 1.0])
    plt.show()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=128, shuffle=True, pin_memory=True)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=1024, shuffle=True, pin_memory=True)

    # Create initial state and transition models
    transformer_degrees = [6, 5, 5]
    conditioner_degrees = [4, 4, 4]
    cond_deg_incr = None #[10] * len(conditioner_degrees)
    init_state_model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, dtype=DTYPE, conditioner_deg_incr=cond_deg_incr, device=device)

    cond_deg_incr = None #[20] * len(conditioner_degrees)
    transition_model = ConditionalBernsteinFlowModel(dim=dim, conditional_dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, dtype=DTYPE, conditioner_deg_incr=cond_deg_incr, device=device)

    # Train the models
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    print("Training initial state model...")
    optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init)
    print("Done training initial state model \n")

    print("Training transition model...")
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-1)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran)
    print("Done training transition model \n")


    # Propagate
    init_model_tfs = init_state_model.get_density_factor_polys(dtype=np.float128)
    trans_model_tfs = transition_model.get_density_factor_polys(dtype=np.float128)

    p_init = poly_product_bernstein_direct(init_model_tfs)
    p_transition = poly_product_bernstein_direct(trans_model_tfs)
    density_polynomials = [p_init]
    xy_marginal_density_polys = [marginal(p_init, {2}, True)]
    for k in range(1, timesteps):
        p_curr = propagate_bfm([density_polynomials[k-1]], [p_transition])
        p_curr_xy_marg = marginal(p_curr, {2}, True)
        density_polynomials.append(p_curr)
        xy_marginal_density_polys.append(p_curr_xy_marg)
        print(f"Computed p(x{k})")


    # Make pdf plotter for interactive vis
    u_bounds = [0.0, 1.0, 0.0, 1.0]
    def pdf_plotter(k : int):
        return grid_eval(lambda u : xy_marginal_density_polys[k](u), u_bounds, dtype=DTYPE)

    #fig2 = plt.figure()
    #ax3d_x = fig2.add_subplot(111, projection='3d')
    #U0, U1, Z = pdf_plotter(2)
    #plot_density_2D_surface(ax3d_x, U0, U1, Z)

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    state_distribution_plot_2D(u_traj_data, pdf_plotter, interactive=True, bounds=u_bounds)