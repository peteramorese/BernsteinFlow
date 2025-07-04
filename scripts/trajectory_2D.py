from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product
from bernstein_flow.Propagate import propagate

from .Systems import Pendulum, sample_trajectories
from .Visualization import interactive_transformer_plot, interactive_state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal


DTYPE = torch.float64

if __name__ == "__main__":

    # System model
    system = Pendulum(dt=0.05, length=1.0, damp=1.1, covariance=0.005 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 2000

    # Number of training epochs
    n_epochs_init = 600
    n_epochs_tran = 50

    # Time horizon
    training_timesteps = 20
    timesteps = 20

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.1, 0.1]), cov = np.diag([0.10, 0.01]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Moment match the GDT to all of the data over the whole horizon
    #gdt = GaussianDistTransform(means=np.array([0.0, 0.0]), variances=[0.3, 1.0])
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[0.2, 0.0])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    #interactive_state_distribution_plot_2D(traj_data)
    interactive_state_distribution_plot_2D(u_traj_data)

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data


    #input("Continue to training...")

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=128, shuffle=True)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=64, shuffle=True)

    # Create initial state and transition models
    transformer_degrees = [10, 7]
    conditioner_degrees = [7, 7]
    init_state_model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, dtype=DTYPE)

    #transformer_degrees = [4, 1]
    #conditioner_degrees = [0, 1]
    transition_model = ConditionalBernsteinFlowModel(dim=dim, conditional_dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, dtype=DTYPE)

    print(f"Created init state model with {init_state_model.n_parameters()} parameters")
    print(f"Created transition model with {transition_model.n_parameters()} parameters")

    # Train the models
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-3)
    print("Training initial state model...")
    optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init)
    print("Done training initial state model \n")

    print("Training transition model...")
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-3)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran)
    print("Done training transition model \n")

    interactive_transformer_plot(transition_model, dim, cond_dim=dim, dtype=DTYPE)    

    #fig, axes = plt.subplots(2, 2)
    #fig.set_figheight(9)
    #fig.set_figwidth(9)
    #for ax in axes.flat:
    #    ax.set_aspect('equal')

    #plot_data(axes[0, 0], X0_data)
    #axes[0, 0].set_xlabel("x0")
    #axes[0, 0].set_ylabel("x1")
    #axes[0, 0].set_title("Data")

    #U_data = gdt.X_to_U(X0_data)

    #plot_data(axes[0, 1], U_data)
    #axes[0, 1].set_xlim((0, 1))
    #axes[0, 1].set_ylim((0, 1))
    #axes[0, 1].set_xlabel("u0")
    #axes[0, 1].set_ylabel("u1")
    #axes[0, 1].set_title("Erf-space Data")

    #model_x_eval = model_x_eval_fcn(init_state_model, gdt)
    #model_u_eval = model_u_eval_fcn(init_state_model)

    #bounds = axes[0, 0].get_xlim() + axes[0, 0].get_ylim()
    #X0, X1, Z_x = grid_eval(model_x_eval, bounds, resolution=100)
    #plot_density(axes[1, 0], X0, X1, Z_x)
    #axes[1, 0].set_xlabel("x0")
    #axes[1, 0].set_ylabel("x1")
    #axes[1, 0].set_title("Feature-space PDF")

    #u_bounds = [0.0, 1.0, 0.0, 1.0]
    #U0, U1, Z_u = grid_eval(model_u_eval, u_bounds, resolution=100)
    #plot_density(axes[1, 1], U0, U1, Z_u)
    #axes[1, 1].set_xlabel("u0")
    #axes[1, 1].set_ylabel("u1")
    #axes[1, 1].set_title("Erf-space PDF")



    # Compute the propagated polynomials
    p_init = poly_product(init_state_model.get_transformer_polynomials())
    p_transition = poly_product(transition_model.get_transformer_polynomials())
    density_polynomials = [p_init]
    for k in range(1, timesteps):
        p_curr = propagate([density_polynomials[k-1]], [p_transition])
        density_polynomials.append(p_curr)
        print(f"Computed p(x{k})")

    # Make pdf plotter for interactive vis
    u_bounds = [0.0, 1.0, 0.0, 1.0]
    def pdf_plotter(k : int):
        return grid_eval(lambda u : density_polynomials[k](u), u_bounds, dtype=DTYPE)

    #fig2 = plt.figure()
    #ax3d_x = fig2.add_subplot(111, projection='3d')
    #U0, U1, Z = pdf_plotter(2)
    #plot_density_2D_surface(ax3d_x, U0, U1, Z)

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    interactive_state_distribution_plot_2D(u_traj_data, pdf_plotter)

    





    ## Plot the density estimate
    #bounds = axes[0, 0].get_xlim() + axes[0, 0].get_ylim()
    #X0, X1, Z_x = evaluate_x_density_on_grid(model, gdt, bounds, resolution=100)
    #plot_density(axes[1, 0], X0, X1, Z_x)
    #axes[1, 0].set_xlabel("x0")
    #axes[1, 0].set_ylabel("x1")
    #axes[1, 0].set_title("Feature-space PDF")

    #U0, U1, Z_u = evaluate_u_density_on_grid(model, resolution=100)
    #plot_density(axes[1, 1], U0, U1, Z_u)
    #axes[1, 1].set_xlabel("u0")
    #axes[1, 1].set_ylabel("u1")
    #axes[1, 1].set_title("Erf-space PDF")


    #fig2 = plt.figure()
    #ax3d_x = fig2.add_subplot(121, projection='3d')
    #plot_density_surface(ax3d_x, X0, X1, Z_x)
    #ax3d_x.set_xlabel("x0")
    #ax3d_x.set_ylabel("x1")
    #ax3d_x.set_zlabel("p(x)")
    #ax3d_x.set_title("Feature-space PDF")

    #ax3d_u = fig2.add_subplot(122, projection='3d')
    #plot_density_surface(ax3d_u, X0, X1, Z_u)
    #ax3d_u.set_xlabel("u0")
    #ax3d_u.set_ylabel("u1")
    #ax3d_u.set_zlabel("p(u)")
    #ax3d_u.set_title("Erf-space PDF")

    ## Plot transformers
    #fig3, axes, sliders = create_interactive_transformer_plot(model, dim)

    plt.show()
