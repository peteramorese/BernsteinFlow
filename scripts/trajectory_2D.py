from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct
from bernstein_flow.Propagate import propagate_bfm

from .Systems import VanDerPol, Pendulum, sample_trajectories
from .Visualization import interactive_transformer_plot, state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

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
    #system = Pendulum(dt=0.05, length=1.0, damp=1.1, covariance=0.005 * np.eye(2))
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 2000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 50

    # Time horizon
    training_timesteps = 10
    timesteps = 10

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Moment match the GDT to all of the data over the whole horizon
    #gdt = GaussianDistTransform(means=np.array([0.0, 0.0]), variances=[0.3, 1.0])
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[2.2, 2.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    #state_distribution_plot_2D(traj_data)
    #state_distribution_plot_2D(u_traj_data)

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data


    #input("Continue to training...")
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("device: ", device)

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=1024, shuffle=True, pin_memory=True)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=1024, shuffle=True, pin_memory=True)

    # Create initial state and transition models
    degrees = [15, 15]
    deg_incr = [10, 10]
    init_state_model = BernsteinFlowModel(dim=dim, 
                                          degrees=degrees, 
                                          dtype=DTYPE, 
                                          device=device, 
                                          deg_incr=deg_incr)

    print(f"Created init state model with {init_state_model.n_parameters()} parameters")

    # Train the Init model
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    print("Training initial state model...")
    optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init, proj_tol=1e-4, proj_min_thresh=1e-3)
    print("Done training initial state model \n")
    init_state_model = init_state_model.to(device=cpu_device)

    degrees = [20, 20]
    cond_degrees = [10, 10]
    deg_incr = [10, 10]
    cond_deg_incr = [0, 0]
    transition_model = ConditionalBernsteinFlowModel(dim=dim, 
                                                     conditional_dim=dim, 
                                                     degrees=degrees, 
                                                     conditional_degrees=cond_degrees, 
                                                     dtype=DTYPE, 
                                                     device=device, 
                                                     deg_incr=deg_incr, 
                                                     cond_deg_incr=cond_deg_incr)

    print(f"Created transition model with {transition_model.n_parameters()} parameters")

    print("Training transition model...")
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran, proj_tol=1e-4, proj_min_thresh=1e-3)
    print("Done training transition model \n")

    #interactive_transformer_plot(transition_model, dim, cond_dim=dim, dtype=DTYPE)    

    #c_alphas = [transition_model.get_constrained_parameters(i, layer_i=0) for i in range(dim)]
    #print("c alpha shapes: ", [alpha.shape for alpha in c_alphas])
    #c_alphas_min_max = [(torch.min(alpha).item(), torch.max(alpha).item()) for alpha in c_alphas]
    #print("trans model coeff min and max: ", c_alphas_min_max)

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
    init_model_tfs = init_state_model.get_density_factor_polys(dtype=np.float128)
    trans_model_tfs = transition_model.get_density_factor_polys(dtype=np.float128)

    p_init = poly_product_bernstein_direct(init_model_tfs)
    p_transition = poly_product_bernstein_direct(trans_model_tfs)
    print("p_init shape: ", p_init.shape(), " dtype: ", p_init.coeffs.dtype)
    print("p_tran shape: ", p_transition.shape())
    density_polynomials = [p_init]
    for k in range(1, timesteps):
        p_curr = propagate_bfm([density_polynomials[k-1]], [p_transition])
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
    state_dist_fig, _ = state_distribution_plot_2D(u_traj_data, pdf_plotter, interactive=False, bounds=u_bounds)

    





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

    #plt.show()

    state_dist_fig.savefig("./figures/trajectory_2D.png")

