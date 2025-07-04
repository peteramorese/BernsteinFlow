from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product
from bernstein_flow.Propagate import propagate

from .Systems import CubicMap, sample_trajectories, sample_io_pairs
from .Visualization import interactive_transformer_plot, interactive_state_distribution_plot_1D, plot_density_1D, plot_data_1D, plot_data_2D, plot_density_2D_surface, plot_density_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from scipy.integrate import quad


DTYPE = torch.float64

if __name__ == "__main__":

    # System model
    system = CubicMap(dt=0.01, alpha=0.5, variance=0.5)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 2000

    # Number of training epochs
    n_epochs_init = 500
    n_epochs_tran = 50

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    # Propagate using
    np_prop = True

    def init_state_sampler():
        mode = np.random.randint(0, 2)
        #return float(mode) * norm.rvs(loc=np.array([1.0]), scale = 1.2) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.0]), scale = 1.2)
        return float(mode) * norm.rvs(loc=np.array([1.5]), scale = 0.5) + (1.0 - float(mode)) * norm.rvs(loc=np.array([-1.5]), scale = 0.5)


    io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-3.0], region_uppers=[3.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    interactive_state_distribution_plot_1D(traj_data, bins=60)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data))

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

    fig = plt.figure()
    plot_data_2D(fig.gca(), Up_io_data)
    #plt.show()

    #input("Continue to training...")

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=128, shuffle=True)

    Up_data_torch = torch.tensor(Up_io_data, dtype=DTYPE)
    #Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=64, shuffle=True)

    # Create initial state and transition models
    transformer_degrees = [25]
    conditioner_degrees = [25]
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


    #cond_u_fcn = model_u_eval_fcn(transition_model)
    #U0, U1, Z = grid_eval(cond_u_fcn, [0.0, 1.0, 0.0, 1.0])
    #fig2 = plt.figure()
    #ax3d_u = fig2.add_subplot(121, projection='3d')
    #plot_density_2D_surface(ax3d_u, U0, U1, Z)
    #ax3d_u.set_xlabel("u0")
    #ax3d_u.set_ylabel("u1")
    #ax3d_u.set_zlabel("p(u)")
    #ax3d_u.set_title("Erf-space PDF")
    #plt.show()

    #interactive_transformer_plot(transition_model, dim, cond_dim=dim, dtype=DTYPE)    
    #input("...")

    #u_curr = torch.linspace(0.0, 1.0, 100, dtype=DTYPE)
    #for u in u_curr:
    #    def true_xp_density(xp):
    #        return torch.from_numpy(system.transition_likelihood(x_slice.numpy() * np.ones_like(xp), xp))
    #def system_trans_xp_density(x_xp : np.ndarray):
    #    return torch.from_numpy(system.transition_likelihood(x_xp[:, 0], x_xp[:, 1]))
    #U, Up, Z = grid_eval(system_trans_density, [0.0, 1.0, 0.0, 1.0], resolution=100)
    #fig, axes = plt.subplots(1, 2)
    #plot_density_2D(axes[0], U, Up, Z)

    #model_u_eval = model_u_eval_fcn(transition_model)
    #U, Up, Z = grid_eval(model_u_eval, [0.0, 1.0, 0.0, 1.0], resolution=100, dtype=DTYPE)
    #plot_density_2D(axes[1], U, Up, Z)
    #plt.show()

    init_model_tfs = init_state_model.get_transformer_polynomials()

    p_init = poly_product(init_state_model.get_transformer_polynomials())

    n_slices = 9 
    x_slices = torch.linspace(0.1, 0.9, 2 * n_slices, dtype=DTYPE)
    fig, axes = plt.subplots(2, n_slices)
    axes = axes.flatten()
    init_densities = p_init(x_slices.unsqueeze(-1))
    for x_slice, ax, init_density in zip(x_slices, axes, init_densities):
        def trans_slice(xp):
            transition_model.eval()
            xp = xp.view(-1, 1)
            x_cat = torch.hstack([x_slice * torch.ones(xp.shape), xp])
            return transition_model(x_cat).squeeze(-1).detach()

        def trans_slice_single_pt(xp_pt):
            transition_model.eval()
            x_cat = torch.tensor([[xp_pt, x_slice]])
            return transition_model(x_cat).squeeze(-1).detach()


        #auc, error = quad(trans_slice_single_pt, 0.0, 1.0)
        #print(f"x = {x_slice} AUC: ", auc)
        up_samples = torch.rand(1000, 1, dtype=DTYPE)
        pdf_samples = trans_slice(up_samples)
        print(f"u = {x_slice} mc AUC: ", torch.mean(pdf_samples))


        Y = torch.linspace(0.01, 0.99, 100, requires_grad=False, dtype=DTYPE)
        Z = trans_slice(Y)
        plot_density_1D(ax, Y, Z)
        ax.set_title(f"p(u' | u={x_slice})")

        
        def true_xp_density(xp):
            return system.transition_likelihood(x_slice.numpy() * np.ones_like(xp), xp)
        Z_true = gdt.u_density(Y.numpy().reshape(-1, 1), true_xp_density)
        ax.plot(Y, Z_true, color='green', linestyle='--')

        up_samples = np.random.rand(10000, 1)
        pdf_samples = gdt.u_density(up_samples, true_xp_density)
        print(f"u = {x_slice} mc true AUC: ", np.mean(pdf_samples))

        #Z_true_weighted = init_density.numpy() * Z_true
        #ax.plot(Y, Z_true_weighted, color='blue', linestyle=':')
    #plt.show()


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
    init_model_tfs = init_state_model.get_transformer_polynomials()
    trans_model_tfs = transition_model.get_transformer_polynomials()

    for i in range(len(init_model_tfs)):
        init_model_tfs[i].coeffs = init_model_tfs[i].coeffs.astype(np.float128)
        trans_model_tfs[i].coeffs = trans_model_tfs[i].coeffs.astype(np.float128)

    p_init = poly_product(init_model_tfs)
    p_transition = poly_product(trans_model_tfs)
    density_polynomials = [p_init]
    for k in range(1, timesteps):
        p_curr = propagate([density_polynomials[k-1]], [p_transition], mag_range=3.0)
        density_polynomials.append(p_curr)
        print(f"Computed p(x{k})")

        u_samples = np.random.rand(10000, 1)
        u_samples = u_samples.astype(np.float128)
        pdf_samples = p_curr(u_samples)
        print(f"p(x{k}) mc AUC: ", np.mean(pdf_samples))

    # Make pdf plotter for interactive vis
    u_bounds = [0.0, 1.0, 0.0, 1.0]
    def pdf_plotter(k : int):
        U = np.linspace(0.0, 1.0, 100, dtype=np.float128)
        U = np.expand_dims(U, axis=1)
        Z = density_polynomials[k](U)
        return U, Z


    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
    interactive_state_distribution_plot_1D(u_traj_data, pdf_plotter, bins=60)

    





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
