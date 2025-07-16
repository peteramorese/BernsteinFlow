from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, optimize
from bernstein_flow.Tools import grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_product_bernstein_direct

from .TestDataSets import sample_modal_gaussian
from .Visualization import interactive_transformer_plot, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons, make_circles

DTYPE = torch.float64

if __name__ == "__main__":

    # Dimension
    dim = 2

    # Number of data points
    n_data = 5000

    # Number of training epochs
    n_epochs = 10

    #gdt = GaussianDistTransform(mean=[0.5, 0.25], variances=[1.0, 0.5])

    means = [[-1.5, -1.5], [1.5, 1.5]]
    covariances = [torch.eye(dim)*1.5, torch.eye(dim)*0.5]
    #X_data = sample_modal_gaussian(n_data, means=means, covariances=covariances, weights=[.3, .7])

    X_data, _ = make_moons(n_data, noise=0.05)
    #X_data, _ = make_circles(n_data, noise=0.1, factor=0.4)

    gdt = GaussianDistTransform.moment_match_data(X_data, variance_pads=[0.5] * dim)

    fig, axes = plt.subplots(2, 2)
    fig.set_figheight(9)
    fig.set_figwidth(9)
    for ax in axes.flat:
        ax.set_aspect('equal')

    plot_data_2D(axes[0, 0], X_data)
    axes[0, 0].set_xlabel("x0")
    axes[0, 0].set_ylabel("x1")
    axes[0, 0].set_title("Data")

    U_data = gdt.X_to_U(X_data)

    plot_data_2D(axes[0, 1], U_data)
    axes[0, 1].set_xlim((0, 1))
    axes[0, 1].set_ylim((0, 1))
    axes[0, 1].set_xlabel("u0")
    axes[0, 1].set_ylabel("u1")
    axes[0, 1].set_title("Erf-space Data")

    plt.show(block=False)
    input("Continue to training...")

    # Create data loader
    U_data_torch = torch.tensor(U_data, dtype=DTYPE)
    dataset = TensorDataset(U_data_torch)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model
    transformer_degrees = [5, 5]
    conditioner_degrees = [5, 5]
    cond_deg_incr = [60] * len(conditioner_degrees)
    model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, layers=2, dtype=DTYPE, conditioner_deg_incr=cond_deg_incr)

    print("Number of parameters in model: ", model.n_parameters())

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimize(model, dataloader, optimizer, epochs=n_epochs)

    # Plot the density estimate
    model_x_eval = model_x_eval_fcn(model, gdt, dtype=DTYPE)
    model_u_eval = model_u_eval_fcn(model)

    bounds = axes[0, 0].get_xlim() + axes[0, 0].get_ylim()
    X0, X1, Z_x = grid_eval(model_x_eval, bounds, resolution=100, dtype=DTYPE)
    plot_density_2D(axes[1, 0], X0, X1, Z_x)
    axes[1, 0].set_xlabel("x0")
    axes[1, 0].set_ylabel("x1")
    axes[1, 0].set_title("Feature-space PDF")

    u_bounds = [0.0, 1.0, 0.0, 1.0]
    U0, U1, Z_u = grid_eval(model_u_eval, u_bounds, resolution=100, dtype=DTYPE)
    plot_density_2D(axes[1, 1], U0, U1, Z_u)
    axes[1, 1].set_xlabel("u0")
    axes[1, 1].set_ylabel("u1")
    axes[1, 1].set_title("Erf-space PDF")


    fig2 = plt.figure()
    ax3d_x = fig2.add_subplot(131, projection='3d')
    plot_density_2D_surface(ax3d_x, X0, X1, Z_x)
    ax3d_x.set_xlabel("x0")
    ax3d_x.set_ylabel("x1")
    ax3d_x.set_zlabel("p(x)")
    ax3d_x.set_title("Feature-space PDF")

    ax3d_u = fig2.add_subplot(132, projection='3d')
    plot_density_2D_surface(ax3d_u, X0, X1, Z_u)
    ax3d_u.set_xlabel("u0")
    ax3d_u.set_ylabel("u1")
    ax3d_u.set_zlabel("p(u)")
    ax3d_u.set_title("Erf-space PDF")


    p_list = model.get_density_factor_polys(dtype=np.float128)
    p_prod = poly_product_bernstein_direct(p_list)
    print("p_prod shape: ", p_prod.shape())

    u_bounds = [0.0, 1.0, 0.0, 1.0]
    ax3d_u = fig2.add_subplot(133, projection='3d')
    plot_density_2D_surface(ax3d_u, *grid_eval(lambda u : p_prod(u), u_bounds, dtype=np.float128))
    ax3d_u.set_xlabel("u0")
    ax3d_u.set_ylabel("u1")
    ax3d_u.set_zlabel("p(u)")
    ax3d_u.set_title("Composed Polynomial Erf-space PDF")

    # Plot transformers
    #fig3, axes, sliders = interactive_transformer_plot(model, dim, dtype=DTYPE)



    plt.show()
