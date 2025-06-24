from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, optimize
from bernstein_flow.Tools import grid_eval, model_u_eval_fcn, model_x_eval_fcn

from .TestDataSets import sample_modal_gaussian, plot_data
from .Visualization import create_interactive_transformer_plot, plot_density, plot_density_surface

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset

#from sklearn.datasets import make_moons, make_circles


if __name__ == "__main__":

    # Dimension
    dim = 2

    # Number of data points
    n_data = 5000

    # Number of training epochs
    n_epochs = 50

    #gdt = GaussianDistTransform(mean=[0.5, 0.25], variances=[1.0, 0.5])

    means = [[-1.5, -1.5], [1.5, 1.5]]
    covariances = [torch.eye(dim)*1.5, torch.eye(dim)*0.5]
    X_data = sample_modal_gaussian(n_data, means=means, covariances=covariances, weights=[.3, .7])

    #X_data, _ = make_moons(n_data, noise=0.05)
    #X_data, _ = make_circles(n_data, noise=0.1, factor=0.4)

    gdt = GaussianDistTransform.moment_match_data(X_data)

    fig, axes = plt.subplots(2, 2)
    fig.set_figheight(9)
    fig.set_figwidth(9)
    for ax in axes.flat:
        ax.set_aspect('equal')

    plot_data(axes[0, 0], X_data)
    axes[0, 0].set_xlabel("x0")
    axes[0, 0].set_ylabel("x1")
    axes[0, 0].set_title("Data")

    U_data = gdt.X_to_U(X_data)

    plot_data(axes[0, 1], U_data)
    axes[0, 1].set_xlim((0, 1))
    axes[0, 1].set_ylim((0, 1))
    axes[0, 1].set_xlabel("u0")
    axes[0, 1].set_ylabel("u1")
    axes[0, 1].set_title("Erf-space Data")

    plt.show(block=False)
    input("Continue to training...")

    # Create data loader
    U_data_torch = torch.tensor(U_data, dtype=torch.float32)
    dataset = TensorDataset(U_data_torch)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create model
    transformer_degrees = [3, 2]
    conditioner_degrees = [2, 3]
    model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees)

    print("Number of parameters in model: ", model.n_parameters())

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimize(model, dataloader, optimizer, epochs=n_epochs)

    # Plot the density estimate
    model_x_eval = model_x_eval_fcn(model, gdt)
    model_u_eval = model_u_eval_fcn(model)

    bounds = axes[0, 0].get_xlim() + axes[0, 0].get_ylim()
    X0, X1, Z_x = grid_eval(model_x_eval, bounds, resolution=100)
    plot_density(axes[1, 0], X0, X1, Z_x)
    axes[1, 0].set_xlabel("x0")
    axes[1, 0].set_ylabel("x1")
    axes[1, 0].set_title("Feature-space PDF")

    U0, U1, Z_u = grid_eval(model_u_eval, resolution=100)
    plot_density(axes[1, 1], U0, U1, Z_u)
    axes[1, 1].set_xlabel("u0")
    axes[1, 1].set_ylabel("u1")
    axes[1, 1].set_title("Erf-space PDF")


    fig2 = plt.figure()
    ax3d_x = fig2.add_subplot(121, projection='3d')
    plot_density_surface(ax3d_x, X0, X1, Z_x)
    ax3d_x.set_xlabel("x0")
    ax3d_x.set_ylabel("x1")
    ax3d_x.set_zlabel("p(x)")
    ax3d_x.set_title("Feature-space PDF")

    ax3d_u = fig2.add_subplot(122, projection='3d')
    plot_density_surface(ax3d_u, X0, X1, Z_u)
    ax3d_u.set_xlabel("u0")
    ax3d_u.set_ylabel("u1")
    ax3d_u.set_zlabel("p(u)")
    ax3d_u.set_title("Erf-space PDF")

    # Plot transformers
    fig3, axes, sliders = create_interactive_transformer_plot(model, dim)

    #print("UNCONSTRAINED Model params:")
    #for i in range(dim):
    #    print(model.A[i])

    #print("Model params:")
    #for i in range(dim):
    #    print(model.get_constrained_parameters(i))


    plt.show()
