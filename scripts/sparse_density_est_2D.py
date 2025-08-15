from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.SparseModel import SparseBetaModel, optimize
from bernstein_flow.Tools import grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_product_bernstein_direct, stable_split_factors, split_factor_poly_product, mc_auc, Polynomial, Basis

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
    n_data = 2000

    # Number of training epochs
    n_epochs = 100

    #gdt = GaussianDistTransform(mean=[0.5, 0.25], variances=[1.0, 0.5])

    #means = [[-2, -2], [2, 2], [-2, 2]]
    #covariances = [torch.eye(dim)*1.5, torch.eye(dim)*0.5, torch.eye(dim)*0.5]
    #X_data = sample_modal_gaussian(n_data, means=means, covariances=covariances, weights=[.4, .3, 0.3])
    #X_data_test = sample_modal_gaussian(n_data, means=means, covariances=covariances, weights=[.4, .3, 0.3])

    X_data, _ = make_moons(n_data, noise=0.15)
    X_data_test, _ = make_moons(n_data, noise=0.15)
    #X_data, _ = make_circles(n_data, noise=0.1, factor=0.5)
    #X_data_test, _ = make_circles(n_data, noise=0.1, factor=0.5)

    #gdt = GaussianDistTransform.moment_match_data(X_data, variance_pads=[0.0] * dim)

    #gdt = GaussianDistTransform(means=[0.0, 2.0], variances=[2.0, 2.0])
    gdt = GaussianDistTransform.moment_match_data(X_data, variance_pads=[0.5] * dim)
    U_data = gdt.X_to_U(X_data)
    U_data_test = gdt.X_to_U(X_data_test)

    #fig, axes = plt.subplots(2, 2)
    figs = [plt.figure() for _ in range(4)]
    axes = [fig.gca() for fig in figs]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plot_data_2D(axes[0], X_data)
    #axes[0].set_xlabel("x0")
    #axes[0].set_ylabel("x1")
    #axes[0].set_title("Data")


    plot_data_2D(axes[1], U_data)
    axes[1].set_xlim((0, 1))
    axes[1].set_ylim((0, 1))
    #axes[1].set_xlabel("u0")
    #axes[1].set_ylabel("u1")
    #axes[1].set_title("Erf-space Data")

    #plt.show(block=False)
    #input("Continue to training...")

    # Create data loader
    U_data_torch = torch.tensor(U_data, dtype=DTYPE)
    U_data_test_torch = torch.tensor(U_data_test, dtype=DTYPE)
    dataset = TensorDataset(U_data_torch)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create model
    model = SparseBetaModel(dim=dim, n_components=60, max_degree=60)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimize(model, dataloader, optimizer, epochs=n_epochs)
    print("Test NLL: ", model.nll_loss(U_data_test_torch))
    #optimize(model, dataloader, optimizer, epochs=n_epochs, hard_constraint=True)

    # Plot the density estimate
    model_x_eval = model_x_eval_fcn(model, gdt, dtype=DTYPE)
    model_u_eval = model_u_eval_fcn(model)


    bounds = axes[0].get_xlim() + axes[0].get_ylim()
    X0, X1, Z_x = grid_eval(model_x_eval, bounds, resolution=100, dtype=DTYPE)
    plot_density_2D(axes[2], X0, X1, Z_x)
    #axes[2].set_xlabel("x0")
    #axes[2].set_ylabel("x1")
    #axes[2].set_title("Feature-space PDF")

    u_bounds = [0.0, 1.0, 0.0, 1.0]
    U0, U1, Z_u = grid_eval(model_u_eval, u_bounds, resolution=100, dtype=DTYPE)
    plot_density_2D(axes[3], U0, U1, Z_u)
    #axes[3].set_xlabel("u0")
    #axes[3].set_ylabel("u1")
    #axes[3].set_title("Erf-space PDF")


    sparse_bernie = model.create_sparse_bern_poly()
    f = plt.figure()
    ax = f.gca()
    ax.set_aspect('equal')
    U0, U1, Z_poly = grid_eval(sparse_bernie, u_bounds, resolution=100, dtype=DTYPE)
    plot_density_2D(ax, U0, U1, Z_poly)

    with torch.no_grad():
        A_bnd, B_bnd, norm_weights = model.get_constrained_parameters()
        print("Model alphas: \n", A_bnd)
        print("Model Coeffs: \n", norm_weights)

    print("Bernie idx   : \n", sparse_bernie.idx)
    print("Bernie coeffs: \n", sparse_bernie.coeffs)
    auc = mc_auc(sparse_bernie, n_samples=10000)
    print("Bernie auc: ", auc)
    #for i, fig in enumerate(figs):
    #    fig.savefig(f"./figures/fig_{i}.pdf")

    plt.show()
