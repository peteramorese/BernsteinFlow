from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, optimize
from bernstein_flow.Tools import grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_product_bernstein_direct

from .TestDataSets import sample_modal_gaussian
from .Visualization import interactive_transformer_plot, plot_density_1D, plot_data_1D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

from sklearn.datasets import make_moons, make_circles

DTYPE = torch.float64

if __name__ == "__main__":

    # Dimension
    dim = 1

    # Number of data points
    n_data = 5000

    # Number of training epochs
    n_epochs = 10

    #gdt = GaussianDistTransform(mean=[0.5, 0.25], variances=[1.0, 0.5])

    means = [[-1.5], [1.5]]
    covariances = [torch.eye(dim)*1.5, torch.eye(dim)*0.5]
    X_data = sample_modal_gaussian(n_data, means=means, covariances=covariances, weights=[.3, .7])


    gdt = GaussianDistTransform.moment_match_data(X_data, variance_pads=[0.5] * dim)

    #fig, axes = plt.subplots(1, 2)
    #fig.set_figheight(9)
    #fig.set_figwidth(9)
    #for ax in axes.flat:
        #ax.set_aspect('equal')

    #plot_data_1D(axes[0], X_data, bins=30)
    #axes[0].set_xlabel("x0")
    #axes[0].set_ylabel("density")
    #axes[0].set_title("Data")

    U_data = gdt.X_to_U(X_data)

    #plot_data_1D(axes[1], U_data, bins=30)
    #axes[1].set_xlabel("u0")
    #axes[1].set_ylabel("u-denstiy")
    #axes[1].set_title("Erf-space Data")

    #plt.show(block=False)
    #input("Continue to training...")

    # Create data loader
    U_data_torch = torch.tensor(U_data, dtype=DTYPE)
    dataset = TensorDataset(U_data_torch)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model
    transformer_degrees = [4]
    conditioner_degrees = [4]
    cond_deg_incr = [60] * len(conditioner_degrees)
    model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, layers=2, dtype=DTYPE, conditioner_deg_incr=cond_deg_incr)

    print("Number of parameters in model: ", model.n_parameters())

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimize(model, dataloader, optimizer, epochs=n_epochs)

    # Plot the density estimate
    model_x_eval = model_x_eval_fcn(model, gdt, dtype=DTYPE)
    model_u_eval = model_u_eval_fcn(model)


    meval = model_u_eval_fcn(model)
    fig, axes = plt.subplots(1, 2)
    U = np.linspace(0.0, 1.0, 100)
    Z = meval(U.reshape(-1, 1))
    plot_density_1D(axes[0], U, Z)


    p_list = model.get_density_factor_polys(dtype=np.float128)
    print("n factors: ", len(p_list))
    p_prod = poly_product_bernstein_direct(p_list)
    print("p_prod shape: ", p_prod.shape())

    Z_poly = p_prod(U.reshape(-1, 1))
    plot_density_1D(axes[1], U, Z_poly)



    plt.show()
