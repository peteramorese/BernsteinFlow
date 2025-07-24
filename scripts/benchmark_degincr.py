from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, avg_log_likelihood
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct, mc_auc
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
import time
import os
import json
from datetime import datetime

from sklearn.datasets import make_moons, make_circles

def get_date_time_str():
    return datetime.now().strftime("%Yy_%mm_%dd_%Hh_%Mm_%Ss")

def save_figure_bundle(fig_bundle, dir):
    os.makedirs(dir, exist_ok=True)
    for k, fig in enumerate(fig_bundle):
        print("saving to: ", dir + f"/k_{k}.pdf")
        fig.savefig(dir + f"/k_{k}.pdf")


DTYPE = torch.float64

if __name__ == "__main__":

    np.random.seed(42)

    benchmark_fields = dict()

    # Dimension
    dim = 2

    # Number of data points
    n_data = 2000

    # Number of training epochs
    n_epochs = 500

    X_data, _ = make_moons(n_data, noise=0.05)
    test_X_data, _ = make_moons(n_data, noise=0.05)
    #X_data, _ = make_circles(n_data, noise=0.1, factor=0.4)

    gdt = GaussianDistTransform.moment_match_data(X_data, variance_pads=[0.5] * dim)
    
    U_data = gdt.X_to_U(X_data)

    #fig, axes = plt.subplots(2, 2)
    #fig.set_figheight(9)
    #fig.set_figwidth(9)
    #for ax in axes.flat:
    #    ax.set_aspect('equal')
    #plot_data_2D(axes[0, 0], X_data)
    #axes[0, 0].set_xlabel("x0")
    #axes[0, 0].set_ylabel("x1")
    #axes[0, 0].set_title("Data")
    #plot_data_2D(axes[0, 1], U_data)
    #axes[0, 1].set_xlim((0, 1))
    #axes[0, 1].set_ylim((0, 1))
    #axes[0, 1].set_xlabel("u0")
    #axes[0, 1].set_ylabel("u1")
    #axes[0, 1].set_title("Erf-space Data")
    #plt.show()

    # Create data loader
    U_data_torch = torch.tensor(U_data, dtype=DTYPE)
    test_X_data_torch = torch.tensor(test_X_data, dtype=DTYPE)
    dataset = TensorDataset(U_data_torch)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model
    degree_increases = [0, 10, 20, 40, 80, 160, 300]
    transformer_degrees = [8, 6]
    conditioner_degrees = [6, 6]
    allhs = []
    training_times = []
    for incr in degree_increases: 
        cond_deg_incr = [incr] * len(conditioner_degrees)
        model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, layers=2, dtype=DTYPE, conditioner_deg_incr=cond_deg_incr)

        # Train
        start = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimize(model, dataloader, optimizer, epochs=n_epochs)
        training_times.append(time.time() - start)

        # Evaluate
        u_eval_f = model_x_eval_fcn(model, gdt, dtype=torch.float64)
        allh = avg_log_likelihood(test_X_data.astype(np.float32), u_eval_f)
        print(f"Average log likelihood using deg incr {incr}: ", allh)
        allhs.append(allh)

    # Write down system properties
    curr_date_time = get_date_time_str()
    benchmark_fields["datetime"] = curr_date_time 
    benchmark_fields["dimension"] = dim
    benchmark_fields["n_data"] = n_data
    benchmark_fields["n_epochs"] = n_epochs
    benchmark_fields["transformer_degrees"] = transformer_degrees
    benchmark_fields["conditioner_degrees"] = conditioner_degrees
    benchmark_fields["cond_deg_incrs"] = degree_increases
    benchmark_fields["training_times"] = training_times
    benchmark_fields["average_log_likelihood"] = allhs



    experiment_name = f"degincr_{curr_date_time}"

    with open(f"./benchmarks/{experiment_name}.json", "w") as f:
        json.dump(benchmark_fields, f, indent=4)
