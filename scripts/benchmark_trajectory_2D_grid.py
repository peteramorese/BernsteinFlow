from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.GPGMM import GMModel, GPModel, fit_gmm, fit_gp
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, avg_log_likelihood
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct
from bernstein_flow.Propagate import propagate_grid_gmm

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

    max_mixands = 10000
    max_time = 2000

    # System model
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 500

    # Number of training epochs
    n_epochs_tran = 200

    # Time horizon
    training_timesteps = 10
    timesteps = 10

    # Grid Resolution
    grid_resolution = 30

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)
    test_traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    x_bounds = [-5.0, 5.0, -5.0, 5.0]

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])
    Xp_data_torch = torch.from_numpy(Xp_data)

    print("Fitting initial state model...")
    start = time.time()
    init_state_model = fit_gmm(X0_data, n_components=10, covariance_type='diag')
    init_train_time = time.time() - start

    print("Fitting transition state model...")
    start = time.time()
    transition_model = fit_gp(Xp=Xp_data_torch[:, dim:], X=Xp_data_torch[:, :dim], num_epochs=n_epochs_tran, dtype=torch.float64)
    tran_train_time = time.time() - start
    print("Done!")


    curr_date_time = get_date_time_str()

    density_gmms = [init_state_model]
    n_mixands = [init_state_model.n_mixands()]
    prop_times = []
    allhs = []
    for k in range(1, timesteps):
        start = time.time()
        p_curr = propagate_grid_gmm(density_gmms[k-1], transition_model, bounds=x_bounds, resolution=grid_resolution)
        prop_times.append(time.time() - start)
        print(f"Computed p(x{k}) in {prop_times[-1]:.2f} seconds. Number of components: ", p_curr.n_mixands())

        # Compute the log likelihood
        allh = avg_log_likelihood(test_traj_data[k], lambda x : p_curr.density(x))
        print(f" - Average log likelihood: {allh:.3f}")
        allhs.append(allh)

        n_mixands.append(p_curr.n_mixands())
        density_gmms.append(p_curr)

    def pdf_plotter(k : int):
        if k < len(density_gmms):
            return grid_eval(lambda x : density_gmms[k].density(x), x_bounds, dtype=DTYPE)
        else:
            return grid_eval(lambda x : 0.0, x_bounds, dtype=DTYPE) # too many mixands
    state_dist_fig, _ = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds)
    particle_figs, ekf_pdf_figs = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds, separate_figures=True, exclude_ticks=False)

    # Write down system properties
    benchmark_fields["datetime"] = curr_date_time 
    benchmark_fields["system"] = system.__class__.__name__
    benchmark_fields["dimension"] = dim
    benchmark_fields["n_traj"] = n_traj
    benchmark_fields["n_epochs_tran"] = n_epochs_tran
    benchmark_fields["training_timesteps"] = training_timesteps
    benchmark_fields["timesteps"] = timesteps
    benchmark_fields["init_train_time"] = init_train_time
    benchmark_fields["tran_train_time"] = tran_train_time
    benchmark_fields["n_mixands"] = n_mixands
    benchmark_fields["prop_times"] = prop_times
    benchmark_fields["average_log_likelihood"] = allhs

    experiment_name = f"trajectory_2D_grid_{curr_date_time}"

    save_figure_bundle(particle_figs, f"./benchmarks/{experiment_name}/particle")
    save_figure_bundle(ekf_pdf_figs, f"./benchmarks/{experiment_name}/pdf")
    state_dist_fig.savefig(f"./benchmarks/{experiment_name}/combined.pdf")
    
    with open(f"./benchmarks/{experiment_name}/data.json", "w") as f:
        json.dump(benchmark_fields, f, indent=4)


