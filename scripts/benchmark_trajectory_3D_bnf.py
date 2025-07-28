from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, avg_log_likelihood
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct, mc_auc, marginal, integrate
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
from scipy.spatial import Rectangle
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

    # System model
    system = DisturbedDubinsCar(dt=1.0, track_heading_function=lambda x : np.sin(1.2*x), noise_magnitude=0.3)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 1000

    # Number of training epochs
    n_epochs_init = 2000
    n_epochs_tran = 200

    # Time horizon
    training_timesteps = 10
    timesteps = 10

    def init_state_sampler():
        xy_init = multivariate_normal.rvs(mean=np.array([0.0, 0.0]), cov = np.diag([0.2, 0.2]))
        theta_init = beta.rvs(a=16, b=16, loc=-0.5, scale=1)
        return np.append(xy_init, theta_init)

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)
    test_traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[2.2, 2.2, 2.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, :dim]), gdt.X_to_U(Xp_data[:, dim:])])  # Transition kernel data

    use_gpu = 
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=128, shuffle=True, pin_memory=use_gpu)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=1024, shuffle=True, pin_memory=use_gpu)

    # Create initial state and transition models
    transformer_degrees = [15, 15, 15]
    conditioner_degrees = [15, 15, 15]
    init_cond_deg_incr = None #[20] * len(conditioner_degrees)
    init_state_model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, dtype=DTYPE, conditioner_deg_incr=init_cond_deg_incr, device=device)

    tran_cond_deg_incr = None #[5] * len(conditioner_degrees)
    transition_model = ConditionalBernsteinFlowModel(dim=dim, conditional_dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees, dtype=DTYPE, conditioner_deg_incr=init_cond_deg_incr, device=device)

    # Train the models
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    
    print("Training initial state model...")
    start = time.time()
    optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init, log_buffer_size=30)
    init_train_time = time.time() - start
    print("Done training initial state model \n")

    print("Training transition model...")
    start = time.time()
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-1)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran, log_buffer_size=30)
    tran_train_time = time.time() - start
    print("Done training transition model \n")


    # Compute the propagated polynomials
    init_model_tfs = init_state_model.get_density_factor_polys(dtype=np.float128)
    trans_model_tfs = transition_model.get_density_factor_polys(dtype=np.float128)

    # Region of integration
    roi = Rectangle(mins=[0.4, 0.0], maxes=[0.6, 0.4])

    p_init = poly_product_bernstein_direct(init_model_tfs)
    p_transition = poly_product_bernstein_direct(trans_model_tfs)
    density_polynomials = [p_init]
    xy_marginal_density_polys = [marginal(p_init, {2}, True)]
    prop_times = []
    mc_aucs = []
    allhs = []
    prob_in_roi = []
    for k in range(1, timesteps):
        start = time.time()
        p_curr = propagate_bfm([density_polynomials[k-1]], [p_transition])
        prop_times.append(time.time() - start)
        mc_aucs.append(mc_auc(p_curr, n_samples=10000))
        print(f"Computed p(x{k}) in {prop_times[-1]:.2f} seconds")

        # Compute the log likelihood using the x density
        x_allh = avg_log_likelihood(test_traj_data[k], lambda x : gdt.x_density(x, p_curr))
        print(f" - Average log likelihood: {x_allh:.3f}")
        allhs.append(x_allh)

        # Compute the xy marginal polynomial
        p_curr_xy_marg = marginal(p_curr, {2}, True)
        prob_in_roi_k = integrate(p_curr_xy_marg, roi)
        print(f" - Probability of being in ROI: {prob_in_roi_k:.3f}")
        prob_in_roi.append(prob_in_roi_k)

        density_polynomials.append(p_curr)
        xy_marginal_density_polys.append(p_curr_xy_marg)

    x_bounds = [-1.0, 8.0, -5.0, 5.0]
    def pdf_plotter(k : int):
        return grid_eval(lambda x : gdt.x_density(x, xy_marginal_density_polys[k], dim_mask=np.array([True, True, False])), x_bounds, dtype=DTYPE)

    x_traj_data_xy_marginal = [timestep_data[:, :2] for timestep_data in u_traj_data]
    state_dist_fig, _ = state_distribution_plot_2D(x_traj_data_xy_marginal, pdf_plotter, interactive=False, bounds=x_bounds)
    particle_figs, pdf_figs = state_distribution_plot_2D(x_traj_data_xy_marginal, pdf_plotter, interactive=False, bounds=x_bounds, separate_figures=True, exclude_ticks=False)

    

    # Write down system properties
    curr_date_time = get_date_time_str()
    benchmark_fields["datetime"] = curr_date_time 
    benchmark_fields["system"] = system.__class__.__name__
    benchmark_fields["dimension"] = dim
    benchmark_fields["n_traj"] = n_traj
    benchmark_fields["n_epochs_init"] = n_epochs_init
    benchmark_fields["n_epochs_tran"] = n_epochs_tran
    benchmark_fields["training_timesteps"] = training_timesteps
    benchmark_fields["timesteps"] = timesteps
    benchmark_fields["device"] = device.type
    benchmark_fields["transformer_degrees"] = transformer_degrees
    benchmark_fields["conditioner_degrees"] = conditioner_degrees
    benchmark_fields["init_cond_deg_incr"] = init_cond_deg_incr
    benchmark_fields["tran_cond_deg_incr"] = tran_cond_deg_incr
    benchmark_fields["init_model_params"] = init_state_model.n_parameters()
    benchmark_fields["tran_model_params"] = transition_model.n_parameters()
    benchmark_fields["init_train_time"] = init_train_time
    benchmark_fields["tran_train_time"] = tran_train_time
    benchmark_fields["prop_times"] = prop_times
    benchmark_fields["mc_auc"] = mc_aucs
    benchmark_fields["average_log_likelihood"] = allhs
    benchmark_fields["prob_in_roi"] = prob_in_roi



    experiment_name = f"trajectory_2D_bnf_{curr_date_time}"

    save_figure_bundle(particle_figs, f"./benchmarks/{experiment_name}/particle")
    save_figure_bundle(pdf_figs, f"./benchmarks/{experiment_name}/pdf")
    state_dist_fig.savefig(f"./benchmarks/{experiment_name}/combined.pdf")

    with open(f"./benchmarks/{experiment_name}/data.json", "w") as f:
        json.dump(benchmark_fields, f, indent=4)


