from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.GPGMM import GMModel, GPModel, fit_gmm, fit_gp
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, avg_log_likelihood, empirical_prob_in_region
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct
from bernstein_flow.Propagate import propagate_gpgmm_ekf, propagate_gpgmm_wsasos, propagate_grid_gmm

from .Systems import VanDerPol, BistableOscillator, sample_trajectories
from .Visualization import interactive_transformer_plot, state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
from scipy.spatial import Rectangle
import time
import os
import json
from datetime import datetime
import copy

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

    max_mixands = 5000
    max_time = 2000
    grid_resolution = 20

    # System model
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))
    #system = BistableOscillator(dt=0.1, a=1.0, d=1.0, cov_scale=0.03)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 500
    n_test_traj = 10000

    # Number of training epochs
    n_epochs_tran = 150

    # Time horizon
    training_timesteps = 10
    timesteps = 10

    # Region of integration
    #roi = Rectangle(mins=[0.0, 0.0], maxes=[2.0, 2.0])
    roi = Rectangle(mins=[-1.0, 1.0], maxes=[-1.0, 1.0])

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)
    test_traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_test_traj)

    x_bounds = [-5.0, 5.0, -5.0, 5.0]

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])
    Xp_data_torch = torch.from_numpy(Xp_data)

    print("Fitting initial state model...")
    start = time.time()
    init_state_model = fit_gmm(X0_data, n_components=10, covariance_type='full')
    init_train_time = time.time() - start

    print("Fitting transition state model...")
    start = time.time()
    transition_model = fit_gp(Xp=Xp_data_torch[:, dim:], X=Xp_data_torch[:, :dim], num_epochs=n_epochs_tran, dtype=torch.float64)
    tran_train_time = time.time() - start
    print("Done!")


    curr_date_time = get_date_time_str()
    def ekf():
        density_gmms_ekf = [init_state_model]
        n_mixands_ekf = [init_state_model.n_mixands()]
        prop_times_ekf = []
        allhs_ekf = []
        prob_in_roi = []
        mc_gt_prob_in_roi = []
        for k in range(1, timesteps):
            start = time.time()
            p_curr = propagate_gpgmm_ekf(density_gmms_ekf[k-1], transition_model)
            prop_times_ekf.append(time.time() - start)
            print(f"Computed p(x{k}) (EKF) in {prop_times_ekf[-1]:.2f} seconds. Number of components: ", p_curr.n_mixands())

            # Compute the log likelihood
            allh = avg_log_likelihood(test_traj_data[k], lambda x : p_curr.density(x))
            print(f" - Average log likelihood: {allh:.3f}")
            allhs_ekf.append(allh)

            # Compute prob in roi
            p_curr_diag = copy.deepcopy(p_curr)
            p_curr_diag.make_cov_diag()
            prob_in_roi_k = p_curr_diag.integrate(roi)
            prob_in_roi.append(prob_in_roi_k)

            # MC "ground truth" prob in roi
            mc_gt_prob_in_roi_k = empirical_prob_in_region(test_traj_data[k], roi)
            mc_gt_prob_in_roi.append(mc_gt_prob_in_roi_k)
            print(f" - Evaluation: {prob_in_roi_k:.3f} / MC ground truth evaluation: {mc_gt_prob_in_roi_k:.3f}")

            n_mixands_ekf.append(p_curr.n_mixands())
            density_gmms_ekf.append(p_curr)

        # EKF
        def pdf_plotter(k : int):
            if k < len(density_gmms_ekf):
                return grid_eval(lambda x : density_gmms_ekf[k].density(x), x_bounds, dtype=DTYPE)
            else:
                return grid_eval(lambda x : 0.0, x_bounds, dtype=DTYPE) # too many mixands
        state_dist_fig_ekf, _ = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds)
        particle_figs, ekf_pdf_figs = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds, separate_figures=True, exclude_ticks=False)

        # EKF

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
        benchmark_fields["n_mixands"] = n_mixands_ekf
        benchmark_fields["prop_times"] = prop_times_ekf
        benchmark_fields["average_log_likelihood"] = allhs_ekf
        benchmark_fields["prob_in_roi"] = prob_in_roi
        benchmark_fields["mc_gt_prob_in_roi"] = mc_gt_prob_in_roi

        experiment_name = f"trajectory_2D_ekf_{curr_date_time}"

        save_figure_bundle(particle_figs, f"./benchmarks/{experiment_name}/particle")
        save_figure_bundle(ekf_pdf_figs, f"./benchmarks/{experiment_name}/pdf")
        state_dist_fig_ekf.savefig(f"./benchmarks/{experiment_name}/combined.pdf")
    
        with open(f"./benchmarks/{experiment_name}/data.json", "w") as f:
            json.dump(benchmark_fields, f, indent=4)

    def wsasos():
        density_gmms_wsasos = [init_state_model]
        n_mixands_wsasos = [init_state_model.n_mixands()]
        prop_times_wsasos = []
        allhs_wsasos = []
        prob_in_roi = []
        mc_gt_prob_in_roi = []
        for k in range(1, timesteps):
            start = time.time()
            p_curr = propagate_gpgmm_wsasos(density_gmms_wsasos[k-1], transition_model)
            prop_times_wsasos.append(time.time() - start)
            print(f"Computed p(x{k}) (WSASOS) in {prop_times_wsasos[-1]:.2f} seconds. Number of components: ", p_curr.n_mixands())

            # Compute the log likelihood
            allh = avg_log_likelihood(test_traj_data[k], lambda x : p_curr.density(x))
            print(f" - Average log likelihood: {allh:.3f}")
            allhs_wsasos.append(allh)

            # Compute prob in roi
            p_curr_diag = copy.deepcopy(p_curr)
            p_curr_diag.make_cov_diag()
            prob_in_roi_k = p_curr_diag.integrate(roi)
            prob_in_roi.append(prob_in_roi_k)

            # MC "ground truth" prob in roi
            mc_gt_prob_in_roi_k = empirical_prob_in_region(test_traj_data[k], roi)
            mc_gt_prob_in_roi.append(mc_gt_prob_in_roi_k)
            print(f" - Evaluation: {prob_in_roi_k:.3f} / MC ground truth evaluation: {mc_gt_prob_in_roi_k:.3f}")

            n_mixands_wsasos.append(p_curr.n_mixands())
            density_gmms_wsasos.append(p_curr)

            if (density_gmms_wsasos[-1].n_mixands() > max_mixands) or (prop_times_wsasos[-1] > max_time):
                break

        # WSASOS
        def pdf_plotter(k : int):
            if k < len(density_gmms_wsasos):
                return grid_eval(lambda x : density_gmms_wsasos[k].density(x), x_bounds, dtype=DTYPE)
            else:
                return grid_eval(lambda x : np.zeros(x.shape[0]), x_bounds, dtype=DTYPE) # too many mixands
        state_dist_fig_wsasos, _ = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds)
        particle_figs, wsasos_pdf_figs = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds, separate_figures=True, exclude_ticks=False)

        # WASOS

        # Write down system properties
        curr_date_time = get_date_time_str()
        benchmark_fields["system"] = system.__class__.__name__
        benchmark_fields["dimension"] = dim
        benchmark_fields["n_traj"] = n_traj
        benchmark_fields["n_epochs_tran"] = n_epochs_tran
        benchmark_fields["training_timesteps"] = training_timesteps
        benchmark_fields["timesteps"] = timesteps
        benchmark_fields["init_train_time"] = init_train_time
        benchmark_fields["tran_train_time"] = tran_train_time
        benchmark_fields["n_mixands"] = n_mixands_wsasos
        benchmark_fields["prop_times"] = prop_times_wsasos
        benchmark_fields["average_log_likelihood"] = allhs_wsasos
        benchmark_fields["prob_in_roi"] = prob_in_roi
        benchmark_fields["mc_gt_prob_in_roi"] = mc_gt_prob_in_roi

        experiment_name = f"trajectory_2D_wsasos_{curr_date_time}"

        save_figure_bundle(particle_figs, f"./benchmarks/{experiment_name}/particle")
        save_figure_bundle(wsasos_pdf_figs, f"./benchmarks/{experiment_name}/pdf")
        state_dist_fig_wsasos.savefig(f"./benchmarks/{experiment_name}/combined.pdf")

        with open(f"./benchmarks/{experiment_name}/data.json", "w") as f:
            json.dump(benchmark_fields, f, indent=4)

    def grid():
        density_gmms = [init_state_model]
        n_mixands = [init_state_model.n_mixands()]
        prop_times = []
        allhs = []
        prob_in_roi = []
        mc_gt_prob_in_roi = []
        for k in range(1, timesteps):
            start = time.time()
            p_curr = propagate_grid_gmm(density_gmms[k-1], transition_model, bounds=x_bounds, resolution=grid_resolution)
            prop_times.append(time.time() - start)
            print(f"Computed p(x{k}) (Grid) in {prop_times[-1]:.2f} seconds. Number of components: ", p_curr.n_mixands())

            # Compute the log likelihood
            allh = avg_log_likelihood(test_traj_data[k], lambda x : p_curr.density(x))
            print(f" - Average log likelihood: {allh:.3f}")
            allhs.append(allh)

            # Compute prob in roi
            p_curr_diag = copy.deepcopy(p_curr)
            p_curr_diag.make_cov_diag()
            prob_in_roi_k = p_curr_diag.integrate(roi)
            prob_in_roi.append(prob_in_roi_k)

            # MC "ground truth" prob in roi
            mc_gt_prob_in_roi_k = empirical_prob_in_region(test_traj_data[k], roi)
            mc_gt_prob_in_roi.append(mc_gt_prob_in_roi_k)
            print(f" - Evaluation: {prob_in_roi_k:.3f} / MC ground truth evaluation: {mc_gt_prob_in_roi_k:.3f}")

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
        benchmark_fields["prob_in_roi"] = prob_in_roi
        benchmark_fields["mc_gt_prob_in_roi"] = mc_gt_prob_in_roi
        benchmark_fields["grid_resolution"] = grid_resolution

        experiment_name = f"trajectory_2D_grid_{curr_date_time}"

        save_figure_bundle(particle_figs, f"./benchmarks/{experiment_name}/particle")
        save_figure_bundle(ekf_pdf_figs, f"./benchmarks/{experiment_name}/pdf")
        state_dist_fig.savefig(f"./benchmarks/{experiment_name}/combined.pdf")
        
        with open(f"./benchmarks/{experiment_name}/data.json", "w") as f:
            json.dump(benchmark_fields, f, indent=4)


    grid()
    ekf()
    wsasos()

