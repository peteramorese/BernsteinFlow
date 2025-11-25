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
import traceback
from datetime import datetime

def get_date_time_str():
    return datetime.now().strftime("%Yy_%mm_%dd_%Hh_%Mm_%Ss")

def save_figure_bundle(fig_bundle, dir):
    os.makedirs(dir, exist_ok=True)
    for k, fig in enumerate(fig_bundle):
        print("saving to: ", dir + f"/k_{k}.pdf")
        fig.savefig(dir + f"/k_{k}.pdf")


DTYPE = torch.float64

def run_trials_gpgmm(train_data, test_data, save_directory, num_trials, 
                     propagation_method='ekf',
                     n_components_init=10,
                     n_epochs_tran=50,
                     grid_resolution=20,
                     x_bounds=[-5.0, 5.0, -5.0, 5.0],
                     max_mixands=5000,
                     max_time=2000,
                     save_figures=True):
    """
    Run multiple trials of training GPGMM models, belief propagation, and evaluation.
    
    Parameters:
    -----------
    train_data : list of np.ndarray
        Training trajectory data. Each element is (N, dim) array for timestep t.
    test_data : list of np.ndarray
        Test trajectory data. Each element is (M, dim) array for timestep t.
    save_directory : str
        Directory to save figures (only saved for first trial).
    num_trials : int
        Number of trials to run.
    propagation_method : str
        Propagation method to use: 'ekf', 'wsasos', or 'grid' (default: 'ekf').
    n_components_init : int
        Number of components for initial state GMM (default: 10).
    n_epochs_tran : int
        Number of epochs for transition GP model training (default: 50).
    grid_resolution : int
        Grid resolution for grid propagation method (default: 20).
    x_bounds : list of float
        Bounds for x-space [x_min, x_max, y_min, y_max] (default: [-5.0, 5.0, -5.0, 5.0]).
    max_mixands : int
        Maximum number of mixands before stopping (for wsasos, default: 5000).
    max_time : float
        Maximum propagation time in seconds before stopping (for wsasos, default: 2000).
    save_figures : bool
        Whether to save figures for the first trial (default: True).
    
    Returns:
    --------
    log_likelihoods : np.ndarray
        Array of shape (num_trials, num_timesteps) containing average log likelihood
        for each belief (timestep) for each trial. The timesteps include t=0 (initial belief)
        through t=num_timesteps-1 (after num_timesteps-1 propagations).
    """
    # Get dimension from first timestep
    dim = train_data[0].shape[1]
    num_timesteps = len(test_data)  # Number of timesteps in test data (includes t=0)
    training_timesteps = len(train_data)
    
    # Initialize results array: (num_trials, num_timesteps)
    log_likelihoods = np.zeros((num_trials, num_timesteps))
    prop_times = np.zeros((num_trials, num_timesteps))
    
    # Run trials
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{num_trials} (method: {propagation_method})")
        print(f"{'='*60}\n")
        
        # Create the data matrices for training
        X0_data = train_data[0]
        Xp_data = create_transition_data_matrix(train_data[:training_timesteps])
        Xp_data_torch = torch.from_numpy(Xp_data)
        
        # Fit initial state model
        print("Fitting initial state model...")
        start = time.time()
        init_state_model = fit_gmm(X0_data, n_components=n_components_init, covariance_type='full')
        init_train_time = time.time() - start
        
        # Fit transition model
        print("Fitting transition state model...")
        start = time.time()
        transition_model = fit_gp(Xp=Xp_data_torch[:, dim:], X=Xp_data_torch[:, :dim], 
                                  num_epochs=n_epochs_tran, dtype=DTYPE, print_interval=10)
        tran_train_time = time.time() - start
        print("Done training models!\n")
        
        # Propagate beliefs based on method
        # For grid method, use make_diagonal() on initial state model
        if propagation_method == 'grid':
            density_gmms = [init_state_model.make_diagonal()]
        else:
            density_gmms = [init_state_model]
        
        # Evaluate initial belief log likelihood
        with torch.no_grad():
            allh_init = avg_log_likelihood(test_data[0], lambda x: density_gmms[0].density(x))
            log_likelihoods[trial, 0] = allh_init
            print(f"t=0: avg_log_likelihood = {allh_init:.6f}")
        
        # Propagate for remaining timesteps
        for k in range(1, num_timesteps):
            start = time.time()
            
            try:
                if propagation_method == 'ekf':
                    p_curr = propagate_gpgmm_ekf(density_gmms[k-1], transition_model)
                elif propagation_method == 'wsasos':
                    p_curr = propagate_gpgmm_wsasos(density_gmms[k-1], transition_model)
                    # Check stopping conditions for wsasos
                    if (p_curr.n_mixands() > max_mixands):
                        print(f"Stopping {propagation_method} early at t={k} due to max_mixands limit")
                        # Fill remaining timesteps with NaN
                        for remaining_k in range(k, num_timesteps):
                            log_likelihoods[trial, remaining_k] = np.nan
                            prop_times[trial, remaining_k] = np.nan
                        break
                elif propagation_method == 'grid':
                    p_curr = propagate_grid_gmm(density_gmms[k-1], transition_model, 
                                               bounds=x_bounds, resolution=grid_resolution)
                else:
                    raise ValueError(f"Unknown propagation method: {propagation_method}")

                prop_time = time.time() - start

                # Check stopping conditions for grid
                if (prop_time > max_time):
                    print(f"Stopping {propagation_method} early at t={k} due to max_time limit")
                    # Fill remaining timesteps with NaN
                    for remaining_k in range(k, num_timesteps):
                        log_likelihoods[trial, remaining_k] = np.nan
                        prop_times[trial, remaining_k] = np.nan
                    break
                
                print(f"Computed p(x{k}) ({propagation_method}) in {prop_time:.2f} seconds. "
                      f"Number of components: {p_curr.n_mixands()}")
                
                # Compute the log likelihood
                allh = avg_log_likelihood(test_data[k], lambda x: p_curr.density(x))
                log_likelihoods[trial, k] = allh
                print(f"  - Average log likelihood: {allh:.6f}")
                prop_times[trial, k] = prop_time
                density_gmms.append(p_curr)
            except Exception as e:
                print(f"Propagation failed at timestep {k}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                # Set remaining NLL and prop_times to NaN
                for remaining_k in range(k, num_timesteps):
                    log_likelihoods[trial, remaining_k] = np.nan
                    prop_times[trial, remaining_k] = np.nan
                break
        
        # Save figures only for the first trial
        if trial == 0 and save_figures:
            os.makedirs(save_directory, exist_ok=True)
            print(f"\nSaving figures to {save_directory}...")
            
            def pdf_plotter(k: int):
                if k < len(density_gmms):
                    return grid_eval(lambda x: density_gmms[k].density(x), x_bounds, dtype=DTYPE)
                else:
                    return grid_eval(lambda x: np.zeros(x.shape[0]), x_bounds, dtype=DTYPE)
            
            # Create and save figures
            state_dist_fig, _ = state_distribution_plot_2D(train_data, pdf_plotter, 
                                                           interactive=False, bounds=x_bounds)
            particle_figs, pdf_figs = state_distribution_plot_2D(train_data, pdf_plotter, 
                                                                  interactive=False, bounds=x_bounds, 
                                                                  separate_figures=True, exclude_ticks=False)
            
            # Save figure bundles
            particle_dir = os.path.join(save_directory, "particle")
            pdf_dir = os.path.join(save_directory, "pdf")
            os.makedirs(particle_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)
            
            for k, fig in enumerate(particle_figs):
                fig.savefig(os.path.join(particle_dir, f"k_{k}.pdf"))
                plt.close(fig)
            
            for k, fig in enumerate(pdf_figs):
                fig.savefig(os.path.join(pdf_dir, f"k_{k}.pdf"))
                plt.close(fig)
            
            state_dist_fig.savefig(os.path.join(save_directory, "combined.pdf"))
            plt.close(state_dist_fig)
            
            print("Figures saved.\n")
    
    return log_likelihoods, prop_times


if __name__ == "__main__":

    np.random.seed(42)

    max_mixands = 5000
    max_time = 20#600
    grid_resolution = 20

    # System model
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))
    #system = BistableOscillator(dt=0.1, a=1.0, d=1.0, cov_scale=0.03)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 50 # Can decrease data for faster results
    n_test_traj = 1000

    # Number of training epochs
    n_epochs_tran = 50

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)
    test_traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_test_traj)

    x_bounds = [-5.0, 5.0, -5.0, 5.0]
    
    # Run trials for each propagation method
    curr_date_time = get_date_time_str()
    
    # Grid method
    print("\n" + "="*60)
    print("Running Grid propagation method")
    print("="*60)
    log_likelihoods_grid, prop_times_grid = run_trials_gpgmm(
        traj_data, test_traj_data, 
        f"./figures/gpgmm_2D/grid", 
        num_trials=1,
        propagation_method='grid',
        n_components_init=10,
        n_epochs_tran=n_epochs_tran,
        grid_resolution=grid_resolution,
        x_bounds=x_bounds,
        save_figures=True
    )
    
    # EKF method
    print("\n" + "="*60)
    print("Running EKF propagation method")
    print("="*60)
    log_likelihoods_ekf, prop_times_ekf = run_trials_gpgmm(
        traj_data, test_traj_data,
        f"./figures/gpgmm_2D/ekf",
        num_trials=1,
        propagation_method='ekf',
        n_components_init=10,
        n_epochs_tran=n_epochs_tran,
        x_bounds=x_bounds,
        save_figures=True
    )
    
    # WSASOS method
    print("\n" + "="*60)
    print("Running WSASOS propagation method")
    print("="*60)
    log_likelihoods_wsasos, prop_times_wsasos = run_trials_gpgmm(
        traj_data, test_traj_data,
        f"./figures/gpgmm_2D/wsasos",
        num_trials=1,
        propagation_method='wsasos',
        n_components_init=10,
        n_epochs_tran=n_epochs_tran,
        x_bounds=x_bounds,
        max_mixands=max_mixands,
        max_time=max_time,
        save_figures=True
    )
    
    print("\n" + "="*60)
    print("All methods completed!")
    print("="*60)
    print(f"Grid log likelihoods: {log_likelihoods_grid}, prop times: {prop_times_grid}")
    print(f"EKF log likelihoods: {log_likelihoods_ekf}, prop times: {prop_times_ekf}")
    print(f"WSASOS log likelihoods: {log_likelihoods_wsasos}, prop times: {prop_times_wsasos}")

