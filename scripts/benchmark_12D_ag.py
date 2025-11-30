from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.GPGMM import GMModel

from .Systems import Quadcopter, sample_trajectories
from .Visualization import plot_2d_marginals_over_time, plot_2d_particle_scatter_over_time

import numpy as np
import torch
from scipy.stats import multivariate_normal
import os
import json
from datetime import datetime
from scipy.linalg import block_diag

from .sos_experiment import run_trials_sos
from .gpgmm_experiment import run_trials_gpgmm
from .true_gmm_experiment import run_trials_true_gmm
from .nf_experiment import run_trials_nf


DTYPE = torch.float64

def get_date_time_str():
    return datetime.now().strftime("%Yy_%mm_%dd_%Hh_%Mm_%Ss")


if __name__ == "__main__":

    np.random.seed(42)

    # ============================================================================
    # CONFIGURATION: Specify which methods to run
    # ============================================================================
    # Available methods:
    #   - "sos"
    #   - "gpgmm_ekf"
    #   - "gpgmm_wsasos"
    #   - "gpgmm_grid"
    #   - "true_gmm_ekf"
    #   - "true_gmm_wsasos"
    #   - "true_gmm_grid"
    #   - "nf"
    # ============================================================================
    methods_to_run = [
        "sos",
        #"gpgmm_ekf",
        #"gpgmm_wsasos",
        #"gpgmm_grid",
        #"true_gmm_ekf",
        #"true_gmm_wsasos",
        #"true_gmm_grid",
        #"nf"
    ]
    # ============================================================================

    print("Benchmarking 12D AG system")

    # System model
    cov_posvel = 0.03 * np.ones((6, 6))
    np.fill_diagonal(cov_posvel, 0.06)
    cov_angles = 0.0005 * np.eye(3)  # further reduce angle noise
    cov_rates = 0.01 * np.eye(3)     # much lower rate noise to reduce oscillations
    quad_covariance = block_diag(cov_posvel, cov_angles, cov_rates)

    system = Quadcopter(
        dt=0.05,
        waypoint=np.array([15.0, 15.0, 5.0]),
        thrust_max=30.0,
        torque_limits=np.array([1.5, 1.5, 0.8]),  
        covariance=quad_covariance,
    )
    # Tuning to keep Euler angles moderate and reduce rate oscillations
    system.c_w = 0.15  # increase angular damping further
    system.kp_pos = np.array([1.2, 1.2, 2.5])
    system.kd_pos = np.array([0.8, 0.8, 1.5])
    system.kp_ang = np.array([2.0, 2.0, 1.5])
    system.kd_ang = np.array([2.0, 2.0, 1.0])
    system.rate_filter_alpha = 0.2  # stronger rate smoothing

    # Dimension
    dim = system.dim()

    ###########################################################################################

    # Number of trajectories
    n_traj_train_sos = 4000 
    n_traj_train_gpgmm = 400
    n_traj_train_nf = 4000  # NF uses same amount as SOS
    n_traj_test = 1000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 150
    n_epochs_tran_refine = 300

    sos_batch_size = 1024
    sos_batch_size_refine = 2048
    sos_learning_rate_init = 1e-1
    sos_learning_rate_refine = 1e-3

    # Variance pads
    variance_pads = [10.0, 10.0, 7.0, 10.0, 10.0, 7.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5]

    # Time horizon
    training_timesteps = 10
    timesteps = 15

    # Number of trials
    num_trials = 3
    
    n_sos = 18
    
    # Grid method parameters (for 6D: [px, pz, theta, vx, vz, omega])
    # Bounds: [px_min, px_max, pz_min, pz_max, theta_min, theta_max, 
    #          vx_min, vx_max, vz_min, vz_max, omega_min, omega_max]
    x_bounds_6d = [-2.0, 2.0, -2.0, 2.0, -1.0, 1.0, 45.0, 55.0, -5.0, 5.0, -1.0, 1.0]
    grid_resolution = 3  # Lower resolution for 6D to keep it feasible
    max_mixands = 5000
    max_time = 1000
    n_components_init_ekf = 50
    n_components_init_wsasos = 10

    sos_tran_params = {
        "min_alpha_beta": 0.1,
        "max_alpha_beta": 300.0,
        "mu": 0.1,
        "min_Q_eigval": 1e-5,
        "regularization_weight": 1e-4,
        "initialization_scale": -1.0
    }
    sos_init_params = {
        "min_alpha_beta": 0.1,
        "max_alpha_beta": 300.0,
        "mu": 0.1,
        "min_Q_eigval": 1e-5,
        "regularization_weight": 1e-4,
        "initialization_scale": -1.0
    }
    ###########################################################################################



    def init_state_sampler():
        # 12D state: [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        # Preset C initial distribution
        mean = np.array([
            -8.0, -8.0, 0.8,
            10.0, 5.0, 0.0,
            0.00, 0.00, 0.00,
            0.0, -0.1, 0.1,
        ])
        cov = np.diag([
            1.5, 1.5, 0.5,
            6.0, 6.0, 2.5,
            0.01, 0.01, 0.01,
            0.35, 0.35, 0.35,
        ])
        return multivariate_normal.rvs(mean=mean, cov=cov)

    # Generate data
    traj_data_train_sos = sample_trajectories(system, init_state_sampler, timesteps, n_traj_train_sos)
    traj_data_train_gpgmm = sample_trajectories(system, init_state_sampler, timesteps, n_traj_train_gpgmm)
    traj_data_train_nf = sample_trajectories(system, init_state_sampler, timesteps, n_traj_train_nf)
    traj_data_test = sample_trajectories(system, init_state_sampler, timesteps, n_traj_test)

    # Create GDT for SOS method
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data_test), variance_pads=variance_pads)

    # Create initial state model for true_gmm
    init_mean = np.array([
        -8.0, -8.0, 0.8,
        10.0, 5.0, 0.0,
        0.00, 0.00, 0.00,
        0.0, -0.1, 0.1,
    ])
    init_cov = np.diag([
        1.5, 1.5, 0.5,
        6.0, 6.0, 2.5,
        0.01, 0.01, 0.01,
        0.35, 0.35, 0.35,
    ])
    init_state_model = GMModel(means=[init_mean], covariances=[init_cov], weights=[1.0])

    # Create benchmark directory with date/time
    curr_date_time = get_date_time_str()
    benchmark_dir = os.path.join("benchmarks", f"benchmark_12D_{curr_date_time}")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Directory for SOS figures
    sos_figures_dir = os.path.join(benchmark_dir, "sos_figures")
    os.makedirs(sos_figures_dir, exist_ok=True)

    # Initialize result variables
    sos_ll = None
    sos_prop_times = None
    gpgmm_ekf_ll = None
    gpgmm_ekf_prop_times = None
    gpgmm_wsasos_ll = None
    gpgmm_wsasos_prop_times = None
    gpgmm_grid_ll = None
    gpgmm_grid_prop_times = None
    true_gmm_ekf_ll = None
    true_gmm_ekf_prop_times = None
    true_gmm_wsasos_ll = None
    true_gmm_wsasos_prop_times = None
    true_gmm_grid_ll = None
    true_gmm_grid_prop_times = None
    nf_ll = None
    nf_prop_times = None

    print("Benchmarking methods: ", methods_to_run)

    if "sos" in methods_to_run:
        print("\n" + "="*60)
        print("Running SOS method")
        print("="*60)
        # Run SOS experiments
        sos_ll, sos_prop_times = run_trials_sos(
            traj_data_train_sos, traj_data_test, sos_figures_dir, 
            num_trials=num_trials, gdt=gdt, n=n_sos, batch_size=sos_batch_size, batch_size_refine=sos_batch_size_refine,
            n_epochs_init=n_epochs_init, n_epochs_tran_coarse=n_epochs_tran, n_epochs_tran_fine=n_epochs_tran_refine,
            tran_params=sos_tran_params, init_params=sos_init_params,
            learning_rate_init=sos_learning_rate_init, learning_rate_refine=sos_learning_rate_refine
        )

    # Run GPGMM experiments with different propagation methods
    if "gpgmm_ekf" in methods_to_run:
        print("\n" + "="*60)
        print("Running GPGMM-EKF method")
        print("="*60)
        gpgmm_ekf_ll, gpgmm_ekf_prop_times = run_trials_gpgmm(
            traj_data_train_gpgmm, traj_data_test,
            os.path.join(benchmark_dir, "gpgmm_ekf_figures"),
            num_trials=num_trials,
            propagation_method='ekf',
            n_components_init=n_components_init_ekf,
            n_epochs_tran=n_epochs_tran,
            save_figures=False
        )

    if "gpgmm_wsasos" in methods_to_run:
        print("\n" + "="*60)
        print("Running GPGMM-WSASOS method")
        print("="*60)
        gpgmm_wsasos_ll, gpgmm_wsasos_prop_times = run_trials_gpgmm(
            traj_data_train_gpgmm, traj_data_test,
            os.path.join(benchmark_dir, "gpgmm_wsasos_figures"),
            num_trials=num_trials,
            propagation_method='wsasos',
            n_components_init=n_components_init_wsasos,
            n_epochs_tran=n_epochs_tran,
            max_mixands=max_mixands,
            max_time=max_time,
            save_figures=False
        )

    if "gpgmm_grid" in methods_to_run:
        print("\n" + "="*60)
        print("Running GPGMM-Grid method")
        print("="*60)
        gpgmm_grid_ll, gpgmm_grid_prop_times = run_trials_gpgmm(
            traj_data_train_gpgmm, traj_data_test,
            os.path.join(benchmark_dir, "gpgmm_grid_figures"),
            num_trials=num_trials,
            propagation_method='grid',
            n_components_init=10,
            n_epochs_tran=n_epochs_tran,
            grid_resolution=grid_resolution,
            x_bounds=x_bounds_6d,
            max_time=max_time,
            save_figures=False
        )

    # Run TRUE_GMM experiments with different propagation methods
    if "true_gmm_ekf" in methods_to_run:
        print("\n" + "="*60)
        print("Running TRUE_GMM-EKF method")
        print("="*60)
        true_gmm_ekf_ll, true_gmm_ekf_prop_times = run_trials_true_gmm(
            system, init_state_model, traj_data_test,
            os.path.join(benchmark_dir, "true_gmm_ekf_figures"),
            num_trials=num_trials,
            propagation_method='ekf',
            save_figures=False,
            traj_data_for_plotting=traj_data_train_gpgmm
        )

    if "true_gmm_wsasos" in methods_to_run:
        print("\n" + "="*60)
        print("Running TRUE_GMM-WSASOS method")
        print("="*60)
        true_gmm_wsasos_ll, true_gmm_wsasos_prop_times = run_trials_true_gmm(
            system, init_state_model, traj_data_test,
            os.path.join(benchmark_dir, "true_gmm_wsasos_figures"),
            num_trials=num_trials,
            propagation_method='wsasos',
            max_mixands=max_mixands,
            max_time=max_time,
            save_figures=False,
            traj_data_for_plotting=traj_data_train_gpgmm
        )

    if "true_gmm_grid" in methods_to_run:
        print("\n" + "="*60)
        print("Running TRUE_GMM-Grid method")
        print("="*60)
        true_gmm_grid_ll, true_gmm_grid_prop_times = run_trials_true_gmm(
            system, init_state_model, traj_data_test,
            os.path.join(benchmark_dir, "true_gmm_grid_figures"),
            num_trials=num_trials,
            propagation_method='grid',
            grid_resolution=grid_resolution,
            x_bounds=x_bounds_6d,
            max_time=max_time,
            save_figures=False,
            traj_data_for_plotting=traj_data_train_gpgmm
        )

    # Run NF experiments
    if "nf" in methods_to_run:
        print("\n" + "="*60)
        print("Running NF (Normalizing Flow) method")
        print("="*60)
        nf_ll, nf_prop_times = run_trials_nf(
            traj_data_train_nf, traj_data_test, init_state_sampler,
            os.path.join(benchmark_dir, "nf_figures"),
            num_trials=num_trials,
            n_particles=1000,
            n_epochs_tran=n_epochs_tran,
            num_layers=8,
            hidden_features=128,
            use_gpu=True,
            batch_size=512,
            n_added_samples=1,
            kde_bandwidth="scott"
        )

    # Prepare data for JSON
    data = {
        "datetime": curr_date_time,
        "system": system.__class__.__name__,
        "dimension": int(dim),
        "n_traj_train_sos": int(n_traj_train_sos),
        "n_traj_train_gpgmm": int(n_traj_train_gpgmm),
        "n_traj_train_nf": int(n_traj_train_nf),
        "n_traj_test": int(n_traj_test),
        "n_epochs_init": int(n_epochs_init),
        "n_epochs_tran": int(n_epochs_tran),
        "n_epochs_tran_refine": int(n_epochs_tran_refine),
        "timesteps": int(timesteps),
        "num_trials": int(num_trials),
        "grid_resolution": int(grid_resolution),
        "max_mixands": int(max_mixands),
        "max_time": float(max_time),
        "methods_run": methods_to_run
    }
    
    # Only include results for methods that were run
    if sos_ll is not None:
        data["sos"] = {
            "negative_log_likelihoods": sos_ll.tolist(),
            "prop_times": sos_prop_times.tolist()
        }
    if gpgmm_ekf_ll is not None:
        data["gpgmm_ekf"] = {
            "negative_log_likelihoods": gpgmm_ekf_ll.tolist(),
            "prop_times": gpgmm_ekf_prop_times.tolist()
        }
    if gpgmm_wsasos_ll is not None:
        data["gpgmm_wsasos"] = {
            "negative_log_likelihoods": gpgmm_wsasos_ll.tolist(),
            "prop_times": gpgmm_wsasos_prop_times.tolist()
        }
    if gpgmm_grid_ll is not None:
        data["gpgmm_grid"] = {
            "negative_log_likelihoods": gpgmm_grid_ll.tolist(),
            "prop_times": gpgmm_grid_prop_times.tolist()
        }
    if true_gmm_ekf_ll is not None:
        data["true_gmm_ekf"] = {
            "negative_log_likelihoods": true_gmm_ekf_ll.tolist(),
            "prop_times": true_gmm_ekf_prop_times.tolist()
        }
    if true_gmm_wsasos_ll is not None:
        data["true_gmm_wsasos"] = {
            "negative_log_likelihoods": true_gmm_wsasos_ll.tolist(),
            "prop_times": true_gmm_wsasos_prop_times.tolist()
        }
    if true_gmm_grid_ll is not None:
        data["true_gmm_grid"] = {
            "negative_log_likelihoods": true_gmm_grid_ll.tolist(),
            "prop_times": true_gmm_grid_prop_times.tolist()
        }
    if nf_ll is not None:
        data["nf"] = {
            "negative_log_likelihoods": nf_ll.tolist(),
            "prop_times": nf_prop_times.tolist()
        }

    # Save data.json
    data_json_path = os.path.join(benchmark_dir, "data.json")
    with open(data_json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved data.json to {data_json_path}")

    # Compute statistics and create results file
    results = []
    results.append("="*60)
    results.append("BENCHMARK RESULTS")
    results.append("="*60)
    results.append(f"System: {system.__class__.__name__}")
    results.append(f"Dimension: {dim}")
    results.append(f"Number of trials: {num_trials}")
    results.append(f"Timesteps: {timesteps}")
    results.append("")

    # Helper function to compute mean and variance, handling NaN
    def compute_stats(arr):
        if arr is None:
            return None, None
        arr_flat = np.array(arr).flatten()
        arr_flat = arr_flat[~np.isnan(arr_flat)]
        if len(arr_flat) == 0:
            return None, None
        return float(np.mean(arr_flat)), float(np.var(arr_flat))

    # Statistics for each method at each timestep (only include methods that were run)
    methods = {}
    if sos_ll is not None:
        methods["SOS"] = (sos_ll, sos_prop_times)
    if gpgmm_ekf_ll is not None:
        methods["GPGMM-EKF"] = (gpgmm_ekf_ll, gpgmm_ekf_prop_times)
    if gpgmm_wsasos_ll is not None:
        methods["GPGMM-WSASOS"] = (gpgmm_wsasos_ll, gpgmm_wsasos_prop_times)
    if gpgmm_grid_ll is not None:
        methods["GPGMM-Grid"] = (gpgmm_grid_ll, gpgmm_grid_prop_times)
    if true_gmm_ekf_ll is not None:
        methods["TRUE_GMM-EKF"] = (true_gmm_ekf_ll, true_gmm_ekf_prop_times)
    if true_gmm_wsasos_ll is not None:
        methods["TRUE_GMM-WSASOS"] = (true_gmm_wsasos_ll, true_gmm_wsasos_prop_times)
    if true_gmm_grid_ll is not None:
        methods["TRUE_GMM-Grid"] = (true_gmm_grid_ll, true_gmm_grid_prop_times)
    if nf_ll is not None:
        methods["NF"] = (nf_ll, nf_prop_times)

    for method_name, (ll, prop_times) in methods.items():
        results.append(f"\n{method_name} Method")
        results.append("-" * 60)
        
        # Negative Log Likelihood statistics
        results.append("\nLog Likelihood:")
        results.append("Timestep | Mean      | Variance")
        results.append("-" * 40)
        for t in range(timesteps):
            ll_t = ll[:, t] if ll is not None else None
            mean_ll, var_ll = compute_stats(ll_t)
            if mean_ll is not None:
                results.append(f"  t={t:2d}   | {mean_ll:8.6f} | {var_ll:8.6f}")
            else:
                results.append(f"  t={t:2d}   |    N/A    |    N/A")
        
        # Overall statistics for ll
        mean_ll_all, var_ll_all = compute_stats(ll)
        if mean_ll_all is not None:
            results.append(f"\nOverall Log Likelihood - Mean: {mean_ll_all:.6f}, Variance: {var_ll_all:.6f}")
        
        # Propagation time statistics
        if prop_times is not None:
            results.append("\nPropagation Time (seconds):")
            results.append("Timestep | Mean      | Variance")
            results.append("-" * 40)
            # prop_times[trial, i] is the time to propagate from t=i to t=i+1
            # So for timestep t, we look at prop_times[:, t-1] (propagation from t-1 to t)
            for t in range(1, timesteps):
                if t-1 < prop_times.shape[1]:
                    prop_t = prop_times[:, t-1]
                    mean_prop, var_prop = compute_stats(prop_t)
                    if mean_prop is not None:
                        results.append(f"  t={t:2d}   | {mean_prop:8.6f} | {var_prop:8.6f}")
                    else:
                        results.append(f"  t={t:2d}   |    N/A    |    N/A")
                else:
                    results.append(f"  t={t:2d}   |    N/A    |    N/A")
            
            # Overall statistics for prop times (exclude zeros/NaNs)
            prop_times_flat = prop_times.flatten()
            prop_times_flat = prop_times_flat[~np.isnan(prop_times_flat)]
            prop_times_flat = prop_times_flat[prop_times_flat > 0]  # Exclude zeros
            if len(prop_times_flat) > 0:
                mean_prop_all = float(np.mean(prop_times_flat))
                var_prop_all = float(np.var(prop_times_flat))
                results.append(f"\nOverall Prop Time - Mean: {mean_prop_all:.6f}, Variance: {var_prop_all:.6f}")
        else:
            results.append("\nPropagation Time: Not tracked")

    # Save results file
    results_text = "\n".join(results)
    results_path = os.path.join(benchmark_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(results_text)
    print(f"Saved results.txt to {results_path}")
    
    # Also print results to console
    print("\n" + results_text)
    
    print(f"\nBenchmark completed! Results saved to: {benchmark_dir}")
