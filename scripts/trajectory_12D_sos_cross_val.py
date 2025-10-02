from bernstein_flow.DistributionTransform import GaussianDistTransform
from sos_form.BetaModel import BetaSOSModel
from sos_form.SOSModel import optimize
from sos_form.SumBetaModel import SumBetaSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, mc_auc

from .Systems import Quadcopter, sample_trajectories

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import os
import json
import traceback


DTYPE = torch.float64

# ---- Cross-Validation Parameters ---- #
regularization_weights = [1e-4]
n_values = [5, 10, 15]
n_terms_values = [0]
n_epochs_values = [100]
n_traj_values = [10000]  


# ---- Plot 2D marginals over time using SumBetaMarginalSOSModel ----
def plot_2d_marginals_over_time(beliefs_list, keep_pair, pair_name, gdt,
                                resolution=60, save_path=None, show_plot=True):
    """
    Plot 2D marginal densities over time for given dimension pair in X space.
    keep_pair: tuple of two indices to keep (others are integrated out)
    pair_name: string for titles/filenames
    """
    dim_total = beliefs_list[0].dy
    keep_pair = tuple(int(i) for i in keep_pair)
    dims_to_integrate = [d for d in range(dim_total) if d not in keep_pair]

    # Create X space grid
    # Get bounds from the GDT transform for the two dimensions we're plotting
    x_bounds = []
    for dim in keep_pair:
        mean = gdt.means[dim]
        std = np.sqrt(gdt.variances[dim])
        x_bounds.extend([mean - 1.0*std, mean + 1.0*std])
    
    x_min, x_max = x_bounds[0], x_bounds[1]
    y_min, y_max = x_bounds[2], x_bounds[3]
    
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    XX, YY = np.meshgrid(xs, ys)
    
    # Create full 12D points for evaluation
    pts_12d = np.zeros((resolution * resolution, dim_total))
    pts_12d[:, keep_pair[0]] = XX.ravel()
    pts_12d[:, keep_pair[1]] = YY.ravel()
    
    # Fill other dimensions with their means
    for dim in dims_to_integrate:
        pts_12d[:, dim] = gdt.means[dim]

    # Evaluate each belief separately with individual color scaling
    grids = []
    with torch.no_grad():
        for belief in beliefs_list:
            marginal_model = belief.marginalize(dims_to_integrate)
            
            # Convert to U space for model evaluation
            pts_u = gdt.X_to_U(pts_12d)
            pts_u_2d = pts_u[:, keep_pair]
            
            # Get U space density
            u_density = marginal_model(torch.from_numpy(pts_u_2d).to(dtype=DTYPE)).cpu().numpy()
            
            # Convert back to X space density using gdt.x_density
            def u_density_func(u_2d):
                return marginal_model(torch.from_numpy(u_2d).to(dtype=DTYPE)).cpu().numpy()
            
            # For marginal density, we need to create a 12D function that evaluates the marginal
            def u_density_12d_func(u_12d):
                u_2d = u_12d[:, keep_pair]
                return marginal_model(torch.from_numpy(u_2d).to(dtype=DTYPE)).cpu().numpy()
            
            x_density = gdt.x_density(pts_12d, u_density_12d_func)
            Z = x_density.reshape(resolution, resolution)
            grids.append(Z)

    # Layout
    t = len(beliefs_list)
    n_cols = min(5, t)
    n_rows = (t + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2*n_cols, 3.0*n_rows), squeeze=False)
    for k, Z in enumerate(grids):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r][c]
        # Use contourf with plasma colormap and contour with white lines
        cf = ax.contourf(XX, YY, Z, levels=30, cmap='plasma')
        ax.contour(XX, YY, Z, levels=10, colors='white', alpha=0.6, linewidths=0.5)
        # Remove axes labels and titles
        ax.set_xticks([])
        ax.set_yticks([])
    # Hide unused axes
    for k in range(t, n_rows*n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r][c].axis('off')
    fig.tight_layout()
    # Note: No global colorbar since each subplot has its own scale
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_2d_particle_scatter_over_time(x_traj_list, keep_pair, pair_name, gdt,
                                       sample_limit=None, save_path=None, show_plot=True):
    """
    Plot 2D marginal trajectory particles (in X-space) over time for a given
    pair of state indices. x_traj_list is a list of length T with arrays (N_t, dy).
    keep_pair: tuple of two indices to keep for scatter.
    """
    keep_pair = tuple(int(i) for i in keep_pair)
    T = len(x_traj_list)

    # Use same bounds as density plots (based on GDT means and variances)
    x_bounds = []
    for dim in keep_pair:
        mean = gdt.means[dim]
        std = np.sqrt(gdt.variances[dim])
        x_bounds.extend([mean - 1.0*std, mean + 1.0*std])
    
    x_min, x_max = x_bounds[0], x_bounds[1]
    y_min, y_max = x_bounds[2], x_bounds[3]

    n_cols = min(5, T)
    n_rows = (T + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 3.0*n_rows), squeeze=False)
    for k in range(T):
        X = x_traj_list[k]
        if sample_limit is not None and X.shape[0] > sample_limit:
            idx = np.random.choice(X.shape[0], size=sample_limit, replace=False)
            Xplot = X[idx]
        else:
            Xplot = X
        r = k // n_cols
        c = k % n_cols
        ax = axes[r][c]
        ax.scatter(Xplot[:, keep_pair[0]], Xplot[:, keep_pair[1]], s=3, alpha=0.5)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        # Remove axes labels and titles
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(T, n_rows*n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r][c].axis('off')
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def run_single_experiment(regularization_weight, n, n_terms, n_epochs, experiment_dir,
                         system, gdt, traj_data, u_traj_data, u_test_traj_data,
                         device, use_gpu, DTYPE, n_traj_used: int):
    """
    Run a single experiment with given parameters and save results.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with:")
    print(f"  regularization_weight: {regularization_weight}")
    print(f"  n: {n}")
    print(f"  n_terms: {n_terms}")
    print(f"  n_epochs: {n_epochs}")
    print(f"{'='*60}")
    
    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment settings
    settings_file = os.path.join(experiment_dir, "experiment_settings.txt")
    with open(settings_file, 'w') as f:
        f.write("Experiment Settings\n")
        f.write("==================\n")
        f.write(f"regularization_weight: {regularization_weight}\n")
        f.write(f"n: {n}\n")
        f.write(f"n_terms: {n_terms}\n")
        f.write(f"n_epochs: {n_epochs}\n")
        f.write(f"n_traj: {n_traj_used}\n")
        f.write(f"device: {device}\n")
        f.write(f"use_gpu: {use_gpu}\n")
    
    # System parameters
    dim = system.dim()
    training_timesteps = 10
    timesteps = training_timesteps
    
    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, dim:]), gdt.X_to_U(Xp_data[:, :dim])])  # Transition kernel data 

    # Create data loaders
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=256, shuffle=True, pin_memory=use_gpu)
    U0_dataloader_refine = DataLoader(U0_dataset, batch_size=2048, shuffle=True, pin_memory=use_gpu)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=256, shuffle=True, pin_memory=use_gpu)
    Up_dataloader_refine = DataLoader(Up_dataset, batch_size=2048, shuffle=True, pin_memory=use_gpu)

    # Create models
    transition_model = BetaSOSModel(dy=dim, 
                                    dx=dim, 
                                    n=n, 
                                    min_alpha_beta=0.1, 
                                    max_alpha_beta=80.0, 
                                    mu=0.1, 
                                    min_Q_eigval=1e-8, 
                                    regularization_weight=regularization_weight)

    print("Training transition model...")
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    _, best_trans_loss_1 = optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
    _, best_trans_loss_2 = optimize(transition_model, Up_dataloader_refine, trans_optimizer, epochs=n_epochs//4)

    transition_model.to(device=torch.device("cpu"))
    print("Done training transition model")

    init_state_model = BetaSOSModel(dy=dim, 
                                    dx=0, 
                                    n=n, 
                                    conditional=False, 
                                    reference_factor_model=transition_model, 
                                    min_alpha_beta=0.4, 
                                    max_alpha_beta=100.0, 
                                    mu=0.1, 
                                    min_Q_eigval=1e-8, 
                                    regularization_weight=regularization_weight)

    print("Training init state model...")
    init_state_model.to(device=device, dtype=DTYPE)
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    _, best_init_loss_1 = optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs)
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-4)
    _, best_init_loss_2 = optimize(init_state_model, U0_dataloader_refine, init_optimizer, epochs=n_epochs//2)

    init_state_model.to(device=torch.device("cpu"))
    print("Done training init state model")

    # Propagate beliefs
    beliefs = [init_state_model]
    for i in range(timesteps):
        beliefs.append(transition_model.propagate(beliefs[i]))

    # Calculate AUC and test log-likelihood for each belief at corresponding timestep
    print("\nBelief metrics:")
    auc_values = []
    test_avg_log_liks = []
    for i, belief in enumerate(beliefs):
        with torch.no_grad():
            auc = mc_auc(12, lambda u : belief(torch.from_numpy(u)).numpy(), n_samples=10000)
            auc_values.append(float(auc))
            # Evaluate avg log-likelihood on test u-space data at timestep i
            U_test_i = u_test_traj_data[i]
            ll = np.mean(np.log(belief(torch.from_numpy(U_test_i)).numpy() + 1e-12))
            test_avg_log_liks.append(float(ll))
            print(f"t={i}: auc={auc:.6f}, test_avg_loglik={ll:.6f}")

    # Generate and save plots
    print("Generating plots...")
    
    # Create subdirectory for individual timestep figures
    belief_density_dir = os.path.join(experiment_dir, "belief_density")
    os.makedirs(belief_density_dir, exist_ok=True)
    
    # Marginal plots - save individual timesteps
    # Using state indices: [0:px, 1:py, 2:pz, 3:vx, 4:vy, 5:vz, 6:phi, 7:theta, 8:psi, 9:p, 10:q, 11:r]
    for t, belief in enumerate(beliefs):
        # Create individual plots for each timestep
        for pair, pair_name in [((0, 1), "px_py"), ((3, 4), "vx_vy"), ((6, 7), "phi_theta"), 
                               ((9, 10), "p_q"), ((2, 5), "pz_vz"), ((8, 11), "psi_r")]:
            plot_2d_marginals_over_time([belief], pair, pair_name, gdt,
                                        resolution=60,
                                        save_path=os.path.join(belief_density_dir, f"{pair_name}_t{t:02d}.pdf"),
                                        show_plot=False)
    
    # Save results.json
    results = {
        "best_transition_loss": float(min(best_trans_loss_1, best_trans_loss_2)),
        "best_init_loss": float(min(best_init_loss_1, best_init_loss_2)),
        "auc_per_timestep": auc_values,
        "test_avg_loglik_per_timestep": test_avg_log_liks,
        "n_traj": int(n_traj_used),
        "n": int(n),
        "n_terms": int(n_terms),
        "n_epochs": int(n_epochs),
        "regularization_weight": float(regularization_weight)
    }
    with open(os.path.join(experiment_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Experiment completed. Results saved to: {experiment_dir}")
    
    # Clean up memory
    del transition_model, init_state_model, beliefs
    del U0_data_torch, Up_data_torch, U0_dataset, Up_dataset
    del U0_dataloader, U0_dataloader_refine, Up_dataloader, Up_dataloader_refine
    if use_gpu:
        torch.cuda.empty_cache()
    
    return True

if __name__ == "__main__":
    import itertools
    from datetime import datetime

    # System model - 12D Quadcopter
    system = Quadcopter(dt=0.10, covariance=0.05 * np.eye(12), waypoint=np.array([10.0, 0.0, 1.0]))

    # Dimension
    dim = system.dim()

    # Number of trajectories (pool); will subsample per experiment to allow tuning n_traj
    n_traj_pool = max(n_traj_values)

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        # 12D state: [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        # Start near hover at origin with small initial conditions
        mean = np.array([0.0, 0.0, 0.0, 0.0, 30.0, -30.0, 0.0, 0.0, -0.8, 0.8, 0.0, 0.0])
        cov = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        return multivariate_normal.rvs(mean=mean, cov=cov)

    # Sample trajectory data once for all experiments
    print("Sampling trajectory data pool...")
    traj_data_pool = sample_trajectories(system, init_state_sampler, timesteps, n_traj_pool)
    test_traj_data_pool = sample_trajectories(system, init_state_sampler, timesteps + 1, n_traj_pool)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data_pool), variance_pads=[5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1])

    # GPU setup
    use_gpu = True
    print("Using GPU: ", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
    print("device: ", device)

    
    # Create root directory for all experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("benchmarks", f"sos_12D_cross_val_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)
    
    # Generate shared MC particles plots once
    print("Generating shared MC particles plots...")
    mc_particles_dir = os.path.join(root_dir, "mc_particles")
    os.makedirs(mc_particles_dir, exist_ok=True)
    
    for t in range(len(traj_data_pool)):
        for pair, pair_name in [((0, 1), "px_py"), ((3, 4), "vx_vy"), ((6, 7), "phi_theta"), 
                               ((9, 10), "p_q"), ((2, 5), "pz_vz"), ((8, 11), "psi_r")]:
            plot_2d_particle_scatter_over_time([traj_data_pool[t]], pair, pair_name, gdt,
                                               sample_limit=10000,
                                               save_path=os.path.join(mc_particles_dir, f"{pair_name}_t{t:02d}.pdf"),
                                               show_plot=False)
    
    # Save overall experiment info
    with open(os.path.join(root_dir, "experiment_info.txt"), 'w') as f:
        f.write("Cross-Validation Experiment\n")
        f.write("==========================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total experiments: {len(regularization_weights) * len(n_values) * len(n_terms_values) * len(n_epochs_values)}\n")
        f.write(f"regularization_weights: {regularization_weights}\n")
        f.write(f"n_values: {n_values}\n")
        f.write(f"n_terms_values: {n_terms_values}\n")
        f.write(f"n_epochs_values: {n_epochs_values}\n")
    
    # Run all parameter combinations
    total_experiments = len(regularization_weights) * len(n_values) * len(n_terms_values) * len(n_epochs_values) * len(n_traj_values)
    experiment_count = 0
    
    for reg_weight, n, n_terms, n_epochs, n_traj in itertools.product(regularization_weights, n_values, n_terms_values, n_epochs_values, n_traj_values):
        experiment_count += 1
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_count}/{total_experiments}")
        print(f"{'='*80}")
        
        # Create experiment directory name
        exp_name = f"exp_{experiment_count:03d}_reg{reg_weight:.0e}_n{n}_terms{n_terms}_epochs{n_epochs}_traj{n_traj}"
        experiment_dir = os.path.join(root_dir, exp_name)
        
        try:
            # Subsample pooled trajectories to requested n_traj for this experiment
            idx = np.random.choice(traj_data_pool[0].shape[0], size=n_traj, replace=False)
            traj_data = [X[idx] for X in traj_data_pool]
            test_idx = np.random.choice(test_traj_data_pool[0].shape[0], size=n_traj, replace=False)
            test_traj_data = [X[test_idx] for X in test_traj_data_pool]
            u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]
            u_test_traj_data = [gdt.X_to_U(X_data) for X_data in test_traj_data]

            # Run the experiment
            success = run_single_experiment(
                regularization_weight=reg_weight,
                n=n,
                n_terms=n_terms,
                n_epochs=n_epochs,
                experiment_dir=experiment_dir,
                system=system,
                gdt=gdt,
                traj_data=traj_data,
                u_traj_data=u_traj_data,
                u_test_traj_data=u_test_traj_data,
                device=device,
                use_gpu=use_gpu,
                DTYPE=DTYPE,
                n_traj_used=n_traj
            )
            
            if success:
                print(f"✓ Experiment {experiment_count} completed successfully")
            else:
                print(f"✗ Experiment {experiment_count} failed")
                
        except Exception as e:
            tb = traceback.format_exc()
            print(f"✗ Experiment {experiment_count} failed with error: {str(e)}")
            # Create error log with full traceback
            error_file = os.path.join(experiment_dir, "error_log.txt")
            os.makedirs(experiment_dir, exist_ok=True)
            with open(error_file, 'w') as f:
                f.write("Experiment failed with error and traceback:\n")
                f.write(str(e) + "\n\n")
                f.write(tb)
        
        # Force garbage collection and memory cleanup
        import gc
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Results saved in: {root_dir}")
    print(f"{'='*80}")
