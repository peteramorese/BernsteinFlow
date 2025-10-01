from bernstein_flow.DistributionTransform import GaussianDistTransform
from sos_form.SOSModel import optimize
from sos_form.SumBetaModel import SumBetaSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, mc_auc

from .Systems import CartPole, sample_trajectories

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
regularization_weights = [1e-2, 1e-3]
n_values = [5, 7]
n_terms_values = [10, 15]
n_epochs_values = [150]
n_traj_values = [4000, 10000]  

# ---- Plot 2D marginals over time using SumBetaMarginalSOSModel ----
def plot_2d_marginals_over_time(beliefs_list, keep_pair, pair_name,
                                resolution=60, save_path=None, show_plot=True):
    """
    Plot 2D marginal densities over time for given dimension pair.
    keep_pair: tuple of two indices to keep (others are integrated out)
    pair_name: string for titles/filenames
    """
    dim_total = beliefs_list[0].dy
    keep_pair = tuple(int(i) for i in keep_pair)
    dims_to_integrate = [d for d in range(dim_total) if d not in keep_pair]

    # Evaluate each belief separately with individual color scaling
    grids = []
    xs = np.linspace(0.05, 0.95, resolution)
    ys = np.linspace(0.05, 0.95, resolution)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    with torch.no_grad():
        for belief in beliefs_list:
            marginal_model = belief.marginalize(dims_to_integrate)
            zz = marginal_model(torch.from_numpy(pts).to(dtype=DTYPE)).cpu().numpy()
            Z = zz.reshape(resolution, resolution)
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
        # Individual color scaling for each subplot
        cf = ax.contourf(XX, YY, Z, levels=30)
        ax.set_title(f"t={k}")
        ax.set_xlabel("u[{}]".format(keep_pair[0]))
        ax.set_ylabel("u[{}]".format(keep_pair[1]))
    # Hide unused axes
    for k in range(t, n_rows*n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r][c].axis('off')
    fig.suptitle(f"2D marginal over time: {pair_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Note: No global colorbar since each subplot has its own scale
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_2d_particle_scatter_over_time(u_traj_list, keep_pair, pair_name,
                                       sample_limit=None, save_path=None, show_plot=True):
    """
    Plot 2D marginal trajectory particles (in U-space) over time for a given
    pair of state indices. u_traj_list is a list of length T with arrays (N_t, dy).
    keep_pair: tuple of two indices to keep for scatter.
    """
    keep_pair = tuple(int(i) for i in keep_pair)
    T = len(u_traj_list)

    # Determine consistent axis limits from data (clipped to [0,1])
    xs_all = []
    ys_all = []
    for U in u_traj_list:
        xs_all.append(U[:, keep_pair[0]])
        ys_all.append(U[:, keep_pair[1]])
    x_min = float(np.clip(np.min([x.min() for x in xs_all]), 0.0, 1.0))
    x_max = float(np.clip(np.max([x.max() for x in xs_all]), 0.0, 1.0))
    y_min = float(np.clip(np.min([y.min() for y in ys_all]), 0.0, 1.0))
    y_max = float(np.clip(np.max([y.max() for y in ys_all]), 0.0, 1.0))
    # Ensure some padding
    pad = 0.02
    x_min, x_max = max(0.0, x_min - pad), min(1.0, x_max + pad)
    y_min, y_max = max(0.0, y_min - pad), min(1.0, y_max + pad)

    n_cols = min(5, T)
    n_rows = (T + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 3.0*n_rows), squeeze=False)
    for k in range(T):
        U = u_traj_list[k]
        if sample_limit is not None and U.shape[0] > sample_limit:
            idx = np.random.choice(U.shape[0], size=sample_limit, replace=False)
            Uplot = U[idx]
        else:
            Uplot = U
        r = k // n_cols
        c = k % n_cols
        ax = axes[r][c]
        ax.scatter(Uplot[:, keep_pair[0]], Uplot[:, keep_pair[1]], s=3, alpha=0.5)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title(f"t={k}")
        ax.set_xlabel("u[{}]".format(keep_pair[0]))
        ax.set_ylabel("u[{}]".format(keep_pair[1]))
    for k in range(T, n_rows*n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r][c].axis('off')
    fig.suptitle(f"2D particle scatter over time: {pair_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    transition_model = SumBetaSOSModel(dy=dim, dx=dim, n=n, n_terms=n_terms, 
                                      min_alpha_beta=0.4, max_alpha_beta=100.0, 
                                      mu=0.1, min_Q_eigval=1e-8, 
                                      regularization_weight=regularization_weight)

    print("Training transition model...")
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    _, best_trans_loss_1 = optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs//2)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
    _, best_trans_loss_2 = optimize(transition_model, Up_dataloader_refine, trans_optimizer, epochs=n_epochs//2)

    transition_model.to(device=torch.device("cpu"))
    print("Done training transition model")

    init_state_model = SumBetaSOSModel(dy=dim, dx=0, n=n, n_terms=n_terms, 
                                      conditional=False, reference_factor_model=transition_model, 
                                      min_alpha_beta=0.4, max_alpha_beta=100.0, 
                                      mu=0.1, min_Q_eigval=1e-8, 
                                      regularization_weight=regularization_weight)

    print("Training init state model...")
    init_state_model.to(device=device, dtype=DTYPE)
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    _, best_init_loss_1 = optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs//2)
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-4)
    _, best_init_loss_2 = optimize(init_state_model, U0_dataloader_refine, init_optimizer, epochs=n_epochs//2)

    init_state_model.to(device=torch.device("cpu"))
    print("Done training init state model")

    # Propagate beliefs
    beliefs = [init_state_model]
    for i in range(timesteps):
        beliefs.append(transition_model.propagate(beliefs[i], n_terms=n_terms))

    # Calculate AUC and test log-likelihood for each belief at corresponding timestep
    print("\nBelief metrics:")
    auc_values = []
    test_avg_log_liks = []
    for i, belief in enumerate(beliefs):
        with torch.no_grad():
            auc = mc_auc(4, lambda u : belief(torch.from_numpy(u)).numpy(), n_samples=10000)
            auc_values.append(float(auc))
            U_test_i = u_test_traj_data[i]
            ll = np.mean(np.log(belief(torch.from_numpy(U_test_i)).numpy() + 1e-12))
            test_avg_log_liks.append(float(ll))
            print(f"t={i}: auc={auc:.6f}, test_avg_loglik={ll:.6f}")

    # Generate and save plots
    print("Generating plots...")
    
    # Marginal plots
    plot_2d_marginals_over_time(beliefs, (0, 2), "cart_pos_pole_angle",
                                resolution=60,
                                save_path=os.path.join(experiment_dir, "marginals_cart_pos_pole_angle.png"),
                                show_plot=False)
    plot_2d_marginals_over_time(beliefs, (1, 3), "cart_vel_pole_angular_vel",
                                resolution=60,
                                save_path=os.path.join(experiment_dir, "marginals_cart_vel_pole_angular_vel.png"),
                                show_plot=False)
    
    # Particle scatter plots
    plot_2d_particle_scatter_over_time(u_traj_data, (0, 2), "cart_pos_pole_angle_particles",
                                       sample_limit=10000,
                                       save_path=os.path.join(experiment_dir, "mc_particles_cart_pos_pole_angle.png"),
                                       show_plot=False)
    plot_2d_particle_scatter_over_time(u_traj_data, (1, 3), "cart_vel_pole_angular_vel_particles",
                                       sample_limit=10000,
                                       save_path=os.path.join(experiment_dir, "mc_particles_cart_vel_pole_angular_vel.png"),
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

    # System model
    system = CartPole(dt=0.1, covariance=0.01 * np.eye(4))

    # Dimension
    dim = system.dim()


    # Number of trajectories (pool); will subsample per experiment to allow tuning n_traj
    n_traj_pool = max(n_traj_values)

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        # 4D state: [cart_pos, cart_vel, pole_angle, pole_angular_vel] near equilibrium
        mean = np.array([0.0, 0.0, 0.0, 0.0])
        cov = np.diag([0.1, 0.1, 0.1, 0.1])
        return multivariate_normal.rvs(mean=mean, cov=cov)

    # Sample trajectory data once for all experiments
    print("Sampling trajectory data pool...")
    traj_data_pool = sample_trajectories(system, init_state_sampler, timesteps, n_traj_pool)
    test_traj_data_pool = sample_trajectories(system, init_state_sampler, timesteps + 1, n_traj_pool)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data_pool), variance_pads=[2.0, 2.0, 5.0, 10.0])

    # GPU setup
    use_gpu = True
    print("Using GPU: ", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
    print("device: ", device)

    # Create root directory for all experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("benchmarks", f"sos_4D_cross_val_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)
    
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
