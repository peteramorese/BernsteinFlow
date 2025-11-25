from bernstein_flow.DistributionTransform import GaussianDistTransform
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
from sos_form.SumBetaModel import SumBetaSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, mc_auc, avg_log_likelihood

from .Systems import PlanarQuadrotor, sample_trajectories, sample_io_pairs
from .Visualization import plot_2d_marginals_over_time, plot_2d_particle_scatter_over_time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import os
import time
import traceback


DTYPE = torch.float64


def run_trials_sos(train_data, test_data, save_directory, num_trials, gdt, n, n_epochs_init=100, n_epochs_tran_coarse=100, n_epochs_tran_fine=50,
               use_gpu=True, batch_size=256, batch_size_refine=2048, tran_params={
                   "min_alpha_beta": 0.1,
                   "max_alpha_beta": 80.0,
                   "mu": 0.1,
                   "min_Q_eigval": 1e-8,
                   "regularization_weight": 1e-4
               },
               init_params={
                   "min_alpha_beta": 0.4,
                   "max_alpha_beta": 100.0,
                   "mu": 0.1,
                   "min_Q_eigval": 1e-8,
                   "regularization_weight": 1e-4
               }):
    """
    Run multiple trials of training BetaSOSModels, belief propagation, and evaluation.
    
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
    gdt : GaussianDistTransform
        GaussianDistTransform object to use for transforming data to U space.
    n : int
        Model parameter n for BetaSOSModel (default: 15).
    n_epochs_init : int
        Number of epochs for initial state model training (default: 100).
    n_epochs_tran_coarse : int
        Number of coarse epochs for transition model (default: 100).
    n_epochs_tran_fine : int
        Number of fine epochs for transition model (default: 50).
    use_gpu : bool
        Whether to use GPU if available (default: True).
    batch_size : int
        Batch size for training (default: 256).
    batch_size_refine : int
        Batch size for refinement training (default: 2048).
    
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
    
    # Convert test data to U space
    u_test_data = [gdt.X_to_U(X_data) for X_data in test_data]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
    print(f"Using device: {device}")

    # Create the data matrices for training
    X0_data = train_data[0]
    Xp_data = create_transition_data_matrix(train_data[:training_timesteps])
    
    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data)  # Initial state data
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, dim:]), gdt.X_to_U(Xp_data[:, :dim])])  # Transition kernel data
    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)

    # Initialize results array: (num_trials, num_timesteps)
    # num_timesteps includes initial belief (t=0) + propagated beliefs (t=1, ..., t=num_timesteps-1)
    negative_log_likelihoods = np.zeros((num_trials, num_timesteps))
    prop_times = np.zeros((num_trials, num_timesteps))
    
    # Run trials
    trial = 0
    while trial < num_trials:
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*60}\n")
        
        # Create data loaders
        U0_dataset = TensorDataset(U0_data_torch)
        U0_dataloader = DataLoader(U0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        U0_dataloader_refine = DataLoader(U0_dataset, batch_size=batch_size_refine, shuffle=True, pin_memory=use_gpu)
        
        Up_dataset = TensorDataset(Up_data_torch)
        Up_dataloader = DataLoader(Up_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        Up_dataloader_refine = DataLoader(Up_dataset, batch_size=batch_size_refine, shuffle=True, pin_memory=use_gpu)
        
        # Create and train transition model
        transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, **tran_params)
        
        print("Training transition model...")
        transition_model.to(device=device, dtype=DTYPE)
        trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
        optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran_coarse, print_interval=10)
        trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
        optimize(transition_model, Up_dataloader_refine, trans_optimizer, epochs=n_epochs_tran_fine, print_interval=10)
        transition_model.to(device=torch.device("cpu"))
        print("Done training transition model\n")
        
        # Check if transition model is PSD
        is_transition_psd = transition_model.is_psd()
        print(f"Transition model is PSD: {is_transition_psd}")
        if not is_transition_psd:
            print("Transition model is not PSD. Discarding trial and retrying...\n")
            del transition_model
            torch.cuda.empty_cache() if use_gpu and torch.cuda.is_available() else None
            continue
        
        # Create and train initial state model
        init_state_model = BetaSOSModel(dy=dim, dx=0, n=n, conditional=False, reference_factor_model=transition_model, **init_params)

        print("Training init state model...")
        init_state_model.to(device=device, dtype=DTYPE)
        init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
        optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init, print_interval=10)
        init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-4)
        optimize(init_state_model, U0_dataloader_refine, init_optimizer, epochs=n_epochs_tran_fine, print_interval=10)
        init_state_model.to(device=torch.device("cpu"))
        print("Done training init state model\n")
        
        # Check if initial state model is PSD
        is_init_psd = init_state_model.is_psd()
        print(f"Initial state model is PSD: {is_init_psd}")
        if not is_init_psd:
            print("Initial state model is not PSD. Discarding trial and retrying...\n")
            del transition_model, init_state_model
            torch.cuda.empty_cache() if use_gpu and torch.cuda.is_available() else None
            continue
        
        # Both models are valid (PSD), proceed with belief propagation and evaluation
        print("Both models are valid (PSD). Proceeding with belief propagation...\n")
        
        # Propagate beliefs
        # We propagate (num_timesteps - 1) times to get beliefs at t=0, t=1, ..., t=num_timesteps-1
        beliefs = [init_state_model]
        for i in range(num_timesteps - 1):
            start = time.time()
            try:
                beliefs.append(transition_model.propagate(beliefs[i]))
                prop_times[trial, i] = time.time() - start
            except Exception as e:
                print(f"Propagation failed at timestep {i+1}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                # Set the failed propagation time to NaN
                prop_times[trial, i] = np.nan
                # Set remaining NLL and prop_times to NaN
                for remaining_k in range(i+1, num_timesteps):
                    negative_log_likelihoods[trial, remaining_k] = np.nan
                    if remaining_k < num_timesteps - 1:
                        prop_times[trial, remaining_k] = np.nan
                break
        
        # Evaluate log likelihood for each belief on test data
        print("Evaluating log likelihoods...")
        for i, belief in enumerate(beliefs):
            with torch.no_grad():
                # Get test data at corresponding timestep
                #U_test_i = u_test_data[i]
                # Evaluate log likelihood
                #nll = -np.mean(np.log(belief(torch.from_numpy(U_test_i)).numpy() + 1e-12))
                def np_belief(u):
                    return belief(torch.from_numpy(u)).numpy()
                nll = avg_log_likelihood(test_data[i], lambda x : gdt.x_density(x, np_belief))
                negative_log_likelihoods[trial, i] = nll
                print(f"  Belief {i}: avg_log_likelihood = {nll:.6f}")
        
        # Save figures only for the first trial
        #if trial == 0:
        os.makedirs(save_directory, exist_ok=True)
        print(f"\nSaving figures to {save_directory}...")
        # Using state indices: [0:px, 1:pz, 2:theta, 3:vx, 4:vz, 5:omega]
        plot_2d_marginals_over_time(beliefs, (0, 1), "px_pz",
                                    resolution=60,
                                    save_path=os.path.join(save_directory, "marginals_px_pz.png"),
                                    show_plot=False)
        plot_2d_marginals_over_time(beliefs, (3, 4), "vx_vz",
                                    resolution=60,
                                    save_path=os.path.join(save_directory, "marginals_vx_vz.png"),
                                    show_plot=False)
        plot_2d_marginals_over_time(beliefs, (2, 5), "theta_omega",
                                    resolution=60,
                                    save_path=os.path.join(save_directory, "marginals_theta_omega.png"),
                                    show_plot=False)
        print("Figures saved.\n")
        
        # Clean up memory
        del transition_model, init_state_model, beliefs
        torch.cuda.empty_cache() if use_gpu and torch.cuda.is_available() else None
        
        # Increment trial counter only after successful completion
        trial += 1
    
    return negative_log_likelihoods, prop_times


if __name__ == "__main__":

    # System model
    system = PlanarQuadrotor(dt=0.01, covariance=0.05 * np.eye(6), waypoint=np.array([5.0, 0.0]))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj_train = 400
    n_traj_test = 1000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 50

    # Variance pads
    variance_pads = [5.2, 5.2, 3.1, 5.2, 5.2, 3.1]

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        # 6D state: [px, pz, theta, vx, vz, omega] near hover at origin
        mean = np.array([0.0, 0.0, 0.1, 50.0, 0.0, 0.0])
        cov = np.diag([0.1, 0.1, 0.05, 0.1, 0.1, 0.05])
        return multivariate_normal.rvs(mean=mean, cov=cov)

    #io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-5.0, -5.0], region_uppers=[5.0, 5.0])
    traj_data_train = sample_trajectories(system, init_state_sampler, timesteps, n_traj_train)
    traj_data_test = sample_trajectories(system, init_state_sampler, timesteps, n_traj_test)

    # Create MC particle figures
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data_test), variance_pads=variance_pads)

    u_traj_data_test = [gdt.X_to_U(X_data) for X_data in traj_data_test]

    os.makedirs("figures/sos_6D", exist_ok=True)
    plot_2d_particle_scatter_over_time(u_traj_data_test, (0, 1), "px_pz_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_6D/mc_particles_px_pz.png",
                                       show_plot=True)
    plot_2d_particle_scatter_over_time(u_traj_data_test, (3, 4), "vx_vz_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_6D/mc_particles_vx_vz.png",
                                       show_plot=True)
    plot_2d_particle_scatter_over_time(u_traj_data_test, (2, 5), "theta_omega_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_6D/mc_particles_theta_omega.png",
                                       show_plot=True)

    negative_log_likelihoods, prop_times = run_trials_sos(traj_data_train, traj_data_test, "figures/sos_6D", num_trials=10, gdt=gdt, n=15, num_epochs_init=n_epochs_init, num_epochs_tran_coarse=n_epochs_tran)
    print(f"Negative log likelihoods: {negative_log_likelihoods}, prop times: {prop_times}")