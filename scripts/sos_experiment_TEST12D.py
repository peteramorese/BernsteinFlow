from bernstein_flow.DistributionTransform import GaussianDistTransform
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
from sos_form.SumBetaModel import SumBetaSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, mc_auc, avg_log_likelihood

from .Systems import Quadcopter, SecondOrderDubinsTrailer, sample_trajectories, sample_io_pairs
from .Visualization import plot_2d_marginals_over_time, plot_2d_particle_scatter_over_time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import os
import time
import traceback
from scipy.linalg import block_diag


DTYPE = torch.float64


def print_num_parameters(model, model_name="Model"):
    """
    Print the number of parameters in a PyTorch model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to count parameters for.
    model_name : str
        Name of the model to display in the output (default: "Model").
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} - Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")


def run_trials_sos(train_data, test_data, save_directory, num_trials, gdt, n, n_epochs_init=100, n_epochs_tran_coarse=100, n_epochs_tran_fine=50,
               use_gpu=True, batch_size=256, batch_size_refine=2048, save_figures=False, tran_params={
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
        
        try:
            print("Training transition model...")
            print_num_parameters(transition_model, "Transition model")
            print()
            transition_model.to(device=device, dtype=DTYPE)
            trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
            optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran_coarse, print_interval=1, not_psd_threshold=10)
            trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-3)
            optimize(transition_model, Up_dataloader_refine, trans_optimizer, epochs=n_epochs_tran_fine, print_interval=10, not_psd_threshold=10)
            transition_model.to(device=torch.device("cpu"))
            print("Done training transition model")
        except Exception as e:
            print(f"Error during transition model training on trial {trial + 1}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            # Clean up and retry this trial
            del transition_model
            torch.cuda.empty_cache() if use_gpu and torch.cuda.is_available() else None
            continue
        
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

        try:
            print("Training init state model...")
            init_state_model.to(device=device, dtype=DTYPE)
            init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-1)
            optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init, print_interval=10, not_psd_threshold=10)
            init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-3)
            optimize(init_state_model, U0_dataloader_refine, init_optimizer, epochs=n_epochs_tran_fine, print_interval=10, not_psd_threshold=10)
            init_state_model.to(device=torch.device("cpu"))
            print("Done training init state model\n")
        except Exception as e:
            print(f"Error during init state model training on trial {trial + 1}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            # Clean up and retry this trial
            del transition_model, init_state_model
            torch.cuda.empty_cache() if use_gpu and torch.cuda.is_available() else None
            continue
        
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
                
                # Calculate mc_auc for this belief (in U space)
                auc = mc_auc(dim, np_belief, n_samples=10000)
                print(f"  Belief {i}: avg_log_likelihood = {nll:.6f}, mc_auc = {auc:.6f}")
        
        # Save figures only for the first trial
        #if trial == 0:
        if save_figures:
            os.makedirs(save_directory, exist_ok=True)
            print(f"\nSaving figures to {save_directory}...")
            # Using state indices: [0:px, 1:pz, 2:theta, 3:vx, 4:vz, 5:omega]
            #plot_2d_marginals_over_time(beliefs, (0, 1), "px_pz",
            #                            resolution=60,
            #                            save_path=os.path.join(save_directory, "marginals_px_pz.png"),
            #                            show_plot=False)
            #plot_2d_marginals_over_time(beliefs, (3, 4), "vx_vz",
            #                            resolution=60,
            #                            save_path=os.path.join(save_directory, "marginals_vx_vz.png"),
            #                            show_plot=False)
            #plot_2d_marginals_over_time(beliefs, (2, 5), "theta_omega",
            #                            resolution=60,
            #                            save_path=os.path.join(save_directory, "marginals_theta_omega.png"),
            #                            show_plot=False)
            plot_2d_marginals_over_time(beliefs, (0, 1), "px_py",
                                        resolution=60,
                                        save_path=os.path.join(save_directory, "marginals_px_py.png"),
                                        show_plot=False)
            plot_2d_marginals_over_time(beliefs, (3, 4), "vx_vy",
                                        resolution=60,
                                        save_path=os.path.join(save_directory, "marginals_vx_vy.png"),
                                        show_plot=False)
            plot_2d_marginals_over_time(beliefs, (9, 10), "p_q",
                                        resolution=60,
                                        save_path=os.path.join(save_directory, "marginals_p_q.png"),
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

    #system = PlanarQuadrotor(
    #    dt=0.03, 
    #    covariance=0.05 * np.eye(6), 
    #    waypoint=np.array([5.0, 5.0]),
    #    m=1.0,
    #    I=0.03,a
    #    ell=0.2,
    #    g=9.81,
    #    c_v=0.05,
    #    c_w=0.12,
    #)
    #system.kp_pos = np.array([1.0, 1.0])
    #system.kd_pos = np.array([0.5, 0.5])
    #system.kp_theta = 3.0
    #system.kd_theta = 2.0

    #system = SecondOrderDubinsTrailer(
    #    dt=0.2,
    #    L_t=1.0,
    #    v_ref=1.0,
    #    k_v=1.0,
    #    k_theta=2.0,
    #    sigma_v=0.1,
    #    sigma_omega=0.5,
    #    cov_scale=0.5
    #)

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

    # Number of trajectories
    n_traj_train = 1000 #4000
    n_traj_test = 10000

    # Number of training epochs
    n_epochs_init = 400
    n_epochs_tran = 100
    n_epochs_refine = 30

    batch_size = 1024
    batch_size_refine = 2048

    tran_params={
        "min_alpha_beta": 0.4,
        "max_alpha_beta": 400.0,
        "mu": 0.1,
        "min_Q_eigval": 1e-5,
        "regularization_weight": 1e-4,
        "initialization_scale": -4.0
    }
    init_params={
        "min_alpha_beta": 0.4,
        "max_alpha_beta": 400.0,
        "mu": 0.1,
        "min_Q_eigval": 1e-5,
        "regularization_weight": 1e-4,
        "initialization_scale": -3.0
    }

    # Variance pads
    variance_pads = [10.0, 10.0, 7.0, 10.0, 10.0, 7.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5]

    # Time horizon
    training_timesteps = 10
    timesteps = 15

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

    #io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-5.0, -5.0], region_uppers=[5.0, 5.0])
    traj_data_train = sample_trajectories(system, init_state_sampler, training_timesteps, n_traj_train)
    traj_data_test = sample_trajectories(system, init_state_sampler, timesteps, n_traj_test)

    # Create MC particle figures
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data_test), variance_pads=variance_pads)

    u_traj_data_test = [gdt.X_to_U(X_data) for X_data in traj_data_test]

    os.makedirs("figures/sos_12D", exist_ok=True)
    plot_2d_particle_scatter_over_time(u_traj_data_test, (0, 1), "px_pz_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_12D/mc_particles_px_pz.png",
                                       show_plot=False)
    plot_2d_particle_scatter_over_time(u_traj_data_test, (3, 4), "vx_vy_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_12D/mc_particles_vx_vy.png",
                                       show_plot=False)
    plot_2d_particle_scatter_over_time(u_traj_data_test, (9, 10), "p_q_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_12D/mc_particles_p_q.png",
                                       show_plot=False)

    negative_log_likelihoods, prop_times = run_trials_sos(traj_data_train, traj_data_test, "figures/sos_12D", 
        num_trials=10, 
        gdt=gdt, 
        n=20, 
        n_epochs_init=n_epochs_init, 
        n_epochs_tran_coarse=n_epochs_tran, 
        n_epochs_tran_fine=n_epochs_refine, 
        save_figures=True, 
        tran_params=tran_params, 
        init_params=init_params, 
        batch_size=batch_size, 
        batch_size_refine=batch_size_refine)
    print(f"Negative log likelihoods: {negative_log_likelihoods}, prop times: {prop_times}")