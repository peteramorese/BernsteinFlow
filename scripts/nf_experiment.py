import numpy as np
from scipy.stats import gaussian_kde
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import traceback

from bernstein_flow.NormalizingFlow import ConditionalNormalizingFlow, optimize
from bernstein_flow.Tools import create_transition_data_matrix, avg_log_likelihood, mc_auc
from bernstein_flow.Propagate import propagate_nf
from .Systems import SecondOrderDubinsTrailer
from scipy.spatial import Rectangle

DTYPE = torch.float32  # nflows typically uses float32


def kde_from_particles(particles: np.ndarray, bandwidth="scott"):
    """
    Fit a Gaussian kernel density estimator to a set of particles.

    Args:
        particles: array of shape (n_particles, dim)
        bandwidth: 'scott', 'silverman', or float / callable passed to gaussian_kde.bw_method

    Returns:
        kde: an object with methods `pdf(x)` and `logpdf(x)`.

        - pdf(x): x can be shape (dim,) or (n_points, dim)
                  returns scalar or array of shape (n_points,)
    """
    particles = np.asarray(particles)
    assert particles.ndim == 2, f"particles must be 2D, got {particles.shape}"
    dim = particles.shape[1]

    # gaussian_kde expects shape (dim, n_samples)
    samples = particles.T  # (dim, n_particles)

    kde = gaussian_kde(samples, bw_method=bandwidth)

    class KDEWrapper:
        def __init__(self, kde, dim):
            self.kde = kde
            self.dim = dim

        def pdf(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            assert x.shape[1] == self.dim, f"Expected dim {self.dim}, got {x.shape[1]}"
            # gaussian_kde wants (dim, n_points)
            return self.kde.evaluate(x.T)

        def logpdf(self, x: np.ndarray) -> np.ndarray:
            p = self.pdf(x)
            return np.log(p + 1e-300)  # avoid log(0)

    return KDEWrapper(kde, dim)


def run_trials_nf(train_data, test_data, init_state_sampler, save_directory, num_trials,
                  n_particles=1000, n_epochs_tran=20, num_layers=8, hidden_features=128,
                  use_gpu=True, batch_size=512, n_added_samples=1, kde_bandwidth="scott"):
    """
    Run multiple trials of training ConditionalNormalizingFlow, belief propagation, and evaluation.
    
    Parameters:
    -----------
    train_data : list of np.ndarray
        Training trajectory data. Each element is (N, dim) array for timestep t.
    test_data : list of np.ndarray
        Test trajectory data. Each element is (M, dim) array for timestep t.
    init_state_sampler : callable
        Function that returns a single sample from the initial state distribution.
        Should return np.ndarray of shape (dim,).
    save_directory : str
        Directory to save figures (only saved for first trial).
    num_trials : int
        Number of trials to run.
    n_particles : int
        Number of particles to use for belief representation (default: 1000).
    n_epochs_tran : int
        Number of epochs for transition model training (default: 20).
    num_layers : int
        Number of layers in the normalizing flow (default: 8).
    hidden_features : int
        Hidden dimension in the normalizing flow (default: 128).
    use_gpu : bool
        Whether to use GPU if available (default: True).
    batch_size : int
        Batch size for training (default: 512).
    n_added_samples : int
        Number of samples to generate per particle during propagation (default: 1).
    kde_bandwidth : str or float
        Bandwidth method for KDE ('scott', 'silverman', or float, default: 'scott').
    
    Returns:
    --------
    negative_log_likelihoods : np.ndarray
        Array of shape (num_trials, num_timesteps) containing average negative log likelihood
        for each belief (timestep) for each trial. The timesteps include t=0 (initial belief)
        through t=num_timesteps-1 (after num_timesteps-1 propagations).
    prop_times : np.ndarray
        Array of shape (num_trials, num_timesteps-1) containing propagation times for each timestep.
    """
    # Get dimension from first timestep
    dim = train_data[0].shape[1]
    num_timesteps = len(test_data)  # Number of timesteps in test data (includes t=0)
    training_timesteps = len(train_data)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
    print(f"Using device: {device}")
    device_str = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Create the data matrices for training (in original state space, no GDT)
    X0_data = train_data[0]  # Initial state data
    Xp_data = create_transition_data_matrix(train_data[:training_timesteps])  # Transition data: [x, x']

    # Normalize data for better NF training (NFs work better with normalized data)
    # Store normalization stats for later use
    x_mean = Xp_data[:, :dim].mean(axis=0, keepdims=True)
    x_std = Xp_data[:, :dim].std(axis=0, keepdims=True) + 1e-8
    xp_mean = Xp_data[:, dim:].mean(axis=0, keepdims=True)
    xp_std = Xp_data[:, dim:].std(axis=0, keepdims=True) + 1e-8
    
    Xp_data_normalized = np.zeros_like(Xp_data)
    Xp_data_normalized[:, :dim] = (Xp_data[:, :dim] - x_mean) / x_std
    Xp_data_normalized[:, dim:] = (Xp_data[:, dim:] - xp_mean) / xp_std

    # Initialize results array: (num_trials, num_timesteps)
    # num_timesteps includes initial belief (t=0) + propagated beliefs (t=1, ..., t=num_timesteps-1)
    negative_log_likelihoods = np.zeros((num_trials, num_timesteps))
    prop_times = np.zeros((num_trials, num_timesteps - 1))
    
    # Run trials
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*60}\n")
        
        # Create data loader for transition data: [x, x'] for p(x' | x)
        # Use normalized data for training
        Xp_data_torch = torch.tensor(Xp_data_normalized, dtype=DTYPE)
        Xp_dataset = TensorDataset(Xp_data_torch)
        Xp_dataloader = DataLoader(Xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        # Create and train transition model (conditional normalizing flow)
        print("Creating transition model...")
        transition_model = ConditionalNormalizingFlow(
            dim_x=dim,  # conditioning variable dimension
            dim_y=dim,  # target variable dimension
            num_layers=num_layers,
            hidden_features=hidden_features,
            device=device_str,
        )

        print("Training transition model...")
        transition_model.to(device)
        transition_model.device = device_str
        
        # Use learning rate schedule
        trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trans_optimizer, mode='min', factor=0.5, patience=20)
        
        # Train using optimize method
        optimize(transition_model, Xp_dataloader, trans_optimizer, epochs=n_epochs_tran, 
                 print_interval=5, scheduler=scheduler)
        
        print("Done training transition model\n")

        # Initialize belief particles from initial state distribution
        print("Initializing belief particles from initial state distribution...")
        initial_particles = np.array([init_state_sampler() for _ in range(n_particles)])

        # Propagate beliefs forward using particle sets
        print("Propagating beliefs...")
        beliefs_particles = [initial_particles.copy()]
        
        # Evaluate initial belief log likelihood
        print("Evaluating log likelihoods...")
        initial_kde = kde_from_particles(initial_particles, bandwidth=kde_bandwidth)
        nll_init = -avg_log_likelihood(test_data[0], initial_kde.pdf)
        negative_log_likelihoods[trial, 0] = nll_init
        
        # Calculate mc_auc for initial KDE belief
        # Use adaptive region based on particle distribution (with padding)
        particle_mins = initial_particles.min(axis=0)
        particle_maxes = initial_particles.max(axis=0)
        particle_std = initial_particles.std(axis=0)
        # Add padding: 5 standard deviations on each side
        padding = 5.0 * particle_std
        adaptive_mins = particle_mins - padding
        adaptive_maxes = particle_maxes + padding
        
        auc_init = mc_auc(dim, initial_kde.pdf, n_samples=10000, region=Rectangle(mins=adaptive_mins, maxes=adaptive_maxes))
        print(f"  Belief 0 (initial): avg_log_likelihood = {-nll_init:.6f}, mc_auc = {auc_init:.6f}")
        
        # Propagate for remaining timesteps
        for i in range(num_timesteps - 1):
            current_particles = beliefs_particles[-1]
            
            start = time.time()
            try:
                # Propagate using propagate_nf function
                next_particles = propagate_nf(
                    current_particles, 
                    transition_model, 
                    n_added_samples=n_added_samples,
                    x_mean=x_mean,
                    x_std=x_std,
                    xp_mean=xp_mean,
                    xp_std=xp_std,
                    dtype=DTYPE,
                    device=device
                )
                
                prop_times[trial, i] = time.time() - start
                beliefs_particles.append(next_particles)
                
                # Convert particles to PDF using KDE and evaluate log likelihood
                belief_kde = kde_from_particles(next_particles, bandwidth=kde_bandwidth)
                nll = -avg_log_likelihood(test_data[i + 1], belief_kde.pdf)
                negative_log_likelihoods[trial, i + 1] = nll
                
                # Calculate mc_auc for KDE belief
                # Use adaptive region based on particle distribution (with padding)
                particle_mins = next_particles.min(axis=0)
                particle_maxes = next_particles.max(axis=0)
                particle_std = next_particles.std(axis=0)
                # Add padding: 5 standard deviations on each side
                padding = 5.0 * particle_std
                adaptive_mins = particle_mins - padding
                adaptive_maxes = particle_maxes + padding
                
                auc = mc_auc(dim, belief_kde.pdf, n_samples=10000, region=Rectangle(mins=adaptive_mins, maxes=adaptive_maxes))
                print(f"  Belief {i + 1}: avg_log_likelihood = {-nll:.6f}, mc_auc = {auc:.6f}, prop_time = {prop_times[trial, i]:.4f}s")
                
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
        
        print("\nDone propagating beliefs\n")
        
        # Clean up memory
        del transition_model, beliefs_particles
        torch.cuda.empty_cache() if use_gpu and torch.cuda.is_available() else None
    
    return negative_log_likelihoods, prop_times


if __name__ == "__main__":
    from .Systems import PlanarQuadrotor, sample_trajectories
    from scipy.stats import multivariate_normal
    import os

    # System model
    #system = PlanarQuadrotor(dt=0.01, covariance=0.05 * np.eye(6), waypoint=np.array([5.0, 0.0]))
    system = SecondOrderDubinsTrailer(
        dt=0.3,
        L_t=1.0,
        v_ref=1.0,
        k_v=1.0,
        k_theta=2.0,
        sigma_v=0.1,
        sigma_omega=0.1,
        cov_scale=0.2
    )

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj_train = 400
    n_traj_test = 10000

    # Number of training epochs
    n_epochs_tran = 100

    # Time horizon
    training_timesteps = 10
    timesteps = 15

    # Number of particles for belief representation
    n_particles = 1000

    def init_state_sampler():
        mean = np.array([0.0, 0.0, 1.0, 0.0, 10.0, -0.5])
        cov = np.diag([0.1, 0.1, 0.05, 0.1, 0.1, 0.05])
        return multivariate_normal.rvs(mean=mean, cov=cov)
    #def init_state_sampler():
    #    # 6D state: [px, pz, theta, vx, vz, omega] near hover at origin
    #    mean = np.array([0.0, 0.0, 0.1, 50.0, 0.0, 0.0])
    #    cov = np.diag([0.1, 0.1, 0.05, 0.1, 0.1, 0.05])
    #    return multivariate_normal.rvs(mean=mean, cov=cov)

    # Sample trajectory data
    print("Sampling training trajectories...")
    traj_data_train = sample_trajectories(system, init_state_sampler, training_timesteps, n_traj_train)
    print("Sampling test trajectories...")
    traj_data_test = sample_trajectories(system, init_state_sampler, timesteps, n_traj_test)

    # Create output directory
    os.makedirs("figures/nf_6D", exist_ok=True)

    # Run trials
    print(f"\n{'='*60}")
    print("Starting NF experiments with 6D PlanarQuadrotor")
    print(f"{'='*60}\n")
    
    negative_log_likelihoods, prop_times = run_trials_nf(
        train_data=traj_data_train,
        test_data=traj_data_test,
        init_state_sampler=init_state_sampler,
        save_directory="figures/nf_6D",
        num_trials=10,
        n_particles=n_particles,
        n_epochs_tran=n_epochs_tran,
        num_layers=8,
        hidden_features=128,
        use_gpu=True,
        batch_size=512,
        n_added_samples=1,
        kde_bandwidth="scott"
    )
    
    print(f"\n{'='*60}")
    print("Experiment Results")
    print(f"{'='*60}")
    print(f"Negative log likelihoods shape: {negative_log_likelihoods.shape}")
    print(f"Negative log likelihoods:\n{negative_log_likelihoods}")
    print(f"\nPropagation times shape: {prop_times.shape}")
    print(f"Propagation times:\n{prop_times}")
    print(f"\nMean NLL per timestep: {np.nanmean(negative_log_likelihoods, axis=0)}")
    print(f"Std NLL per timestep: {np.nanstd(negative_log_likelihoods, axis=0)}")
    print(f"Mean prop time per timestep: {np.nanmean(prop_times, axis=0)}")
    print(f"Std prop time per timestep: {np.nanstd(prop_times, axis=0)}")