import numpy as np
from scipy.stats import gaussian_kde
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import traceback
import matplotlib.pyplot as plt
import os

from bernstein_flow.NormalizingFlow import ConditionalNormalizingFlow, optimize
from bernstein_flow.Tools import create_transition_data_matrix, avg_log_likelihood, mc_auc
from bernstein_flow.Propagate import propagate_nf
from .Systems import SecondOrderDubinsTrailer
from .Visualization import plot_2d_particle_scatter_over_time
from scipy.spatial import Rectangle
from sklearn.mixture import GaussianMixture

DTYPE = torch.float32  # nflows typically uses float32


#def kde_from_particles(particles: np.ndarray, bandwidth="scott"):
#    """
#    Fit a Gaussian kernel density estimator to a set of particles.
#
#    Args:
#        particles: array of shape (n_particles, dim)
#        bandwidth: 'scott', 'silverman', or float / callable passed to gaussian_kde.bw_method
#
#    Returns:
#        kde: an object with methods `pdf(x)` and `logpdf(x)`.
#
#        - pdf(x): x can be shape (dim,) or (n_points, dim)
#                  returns scalar or array of shape (n_points,)
#    """
#    particles = np.asarray(particles)
#
#    # Clamp all particles to be within 10 standard deviations of the mean
#    mean = particles.mean(axis=0)
#    std = particles.std(axis=0)
#    particles = np.clip(particles, mean - 5 * std, mean + 5 * std)
#
#    assert particles.ndim == 2, f"particles must be 2D, got {particles.shape}"
#    dim = particles.shape[1]
#
#    # gaussian_kde expects shape (dim, n_samples)
#    samples = particles.T  # (dim, n_particles)
#
#    kde = gaussian_kde(samples, bw_method=bandwidth)
#
#    #def regularize_kde_covariance(kde, eps=1e-6):
#    #    """Add eps*I to the KDE covariance to prevent blow-ups."""
#    #    cov = kde.covariance
#    #    d = cov.shape[0]
#
#    #    # Regularize covariance
#    #    cov_reg = cov + eps * np.eye(d)
#
#    #    # Store updates back into KDE
#    #    kde.covariance = cov_reg
#    #    kde.inv_cov = np.linalg.inv(cov_reg)
#
#    #    # Recompute normalization constant
#    #    det = np.linalg.det(cov_reg)
#    #    kde._norm_factor = np.sqrt(det * (2 * np.pi) ** d) * kde.n
#
#    #regularize_kde_covariance(kde, eps=1e-6)
#
#    class KDEWrapper:
#        def __init__(self, kde, dim):
#            self.kde = kde
#            self.dim = dim
#
#        def pdf(self, x: np.ndarray) -> np.ndarray:
#            x = np.asarray(x)
#            if x.ndim == 1:
#                x = x.reshape(1, -1)
#            assert x.shape[1] == self.dim, f"Expected dim {self.dim}, got {x.shape[1]}"
#            # gaussian_kde wants (dim, n_points)
#            return self.kde.evaluate(x.T)
#
#        def logpdf(self, x: np.ndarray) -> np.ndarray:
#            p = self.pdf(x)
#            return np.log(p + 1e-300)  # avoid log(0)
#
#    return KDEWrapper(kde, dim)

def kde_from_particles(
    particles: np.ndarray,
    bandwidth="scott",
    min_bw_factor = 0.5
):
    """
    Fit a Gaussian kernel density estimator to a set of particles.

    Args:
        particles: array of shape (n_particles, dim)
        bandwidth: 'scott', 'silverman', or float / callable passed to gaussian_kde.bw_method
        min_bw_factor: if not None, enforce that the scalar bandwidth factor is at least this value.
                       This applies to 'scott', 'silverman', or a scalar bandwidth.

    Returns:
        kde: an object with methods `pdf(x)` and `logpdf(x)`.
    """
    particles = np.asarray(particles)

    mean = particles.mean(axis=0)
    std = particles.std(axis=0)
    particles = np.clip(particles, mean - 5 * std, mean + 5 * std)

    assert particles.ndim == 2, f"particles must be 2D, got {particles.shape}"
    dim = particles.shape[1]

    # gaussian_kde expects shape (dim, n_samples)
    samples = particles.T  # (dim, n_particles)

    # Build bw_method with optional minimum factor
    bw_method = bandwidth
    if min_bw_factor is not None:
        # Case 1: bandwidth is one of the built-in rules
        if isinstance(bandwidth, str) and bandwidth in ("scott", "silverman"):
            def bw(kde):
                if bandwidth == "scott":
                    base = kde.scotts_factor()
                else:
                    base = kde.silverman_factor()
                return max(base, min_bw_factor)
            bw_method = bw

        # Case 2: bandwidth is a scalar
        elif isinstance(bandwidth, (int, float)):
            bw_method = max(float(bandwidth), min_bw_factor)
        # If it's a callable, we leave it alone and assume you know what you're doing

    kde = gaussian_kde(samples, bw_method=bw_method)

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

def gmm_from_particles(particles: np.ndarray,
                       n_components: int = 100,
                       reg_covar: float = 1e-4):
    """
    Fit a Gaussian Mixture Model to a set of particles.

    Args:
        particles: array of shape (n_particles, dim)
        n_components: number of Gaussian components (default: 3)
        reg_covar: regularization for covariance matrices (default: 1e-4)

    Returns:
        gmm: an object with methods `pdf(x)` and `logpdf(x)`.
    """
    particles = np.asarray(particles)
    assert particles.ndim == 2, f"particles must be 2D, got {particles.shape}"
    
    # Ensure we have enough particles for the number of components
    n_particles = particles.shape[0]
    n_components = min(n_components, n_particles)
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=reg_covar,
        max_iter=500
    )
    gmm.fit(particles)

    class GMMDensity:
        def __init__(self, gmm, dim):
            self.gmm = gmm
            self.dim = dim

        def pdf(self, x):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            assert x.shape[1] == self.dim, f"Expected dim {self.dim}, got {x.shape[1]}"
            return np.exp(self.gmm.score_samples(x))

        def logpdf(self, x):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            assert x.shape[1] == self.dim, f"Expected dim {self.dim}, got {x.shape[1]}"
            return self.gmm.score_samples(x)

    return GMMDensity(gmm, particles.shape[1])


def plot_2d_marginals_over_time_nf(beliefs_particles_list, keep_pair, pair_name,
                                   resolution=60, save_path=None, show_plot=True, gmm_n_components=3, gmm_reg_covar=1e-4):
    """
    Plot 2D marginal densities over time for given dimension pair using particle beliefs.
    This creates proper marginals by extracting the 2D coordinates and creating 2D GMMs.
    
    Args:
        beliefs_particles_list: List of particle arrays, each of shape (n_particles, dim)
        keep_pair: tuple of two indices to keep (others are integrated out)
        pair_name: string for titles/filenames
        resolution: grid resolution
        save_path: path to save figure
        show_plot: whether to show the plot
        gmm_n_components: number of components for GMM
        gmm_reg_covar: regularization for GMM covariance
    """
    if len(beliefs_particles_list) == 0:
        return
    
    keep_pair = tuple(int(i) for i in keep_pair)
    
    # Determine grid bounds from all particles
    all_particles_2d = []
    for particles in beliefs_particles_list:
        particles_2d = particles[:, keep_pair]
        all_particles_2d.append(particles_2d)
    
    all_particles_2d = np.vstack(all_particles_2d)
    
    # Add padding
    x_margin = (all_particles_2d[:, 0].max() - all_particles_2d[:, 0].min()) * 0.1
    y_margin = (all_particles_2d[:, 1].max() - all_particles_2d[:, 1].min()) * 0.1
    
    x_min = all_particles_2d[:, 0].min() - x_margin
    x_max = all_particles_2d[:, 0].max() + x_margin
    y_min = all_particles_2d[:, 1].min() - y_margin
    y_max = all_particles_2d[:, 1].max() + y_margin
    
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    XX, YY = np.meshgrid(xs, ys)
    pts_2d = np.stack([XX.ravel(), YY.ravel()], axis=1)
    
    # Evaluate each belief separately with individual color scaling
    grids = []
    for particles in beliefs_particles_list:
        # Extract 2D coordinates for the marginal
        particles_2d = particles[:, keep_pair]
        
        # Create 2D GMM from the marginal particles
        gmm_2d = gmm_from_particles(particles_2d, n_components=gmm_n_components, reg_covar=gmm_reg_covar)
        
        # Evaluate on grid
        zz = gmm_2d.pdf(pts_2d)
        Z = zz.reshape(resolution, resolution)
        grids.append(Z)
    
    # Layout
    t = len(beliefs_particles_list)
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
        ax.set_xlabel(f"x[{keep_pair[0]}]")
        ax.set_ylabel(f"x[{keep_pair[1]}]")
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


def run_trials_nf(train_data, test_data, init_state_sampler, save_directory, num_trials,
                  n_particles=1000, n_epochs_tran=20, num_layers=8, hidden_features=128,
                  use_gpu=True, batch_size=512, n_added_samples=1, gmm_n_components=100, gmm_reg_covar=1e-4,
                  save_figures=False):
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
    save_figures : bool
        Whether to save figures (default: False).
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
    gmm_n_components : int
        Number of components for Gaussian Mixture Model (default: 3).
    gmm_reg_covar : float
        Regularization for GMM covariance matrices (default: 1e-4).
    
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
        
        # Remove any particles that are infinite or NaN
        valid_mask = np.isfinite(initial_particles).all(axis=1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            initial_particles = initial_particles[valid_mask]
            if len(initial_particles) == 0:
                print(f"Error: All initial particles are invalid. Skipping trial {trial + 1}.")
                # Set all NLL and prop_times to NaN for this trial
                for k in range(num_timesteps):
                    negative_log_likelihoods[trial, k] = np.nan
                    if k < num_timesteps - 1:
                        prop_times[trial, k] = np.nan
                continue
            elif len(initial_particles) < n_particles * 0.9:
                print(f"Warning: Removed {n_invalid}/{n_particles} invalid initial particles ({len(initial_particles)} remaining)")

        # Propagate beliefs forward using particle sets
        print("Propagating beliefs...")
        beliefs_particles = [initial_particles.copy()]
        
        # Evaluate initial belief log likelihood
        print("Evaluating log likelihoods...")
        initial_gmm = gmm_from_particles(initial_particles, n_components=gmm_n_components, reg_covar=gmm_reg_covar)
        nll_init = -avg_log_likelihood(test_data[0], initial_gmm.pdf)
        negative_log_likelihoods[trial, 0] = nll_init
        
        # Calculate mc_auc for initial GMM belief
        # Use adaptive region based on particle distribution (with padding)
        particle_mins = initial_particles.min(axis=0)
        particle_maxes = initial_particles.max(axis=0)
        particle_std = initial_particles.std(axis=0)
        # Add padding: 5 standard deviations on each side
        padding = 5.0 * particle_std
        adaptive_mins = particle_mins - padding
        adaptive_maxes = particle_maxes + padding
        
        auc_init = mc_auc(dim, initial_gmm.pdf, n_samples=10000, region=Rectangle(mins=adaptive_mins, maxes=adaptive_maxes))
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
                
                # Remove particles that are infinite or NaN before clipping
                valid_mask = np.isfinite(next_particles).all(axis=1)
                if not valid_mask.all():
                    n_invalid = (~valid_mask).sum()
                    n_before = len(next_particles)
                    next_particles = next_particles[valid_mask]
                    n_after = len(next_particles)
                    if n_after == 0:
                        print(f"Warning: All particles invalid after filtering at timestep {i+1}")
                        # Set the failed propagation time to NaN
                        prop_times[trial, i] = np.nan
                        # Set remaining NLL and prop_times to NaN
                        for remaining_k in range(i+1, num_timesteps):
                            negative_log_likelihoods[trial, remaining_k] = np.nan
                            if remaining_k < num_timesteps - 1:
                                prop_times[trial, remaining_k] = np.nan
                        break
                    elif n_after < n_before * 0.5:
                        print(f"Warning: Removed {n_invalid}/{n_before} invalid particles at timestep {i+1} ({n_after} remaining)")

                mean = next_particles.mean(axis=0)
                std = next_particles.std(axis=0)
                next_particles = np.clip(next_particles, -100, 100)
                
                prop_times[trial, i] = time.time() - start
                beliefs_particles.append(next_particles)
                
                # Convert particles to PDF using GMM and evaluate log likelihood
                belief_gmm = gmm_from_particles(next_particles, n_components=gmm_n_components, reg_covar=gmm_reg_covar)
                nll = -avg_log_likelihood(test_data[i + 1], belief_gmm.pdf)
                negative_log_likelihoods[trial, i + 1] = nll
                
                # Calculate mc_auc for GMM belief
                # Use adaptive region based on particle distribution (with padding)
                particle_mins = next_particles.min(axis=0)
                particle_maxes = next_particles.max(axis=0)
                particle_std = next_particles.std(axis=0)
                # Add padding: 5 standard deviations on each side
                padding = 5.0 * particle_std
                adaptive_mins = particle_mins - padding
                adaptive_maxes = particle_maxes + padding
                
                #auc = mc_auc(dim, belief_gmm.pdf, n_samples=10000, region=Rectangle(mins=adaptive_mins, maxes=adaptive_maxes))
                #print(f"  Belief {i + 1}: avg_log_likelihood = {-nll:.6f}, mc_auc = {auc:.6f}, prop_time = {prop_times[trial, i]:.4f}s")
                print(f"  Belief {i + 1}: avg_log_likelihood = {-nll:.6f}, prop_time = {prop_times[trial, i]:.4f}s")
                
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
        
        # Save figures only for the first trial
        if save_figures and trial == 0:
            os.makedirs(save_directory, exist_ok=True)
            print(f"\nSaving figures to {save_directory}...")
            
            # Plot 2D marginals (same as sos_experiment)
            # Using state indices: [0:px, 1:pz, 2:thetac, 3:thetat, 4:v, 5:omega]
            # beliefs_particles already contains all the particle sets we need
            plot_2d_marginals_over_time_nf(beliefs_particles, (0, 1), "px_pz",
                                          resolution=60,
                                          save_path=os.path.join(save_directory, "marginals_px_pz.png"),
                                          show_plot=False,
                                          gmm_n_components=gmm_n_components,
                                          gmm_reg_covar=gmm_reg_covar)
            plot_2d_marginals_over_time_nf(beliefs_particles, (2, 3), "thetac_thetat",
                                          resolution=60,
                                          save_path=os.path.join(save_directory, "marginals_thetac_thetat.png"),
                                          show_plot=False,
                                          gmm_n_components=gmm_n_components,
                                          gmm_reg_covar=gmm_reg_covar)
            plot_2d_marginals_over_time_nf(beliefs_particles, (4, 5), "v_omega",
                                          resolution=60,
                                          save_path=os.path.join(save_directory, "marginals_v_omega.png"),
                                          show_plot=False,
                                          gmm_n_components=gmm_n_components,
                                          gmm_reg_covar=gmm_reg_covar)
            print("Figures saved.\n")
        
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
        dt=0.2,
        L_t=1.0,
        v_ref=1.0,
        k_v=1.0,
        k_theta=2.0,
        sigma_v=0.1,
        sigma_omega=0.5,
        cov_scale=0.5
    )

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj_train = 4000
    n_traj_test = 10000

    # Number of training epochs
    n_epochs_tran = 50

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
        gmm_n_components=50,
        gmm_reg_covar=1e-4,
        save_figures=False
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