from bernstein_flow.DistributionTransform import GaussianDistTransform
#from bernstein_flow.Model import BernsteinFlowModel, ConditionalBernsteinFlowModel, optimize
from sos_form.SOSModel import optimize
from sos_form.BetaModel import BetaSOSModel
from sos_form.SumBetaModel import SumBetaSOSModel
from sos_form.PowerFunctionModel import PowerFunctionSOSModel
from sos_form.SignomialModel import SignomialSOSModel

from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn, mc_auc

from .Systems import PlanarQuadrotor, sample_trajectories, sample_io_pairs
from .Visualization import interactive_transformer_plot, state_distribution_plot_2D, plot_density_2D, plot_density_2D_surface, plot_data_2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import os


DTYPE = torch.float64

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

if __name__ == "__main__":

    # System model
    system = PlanarQuadrotor(dt=0.01, covariance=0.05 * np.eye(6), waypoint=np.array([5.0, 0.0]))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 4000

    # Number of training epochs
    n_epochs_init = 100
    n_epochs_tran = 1000

    # Time horizon
    training_timesteps = 10
    timesteps = training_timesteps

    def init_state_sampler():
        # 6D state: [px, pz, theta, vx, vz, omega] near hover at origin
        mean = np.array([0.0, 0.0, 0.1, 50.0, 0.0, 0.0])
        cov = np.diag([0.1, 0.1, 0.05, 0.1, 0.1, 0.05])
        return multivariate_normal.rvs(mean=mean, cov=cov)

    #io_data = sample_io_pairs(system, n_pairs=n_traj * training_timesteps, region_lowers=[-5.0, -5.0], region_uppers=[5.0, 5.0])
    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    # Moment match the GDT to all of the data over the whole horizon
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[5.2, 5.2, 3.1, 5.2, 5.2, 3.1])
    #gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[0.2, 0.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]

    os.makedirs("figures/sos_6D", exist_ok=True)
    plot_2d_particle_scatter_over_time(u_traj_data, (0, 1), "px_pz_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_6D/mc_particles_px_pz.png",
                                       show_plot=True)
    plot_2d_particle_scatter_over_time(u_traj_data, (3, 4), "vx_vz_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_6D/mc_particles_vx_vz.png",
                                       show_plot=True)
    plot_2d_particle_scatter_over_time(u_traj_data, (2, 5), "theta_omega_particles",
                                       sample_limit=10000,
                                       save_path="figures/sos_6D/mc_particles_theta_omega.png",
                                       show_plot=True)

    #input("Continue to training...")

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, dim:]), gdt.X_to_U(Xp_data[:, :dim])])  # Transition kernel data 

    #use_gpu = torch.cuda.is_available()
    use_gpu = True
    print("Using GPU: ", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
    #device = torch.device("cpu")
    print("device: ", device)

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=256, shuffle=True, pin_memory=use_gpu)
    U0_dataloader_refine = DataLoader(U0_dataset, batch_size=2048, shuffle=True, pin_memory=use_gpu)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=256, shuffle=True, pin_memory=use_gpu)
    Up_dataloader_refine = DataLoader(Up_dataset, batch_size=2048, shuffle=True, pin_memory=use_gpu)

    ## Create initial state and transition models


    n = 15
    #n_terms = 10
    transition_model = BetaSOSModel(dy=dim, dx=dim, n=n, min_alpha_beta=0.1, max_alpha_beta=80.0, mu=0.1, min_Q_eigval=1e-8, regularization_weight=1e-4)
    #transition_model = SumBetaSOSModel(dy=dim, dx=dim, n=n, n_terms=n_terms, min_alpha_beta=0.4, max_alpha_beta=100.0, mu=0.1, min_Q_eigval=1e-8, regularization_weight=4e-4)

    print("Training transition model...")
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    optimize(transition_model, Up_dataloader, trans_optimizer, epochs=100)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
    optimize(transition_model, Up_dataloader_refine, trans_optimizer, epochs=50)

    transition_model.to(device=torch.device("cpu"))
    print("Done training transition model \n")

    init_state_model = BetaSOSModel(dy=dim, dx=0, n=n, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.4, max_alpha_beta=100.0, mu=0.1, min_Q_eigval=1e-8, regularization_weight=1e-4)
    #init_state_model = SumBetaSOSModel(dy=dim, dx=0, n=n, n_terms=n_terms, conditional=False, reference_factor_model=transition_model, min_alpha_beta=0.4, max_alpha_beta=100.0, mu=0.1, min_Q_eigval=1e-8, regularization_weight=4e-4)

    print("Training init state model...")
    init_state_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    optimize(init_state_model, U0_dataloader, trans_optimizer, epochs=100)
    trans_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-4)
    optimize(init_state_model, U0_dataloader_refine, trans_optimizer, epochs=50)

    init_state_model.to(device=torch.device("cpu"))
    print("Done training init state model \n")

    beliefs = [init_state_model]
    for i in range(timesteps):
        beliefs.append(transition_model.propagate(beliefs[i]))
        #beliefs.append(transition_model.propagate(beliefs[i], n_terms=n_terms)) #, n_terms=n_terms)

    print("\n")
    for i, belief in enumerate(beliefs):
        with torch.no_grad():
            auc = mc_auc(6, lambda u : belief(torch.from_numpy(u)).numpy(), n_samples=10000)
            print(f"Belief {i} auc: ", auc)


    os.makedirs("figures/sos_6D", exist_ok=True)
    # Using state indices: [0:px, 1:pz, 2:theta, 3:vx, 4:vz, 5:omega]
    plot_2d_marginals_over_time(beliefs, (0, 1), "px_pz",
                                resolution=60,
                                save_path="figures/sos_6D/marginals_px_pz.png",
                                show_plot=True)
    plot_2d_marginals_over_time(beliefs, (3, 4), "vx_vz",
                                resolution=60,
                                save_path="figures/sos_6D/marginals_vx_vz.png",
                                show_plot=True)
    plot_2d_marginals_over_time(beliefs, (2, 5), "theta_omega",
                                resolution=60,
                                save_path="figures/sos_6D/marginals_theta_omega.png",
                                show_plot=True)
