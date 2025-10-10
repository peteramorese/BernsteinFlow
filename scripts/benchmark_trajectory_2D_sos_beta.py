from bernstein_flow.DistributionTransform import GaussianDistTransform
from sos_form.BetaModel import BetaSOSModel
from sos_form.SOSModel import optimize

from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, mc_auc, avg_log_likelihood, empirical_prob_in_region

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

    # System model
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))
    #system = BistableOscillator(dt=0.1, a=1.0, d=1.0, cov_scale=0.03)

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 1000
    n_test_traj = 10000

    # Number of training epochs
    n_epochs_init = 300#3000
    n_epochs_tran = 200#150

    # Time horizon
    training_timesteps = 10
    timesteps = 12

    # Region of integration
    roi = Rectangle(mins=[-1.0, -1.0], maxes=[1.0, 1.0])

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)
    test_traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_test_traj)

    # Moment match the GDT to all of the data over the whole horizon
    #gdt = GaussianDistTransform(means=np.array([0.0, 0.0]), variances=[0.3, 1.0])
    gdt = GaussianDistTransform.moment_match_data(np.vstack(traj_data), variance_pads=[2.2, 2.2])

    u_traj_data = [gdt.X_to_U(X_data) for X_data in traj_data]

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])

    # Convert the data to the U space for training
    U0_data = gdt.X_to_U(X0_data) # Initial state data
    Up_data = np.hstack([gdt.X_to_U(Xp_data[:, dim:]), gdt.X_to_U(Xp_data[:, :dim])])  # Transition kernel data (y, x order)

    use_gpu = True
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    cpu_device = torch.device("cpu")

    # Create data loader
    U0_data_torch = torch.tensor(U0_data, dtype=DTYPE)
    U0_dataset = TensorDataset(U0_data_torch)
    U0_dataloader = DataLoader(U0_dataset, batch_size=128, shuffle=True, pin_memory=use_gpu)

    Up_data_torch = torch.tensor(Up_data, dtype=DTYPE)
    Up_dataset = TensorDataset(Up_data_torch)
    Up_dataloader = DataLoader(Up_dataset, batch_size=256, shuffle=True, pin_memory=use_gpu)

    # Create initial state and transition models using SOS Beta model
    n_basis = 25  # Number of basis functions
    
    # Create transition model first (needed as reference for init state model)
    transition_model = BetaSOSModel(dy=dim, 
                                   dx=dim, 
                                   n=n_basis, 
                                   min_alpha_beta=0.1, 
                                   max_alpha_beta=100.0, 
                                   mu=0.1, 
                                   min_Q_eigval=1e-8, 
                                   regularization_weight=5e-5)

    print(f"Created transition model with {transition_model.n_parameters()} parameters")

    print("Training transition model...")
    start = time.time()
    transition_model.to(device=device, dtype=DTYPE)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-2)
    _, best_trans_loss_1 = optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran)
    trans_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
    _, best_trans_loss_2 = optimize(transition_model, Up_dataloader, trans_optimizer, epochs=n_epochs_tran//2)
    tran_train_time = time.time() - start
    print("Done training transition model \n")
    transition_model = transition_model.to(device=cpu_device)

    # Now create init state model with reference to transition model
    init_state_model = BetaSOSModel(dy=dim, 
                                   dx=0, 
                                   n=n_basis, 
                                   conditional=False, 
                                   reference_factor_model=transition_model,
                                   min_alpha_beta=0.1, 
                                   max_alpha_beta=100.0, 
                                   mu=0.1, 
                                   min_Q_eigval=1e-8, 
                                   regularization_weight=1e-4)

    print(f"Created init state model with {init_state_model.n_parameters()} parameters")

    # Train the Init model
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-2)
    print("Training initial state model...")
    start = time.time()
    init_state_model.to(device=device, dtype=DTYPE)
    _, best_init_loss_1 = optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init)
    init_optimizer = torch.optim.Adam(init_state_model.parameters(), lr=1e-4)
    _, best_init_loss_2 = optimize(init_state_model, U0_dataloader, init_optimizer, epochs=n_epochs_init//2)
    init_train_time = time.time() - start
    print("Done training initial state model \n")
    init_state_model = init_state_model.to(device=cpu_device)

    # Propagate beliefs using SOS model propagation
    beliefs = [init_state_model]
    prop_times = []
    mc_aucs = []
    allhs = []
    
    for k in range(1, timesteps):
        start = time.time()
        # Propagate belief using SOS model
        new_belief = transition_model.propagate(beliefs[k-1])
        prop_times.append(time.time() - start)
        
        # Calculate MC AUC for the belief
        with torch.no_grad():
            mc_auc_val = mc_auc(dim, lambda u: new_belief(torch.from_numpy(u).to(dtype=DTYPE)).detach().cpu().numpy(), n_samples=10000)
        mc_aucs.append(mc_auc_val)
        print(f"Computed belief at timestep {k} in {prop_times[-1]:.2f} seconds")

        # Compute the log likelihood using the x density
        def x_density_func(x):
            u = gdt.X_to_U(x)
            return gdt.x_density(x, lambda u: new_belief(torch.from_numpy(u).to(dtype=DTYPE)).detach().cpu().numpy())
        
        x_allh = avg_log_likelihood(test_traj_data[k], x_density_func)
        print(f" - Average log likelihood: {x_allh:.3f}")
        allhs.append(x_allh)

        beliefs.append(new_belief)

    x_bounds = [-5.0, 5.0, -5.0, 5.0]
    def pdf_plotter(k : int):
        def x_density_func(x):
            u = gdt.X_to_U(x)
            return gdt.x_density(x, lambda u: beliefs[k](torch.from_numpy(u).to(dtype=DTYPE)).detach().cpu().numpy())
        return grid_eval(x_density_func, x_bounds, dtype=DTYPE)

    state_dist_fig, _ = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds)
    particle_figs, pdf_figs = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds, separate_figures=True, exclude_ticks=False)

    

    # Write down system properties
    curr_date_time = get_date_time_str()
    benchmark_fields["datetime"] = curr_date_time 
    benchmark_fields["system"] = system.__class__.__name__
    benchmark_fields["dimension"] = dim
    benchmark_fields["n_traj"] = n_traj
    benchmark_fields["n_epochs_init"] = n_epochs_init
    benchmark_fields["n_epochs_tran"] = n_epochs_tran
    benchmark_fields["training_timesteps"] = training_timesteps
    benchmark_fields["timesteps"] = timesteps
    benchmark_fields["device"] = device.type
    benchmark_fields["n_basis"] = n_basis
    benchmark_fields["min_alpha_beta"] = 0.1
    benchmark_fields["max_alpha_beta"] = 80.0
    benchmark_fields["mu"] = 0.1
    benchmark_fields["min_Q_eigval"] = 1e-8
    benchmark_fields["regularization_weight"] = 1e-4
    benchmark_fields["init_model_params"] = init_state_model.n_parameters()
    benchmark_fields["tran_model_params"] = transition_model.n_parameters()
    benchmark_fields["init_train_time"] = init_train_time
    benchmark_fields["tran_train_time"] = tran_train_time
    benchmark_fields["prop_times"] = prop_times
    benchmark_fields["mc_auc"] = mc_aucs
    benchmark_fields["average_log_likelihood"] = allhs


    experiment_name = f"trajectory_2D_sos_beta_{curr_date_time}"

    save_figure_bundle(particle_figs, f"./benchmarks/{experiment_name}/particle")
    save_figure_bundle(pdf_figs, f"./benchmarks/{experiment_name}/pdf")
    state_dist_fig.savefig(f"./benchmarks/{experiment_name}/combined.pdf")

    with open(f"./benchmarks/{experiment_name}/data.json", "w") as f:
        json.dump(benchmark_fields, f, indent=4)
