from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.GPGMM import GMModel, GPModel, fit_gmm, fit_gp
from bernstein_flow.Tools import create_transition_data_matrix, grid_eval, model_u_eval_fcn, model_x_eval_fcn
from bernstein_flow.Polynomial import poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct
from bernstein_flow.Propagate import propagate_gpgmm_ekf, propagate_gpgmm_wsasos

from .Systems import VanDerPol, Pendulum, sample_trajectories
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

if __name__ == "__main__":

    # System model
    #system = Pendulum(dt=0.05, length=1.0, damp=1.1, covariance=0.005 * np.eye(2))
    system = VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * np.eye(2))

    # Dimension
    dim = system.dim()

    # Number of trajectories
    n_traj = 100

    # Number of training epochs
    n_epochs_tran = 4

    # Time horizon
    training_timesteps = 5
    timesteps = 5

    def init_state_sampler():
        return multivariate_normal.rvs(mean=np.array([0.2, 0.1]), cov = np.diag([0.2, 0.2]))

    traj_data = sample_trajectories(system, init_state_sampler, timesteps, n_traj)

    x_bounds = [-5.0, 5.0, -5.0, 5.0]
    state_distribution_plot_2D(traj_data, interactive=True, bounds=x_bounds)
    plt.show()

    # Create the data matrices for training
    X0_data = traj_data[0]
    Xp_data = create_transition_data_matrix(traj_data[:training_timesteps])
    Xp_data_torch = torch.from_numpy(Xp_data)

    print("Fitting initial state model...")
    init_state_model = fit_gmm(X0_data, n_components=10, covariance_type='full')
    print("Fitting transition state model...")
    transition_model = fit_gp(Xp=Xp_data_torch[:, dim:], X=Xp_data_torch[:, :dim], num_epochs=n_epochs_tran, dtype=torch.float64)
    print("Done!")


    density_gmms_ekf = [init_state_model]
    for k in range(1, timesteps):
        p_curr = propagate_gpgmm_ekf(density_gmms_ekf[k-1], transition_model)
        print(f"Computed p(x{k}) (EKF). Number of components: ", p_curr.n_mixands())
        density_gmms_ekf.append(p_curr)

    #density_gmms_wsasos = [init_state_model]
    #for k in range(1, timesteps):
    #    p_curr = propagate_gpgmm_wsasos(density_gmms_wsasos[k-1], transition_model)
    #    print(f"Computed p(x{k}) (WSASOS). Number of components: ", p_curr.n_mixands())
    #    density_gmms_wsasos.append(p_curr)

    # Make pdf plotter for interactive vis
    def pdf_plotter(k : int):
        return grid_eval(lambda x : density_gmms_ekf[k].density(x), x_bounds, dtype=DTYPE)
    state_dist_fig_ekf, _ = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds)

    ## Make pdf plotter for interactive vis
    #def pdf_plotter(k : int):
    #    return grid_eval(lambda x : density_gmms_wsasos[k].density(x), x_bounds, dtype=DTYPE)
    #state_dist_fig_wsasos, _ = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds)

    particle_figs, ekf_figs = state_distribution_plot_2D(traj_data, pdf_plotter, interactive=False, bounds=x_bounds, separate_figures=True, exclude_ticks=False)


    def save_figure_bundle(fig_bundle, dir):
        print("cwd: ", os.getcwd())
        os.makedirs(dir, exist_ok=True)
        for k, fig in enumerate(fig_bundle):
            print("saving to: ", dir + f"/k_{k}.png")
            fig.savefig(dir + f"/k_{k}.png")



    state_dist_fig_ekf.savefig("./figures/trajectory_2D_gpgmm_ekf.png")
    #state_dist_fig_wsasos.savefig("./figures/trajectory_2D_gpgmm_wsasos.png")

    save_figure_bundle(particle_figs, "./figures/trajectory_2D_gpgmm_ekf_separate_particle")
    #save_figure_bundle(wsasos_figs, "./figures/trajectory_2D_gpgmm_wsasos_separate_pdf")

