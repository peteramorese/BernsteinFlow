import numpy as np
import torch
import copy
from scipy.spatial import Rectangle
from itertools import product

from .Polynomial import Polynomial, poly_sum, poly_product, split_factor_poly_product, stable_split_factors, marginal, monomial_to_bernstein, bernstein_to_monomial, poly_product_bernstein_direct
from .GPGMM import GMModel, MultivariateGPModel, compute_mean_jacobian, compute_mean_hessian_tensor
from .WSASOS import WSASOS


def propagate_bfm(belief_p_factors : list[Polynomial], transition_p_factors : list[Polynomial], mag_range=None):
    """
    Given a belief marginal distribution (Bernstein Flow Model) and a stochastic transition distribution, compute the next marginal distribution

    Args:
        belief_p_factors : polynomial pdf (bernstein basis) of p(x). Can be in factor form (with multiple factors in the list) or just a single polynomial (list with single element)
        transition_p_factors : conditional polynomial pdf (bernstein basis) of p(x' | x)
        mag_range : magnitude range for stable splitting. If provided, all operations will be done with stability.
    """
    if mag_range is None:
        curr_belief_dim = max([p.dim() for p in belief_p_factors])
        p_list = belief_p_factors + transition_p_factors # Ordered since all belief factors will be smaller dimension than transition factors
        #p_joint = poly_product(p_list) # Compute p(x) * p(x' | x)
        p_joint = poly_product_bernstein_direct(p_list) # Compute p(x) * p(x' | x)
        #p_next_marginal = marginal(p_joint, list(range(curr_belief_dim))) # Marginalize out the x dimensions
        p_next_marginal = marginal(p_joint, list(range(curr_belief_dim)), stable=True) # Marginalize out the x dimensions
    else:
        curr_belief_dim = max([p.dim() for p in belief_p_factors])
        p_list = belief_p_factors + transition_p_factors 
        p_list_split = stable_split_factors(p_list, mag_range)
        p_joint_list = split_factor_poly_product(p_list_split) # Compute the split product
        p_next_marginal_list = [marginal(p_joint, list(range(curr_belief_dim)), stable=True) for p_joint in p_joint_list] # Marginalize each list to be more stable
        p_next_marginal = poly_sum(p_next_marginal_list, stable=True) # Sum each marginal output
    return p_next_marginal

def propagate_gpgmm_ekf(belief : GMModel, transition_p : MultivariateGPModel):
    """
    Propagate the GP-GMM model forward using an EKF-style linearization for lin-gaussian prop of each GMM component
    """
    means = belief.means
    covs = belief.covariances

    next_belief = GMModel(means=[], covariances=[], weights=copy.deepcopy(belief.weights))

    # Propagate each component of the belief GMM
    for mean, cov in zip(means, covs):
        # Propagate mean
        mean = mean.reshape(1, -1)
        next_mean, pred_stds = transition_p.predict(mean)
        pred_cov = np.diag(pred_stds**2)

        # Propagate covariance
        J = compute_mean_jacobian(transition_p, mean)

        next_cov = J @ cov @ J.T + pred_cov

        next_belief.means.append(next_mean)
        next_belief.covariances.append(next_cov)

    return next_belief

def propagate_gpgmm_wsasos(belief : GMModel, transition_p : MultivariateGPModel, n_samples = 500):
    wsasos = WSASOS(n_samples=n_samples)
    return wsasos.wsasos_split_and_propagate(belief, transition_p)

def propagate_grid_gmm(belief : GMModel, transition_p : MultivariateGPModel, bounds : list, resolution = 10):
    domain = Rectangle(mins=np.array(bounds[::2]), maxes=np.array(bounds[1::2]))
    diag_vector = (np.array(bounds[1::2]) - np.array(bounds[::2])) / resolution

    linspaces = [np.linspace(domain_min, domain_max, resolution, endpoint=False) for domain_min, domain_max in zip(domain.mins, domain.maxes)]
    #mesh = np.meshgrid(*linspaces, indexing='ij')

    next_belief = GMModel(means=[], covariances=[], weights=[])

    for min_coord in product(*linspaces):
        min_coord = np.asarray(min_coord)
        max_coord = min_coord + diag_vector
        center_point = min_coord + 0.5 * diag_vector

        region = Rectangle(mins=min_coord, maxes=max_coord)

        next_weight = belief.integrate(region)
        #print("centerjpoint :" ,center_point.reshape(1, -1))
        next_mean, pred_stds = transition_p.predict(center_point.reshape(1, -1))
        pred_cov = pred_stds**2

        next_belief.means.append(next_mean)
        next_belief.covariances.append(pred_cov)
        next_belief.weights.append(next_weight)
    
    return next_belief

    #cell_counts = [resolution - 1 for _ in range(transition_p.dim)]

    #coords = [np.linspace(domain_min, domain_max, resolution) for domain_min, domain_max in zip(domain.mins, domain.maxes)]
    #for idx in product(*[range(n) for n in cell_counts]):
    #    region = Rectangle()
