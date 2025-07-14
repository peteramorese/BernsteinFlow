import numpy as np
import torch
import copy

from .Polynomial import Polynomial, poly_sum, poly_product, split_factor_poly_product, stable_split_factors, marginal, monomial_to_bernstein, bernstein_to_monomial
from .GPGMM import GMModel, MultivariateGPModel, compute_mean_jacobian


def propagate_bfn(belief_p_factors : list[Polynomial], transition_p_factors : list[Polynomial], mag_range=None):
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
        p_joint = poly_product(p_list) # Compute p(x) * p(x' | x)
        p_next_marginal = marginal(p_joint, list(range(curr_belief_dim))) # Marginalize out the x dimensions
    else:
        curr_belief_dim = max([p.dim() for p in belief_p_factors])
        p_list = belief_p_factors + transition_p_factors 
        p_list_split = stable_split_factors(p_list, mag_range)
        p_joint_list = split_factor_poly_product(p_list_split) # Compute the split product
        p_next_marginal_list = [marginal(p_joint, list(range(curr_belief_dim))) for p_joint in p_joint_list] # Marginalize each list to be more stable
        p_next_marginal = poly_sum(p_next_marginal_list, stable=True) # Sum each marginal output
    return p_next_marginal

def propagate_gpgmm_ekf(belief : GMModel, transition_p : MultivariateGPModel):
    means = belief.means
    covs = belief.covariances

    next_belief = GMModel(means=[], covariances=[], weights=copy.deepcopy(belief.weights))

    # Propagate each component of the belief GMM
    for mean, cov in zip(means, covs):
        # Propagate mean
        next_mean, pred_stds = transition_p.predict(mean)
        pred_cov = np.diag(pred_stds**2)

        # Propagate covariance
        J = compute_mean_jacobian(transition_p, mean)

        next_cov = J * cov * J.T + pred_cov

        next_belief.means.append(next_mean)
        next_belief.covariances.append(next_cov)

    return next_belief

