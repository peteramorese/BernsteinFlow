import torch

from .Polynomial import Polynomial, poly_sum, poly_product, split_factor_poly_product, stable_split_factors, marginal, monomial_to_bernstein, bernstein_to_monomial


def propagate(belief_p_factors : list[Polynomial], transition_p_factors : list[Polynomial], mag_range=None):
    """
    Given a belief marginal distribution and a stochastic transition distribution, compute the next marginal distribution

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


