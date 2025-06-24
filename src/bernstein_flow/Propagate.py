import torch

from .Polynomial import Polynomial, poly_product, bernstein_marginal


def propagate(belief_p_factors : list[Polynomial], transition_p_factors : list[Polynomial]):
    """
    Given a belief marginal distribution and a stochastic transition distribution, compute the next marginal distribution

    Args:
        belief_p_factors : polynomial pdf (bernstein basis) of p(x). Can be in factor form (with multiple factors in the list) or just a single polynomial (list with single element)
        transition_p_factors : conditional polynomial pdf (bernstein basis) of p(x' | x)
    """
    curr_belief_dim = max([p.ndim for p in belief_p_factors])
    p_list = belief_p_factors + transition_p_factors # Ordered since all belief factors will be smaller dimension than transition factors
    p_joint = poly_product(p_list) # Compute p(x) * p(x' | x)
    p_next_marginal = bernstein_marginal(p_joint, list(range(curr_belief_dim))) # Marginalize out the x dimensions

    return p_next_marginal


