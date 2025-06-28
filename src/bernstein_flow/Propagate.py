import torch

from .Polynomial import Polynomial, poly_product, marginal, monomial_to_bernstein, bernstein_to_monomial


def propagate(belief_p_factors : list[Polynomial], transition_p_factors : list[Polynomial]):
    """
    Given a belief marginal distribution and a stochastic transition distribution, compute the next marginal distribution

    Args:
        belief_p_factors : polynomial pdf (bernstein basis) of p(x). Can be in factor form (with multiple factors in the list) or just a single polynomial (list with single element)
        transition_p_factors : conditional polynomial pdf (bernstein basis) of p(x' | x)
    """
    #print("belief: \n", bernstein_to_monomial(belief_p_factors), "\ntransition: \n", bernstein_to_monomial(transition_p_factors))
    curr_belief_dim = max([p.dim() for p in belief_p_factors])
    p_list = belief_p_factors + transition_p_factors # Ordered since all belief factors will be smaller dimension than transition factors
    p_joint = poly_product(p_list) # Compute p(x) * p(x' | x)
    p_next_marginal = marginal(p_joint, list(range(curr_belief_dim))) # Marginalize out the x dimensions
    return p_next_marginal


