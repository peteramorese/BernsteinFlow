import itertools
import torch
from sympy import symbols, binomial, lambdify

def bernstein_basis_functions(dim : int, degree : int):
    """
    Returns a list of lambdified multivariate Bernstein basis functions and their multi-indices.
    
    Args:
        dim (int): number of variables (dimension)
        degree (int): Bernstein polynomial degree in each variable
    
    Returns:
        basis_funcs (List[Callable]): List of lambdified basis functions compatible with torch
        multi_indices (List[Tuple[int]]): Corresponding multi-indices for each basis function
        x_syms (Tuple[sympy.Symbol]): Tuple of sympy symbols (x0, x1, ..., xd-1)
    """
    x_syms = symbols(f'x0:{dim}')
    multi_indices = list(itertools.product(range(degree + 1), repeat=dim))
    
    basis_funcs = []
    for alpha in multi_indices:
        B = 1
        for i in range(dim):
            B *= binomial(degree, alpha[i]) * x_syms[i]**alpha[i] * (1 - x_syms[i])**(degree - alpha[i])
        basis_funcs.append(lambdify(x_syms, B, modules='torch'))
    
    return basis_funcs, multi_indices, x_syms

