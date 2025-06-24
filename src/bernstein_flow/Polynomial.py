from itertools import product
import math
import torch
import torch.fft
from functools import reduce
import copy
import numpy as np

from scipy.fft import next_fast_len

Polynomial = torch.Tensor

def eval(p : Polynomial, x : torch.Tensor):
    """
    Evaluate a multivariate polynomial at a point x ∈ R^d.
    
    Args:
        coeffs: A d-dimensional torch.Tensor of shape (n₁, n₂, ..., n_d), 
                where coeffs[i₁, i₂, ..., i_d] is the coefficient of 
                x₁^i₁ * x₂^i₂ * ... * x_d^i_d.
        x: A torch.Tensor of shape (m, d) representing m points in R^d.
        
    Returns:
        A scalar torch.Tensor containing the polynomial value at x.
    """
    assert x.ndim == 2, "x must be a 2D tensor of shape (m, d)"
    d = p.ndim
    m = x.shape[0]
    assert x.shape[1] == d, f"Each point must have dimension {d}"

    # Generate all exponent combinations (m, d)
    exponents = torch.cartesian_prod(*[torch.arange(n, device=p.device) for n in p.shape])  # (m, d)

    # Evaluate monomials at all x: x^exponents using broadcasting
    # x shape: (p, d), exponents shape: (m, d)
    x_powers = x.unsqueeze(1) ** exponents.unsqueeze(0)  # shape: (p, m, d)
    monomials = torch.prod(x_powers, dim=2)  # shape: (p, m)

    # Dot each row of monomials with the flattened coeffs
    coeffs_flat = p.flatten()  # shape: (m,)
    result = monomials @ coeffs_flat  # shape: (p,)

    return result

def binomial(n, k):
    return math.comb(n, k)

def binomial_tensor(n, k, dtype=torch.double, device='cpu'):
    n = torch.as_tensor(n, dtype=dtype, device=device)
    k = torch.as_tensor(k, dtype=dtype, device=device)
    
    # Create a mask for valid k values (0 <= k <= n)
    valid_mask = (k >= 0) & (k <= n)
    
    # lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    log_comb = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    
    # Use the mask to handle invalid k values, which should product in 0
    # exp(log_comb) for valid cases, 0 otherwise
    # We round to handle potential floating point inaccuracies since binomial_tensor should be integers
    return torch.where(valid_mask, torch.exp(log_comb).round(), torch.zeros_like(log_comb))

def bernstein_to_monomial(bernstein_coeffs: torch.Tensor) -> torch.Tensor:
    if not isinstance(bernstein_coeffs, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    d = bernstein_coeffs.ndim
    monomial_coeffs = bernstein_coeffs.clone()
    dtype = bernstein_coeffs.dtype
    device = bernstein_coeffs.device
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for i in range(d):
        n = monomial_coeffs.shape[i] - 1
        if n < 0:
            continue

        # Build the Bernstein-to-Monomial conversion matrix M
        # The (k, j) entry is C(n, k) * C(k, j) * (-1)^(k-j)
        k_ = torch.arange(n + 1, device=device, dtype=dtype).view(-1, 1) # Row index
        j_ = torch.arange(n + 1, device=device, dtype=dtype).view(1, -1) # Col index
        
        mask = (j_ <= k_).float()
        
        comb_nk = binomial_tensor(n, k_, dtype=dtype, device=device)
        comb_kj = binomial_tensor(k_, j_, dtype=dtype, device=device)
        
        signs = torch.pow(-1.0, k_ - j_)
        
        M = comb_nk * comb_kj * signs * mask
        M = M.to(dtype)

        # Apply the transformation using einsum for clarity and efficiency
        # e.g., for dim 0 of a 3D tensor: 'abc,ka->kbc'
        subscripts = list(alphabet[:d])
        contract_idx = subscripts[i]
        new_idx = alphabet[d] 
        
        mat_subscripts = f"{new_idx}{contract_idx}"
        
        product_subscripts = list(subscripts)
        product_subscripts[i] = new_idx
        
        einsum_str = f"{''.join(subscripts)},{mat_subscripts}->{''.join(product_subscripts)}"
        
        monomial_coeffs = torch.einsum(einsum_str, monomial_coeffs, M)
        
    return monomial_coeffs

def monomial_to_bernstein(monomial_coeffs: torch.Tensor) -> torch.Tensor:
    if not isinstance(monomial_coeffs, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    d = monomial_coeffs.ndim
    bernstein_coeffs = monomial_coeffs.clone()
    dtype = monomial_coeffs.dtype
    device = monomial_coeffs.device

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(d):
        n = bernstein_coeffs.shape[i] - 1
        if n < 0:
            continue

        # Build the Monomial-to-Bernstein conversion matrix N
        # The (j, k) entry is C(j, k) / C(n, k)
        j_ = torch.arange(n + 1, device=device, dtype=dtype).view(-1, 1) # Row index
        k_ = torch.arange(n + 1, device=device, dtype=dtype).view(1, -1) # Col index

        mask = (k_ <= j_).float()
        
        comb_jk = binomial_tensor(j_, k_, dtype=dtype, device=device)
        comb_nk = binomial_tensor(n, k_, dtype=dtype, device=device)
        
        # Add a small epsilon to denominator to prevent division by zero
        # This is a safeguard; C(n, k) should be non-zero for k <= n.
        N = (comb_jk / (comb_nk + 1e-12)) * mask
        N = N.to(dtype)

        # Apply the transformation using einsum
        subscripts = list(alphabet[:d])
        contract_idx = subscripts[i]
        new_idx = alphabet[d]
        
        mat_subscripts = f"{new_idx}{contract_idx}"
        
        product_subscripts = list(subscripts)
        product_subscripts[i] = new_idx
        
        einsum_str = f"{''.join(subscripts)},{mat_subscripts}->{''.join(product_subscripts)}"
        
        bernstein_coeffs = torch.einsum(einsum_str, bernstein_coeffs, N)

    return bernstein_coeffs

def create_d_separable_tensor(index_fcn, shape):
    # Compute g values for each dimension
    g_vectors = [torch.tensor([index_fcn(d, i) for i in range(shape[d])], dtype=torch.float32) for d in range(len(shape))]

    # Broadcasting multiplication (outer product)
    product = torch.ones(shape)
    for dim, g_vec in enumerate(g_vectors):
        broadcast_shape = [1] * len(shape)
        broadcast_shape[dim] = shape[dim]
        product *= g_vec.reshape(broadcast_shape)
    
    return product

def fft_convolve_nd(a, b):
    """
    Computes the convolution of two tensors using FFT, matching the broadcasted shape.
    """
    size = [a.shape[i] + b.shape[i] - 1 for i in range(a.ndim)]
    fft_size = [next_fast_len(s) for s in size]
    print("size: ", size)
    print("fft_size: ", fft_size)

    A = torch.fft.rfftn(a, s=fft_size)
    B = torch.fft.rfftn(b, s=fft_size)
    C = A * B
    c = torch.fft.irfftn(C, s=fft_size)

    # Truncate to actual convolution size
    slices = tuple(slice(0, s) for s in size)
    return c[slices]

def poly_product(p_list : list[Polynomial], bernstein_basis = False):
    """
    Computes the product of a list of multivariate polynomials using FFT-based convolution.
    
    Each polynomial in p_list is a torch.Tensor of coefficients of increasing dimension:
    e.g. p_list = [p(x1), p(x1,x2), ..., p(x1,...,xd)]. The product is most efficient
    when this order is given, but it is not necessary
    
    Returns:
        product: torch.Tensor of coefficients of the product polynomial.
    """
    assert len(p_list) >= 1, "p_list must contain at least one polynomial"
    
    p_list_lcl = copy.deepcopy(p_list)

    if bernstein_basis:
        # Pre-weight each polynomial before convolution
        for l in range(len(p_list_lcl)):
            p = p_list_lcl[l]
            pre_weight = create_d_separable_tensor(lambda dim, i : binomial(p.shape[dim] - 1, i), p.shape)
            p_list_lcl[l] = p * pre_weight
    
    product = p_list_lcl[0]
    for i in range(1, len(p_list_lcl)):
        p = p_list_lcl[i]

        # Pad product to match p's dimensionality
        if product.ndim < p.ndim:
            pad_dims = p.ndim - product.ndim
            product = product[(...,) + (None,) * pad_dims]

        # Broadcast shapes to match for convolution
        product_shape = product.shape
        p_shape = p.shape
        max_shape = [max(r, s) for r, s in zip(product_shape, p_shape)]

        # Pad each tensor to the appropriate shape
        product_pad = list(product_shape)
        p_pad = list(p_shape)
        while len(product_pad) < len(max_shape):
            product_pad.append(1)
        while len(p_pad) < len(max_shape):
            p_pad.append(1)

        # Compute FFT-based convolution
        product = fft_convolve_nd(product, p)
    
    if bernstein_basis:
        # Post-weight the convoluted polynomial
        post_weight = create_d_separable_tensor(lambda dim, s : 1.0 / binomial(product.shape[dim] - 1, s), product.shape)
        product *= post_weight

    return product

def bernstein_marginal(p : Polynomial, dims : list[int]):
    """
    Integrate a bernstein polynomial over the specified dims from x_l = 0 to x_l = 1

    Args:
        p : coefficient tensor in the bernstein basis
        dims : list of integers describing which dimensions to integrate over
    """
    weight = 1.0
    for d in dims:
        weight /= p.shape[d]
    
    # Sum along the desired dimensions (integrate) then weight by the area of the basis polynomials
    return weight * torch.sum(p, dim=dims)

def monomial_marginal(p : Polynomial, dims : list[int]):
    """
    Integrate a polynomial over the specified dims from x_l = 0 to x_l = 1

    Args:
        p : coefficient tensor in the monomial basis
        dims : list of integers describing which dimensions to integrate over
    """
    result = p.clone()
    
    # Sort dimensions in descending order to avoid index shifting issues
    dim_list_sorted = sorted(dims, reverse=True)
    
    for dim in dim_list_sorted:
        assert dim < p.ndim

        # Get the size of this dimension (max degree + 1)
        degree_plus_one = result.shape[dim]
        
        # Create integration weights: 1/(k+1) for degree k
        # Shape: [1, 1, ..., degree_plus_one, 1, 1, ...]
        weights_shape = [1] * result.ndim
        weights_shape[dim] = degree_plus_one
        
        weights = torch.zeros(weights_shape, dtype=result.dtype, device=result.device)
        for k in range(degree_plus_one):
            # For term x^k, integral from 0 to 1 is 1/(k+1)
            weights_index = [slice(None)] * result.ndim
            weights_index[dim] = k
            weights[tuple(weights_index)] = 1.0 / (k + 1)
        
        # Apply integration weights
        result = result * weights
        
        # Sum along this dimension (integrate)
        result = torch.sum(result, dim=dim, keepdim=False)
    
    return result


if __name__ == "__main__":

    ## Bernstein
    #p_list = [
    #    torch.Tensor([1, 2, 3]),
    #    torch.Tensor([2, 3, 8, 6]),
    #    #torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 5, 3]]),
    #]

    #prod = triangular_poly_product(p_list, bernstein_basis=True)
    #print("Bernstein prod: ", prod, "\n")

    ## Power
    #p_list = [bernstein_to_monomial(p) for p in p_list]

    #print("pwr basis polys: ", p_list)
    #prod = triangular_poly_product(p_list)
    #print("pwr prod: ", prod)
    #prod_bern = monomial_to_bernstein(prod)
    #print("Power prod in bern: ", prod_bern)


    dims = [1, 3]
    p_bern = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    p_bern = torch.rand(3, 3, 3, 3)
    p_bern_marg = bernstein_marginal(p_bern, dims)
    print("Bern marg: ", p_bern_marg)

    p_pwr = bernstein_to_monomial(p_bern)
    p_pwr_marg = monomial_marginal(p_pwr, dims)
    p_pwr_marg_bern = monomial_to_bernstein(p_pwr_marg)
    print("Pwr marg in bern: ", p_pwr_marg_bern)



    #p_pwr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    #p_bern = monomial_to_bernstein(p_pwr)
    #print("Power: ", p_pwr)
    #print("Bernstein: ", p_bern)
    #print("Power: ", bernstein_to_monomial(p_bern))