from itertools import product
import torch
import torch.fft

from functools import reduce
import copy
import numpy as np
from enum import Enum

import scipy.fft as scifft
from scipy.special import comb

class Basis(Enum):
    MONO = 1 # monomial (power basis)
    BERN = 2 # bernstein

Polynomial = torch.Tensor
class Polynomial:
    def __init__(self, coeffs, basis = Basis.MONO):
        if not (isinstance(coeffs, torch.Tensor) or isinstance(coeffs, np.ndarray)):
            raise ValueError("Coeffs supported as torch.Tensor or np.ndarray")
        self.coeffs = coeffs
        self._basis = basis
    
    def ten(self):
        return self.coeffs

    def basis(self):
        return self._basis

    def __call__(self, x):
        """
        Evaluate a multivariate polynomial at a point x in^d.
        """
        if self._basis == Basis.BERN:
            return decasteljau(self, x)
        elif self._basis == Basis.MONO: 
            return poly_eval(self, x)
        else:
            raise ValueError("Unrecognized basis type")
    
    def dim(self):
        return self.coeffs.ndim

def poly_eval(p : Polynomial, x : torch.Tensor):
    """
    Evaluate a multivariate polynomial at a point x in^d.
    
    Args:
        p: polynomial to evaluate
        x: a torch.Tensor of shape (m, d) representing m points in R^d.
        
    Returns:
        a scalar torch.Tensor containing the polynomial value at x.
    """
    assert p.basis() == Basis.MONO, "polynomial must be monomial basis"
    assert x.ndim == 2, "x must be a 2D tensor of shape (m, d)"
    p_ten = p.ten()
    d = p_ten.ndim
    m = x.shape[0]
    assert x.shape[1] == d, f"Each point must have dimension {d}"

    # Handle 1D case separately to prevent dimension mismatch
    if d == 1:
        n = p_ten.shape[0]
        monomials = x ** torch.arange(n, device=x.device, dtype=x.dtype)
        return monomials @ p_ten

    # Generate all exponent combinations (m, d)
    exponents = torch.cartesian_prod(*[torch.arange(n, device=p_ten.device) for n in p_ten.shape])  # (m, d)

    # Evaluate monomials at all x: x^exponents using broadcasting
    x_powers = x.unsqueeze(1) ** exponents.unsqueeze(0)  # shape: (p, m, d)
    monomials = torch.prod(x_powers, dim=2)  # shape: (p, m)

    # Dot each row of monomials with the flattened coeffs
    coeffs_flat = p_ten.flatten()  # shape: (m,)
    result = monomials @ coeffs_flat  # shape: (p,)

    return result

def decasteljau(p : Polynomial, x : torch.Tensor):
    """
    Evaluates a multivariate Bernstein polynomial for a batch of input vectors
    using the De Casteljau algorithm.

    Args:
        p: polynomial in Bernstein basis
        x: a torch.Tensor of shape (m, d) representing m points in R^d.

    Returns:
        a scalar torch.Tensor containing the polynomial value at x.

    Raises:
        ValueError: If the dimensionality of x (x.shape[1]) does not match the
                    number of dimensions of the coefficient tensor (p.coeffs.dim()).
    """
    assert p.basis() == Basis.BERN, "polynomial must be Bernstein basis"
    coeffs = p.coeffs
    
    # Validate input shapes
    if x.dim() != 2:
        raise ValueError(f"Input tensor x must be 2-dimensional (batch_size, d), but got {x.dim()} dimensions.")
        
    batch_size, d = x.shape

    if d != coeffs.dim():
        raise ValueError(
            f"Dimensionality of x ({d}) must match the dimensionality of the "
            f"polynomial's coefficient tensor ({coeffs.dim()})."
        )

    # Store the degrees of the polynomial in each dimension.
    # The degree in a dimension is one less than the size of the tensor in that dimension.
    degrees = [s - 1 for s in coeffs.shape]

    # Start with the original coefficients. We add a batch dimension to handle all
    # 'batch_size' evaluations simultaneously using broadcasting.
    # Shape changes from (n1+1, ..., nd+1) to (batch_size, n1+1, ..., nd+1).
    current_coeffs = coeffs.unsqueeze(0).repeat(batch_size, *([1] * d))

    # Iterate over each of the 'd' dimensions to apply De Casteljau's algorithm.
    for i in range(d):
        # Get the evaluation points for the current dimension for all vectors in the batch.
        t = x[:, i]

        # Reshape t to allow broadcasting with the coefficient tensor.
        # For dimension i, the coefficient tensor has d-i polynomial dimensions remaining.
        # Shape of t changes from (batch_size,) to (batch_size, 1, 1, ...).
        view_shape = [batch_size] + [1] * (d - i)
        t = t.view(*view_shape)

        # The degree for the current dimension being processed.
        degree = degrees[i]

        # Apply the De Casteljau recurrence 'degree' times.
        for _ in range(degree):
            current_coeffs = (
                (1 - t) * current_coeffs[:, :-1, ...] +
                t * current_coeffs[:, 1:, ...]
            )

        # After reducing a dimension, its size becomes 1. We squeeze it out
        # before processing the next dimension, unless it's the last one.
        if i < d - 1:
            current_coeffs = current_coeffs.squeeze(1)

    # After the loop, current_coeffs has shape (batch_size, 1, 1, ..., 1).
    # Squeeze all singleton dimensions to get the final (batch_size,) result.
    return current_coeffs.squeeze()

#def decasteljau_composition(p : Polynomial, q_list : list[Polynomial]):
#    """
#    Evaluates a multivariate Bernstein polynomial for a batch of input vectors
#    using the De Casteljau algorithm.
#
#    Args:
#        p: polynomial in Bernstein basis
#        q_list: a list of polynomials in the Bernstein basis to compose with p
#
#    Returns:
#        a scalar torch.Tensor containing the composed polynomial
#
#    Raises:
#        ValueError: If the dimensionality of x (x.shape[1]) does not match the
#                    number of dimensions of the coefficient tensor (p.coeffs.dim()).
#    """
#    assert p.basis() == Basis.BERN, "p must be Bernstein basis"
#    d = p.dim()
#    q_d = q_list[0].dim()
#    for q in q_list:
#        assert q.basis() == Basis.BERN, "All polynomials in q_list must be Bernstein basis"
#        assert q.dim() == q_d, "All polynomials in q_list must have the same dimensionality"
#
#    p_coeffs = p.coeffs
#    
#    # Validate input shapes
#    if len(q_list) != p.dim():
#        raise ValueError(f"Must supply {p.dim()} polynomials in q_list, but got {len(q_list)}.")
#        
#    # Store the degrees of the polynomial in each dimension.
#    # The degree in a dimension is one less than the size of the tensor in that dimension.
#    degrees = [s - 1 for s in p_coeffs.shape]
#
#    # Iterate over each of the 'd' dimensions to apply De Casteljau's algorithm.
#    for i, q_i in enumerate(q_list):
#        
#        q_i_fft = 
#
#        nk = p_coeffs.shape[i] - 1
#        assert nk >= 0, f"Polynomial p must have at least one coefficient in dimension {i}."
#
#        for _ in range(nk):
#
#    # After the loop, current_coeffs has shape (batch_size, 1, 1, ..., 1).
#    # Squeeze all singleton dimensions to get the final (batch_size,) result.
#    return current_coeffs.squeeze()

def binomial_torch(n, k, dtype=torch.double, device='cpu'):
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

def binomial(n, k):
    assert type(n) == type(k), "Type of n and k must match"
    if isinstance(n, np.ndarray):
        return np.round(comb(n, k, exact=False))
    else:
        assert type(n) == torch.Tensor, "Unrecognized tensor type"
        return binomial_torch(n, k)

def bernstein_to_monomial(p: Polynomial) -> Polynomial:
    if p.basis() != Basis.BERN:
        raise TypeError("Input must be a Polynomial in Bernstein basis.")

    bernstein_coeffs = p.ten()

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
        
        comb_nk = binomial_torch(n, k_, dtype=dtype, device=device)
        comb_kj = binomial_torch(k_, j_, dtype=dtype, device=device)
        
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
        
    return Polynomial(monomial_coeffs, basis=Basis.MONO)

def monomial_to_bernstein(p : Polynomial) -> Polynomial:
    if p.basis() != Basis.MONO:
        raise TypeError("Input must be a Polynomial in monomial basis.")

    monomial_coeffs = p.ten()
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
        
        comb_jk = binomial_torch(j_, k_, dtype=dtype, device=device)
        comb_nk = binomial_torch(n, k_, dtype=dtype, device=device)
        
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

    return Polynomial(bernstein_coeffs, basis=Basis.BERN)

def create_d_separable_tensor(index_fcn, shape, tensor_type=torch.Tensor, dtype=torch.float64):
    if tensor_type == torch.Tensor:
        # Compute g values for each dimension
        g_vectors = [torch.tensor([index_fcn(d, i) for i in range(shape[d])], dtype=dtype) for d in range(len(shape))]

        # Broadcasting multiplication (outer product)
        product = torch.ones(shape, dtype=dtype)
    else:
        g_vectors = [np.array([index_fcn(d, i) for i in range(shape[d])], dtype=dtype) for d in range(len(shape))]
        product = np.ones(shape, dtype=dtype)

    for dim, g_vec in enumerate(g_vectors):
        broadcast_shape = [1] * len(shape)
        broadcast_shape[dim] = shape[dim]
        product *= g_vec.reshape(broadcast_shape)
    
    return product

def fft_convolve_nd(a, b):
    """
    Computes the convolution of two tensors using FFT, matching the broadcasted shape. Accepts numpy or torch tensors.
    """
    assert type(a) == type(b), "Both inputs must be same type"
    is_np = isinstance(a, np.ndarray)

    size = [a.shape[i] + b.shape[i] - 1 for i in range(a.ndim)]
    fft_size = [scifft.next_fast_len(s) for s in size]

    if is_np: 
        A = scifft.rfftn(a, s=fft_size)
        B = scifft.rfftn(b, s=fft_size)
        C = A * B
        c = scifft.irfftn(C, s=fft_size)
    else:
        A = torch.fft.rfftn(a, s=fft_size)
        B = torch.fft.rfftn(b, s=fft_size)
        C = A * B
        c = torch.fft.irfftn(C, s=fft_size)

    # Truncate to actual convolution size
    slices = tuple(slice(0, s) for s in size)
    return c[slices]

def poly_product(p_list : list[Polynomial]):
    """
    Computes the product of a list of multivariate polynomials using FFT-based convolution.
    
    Each polynomial in p_list is a torch.Tensor of coefficients of increasing dimension:
    e.g. p_list = [p(x1), p(x1,x2), ..., p(x1,...,xd)]. The product is most efficient
    when this order is given, but it is not necessary
    
    Returns:
        product: torch.Tensor of coefficients of the product polynomial.
    """
    assert len(p_list) >= 1, "p_list must contain at least one polynomial"
    p_basis = p_list[0].basis()
    for p in p_list[1:]:
        assert p.basis() == p_basis, "All polynomials must be in the same basis"
    
    p_list_lcl = copy.deepcopy(p_list)

    if p_basis == Basis.BERN:
        # Pre-weight each polynomial before convolution
        for l in range(len(p_list_lcl)):
            p_ten = p_list_lcl[l].ten()
            pre_weight = create_d_separable_tensor(lambda dim, i : comb(p_ten.shape[dim] - 1, i), p_ten.shape, tensor_type=type(p_ten), dtype=p_ten.dtype)
            p_list_lcl[l] = Polynomial(p_ten * pre_weight, basis=Basis.BERN)
    
    product_ten = p_list_lcl[0].ten()
    for i in range(1, len(p_list_lcl)):
        p_ten = p_list_lcl[i].ten()

        # Pad product to match p's dimensionality
        if product_ten.ndim < p_ten.ndim:
            pad_dims = p_ten.ndim - product_ten.ndim
            product_ten = product_ten[(...,) + (None,) * pad_dims]

        # Broadcast shapes to match for convolution
        product_shape = product_ten.shape
        p_shape = p_ten.shape
        max_shape = [max(r, s) for r, s in zip(product_shape, p_shape)]

        # Pad each tensor to the appropriate shape
        product_pad = list(product_shape)
        p_pad = list(p_shape)
        while len(product_pad) < len(max_shape):
            product_pad.append(1)
        while len(p_pad) < len(max_shape):
            p_pad.append(1)

        # Compute FFT-based convolution
        product_ten = fft_convolve_nd(product_ten, p_ten)
    
    if p_basis == Basis.BERN:
        # Post-weight the convoluted polynomial
        post_weight = create_d_separable_tensor(lambda dim, s : 1.0 / comb(product_ten.shape[dim] - 1, s), product_ten.shape, tensor_type=type(p_ten), dtype=p_ten.dtype)
        product_ten *= post_weight

    return Polynomial(product_ten, basis=p_basis)

def marginal(p : Polynomial, dims : list[int]):
    """
    Integrate a polynomial over the specified dims from x_l = 0 to x_l = 1

    Args:
        p : polynomial
        dims : list of integers describing which dimensions to integrate over
    """
    if p.basis() == Basis.BERN:
        weight = 1.0
        p_ten = p.ten()
        for d in dims:
            weight /= p_ten.shape[d]
        
        # Sum along the desired dimensions (integrate) then weight by the area of the basis polynomials
        summed_tensor = p.ten()
        if isinstance(summed_tensor, torch.Tensor):
            summed_tensor = torch.sum(p_ten, dim=dims)
        else:
            summed_tensor = np.sum(summed_tensor, axis=tuple(dims))
        return Polynomial(weight * summed_tensor, basis=Basis.BERN)
    else:
        result = p.ten().clone()
        
        # Sort dimensions in descending order to avoid index shifting issues
        dim_list_sorted = sorted(dims, reverse=True)
        
        for dim in dim_list_sorted:
            assert dim < p.ten().ndim

            # Get the size of this dimension (max degree + 1)
            degree_plus_one = result.shape[dim]
            
            # Create integration weights: 1/(k+1) for degree k
            # Shape: [1, 1, ..., degree_plus_one, 1, 1, ...]
            weights_shape = [1] * result.ndim
            weights_shape[dim] = degree_plus_one
            
            if isinstance(result, torch.Tensor):
                weights = torch.zeros(weights_shape, dtype=result.dtype, device=result.device)
            else:
                weights = np.zeros(weights_shape, dtype=result.dtype)
            for k in range(degree_plus_one):
                # For term x^k, integral from 0 to 1 is 1/(k+1)
                weights_index = [slice(None)] * result.ndim
                weights_index[dim] = k
                weights[tuple(weights_index)] = 1.0 / (k + 1)
            
            # Apply integration weights
            result = result * weights
            
            # Sum along this dimension (integrate)
            if isinstance(result, torch.Tensor):
                summed_tensor = torch.sum(result, dim=dim, keepdim=False)
            else:
                result = np.sum(result, axis=dim, keepdims=False)
        
        return Polynomial(result, basis=Basis.MONO)

if __name__ == "__main__":

    #p_bern_1 = Polynomial(torch.randn(4,3,4,5), basis=Basis.BERN)
    #p_mono = bernstein_to_monomial(p_bern)

    #x = torch.rand(10, 4)
    #p_bern_eval = p_bern(x)
    #p_mono_eval = p_mono(x)
    #print("Bernstein eval: \n", p_bern_eval)
    #print("Monomial eval: \n", p_mono_eval)

    p_bern_1_tch = Polynomial(torch.randn(4,3,4,5), basis=Basis.BERN)
    p_bern_2_tch = Polynomial(torch.randn(4,3,4,5), basis=Basis.BERN)
    p_bern_1_np = Polynomial(p_bern_1_tch.ten().numpy(), basis=Basis.BERN)
    p_bern_2_np = Polynomial(p_bern_2_tch.ten().numpy(), basis=Basis.BERN)

    tch_prod = poly_product([p_bern_1_tch, p_bern_2_tch])
    np_prod = poly_product([p_bern_1_np, p_bern_2_np])
    print(np.max(np.abs(tch_prod.ten().numpy() - np_prod.ten())))
