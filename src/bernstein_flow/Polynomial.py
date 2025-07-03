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
        if isinstance(coeffs, torch.Tensor):
            self.coeffs = coeffs.detach().cpu().numpy()
        elif isinstance(coeffs, np.ndarray):
            self.coeffs = coeffs
        else:
            raise ValueError("Coeffs supported as torch.Tensor or np.ndarray")

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
    
def poly_eval(p : Polynomial, x : np.ndarray):
    """
    Evaluate a multivariate polynomial in the monomial basis at points x.

    Args:
        p: polynomial to evaluate; must be monomial basis 
        x: a 2D array of shape (m, d), where m is the number of points and d is the dimension.

    Returns:
        An array of shape (m,) with the polynomial evaluated at each point.
    """
    assert p.basis() == Basis.MONO, "Polynomial must be in monomial basis"
    p_ten = p.ten()
    d = p_ten.ndim

    assert x.ndim == 2, "Input x must be a 2D array (m, d)"
    assert x.shape[1] == d, f"Each point must have dimension {d}"

    # Handle univariate case
    if d == 1:
        n = p_ten.shape[0]
        exponents = np.arange(n, dtype=p_ten.dtype)
        monomials = x ** exponents  # shape: (m, n)
        return monomials @ p_ten

    # Multivariate case
    # Generate all combinations of exponents
    shapes = [range(n) for n in p_ten.shape]
    grids = np.meshgrid(*shapes, indexing='ij')
    exponents = np.stack([g.flatten() for g in grids], axis=1)  # shape: (num_terms, d)
    x_powers = x[:, None, :] ** exponents[None, :, :]  # shape: (m, num_terms, d)
    monomials = np.prod(x_powers, axis=2)  # shape: (m, num_terms)
    result = monomials @ p_ten.flatten()

    return result

def decasteljau(p : Polynomial, x : np.ndarray):
    """
    Evaluates a multivariate Bernstein polynomial for a batch of input vectors
    using the De Casteljau algorithm.

    Args:
        p: polynomial in Bernstein basis
        x: a np.ndarray of shape (m, d) representing m points in R^d.

    Returns:
        An array of shape (m,) with the polynomial evaluated at each point.

    """
    assert p.basis() == Basis.BERN, "polynomial must be Bernstein basis"
    coeffs = p.coeffs

    # Validate input shapes
    if x.ndim != 2:
        raise ValueError(f"Input tensor x must be 2-dimensional (batch_size, d), but got {x.ndim} dimensions.")
        
    batch_size, d = x.shape

    if d != coeffs.ndim:
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
    expand_shape = (batch_size,) + coeffs.shape
    current_coeffs = np.broadcast_to(coeffs, expand_shape) 

    # Iterate over each of the 'd' dimensions to apply De Casteljau's algorithm.
    for i in range(d):
        # Get the evaluation points for the current dimension for all vectors in the batch.
        t = x[:, i]

        # Reshape t to allow broadcasting with the coefficient tensor.
        # For dimension i, the coefficient tensor has d-i polynomial dimensions remaining.
        # Shape of t changes from (batch_size,) to (batch_size, 1, 1, ...).
        view_shape = [batch_size] + [1] * (d - i)
        t = t.reshape(*view_shape) 

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
            current_coeffs = np.squeeze(current_coeffs, axis=1)

    # After the loop, current_coeffs has shape (batch_size, 1, 1, ..., 1).
    # Squeeze all singleton dimensions to get the final (batch_size,) result.
    return np.squeeze(current_coeffs)

def binomial(n, k, dtype=None):
    n = np.asarray(n, dtype=dtype)
    k = np.asarray(k, dtype=dtype)
    return np.round(comb(n, k, exact=False)).astype(dtype if dtype else float)

def bernstein_to_monomial(p: Polynomial) -> Polynomial:
    if p.basis() != Basis.BERN:
        raise TypeError("Input must be a Polynomial in Bernstein basis.")

    bernstein_coeffs = p.ten()

    d = bernstein_coeffs.ndim
    monomial_coeffs = bernstein_coeffs.copy()

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(d):
        n = monomial_coeffs.shape[i] - 1
        if n < 0:
            continue

        # Construct conversion matrix M
        k_ = np.arange(n + 1).reshape(-1, 1)  # (n+1, 1)
        j_ = np.arange(n + 1).reshape(1, -1)  # (1, n+1)

        mask = (j_ <= k_).astype(float)

        comb_nk = binomial(n, k_, dtype=bernstein_coeffs.dtype)
        comb_kj = binomial(k_, j_, dtype=bernstein_coeffs.dtype)

        signs = (-1.0) ** (k_ - j_)

        M = comb_nk * comb_kj * signs * mask
        M = M.astype(bernstein_coeffs.dtype)

        # Build einsum string for contraction
        subscripts = list(alphabet[:d])
        contract_idx = subscripts[i]
        new_idx = alphabet[d]

        mat_subscripts = f"{new_idx}{contract_idx}"
        product_subscripts = list(subscripts)
        product_subscripts[i] = new_idx
        einsum_str = f"{''.join(subscripts)},{mat_subscripts}->{''.join(product_subscripts)}"

        monomial_coeffs = np.einsum(einsum_str, monomial_coeffs, M)

    return Polynomial(monomial_coeffs, basis=Basis.MONO)

def monomial_to_bernstein(p: Polynomial) -> Polynomial:
    if p.basis() != Basis.MONO:
        raise TypeError("Input must be a Polynomial in monomial basis.")

    monomial_coeffs = p.ten()

    d = monomial_coeffs.ndim
    bernstein_coeffs = monomial_coeffs.copy()
    dtype = monomial_coeffs.dtype

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(d):
        n = bernstein_coeffs.shape[i] - 1
        if n < 0:
            continue

        # Build the Monomial-to-Bernstein conversion matrix N
        j_ = np.arange(n + 1).reshape(-1, 1)  # (n+1, 1)
        k_ = np.arange(n + 1).reshape(1, -1)  # (1, n+1)

        mask = (k_ <= j_).astype(float)
        comb_jk = binomial(j_, k_, dtype=dtype)
        comb_nk = binomial(n, k_, dtype=dtype)

        N = (comb_jk / (comb_nk + 1e-12)) * mask
        N = N.astype(dtype)

        # Apply the transformation using einsum
        subscripts = list(alphabet[:d])
        contract_idx = subscripts[i]
        new_idx = alphabet[d]

        mat_subscripts = f"{new_idx}{contract_idx}"
        product_subscripts = list(subscripts)
        product_subscripts[i] = new_idx

        einsum_str = f"{''.join(subscripts)},{mat_subscripts}->{''.join(product_subscripts)}"
        bernstein_coeffs = np.einsum(einsum_str, bernstein_coeffs, N)

    return Polynomial(bernstein_coeffs, basis=Basis.BERN)

def create_d_separable_tensor(index_fcn, shape, dtype=np.float64):
    g_vectors = [np.array([index_fcn(d, i) for i in range(shape[d])], dtype=dtype) for d in range(len(shape))]
    product = np.ones(shape, dtype=dtype)

    for dim, g_vec in enumerate(g_vectors):
        broadcast_shape = [1] * len(shape)
        broadcast_shape[dim] = shape[dim]
        product *= g_vec.reshape(broadcast_shape)
    
    return product

def poly_product(p_list : list[Polynomial]):
    """
    Compute the product of a list of polynomials.
    """
    assert len(p_list) >= 1, "p_list must contain at least one polynomial"
    p_basis = p_list[0].basis()
    for p in p_list[1:]:
        assert p.basis() == p_basis, "All polynomials must be in the same basis"
    
    tensors = [p.ten() for p in p_list]

    if p_basis == Basis.BERN:
        # Pre-weight each polynomial before convolution
        for l in range(len(tensors)):
            p_ten = tensors[l]
            pre_weight = create_d_separable_tensor(lambda dim, i : comb(p_ten.shape[dim] - 1, i), p_ten.shape, dtype=p_ten.dtype)
            tensors[l] = p_ten * pre_weight
    
    # Expand all tensors to be the same dimensionality
    max_dim = max(t.ndim for t in tensors)
    for i in range(len(tensors)):
        t = tensors[i]
        if t.ndim < max_dim:
            tensors[i] = t.reshape(t.shape + (1,) * (max_dim - t.ndim))
    
    # Compute the shape of the final product
    final_shape = np.array(tensors[0].shape, dtype=np.int32)
    for t in tensors[1:]:
        final_shape += np.array(t.shape, dtype=np.int32) - 1
    
    # FFT all tensors
    fft_size = [scifft.next_fast_len(s) for s in final_shape]
    freq_tensors = [scifft.rfftn(t, s=fft_size) for t in tensors]
    freq_product = np.multiply.reduce(freq_tensors)
    product = scifft.irfftn(freq_product, s=fft_size)


    slices = tuple(slice(0, s) for s in final_shape)
    product = product[slices]

    if p_basis == Basis.BERN:
        # Post-weight the convoluted polynomial
        post_weight = create_d_separable_tensor(lambda dim, s : 1.0 / comb(product.shape[dim] - 1, s), product.shape, dtype=p_ten.dtype)
        product *= post_weight

    return Polynomial(product, basis=p_basis)

def stable_split_factors(p_list : list[Polynomial], mag_range : float = 6.0):
    """
    Splits each factor into multiple polynomials depending on the magnitudes of the coefficients to achieve better numerical stability. Returns a list of polynomials
    whose sum is the product of the input polynomials

    Args:
        p_list : list polynomial factors to split
        mag_range : range of (base-10) magnitude for each factor. The range is used to determine how many splits to make based on the magnitudes of the coefficients.
                   A larger range will result in fewer splits, while a smaller range will result in more splits
    """
    basis = p_list[0].basis()
    factor_list = [] 
    #is_np = isinstance(p_list[0].ten(), np.ndarray)
    for p in p_list:
        p_ten = p.ten()
        p_ten_abs = np.abs(p_ten)
        mag_min, mag_max = p_ten_abs.min(), p_ten_abs.max()
        #print("mag min: ", mag_min, " mag max: ", mag_max)
        #print("log min: " , np.log10(mag_min), " log max: ", np.log10(mag_max), " mag range: " , mag_range)
        n_splits = int(np.ceil((np.log10(mag_max) - np.log10(mag_min)) / mag_range))
        #n_splits = int(np.ceil((mag_max - mag_min) / threshold))
        #print("n_splits: ", n_splits)
        #input("...")
        summands = []
        mask = np.zeros_like(p_ten).astype(bool) # initially no elements are masked out
        for i in range(n_splits + 1):
            threshold_i = 10.0 ** (np.log10(mag_min) + i * mag_range)
            #print("i : ", i, " threshold_i: ", threshold_i)
            below_thresh = (p_ten_abs <= threshold_i) if i < n_splits else True
            #print("  below thresh: \n", below_thresh)
            include = ~mask & below_thresh # Determine which elements to include in this summand: not elements already included (in the mask) and elements that are above the threshold
            #print("  include mask: \n", include)
            summands.append(Polynomial(include * p_ten, basis=basis))
            mask = mask | below_thresh # Update the mask to reflect elements included in this summand
        factor_list.append(summands)
    
    return factor_list
    
    #product_terms = [poly_product(factors) for factors in product(*factor_list)]
    
    #return [Polynomial(term, basis=p_list[0].basis()) for term in product_terms]
    

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
        summed_tensor = np.sum(p.ten(), axis=tuple(dims))
        return Polynomial(weight * summed_tensor, basis=Basis.BERN)
    else:
        result = p.ten().copy()
        
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
            
            weights = np.zeros(weights_shape, dtype=result.dtype)
            for k in range(degree_plus_one):
                # For term x^k, integral from 0 to 1 is 1/(k+1)
                weights_index = [slice(None)] * result.ndim
                weights_index[dim] = k
                weights[tuple(weights_index)] = 1.0 / (k + 1)
            
            # Apply integration weights
            result = result * weights
            
            # Sum along this dimension (integrate)
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

    p_bern_1 = Polynomial(np.random.randn(4,3,4,5), basis=Basis.BERN)
    p_bern_2 = Polynomial(np.random.randn(4,3,4,5), basis=Basis.BERN)

    p_prod = poly_product([p_bern_1, p_bern_2]) 

    x = np.random.rand(5, 4)
    print("prod: ", p_prod(x))

    #tch_prod = poly_product([p_bern_1_tch, p_bern_2_tch])
    #np_prod = poly_product([p_bern_1_np, p_bern_2_np])
    #print(np.max(np.abs(tch_prod.ten().numpy() - np_prod.ten())))

    #p_tch = Polynomial(torch.randn(4,3,4,5), basis=Basis.BERN)
    #p_np = Polynomial(p_tch.ten().numpy(), basis=Basis.BERN)
    #x = torch.rand(5, 4)
    #print("tch: ", decasteljau(p_tch, x))
    #print("np:  ", decasteljau(p_np, x.numpy()))

    #p_bern_tch = Polynomial(torch.randn(2, 2), basis=Basis.BERN)
    #p_bern_np = Polynomial(p_bern_tch.ten().numpy(), basis=Basis.BERN)

    #p_mono_tch = bernstein_to_monomial(p_bern_tch)
    #print(p_mono_tch.ten())
    #p_mono_np = bernstein_to_monomial(p_bern_np)
    #print(p_mono_np.ten())
    #x = torch.rand(5, 2)
    #print("tch: ", p_mono_tch(x))
    #print("np:  ", p_mono_np(x.numpy()))

    #p_bern_tch_a = monomial_to_bernstein(p_mono_tch)
    #p_bern_np_a = monomial_to_bernstein(p_mono_np)
    #print("tch: ", p_bern_tch_a(x))
    #print("np:  ", p_bern_np_a(x.numpy()))

    #p = Polynomial(np.exp(np.random.randint(1, 10, (4, 4))), basis=Basis.MONO)
    #q = Polynomial(np.exp(np.random.randint(1, 10, (4, 4))), basis=Basis.MONO)
    #print("p : \n", p.ten())
    #print("q : \n", q.ten())
    #print("")
    #factors = stable_split_factors([p, q], mag_range=2.0)
    #for f in factors[0]:
    #    print("p term: \n", f.ten().astype(int))
    #print("")
    #print("Sum of p terms: \n", sum([f.ten() for f in factors[0]]).astype(int))
    #print("")
    #for f in factors[1]:
    #    print("q term: \n", f.ten().astype(int))
    #print("")
    #print("Sum of q terms: \n", sum([f.ten() for f in factors[1]]).astype(int))
