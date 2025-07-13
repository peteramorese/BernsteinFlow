import itertools as it
import torch
import math

from functools import reduce
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
            raise ValueError("Input coeffs tensor supported as torch.Tensor or np.ndarray")

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

    def shape(self):
        return self.coeffs.shape
    
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
    p_ten = p.ten()

    # Validate input shapes
    if x.ndim != 2:
        raise ValueError(f"Input tensor x must be 2-dimensional (batch_size, d), but got {x.ndim} dimensions.")
        
    batch_size, d = x.shape

    if d != p_ten.ndim:
        raise ValueError(
            f"Dimensionality of x ({d}) must match the dimensionality of the "
            f"polynomial's coefficient tensor ({p_ten.ndim})."
        )

    degrees = [s - 1 for s in p_ten.shape]

    expand_shape = (batch_size,) + p_ten.shape
    current_coeffs = np.broadcast_to(p_ten, expand_shape) 

    # Iterate over each of the 'd' dimensions to apply De Casteljau's algorithm.
    for i in range(d):
        t = x[:, i]

        view_shape = [batch_size] + [1] * (d - i)
        t = t.reshape(*view_shape) 

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

def poly_sum(p_list : list[Polynomial], stable : bool = False):
    tensors = [p.ten() for p in p_list]

    # Compute the max size of each tensor
    max_size = np.zeros_like(tensors[0].shape, dtype=np.int32)
    for t in tensors:
        max_size = np.maximum(max_size, t.shape)

    # Pad all lower-degree tensors to the max size 
    for i in range(len(tensors)):
        pad_size = max_size - np.array(tensors[i].shape)
        pad = [(0, s) for s in pad_size]
        tensors[i] = np.pad(tensors[i], pad)

    if not stable:
        return Polynomial(np.add.reduce(tensors), basis=p_list[0].basis())
    else:
        # Kahan summation
        total = np.zeros(max_size, dtype=tensors[0].dtype)
        c = np.zeros(max_size, dtype=tensors[0].dtype)

        for t in tensors:
            y = t - c
            t_sum = total + y
            c = (t_sum - total) - y
            total = t_sum

        return Polynomial(total, basis=p_list[0].basis())


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

def stable_sum_reduction(a : np.ndarray, axis=None, keepdims=False):
    """
    Substitute for np.sum with better numerical stability (at the cost of performance)
    """
    arr = np.asanyarray(a)
    # Global sum
    if axis is None:
        # fsum over the flattened data
        total = math.fsum(arr.ravel().tolist())
        if keepdims:
            # return as an array of shape (1,1,...,1)
            shape = tuple(1 for _ in range(arr.ndim))
            return np.full(shape, total)
        else:
            return total

    # Normalize axis to a tuple of positive ints
    if isinstance(axis, int):
        axes = (axis % arr.ndim, )
    else:
        axes = tuple(ax % arr.ndim for ax in axis)
    axes = sorted(axes)

    # Determine the shape of the result
    out_shape = []
    for i, dim in enumerate(arr.shape):
        if i not in axes:
            out_shape.append(dim)
        elif keepdims:
            out_shape.append(1)

    # Move the summation axes to the end, reshape so that
    # we have shape (..., M), where M is the product of lengths over axes
    perm = [i for i in range(arr.ndim) if i not in axes] + axes
    arr_t = np.transpose(arr, perm)
    outer_shape = arr_t.shape[:arr.ndim - len(axes)]
    M = int(np.prod(arr_t.shape[arr.ndim - len(axes):]))
    flat = arr_t.reshape(-1, M)

    # Apply math.fsum to each row
    summed = [math.fsum(row.tolist()) for row in flat]
    result = np.array(summed, dtype=float).reshape(outer_shape)

    # If keepdims, insert back the reduced axes as length-1 dims
    if keepdims:
        for ax in axes:
            result = np.expand_dims(result, axis=ax)

    return result

def split_factor_poly_product(summands_list : list[list[Polynomial]], shrink_to_size : bool = False):
    """
    Compute the product of a list polynomial sums. Returns a list of polynomials whose sum is the product of the input polynomials.
    """
    assert len(summands_list) >= 1, "p_list must contain at least one polynomial"
    p_basis = summands_list[0][0].basis()
    for summands in summands_list[1:]:
        for p in summands:
            assert p.basis() == p_basis, "All polynomials must be in the same basis"
    
    summands_tensors = [[p.ten() for p in p_list] for p_list in summands_list]

    if p_basis == Basis.BERN:
        # Pre-weight each polynomial before convolution
        for summands in summands_tensors:
            for l in range(len(summands)):
                p_ten = summands[l]
                pre_weight = create_d_separable_tensor(lambda dim, i : comb(p_ten.shape[dim] - 1, i), p_ten.shape, dtype=p_ten.dtype)
                summands[l] = p_ten * pre_weight
    

    # Expand all tensors to be the same dimensionality
    max_dim = max(t.ndim for summands in summands_tensors for t in summands)
    for summands in summands_tensors:
        for i in range(len(summands)):
            t = summands[i]
            if t.ndim < max_dim:
                summands[i] = t.reshape(t.shape + (1,) * (max_dim - t.ndim))
    
    # Compute the final shape and fft sizes of each product
    final_shapes = []
    max_fft_size = np.zeros_like(summands_tensors[0][0].shape, dtype=np.int32) # Track the largest fft_size, and use that for all products
    for factor_tensor_list in it.product(*summands_tensors):
        final_shape = np.array(factor_tensor_list[0].shape, dtype=np.int32)
        for t in factor_tensor_list[1:]:
            final_shape += np.array(t.shape, dtype=np.int32) - 1

        for i in range(len(final_shape)):
            nfl = scifft.next_fast_len(final_shape[i])
            if nfl > max_fft_size[i]:
                max_fft_size[i] = nfl

        final_shapes.append(final_shape)

    # FFT all tensors
    test = summands_tensors[0][0]
    scifft.rfftn(test, s=max_fft_size)
    freq_summands_tensors = [[scifft.rfftn(t, s=max_fft_size) for t in summands] for summands in summands_tensors]
    products = []
    for factor_tensor_list in it.product(*freq_summands_tensors):
        freq_product = np.multiply.reduce(factor_tensor_list)
        prod = scifft.irfftn(freq_product, s=max_fft_size)
        products.append(prod)


    if shrink_to_size:
        for prod, final_shape in zip(products, final_shapes):
            slices = tuple(slice(0, s) for s in final_shape)
            prod = prod[slices]

    if p_basis == Basis.BERN:
        for prod in products:
            # Post-weight the convoluted polynomial
            post_weight = create_d_separable_tensor(lambda dim, s : 1.0 / comb(prod.shape[dim] - 1, s), prod.shape, dtype=p_ten.dtype)
            prod *= post_weight

    return [Polynomial(product, basis=p_basis) for product in products]

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
        n_splits = int(np.ceil((np.log10(mag_max) - np.log10(mag_min)) / mag_range))
        summands = []
        mask = np.zeros_like(p_ten).astype(bool) # initially no elements are masked out
        for i in range(n_splits + 1):
            threshold_i = 10.0 ** (np.log10(mag_min) + i * mag_range)
            below_thresh = (p_ten_abs <= threshold_i) if i < n_splits else True
            include = ~mask & below_thresh # Determine which elements to include in this summand: not elements already included (in the mask) and elements that are above the threshold
            summands.append(Polynomial(include * p_ten, basis=basis))
            mask = mask | below_thresh # Update the mask to reflect elements included in this summand
        factor_list.append(summands)
    
    return factor_list
    
    #product_terms = [poly_product(factors) for factors in product(*factor_list)]
    
    #return [Polynomial(term, basis=p_list[0].basis()) for term in product_terms]
    
def marginal(p : Polynomial, dims : list[int], stable : bool = False):
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
        summed_tensor = np.sum(p.ten(), axis=tuple(dims), keepdims=False) if not stable else stable_sum_reduction(p.ten(), axis=tuple(dims), keepdims=False)
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
            result = np.sum(result, axis=dim, keepdims=False) if not stable else stable_sum_reduction(result, axis=dim, keepdims=False)
        
        return Polynomial(result, basis=Basis.MONO)

if __name__ == "__main__":

    #p_bern_1 = Polynomial(torch.randn(4,3,4,5), basis=Basis.BERN)
    #p_mono = bernstein_to_monomial(p_bern)

    #x = torch.rand(10, 4)
    #p_bern_eval = p_bern(x)
    #p_mono_eval = p_mono(x)
    #print("Bernstein eval: \n", p_bern_eval)
    #print("Monomial eval: \n", p_mono_eval)

    p = Polynomial(np.exp(np.random.uniform(low=-5, high=12, size=(4, 4))), basis=Basis.MONO)
    q = Polynomial(np.exp(np.random.uniform(low=-5, high=12, size=(4, 4))), basis=Basis.MONO)

    split = stable_split_factors([p, q], mag_range=2.0)

    prod = poly_product([p, q])
    stable_prod = split_factor_poly_product(split)

    x = np.random.rand(5, 2)
    print("prod: ", prod(x))
    print("prod sep: ", sum(sprod(x) for sprod in stable_prod))


    p = Polynomial(np.array([[1, 2], [3, 4.5]]))
    q = Polynomial(np.array([[6, 7], [2, 5.5]]))

    print(poly_sum([p, q], stable=False).ten())
    print(poly_sum([p, q], stable=True).ten())

    #print("old prod: ", p_prod_old(x))

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
