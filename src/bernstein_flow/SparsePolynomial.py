import itertools as it
import torch
import math

from functools import reduce
import numpy as np
from enum import Enum

import scipy.fft as scifft
from scipy.special import gammaln
from scipy.spatial import Rectangle
from scipy.signal import convolve
import scipy.sparse as sp

from .Polynomial import Basis, Polynomial

class SparseBernsteinPolynomial:
    def __init__(self, coeffs : np.ndarray, indices : np.ndarray, degrees : np.ndarray, stable = True):
        assert degrees.shape == indices.shape
        assert coeffs.ndim == 1 and coeffs.shape[0] == indices.shape[0]
        self.coeffs = coeffs
        self.n_components, self.dim = indices.shape
        self.idx = indices
        self.deg = degrees
        self.stable = stable

    
    def __call__(self, x : np.ndarray):
        batch_size, d = x.shape 

        assert d == self.dim, "Dimension of x does not match polynoial dimension"

        log_basis_total = np.zeros((batch_size, self.n_components))

        for i in range(d):
            idx_i = self.idx[:, i][None, :]
            deg_i = self.deg[:, i][None, :]
            x_i = x[:, i][:, None]

            log_binoms = gammaln(deg_i + 1) - gammaln(idx_i + 1) - gammaln(deg_i - idx_i + 1)

            # Evaluate the bernstein basis functions for the current dimension and add it to the total
            log_basis_total += (
                log_binoms
                + idx_i * np.log(x_i)
                + (deg_i - idx_i) * np.log(1.0 - x_i)
            )
        
        basis_vals = np.exp(log_basis_total)
        return basis_vals @ self.coeffs
    
#class SparsePolynomial:
#    def __init__(self, coeffs, basis = Basis.MONO, stable = False, dtype = None):
#        """
#        Create a d-dimensional polynomial with a coefficient tensor.
#
#        Args:
#            coeffs : coefficient tensor (scipy coo_array)
#            basis : coefficient basis
#            operation_mode : ['fast', 'stable'] specify if default is to use fast operations or numerically stable operations
#        """
#        self.coeffs = coeffs
#
#        self._basis = basis
#        self.stable = stable
#    
#    def scoeffs(self):
#        return self.coeffs
#
#    def basis(self):
#        return self._basis
#    
#    def set_type(self, dtype):
#        if dtype != self.coeffs.dtype:
#            self.coeffs = self.coeffs.astype(dtype)
#
#    def __call__(self, x):
#        """
#        Evaluate a multivariate polynomial at a point x in^d.
#        """
#        if self._basis == Basis.BERN:
#            return decasteljau(self, x)
#        elif self._basis == Basis.MONO: 
#            return poly_eval(self, x)
#        else:
#            raise ValueError("Unrecognized basis type")
#    
#    def dim(self):
#        return self.coeffs.ndim
#
#    def shape(self):
#        return self.coeffs.shape
    
#def sparse_poly_eval(p : SparsePolynomial, x : np.ndarray):
#    """
#    Evaluate a multivariate sparse polynomial in the monomial basis at points x.
#
#    Args:
#        p: polynomial to evaluate; must be monomial basis 
#        x: a 2D array of shape (m, d), where m is the number of points and d is the dimension.
#
#    Returns:
#        An array of shape (m,) with the polynomial evaluated at each point.
#    """
#    assert p.basis() == Basis.MONO, "Polynomial must be in monomial basis"
#
#    coeffs = p.scoeffs()
#    d = p.dim()
#
#    assert x.ndim == 2, "Input x must be a 2D array (m, d)"
#    assert x.shape[1] == d, f"Each point must have dimension {d}"
#
#    batch_size = x.shape[0]
#
#    if coeffs.nnz == 0:
#        return np.zeros(batch_size)
#
#    values = coeffs.data
#    exponents = np.array(coeffs.coords).T
#
#    powered_x = np.power(x[:, np.newaxis, :], exponents[np.newaxis, :, :])
#
#    monomial_vals = np.prod(powered_x, axis=2)
#
#    weighted_sum = np.sum(values * monomial_vals, axis=1)
#    return weighted_sum

    
#def sparse_decasteljau(p : SparsePolynomial, x : np.ndarray):
#    """
#    Evaluates a multivariate sparse Bernstein polynomial for a batch of input vectors
#    using the De Casteljau algorithm.
#
#    Args:
#        p: polynomial in Bernstein basis
#        x: a np.ndarray of shape (m, d) representing m points in R^d.
#
#    Returns:
#        An array of shape (m,) with the polynomial evaluated at each point.
#
#    """
#    assert p.basis() == Basis.BERN, "polynomial must be Bernstein basis"
#    p_ten = p.ten()
#
#    # Validate input shapes
#    if x.ndim != 2:
#        raise ValueError(f"Input tensor x must be 2-dimensional (batch_size, d), but got {x.ndim} dimensions.")
#        
#    batch_size, d = x.shape
#
#    assert p.dim() == d, "x vector dimension does not match dimension of p"
#
#    degrees = [s - 1 for s in p_ten.shape]


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

        return Polynomial(total, basis=p_list[0].basis(), stable=stable)


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
            pre_weight = _create_d_separable_tensor(lambda dim, i : comb(p_ten.shape[dim] - 1, i), p_ten.shape, dtype=p_ten.dtype)
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
        post_weight = _create_d_separable_tensor(lambda dim, s : 1.0 / comb(product.shape[dim] - 1, s), product.shape, dtype=p_ten.dtype)
        product *= post_weight

    return Polynomial(product, basis=p_basis)

def poly_product_bernstein_direct(p_list : list[Polynomial]):
    for p in p_list:
        assert p.basis() == Basis.BERN, "All polynomials must be in the Bernstein same basis"
    p_list.sort(key=lambda p : p.dim())

    dtype = p_list[0].ten().dtype

    def mult(p : Polynomial, q : Polynomial):
        p_ten, q_ten = p.ten(), q.ten()
        max_dim = max(p_ten.ndim, q_ten.ndim)
        if p_ten.ndim < max_dim:
            p_ten = p_ten.reshape(p_ten.shape + (1,) * (max_dim - p_ten.ndim))
        elif q_ten.ndim < max_dim:
            q_ten = q_ten.reshape(q_ten.shape + (1,) * (max_dim - q_ten.ndim))

        pre_weight_A = _create_d_separable_tensor(lambda dim, i : comb(p_ten.shape[dim] - 1, i), p_ten.shape, dtype=dtype)
        pre_weight_B = _create_d_separable_tensor(lambda dim, i : comb(q_ten.shape[dim] - 1, i), q_ten.shape, dtype=dtype)

        # weighted control nets
        A_w = p_ten * pre_weight_A
        B_w = q_ten * pre_weight_B

        #if dtype == np.float128:
        #    product = direct_nd_convolve(A_w, B_w)
        #else:
        #    product = convolve(A_w, B_w, mode='full', method='direct')
        #print("convs dtype b4: ", A_w.dtype)
        product = convolve(A_w, B_w, mode='full', method='direct')
        #print("product_dtype af: ", product.dtype)

        post_weight = _create_d_separable_tensor(lambda dim, s : 1.0 / comb(product.shape[dim] - 1, s), product.shape, dtype=dtype)
        product *= post_weight
        #print("product dtype: " ,product.dtype, " poly dtype: ", Polynomial(product, basis=Basis.BERN, stable=True).coeffs.dtype)
        return Polynomial(product, basis=Basis.BERN, stable=True)
    
    prod = p_list[0]
    for p in p_list[1:]:
        prod = mult(prod, p)
    
    return prod

def marginal(p : Polynomial, dims : set[int], stable : bool = False):
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
        summed_tensor = np.sum(p.ten(), axis=tuple(dims), keepdims=False) if not stable else _stable_sum_reduction(p.ten(), axis=tuple(dims), keepdims=False)
        return Polynomial(weight * summed_tensor, basis=Basis.BERN, stable=stable) if len(dims) < p.dim() else weight * summed_tensor
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
            result = np.sum(result, axis=dim, keepdims=False) if not stable else _stable_sum_reduction(result, axis=dim, keepdims=False)
        
        return Polynomial(result, basis=Basis.MONO, stable=stable) if len(dims) < p.dim() else weight * summed_tensor

def integrate(p : Polynomial, region : Rectangle, stable : bool = False):
    assert p.dim() == region.m, "Region must have the same dimension as the polynomial being integrated"

    p_shape = np.array(p.shape())

    total_integral = 0.0
    for midx in np.ndindex(p.coeffs.shape):
        coeff = p.coeffs[midx]
        if coeff == 0.0:
            continue

        midx = np.array(midx)

        if p.basis() == Basis.BERN:
            int_values_1d = beta(midx + 1, p_shape - midx) * comb(p_shape - 1, midx) * (betainc(midx + 1, p_shape - midx, region.maxes) - betainc(midx + 1, p_shape - midx, region.mins))
        else:
            int_values_1d = (region.maxes**(midx + 1) - region.mins**(midx + 1)) / (midx + 1)
        total_integral += coeff * np.prod(int_values_1d)
    
    return total_integral

def mc_auc(p : Polynomial, n_samples : int, region : Rectangle = None):
    d = p.dim()
    if region is None:
        X = np.random.rand(n_samples, d)
        vol = 1.0
    else:
        X = np.random.uniform(low=region.mins, high=region.maxes, size=(n_samples, d))
        vol = region.volume()
    p_evals = p(X)
    return np.mean(p_evals) * vol


if __name__ == "__main__":

    coeffs = np.array([3.5, 9.2, -1.8])
    idx = np.array([[4,5,6], [9, 4, 3], [8, 2, 5]])
    deg = np.array([[10, 10, 10], [10, 10, 10], [12, 8, 9]])
    
    p_sparse = SparseBernsteinPolynomial(coeffs, idx, deg)
    
    coeffs_dense_1 = np.zeros(deg[0] + 1)
    coeffs_dense_1[*idx[0]] = coeffs[0]
    coeffs_dense_1[*idx[1]] = coeffs[1]
    p_dense_1 = Polynomial(coeffs_dense_1, basis=Basis.BERN)

    coeffs_dense_2 = np.zeros(deg[2] + 1)
    coeffs_dense_2[*idx[2]] = coeffs[2]
    p_dense_2 = Polynomial(coeffs_dense_2, basis=Basis.BERN)

    x = np.random.rand(5, 3)

    print("Sparse: ", p_sparse(x))
    print("Dense: ", p_dense_1(x) + p_dense_2(x))