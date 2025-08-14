import itertools as it
import torch
import math

from functools import reduce
import numpy as np
from enum import Enum
from copy import copy

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
    
    def __mul__(self, other):
        if isinstance(other, SparseBernsteinPolynomial):
            assert other.dim == self.dim, "Dimensions do not match"

            return self.multiply(other) 
        else:
            raise ValueError(f"Unrecognized (*) operand of type f{type(other)}")


    def multiply(self, other):
        """
        Multiply two sparse Bernstein polynomials.
        Returns the combination product (terms are not consolidated automatically)
        """
        assert self.dim == other.dim, "Polynomial dimensions must match"

        Ka, d = self.n_components, self.dim
        Kb = other.n_components

        # Shapes: (Ka, d) and (Kb, d) → (Ka, Kb, d)
        idx_a = self.idx[:, None, :]  # (Ka, 1, d)
        deg_a = self.deg[:, None, :]  # (Ka, 1, d)
        idx_b = other.idx[None, :, :] # (1, Kb, d)
        deg_b = other.deg[None, :, :] # (1, Kb, d)

        idx_sum = idx_a + idx_b      # (Ka, Kb, d)
        deg_sum = deg_a + deg_b      # (Ka, Kb, d)

        # Compute log gamma factor in one go
        log_binom_a = gammaln(deg_a + 1) - gammaln(idx_a + 1) - gammaln(deg_a - idx_a + 1)
        log_binom_b = gammaln(deg_b + 1) - gammaln(idx_b + 1) - gammaln(deg_b - idx_b + 1)
        log_binom_sum = gammaln(deg_sum + 1) - gammaln(idx_sum + 1) - gammaln(deg_sum - idx_sum + 1)

        log_gamma = np.sum(log_binom_a + log_binom_b - log_binom_sum, axis=2)  # (Ka, Kb)

        # Coefficient signs & logs
        sign_a = np.sign(self.coeffs)[:, None]   # (Ka, 1)
        sign_b = np.sign(other.coeffs)[None, :]  # (1, Kb)
        sign_product = sign_a * sign_b           # (Ka, Kb)

        log_abs_a = np.log(np.abs(self.coeffs))[:, None]  # (Ka, 1)
        log_abs_b = np.log(np.abs(other.coeffs))[None, :] # (1, Kb)

        log_abs_product = log_abs_a + log_abs_b + log_gamma  # (Ka, Kb)

        # Flatten all
        new_idx = idx_sum.reshape(-1, d)
        new_deg = deg_sum.reshape(-1, d)
        new_coeffs = sign_product.ravel() * np.exp(log_abs_product.ravel())

        return SparseBernsteinPolynomial(new_coeffs, new_idx, new_deg, stable=self.stable)
    
    def marginal(self, dims : set[int]):
        """
        Integrate the polynomial over the given dimensions (dims is a set of axis indices).
        Returns a new SparseBernsteinPolynomial in the remaining dimensions.
        """
        new_coeffs = self.coeffs.copy()
        
        removed_degs = self.deg[:, list(dims)]
        new_coeffs /= np.prod(removed_degs + 1, axis=1)
        
        keep_dims = [i for i in range(self.dim) if i not in dims]
        return SparseBernsteinPolynomial(new_coeffs, self.idx[:, keep_dims], self.deg[:, keep_dims], stable=self.stable)


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


    coeffs = np.array([6.5, 4.2, -2.8, 1.1])
    idx = np.array([[4,5,6], [9, 4, 3], [8, 2, 5], [0, 5, 4]])
    deg = np.array([[10, 10, 10], [10, 10, 10], [12, 8, 9], [12, 8, 9]])
    
    q_sparse = SparseBernsteinPolynomial(coeffs, idx, deg)
    
    coeffs_dense_1 = np.zeros(deg[0] + 1)
    coeffs_dense_1[*idx[0]] = coeffs[0]
    coeffs_dense_1[*idx[1]] = coeffs[1]
    q_dense_1 = Polynomial(coeffs_dense_1, basis=Basis.BERN)

    coeffs_dense_2 = np.zeros(deg[2] + 1)
    coeffs_dense_2[*idx[2]] = coeffs[2]
    coeffs_dense_2[*idx[3]] = coeffs[3]
    q_dense_2 = Polynomial(coeffs_dense_2, basis=Basis.BERN)


    x = np.random.rand(5, 3)

    print("Sparse: ", (p_sparse * q_sparse)(x))
    print("Dense: ", (p_dense_1(x) + p_dense_2(x)) * (q_dense_1(x) + q_dense_2(x)))