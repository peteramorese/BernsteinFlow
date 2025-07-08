import numpy as np
from math import comb
from functools import reduce
import matplotlib.pyplot as plt

from .Polynomial import Polynomial, Basis

class PolynomialTransformation:
    def __init__(self, A : np.ndarray, pre_shape : tuple[int, ...], post_shape : tuple[int, ...], pre_basis=None, post_basis=None):
        self.A = A
        self.dim = len(pre_shape)
        assert len(post_shape) == self.dim, "Pre and post shapes dimensions must match."
        assert A.shape == (self.vector_length(post_shape), self.vector_length(pre_shape)), "Transformation matrix A is inconsistent with pre and post shapes."
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        self.pre_basis = pre_basis if pre_basis is not None else Basis.BERN
        self.post_basis = post_basis if pre_basis is not None else Basis.BERN
    
    @staticmethod
    def vector_length(poly_shape : tuple[int, ...]):
        return np.prod(poly_shape)

def bernstein_degree_raise_matrix(n: int, m: int, d: int) -> np.ndarray:
    """
    Calculate the degree raising matrix for a Bernstein polynomial of degree n to degree m
    """
    if m <= n:
        raise ValueError("Raised degree m must be strictly greater than original degree n.")
    # Build 1D degree-raising matrix U of shape (m+1)x(n+1)
    U = np.zeros((m+1, n+1), dtype=float)
    for i in range(m+1):
        for j in range(n+1):
            # valid only if 0 <= j <= n and i-j <= m-n and i >= j
            if 0 <= j <= i and (i-j) <= (m-n):
                U[i, j] = comb(n, j) * comb(m-n, i-j) / comb(m, i)
    # Now form the d‑fold Kronecker product 
    # which has shape ((m+1)**d, (n+1)**d)
    A = reduce(lambda X, Y: np.kron(X, Y), [U]*d)
    return PolynomialTransformation(A, pre_shape=(n+1,)*d, post_shape=(m+1,)*d, pre_basis=Basis.BERN, post_basis=Basis.BERN)

def bernstein_raised_degree_tf(pre_shape, post_shape):
    if len(pre_shape) != len(post_shape):
        raise ValueError("pre_shape and post_shape must have the same length d.")
    # For each axis k build the 1D raising matrix U_k of shape (m_k+1)x(n_k+1)
    U_matrices = []
    for axis, (n_plus1, m_plus1) in enumerate(zip(pre_shape, post_shape)):
        n = n_plus1 - 1
        m = m_plus1 - 1
        if m <= n:
            raise ValueError(f"Axis {axis}: target degree m={m} must exceed n={n}.")
        U = np.zeros((m+1, n+1), dtype=float)
        for i in range(m+1):
            # Only j up to min(i,n)
            for j in range(min(i, n)+1):
                # And i-j <= m-n automatically holds since i <= m
                U[i, j] = comb(n, j) * comb(m-n, i-j) / comb(m, i)
        U_matrices.append(U)
    # Kronecker together: U1 ⊗ U2 ⊗ ... ⊗ Ud
    A = reduce(lambda X, Y: np.kron(X, Y), U_matrices)
    return PolynomialTransformation(A, pre_shape=pre_shape, post_shape=post_shape, pre_basis=Basis.BERN, post_basis=Basis.BERN)

def apply_transformation(p : Polynomial, tf : PolynomialTransformation):
    """
    Apply a linear matrix transformation the the coefficients of a polynomial
    """
    assert p.shape() == tf.pre_shape, f"Polynomial shape {p.shape()} is inconsistent with transformation pre-shape {tf.pre_shape}."
    assert p.basis() == tf.pre_basis, "Polynomial basis is inconsistent with transformation pre-basis."
    d = p.dim()
    p_coeffs = p.coeffs
    p_coeffs_v = p_coeffs.reshape((-1, 1))

    p_coeffs_v_new = tf.A @ p_coeffs_v

    p_coeffs_new = p_coeffs_v_new.reshape(tf.post_shape)
    return Polynomial(p_coeffs_new, basis=tf.post_basis)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2)

    def plot_poly(p, ax, **kwargs):
        x = np.linspace(0, 1, 100)
        x = x.reshape((-1, 1))
        y = p(x)
        ax.plot(x, y, **kwargs)
    
    def plot_poly_2D(p, ax, **kwargs):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = p(np.stack((X.ravel(), Y.ravel()), axis=-1)).reshape(X.shape)
        ax.contourf(X, Y, Z, **kwargs)

    #p_bern = Polynomial(np.array([1,4,2]), basis=Basis.BERN)

    #plot_poly(p_bern, axes[0])

    #tf = bernstein_degree_raise_matrix(2, 13, p_bern.dim())

    #p_bern_raised = apply_transformation(p_bern, tf)
    #print("coeffs: ", p_bern.coeffs)
    #print("coeffs raised: ", p_bern_raised.coeffs)

    #plot_poly(p_bern_raised, axes[1])

    #plt.show()

    p_bern = Polynomial(np.array([[1,4,2], [3, 4, 5]]), basis=Basis.BERN)

    plot_poly_2D(p_bern, axes[0])

    #tf = bernstein_degree_raise_matrix(2, 5, p_bern.dim())
    tf = bernstein_raised_degree_tf(p_bern.shape(), (5, 7))

    p_bern_raised = apply_transformation(p_bern, tf)
    print("coeffs: ", p_bern.coeffs)
    print("coeffs raised: ", p_bern_raised.coeffs)

    plot_poly_2D(p_bern_raised, axes[1])

    plt.show()