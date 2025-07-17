import numpy as np
from scipy.linalg import toeplitz

from bernstein_flow.Polynomial import Polynomial, poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct, marginal

def integrate_product_over_x(A: np.ndarray, b: np.ndarray, basis_integrals: np.ndarray):
    """
    Computes the y-coefficients of the polynomial resulting from:
        ∫ (p(y, x) * q(x)) dx
    where:
        p(y, x) = φ_y(y)^T @ A @ φ_x(x)
        q(x) = b @ φ_x(x)
    and basis_integrals contains ∫ φ^{(2)}_x(x) dx over [0, 1]

    Parameters:
    - A: (m, n) matrix of coefficients for p(y, x)
    - b: (n,) vector of coefficients for q(x)
    - basis_integrals: (2n - 1,) vector of ∫ φ_x^{(2)}(x) dx over [0,1]

    Returns:
    - y_coeffs: (m,) vector of resulting y polynomial coefficients
    """
    n = len(b)
    # Construct Toeplitz matrix T_b: shape (2n - 1, n)
    first_col = np.concatenate([b, np.zeros(n - 1)])
    first_row = np.concatenate([b[:1], np.zeros(n - 1)])
    T_b = toeplitz(first_col, first_row)

    # Multiply A with T_b.T to get the new coefficient matrix \tilde{A}
    A_tilde = A @ T_b.T  # shape: (m, 2n - 1)

    # Integrate over x by multiplying with the basis integral vector I_x
    y_coeffs = A_tilde @ basis_integrals  # shape: (m,)

    return y_coeffs

if __name__ == "__main__":
    n = 3
    m = 4
    A = np.random.rand(m, n)
    b = np.random.rand(n)

    # Integral of monomial basis from 0 to 1: [1, 1/2, ..., 1/(2n - 1)]
    I_x = 1 / np.arange(1, 2 * n)

    y_coeffs = integrate_product_over_x(A, b, I_x)
    print("Resulting y coefficients:", y_coeffs)

    p = Polynomial(A.T)
    q = Polynomial(b)

    prod = poly_product([p, q])
    marg = marginal(prod, dims=[0])
    print("True y coeffs: ", marg.coeffs)

