import numpy as np
from math import comb
from functools import reduce
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.optimize import linprog
import scipy.sparse as sp

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

def bernstein_raised_degree_tf(pre_shape, post_shape, sparse=True):
    if len(pre_shape) != len(post_shape):
        raise ValueError("pre_shape and post_shape must have the same length d.")

    # For each axis k build the 1D raising matrix U_k of shape (m_k+1)x(n_k+1)
    U_matrices = []
    for axis, (n_plus1, m_plus1) in enumerate(zip(pre_shape, post_shape)):
        n = n_plus1 - 1
        m = m_plus1 - 1
        if m < n:
            raise ValueError(f"Axis {axis}: target degree m={m} must exceed n={n}.")
        elif m == n: 
            if sparse:
                U = sp.eye(n + 1, format='coo')
            else:
                U = np.eye(n + 1)
        else:
            if sparse:
                data, row_ind, col_ind = [], [], []
                r = m - n  # The degree increase
                for i in range(m + 1):
                    # Determine the range of j for non-zero entries: max(0, i-r) <= j <= min(i, n)
                    start_j = max(0, i - r)
                    end_j = min(i, n)
                    for j in range(start_j, end_j + 1):
                        val = comb(n, j) * comb(r, i - j) / comb(m, i)
                        data.append(val)
                        row_ind.append(i)
                        col_ind.append(j)
                U = sp.coo_matrix((data, (row_ind, col_ind)), shape=(m + 1, n + 1))
            else:
                U = np.zeros((m+1, n+1), dtype=float)
                for i in range(m+1):
                    # Only j up to min(i,n)
                    for j in range(min(i, n)+1):
                        # And i-j <= m-n automatically holds since i <= m
                        U[i, j] = comb(n, j) * comb(m-n, i-j) / comb(m, i)
        U_matrices.append(U)
    # Kronecker together: U1 ⊗ U2 ⊗ ... ⊗ Ud
    A = reduce(lambda X, Y: sp.kron(X, Y, format='csr'), U_matrices) if sparse else reduce(lambda X, Y: np.kron(X, Y), U_matrices) 
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

def plot_projected_basis_vectors(ax : plt.Axes, A, label_prefix=r'$\hat{e}_', color='blue', show_line=True, label=r'$e_i$'):
    """
    Projects standard basis vectors of R^m onto the column space of A (an m x 3 matrix),
    and plots the resulting 3D vectors in the coordinate system of the embedded hyperplane.

    Parameters:
    - ax: matplotlib 3D axis (Axes3D) to plot on
    - A: (m x 3) matrix with full column rank
    - label_prefix: optional LaTeX label prefix (default: '$\\hat{{e}}_')
    """
    m, n = A.shape
    if n != 3:
        raise ValueError("Matrix A must have 3 columns (embedding into 3D space).")

    # Step 1: Get orthonormal basis Q for col(A)
    Q, _ = qr(A, mode='economic')  # Q: m x 3

    # Step 2: Project each standard basis vector and convert to 3D coordinates
    for i in range(m):
        e_i = np.zeros(m)
        e_i[i] = 1
        c_i = Q.T @ e_i  # 3D coordinates in basis of col(A)

        if show_line:
            # Step 3: Plot line from origin to projected coordinate
            ax.plot([0, c_i[0]],
                    [0, c_i[1]],
                    [0, c_i[2]], color=color)
                    #label=f"{label_prefix}{i+1}$", color=color)

        # Plot endpoint
        ax.scatter(c_i[0], c_i[1], c_i[2], s=40, color=color, label=label if i == 0 else None)

    # Optional styling (can override outside this function)
    ax.set_xlabel("Basis 1")
    ax.set_ylabel("Basis 2")
    ax.set_zlabel("Basis 3")
    ax.grid(True)
    ax.set_aspect('equal')
    #ax.set_box_aspect([1,1,1])

def reduce_cone_basis(V : np.ndarray, tol=1e-8):
    assert V.ndim == 2
    n, N = V.shape

    selected = []
    for i in range(N):
        v = V[:, i]
        A_eq = V[:, [j for j in range(N) if j != i]]
        c = np.ones(N-1)
        res = linprog(c, A_eq=A_eq, b_eq=v, bounds=(0, None), method='highs')
        if not res.success:
            selected.append(i)
        elif np.any(res.x < tol):
            selected.append(i)

        if len(selected) == n:
            break

    return V[:, selected], selected

if __name__ == "__main__":

    #fig = plt.figure()
    #ax3d = fig.add_subplot(111, projection='3d')



    ## Show the embedded hyperplane
    #raised_deg = 3
    #tf = bernstein_raised_degree_tf((2,), (raised_deg,))
    #A = tf.A
    #assert A.shape == (3, 2)
    #b0 = np.linspace(-1, 1, 100)
    #b1 = np.linspace(-1, 1, 100)
    #B0, B1 = np.meshgrid(b0, b1)
    #coeffs2 = np.vstack([B0.ravel(), B1.ravel()])
    #print("coeffs2 shape: ", coeffs2.shape)
    #coeffs3 = (A @ coeffs2)
    #B0 = coeffs3[0, :].reshape(B0.shape)
    #B1 = coeffs3[1, :].reshape(B0.shape)
    #B2 = coeffs3[2, :].reshape(B0.shape)
    #ax3d.plot_surface(B0, B1, B2)
    #ax3d.set_xlabel("b0")
    #ax3d.set_ylabel("b1")
    #ax3d.set_zlabel("b2")
    #plt.show()


    original_shape = (100,)
    new_shape = (120,)

    A = bernstein_raised_degree_tf(original_shape, new_shape).A
    A_dense = bernstein_raised_degree_tf(original_shape, new_shape, sparse=False).A
    print("eq: ", np.allclose(A.toarray(), A_dense))



    ##M = np.eye(A.shape[0])
    #P = A @ np.linalg.inv(A.T @ A) @ A.T
    #n_vecs = 12
    #proj_curr = np.random.uniform(low=0, high=1, size=(P.shape[0], n_vecs))
    ##proj_curr = np.random.uniform(low=0, high=10, size=(P.shape[0], n_vecs))
    #for i in range(100):
    #    
    #    norm_rect = np.maximum(1e-1, proj_curr) / np.linalg.norm(proj_curr, axis=0)
    #    proj_curr = P @ norm_rect 
    #    with np.printoptions(precision=2, suppress=True):
    #        #print(np.min(proj_curr,axis=0))
    #        print("proj N curr: \n", proj_curr)
    #        n_vecs = np.linalg.pinv(A) @ proj_curr
    #        n_vecs_normalized = n_vecs / np.linalg.norm(n_vecs, axis=0)
    #        n_vecs_neg = np.sum(np.min(n_vecs_normalized, axis=0) < 0)
    #        n_vecs_reduced, _ = reduce_cone_basis(n_vecs_normalized)
    #        #print("proj n curr: \n", n_vecs_normalized)
    #        print("proj n curr minimal: \n", n_vecs_reduced)
    #        print("Min val: ", np.min(proj_curr), " neg vecs: ", n_vecs_neg)
    #        
    #        print("argmin co: ", np.argmin(np.min(proj_curr, axis=0)))
    #        min_col_idx = np.argmin(np.min(proj_curr, axis=0))
    #        proj_curr = np.delete(proj_curr, min_col_idx, axis=1)
    #    input("...")
    ##for i in range(A.shape[1]):

    #fig = plt.figure(figsize=(10, 7))
    #ax = fig.add_subplot(111, projection='3d')


    #raised_degs = [10, 12, 15, 20, 30, 50, 100, 1000, 10000]
    #for raised_deg in raised_degs:
    #    r = np.random.rand()  # Random float between 0 and 1 for red
    #    g = np.random.rand()  # Random float between 0 and 1 for green
    #    b = np.random.rand()  # Random float between 0 and 1 for blue
    #    color = (r, g, b)

    #    tf = bernstein_raised_degree_tf((3,), (raised_deg,))
    #    plot_projected_basis_vectors(ax, tf.A, color=color, label=f"m={raised_deg}")
    

    #ax.legend()

    ## Equal aspect ratio for clarity
    #plt.tight_layout()
    #plt.show()





    #fig, axes = plt.subplots(1, 2)

    #def plot_poly(p, ax, **kwargs):
    #    x = np.linspace(0, 1, 100)
    #    x = x.reshape((-1, 1))
    #    y = p(x)
    #    ax.plot(x, y, **kwargs)
    
    #def plot_poly_2D(p, ax, **kwargs):
    #    x = np.linspace(0, 1, 100)
    #    y = np.linspace(0, 1, 100)
    #    X, Y = np.meshgrid(x, y)
    #    Z = p(np.stack((X.ravel(), Y.ravel()), axis=-1)).reshape(X.shape)
    #    ax.contourf(X, Y, Z, **kwargs)

    p_bern = Polynomial(np.random.uniform(low=-10, high=10, size=(5, 6, 4, 3)), basis=Basis.BERN)

    ##plot_poly(p_bern, axes[0])

    new_shape = tuple(np.array(p_bern.shape()) + 4)
    tf = bernstein_raised_degree_tf(p_bern.shape(), new_shape)
    A = bernstein_raised_degree_tf(p_bern.shape(), new_shape).A
    A_dense = bernstein_raised_degree_tf(p_bern.shape(), new_shape, sparse=False).A
    print("eq: ", np.allclose(A.toarray(), A_dense))

    p_bern_raised = apply_transformation(p_bern, tf)

    x = np.random.rand(10, p_bern.dim())

    print("orig values: ", p_bern(x))
    print("raised values: ", p_bern_raised(x))

    ##p_bern_raised = apply_transformation(p_bern, tf)
    ##print("coeffs: ", p_bern.coeffs)
    ##print("coeffs raised: ", p_bern_raised.coeffs)

    ##plot_poly(p_bern_raised, axes[1])

    ##plt.show()

    #p_bern = Polynomial(np.array([[1,4,2], [3, 4, 5]]), basis=Basis.BERN)

    #plot_poly_2D(p_bern, axes[0])

    ##tf = bernstein_degree_raise_matrix(2, 5, p_bern.dim())
    #tf = bernstein_raised_degree_tf(p_bern.shape(), (5, 7))

    #p_bern_raised = apply_transformation(p_bern, tf)
    #print("coeffs: ", p_bern.coeffs)
    #print("coeffs raised: ", p_bern_raised.coeffs)

    #plot_poly_2D(p_bern_raised, axes[1])

    #plt.show()