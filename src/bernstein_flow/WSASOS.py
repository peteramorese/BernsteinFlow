import numpy as np
from scipy.linalg import sqrtm, inv, eigh
from scipy.stats import multivariate_normal

from .GPGMM import GMModel, MultivariateGPModel, compute_mean_jacobian, compute_mean_hessian_tensor

class WSASOS:
    """
    Implementation of the `whitened spherical-average second-order stretching` propagation algorithm
    """
    def __init__(self, n_samples : int = 500):
        self.n_samples = n_samples 
        self.n_samples = n_samples

    def J2_action(J2: np.ndarray, u: np.ndarray):
        """
        Contract the second-derivative tensor J2 (shape m x n x n)
        with u twice to produce a vector of length m.
        """
        # J2[i, j, k] * u[j] * u[k]
        return np.tensordot(J2, np.outer(u, u), axes=([1, 2], [0, 1]))

    def wsasos_direction(self, mean: np.ndarray, cov: np.ndarray, J: np.ndarray, H: np.ndarray):
        """
        Compute the WSASOS splitting direction at a Gaussian mixand.
        mean:    n-dimensional mean
        cov:     n x n covariance
        J:       m x n Jacobian of g at mean
        H:       m x n x n second-derivative Hessian tensor of function at mean
        Returns an n-dimensional unit vector direction.
        """
        assert J.ndim == 2, "Jacobian must be 2D array"
        assert H.ndim == 3, "Second order tensor must be 3D array"

        # 1. Linearized output covariance and its inverse
        Pz = J @ cov @ J.T
        Pz_inv = inv(Pz)

        # 2. Input whitening factors
        P_half = sqrtm(cov)
        # ensure real
        P_half = np.real_if_close(P_half)

        # 3. Monte Carlo approximate Qw: an n x n matrix
        n = mean.shape[0]
        Qw = np.zeros((n, n))
        for _ in range(self.n_samples):
            # sample on unit sphere in whitened space
            y = np.random.normal(size=n)
            y /= np.linalg.norm(y)
            # map back to original input space
            u = P_half @ y
            # second-order action
            v = self.J2_action(H, u)
            # scalar weight: Mahalanobis of v in output
            weight = float(v.T @ Pz_inv @ v)
            # accumulate
            Qw += weight * np.outer(u, u)
        Qw /= self.n_samples

        # 4. principal eigenvector of Qw
        vals, vecs = eigh(Qw)
        # pick largest
        principal = vecs[:, np.argmax(vals)]
        # normalize
        return principal / np.linalg.norm(principal)


    def simple_univariate_split(self, var: float) -> tuple[list[float], list[float], list[float]]:
        """
        Simple symmetric univariate split into two mixands:
        - even weights, means at +/- delta, variances = var - delta^2.
        """
        w = [0.5, 0.5] # weight the mixands equally
        delta = np.sqrt(var) / 2
        mu1, mu2 = -delta, delta
        sigma2 = var - delta**2
        return w, [mu1, mu2], [sigma2, sigma2]


    def wsasos_split_and_propagate(self, belief : GMModel, transition_p : MultivariateGPModel):
        """
        Apply one WSASOS split to each mixand and propagate via linearization.
        Returns a new GMModel with 2*len(gm.means) components.
        """
        new_means = list()
        new_covs = list()
        new_weights = list()

        for mean, cov, weight in zip(belief.means, belief.covariances, belief.weights):
            mean = mean.reshape(1, -1)

            # 1. compute split direction
            J = compute_mean_jacobian(transition_p, mean)
            H = compute_mean_hessian_tensor(transition_p, mean)
            d = self.wsasos_direction(mean, cov, J, H)

            # 2. directional variance: reciprocal precision along d
            prec = float(d.T @ inv(cov) @ d)
            var_dir = 1 / prec

            # 3. univariate split along d
            w_u, mus_u, sig2_u = self.simple_univariate_split(var_dir)

            # 4. compute Pbar for mixand covariances
            sum_w_mmT = sum(wu * np.outer(mu_ui * d, mu_ui * d)
                            for wu, mu_ui in zip(w_u, mus_u))
            sum_w_sig2 = sum(wu * sig2_ui for wu, sig2_ui in zip(w_u, sig2_u))
            Pbar = (cov - sum_w_mmT) / sum_w_sig2

            # 5. assemble children mixands, propagate each (in this case only two children)
            for wu, mu_ui, sig2_ui in zip(w_u, mus_u, sig2_u):
                # new weight
                w_child = weight * wu
                # new mean
                child_mean = mean + mu_ui * d
                # new covariance
                Pi = sig2_ui * Pbar
                # propagate via linearization
                next_child_mean, pred_stds = transition_p.predict(child_mean)
                pred_cov = np.diag(pred_stds**2)

                J_child = compute_mean_jacobian(transition_p, child_mean)
                next_child_cov = J_child @ Pi @ J_child.T + pred_cov

                new_means.append(next_child_mean)
                new_covs.append(next_child_cov)
                new_weights.append(w_child)

        return GMModel(new_means, new_covs, new_weights)