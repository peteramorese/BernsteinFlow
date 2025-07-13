import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

def fit_gmm(X, n_components=1, covariance_type='full', random_state=None):
    X = np.asarray(X)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    gmm.fit(X)
    return gmm

def fit_gp(X, Xp, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=True):
    """
    Fit a Gaussian Process to learn the conditional distribution p(x' | x).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input states.
    Xp : array-like, shape (n_samples, n_features)
        Output targets.
    kernel : sklearn.gaussian_process.kernels.Kernel, default=None
        Kernel to use in the GP. If None, uses a default RBF+WhiteKernel.
    alpha : float, default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
    optimizer : str or optimizer callable, default='fmin_l_bfgs_b'
        Optimizer to use for kernel hyperparameter tuning.
    n_restarts_optimizer : int, default=5
        Number of times to restart the optimizer to find better hyperparameters.
    normalize_y : bool, default=True
        Whether to normalize the targets before fitting.
    """
    X = np.asarray(X)
    Xp = np.asarray(Xp)
    
    if kernel is None:
        # Default: constant*RBF + White noise
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
    
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        optimizer=optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=normalize_y
    )
    gpr.fit(X, Xp)
    return gpr

#if __name__ == "__main__":
#    # Simulated data
#    X0_samples = np.random.randn(100, 2)  # 100 initial states in 2D
#    X_samples = np.random.randn(200, 2)
#    X_prime_samples = X_samples * 0.5 + np.sin(X_samples) + 0.1 * np.random.randn(200, 2)
#    
#    # Estimate initial distribution
#    gmm = fit_gmm(X0_samples, n_components=3, random_state=42)
#    print("GMM means:", gmm.means_)
#    
#    # Fit transition GP
#    gp = fit_gp(X_samples, X_prime_samples)
#    print("Learned kernel:", gp.kernel_)
#    
#    # Predict next state for new point
#    x_new = np.array([[0.1, -0.2]])
#    x_pred, x_std = gp.predict(x_new, return_std=True)
#    print("Predicted next state:", x_pred, "Std dev:", x_std)