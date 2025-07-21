import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import multivariate_normal

import torch
import gpytorch
import time

class GMModel:
    def __init__(self, means : list[np.ndarray], covariances : list[np.ndarray], weights : list[float]):
        self.means = means
        self.covariances = covariances
        self.weights = weights
    
    @classmethod
    def from_sklearn_gmm(cls, model : GaussianMixture):
        covariances_shape = model.covariances_.shape
        if len(covariances_shape) == 2: # Diag
            covariances = [np.diag(cov_diag) for cov_diag in model.covariances_]
        elif len(covariances_shape) == 3: # Full
            covariances = [cov for cov in model.covariances_]
        else:
            assert False, "Not implemented"
        return cls(model.means_, covariances, model.weights_)
    
    def density(self, x : np.ndarray):
        return sum(w * multivariate_normal.pdf(x, mean, cov) for mean, cov, w in zip(self.means, self.covariances, self.weights))

    def n_mixands(self):
        return len(self.means)

def fit_gmm(X, n_components=1, covariance_type='diag', random_state=None):
    X = np.asarray(X)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    gmm.fit(X)
    return GMModel.from_sklearn_gmm(gmm)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_xp, likelihood):
        super().__init__(train_x, train_xp, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x : torch.Tensor):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

class MultivariateGPModel:
    def __init__(self, component_models : list, likelihoods = list, dtype=torch.float32):
        assert len(component_models) == len(likelihoods)
        self.component_models = component_models 
        self.likelihoods = likelihoods 
        self.dim = len(component_models)
        self.dtype = dtype
    
    def predict(self, x):
        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x).to(dtype=self.dtype)
        else:
            assert isinstance(x, torch.Tensor)
            if x.dtype != self.dtype:
                x = x.to(dtype=self.dtype)

        means = []
        stds = []
        for model, likelihood in zip(self.component_models, self.likelihoods):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                xp_pred_dist = likelihood(model(x))
                means.append(xp_pred_dist.mean.item())
                stds.append(xp_pred_dist.stddev.item())
        if is_np:
            return np.array(means), np.array(stds)
        else:
            return torch.tensor(means), torch.tensor(stds)
    
    def density(self, x, xp):
        assert type(x) == type(xp)
        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x).to(dtype=self.dtype)
        else:
            assert isinstance(x, torch.Tensor)
        log_density = 0.0
        for i in range(self.dim):
            model = self.component_models[i]
            likelihood = self.likelihoods[i]
            with torch.no_grad():
                dist = likelihood(model(x))
                log_density += dist.log_prob(xp[i])
        return np.exp(log_density.item())
    
def fit_gp(X, Xp, num_iter=100, lr=0.1, device='cpu', dtype=torch.float32):
    """
    Fit a GP using GPyTorch for each output dimension of Xp.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features.
    Xp : array-like, shape (n_samples, n_targets)
        Target outputs.
    num_iter : int, default=100
        Number of training iterations.
    learning_rate : float, default=0.1
        Learning rate for optimizer.
    normalize_xp : bool, default=True
        Whether to normalize outputs.
    device : str, default='cpu'
        Device to run the model on.
        
    Returns
    -------
    models : list of trained GPyTorch models (one per output dimension)
    likelihoods : list of corresponding likelihood modules
    """  
    X = torch.tensor(X, dtype=dtype, device=device)
    Xp = torch.tensor(Xp, dtype=dtype, device=device)

    models = []
    likelihoods = []
    for d in range(Xp.shape[1]):
        xp = Xp[:, d]

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype=dtype)
        model = GPModel(X, xp, likelihood).to(device, dtype=dtype)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for epoch in range(num_iter):
            start_time = time.time()
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, xp)
            loss.backward()
            optimizer.step()

            line = f"Epoch {epoch+1}: Loss = {loss:.6f}, time: {time.time() - start_time:.3f}"
            print(line)

        models.append(model.eval())
        likelihoods.append(likelihood.eval())

    return MultivariateGPModel(models, likelihoods, dtype=dtype)

#def fit_gp(X, Xp, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=True):
#    """
#    Fit a Gaussian Process to learn the conditional distribution p(x' | x).
#    
#    Parameters
#    ----------
#    X : array-like, shape (n_samples, n_features)
#        Input states.
#    Xp : array-like, shape (n_samples, n_features)
#        Output targets.
#    kernel : sklearn.gaussian_process.kernels.Kernel, default=None
#        Kernel to use in the GP. If None, uses a default RBF+WhiteKernel.
#    alpha : float, default=1e-10
#        Value added to the diagonal of the kernel matrix during fitting.
#    optimizer : str or optimizer callable, default='fmin_l_bfgs_b'
#        Optimizer to use for kernel hyperparameter tuning.
#    n_restarts_optimizer : int, default=5
#        Number of times to restart the optimizer to find better hyperparameters.
#    normalize_y : bool, default=True
#        Whether to normalize the targets before fitting.
#    """
#    X = np.asarray(X)
#    Xp = np.asarray(Xp)
#    
#    if kernel is None:
#        # Default: constant*RBF + White noise
#        kernel = (
#            ConstantKernel(1.0, (1e-3, 1e3)) *
#            RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e3))
#            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
#        )
#    
#    gpr = GaussianProcessRegressor(
#        kernel=kernel,
#        alpha=alpha,
#        optimizer=optimizer,
#        n_restarts_optimizer=n_restarts_optimizer,
#        normalize_y=normalize_y
#    )
#    gpr.fit(X, Xp)
#    return gpr

def compute_mean_jacobian(model : MultivariateGPModel, x):
    """
    Compute the jacobian of the mean function about a given point
    """
    is_np = isinstance(x, np.ndarray)
    if is_np:
        x = torch.from_numpy(x).to(dtype=model.dtype)
    else:
        assert isinstance(x, torch.Tensor)

    x_grad = torch.autograd.Variable(x, requires_grad=True)
    jacobian = []
    for component_model in model.component_models:
        mean = component_model(x_grad).mean 
        grad = torch.autograd.grad(mean, x_grad, retain_graph=True)[0]
        jacobian.append(grad.squeeze(0))
    J = torch.stack(jacobian)
    return J.numpy() if is_np else J

def compute_mean_hessian_tensor(model : MultivariateGPModel, x):
    """
    Compute the second-order derivative tensor (array of 2D hessians for each component) of the mean function about a given point
    """
    is_np = isinstance(x, np.ndarray)
    if is_np:
        x = torch.from_numpy(x).to(dtype=model.dtype)
    else:
        assert isinstance(x, torch.Tensor)

    x_grad = torch.autograd.Variable(x, requires_grad=True)
    hessians = []

    for component_model in model.component_models:
        mean = component_model(x_grad).mean  # scalar-valued output

        # Compute gradient (first derivative)
        grad = torch.autograd.grad(mean, x_grad, create_graph=True)[0]  # shape: (input_dim,)

        # Compute second derivative (Hessian)
        hessian_rows = []
        for i in range(grad.shape[-1]):
            grad_i = grad[..., i]
            hess_row = torch.autograd.grad(grad_i, x_grad, retain_graph=True)[0].squeeze(0)
            hessian_rows.append(hess_row)
        hessian = torch.stack(hessian_rows, dim=0)
        hessians.append(hessian)

    H = torch.stack(hessians)  # shape: (output_dim, input_dim, input_dim)
    return H.numpy() if is_np else H