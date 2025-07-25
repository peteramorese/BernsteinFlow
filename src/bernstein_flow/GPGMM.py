import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.spatial import Rectangle

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
            covariances = [cov_diag for cov_diag in model.covariances_]
        elif len(covariances_shape) == 3: # Full
            covariances = [cov for cov in model.covariances_]
        else:
            assert False, "Not implemented"
        return cls(model.means_, covariances, model.weights_)
    
    def density(self, x : np.ndarray):
        dim = x.shape[1] if x.ndim == 2 else x.shape[0]
        return sum(w * multivariate_normal.pdf(x, mean.reshape((dim,)), cov) for mean, cov, w in zip(self.means, self.covariances, self.weights))

    def n_mixands(self):
        return len(self.means)
    
    def integrate(self, region : Rectangle):
        prob_mass = 0.0
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            assert len(cov.shape) == 1, "Covariance must be diagonal to integrate"
            component_integrals = norm.cdf(region.maxes, loc=mean, scale=np.sqrt(cov)) - norm.cdf(region.mins, loc=mean, scale=np.sqrt(cov))
            prob_mass += weight * np.prod(component_integrals)
        return prob_mass


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
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x : torch.Tensor):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

class MultitaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dtype = torch.float64):
        super().__init__(train_x, train_y, likelihood)
        self.dtype = dtype
        num_tasks = train_y.shape[1]

        # A mean for each task
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # An LCM kernel to model correlations between tasks
        self.covar_module = gpytorch.kernels.LCMKernel(
            base_kernels=[gpytorch.kernels.RBFKernel()],
            num_tasks=num_tasks,
            rank=1 # Rank determines the complexity of the correlation
        )

    def forward(self, x : torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultivariateGPModel:
    def __init__(self, gp : MultitaskGP, likelihood, dtype = torch.float64):
        self.gp = gp
        self.likelihood = likelihood
        self.dtype = dtype
    
    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            is_np = True
            x = torch.from_numpy(x).to(self.dtype)
        else:
            assert isinstance(x, torch.Tensor)
            is_np = False
            x = x.to(dtype=self.dtype)

        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(x))

            if is_np:
                return pred.mean.numpy(), pred.covariance_matrix.numpy()
            else:
                return pred.mean, pred.covariance_matrix
    
    def jacobian(self, x):
        self.gp.eval()
        self.likelihood.eval()

        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x)
        else:
            assert isinstance(x, torch.Tensor)

        x = x.flatten()

        x = x.to(dtype=self.dtype, device=next(self.gp.parameters()).device)
        x = torch.autograd.Variable(x, requires_grad=True)

        def mean_fcn(x_in : torch.Tensor):
            pred = self.likelihood(self.gp(x_in.reshape(1, -1)))
            return pred.mean.squeeze(0)

        J = torch.autograd.functional.jacobian(mean_fcn, x)
        return J.numpy() if is_np else J

    def hessian_tensor(self, x):
        """
        Compute the second-order derivative tensor (array of 2D hessians for each component) of the mean function about a given point
        """
        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x)
        else:
            assert isinstance(x, torch.Tensor)
        
        x = x.flatten()

        x = x.to(dtype=self.dtype, device=next(self.gp.parameters()).device)
        x = torch.autograd.Variable(x, requires_grad=True)

        hessian_layers = []
        for i in range(x.shape[0]):
            def mean_fcn(x_in : torch.Tensor):
                pred = self.likelihood(self.gp(x_in.reshape(1, -1)))
                return pred.mean[:,i].squeeze(0)
            H_i = torch.autograd.functional.hessian(mean_fcn, x)
            hessian_layers.append(H_i)

        H = torch.stack(hessian_layers)
        print(H.shape)
        return H.numpy() if is_np else H


def fit_gp(X : torch.Tensor, Xp : torch.Tensor, num_epochs=100, lr=0.1, device='cpu', dtype=torch.float64):
    dim = X.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=dim, rank=1).to(dtype=dtype)
    model = MultitaskGP(X, Xp, likelihood).to(dtype=dtype)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(num_epochs):
        start_time = time.time()
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Xp)
        loss.backward()
        optimizer.step()

        line = f"Epoch {epoch+1}: Loss = {loss:.6f}, time: {time.time() - start_time:.3f}"
        print(line)

    return MultivariateGPModel(model, likelihood)

import matplotlib.pyplot as plt
if __name__ == "__main__":
    n_dim = 2

    true_cov = np.array([[1.0, -0.2], [-0.2, 1.0]])
    true_dist = multivariate_normal(mean=np.zeros(2), cov=true_cov)

    N = 100
    train_x = torch.randn(N, n_dim)
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(train_x[:, 0], train_x[:, 1])

    train_y = torch.sin(4.0 * train_x) + torch.from_numpy(true_dist.rvs(size=N))

    mvgp = fit_gp(train_x, train_y)

    test_x = np.array([0.2, 0.3])
    mean, cov = mvgp.predict(test_x.reshape(1, -1))
    print("mean: ", mean, " cov: ", cov)

    y_ls = np.linspace(-3, 3, 100)
    print("true test mean: \n", np.sin(test_x), " true test cov: \n", true_cov)
    p_y_true = multivariate_normal(mean=np.sin(test_x), cov=true_cov)

    p_y_model = multivariate_normal(mean=mean.reshape((2,)), cov=cov)

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z_true = p_y_true.pdf(pos)
    Z_model = p_y_model.pdf(pos)

    # Plot the PDF using a contour plot
    fig, axes = plt.subplots(1, 2)
    axes[0].contourf(X, Y, Z_true, levels=20, cmap='viridis')
    axes[0].set_title('True')
    axes[0].grid(True)
    axes[1].contourf(X, Y, Z_model, levels=20, cmap='viridis')
    axes[1].set_title('Model')
    axes[1].grid(True)

    J = mvgp.jacobian(test_x.reshape(1, -1))
    H = mvgp.hessian_tensor(test_x.reshape(1, -1))
    print("Jacobian: \n", J)
    print("Hessian: \n", H)

    ## Optional: Plot as a 3D surface plot
    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    #ax.set_zlabel('Probability Density')
    #ax.set_title('3D Surface Plot of 2D Multivariate Normal PDF')
    plt.show()

    