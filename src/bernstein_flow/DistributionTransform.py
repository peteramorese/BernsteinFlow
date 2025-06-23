import numpy as np
import scipy.stats as stats

class GaussianDistTransform:
    def __init__(self, means : np.ndarray, variances : np.ndarray):
        self.means = means
        self.variances = variances

    @classmethod
    def moment_match_data(cls, data : np.ndarray):
        """
        Compute the "best" GDT assuming the data follows independent gaussian trends
        """
        means = np.mean(data, axis=0)
        variances = np.var(data, axis=0)
        return cls(means, variances)

    def x_to_u(self, x):
        return stats.norm.cdf(x, loc=self.means, scale=np.sqrt(self.variances))

    def u_to_x(self, u):
        return stats.norm.ppf(u, loc=self.means, scale=np.sqrt(self.variances))
    
    def X_to_U(self, X):
        return np.apply_along_axis(self.x_to_u, 1, X)

    def U_to_X(self, U):
        return np.apply_along_axis(self.u_to_x, 1, U)

    def x_density(self, x, u_density_fcn):
        u = self.x_to_u(x)
        u_density = u_density_fcn(u)

        component_multipliers = stats.norm.pdf(x, loc=self.means, scale=np.sqrt(self.variances))
        volume_change = np.prod(component_multipliers, axis=1)
        return u_density * volume_change
