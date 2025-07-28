import numpy as np
import scipy.stats as stats

class GaussianDistTransform:
    def __init__(self, means : np.ndarray, variances : np.ndarray):
        self.means = means
        self.variances = variances

    @classmethod
    def moment_match_data(cls, data : np.ndarray, variance_pads=None):
        """
        Compute the "best" GDT assuming the data follows independent gaussian trends

        Args:
            data : np.ndarray
            variance_pads : list of values to add to the variances if provided
        """
        means = np.mean(data, axis=0)
        variances = np.var(data, axis=0)
        if variance_pads is not None:
            variances += np.array(variance_pads)
        return cls(means, variances)

    def x_to_u(self, x, dim_mask : np.ndarray = None):
        if dim_mask is None:
            return stats.norm.cdf(x, loc=self.means, scale=np.sqrt(self.variances))
        else:
            return stats.norm.cdf(x, loc=self.means[dim_mask], scale=np.sqrt(self.variances[dim_mask]))

    def u_to_x(self, u, dim_mask : np.ndarray = None):
        if dim_mask is None:
            return stats.norm.ppf(u, loc=self.means, scale=np.sqrt(self.variances))
        else:
            return stats.norm.ppf(u, loc=self.means[dim_mask], scale=np.sqrt(self.variances[dim_mask]))
    
    def X_to_U(self, X): # TODO dim mask
        return np.apply_along_axis(self.x_to_u, 1, X)

    def U_to_X(self, U):
        return np.apply_along_axis(self.u_to_x, 1, U)

    def x_density(self, x, u_density_fcn, dim_mask : np.ndarray = None):
        u = self.x_to_u(x, dim_mask) 
        u_density = u_density_fcn(u)
        if dim_mask is None:
            component_multipliers = stats.norm.pdf(x, loc=self.means, scale=np.sqrt(self.variances))
            volume_change = np.prod(component_multipliers, axis=1) if len(self.means) > 1 else component_multipliers
        else:
            component_multipliers = stats.norm.pdf(x, loc=self.means[dim_mask], scale=np.sqrt(self.variances[dim_mask]))
            volume_change = np.prod(component_multipliers, axis=1) if len(self.means[dim_mask]) > 1 else component_multipliers
        return u_density * volume_change
    
    def u_density(self, u, x_density_fcn, dim_mask : np.ndarray = None):
        x = self.u_to_x(u, dim_mask)
        x_density = x_density_fcn(x)
        if dim_mask is None:
            component_multipliers = stats.norm.pdf(x, loc=self.means, scale=np.sqrt(self.variances))
            volume_change = np.prod(component_multipliers, axis=1) if len(self.means) > 1 else component_multipliers
        else:
            component_multipliers = stats.norm.pdf(x, loc=self.means[dim_mask], scale=np.sqrt(self.variances[dim_mask]))
            volume_change = np.prod(component_multipliers, axis=1) if len(self.means[dim_mask]) > 1 else component_multipliers
        return x_density / volume_change

class GammaDistTransform:
    def __init__(self, means : np.ndarray, variances : np.ndarray, a : np.ndarray):
        self.means = means
        self.variances = variances
        self.a = a

    #@classmethod
    #def moment_match_data(cls, data : np.ndarray, variance_pads=None):
    #    """
    #    Compute the "best" GDT assuming the data follows independent gaussian trends

    #    Args:
    #        data : np.ndarray
    #        variance_pads : list of values to add to the variances if provided
    #    """
    #    means = np.mean(data, axis=0)
    #    variances = np.var(data, axis=0)
    #    if variance_pads is not None:
    #        variances += np.array(variance_pads)
    #    return cls(means, variances)

    def x_to_u(self, x):
        return stats.gamma.cdf(x, self.a, loc=self.means, scale=np.sqrt(self.variances))

    def u_to_x(self, u):
        return stats.gamma.ppf(u, self.a, loc=self.means, scale=np.sqrt(self.variances))
    
    def X_to_U(self, X):
        return np.apply_along_axis(self.x_to_u, 1, X)

    def U_to_X(self, U):
        return np.apply_along_axis(self.u_to_x, 1, U)

    def x_density(self, x, u_density_fcn):
        u = self.x_to_u(x)
        u_density = u_density_fcn(u)
        component_multipliers = stats.gamma.pdf(x, self.a, loc=self.means, scale=np.sqrt(self.variances))
        volume_change = np.prod(component_multipliers, axis=1) if len(self.means) > 1 else component_multipliers
        return u_density * volume_change
    
    def u_density(self, u, x_density_fcn):
        x = self.u_to_x(u)
        x_density = x_density_fcn(x)
        component_multipliers = stats.gamma.pdf(x, self.a, loc=self.means, scale=np.sqrt(self.variances))
        volume_change = np.prod(component_multipliers, axis=1) if len(self.means) > 1 else component_multipliers
        return x_density / volume_change