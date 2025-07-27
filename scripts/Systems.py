import numpy as np
import scipy.stats as stats
from abc import ABC, abstractmethod

class DiscreteTimeStochasticSystem(ABC):
    def __init__(self, dim : int, v_dist = None):
        self._dim = dim

        if v_dist is not None:
            self._v_dist = v_dist
        else:
            def uniform():
                return np.random.uniform(size=self._dim)
            self._v_dist = stats.uniform()

    @abstractmethod
    def next_state(self, x : np.ndarray, v : np.ndarray):
        """
        Args:
            x : current state
            v : realization of the noise parameters
        """
        pass

    def __call__(self, x : np.ndarray):
        v = self._v_dist() # Sample a random v to be plugged into the difference function
        return self.next_state(x, v)

    def dim(self):
        return self._dim

def sample_trajectories(system : DiscreteTimeStochasticSystem, initial_state_sampler, n_timesteps : int, n_trajectories : int):
    """
    Sample trajectory data from a system under an initial state distribution

    Args:
        system : system model
        initial_state_sampler : callable (no args) that returns a randomly sampled initial state
        n_timesteps : time horizon of each trajectory
        n_trajectories : number of trajectories to sample

    Returns:
        traj_data (list) : list of length n_timesteps marginal data sets indexed by time step
    """
    dim = system.dim()
    traj_data = [np.zeros((n_trajectories, dim)) for _ in range(n_timesteps)]

    # Sample initial conditions
    for i in range(n_trajectories):
        traj_data[0][i, :] = initial_state_sampler()

    for k in range(n_timesteps - 1):
        for i in range(n_trajectories):
            xk = traj_data[k][i, :]
            xkp1 = system(xk)
            traj_data[k + 1][i, :] = xkp1

    return traj_data

def sample_io_pairs(system : DiscreteTimeStochasticSystem, n_pairs : int, region_lowers : list[float], region_uppers : list[float]):
    """
    Sample input (x', x) pairs from a system model, where the starting state x is sampled uniformly from a specified region
    """
    x_data = np.random.uniform(low=region_lowers, high=region_uppers, size=(n_pairs, system.dim()))
    xp_data = np.zeros_like(x_data)
    for i in range(n_pairs):
        xp_data[i, :] = system(x_data[i, :])
    
    return np.hstack((xp_data, x_data))



class CubicMap(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, alpha : float = 1.0, variance = 0.1):
        def additive_gaussian():
            return stats.norm(loc=0.0, scale=variance).rvs()
        
        super().__init__(dim=1, v_dist=additive_gaussian)

        self.dt = dt
        self.alpha = alpha
        self.variance  = variance

    def next_state(self, x : np.ndarray, v : np.ndarray):
        x_next = x - self.dt * self.alpha * x**3
        return x_next + v
    
    def transition_likelihood(self, x : np.ndarray, x_next : np.ndarray):
        mean = self.next_state(x, np.zeros_like(x))
        likelihood = stats.norm.pdf(x_next, loc=mean, scale=self.variance)
        return likelihood

class Pendulum(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, length : float = 1.0, damp : float = 0.1, covariance : np.ndarray = np.eye(2)):
        """
        Pendulum with additive Gaussian noise

        Args:
            dt : time step
            length : length of the pendulum
            damp : velocity damping coefficient
            covariance: 2x2 covariance matrix for process noise
        """

        def additive_gaussian():
            return stats.multivariate_normal.rvs(mean = np.zeros(2), cov=covariance)

        super().__init__(dim=2, v_dist=additive_gaussian)

        self.dt = dt
        self.g = 9.81
        self.l = length
        self.d = damp

    def next_state(self, x : np.ndarray, v : np.ndarray):
        theta, theta_dot = x
        theta_ddot = - (self.g / self.l) * np.sin(theta) - self.d * theta_dot
        
        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_ddot
        
        return np.array([theta_next, theta_dot_next]) + v

class VanDerPol(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, mu : float = 1.0, covariance : np.ndarray = np.eye(2)):
        """
        Van der Pol oscillator with additive Gaussian noise.
        
        Args:
            dt : time step
            mu : nonlinearity parameter
            covariance : 2x2 covariance matrix for process noise
        """
        def additive_gaussian():
            return stats.multivariate_normal.rvs(mean=np.zeros(2), cov=covariance)

        super().__init__(dim=2, v_dist=additive_gaussian)

        self.dt = dt
        self.mu = mu

    def next_state(self, x : np.ndarray, v : np.ndarray):
        x1, x2 = x
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return np.array([x1_next, x2_next]) + v

class VanDerPolMN(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, mu : float = 1.0, covariance : np.ndarray = np.eye(2)):
        """
        Van der Pol oscillator with multiplicative noise.
        
        Args:
            dt : time step
            mu : nonlinearity parameter
            covariance : 2x2 covariance matrix for process noise
        """
        def noise():
            return stats.multivariate_normal.rvs(mean=np.ones(2), cov=covariance)
            #return stats.beta.rvs(a=2, b=2, loc=0.5, scale=scale, size=2)

        super().__init__(dim=2, v_dist=noise)

        self.dt = dt
        self.mu = mu

    def next_state(self, x : np.ndarray, v : np.ndarray):
        x1, x2 = x
        dx1 = x2
        v1, v2 = v
        dx2 = (self.mu*v1) * (1 - x1**2) * x2 - v2*x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return np.array([x1_next, x2_next])
    
class LotkaVolterra(DiscreteTimeStochasticSystem):
    def __init__(self, dt: float, alpha = 1.0, beta = 0.1, delta = 0.075, gamma = 1.5, covariance : np.ndarray = np.eye(2), alpha_scale = 0.1):
        """
        LotkaVolterra population dynamics with multiplicative noise

        Args:
            dt : time step
            alpha : prey birth rate
            beta : predation rate
            delta : predator reproduction per prey consumed
            gamma : predator death rate
            covariance : 3D covariance of 1) multiplicative noise for prey, 2) multiplicative noise for predators, and 3) noise in the prey birth rate
        """
        def v_dist():
            v_pop = stats.multivariate_normal.rvs(mean=np.zeros(2), cov=covariance)
            return v_pop
            #v_alpha = stats.beta.rvs(a=2, b=5, loc=0.7, scale=np.sqrt(alpha_scale))
            #return np.append(v_pop, v_alpha)
        
        super().__init__(dim=2, v_dist=v_dist)

        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
    
    def next_state(self, x : np.ndarray, v : np.ndarray):
        x1, x2 = x
        v1, v2 = v

        x1_next = x1 + self.dt * ((1.0 * self.alpha)* x1 - self.beta * x1 * x2) + v1 * x1
        x2_next = x2 + self.dt * (self.delta * x1 * x2 - self.gamma * x2) + v2 * x2

        return np.array([x1_next, x2_next])
    
class BistableOscillator(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, a = 1.0, b = 1.0, c = 0.5, d = 1.0, e = 1.0, f = 0.5, cov_scale=0.01):

        n_components = 2
        means = [np.array([0.0, 0.0]), 0.5 * np.array([1.0, 1.0])]
        #means = [np.array([0.5, 0.5]), 2.0 * np.array([1.2, 1.2])]
        covariances = [cov_scale * np.array([[1.0, 0.2], [0.2, 1.0]]), cov_scale * np.array([[1.0, -0.2], [-0.2, 1.0]])]
        def v_dist():
            component = np.random.choice(n_components, size=1, p=[0.6, 0.4])[0]
            return stats.multivariate_normal.rvs(mean=means[component], cov=covariances[component])
        
        super().__init__(dim=2, v_dist=v_dist)

        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
    
    def next_state(self, x : np.ndarray, v : np.ndarray):
        x1, x2 = x
        v1, v2 = v

        x1_next = x1 + self.dt * (self.a * x1 - self.b * x1**3 - self.c * x2) + x1 * v1
        x2_next = x2 + self.dt * (self.d * x2 - self.e * x2**3 - self.f * x1) + x2 * v2

        return np.array([x1_next, x2_next])
