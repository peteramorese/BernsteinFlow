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
    def __init__(self, dt: float, mu: float = 1.0, covariance: np.ndarray = np.eye(2)):
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

    def next_state(self, x: np.ndarray, v: np.ndarray):
        x1, x2 = x
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return np.array([x1_next, x2_next]) + v