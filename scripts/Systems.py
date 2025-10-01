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
        self.cov = covariance

    def next_state(self, x : np.ndarray, v : np.ndarray):
        x1, x2 = x
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return np.array([x1_next, x2_next]) + v
    
    # Methods for matching GP model
    def predict(self, x : np.ndarray):
        x1, x2 = x.flatten()
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return np.array([x1_next, x2_next]), self.cov
    
    def jacobian(self, x : np.ndarray):
        x1, x2 = x.flatten()
        mu = self.mu
        dt = self.dt

        # Partial derivatives
        dfdx1 = np.array([
            [1,         dt],
            [dt * (-2 * mu * x1 * x2 - 1), 1 + dt * mu * (1 - x1**2)]
        ])

        return dfdx1


    def hessian_tensor(self, x : np.ndarray):
        x1, x2 = x.flatten()
        mu = self.mu
        dt = self.dt

        H = np.zeros((2, 2, 2))


        H[1, 0, 0] = dt * (-2 * mu * x2)
        H[1, 0, 1] = dt * (-2 * mu * x1)
        H[1, 1, 0] = dt * (-2 * mu * x1)

        return H

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

class DisturbedDubinsCar(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, track_heading_function, velocity = 1.0, noise_magnitude = 0.01, controller_gain = 1.0):
        """
        Dubins car with feedback controller and turn angle disturbance
        """

        def v_dist():
            return stats.multivariate_normal.rvs(mean=np.array([0.0, 0.0]), cov=noise_magnitude*np.eye(2))

        super().__init__(dim=3, v_dist=v_dist)

        self.dt = dt
        self.track_heading_function = track_heading_function
        self.velocity = velocity
        self.noise_magnitude = noise_magnitude
        self.controller_gain = controller_gain

    def next_state(self, x : np.ndarray, v : np.ndarray):
        x, y, theta = x
        v1, v2 = v

        dy_ref_dx = self.track_heading_function(x)
        reference_heading = np.arctan(dy_ref_dx) + v2
        heading_error = reference_heading - theta

        x_next = x + self.dt * self.velocity * np.cos(theta + v1)
        y_next = y + self.dt * self.velocity * np.sin(theta + v1)
        theta_next = theta + self.controller_gain * heading_error

        return np.array([x_next, y_next, theta_next])

class CartPole(DiscreteTimeStochasticSystem):
    def __init__(self, dt: float,
                 m_c: float = 1.0,   # cart mass
                 m_p: float = 0.1,   # pole mass
                 l: float = 0.5,     # half pole length
                 g: float = 9.81,
                 covariance: np.ndarray = 0.001*np.eye(4)):
        """
        Cart-pole system (open-loop, no control).
        State: [cart position, cart velocity, pole angle, pole angular velocity]

        Args:
            dt : time step
            m_c : cart mass
            m_p : pole mass
            l : half-length of the pole
            g : gravity
            covariance : 4x4 process noise covariance (additive Gaussian)
        """
        def additive_gaussian():
            return stats.multivariate_normal.rvs(mean=np.zeros(4), cov=covariance)

        super().__init__(dim=4, v_dist=additive_gaussian)

        self.dt, self.m_c, self.m_p, self.l, self.g = dt, m_c, m_p, l, g

    def next_state(self, x: np.ndarray, v: np.ndarray):
        p, p_dot, theta, theta_dot = x
        m_c, m_p, l, g = self.m_c, self.m_p, self.l, self.g

        # No control force (u = 0)
        u = 0.0

        # Equations of motion (continuous time)
        total_mass = m_c + m_p
        sin_th, cos_th = np.sin(theta), np.cos(theta)

        temp = (u + m_p * l * theta_dot**2 * sin_th) / total_mass
        theta_acc = (g * sin_th - cos_th * temp) / (l * (4.0/3.0 - (m_p * cos_th**2) / total_mass))
        p_acc = temp - (m_p * l * theta_acc * cos_th) / total_mass

        # Euler integration
        p_next      = p + self.dt * p_dot
        p_dot_next  = p_dot + self.dt * p_acc
        theta_next  = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_acc

        return np.array([p_next, p_dot_next, theta_next, theta_dot_next]) + v


class PlanarQuadrotor(DiscreteTimeStochasticSystem):
    def __init__(self, dt: float,
                 waypoint: np.ndarray = np.array([0.0, 0.0]),   # [px_ref, pz_ref]
                 m: float = 1.0, I: float = 0.02, ell: float = 0.2, g: float = 9.81,
                 c_v: float = 0.05, c_w: float = 0.02,
                 covariance: np.ndarray = 0.01*np.eye(6),
                 thrust_min: float = 0.0, thrust_max: float = 20.0):
        """
        Planar quadrotor with state-feedback waypoint tracking (6D autonomous system).
        State x = [px, pz, theta, vx, vz, omega]

        Controller drives (px, pz) -> waypoint using PD position control, pitch control from desired horizontal accel.
        Inputs are internal (no extra args to next_state), so the system is autonomous.
        """
        def additive_gaussian():
            return stats.multivariate_normal.rvs(mean=np.zeros(6), cov=covariance)

        super().__init__(dim=6, v_dist=additive_gaussian)

        self.dt = dt
        self.m, self.I, self.ell, self.g = m, I, ell, g
        self.c_v, self.c_w = c_v, c_w
        self.thrust_min, self.thrust_max = thrust_min, thrust_max

        # Fixed waypoint (parameter, not part of the state)
        self.waypoint = np.asarray(waypoint, dtype=float).reshape(2,)

        # PD gains (tune as needed)
        self.kp_pos = np.array([2.0, 2.0])   # [x, z]
        self.kd_pos = np.array([1.0, 1.0])
        self.kp_theta = 5.0
        self.kd_theta = 3.0

        # Convenience: hover thrust per rotor (not used directly but useful bound)
        self.u_hover = np.array([m*g/2, m*g/2])

    def set_waypoint(self, waypoint: np.ndarray):
        self.waypoint = np.asarray(waypoint, dtype=float).reshape(2,)

    # ---------- internal helpers ----------
    def _dynamics(self, x: np.ndarray, u: np.ndarray):
        px, pz, th, vx, vz, w = x
        u1, u2 = u
        T = u1 + u2
        tau = self.ell * (u2 - u1)

        dx = np.zeros(6)
        dx[0] = vx
        dx[1] = vz
        dx[2] = w
        dx[3] = -(T/self.m) * np.sin(th) - self.c_v * vx
        dx[4] =  (T/self.m) * np.cos(th) - self.g - self.c_v * vz
        dx[5] =  (tau/self.I) - self.c_w * w
        return dx

    def _state_feedback(self, x: np.ndarray):
        """
        State-feedback thrusts to move toward self.waypoint.
        Returns rotor thrusts u = [u1, u2] with saturation and non-negativity.
        """
        px, pz, th, vx, vz, w = x
        px_ref, pz_ref = self.waypoint

        # Position & velocity errors
        ex, ez = (px_ref - px), (pz_ref - pz)
        evx, evz = (-vx), (-vz)

        # Desired accelerations
        ax_des = self.kp_pos[0] * ex + self.kd_pos[0] * evx
        az_des = self.kp_pos[1] * ez + self.kd_pos[1] * evz + self.g  # add g so az_des = g at zero error

        # Desired pitch from horizontal accel (small-angle compatible, globally well-defined)
        theta_des = -np.arctan2(ax_des, az_des)

        # Inner-loop attitude control -> desired torque
        e_theta = theta_des - th
        e_w = -w
        tau_des = self.kp_theta * e_theta + self.kd_theta * e_w

        # Total thrust to realize resultant accel magnitude
        T_des = self.m * np.sqrt(ax_des**2 + az_des**2)

        # Map to rotor thrusts
        u1 = 0.5 * (T_des - tau_des / self.ell)
        u2 = 0.5 * (T_des + tau_des / self.ell)

        # Enforce actuator limits and non-negativity
        u1 = float(np.clip(u1, self.thrust_min, self.thrust_max))
        u2 = float(np.clip(u2, self.thrust_min, self.thrust_max))
        return np.array([u1, u2])

    # ---------- required by your framework ----------
    def next_state(self, x: np.ndarray, v: np.ndarray):
        """
        Autonomous closed-loop: x_{k+1} = f(x_k) + v_k
        """
        u = self._state_feedback(x)
        dx = self._dynamics(x, u)
        x_next = x + self.dt * dx   # Euler step (matches style of your other systems)
        return x_next + v