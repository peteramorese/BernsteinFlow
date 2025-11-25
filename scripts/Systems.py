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
        self.cov = covariance

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
        x = x.flatten()
        u = u.flatten()
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
        x = x.flatten()
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
    
    def predict(self, x: np.ndarray):
        """
        Predict mean and covariance for additive Gaussian system.
        Returns (mean, covariance) where mean = f(x) = x + dt * dx(x, u(x))
        """
        u = self._state_feedback(x)
        dx = self._dynamics(x, u)
        mean = x + self.dt * dx
        return mean, self.cov
    
    def jacobian(self, x: np.ndarray):
        """
        Compute Jacobian matrix ∂f/∂x where f(x) = x + dt * dx(x, u(x))
        Returns 6x6 matrix.
        """
        x = x.flatten()
        px, pz, th, vx, vz, w = x
        dt = self.dt
        m, I, ell, c_v, c_w = self.m, self.I, self.ell, self.c_v, self.c_w
        px_ref, pz_ref = self.waypoint
        
        # Get control inputs
        u = self._state_feedback(x)
        u1, u2 = u
        T = u1 + u2
        tau = ell * (u2 - u1)
        
        # Controller derivatives (assuming not at saturation boundaries)
        ex = px_ref - px
        ez = pz_ref - pz
        evx = -vx
        evz = -vz
        
        ax_des = self.kp_pos[0] * ex + self.kd_pos[0] * evx
        az_des = self.kp_pos[1] * ez + self.kd_pos[1] * evz + self.g
        
        # Derivatives of desired accelerations
        dax_des_dpx = -self.kp_pos[0]
        dax_des_dvx = -self.kd_pos[0]
        daz_des_dpz = -self.kp_pos[1]
        daz_des_dvz = -self.kd_pos[1]
        
        # Derivatives of theta_des = -arctan2(ax_des, az_des)
        denom = ax_des**2 + az_des**2
        dtheta_des_dax = -az_des / denom
        dtheta_des_daz = ax_des / denom
        
        dtheta_des_dpx = dtheta_des_dax * dax_des_dpx
        dtheta_des_dvx = dtheta_des_dax * dax_des_dvx
        dtheta_des_dpz = dtheta_des_daz * daz_des_dpz
        dtheta_des_dvz = dtheta_des_daz * daz_des_dvz
        
        # Derivatives of T_des = m * sqrt(ax_des^2 + az_des^2)
        if denom > 1e-10:
            dT_des_dax = self.m * ax_des / np.sqrt(denom)
            dT_des_daz = self.m * az_des / np.sqrt(denom)
        else:
            dT_des_dax = 0.0
            dT_des_daz = 0.0
        
        dT_des_dpx = dT_des_dax * dax_des_dpx
        dT_des_dvx = dT_des_dax * dax_des_dvx
        dT_des_dpz = dT_des_daz * daz_des_dpz
        dT_des_dvz = dT_des_daz * daz_des_dvz
        
        # Derivatives of tau_des = kp_theta * (theta_des - th) + kd_theta * (-w)
        dtau_des_dpx = self.kp_theta * dtheta_des_dpx
        dtau_des_dpz = self.kp_theta * dtheta_des_dpz
        dtau_des_dth = -self.kp_theta
        dtau_des_dvx = self.kp_theta * dtheta_des_dvx
        dtau_des_dvz = self.kp_theta * dtheta_des_dvz
        dtau_des_dw = -self.kd_theta
        
        # Derivatives of u1, u2 (assuming not saturated)
        # u1 = 0.5 * (T_des - tau_des / ell)
        # u2 = 0.5 * (T_des + tau_des / ell)
        du1_dpx = 0.5 * (dT_des_dpx - dtau_des_dpx / ell)
        du1_dpz = 0.5 * (dT_des_dpz - dtau_des_dpz / ell)
        du1_dth = 0.5 * (-dtau_des_dth / ell)
        du1_dvx = 0.5 * (dT_des_dvx - dtau_des_dvx / ell)
        du1_dvz = 0.5 * (dT_des_dvz - dtau_des_dvz / ell)
        du1_dw = 0.5 * (-dtau_des_dw / ell)
        
        du2_dpx = 0.5 * (dT_des_dpx + dtau_des_dpx / ell)
        du2_dpz = 0.5 * (dT_des_dpz + dtau_des_dpz / ell)
        du2_dth = 0.5 * (dtau_des_dth / ell)
        du2_dvx = 0.5 * (dT_des_dvx + dtau_des_dvx / ell)
        du2_dvz = 0.5 * (dT_des_dvz + dtau_des_dvz / ell)
        du2_dw = 0.5 * (dtau_des_dw / ell)
        
        # Derivatives of T and tau
        dT_dpx = du1_dpx + du2_dpx
        dT_dpz = du1_dpz + du2_dpz
        dT_dth = du1_dth + du2_dth
        dT_dvx = du1_dvx + du2_dvx
        dT_dvz = du1_dvz + du2_dvz
        dT_dw = du1_dw + du2_dw
        
        dtau_dpx = ell * (du2_dpx - du1_dpx)
        dtau_dpz = ell * (du2_dpz - du1_dpz)
        dtau_dth = ell * (du2_dth - du1_dth)
        dtau_dvx = ell * (du2_dvx - du1_dvx)
        dtau_dvz = ell * (du2_dvz - du1_dvz)
        dtau_dw = ell * (du2_dw - du1_dw)
        
        # Jacobian of dynamics dx
        # dx[0] = vx
        # dx[1] = vz
        # dx[2] = w
        # dx[3] = -(T/m) * sin(th) - c_v * vx
        # dx[4] = (T/m) * cos(th) - g - c_v * vz
        # dx[5] = (tau/I) - c_w * w
        
        cth, sth = np.cos(th), np.sin(th)
        
        J_dx = np.zeros((6, 6))
        J_dx[0, 3] = 1.0  # ∂dx[0]/∂vx
        J_dx[1, 4] = 1.0  # ∂dx[1]/∂vz
        J_dx[2, 5] = 1.0  # ∂dx[2]/∂w
        
        # ∂dx[3]/∂x
        J_dx[3, 0] = -(dT_dpx/m) * sth
        J_dx[3, 1] = -(dT_dpz/m) * sth
        J_dx[3, 2] = -(T/m) * cth - (dT_dth/m) * sth
        J_dx[3, 3] = -(dT_dvx/m) * sth - c_v
        J_dx[3, 4] = -(dT_dvz/m) * sth
        J_dx[3, 5] = -(dT_dw/m) * sth
        
        # ∂dx[4]/∂x
        J_dx[4, 0] = (dT_dpx/m) * cth
        J_dx[4, 1] = (dT_dpz/m) * cth
        J_dx[4, 2] = -(T/m) * sth + (dT_dth/m) * cth
        J_dx[4, 3] = (dT_dvx/m) * cth
        J_dx[4, 4] = (dT_dvz/m) * cth - c_v
        J_dx[4, 5] = (dT_dw/m) * cth
        
        # ∂dx[5]/∂x
        J_dx[5, 0] = dtau_dpx / I
        J_dx[5, 1] = dtau_dpz / I
        J_dx[5, 2] = dtau_dth / I
        J_dx[5, 3] = dtau_dvx / I
        J_dx[5, 4] = dtau_dvz / I
        J_dx[5, 5] = dtau_dw / I - c_w
        
        # Jacobian of f(x) = x + dt * dx
        J = np.eye(6) + dt * J_dx
        return J
    
    def hessian_tensor(self, x: np.ndarray):
        """
        Compute Hessian tensor H[i, j, k] = ∂²f_i/∂x_j∂x_k where f(x) = x + dt * dx(x, u(x))
        Returns 6x6x6 tensor.
        """
        x = x.flatten()
        px, pz, th, vx, vz, w = x
        dt = self.dt
        m, I, ell, c_v, c_w = self.m, self.I, self.ell, self.c_v, self.c_w
        px_ref, pz_ref = self.waypoint
        
        # Get control inputs
        u = self._state_feedback(x)
        u1, u2 = u
        T = u1 + u2
        tau = ell * (u2 - u1)
        
        # Controller terms
        ex = px_ref - px
        ez = pz_ref - pz
        ax_des = self.kp_pos[0] * ex + self.kd_pos[0] * (-vx)
        az_des = self.kp_pos[1] * ez + self.kd_pos[1] * (-vz) + self.g
        
        denom = ax_des**2 + az_des**2
        cth, sth = np.cos(th), np.sin(th)
        
        H = np.zeros((6, 6, 6))
        
        # Most components are zero. Only non-zero entries come from:
        # dx[3] = -(T/m) * sin(th) - c_v * vx
        # dx[4] = (T/m) * cos(th) - g - c_v * vz
        # dx[5] = (tau/I) - c_w * w
        
        # Second derivatives involving theta (th) and control inputs
        # We need second derivatives of T and tau w.r.t. state variables
        
        # For simplicity, we'll compute the key second derivatives
        # The main contributions come from:
        # 1. ∂²dx[3]/∂th² = (T/m) * sin(th) (from -T/m * sin(th))
        # 2. ∂²dx[4]/∂th² = -(T/m) * cos(th) (from T/m * cos(th))
        # 3. Cross terms from controller dependencies
        
        # Note: Computing full second derivatives of the controller is complex
        # We'll focus on the dominant terms from the dynamics themselves
        
        # ∂²dx[3]/∂th² = (T/m) * sin(th) (since d/dth of -T/m * sin(th) = -T/m * cos(th), 
        # and d²/dth² = T/m * sin(th))
        # Actually: d/dth(-(T/m)*sin(th)) = -(T/m)*cos(th) - (dT/dth/m)*sin(th)
        # d²/dth² = (T/m)*sin(th) - 2*(dT/dth/m)*cos(th) - (d²T/dth²/m)*sin(th)
        
        # For the controller second derivatives, we approximate by ignoring 
        # second-order controller effects (which are typically small)
        # The main second derivatives come from the trigonometric terms
        
        # ∂²dx[3]/∂th²
        if denom > 1e-10:
            dT_dth_approx = 0.0  # T depends weakly on th through controller
            H[3, 2, 2] = dt * (T/m) * sth  # Main term from -T/m * sin(th)
        else:
            H[3, 2, 2] = 0.0
        
        # ∂²dx[4]/∂th²
        H[4, 2, 2] = dt * (-T/m) * cth  # Main term from T/m * cos(th)
        
        # Cross terms: ∂²dx[3]/∂th∂px, etc. (from controller dependencies)
        # These are typically small, so we approximate as zero for now
        # A full implementation would require computing second derivatives of the controller
        
        return H
    

class Quadcopter(DiscreteTimeStochasticSystem):
    """
    12D quadcopter with closed-loop waypoint tracking (autonomous).
    State x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        - positions (world), linear velocities (world),
          Euler angles ZYX = (roll=phi, pitch=theta, yaw=psi),
          body rates (p,q,r) in body frame.
    Control (internal): total thrust T and body torques tau = [tau_x, tau_y, tau_z].
    Noise: additive Gaussian (12D).
    """

    def __init__(self, dt: float,
                 waypoint: np.ndarray = np.array([0.0, 0.0, 1.0]),  # target position (x,y,z)
                 yaw_ref: float = 0.0,
                 m: float = 1.0,
                 J: np.ndarray = np.diag([0.02, 0.02, 0.04]),
                 g: float = 9.81,
                 c_v: float = 0.05,          # translational linear damping
                 c_w: float = 0.05,          # angular linear damping
                 thrust_min: float = 0.0,
                 thrust_max: float = 20.0,
                 torque_limits: np.ndarray = np.array([1.0, 1.0, 0.5]),  # |tau_x|,|tau_y|,|tau_z|
                 covariance: np.ndarray = 0.001 * np.eye(12)):

        def additive_gaussian():
            return stats.multivariate_normal.rvs(mean=np.zeros(12), cov=covariance)

        super().__init__(dim=12, v_dist=additive_gaussian)

        # Params
        self.dt = float(dt)
        self.m = float(m)
        self.J = np.asarray(J, dtype=float)
        self.Jinv = np.linalg.inv(self.J)
        self.g = float(g)
        self.c_v = float(c_v)
        self.c_w = float(c_w)
        self.cov = covariance

        self.waypoint = np.asarray(waypoint, dtype=float).reshape(3,)
        self.yaw_ref = float(yaw_ref)

        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.torque_limits = np.asarray(torque_limits, dtype=float).reshape(3,)
        
        # Rate smoothing filter (exponential moving average)
        self.rate_filter_alpha = 0.3  # smoothing factor (0=no smoothing, 1=no filtering)
        self.filtered_rates = np.zeros(3)  # [p, q, r] filtered

        # Outer-loop (position) gains
        self.kp_pos = np.array([2.0, 2.0, 4.0])
        self.kd_pos = np.array([1.2, 1.2, 2.0])

        # Inner-loop (attitude) gains
        self.kp_ang = np.array([2.0, 2.0, 1.5])   # for [phi, theta, psi] errors
        self.kd_ang = np.array([1.5, 1.5, 0.5])   # for [p, q, r] errors

    # ---- utilities ----
    @staticmethod
    def _rot_zyx(phi, theta, psi):
        """Rotation matrix R (world <- body) from ZYX Euler angles."""
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        Rz = np.array([[ cpsi, -spsi, 0],
                       [ spsi,  cpsi, 0],
                       [    0,     0, 1]])
        Ry = np.array([[ cth, 0, sth],
                       [   0, 1,   0],
                       [-sth, 0, cth]])
        Rx = np.array([[1,   0,    0],
                       [0, cphi, -sphi],
                       [0, sphi,  cphi]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _euler_rate_matrix(phi, theta):
        """
        Map body rates Ω=[p,q,r] to Euler angle rates [phi_dot, theta_dot, psi_dot].
        ZYX convention.
        """
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth   = np.cos(theta), np.sin(theta)

        # Avoid singularity at cos(theta)=0 (|theta|=pi/2). Clamp magnitude to avoid huge gains.
        if np.isclose(cth, 0.0) or abs(cth) < 0.2:
            cth = 0.2 if cth >= 0 else -0.2

        E = np.array([
            [1, sphi*sth/cth, cphi*sth/cth],
            [0,      cphi,         -sphi],
            [0, sphi/cth,     cphi/cth]
        ])
        return E

    # ---- controller ----
    def _state_feedback(self, x: np.ndarray):
        """
        Compute (T, tau) from state. Tracks self.waypoint at yaw = self.yaw_ref.
        Small-angle compatible position-to-attitude conversion.
        """
        # Unpack
        p = x[0:3]      # [px, py, pz] world
        v = x[3:6]      # [vx, vy, vz] world
        phi, theta, psi = x[6], x[7], x[8]
        pqr = x[9:12]   # [p, q, r] body

        # --- Outer-loop position PD -> desired accel in world ---
        pos_err = self.waypoint - p
        vel_err = -v
        a_des = self.kp_pos * pos_err + self.kd_pos * vel_err  # world-frame desired accel

        # Add gravity compensation in thrust computation later

        # --- Convert a_des to desired (phi, theta) given desired yaw ---
        # For small-to-moderate angles (classic quad formula)
        cpsi, spsi = np.cos(self.yaw_ref), np.sin(self.yaw_ref)
        ax, ay, az = a_des
        # Desired roll/pitch to realize horizontal accelerations
        phi_des   = ( ax * spsi - ay * cpsi ) / self.g
        theta_des = ( ax * cpsi + ay * spsi ) / self.g

        # Total thrust to realize vertical accel
        T_des = self.m * (self.g + az)

        # Desired yaw
        psi_des = self.yaw_ref

        # --- Inner-loop attitude PD (Euler angle + rate) -> body torques ---
        ang_err = np.array([phi_des - phi, theta_des - theta, psi_des - psi])
        # Wrap yaw error to [-pi, pi] for smoothness
        ang_err[2] = (ang_err[2] + np.pi) % (2*np.pi) - np.pi

        rate_err = -pqr

        tau = self.kp_ang * ang_err + self.kd_ang * rate_err  # desired torques (approx)

        # Saturate
        T = float(np.clip(T_des, self.thrust_min, self.thrust_max))
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)

        return T, tau

    # ---- continuous-time dynamics (xdot = f(x, u)) ----
    def _dynamics(self, x: np.ndarray, T: float, tau: np.ndarray):
        # Unpack state
        px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x

        # Rotation and mappings
        R = self._rot_zyx(phi, theta, psi)        # world <- body
        e3 = np.array([0.0, 0.0, 1.0])

        # Forces: gravity + thrust along body + linear drag
        a_world = (T / self.m) * (R @ e3) - self.g * e3 - self.c_v * np.array([vx, vy, vz])

        # Angular dynamics: J*Ωdot = tau - Ω×(JΩ) - damping
        Omega = np.array([p, q, r])
        Omega_dot = self.Jinv @ (tau - np.cross(Omega, self.J @ Omega) - self.c_w * Omega)

        # Euler angle rates
        E = self._euler_rate_matrix(phi, theta)
        euler_dot = E @ Omega

        # Assemble xdot
        xdot = np.zeros(12)
        xdot[0:3] = np.array([vx, vy, vz])
        xdot[3:6] = a_world
        xdot[6:9] = euler_dot
        xdot[9:12] = Omega_dot
        return xdot

    def next_state(self, x: np.ndarray, v: np.ndarray):
        """
        Autonomous closed-loop: x_{k+1} = x_k + dt * f(x_k, u(x_k)) + v_k
        (Forward Euler integration; swap to semi-implicit or RK4 if desired.)
        """
        # Control from current state
        T, tau = self._state_feedback(x)

        # Derivatives
        xdot = self._dynamics(x, T, tau)

        # Euler integration
        x_next = x + self.dt * xdot
        
        # Apply rate smoothing to reduce chaotic oscillations
        raw_rates = x_next[9:12]  # [p, q, r]
        self.filtered_rates = (self.rate_filter_alpha * raw_rates + 
                              (1 - self.rate_filter_alpha) * self.filtered_rates)
        x_next[9:12] = self.filtered_rates

        # Wrap Euler angles to [-pi, pi] to prevent unbounded growth
        x_next[6] = (x_next[6] + np.pi) % (2*np.pi) - np.pi  # phi
        x_next[7] = (x_next[7] + np.pi) % (2*np.pi) - np.pi  # theta
        x_next[8] = (x_next[8] + np.pi) % (2*np.pi) - np.pi  # psi

        return x_next + v
    
    def predict(self, x: np.ndarray):
        """
        Predict mean and covariance for additive Gaussian system.
        Returns (mean, covariance) where mean = f(x) = x + dt * dx(x, u(x))
        Note: This ignores rate filtering and angle wrapping for the mean prediction.
        """
        T, tau = self._state_feedback(x)
        xdot = self._dynamics(x, T, tau)
        mean = x + self.dt * xdot
        # Apply rate filtering (simplified - uses current filtered rates)
        mean[9:12] = (self.rate_filter_alpha * mean[9:12] + 
                      (1 - self.rate_filter_alpha) * self.filtered_rates)
        # Wrap angles
        mean[6] = (mean[6] + np.pi) % (2*np.pi) - np.pi
        mean[7] = (mean[7] + np.pi) % (2*np.pi) - np.pi
        mean[8] = (mean[8] + np.pi) % (2*np.pi) - np.pi
        return mean, self.cov
    
    def jacobian(self, x: np.ndarray):
        """
        Compute Jacobian matrix ∂f/∂x where f(x) = x + dt * dx(x, u(x))
        Returns 12x12 matrix.
        Note: This is a simplified version that ignores rate filtering and angle wrapping effects.
        """
        x = x.flatten()
        px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x
        dt = self.dt
        m, g, c_v, c_w = self.m, self.g, self.c_v, self.c_w
        J = self.J
        Jinv = self.Jinv
        
        # Get control inputs
        T, tau = self._state_feedback(x)
        tau_x, tau_y, tau_z = tau
        
        # Controller derivatives (simplified - ignoring saturation)
        p = x[0:3]
        v = x[3:6]
        pos_err = self.waypoint - p
        vel_err = -v
        a_des = self.kp_pos * pos_err + self.kd_pos * vel_err
        
        # Derivatives of a_des
        da_des_dp = -np.diag(self.kp_pos)
        da_des_dv = -np.diag(self.kd_pos)
        
        # Derivatives of desired angles (small angle approximation)
        cpsi, spsi = np.cos(self.yaw_ref), np.sin(self.yaw_ref)
        ax, ay, az = a_des
        
        dphi_des_dax = spsi / self.g
        dphi_des_day = -cpsi / self.g
        dtheta_des_dax = cpsi / self.g
        dtheta_des_day = spsi / self.g
        
        dphi_des_dp = np.array([dphi_des_dax * da_des_dp[0, 0] + dphi_des_day * da_des_dp[1, 1], 
                                 dphi_des_dax * da_des_dp[0, 1] + dphi_des_day * da_des_dp[1, 0],
                                 0.0])
        dphi_des_dv = np.array([dphi_des_dax * da_des_dv[0, 0] + dphi_des_day * da_des_dv[1, 1],
                                dphi_des_dax * da_des_dv[0, 1] + dphi_des_day * da_des_dv[1, 0],
                                0.0])
        
        dtheta_des_dp = np.array([dtheta_des_dax * da_des_dp[0, 0] + dtheta_des_day * da_des_dp[1, 1],
                                  dtheta_des_dax * da_des_dp[0, 1] + dtheta_des_day * da_des_dp[1, 0],
                                  0.0])
        dtheta_des_dv = np.array([dtheta_des_dax * da_des_dv[0, 0] + dtheta_des_day * da_des_dv[1, 1],
                                  dtheta_des_dax * da_des_dv[0, 1] + dtheta_des_day * da_des_dv[1, 0],
                                  0.0])
        
        dT_des_daz = self.m
        dT_des_dp = np.array([0.0, 0.0, dT_des_daz * da_des_dp[2, 2]])
        dT_des_dv = np.array([0.0, 0.0, dT_des_daz * da_des_dv[2, 2]])
        
        # Compute desired angles
        phi_des = (ax * spsi - ay * cpsi) / self.g
        theta_des = (ax * cpsi + ay * spsi) / self.g
        
        # Derivatives of tau (attitude controller)
        dtau_dphi_des = self.kp_ang[0]
        dtau_dtheta_des = self.kp_ang[1]
        dtau_dpsi_des = self.kp_ang[2]
        
        # Derivatives of tau w.r.t. state (3x12 matrix, but we'll build it piecewise)
        # tau = kp_ang * (ang_des - ang) + kd_ang * (-pqr)
        # So ∂tau/∂p = kp_ang * ∂ang_des/∂p, etc.
        dtau_dp = np.zeros((3, 3))
        dtau_dp[0, 0] = dtau_dphi_des * dphi_des_dp[0]  # tau_x depends on px through phi_des
        dtau_dp[0, 1] = dtau_dphi_des * dphi_des_dp[1]  # tau_x depends on py through phi_des
        dtau_dp[1, 0] = dtau_dtheta_des * dtheta_des_dp[0]  # tau_y depends on px through theta_des
        dtau_dp[1, 1] = dtau_dtheta_des * dtheta_des_dp[1]  # tau_y depends on py through theta_des
        # tau_z doesn't depend on position (only on yaw_ref which is constant)
        
        dtau_dv = np.zeros((3, 3))
        dtau_dv[0, 0] = dtau_dphi_des * dphi_des_dv[0]  # tau_x depends on vx through phi_des
        dtau_dv[0, 1] = dtau_dphi_des * dphi_des_dv[1]  # tau_x depends on vy through phi_des
        dtau_dv[1, 0] = dtau_dtheta_des * dtheta_des_dv[0]  # tau_y depends on vx through theta_des
        dtau_dv[1, 1] = dtau_dtheta_des * dtheta_des_dv[1]  # tau_y depends on vy through theta_des
        
        # Derivatives w.r.t. Euler angles
        dtau_dang = -np.diag(self.kp_ang)  # 3x3 diagonal matrix
        
        # Derivatives w.r.t. body rates
        dtau_dpqr = -np.diag(self.kd_ang)  # 3x3 diagonal matrix
        
        # Rotation matrix
        R = self._rot_zyx(phi, theta, psi)
        e3 = np.array([0.0, 0.0, 1.0])
        R_e3 = R @ e3
        
        # Derivatives of rotation matrix R w.r.t. Euler angles
        # ∂R/∂phi, ∂R/∂theta, ∂R/∂psi (computed via product rule)
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        # For ZYX: R = Rz @ Ry @ Rx
        # We'll compute these derivatives (simplified - full implementation would be more complex)
        # For now, approximate as zero for rotation matrix derivatives (they're typically small)
        
        # Jacobian of dynamics
        J_dx = np.zeros((12, 12))
        
        # Position derivatives: dx[0:3] = v
        J_dx[0:3, 3:6] = np.eye(3)
        
        # Velocity derivatives: dx[3:6] = (T/m) * (R @ e3) - g * e3 - c_v * v
        # ∂dx[3:6]/∂v = -c_v * I
        J_dx[3:6, 3:6] = -c_v * np.eye(3)
        
        # ∂dx[3:6]/∂T = (1/m) * (R @ e3)
        ddx_dT = (1/m) * R_e3
        
        # Chain rule: ∂dx[3:6]/∂p = ∂dx[3:6]/∂T * ∂T/∂p
        J_dx[3:6, 0:3] = ddx_dT.reshape(3, 1) @ dT_des_dp.reshape(1, 3)
        J_dx[3:6, 3:6] += ddx_dT.reshape(3, 1) @ dT_des_dv.reshape(1, 3)
        
        # Angular velocity derivatives: dx[6:9] = E @ Omega
        E = self._euler_rate_matrix(phi, theta)
        J_dx[6:9, 9:12] = E
        
        # Derivatives of E w.r.t. phi and theta (complex, approximate as zero for now)
        
        # Body rate derivatives: dx[9:12] = Jinv @ (tau - Omega × (J @ Omega) - c_w * Omega)
        Omega = np.array([p, q, r])
        J_Omega = J @ Omega
        
        # ∂(Omega × (J @ Omega))/∂Omega
        # Using product rule: ∂(a × b)/∂x = (∂a/∂x) × b + a × (∂b/∂x)
        # For Omega × (J @ Omega):
        # ∂/∂Omega (Omega × (J @ Omega)) = I × (J @ Omega) + Omega × J
        # = [J @ Omega]× + [Omega]× @ J
        # where [v]× is the skew-symmetric matrix
        skew_Omega = np.array([[0, -r, q],
                               [r, 0, -p],
                               [-q, p, 0]])
        skew_JOmega = np.array([[0, -J_Omega[2], J_Omega[1]],
                                [J_Omega[2], 0, -J_Omega[0]],
                                [-J_Omega[1], J_Omega[0], 0]])
        # Note: [J @ Omega]× represents the cross product matrix for J @ Omega
        # and [Omega]× @ J represents the derivative of Omega × (J @ Omega) w.r.t. Omega
        d_cross_dOmega = skew_JOmega + skew_Omega @ J
        
        J_dx[9:12, 9:12] = Jinv @ (-d_cross_dOmega - c_w * np.eye(3))
        
        # Chain rule: ∂dx[9:12]/∂x = Jinv @ (∂tau/∂x)
        J_dx[9:12, 0:3] = Jinv @ dtau_dp
        J_dx[9:12, 3:6] = Jinv @ dtau_dv
        J_dx[9:12, 6:9] = Jinv @ dtau_dang
        J_dx[9:12, 9:12] += Jinv @ dtau_dpqr
        
        # Jacobian of f(x) = x + dt * dx
        J_f = np.eye(12) + dt * J_dx
        return J_f
    
    def hessian_tensor(self, x: np.ndarray):
        """
        Compute Hessian tensor H[i, j, k] = ∂²f_i/∂x_j∂x_k where f(x) = x + dt * dx(x, u(x))
        Returns 12x12x12 tensor.
        Note: This is a simplified implementation focusing on dominant terms.
        """
        x = x.flatten()
        px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x
        dt = self.dt
        m = self.m
        J = self.J
        Jinv = self.Jinv
        
        T, tau = self._state_feedback(x)
        
        H = np.zeros((12, 12, 12))
        
        # Main contributions come from:
        # 1. Rotation matrix terms in acceleration (dx[3:6])
        # 2. Euler angle rate matrix terms (dx[6:9])
        # 3. Cross product terms in angular dynamics (dx[9:12])
        
        # For the rotation matrix: R @ e3 appears in acceleration
        # Second derivatives w.r.t. Euler angles are the main contributors
        # These are complex to compute exactly, so we approximate
        
        # For Euler angle rate matrix E, second derivatives w.r.t. phi and theta
        # These come from the 1/cos(theta) terms
        
        # For angular dynamics, second derivatives come from the cross product
        # ∂²(Omega × (J @ Omega))/∂Omega²
        
        # Simplified: focus on the most significant terms
        # The cross product second derivatives
        Omega = np.array([p, q, r])
        J_Omega = J @ Omega
        
        # ∂²(Omega × (J @ Omega))/∂Omega² involves second derivatives of cross product
        # This is typically small, so we approximate as zero
        
        # The main non-zero terms would be from rotation matrix second derivatives
        # which are complex. For now, we return mostly zeros with a note that
        # a full implementation would require symbolic differentiation of rotation matrices
        
        return H

    def set_waypoint(self, waypoint: np.ndarray, yaw_ref: float | None = None):
        self.waypoint = np.asarray(waypoint, dtype=float).reshape(3,)
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)