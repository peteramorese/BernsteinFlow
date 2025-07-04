import numpy as np
import torch

def create_transition_data_matrix(trajectory_data):
    """
    Create a data matrix of (x_{k+1}, x_k) pairs from trajectory data.

    Parameters:
    -----------
    trajectory_data : list of np.ndarray
        A list of length k, where each element is a (p x n) array.
        Each array contains p samples from the n-dimensional state distribution at time step k.

    Returns:
    --------
    data_matrix : np.ndarray
        A ((k-1) * p) x (2n) array where each row is a pair (x_k, x_{k+1}).
    """
    k = len(trajectory_data)
    if k < 2:
        raise ValueError("Need at least two time steps to form transition pairs.")
    
    p, n = trajectory_data[0].shape

    data_matrix = np.empty(((k - 1) * p, 2 * n))

    for i in range(k - 1):
        x_k = trajectory_data[i]      # shape: (p, n)
        x_kp1 = trajectory_data[i+1]  # shape: (p, n)
        data_matrix[i * p : (i + 1) * p, :] = np.hstack((x_k, x_kp1))

    return data_matrix 

def grid_eval(f, bounds : list, resolution=100, device=None, dtype=torch.float32):
    """
    Evaluate a function f: R^d -> R over a rectangular domain in R^d on a grid.

    Args:
        f: Callable that takes a torch tensor of shape (N, d) and returns (N,)
        bounds: List of length 2d, e.g. [x0_min, x0_max, x1_min, x1_max, ..., xd_min, xd_max]
        resolution: Number of points along each axis.
        device: Device to run the function on.

    Returns:
        meshgrids: A tuple of d meshgrid arrays (numpy)
        Z: Evaluated values reshaped to (resolution,) * d
    """
    d = len(bounds) // 2
    assert len(bounds) == 2 * d

    axes = [np.linspace(bounds[2*i], bounds[2*i + 1], resolution) for i in range(d)]
    meshgrids = np.meshgrid(*axes, indexing="ij")

    grid_points = np.stack([mg.ravel() for mg in meshgrids], axis=-1)
    grid_tensor = torch.tensor(grid_points, dtype=dtype, device=device)

    with torch.no_grad():
        values = f(grid_tensor).cpu().numpy()

    Z = values.reshape([resolution] * d)
    return *meshgrids, Z

def model_u_eval_fcn(model):
    def f(x):
        model.eval()
        return model(x).squeeze(-1)
    return f

def model_x_eval_fcn(model, dt, device=None, dtype=torch.float32):
    """
    Wraps a u-space model for evaluating in x, given a distribution transform (dt)
    """
    if device is None:
        device = next(model.parameters()).device

    def f(x):
        def u_density(u):
            u = torch.tensor(u, dtype=dtype, device=device)
            return model(u).detach().cpu().numpy()

        x_density_np = dt.x_density(x, u_density)
        return torch.from_numpy(x_density_np).to(device=device)

    return f  