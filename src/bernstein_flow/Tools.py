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
        A ((k-1) * p) x (2n) array where each row is a pair (x_{k+1}, x_k).
    """
    k = len(trajectory_data)
    if k < 2:
        raise ValueError("Need at least two time steps to form transition pairs.")
    
    p, n = trajectory_data[0].shape

    data_matrix = np.empty(((k - 1) * p, 2 * n))

    for i in range(k - 1):
        x_k = trajectory_data[i]      # shape: (p, n)
        x_kp1 = trajectory_data[i+1]  # shape: (p, n)
        data_matrix[i * p : (i + 1) * p, :] = np.hstack((x_kp1, x_k))

    return data_matrix 