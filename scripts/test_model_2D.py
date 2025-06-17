from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, train

from .TestDataSets import sample_modal_gaussian, plot_data
from .Visualization import create_interactive_transformer_plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons, make_circles

def evaluate_u_density_on_grid(model, resolution=100, device=None):
    """
    Evaluate a trained model on a 2D grid over [0, 1]^2.

    Args:
        model: A trained PyTorch model that maps 2D inputs to scalar densities.
        resolution: Number of points along each axis (default: 100).
        device: Device to run the model on. If None, uses model's device.

    Returns:
        U0, U1`: Meshgrid coordinates (numpy arrays)
        Z: Density values on the grid (numpy array of shape [resolution, resolution])
    """
    if device is None:
        device = next(model.parameters()).device

    # Create 2D grid over [0, 1] x [0, 1]
    u0 = np.linspace(0, 1, resolution)
    u1 = np.linspace(0, 1, resolution)
    U0, U1 = np.meshgrid(u0, u1)

    # Flatten grid and convert to tensor
    grid_points = np.stack([U0.ravel(), U1.ravel()], axis=-1)  # shape: (resolution^2, 2)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # Evaluate model
    with torch.no_grad():
        model.eval()
        densities = model(grid_tensor).cpu().numpy()  # shape: (resolution^2,)

    Z = densities.reshape(resolution, resolution)
    return U0, U1, Z

def evaluate_x_density_on_grid(model, gdt: GaussianDistTransform, bounds : list, resolution=100, device=None):
    """
    Evaluate a trained model on a 2D grid over R^n

    Args:
        model: A trained PyTorch model that maps 2D inputs to scalar densities.
        resolution: Number of points along each axis (default: 100).
        device: Device to run the model on. If None, uses model's device.

    Returns:
        X0, X1: Meshgrid coordinates (numpy arrays)
        Z: Density values on the grid (numpy array of shape [resolution, resolution])
    """
    if device is None:
        device = next(model.parameters()).device

    # Create 2D grid over [0, 1] x [0, 1]
    x0 = np.linspace(bounds[0], bounds[1], resolution)
    x1 = np.linspace(bounds[2], bounds[3], resolution)
    X0, X1 = np.meshgrid(x0, x1)

    # Flatten grid and convert to tensor
    x_grid_points = np.stack([X0.ravel(), X1.ravel()], axis=-1)  # shape: (resolution^2, 2)
    x_grid_tensor = torch.tensor(x_grid_points, dtype=torch.float32, device=device)

    x_grid_tensor.dtype

    # Evaluate model
    def u_density(u : np.ndarray):
        u = torch.from_numpy(u)
        u = u.to(dtype=x_grid_tensor.dtype)
        with torch.no_grad():
            model.eval()
            densities = model(u).cpu().numpy()  # shape: (resolution^2,)
            return densities
        
    x_densities = gdt.x_density(x_grid_tensor, u_density)

    Z = x_densities.reshape(resolution, resolution)
    return X0, X1, Z

def plot_density_surface(ax, X0, X1, Z):
    # Create surface plot
    surf = ax.plot_surface(X0, X1, Z, cmap='viridis', linewidth=0, antialiased=True)
    return ax

def plot_density(ax : plt.Axes, X0, X1, Z):
    ax.contourf(X0, X1, Z, levels=50, cmap='viridis')


if __name__ == "__main__":

    # Dimension
    dim = 2

    # Number of data points
    n_data = 5000

    # Number of training epochs
    n_epochs = 1000

    #gdt = GaussianDistTransform(mean=dim*[0.5], variances=[1.0, 0.5])
    gdt = GaussianDistTransform(mean=[0.5, 0.25], variances=[1.0, 0.5])

    means = [[-1.5, -1.5], [1.5, 1.5]]
    covariances = [torch.eye(dim)*1.5, torch.eye(dim)*0.5]
    X_data = sample_modal_gaussian(n_data, means=means, covariances=covariances, weights=[.3, .7])

    X_data, _ = make_moons(n_data, noise=0.05)
    #X_data, _ = make_circles(n_data, noise=0.1, factor=0.4)


    fig, axes = plt.subplots(2, 2)
    fig.set_figheight(9)
    fig.set_figwidth(9)
    for ax in axes.flat:
        ax.set_aspect('equal')

    plot_data(axes[0, 0], X_data)
    axes[0, 0].set_xlabel("x0")
    axes[0, 0].set_ylabel("x1")
    axes[0, 0].set_title("Data")

    U_data = gdt.X_to_U(X_data)

    plot_data(axes[0, 1], U_data)
    axes[0, 1].set_xlim((0, 1))
    axes[0, 1].set_ylim((0, 1))
    axes[0, 1].set_xlabel("u0")
    axes[0, 1].set_ylabel("u1")
    axes[0, 1].set_title("Erf-space Data")

    plt.show(block=False)
    input("Continue to training...")

    # Create data loader
    U_data_torch = torch.tensor(U_data, dtype=torch.float32)
    dataset = TensorDataset(U_data_torch)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create model
    transformer_degrees = [100, 50]
    conditioner_degrees = [20, 70]
    model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees)

    print("Number of parameters in model: ", model.n_parameters())

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, dataloader, optimizer, epochs=n_epochs)

    # Plot the density estimate
    bounds = axes[0, 0].get_xlim() + axes[0, 0].get_ylim()
    X0, X1, Z_x = evaluate_x_density_on_grid(model, gdt, bounds, resolution=100)
    plot_density(axes[1, 0], X0, X1, Z_x)
    axes[1, 0].set_xlabel("x0")
    axes[1, 0].set_ylabel("x1")
    axes[1, 0].set_title("Feature-space PDF")

    U0, U1, Z_u = evaluate_u_density_on_grid(model, resolution=100)
    plot_density(axes[1, 1], U0, U1, Z_u)
    axes[1, 1].set_xlabel("u0")
    axes[1, 1].set_ylabel("u1")
    axes[1, 1].set_title("Erf-space PDF")


    fig2 = plt.figure()
    ax3d_x = fig2.add_subplot(121, projection='3d')
    plot_density_surface(ax3d_x, X0, X1, Z_x)
    ax3d_x.set_xlabel("x0")
    ax3d_x.set_ylabel("x1")
    ax3d_x.set_zlabel("p(x)")
    ax3d_x.set_title("Feature-space PDF")

    ax3d_u = fig2.add_subplot(122, projection='3d')
    plot_density_surface(ax3d_u, X0, X1, Z_u)
    ax3d_u.set_xlabel("u0")
    ax3d_u.set_ylabel("u1")
    ax3d_u.set_zlabel("p(u)")
    ax3d_u.set_title("Erf-space PDF")

    # Plot transformers
    fig3, axes, sliders = create_interactive_transformer_plot(model, dim)

    #print("UNCONSTRAINED Model params:")
    #for i in range(dim):
    #    print(model.A[i])

    #print("Model params:")
    #for i in range(dim):
    #    print(model.get_constrained_parameters(i))


    plt.show()
