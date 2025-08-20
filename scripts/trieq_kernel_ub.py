import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List

# ---------------- Kernel Class ----------------
class Kernel:
    def __init__(self, h: Callable[[float], float], center: np.ndarray, name: str = ""):
        """
        h: callable, radial kernel function h(r) (monotonically decreasing in r)
        center: np.ndarray, center point c_i
        name: optional label
        """
        self.h = h
        self.center = np.array(center)
        self.name = name if name else "kernel"

# ---------------- Density & Bound ----------------
def sum_density(x: np.ndarray, kernels: List[Kernel]) -> float:
    """Compute the exact density sum S(x) = sum h_i(||x - c_i||)."""
    total = 0.0
    for k in kernels:
        r = np.linalg.norm(x - k.center)
        total += k.h(r)
    return total

def triangle_bound(rho: float, kernels: List[Kernel], g: np.ndarray) -> float:
    """Compute the triangle inequality upper bound U(rho)."""
    total = 0.0
    for k in kernels:
        R_i = np.linalg.norm(k.center - g)
        total += k.h(abs(R_i - rho))
    return total

# ---------------- Visualization ----------------
def plot_density_field(kernels: List[Kernel], bounds=[-3, 3, -3, 3], resolution=100):
    """Plot the 2D kernel sum as a 3D surface."""
    x_vals = np.linspace(bounds[0], bounds[1], resolution)
    y_vals = np.linspace(bounds[2], bounds[3], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = sum_density(point, kernels)

    # 3D Surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.set_title("2D Kernel Density Field (Sum of Kernels)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Density")

    # Mark kernel centers
    for k in kernels:
        ax.scatter(k.center[0], k.center[1], sum_density(k.center, kernels),
                   color="red", s=50, label=k.name)
    ax.legend()
    plt.show()

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Example: define some radial decreasing kernels
    kernels = [
        Kernel(lambda r: np.exp(-r**2), center=[0,0], name="Gaussian_0"),
        Kernel(lambda r: 1/(1+r), center=[2,0], name="Rational_1"),
        Kernel(lambda r: np.exp(-r), center=[1,1], name="Exponential_2"),
    ]

    # Reference point g (choose centroid of centers)
    g = np.mean([k.center for k in kernels], axis=0)

    # --- Test bound ---
    rhos = np.linspace(0, 5, 200)
    U_vals = [triangle_bound(rho, kernels, g) for rho in rhos]

    plt.figure(figsize=(8,5))
    plt.plot(rhos, U_vals, label="Triangle Inequality Bound U(rho)")
    plt.xlabel("rho")
    plt.ylabel("Upper Bound")
    plt.title("Triangle Inequality Bound vs. rho")
    plt.legend()
    plt.grid(True)
    #plt.show()

    # --- 3D plot of density field ---
    plot_density_field(kernels, bounds=[-2,4,-2,4], resolution=100)
