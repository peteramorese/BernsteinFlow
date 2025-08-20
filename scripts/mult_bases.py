import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import beta

def beta2d_pdf(x, y, a1, b1, a2, b2):
    """2D beta pdf = product of independent beta distributions."""
    return beta.pdf(x, a1, b1) * beta.pdf(y, a2, b2)

def plot_beta_slider(c1a_init, c1b_init, c2a_init, c2b_init, n):
    # Grid for evaluation
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)

    # Initial parameters
    i0 = 1
    def compute_Z(c1a, c1b, c2a, c2b, i):
        a1, b1 = c1a * i, c1b * i
        a2, b2 = c2a * i, c2b * i
        return beta2d_pdf(X, Y, a1, b1, a2, b2)

    Z = compute_Z(c1a_init, c1b_init, c2a_init, c2b_init, i0)

    # Plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.45)  # Make space for sliders
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax.set_title(f"2D Beta PDF (i={i0})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Slider axes
    ax_i   = plt.axes([0.25, 0.35, 0.65, 0.03])
    ax_c1a = plt.axes([0.25, 0.30, 0.65, 0.03])
    ax_c1b = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_c2a = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_c2b = plt.axes([0.25, 0.15, 0.65, 0.03])

    # Sliders
    slider_i   = Slider(ax_i,   'i',   1, n, valinit=i0, valstep=1)
    slider_c1a = Slider(ax_c1a, 'c1a', 0.1, 10.0, valinit=c1a_init)
    slider_c1b = Slider(ax_c1b, 'c1b', 0.1, 10.0, valinit=c1b_init)
    slider_c2a = Slider(ax_c2a, 'c2a', 0.1, 10.0, valinit=c2a_init)
    slider_c2b = Slider(ax_c2b, 'c2b', 0.1, 10.0, valinit=c2b_init)

    # Update function
    def update(val):
        i   = int(slider_i.val)
        c1a = slider_c1a.val
        c1b = slider_c1b.val
        c2a = slider_c2a.val
        c2b = slider_c2b.val

        Z = compute_Z(c1a, c1b, c2a, c2b, i)
        ax.clear()
        ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax.set_title(f"2D Beta PDF (i={i})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.canvas.draw_idle()

    # Connect sliders
    slider_i.on_changed(update)
    slider_c1a.on_changed(update)
    slider_c1b.on_changed(update)
    slider_c2a.on_changed(update)
    slider_c2b.on_changed(update)

    plt.show()

# Example usage:
plot_beta_slider(c1a_init=2, c1b_init=3, c2a_init=4, c2b_init=2, n=10)
