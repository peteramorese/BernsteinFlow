import numpy as np
import matplotlib.pyplot as plt

from bernstein_flow.Polynomial import Polynomial, Basis, poly_eval, bernstein_to_monomial, poly_product, poly_product_bernstein_direct, marginal

def plot_poly(ax, poly):
    x = np.linspace(0, 1, 300).reshape(-1, 1)
    z = poly(x)
    ax.plot(x, z)
    return ax

if __name__ == "__main__":
    ax = plt.gca()
    
    weights = np.linspace(0, 1, 10)
    for i in range(len(weights)):
        bern_coeffs = np.zeros(50)
        bern_coeffs[3] = weights[i]
        bern_coeffs[4] = 1.0 - weights[i]

        p = Polynomial(bern_coeffs, basis=Basis.BERN)    



        plot_poly(ax, p)
    plt.show()
    