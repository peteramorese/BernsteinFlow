import numpy as np

from .Polynomial import Polynomial


class FactorizedPolynomial:
    def __init__(self, factors : list[Polynomial]):
        self.basis = factors[0].basis()
        self.stable = True

        self.dim = 1
        for factor in factors[1:]:
            assert factor.basis() == self.basis, "Not all factors are in the same basis"
            if factor.dim() > self.dim:
                self.dim = factor.dim()

        self.factors = factors
    
        #self.list_to_ndarray = np.frompyfunc(lambda c : Polynomial(np.array(c).reshape((1,) * self.dim), basis=self.basis, stable=self.stable), 1, 1)


