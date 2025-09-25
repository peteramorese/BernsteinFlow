import torch
from .SOSModel import SOSModel

#class SignomialSOSModel(SOSModel):
#    def __init__(self, dy : int, dx : int, n : int, m : int, n_terms : int, min_exp : float = -1.0, max_exp : float = 50.0, **kwargs):
#        # One parameter for each basis function (alpha) for each dimension
#        super().__init__(dy, dx, n, m, n_terms * (dx + 1), n_terms * (dy + 1), **kwargs)
#
#        self.min_exp = min_exp
#        self.max_exp = max_exp
#        self.n_terms = n_terms
#    
#    def phi(self, x : torch.Tensor):
#        # Split parameters: first n_terms are coefficients, next n_terms*dx are exponents
#        coefficient_params = self.phi_params[:, :self.n_terms]
#        exponent_params = self.phi_params[:, self.n_terms:]
#        assert exponent_params.shape[1] == self.n_terms * self.dx
#
#        # Make exponents positive and add minimum exp
#        exponent_params = (self.max_exp - self.min_exp) * torch.nn.functional.softplus(exponent_params) + self.min_exp
#        
#        # Reshape to (p, n, n_terms, dx) 
#        exponent_params = exponent_params.view(self.n - 1, self.n_terms, self.dx).unsqueeze(0)
#        coefficient_params = coefficient_params.view(self.n - 1, self.n_terms).unsqueeze(0) # (p, n, n_terms)
#        
#        # Compute x^exponent for each term and dimension
#        log_x = torch.log(x)  # (p, dx)
#
#        log_x = log_x[:, None, None, :]
#        
#        # Compute x^exponent for each term: (p, n, n_terms, dx)
#        log_power_terms = exponent_params * log_x
#
#        # Sum over dimensions for each term: (p, n, n_terms)
#        log_power_terms = torch.sum(log_power_terms, dim=3)  # (p, n, n_terms)
#
#        power_terms = torch.exp(log_power_terms)
#        
#        # Sum over terms
#        sum_of_terms = torch.sum(power_terms * coefficient_params, dim=2)
#
#        log_normalization_constant = -torch.sum(exponent_params + 1, dim=3)
#        norm_constant = torch.sum(torch.exp(log_normalization_constant) * coefficient_params, dim=2)
#        
#        return sum_of_terms / norm_constant
#
#    def psi(self, y : torch.Tensor):
#        # Split parameters: first n_terms are coefficients, next n_terms*dy are exponents
#        coefficient_params = self.psi_params[:, :self.n_terms]
#        exponent_params = self.psi_params[:, self.n_terms:]
#        assert exponent_params.shape[1] == self.n_terms * self.dy
#
#        # Make exponents positive and add minimum exp
#        exponent_params = (self.max_exp - self.min_exp) * torch.nn.functional.softplus(exponent_params) + self.min_exp
#        
#        # Reshape to (n*m, n_terms, dy) for easier computation
#        exponent_params = exponent_params.view(self.n * self.m, self.n_terms, self.dy).unsqueeze(0)
#        coefficient_params = coefficient_params.view(self.n * self.m, self.n_terms)
#
#        # y shape: (p, dy), we need (p, 1, 1, dy) for broadcasting
#        log_y = torch.log(y)  # (p, dy)
#        log_y = log_y[:, None, None, :]
#        
#        # Compute y^exponent for each term and dimension
#        log_power_terms = exponent_params * log_y
#
#        # Sum over dimensions for each term: (p, n*m, n_terms)
#        log_power_terms = torch.sum(log_power_terms, dim=3)  # (p, n*m, n_terms)
#
#        power_terms = torch.exp(log_power_terms)
#        
#        # Sum over terms
#        sum_of_terms = torch.sum(power_terms * coefficient_params, dim=2)
#
#        log_normalization_constant = -torch.sum(exponent_params + 1, dim=3)
#        norm_constant = torch.sum(torch.exp(log_normalization_constant) * coefficient_params, dim=2)
#        
#        return sum_of_terms / norm_constant
#        
#    def psi_inner_product_mat(self):
#        # Split parameters
#        coefficient_params = self.psi_params[:, :self.n_terms]
#        exponent_params = self.psi_params[:, self.n_terms:]
#        assert exponent_params.shape[1] == self.n_terms * self.dy
#
#        # Make exponents positive and add minimum alpha
#        exponent_params = torch.nn.functional.softplus(exponent_params) + self.min_exp
#        
#        # Reshape
#        exponent_params = exponent_params.view(self.n * self.m, self.n_terms, self.dy)
#        coefficient_params = coefficient_params.view(self.n * self.m, self.n_terms)
#
#        # ---- Normalization constants ----
#        # N_i = Σ_k c_{i,k} * ∏_d 1/(α_{i,k,d} + 1)
#        norm_factors = torch.prod(1.0 / (exponent_params + 1.0), dim=2)  # (n*m, n_terms)
#        normalization_constants = torch.sum(coefficient_params * norm_factors, dim=1)  # (n*m,)
#
#        # ---- Pairwise inner products ----
#        exp_i = exponent_params.unsqueeze(1).unsqueeze(3)  # (n*m, 1, 1, n_terms, dy)
#        exp_j = exponent_params.unsqueeze(0).unsqueeze(2)  # (1, n*m, n_terms, 1, dy)
#
#        alpha_sum = exp_i + exp_j + 1.0
#        term_inner_products = torch.prod(1.0 / alpha_sum, dim=4)  # (n*m, n*m, n_terms, n_terms)
#
#        coeff_i = coefficient_params.unsqueeze(1).unsqueeze(3)  # (n*m, 1, 1, n_terms)
#        coeff_j = coefficient_params.unsqueeze(0).unsqueeze(2)  # (1, n*m, n_terms, 1)
#        coeff_products = coeff_i * coeff_j
#
#        unnormalized_inner_products = torch.sum(coeff_products * term_inner_products, dim=(2, 3))  # (n*m, n*m)
#
#        # ---- Apply normalization ----
#        norm_i = normalization_constants.unsqueeze(1)  # (n*m, 1)
#        norm_j = normalization_constants.unsqueeze(0)  # (1, n*m)
#        inner_products = unnormalized_inner_products / (norm_i * norm_j)
#
#        return inner_products
#    
#    def get_phi_params(self):
#        return (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.phi_params) + self.min_exp
#
#    def get_psi_params(self):
#        return (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.psi_params) + self.min_exp


class SignomialSOSModel(SOSModel):
    def __init__(self, dy: int, dx: int, n: int, m: int, n_terms: int,
                 min_exp: float = -0.999, max_exp: float = 50.0,  # avoid -1 exactly
                 eps: float = 1e-12, pos_coeffs: bool = True, **kwargs):
        super().__init__(dy, dx, n, m, n_terms * (dx + 1), n_terms * (dy + 1), **kwargs)
        self.min_exp = min_exp
        self.max_exp = max_exp
        self.n_terms = n_terms
        self.eps = eps
        self.pos_coeffs = pos_coeffs  # if True, enforce c_k >= 0 via softplus

    # --- helpers --------------------------------------------------------------

    def _map_exponents(self, raw):
        # same mapping everywhere
        return (self.max_exp - self.min_exp) * torch.nn.functional.softplus(raw) + self.min_exp

    def _map_coeffs(self, raw):
        if self.pos_coeffs:
            return torch.nn.functional.softplus(raw) + self.eps  # strictly positive
        return raw

    def _log_norm_terms(self, exps):
        # exps shape (..., dim)
        # log ∏_d 1/(α_d+1) = -∑_d log(α_d+1)
        return -torch.sum(torch.log(exps + 1.0 + self.eps), dim=-1)

    # --- phi ------------------------------------------------------------------

    def phi(self, x: torch.Tensor):
        # params
        coeff_raw = self.phi_params[:, :self.n_terms]
        exp_raw   = self.phi_params[:, self.n_terms:]
        assert exp_raw.shape[1] == self.n_terms * self.dx

        coeff = self._map_coeffs(coeff_raw)  # (n-1, n_terms)
        exps  = self._map_exponents(exp_raw).view(self.n - 1, self.n_terms, self.dx)  # (n-1, n_terms, dx)

        # shapes for batch p
        coeff = coeff.unsqueeze(0)                     # (1, n-1, n_terms)
        exps  = exps.unsqueeze(0)                      # (1, n-1, n_terms, dx)

        # clamp x to avoid log(0)
        x = torch.clamp(x, self.eps, 1.0 - self.eps)   # (p, dx)
        log_x = torch.log(x)[:, None, None, :]         # (p, 1, 1, dx)

        # log power per term: ∑_d α_d log x_d
        log_power_terms = torch.sum(exps * log_x, dim=3)     # (p, n-1, n_terms)
        power_terms = torch.exp(log_power_terms)             # (p, n-1, n_terms)

        # numerator: Σ_k c_k * ∏_d x_d^{α_d}
        numer = torch.sum(power_terms * coeff, dim=2)        # (p, n-1)

        # denominator: Σ_k c_k * ∏_d 1/(α_d+1)
        log_den_terms = self._log_norm_terms(exps)           # (1, n-1, n_terms)
        # stable log-sum-exp over k with weights c_k:
        # log Σ_k c_k * a_k = log Σ_k exp(log c_k + log a_k)
        log_coeff = torch.log(coeff)                          # (1, n-1, n_terms)
        log_den = torch.logsumexp(log_coeff + log_den_terms, dim=2)  # (1, n-1)
        den = torch.exp(log_den)                              # (1, n-1)

        return numer / (den + self.eps)                       # (p, n-1)

    # --- psi ------------------------------------------------------------------

    def psi(self, y: torch.Tensor):
        coeff_raw = self.psi_params[:, :self.n_terms]
        exp_raw   = self.psi_params[:, self.n_terms:]
        assert exp_raw.shape[1] == self.n_terms * self.dy

        coeff = self._map_coeffs(coeff_raw).view(self.n * self.m, self.n_terms)     # (nm, n_terms)
        exps  = self._map_exponents(exp_raw).view(self.n * self.m, self.n_terms, self.dy)  # (nm, n_terms, dy)

        # batch shapes
        exps  = exps.unsqueeze(0)                         # (1, nm, n_terms, dy)
        coeff = coeff.unsqueeze(0)                        # (1, nm, n_terms)

        y = torch.clamp(y, self.eps, 1.0 - self.eps)      # (p, dy)
        log_y = torch.log(y)[:, None, None, :]            # (p, 1, 1, dy)

        log_power_terms = torch.sum(exps * log_y, dim=3)  # (p, nm, n_terms)
        power_terms = torch.exp(log_power_terms)          # (p, nm, n_terms)

        numer = torch.sum(power_terms * coeff, dim=2)     # (p, nm)

        log_den_terms = self._log_norm_terms(exps)        # (1, nm, n_terms)
        log_den = torch.logsumexp(torch.log(coeff) + log_den_terms, dim=2)  # (1, nm)
        den = torch.exp(log_den)

        return numer / (den + self.eps)                   # (p, nm)

    # --- Gram of psi ----------------------------------------------------------

    def psi_inner_product_mat(self):
        coeff_raw = self.psi_params[:, :self.n_terms]
        exp_raw   = self.psi_params[:, self.n_terms:]
        assert exp_raw.shape[1] == self.n_terms * self.dy

        # use the SAME mappings as in psi()
        coeff = self._map_coeffs(coeff_raw).view(self.n * self.m, self.n_terms)           # (nm, n_terms)
        exps  = self._map_exponents(exp_raw).view(self.n * self.m, self.n_terms, self.dy)  # (nm, n_terms, dy)

        # normalization constants N_i = Σ_k c_{i,k} ∏_d 1/(α_{i,k,d}+1)
        log_norm_terms = -torch.sum(torch.log(exps + 1.0 + self.eps), dim=2)   # (nm, n_terms)
        log_norm_i = torch.logsumexp(torch.log(coeff) + log_norm_terms, dim=1) # (nm,)
        N = torch.exp(log_norm_i) + self.eps                                   # (nm,)

        # pairwise ∑_{k,l} c_i,k c_j,l ∏_d 1/(α_i,k,d + α_j,l,d + 1)
        exp_i = exps.unsqueeze(1).unsqueeze(3)   # (nm, 1, 1, n_terms, dy)
        exp_j = exps.unsqueeze(0).unsqueeze(2)   # (1, nm, n_terms, 1, dy)

        denom = exp_i + exp_j + 1.0
        # product over dimensions with eps
        term_ip = torch.prod(1.0 / (denom + self.eps), dim=4)  # (nm, nm, n_terms, n_terms)

        c_i = coeff.unsqueeze(1).unsqueeze(3)                  # (nm, 1, 1, n_terms)
        c_j = coeff.unsqueeze(0).unsqueeze(2)                  # (1, nm, n_terms, 1)
        coeff_prod = c_i * c_j                                 # (nm, nm, n_terms, n_terms)

        unnorm = torch.sum(coeff_prod * term_ip, dim=(2, 3))   # (nm, nm)

        # normalize
        G = unnorm / (N.unsqueeze(1) * N.unsqueeze(0))         # (nm, nm)
        return G

    # (optional) expose bounded params if you still want these:
    def get_phi_params(self):
        # match _map_exponents if you use them downstream
        exps = self._map_exponents(self.phi_params[:, self.n_terms:])
        coeff = self._map_coeffs(self.phi_params[:, :self.n_terms])
        return torch.cat([coeff, exps], dim=1)

    def get_psi_params(self):
        exps = self._map_exponents(self.psi_params[:, self.n_terms:])
        coeff = self._map_coeffs(self.psi_params[:, :self.n_terms])
        return torch.cat([coeff, exps], dim=1)
