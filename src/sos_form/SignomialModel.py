import torch
from .SOSModel import SOSModel

class SignomialSOSModel(SOSModel):
    def __init__(self, dy : int, dx : int, n : int, m : int, n_terms : int, min_alpha : float = -1.0, **kwargs):
        # One parameter for each basis function (alpha) for each dimension
        super().__init__(dy, dx, n, m, n_terms * (dx + 1), n_terms * (dy + 1), **kwargs)

        self.min_alpha = min_alpha
        self.n_terms = n_terms
    
    def phi(self, x : torch.Tensor):
        # Split parameters: first n_terms are coefficients, next n_terms*dx are exponents
        coefficient_params = self.phi_params[:, :self.n_terms]
        exponent_params = self.phi_params[:, self.n_terms:]
        assert exponent_params.shape[1] == self.n_terms * self.dx

        # Make exponents positive and add minimum alpha
        exponent_params = torch.nn.functional.softplus(exponent_params) + self.min_alpha
        
        # Reshape to (n, n_terms, dx) for easier computation
        exponent_params = exponent_params.view(self.n - 1, self.n_terms, self.dx)
        coefficient_params = coefficient_params.view(self.n - 1, self.n_terms)
        
        # x shape: (p, dx), we need (p, 1, 1, dx) for broadcasting
        x_expanded = x.unsqueeze(1).unsqueeze(1)  # (p, 1, 1, dx)
        
        # Compute x^exponent for each term and dimension
        log_x = torch.log(x_expanded)  # (p, 1, 1, dx)
        
        # exponent_params: (n, n_terms, dx) -> (1, n, n_terms, dx)
        exponent_params = exponent_params.unsqueeze(0)  # (1, n, n_terms, dx)
        
        # Compute x^exponent for each term: (p, n, n_terms, dx)
        log_power_terms = exponent_params * log_x.unsqueeze(2)  # (p, n, n_terms, dx)
        
        # Sum over dimensions for each term: (p, n, n_terms)
        log_power_terms = torch.sum(log_power_terms, dim=3)  # (p, n, n_terms)
        
        # Apply coefficients and sum over terms
        coefficient_params = coefficient_params.unsqueeze(0)  # (1, n, n_terms)
        
        # Compute coefficient * x^exponent for each term
        log_coeff_terms = torch.log(coefficient_params + 1e-10) + log_power_terms  # (p, n, n_terms)
        
        # Sum over terms: (p, n)
        log_phi = torch.logsumexp(log_coeff_terms, dim=2)  # (p, n)
        
        return torch.exp(log_phi)

    def psi(self, y : torch.Tensor):
        # Split parameters: first n_terms are coefficients, next n_terms*dy are exponents
        coefficient_params = self.psi_params[:, :self.n_terms]
        exponent_params = self.psi_params[:, self.n_terms:]
        assert exponent_params.shape[1] == self.n_terms * self.dy

        # Make exponents positive and add minimum alpha
        exponent_params = torch.nn.functional.softplus(exponent_params) + self.min_alpha
        
        # Reshape to (n*m, n_terms, dy) for easier computation
        exponent_params = exponent_params.view(self.n * self.m, self.n_terms, self.dy)
        coefficient_params = coefficient_params.view(self.n * self.m, self.n_terms)

        # y shape: (p, dy), we need (p, 1, 1, dy) for broadcasting
        y_expanded = y.unsqueeze(1).unsqueeze(1)  # (p, 1, 1, dy)
        
        # Compute y^exponent for each term and dimension
        log_y = torch.log(y_expanded)  # (p, 1, 1, dy)
        
        # exponent_params: (n*m, n_terms, dy) -> (1, n*m, n_terms, dy)
        exponent_params = exponent_params.unsqueeze(0)  # (1, n*m, n_terms, dy)
        
        # Compute y^exponent for each term: (p, n*m, n_terms, dy)
        log_power_terms = exponent_params * log_y.unsqueeze(2)  # (p, n*m, n_terms, dy)
        
        # Sum over dimensions for each term: (p, n*m, n_terms)
        log_power_terms = torch.sum(log_power_terms, dim=3)  # (p, n*m, n_terms)
        
        # Apply coefficients and sum over terms
        coefficient_params = coefficient_params.unsqueeze(0)  # (1, n*m, n_terms)
        
        # Compute coefficient * y^exponent for each term
        log_coeff_terms = torch.log(coefficient_params + 1e-10) + log_power_terms  # (p, n*m, n_terms)
        
        # Sum over terms: (p, n*m)
        log_psi = torch.logsumexp(log_coeff_terms, dim=2)  # (p, n*m)
        
        return torch.exp(log_psi)

    def psi_inner_product_mat(self):
        # Split parameters: first n_terms are coefficients, next n_terms*dy are exponents
        coefficient_params = self.psi_params[:, :self.n_terms]
        exponent_params = self.psi_params[:, self.n_terms:]
        assert exponent_params.shape[1] == self.n_terms * self.dy

        # Make exponents positive and add minimum alpha
        exponent_params = torch.nn.functional.softplus(exponent_params) + self.min_alpha
        
        # Reshape to (n*m, n_terms, dy) for easier computation
        exponent_params = exponent_params.view(self.n * self.m, self.n_terms, self.dy)
        coefficient_params = coefficient_params.view(self.n * self.m, self.n_terms)
        
        # Initialize the inner product matrix
        inner_products = torch.zeros(self.n * self.m, self.n * self.m, 
                                   dtype=exponent_params.dtype, device=exponent_params.device)
        
        # For signomials, we need to compute the inner product of each pair of basis functions
        # Each basis function is a sum of terms: sum_k c_k * ∏_d y_d^α_k_d
        # The inner product of two signomials is:
        # ∫ sum_i c_i * ∏_d y_d^α_i_d * sum_j d_j * ∏_d y_d^β_j_d dy
        # = sum_i sum_j c_i * d_j * ∫ ∏_d y_d^(α_i_d + β_j_d) dy
        # = sum_i sum_j c_i * d_j * ∏_d ∫[0,1] y_d^(α_i_d + β_j_d) dy_d
        # = sum_i sum_j c_i * d_j * ∏_d 1/(α_i_d + β_j_d + 1)
        
        for i in range(self.n * self.m):
            for j in range(self.n * self.m):
                # Get parameters for basis functions i and j
                exp_i = exponent_params[i]  # (n_terms, dy)
                coeff_i = coefficient_params[i]  # (n_terms,)
                exp_j = exponent_params[j]  # (n_terms, dy)
                coeff_j = coefficient_params[j]  # (n_terms,)
                
                # Compute inner product between all pairs of terms
                inner_product = 0.0
                
                for k in range(self.n_terms):
                    for l in range(self.n_terms):
                        # Get the k-th term of basis i and l-th term of basis j
                        exp_i_k = exp_i[k]  # (dy,)
                        coeff_i_k = coeff_i[k]  # scalar
                        exp_j_l = exp_j[l]  # (dy,)
                        coeff_j_l = coeff_j[l]  # scalar
                        
                        # Compute the inner product of the k-th term of i and l-th term of j
                        # This is: ∫[0,1]^dy c_i_k * ∏_d y_d^α_i_k_d * c_j_l * ∏_d y_d^β_j_l_d dy
                        # = c_i_k * c_j_l * ∫[0,1]^dy ∏_d y_d^(α_i_k_d + β_j_l_d) dy
                        # = c_i_k * c_j_l * ∏_d ∫[0,1] y_d^(α_i_k_d + β_j_l_d) dy_d
                        # = c_i_k * c_j_l * ∏_d 1/(α_i_k_d + β_j_l_d + 1)
                        
                        # Compute the product over dimensions
                        term_inner_product = 1.0
                        for d in range(self.dy):
                            alpha_sum = exp_i_k[d] + exp_j_l[d] + 1
                            if alpha_sum > 0:
                                term_inner_product *= 1.0 / alpha_sum
                            else:
                                term_inner_product = 0.0
                                break
                        
                        # Multiply by coefficients
                        coeff_product = coeff_i_k * coeff_j_l
                        inner_product += coeff_product * term_inner_product
                
                inner_products[i, j] = inner_product
        
        return inner_products