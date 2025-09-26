import torch
from .SOSModel import SOSModel

class PowerFunctionSOSModel(SOSModel):
    def __init__(self, dy : int, dx : int, n : int, m : int, min_exp : float = -1.0, max_exp : float = 50.0, **kwargs):
        # One parameter for each basis function (alpha) for each dimension
        super().__init__(dy, dx, n, m, dy, dx, **kwargs)

        self.min_exp = min_exp
        self.max_exp = max_exp
    
    def phi(self, x : torch.Tensor):
        # Make each parameter positive and unsqeeze to data shape
        alpha = self.get_phi_params()
        alpha = alpha[None, :, :] # (p, n, dx)

        log_x = torch.log(x) # Shape (p, dx)
        log_x = log_x[:, None, :] # (p, n, dx)

        log_phi_per_dim = alpha * log_x

        # Sum in log space over the dimension
        log_phi = torch.sum(log_phi_per_dim, dim=2) # Shape (p, n)

        # Normalize to sum to 1 over [0,1]^dx
        # For x^α, the integral over [0,1] is 1/(α+1) for each dimension
        # So the normalization factor is ∏_d (α_d + 1)
        normalization = torch.prod(alpha + 1, dim=2) # Shape (p, n)
        log_normalization = torch.log(normalization)
        
        # Apply normalization in log space
        log_phi_normalized = log_phi + log_normalization

        return torch.exp(log_phi_normalized)

    def psi(self, y : torch.Tensor):
        # Make each parameter positive and unsqeeze to data shape
        alpha = self.get_psi_params()
        alpha = alpha[None, :, :] # (p, n*m, dy)

        log_y = torch.log(y)
        log_y = log_y[:, None, :]  # Shape (p, n*m, dy)

        log_psi_per_dim = alpha * log_y

        # Sum in log space over the dimension
        log_psi = torch.sum(log_psi_per_dim, dim=2) # Shape (p, n*m)

        # Normalize to sum to 1 over [0,1]^dy
        # For y^α, the integral over [0,1] is 1/(α+1) for each dimension
        # So the normalization factor is ∏_d (α_d + 1)
        normalization = torch.prod(alpha + 1, dim=2) # Shape (p, n*m)
        log_normalization = torch.log(normalization)
        
        # Apply normalization in log space
        log_psi_normalized = log_psi + log_normalization

        return torch.exp(log_psi_normalized)

    def psi_gram(self):
        # Make each parameter positive and unsqeeze to data shape
        alpha = self.get_psi_params()
        
        # For normalized power functions (α+1)x^α, the inner product over [0,1]^d is:
        # ∫₀¹ (αᵢ+1)x^αᵢ (αⱼ+1)x^αⱼ dx = (αᵢ+1)(αⱼ+1)/(αᵢ + αⱼ + 1) for each dimension
        
        log_inner_products = torch.zeros(self.n * self.m, self.n * self.m, dtype=alpha.dtype, device=alpha.device)
        
        for d in range(self.dy):
            # Get alpha for dimension d
            alpha_d = alpha[:, d]  # Shape: ((n+1)*m,)
            
            # Compute inner products for this dimension
            # For each pair (i, j), compute the inner product of normalized psi_i^d and psi_j^d
            alpha_i = alpha_d.unsqueeze(1)  # Shape: ((n+1)*m, 1)
            alpha_j = alpha_d.unsqueeze(0)  # Shape: (1, (n+1)*m)
            
            # Compute the inner product for this dimension
            # ∫₀¹ (αᵢ+1)x^αᵢ (αⱼ+1)x^αⱼ dx = (αᵢ+1)(αⱼ+1)/(αᵢ + αⱼ + 1)
            normalization_i = alpha_i + 1
            normalization_j = alpha_j + 1
            alpha_sum = alpha_i + alpha_j + 1
            
            # Compute log of the inner product to avoid overflow
            # log((αᵢ+1)(αⱼ+1)/(αᵢ + αⱼ + 1)) = log(αᵢ+1) + log(αⱼ+1) - log(αᵢ + αⱼ + 1)
            log_inner_d = torch.log(normalization_i) + torch.log(normalization_j) - torch.log(alpha_sum)
            
            # Add to existing inner products (multiply in log space)
            log_inner_products += log_inner_d
        
        return torch.exp(log_inner_products)
    
    def get_phi_params(self):
        return (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.phi_params) + self.min_exp

    def get_psi_params(self):
        return (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.psi_params) + self.min_exp
