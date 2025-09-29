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
        alpha = (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.phi_params) + self.min_exp
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
        alpha = (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.psi_params) + self.min_exp
        alpha = alpha[None, :, :] # (p, n*m-1, dy)

        log_y = torch.log(y)
        log_y = log_y[:, None, :]  # Shape (p, n*m-1, dy)

        log_psi_per_dim = alpha * log_y

        # Sum in log space over the dimension
        log_psi = torch.sum(log_psi_per_dim, dim=2) # Shape (p, n*m-1)

        # Normalize to sum to 1 over [0,1]^dy
        # For y^α, the integral over [0,1] is 1/(α+1) for each dimension
        # So the normalization factor is ∏_d (α_d + 1)
        normalization = torch.prod(alpha + 1, dim=2) # Shape (p, n*m-1)
        log_normalization = torch.log(normalization)
        
        # Apply normalization in log space
        log_psi_normalized = log_psi + log_normalization

        return torch.exp(log_psi_normalized)

    def gram_tensor(self, cross_gram_model : 'PowerFunctionSOSModel' = None):
        """
        Compute the gram tensor using self's phi and psi parameters. If cross_gram_model is provided, use its psi parameters.
        """
        if cross_gram_model is None:
            phi_alpha = (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.phi_params) + self.min_exp
            psi_alpha = (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.psi_params) + self.min_exp
        else:
            phi_alpha = (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.phi_params) + self.min_exp
            psi_alpha = (cross_gram_model.max_exp - cross_gram_model.min_exp) * torch.nn.functional.sigmoid(cross_gram_model.psi_params) + cross_gram_model.min_exp
        
        # Initialize the 4D gram tensor: (n, n, n*m-1, n*m-1)
        log_gram_tensor = torch.zeros(self.n, self.n, self.n * self.m - 1, self.n * self.m - 1, dtype=phi_alpha.dtype, device=phi_alpha.device)
        
        # For each dimension, compute the inner products
        for d in range(self.dy):
            phi_alpha_d = phi_alpha[:, d]  # Shape: (n,)
            psi_alpha_d = psi_alpha[:, d]  # Shape: (n*m-1,)
            
            # Create all combinations of indices
            i_idx = torch.arange(self.n, device=phi_alpha.device)
            j_idx = torch.arange(self.n, device=phi_alpha.device)
            k_idx = torch.arange(self.n * self.m - 1, device=phi_alpha.device)
            l_idx = torch.arange(self.n * self.m - 1, device=phi_alpha.device)
            
            # Always compute as φᵢ φⱼ ψₖ ψₗ (regular gram tensor)
            phi_alpha_i = phi_alpha_d[i_idx, None, None, None]  # (n, 1, 1, 1)
            phi_alpha_j = phi_alpha_d[None, j_idx, None, None]  # (1, n, 1, 1)
            psi_alpha_k = psi_alpha_d[None, None, k_idx, None]  # (1, 1, n*m-1, 1)
            psi_alpha_l = psi_alpha_d[None, None, None, l_idx]  # (1, 1, 1, n*m-1)
            
            # The integral of the product of four normalized power functions is:
            # ∫₀¹ (αᵢ+1)x^αᵢ (αⱼ+1)x^αⱼ (αₖ+1)x^αₖ (αₗ+1)x^αₗ dx
            # = (αᵢ+1)(αⱼ+1)(αₖ+1)(αₗ+1) / (αᵢ + αⱼ + αₖ + αₗ + 1)
            
            # Sum all alpha parameters
            total_alpha = phi_alpha_i + phi_alpha_j + psi_alpha_k + psi_alpha_l + 1
            
            # Compute log of the integral to avoid overflow
            log_integral = (
                torch.log(phi_alpha_i + 1) + torch.log(phi_alpha_j + 1) 
                + torch.log(psi_alpha_k + 1) + torch.log(psi_alpha_l + 1)
                - torch.log(total_alpha)
            )
            
            # Add to the gram tensor for this dimension
            log_gram_tensor += log_integral
        
        # Convert from log space
        gram_tensor = torch.exp(log_gram_tensor)
        
        # If cross_gram_model is provided, permute indices to get ψ'ᵢ ψ'ⱼ φₖ φₗ
        if cross_gram_model is not None:
            # Permute from (i,j,k,l) to (k,l,i,j) to get ψ'ᵢ ψ'ⱼ φₖ φₗ
            gram_tensor = gram_tensor.permute(2, 3, 0, 1)
        
        return gram_tensor
    
    def get_phi_params(self):
        return (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.phi_params) + self.min_exp

    def get_psi_params(self):
        return (self.max_exp - self.min_exp) * torch.nn.functional.sigmoid(self.psi_params) + self.min_exp
