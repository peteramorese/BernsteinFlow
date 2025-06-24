import itertools
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from sympy import symbols, binomial, lambdify

def bernstein_basis_functions(dim : int, degree : int, coefficients : torch.Tensor = None, scale : float = 1.0):
    """
    Returns a list of lambdified multivariate Bernstein basis functions and their multi-indices.
    
    Args:
        dim (int): number of variables (dimension)
        degree (int): Bernstein polynomial degree in each variable
        coefficients (torch.Tensor): optional coefficients for the basis functions (flattened)
    
    Returns:
        basis_funcs (List[Callable]): List of lambdified basis functions compatible with torch
        multi_indices (List[Tuple[int]]): Corresponding multi-indices for each basis function
        x_syms (Tuple[sympy.Symbol]): Tuple of sympy symbols (x0, x1, ..., xd-1)
    """
    x_syms = symbols(f'x0:{dim}')
    multi_indices = list(itertools.product(range(degree + 1), repeat=dim))
    
    basis_funcs = []
    for flat_idx, alpha in enumerate(multi_indices):
        B = 1
        for i in range(dim):
            coeff = float(math.comb(degree, alpha[i]))
            B *= coeff * x_syms[i]**alpha[i] * (1 - x_syms[i])**(degree - alpha[i])
            #B *= binomial(degree, alpha[i]) * x_syms[i]**alpha[i] * (1 - x_syms[i])**(degree - alpha[i])
        
        # Incorporate scale/coefficients
        #print("B: ", B)
        B *= scale
        if coefficients is not None:
            B *= coefficients[flat_idx]

        basis_funcs.append(lambdify(x_syms, B, modules='torch'))
    
    return basis_funcs

class BernsteinFlowModel(torch.nn.Module):
    def __init__(self, dim : int, transformer_degrees : int, conditioner_degrees : int):
        super().__init__()

        self.dim = dim

        self.tf_degs = transformer_degrees
        self.cond_degs = conditioner_degrees

        # Parameters
        self.A = torch.nn.ParameterList()
        for i in range(dim):
            # Number of coefficients in each conditioner polynomial (flattened tensor)
            alpha_j_size = (conditioner_degrees[i] + 1)**(i)

            # Initialize the coefficients of each Bernstein conditioner except the first and last one (which are always 0 and 1)
            # Rows correspond to basis polynomials, columns correspond to which TF coefficient the conditioner outputs
            alpha_matrix = torch.nn.Parameter(torch.rand(alpha_j_size, transformer_degrees[i] - 1))
        
            self.A.append(alpha_matrix)
        
        # Basis functions
        self.tf_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i]) for i in range(dim)]
        self.tf_deriv_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i] - 1, scale=transformer_degrees[i]) for i in range(dim)]
        self.cond_basis_funcs = [bernstein_basis_functions(i, conditioner_degrees[i]) for i in range(dim)]
        self.cond_input_dims = list(range(dim))
    
    def n_parameters(self):
        n_params = 0
        for alpha_matrix in self.A:
            n_params += torch.numel(alpha_matrix)
        return n_params

    def forward(self, x : torch.Tensor):
        density = torch.ones(x.shape[0], device=x.device)
        for i in range(self.dim):
            tf_val = self.transformer_deriv(x, i)
            density *= tf_val
        return density

    def get_constrained_parameters(self, i : int):
        alpha_matrix = self.A[i]
        c_alpha_matrix = torch.zeros_like(alpha_matrix) # Constrained alpha matrix
        prev = torch.zeros_like(alpha_matrix[:, 0])
        for j in range(alpha_matrix.shape[1]): # Iterate through the vectors of each conditioner component
            difference = torch.nn.functional.softplus(alpha_matrix[:, j])
            calpha_j = difference + prev
            c_alpha_matrix[:, j] = calpha_j
            prev = calpha_j

        # Normalize all coefficients to be between 0 and 1 (they are already ordered by design)
        # All coefficients are > 0 so the range of sigmoid is (0.5, 1.0); shift and rescale to make range (0.0, 1.0)
        c_alpha_matrix = 2.0 * (torch.sigmoid(c_alpha_matrix) - 0.5)
        return c_alpha_matrix
    
    def transformer(self, x : torch.Tensor, i : int):
        assert i < self.dim, "Index of transformer is greater than the dimension of the random variable"

        tf_basis_vals = torch.stack([phi_j(x[:, i]) for phi_j in self.tf_basis_funcs[i]], dim=1)

        # Evaluate the basis functions for each term in the conditioner
        cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(self.cond_input_dims[i])]) for phi_k in self.cond_basis_funcs[i]], dim=1) if self.cond_input_dims[i] > 0 else torch.ones(x.shape[0], 1, device=x.device)
        
        c_alpha_matrix = self.get_constrained_parameters(i)

        tf_coeffs = cond_basis_vals @ c_alpha_matrix

        # Augment the coefficient matrices with the first and last coefficients
        # First weight is 0, last weight is 1
        coeff_ones = torch.ones(tf_coeffs.shape[0], 1, device=x.device)
        coeff_zeros = torch.zeros(tf_coeffs.shape[0], 1, device=x.device)
        tf_coeffs = torch.cat([coeff_zeros, tf_coeffs, coeff_ones], dim=1)

        tf_val = torch.sum(torch.mul(tf_coeffs, tf_basis_vals), 1) 
        return tf_val
    
    def transformer_deriv(self, x : torch.Tensor, i : int):
        assert i < self.dim, "Number of transformers is the dimension of the random variable"

        # Evaluate the basis functions for each term in the transformer
        #print("number of tf basis funcs", len(self.tf_deriv_basis_funcs[i]))
        tf_deriv_basis_vals = torch.stack([phi_j(x[:, i]) for phi_j in self.tf_deriv_basis_funcs[i]], dim=1)

        # Evaluate the basis functions for each term in the conditioner
        cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(self.cond_input_dims[i])]) for phi_k in self.cond_basis_funcs[i]], dim=1) if self.cond_input_dims[i] > 0 else torch.ones(x.shape[0], 1, device=x.device)
        
        c_alpha_matrix = self.get_constrained_parameters(i)

        tf_coeffs = cond_basis_vals @ c_alpha_matrix

        # Compute the difference coefficients used in the derivative of bernstein polynomial (c_{j+1} - c_j)
        # First weight is 0, last weight is 1
        coeff_ones = torch.ones(tf_coeffs.shape[0], 1, device=x.device)
        coeff_zeros = torch.zeros(tf_coeffs.shape[0], 1, device=x.device)
        tf_deriv_coeffs = torch.cat([tf_coeffs, coeff_ones], dim=1) - torch.cat([coeff_zeros, tf_coeffs], dim=1)

        tf_val = torch.sum(torch.mul(tf_deriv_coeffs, tf_deriv_basis_vals), 1) 
        return tf_val
    
    def get_transformer_polynomials(self):
        p_list = []
        for i in range(self.dim):
            c_alpha_matrix = self.get_constrained_parameters(i).detach().clone()
            coeff_ones = torch.ones(c_alpha_matrix.shape[0], 1, device=c_alpha_matrix.device)
            coeff_zeros = torch.zeros(c_alpha_matrix.shape[0], 1, device=c_alpha_matrix.device)
            # Formula for derivative of transformer dimension
            tf_deriv_coefficients = self.tf_degs[i] * (torch.cat([c_alpha_matrix, coeff_ones], dim=1) - torch.cat([coeff_zeros, c_alpha_matrix], dim=1))

            # These coefficients are correct, but in matrix form; port them to tensor form
            cond_input_dim = self.cond_input_dims[i]
            cond_input_deg = self.cond_degs[i]

            tf_deriv_coeffs_shape = (cond_input_dim) * [cond_input_deg + 1] + [tf_deriv_coefficients.shape[1]]
            p_list.append(tf_deriv_coefficients.view(tf_deriv_coeffs_shape))
        return p_list


class ConditionalBernsteinFlowModel(BernsteinFlowModel):
    def __init__(self, dim : int, conditional_dim : int, transformer_degrees : int, conditioner_degrees : int):
        torch.nn.Module.__init__(self)

        self.dim = dim
        self.joint_dim = dim + conditional_dim

        self.tf_degs = transformer_degrees
        self.cond_degs = conditioner_degrees

        # Parameters
        self.A = torch.nn.ParameterList()
        for i in range(dim):
            # Number of coefficients in each conditioner polynomial (flattened tensor)
            # For conditional model, the dimension of each condition effectively increase by the conditional dim
            alpha_j_size = (conditioner_degrees[i] + 1)**(i + conditional_dim)

            # Initialize the coefficients of each Bernstein conditioner except the first and last one (which are always 0 and 1)
            # Rows correspond to basis polynomials, columns correspond to which TF coefficient the conditioner outputs
            alpha_matrix = torch.nn.Parameter(torch.rand(alpha_j_size, transformer_degrees[i] - 1))
        
            self.A.append(alpha_matrix)
        
        # Basis functions
        self.tf_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i]) for i in range(dim)]
        self.tf_deriv_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i] - 1, scale=transformer_degrees[i]) for i in range(dim)]
        self.cond_basis_funcs = [bernstein_basis_functions(i + conditional_dim, conditioner_degrees[i]) for i in range(dim)]
        self.cond_input_dims = list(range(conditional_dim, conditional_dim + dim))


def nll_loss(model, data):
    density = model(data)
    log_density = torch.log(density + 1e-10)
    loss = -log_density.mean()
    return loss

def train_step(model, x_data, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = nll_loss(model, x_data)
    loss.backward()
    optimizer.step()
    return loss.item()

def optimize(model, data_loader : DataLoader, optimizer, epochs=100):
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss = train_step(model, x_batch, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")