import itertools
import torch
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
            B *= binomial(degree, alpha[i]) * x_syms[i]**alpha[i] * (1 - x_syms[i])**(degree - alpha[i])
        
        # Incorporate scale/coefficients
        B *= scale
        if coefficients is not None:
            B *= coefficients[flat_idx]

        #print("flat idx: ", flat_idx, "b: ", B)
        basis_funcs.append(lambdify(x_syms, B, modules='torch'))
    
    #print("bais funcs:", len(basis_funcs))
    return basis_funcs #, multi_indices, x_syms

class BernsteinFlowModel(torch.nn.Module):
    def __init__(self, dim : int, transformer_degrees : int, conditioner_degrees : int):
        super().__init__()

        self.dim = dim

        self.tf_degs = transformer_degrees
        self.cond_degs = conditioner_degrees

        ## Parameters
        #self.A = torch.nn.ModuleList()
        #for i in range(dim):
        #    # Number of coefficients in each conditioner matrix
        #    alpha_j_size = (conditioner_degrees[i] + 1)**(i)

        #    # Initialize the coefficients of each Bernstein conditioner except the first and last one (which are always 0 and 1)
        #    A_i = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(alpha_j_size)) for _ in range(transformer_degrees[i] - 1)])

        #    self.A.append(A_i)

        # Parameters
        #self.A = torch.nn.ModuleList()
        self.A = torch.nn.ParameterList()
        for i in range(dim):
            # Number of coefficients in each conditioner polynomial (flattened tensor)
            alpha_j_size = (conditioner_degrees[i] + 1)**(i)

            # Initialize the coefficients of each Bernstein conditioner except the first and last one (which are always 0 and 1)
            # Rows correspond to basis polynomials, columns correspond to which TF coefficient the conditioner outputs
            alpha_matrix = torch.nn.Parameter(torch.rand(alpha_j_size, transformer_degrees[i] - 1))
        
            self.A.append(alpha_matrix)
            #A_i = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(alpha_j_size)) for _ in range(transformer_degrees[i] - 1)])

        
        # Basis functions
        self.tf_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i]) for i in range(dim)]
        self.tf_deriv_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i] - 1, scale=transformer_degrees[i]) for i in range(dim)]
        self.cond_basis_funcs = [bernstein_basis_functions(i, conditioner_degrees[i]) for i in range(dim)]

    def forward(self, x : torch.Tensor):
        density = torch.ones(x.shape[0], device=x.device)
        for i in range(self.dim):
            #print("tf deriv basis funcs:", self.tf_deriv_basis_funcs[i])

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
        c_alpha_matrix = torch.sigmoid(c_alpha_matrix)
        return c_alpha_matrix
    
    def transformer_deriv(self, x : torch.Tensor, i : int):
        # Evaluate the basis functions for each term in the transformer
        #print("number of tf basis funcs", len(self.tf_deriv_basis_funcs[i]))
        tf_deriv_basis_vals = torch.stack([phi_j(x[:, i]) for phi_j in self.tf_deriv_basis_funcs[i]], dim=1)
        #print("TF basis vals: \n", [phi_j(x[:, i]) for phi_j in self.tf_deriv_basis_funcs[i]])
        #print("TF basis vals stacked: \n", tf_deriv_basis_vals)

        # Evaluate the basis functions for each term in the conditioner
        cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(i)]) for phi_k in self.cond_basis_funcs[i]], dim=1) if i > 0 else torch.ones(x.shape[0], 1, device=x.device)
        

        c_alpha_matrix = self.get_constrained_parameters(i)

        # Repeat the basis values for each tf coefficient
        #cond_basis_vals = cond_basis_vals.repeat(1, len(cA_i))
        

        tf_coeffs = cond_basis_vals @ c_alpha_matrix

        # Compute the difference coefficients used in the derivative of bernstein polynomial (c_{j+1} - c_j)
        # First weight is 0, last weight is 1
        coeff_ones = torch.ones(tf_coeffs.shape[0], 1, device=x.device)
        coeff_zeros = torch.zeros(tf_coeffs.shape[0], 1, device=x.device)
        tf_deriv_coeffs = torch.cat([tf_coeffs, coeff_ones], dim=1) - torch.cat([coeff_zeros, tf_coeffs], dim=1)

        #print("tf deriv coef: ", tf_deriv_coeffs)
        #print("tf basis vals: ", tf_deriv_basis_vals)
        tf_val = torch.sum(torch.mul(tf_deriv_coeffs, tf_deriv_basis_vals), 1) 
        return tf_val
    
    def transformer(self, x : torch.Tensor, i : int):
        tf_basis_vals = torch.stack([phi_j(x[:, i]) for phi_j in self.tf_basis_funcs[i]], dim=1)

        # Evaluate the basis functions for each term in the conditioner
        cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(i)]) for phi_k in self.cond_basis_funcs[i]], dim=1) if i > 0 else torch.ones(x.shape[0], 1, device=x.device)
        

        c_alpha_matrix = self.get_constrained_parameters(i)

        tf_coeffs = cond_basis_vals @ c_alpha_matrix

        # Compute the difference coefficients used in the derivative of bernstein polynomial (c_{j+1} - c_j)
        # First weight is 0, last weight is 1
        coeff_ones = torch.ones(tf_coeffs.shape[0], 1, device=x.device)
        coeff_zeros = torch.zeros(tf_coeffs.shape[0], 1, device=x.device)
        tf_coeffs = torch.cat([coeff_zeros, tf_coeffs, coeff_ones], dim=1)

        #print("tf deriv coef: ", tf_deriv_coeffs)
        #print("tf basis vals: ", tf_deriv_basis_vals)
        tf_val = torch.sum(torch.mul(tf_coeffs, tf_basis_vals), 1) 
        return tf_val

def nll_loss(model, x_data):
    density = model(x_data)
    #print("densities: ", density)
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

def train(model : BernsteinFlowModel, data_loader : DataLoader, optimizer, epochs=100):
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss = train_step(model, x_batch, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")



