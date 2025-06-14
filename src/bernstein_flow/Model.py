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

        basis_funcs.append(lambdify(x_syms, B, modules='torch'))
    
    return basis_funcs, multi_indices, x_syms

class BernsteinFlowModel(torch.nn.Module):
    def __init__(self, dim : int, transformer_degrees : int, conditioner_degrees : int):
        super().__init__()

        self.dim = dim

        self.tf_degs = transformer_degrees
        self.cond_degs = conditioner_degrees

        # Parameters
        self.A = torch.nn.ModuleList()
        for i in range(dim):
            # Number of coefficients in each conditioner matrix
            alpha_j_size = (conditioner_degrees[i] + 1)**(i)

            # Initialize the coefficients of each Bernstein conditioner except the first and last one (which are always 0 and 1)
            A_i = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(alpha_j_size)) for _ in range(transformer_degrees[i] - 1)])

            self.A.append(A_i)
        
        # Basis functions
        self.tf_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i]) for i in range(dim)]
        self.tf_deriv_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i] - 1, scale=transformer_degrees[i]) for i in range(dim)]
        self.cond_basis_funcs = [bernstein_basis_functions(i + 1, conditioner_degrees[i]) for i in range(dim)]

    def forward(self, x : torch.Tensor):
        density = torch.ones(x.shape[0], device=x.device)
        for i in range(self.dim):

            # Evaluate the basis functions for each term in the transformer
            tf_basis_vals = torch.stack([phi_j(x[:, i]) for phi_j in self.tf_deriv_basis_funcs[i]], dim=1)

            # Evaluate the basis functions for each term in the conditioner
            cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(i)]) for phi_k in self.cond_basis_funcs[i]], dim=1) if i > 0 else torch.ones(x.shape[0], 1, device=x.device)

            cA_i = self.get_constrained_parameters(i)

            tf_coeffs = torch.stack([alpha_j @ cond_basis_vals for alpha_j in cA_i])

            tf_val = tf_coeffs @ tf_basis_vals[1:-1] + tf_basis_vals[-1] # First weight is 0, last weight is 1

            density *= tf_val
        return density

    def get_constrained_parameters(self, dim : int):
        cA_i = [] # Constrained Ai
        prev = torch.zeros_like(self.A[dim][0])
        for alpha_j in self.A[dim]:
            difference = torch.nn.functional.softplus(alpha_j)
            calpha_j = difference + prev
            cA_i.append(calpha_j)
            prev = calpha_j
        ordered_alpha = torch.stack(cA_i)
        normalized_alpha = torch.sigmoid(ordered_alpha)
        return normalized_alpha

def nll_loss(model, x_data):
    density = model(x_data)
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
            x_batch = x_batch.to(next(model.parameters()).device)
            loss = train_step(model, x_batch, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")



