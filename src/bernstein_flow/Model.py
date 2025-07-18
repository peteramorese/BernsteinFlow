import itertools
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from sympy import symbols, binomial, lambdify
import numpy as np
import time
import sys

from .Polynomial import Polynomial, Basis, decasteljau_composition
from .HyperProjection import bernstein_raised_degree_tf

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
    def __init__(self, dim : int, transformer_degrees : list[int], conditioner_degrees : list[int], layers : int = 1, conditioner_deg_incr : list[int] = None, device = None, dtype = torch.float32):
        """
        Create a BFM simple density estimation model

        Args:
            dim : dimension of the support
            transformer_degrees : dim-length list of degrees for each transformer
            conditioner_degrees : dim-length list of degrees for each conditioner
            layers : number of compositional layers to use. If `layers == 1`, avoids using compositional operations
            conditioner_deg_incr : degree increase applied to each conditioner to increase model expressiveness. Slows down training, but does not affect complexity of final polynomial
            device : device to store tensors on
            dtype : data type of tensors
        """
        super().__init__()

        self.dim = dim

        self.tf_degs = transformer_degrees
        self.cond_degs = conditioner_degrees
        self.n_layers = layers
        
        if conditioner_deg_incr is not None:
            assert len(conditioner_degrees) == len(conditioner_deg_incr)
        self.cond_deg_incr = conditioner_deg_incr

        self.device = device
        self.dtype = dtype

        # Parameters
        self.layers = torch.nn.ModuleList([torch.nn.ParameterList() for _ in range(self.n_layers)])
        self.deg_incr_matrices = list() # Transformation matrices to raise the conditioner degrees
        self.mpsi = list() # Moore Penrose Psuedo Inverse for bernstein degree increase
        for i in range(dim):
            # Number of coefficients in each conditioner polynomial (flattened tensor)
            alpha_j_size = (conditioner_degrees[i] + 1)**(i)

            # Initialize the coefficients of each Bernstein conditioner except the first and last one (which are always 0 and 1)
            # Rows correspond to basis polynomials, columns correspond to which TF coefficient the conditioner outputs. Assumes same size across all layers
            for param_list in self.layers:
                # Offset the initial values such that the initialized transformers are near linear, to avoid nan loss
                unconstrained_param_mat = torch.nn.Parameter(0.5 * torch.rand(alpha_j_size, transformer_degrees[i] - 1, dtype=self.dtype, device=self.device) - 1.2) 
                param_list.append(unconstrained_param_mat)
        
            if self.cond_deg_incr is not None:
                if i == 0:
                    self.deg_incr_matrices.append(torch.eye(1, dtype=self.dtype, device=self.device))
                    self.mpsi.append(torch.eye(1, dtype=self.dtype, device=self.device))
                    continue

                # Create the MPSI matrix based on the shape transformation of the bernstein polynomial
                original_shape = (conditioner_degrees[i] + 1,) * (i)
                deg_incr_shape = (conditioner_degrees[i] + conditioner_deg_incr[i] + 1,) * (i)
                deg_incr_matrix_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape).A

                self.deg_incr_matrices.append(torch.from_numpy(deg_incr_matrix_np).to(dtype=self.dtype, device=self.device))
                self.mpsi.append(torch.from_numpy(np.linalg.pinv(deg_incr_matrix_np)).to(dtype=self.dtype, device=self.device)) # Left psuedo-inverse
        
        # Basis functions
        self.tf_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i]) for i in range(dim)]
        self.tf_deriv_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i] - 1, scale=transformer_degrees[i]) for i in range(dim)]
        self.cond_basis_funcs = [bernstein_basis_functions(i, conditioner_degrees[i]) for i in range(dim)]
        self.cond_input_dims = list(range(dim))
    
    def n_parameters(self):
        n_params = 0
        for layer in self.layers:
            for alpha_matrix in layer:
                n_params += torch.numel(alpha_matrix)
        return n_params

    def forward(self, x : torch.Tensor):
        density = torch.ones(x.shape[0], device=x.device)

        if self.n_layers == 1:
            for i in range(self.dim):
                tf_val = self.transformer_deriv(x, i, layer_i=0)
                density *= tf_val
        else:
            layer_input = x
            for layer_i in range(len(self.layers)):
                next_layer_input = []
                #print("layer input bounds: ", torch.min(layer_input).item(), torch.max(layer_input).item())
                for i in range(self.dim):
                    tf_val = self.transformer_deriv(layer_input, i, layer_i=layer_i)
                    density *= tf_val
                    
                    # Compute the next input by moving the previous 'x' through the current transformer
                    y_i = self.transformer(layer_input, i, layer_i=layer_i)
                    next_layer_input.append(y_i)
                layer_input = torch.vstack(next_layer_input).t()
        return density

    def get_constrained_parameters(self, i : int, layer_i : int = 0):
        alpha_matrix = self.layers[layer_i][i] if self.cond_deg_incr is None else self.deg_incr_matrices[i] @ self.layers[layer_i][i]
        #print(" ------ cond deg incr: ", self.cond_deg_incr, " A size: ", self.A[i].shape, " deg incr size: ", self.deg_incr_matrices[i].shape)
        #print(" ------ alpha matrix size: ", alpha_matrix.shape, " using deg raise? : ", self.cond_deg_incr is not None)
        #input("...")

        c_alpha_matrix = torch.zeros_like(alpha_matrix, device=self.device, dtype=self.dtype) # Constrained alpha matrix
        prev = torch.zeros_like(alpha_matrix[:, 0], device=self.device, dtype=self.dtype)
        for j in range(alpha_matrix.shape[1]): # Iterate through the vectors of each conditioner component
            difference = torch.nn.functional.softplus(alpha_matrix[:, j])
            calpha_j = difference + prev
            c_alpha_matrix[:, j] = calpha_j
            prev = calpha_j

        # Normalize all coefficients to be between 0 and 1 (they are already ordered by design)
        # All coefficients are > 0 so the range of sigmoid is (0.5, 1.0); shift and rescale to make range (0.0, 1.0)
        c_alpha_matrix = 2.0 * (torch.sigmoid(c_alpha_matrix) - 0.5)

        # If the degree was raised, project the coefficients back down to the space of actual coefficients using the MPSI
        if self.cond_deg_incr is not None:
            c_alpha_matrix = self.mpsi[i] @ c_alpha_matrix

        #if torch.any(c_alpha_matrix < 0.0):
        #    print("Less than zero! min coeff: ", torch.min(c_alpha_matrix))
        return c_alpha_matrix
    
    def transformer(self, x : torch.Tensor, i : int, layer_i : int):
        assert i < self.dim, "Index of transformer is greater than the dimension of the random variable"
        #if self.device is not None:
        #    assert x.device == self.device, "Devices don't match"

        tf_basis_vals = torch.stack([phi_j(x[:, self.cond_input_dims[i]]) for phi_j in self.tf_basis_funcs[i]], dim=1)

        # Evaluate the basis functions for each term in the conditioner
        cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(self.cond_input_dims[i])]) for phi_k in self.cond_basis_funcs[i]], dim=1) if self.cond_input_dims[i] > 0 else torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype)
        
        c_alpha_matrix = self.get_constrained_parameters(i, layer_i)

        tf_coeffs = cond_basis_vals @ c_alpha_matrix

        # Augment the coefficient matrices with the first and last coefficients
        # First weight is 0, last weight is 1
        coeff_ones = torch.ones(tf_coeffs.shape[0], 1, device=self.device, dtype=self.dtype)
        coeff_zeros = torch.zeros(tf_coeffs.shape[0], 1, device=self.device, dtype=self.dtype)
        tf_coeffs = torch.cat([coeff_zeros, tf_coeffs, coeff_ones], dim=1)

        tf_val = torch.sum(torch.mul(tf_coeffs, tf_basis_vals), 1) 
        return tf_val
    
    def transformer_deriv(self, x : torch.Tensor, i : int, layer_i : int):
        assert i < self.dim, "Number of transformers is the dimension of the random variable"
        #if self.device is not None:
        #    assert x.device == self.device, f"Devices don't match. x device: {x.device}, model device: {self.device}"

        # Evaluate the basis functions for each term in the transformer
        #print("number of tf basis funcs", len(self.tf_deriv_basis_funcs[i]))
        tf_deriv_basis_vals = torch.stack([phi_j(x[:, self.cond_input_dims[i]]) for phi_j in self.tf_deriv_basis_funcs[i]], dim=1)

        # Evaluate the basis functions for each term in the conditioner
        #print("range : ", self.cond_input_dims[i])
        #print("input x: ", [x[:, j] for j in range(self.cond_input_dims[i])])
        ##print("output: ", self.cond_basis_funcs[0](*[x[:, j] for j in range(self.cond_input_dims[i])]))
        #print("cond basis func [0]: ", self.cond_basis_funcs[0])
        cond_basis_vals = torch.stack([phi_k(*[x[:, j] for j in range(self.cond_input_dims[i])]) for phi_k in self.cond_basis_funcs[i]], dim=1) if self.cond_input_dims[i] > 0 else torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype)
        
        c_alpha_matrix = self.get_constrained_parameters(i, layer_i)

        tf_coeffs = cond_basis_vals @ c_alpha_matrix

        # Compute the difference coefficients used in the derivative of bernstein polynomial (c_{j+1} - c_j)
        # First weight is 0, last weight is 1
        coeff_ones = torch.ones(tf_coeffs.shape[0], 1, device=self.device, dtype=self.dtype)
        coeff_zeros = torch.zeros(tf_coeffs.shape[0], 1, device=self.device, dtype=self.dtype)
        tf_deriv_coeffs = torch.cat([tf_coeffs, coeff_ones], dim=1) - torch.cat([coeff_zeros, tf_coeffs], dim=1)

        tf_val = torch.sum(torch.mul(tf_deriv_coeffs, tf_deriv_basis_vals), 1) 
        return tf_val
    
    def get_density_factor_polys(self, dtype = np.float64):
        """
        Retrieve a list of all the polynomial factors used to calculate the density
        """
        decomposed_tf_derivs = []
        decomposed_tfs = []
        for layer_i in range(self.n_layers):
            layer_i_tf_derivs = []
            layer_i_tfs = []
            for i in range(self.dim):
                c_alpha_matrix = self.get_constrained_parameters(i, layer_i=layer_i).detach().clone()
                coeff_ones = torch.ones(c_alpha_matrix.shape[0], 1, device=c_alpha_matrix.device)
                coeff_zeros = torch.zeros(c_alpha_matrix.shape[0], 1, device=c_alpha_matrix.device)
                # Formula for derivative of transformer dimension
                tf_deriv_coeffs = self.tf_degs[i] * (torch.cat([c_alpha_matrix, coeff_ones], dim=1) - torch.cat([coeff_zeros, c_alpha_matrix], dim=1))

                # These coefficients are correct, but in matrix form; port them to tensor form
                cond_input_dim = self.cond_input_dims[i]
                cond_input_deg = self.cond_degs[i]

                tf_deriv_coeffs_shape = (cond_input_dim) * [cond_input_deg + 1] + [tf_deriv_coeffs.shape[1]]
                layer_i_tf_derivs.append(Polynomial(tf_deriv_coeffs.view(tf_deriv_coeffs_shape), basis=Basis.BERN, dtype=dtype))

                if self.n_layers > 1:
                    print("coeff zeros shape :", coeff_zeros.shape, " c alpha shape: ", c_alpha_matrix.shape)
                    tf_coeffs = torch.cat([coeff_zeros, c_alpha_matrix, coeff_ones], dim=1)
                    tf_coeffs_shape = (cond_input_dim) * [cond_input_deg + 1] + [tf_coeffs.shape[1]]
                    layer_i_tfs.append(Polynomial(tf_coeffs.view(tf_coeffs_shape), basis=Basis.BERN, dtype=dtype))
            decomposed_tf_derivs.append(layer_i_tf_derivs)
            decomposed_tfs.append(layer_i_tfs)

        # If there is only a single layer, skip composition
        if self.n_layers == 1:
            return decomposed_tf_derivs[0]
        else:
            factors = decomposed_tf_derivs[0]
            input_polynomial_vec = decomposed_tfs[0]
            for layer_i in range(1, self.n_layers):
                #prev_layer_vector = factors[(layer_i-1)*self.dim:(layer_i)*self.dim]
                curr_layer_tf_derivs = decomposed_tf_derivs[layer_i]
                for i, p in enumerate(curr_layer_tf_derivs):
                    print("p dim: ", p.dim(), " l q vec: ", len(input_polynomial_vec[:i+1]))
                    print("q dims: ", [q.dim() for q in input_polynomial_vec[:i+1]])
                    p_composed = decasteljau_composition(p, input_polynomial_vec[:i+1])
                    factors.append(p_composed)

                if layer_i < self.n_layers - 1:
                    curr_layer_tfs = decomposed_tfs[layer_i]
                    for i, p in enumerate(curr_layer_tfs):
                        print("p dim: ", p.dim(), " l q vec: ", len(input_polynomial_vec[:i+1]))
                        #print("q dims: ", [q.dim() for q in prev_layer_vector[:i+1]])
                        input_polynomial_vec[i] = decasteljau_composition(p, input_polynomial_vec[:i+1])
            return factors




class ConditionalBernsteinFlowModel(BernsteinFlowModel):
    def __init__(self, dim : int, conditional_dim : int, transformer_degrees : int, conditioner_degrees : int, layers : int = 1, conditioner_deg_incr : list[int] = None, device = None, dtype = torch.float32):
        """
        Conditional flow model for p(x | y). The data must be supplied IN THE FORM [y, x] to evaluation/training
        """
        torch.nn.Module.__init__(self)

        self.dim = dim
        self.joint_dim = dim + conditional_dim

        self.tf_degs = transformer_degrees
        self.cond_degs = conditioner_degrees
        self.n_layers = layers

        if conditioner_deg_incr is not None:
            assert len(conditioner_degrees) == len(conditioner_deg_incr)
        self.cond_deg_incr = conditioner_deg_incr

        self.device = device
        self.dtype = dtype

        # Parameters
        self.layers = torch.nn.ModuleList([torch.nn.ParameterList() for _ in range(self.n_layers)])
        self.deg_incr_matrices = list() # Transformation matrices to raise the conditioner degrees
        self.mpsi = list() # Moore Penrose Psuedo Inverse for bernstein degree increase
        for i in range(dim):
            # Number of coefficients in each conditioner polynomial (flattened tensor)
            # For conditional model, the dimension of each condition effectively increase by the conditional dim
            alpha_j_size = (conditioner_degrees[i] + 1)**(i + conditional_dim)

            for param_list in self.layers:
                # Offset the initial values such that the initialized transformers are near linear, to avoid nan loss
                unconstrained_param_mat = torch.nn.Parameter(0.5 * torch.rand(alpha_j_size, transformer_degrees[i] - 1, dtype=self.dtype, device=self.device) - 1.2) 
                param_list.append(unconstrained_param_mat)
        
            if self.cond_deg_incr is not None:
                # Create the MPSI matrix based on the shape transformation of the bernstein polynomial
                original_shape = (conditioner_degrees[i] + 1,) * (i + conditional_dim)
                deg_incr_shape = (conditioner_degrees[i] + conditioner_deg_incr[i] + 1,) * (i + conditional_dim)
                print("orig shape: ",original_shape)
                print("incr shape: ",deg_incr_shape)
                deg_incr_matrix_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape).A
                print("mat shape: ",deg_incr_matrix_np.shape)

                self.deg_incr_matrices.append(torch.from_numpy(deg_incr_matrix_np).to(dtype=self.dtype, device=self.device))
                self.mpsi.append(torch.from_numpy(np.linalg.pinv(deg_incr_matrix_np)).to(dtype=self.dtype, device=self.device)) # Left psuedo-inverse
        
        # Basis functions
        self.tf_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i]) for i in range(dim)]
        self.tf_deriv_basis_funcs = [bernstein_basis_functions(1, transformer_degrees[i] - 1, scale=transformer_degrees[i]) for i in range(dim)]
        self.cond_basis_funcs = [bernstein_basis_functions(i + conditional_dim, conditioner_degrees[i]) for i in range(dim)]
        self.cond_input_dims = list(range(conditional_dim, conditional_dim + dim))


def nll_loss(model, data):
    density = model(data)
    log_density = torch.log(density + 1e-10)
    loss = -log_density.mean()
    #if math.isnan(loss):
    #    print("Parameters: \n", model.get_constrained_parameters(0, 0), model.get_constrained_parameters(1, 0))
    return loss

def train_step(model, x_data, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = nll_loss(model, x_data)
    loss.backward()
    optimizer.step()
    return loss.item()

def optimize(model, data_loader : DataLoader, optimizer, epochs=100, buffer_size = 20):
    stdout_buffer = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            #print("x_batch device: ",x_batch.device) 
            loss = train_step(model, x_batch, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)
        
        line = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, time: {time.time() - start_time:.3f}"
        stdout_buffer.append(line)
        if len(stdout_buffer) <= buffer_size:
            print(line)
        else:
            stdout_buffer.pop(0)
            sys.stdout.write("\033[F" * len(stdout_buffer))
            for l in stdout_buffer:
                sys.stdout.write("\033[K")
                print(l)
