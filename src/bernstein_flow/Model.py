import itertools
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from sympy import symbols, binomial, lambdify
import numpy as np
import time
import sys
import gc
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import issparse

from .Polynomial import Polynomial, Basis, decasteljau_composition
from .HyperProjection import bernstein_raised_degree_tf


class BernsteinFlowModel(torch.nn.Module):
    def __init__(self, dim : int, degrees : list[int], layers : int = 1, deg_incr : list[int] = None, device = None, dtype = torch.float32, sparse_di=True):
        """
        Create a BFM simple density estimation model

        Args:
            dim : dimension of the support
            degrees : dim-length list of degrees for each dimension (applied to all components of g)
            layers : number of compositional layers to use. If `layers == 1`, avoids using compositional operations
            deg_incr : degree increase for each dimension. If None, avoids using deg incr training
            device : device to store tensors on
            dtype : data type of tensors
            sparse_di : force degree-increase matrices to be stored in sparse format
        """
        super().__init__()

        self.dim = dim

        assert len(degrees) == dim

        self.degrees = torch.tensor(degrees)
        self.n_layers = layers
        
        if deg_incr is not None:
            assert len(deg_incr) == dim
        self.deg_incr = deg_incr

        self.constrained = True if deg_incr is None else False

        self.device = device
        self.dtype = dtype

        # Parameters
        self.layers = torch.nn.ModuleList([torch.nn.ParameterList() for _ in range(self.n_layers)])

        for i in range(dim):
            tf_deriv_degrees = self.degrees[:i + 1].clone()
            tf_deriv_degrees[i] -= 1

            # Number of coefficients in each polynomial (flattened tensor)
            poly_size = torch.prod(tf_deriv_degrees + 1).item()
            

            for param_list in self.layers:
                unconstrained_param_mat = torch.nn.Parameter(torch.rand(poly_size, dtype=self.dtype, device=self.device)) 
                param_list.append(unconstrained_param_mat)
        
            if self.deg_incr is not None:

                original_shape = (tf_deriv_degrees + 1).tolist()
                deg_incr_shape = [og_shape + self.deg_incr[j] for j, og_shape in enumerate(original_shape)]
                di_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape, sparse=sparse_di).A
                di_np_sparse = issparse(di_np)


                if di_np_sparse:
                    di_np_coo = di_np.tocoo()
                    values = torch.FloatTensor(di_np_coo.data)
                    indices = torch.LongTensor(np.vstack((di_np_coo.row, di_np_coo.col)))
                    shape = torch.Size(di_np_coo.shape)
                    sparse_di_mat = torch.sparse_coo_tensor(indices=indices, values=values, size=shape).to(dtype=self.dtype, device=self.device)
                    self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                else:
                    dense_di_mat = torch.from_numpy(di_np).to(dtype=self.dtype, device=self.device)
                    n_zeros = torch.sum(dense_di_mat == 0).item()
                    sparsity = n_zeros / dense_di_mat.numel()
                    print(f"DI matrix sparsity {sparsity * 100:.2f}%")
                    if sparsity > 0.7:
                        print("Using sparse matrix for dimension ", i)
                        sparse_di_mat = dense_di_mat.to_sparse_coo()
                        self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                    else:
                        print("Using dense matrix for dimension ", i)
                        self.register_buffer(f"deg_incr_{i}", dense_di_mat)
        
        self.input_dims = list(range(dim))
    
    
    def n_parameters(self):
        n_params = 0
        for layer in self.layers:
            for alpha_matrix in layer:
                n_params += torch.numel(alpha_matrix)
        return n_params

    def forward(self, x : torch.Tensor):
        density = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)

        if self.n_layers == 1:
            for i in range(self.dim):
                tf_val = self.transformer_deriv(x, i, layer_i=0)
                density *= tf_val
        else:
            raise NotImplementedError()
            layer_input = x
            for layer_i in range(len(self.layers)):
                next_layer_input = []
                for i in range(self.dim):
                    tf_val = self.transformer_deriv(layer_input, i, layer_i=layer_i)
                    density *= tf_val
                    
                    # Compute the next input by moving the previous 'x' through the current transformer
                    y_i = self.transformer(layer_input, i, layer_i=layer_i)
                    next_layer_input.append(y_i)
                layer_input = torch.vstack(next_layer_input).t()
        return density
    
    
    def get_constrained_coeff_tensor(self, i : int, layer_i : int = 0, project_no_grad=False):

        # Unconstrained
        param_vec = self.layers[layer_i][i]        
        
        if self.constrained:
            # Make each coefficient positive to ensure invertibility
            param_vec = torch.nn.functional.softplus(param_vec)

        input_dim = self.input_dims[i]

        tensor_shape = self.degrees[:input_dim+1] + 1
        tensor_shape[input_dim] -= 1
        coeff_tensor = param_vec.reshape(tuple(tensor_shape))
        
        
        if self.constrained:
            normalizing_coeffs = self.degrees[input_dim] / coeff_tensor.sum(dim=input_dim, keepdim=True)
        else:
            # Perform normalization in the raised degree space for better numerical stability
            di = getattr(self, f"deg_incr_{i}")
            raised_deg_param_vec = torch.sparse.mm(di, param_vec.unsqueeze(1)) if di.is_sparse else di @ param_vec.unsqueeze(1)
            di_tensor_shape = self.degrees[:input_dim+1] + torch.tensor(self.deg_incr[:input_dim+1]) + 1
            di_tensor_shape[input_dim] -= 1
            coeff_tensor = raised_deg_param_vec.reshape(tuple(di_tensor_shape))
            normalizing_coeffs = (self.degrees[input_dim] + self.deg_incr[input_dim]) / torch.clamp(coeff_tensor.sum(dim=input_dim, keepdim=False), min=1e-12).unsqueeze(dim=input_dim)


        constrained_coeffs = coeff_tensor * normalizing_coeffs

        if not self.constrained and project_no_grad:
            with torch.no_grad():
                constrained_coeffs_vec = constrained_coeffs.reshape(di.shape[0], 1).detach()
                constrained_coeffs_vec = cg_projection(di, constrained_coeffs_vec)
                return constrained_coeffs_vec.reshape(tuple(tensor_shape))
        return constrained_coeffs

    def transformer_deriv(self, x : torch.Tensor, i : int, layer_i : int = 0):
        coeff_tensor = self.get_constrained_coeff_tensor(i, layer_i=layer_i)

        return self.decasteljau_torch(coeff_tensor, x[:, :self.input_dims[i]+1])
    
    def transformer(self, x : torch.Tensor, i : int, layer_i : int = 0):
        coeff_tensor = self.get_constrained_coeff_tensor(i, layer_i=layer_i)

        antideriv_coeff_tensor = self._antiderivative_torch(coeff_tensor, axis=i)

        return self.decasteljau_torch(antideriv_coeff_tensor, x[:, :self.input_dims[i]+1])

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
                tf_deriv_coeffs = self.get_constrained_coeff_tensor(i, layer_i=layer_i, project_no_grad=True)
                layer_i_tf_derivs.append(Polynomial(tf_deriv_coeffs, basis=Basis.BERN, dtype=dtype))

                if self.n_layers > 1:
                    antideriv_coeff_tensor = self._antiderivative_torch(tf_deriv_coeffs, axis=i)
                    layer_i_tfs.append(Polynomial(antideriv_coeff_tensor, basis=Basis.BERN, dtype=dtype))

            decomposed_tf_derivs.append(layer_i_tf_derivs)
            decomposed_tfs.append(layer_i_tfs)

        # If there is only a single layer, skip composition
        if self.n_layers == 1:
            return decomposed_tf_derivs[0]
        else:
            raise NotImplementedError()
            factors = decomposed_tf_derivs[0]
            input_polynomial_vec = decomposed_tfs[0]
            for layer_i in range(1, self.n_layers):
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
                        input_polynomial_vec[i] = decasteljau_composition(p, input_polynomial_vec[:i+1])
            return factors

    def nll_loss(self, data, hard_constraint = True):
        if self.constrained or hard_constraint:
            density = self(data)
            log_density = torch.log(density + 1e-10)
            loss = -log_density.mean()
            return loss
        else:
            density = self(data)
            log_density = torch.log(density + 1e-10)
            
            # Compute the constraint violation
            penalty = 0.0
            for i in range(self.dim):
                deg_incr_mat = getattr(self, f"deg_incr_{i}")
                params = self.layers[0][i]
                raised_deg_params = deg_incr_mat @ params
                violation = torch.clamp(-raised_deg_params, min=0.0)
                penalty += torch.sum(violation**2 + violation)

            loss = -log_density.mean() + penalty
            return loss


    def feasible_projection(self, max_iterations=50, tol=1e-2, min_thresh=1e-2):
        # Only project if model is unconstrained
        if not self.constrained:
            with torch.no_grad():
                for i in range(self.dim):
                    di = getattr(self, f"deg_incr_{i}")
                    params = self.layers[0][i].detach().clone().unsqueeze(1)
                    feasible = False

                    for iter in range(max_iterations):
                        raised_deg_params = torch.sparse.mm(di, params)
                        if torch.all(raised_deg_params >= min_thresh):
                            self.layers[0][i].copy_(params.squeeze())
                            feasible = True
                            break

                        # Clamp raised degree parameters, then project back to original degree
                        print(f"Projection iteration {iter + 1} / {max_iterations}. Min value: {torch.min(raised_deg_params)} (clamp: {min_thresh + iter * tol}, thresh: {min_thresh})")
                        raised_deg_params = torch.clamp(raised_deg_params, min=(min_thresh + iter * tol))
                        params = cg_projection(di, raised_deg_params)
                    if not feasible:

                        raised_deg_params = torch.sparse.mm(di, params)
                        min_val = torch.min(raised_deg_params).item()
                        print(f"Cound not find feasible projection for transformer {i} after {max_iterations} iterations. Min raised degree param {min_val} is below {min_thresh}")
                        return False
        return True
    
    def get_raised_degree_params(self, i : int, layer_i : int = 0):
        if self.constrained:
            raise ValueError("Model was not given degree increase")
        deg_incr_mat = getattr(self, f"deg_incr_{i}")
        with torch.no_grad():
            params = self.get_constrained_coeff_tensor(i, layer_i=layer_i).reshape(-1)
        raised_deg_params = deg_incr_mat @ params

        tensor_shape = self.degrees[:i+1] + torch.tensor(self.deg_incr[:i+1]) + 1
        tensor_shape[i] -= 1
        coeff_tensor = raised_deg_params.reshape(tuple(tensor_shape))
        return coeff_tensor
        

    def decasteljau_torch(self, coeffs : torch.Tensor, x : torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"Input tensor x must be 2-dimensional (batch_size, d), but got {x.ndim} dimensions.")

        batch_size, d = x.shape

        assert coeffs.ndim == d, "x vector dimension does not match dimension of p"

        degrees = [s - 1 for s in coeffs.shape]

        expand_shape = (batch_size,) + coeffs.shape
        current_coeffs = coeffs.expand(*expand_shape) 

        for i in range(d):
            t = x[:, i]
            view_shape = [batch_size] + [1] * (d - i)
            t = t.reshape(*view_shape)

            degree = degrees[i]

            for _ in range(degree):
                current_coeffs = (
                    (1 - t) * current_coeffs[:, :-1, ...] +
                    t * current_coeffs[:, 1:, ...]
                )
            
            if i < d - 1:
                current_coeffs = current_coeffs.squeeze(dim=1)
        return current_coeffs.squeeze()
    
    def _antiderivative_torch(self, coeffs : torch.Tensor, axis : int):
        deg = coeffs.shape[axis] - 1

        sum = torch.cumsum(coeffs, dim=axis)
        new_shape = list(coeffs.shape)
        new_shape[axis] += 1
        antiderivative_coeffs = torch.zeros(new_shape, dtype=coeffs.dtype, device=coeffs.device)

        slice_obj = [slice(None)] * coeffs.ndim
        slice_obj[axis] = slice(1, None)
        antiderivative_coeffs[tuple(slice_obj)] = sum

        return antiderivative_coeffs / (deg + 1)


class ConditionalBernsteinFlowModel(BernsteinFlowModel):
    def __init__(self, dim : int, 
                 conditional_dim : int, 
                 degrees : list[int], 
                 conditional_degrees : list[int], 
                 layers : int = 1, 
                 deg_incr : list[int] = None, 
                 cond_deg_incr : list[int] = None, 
                 device = None, 
                 dtype = torch.float32, 
                 sparse_di=True):
        """
        Conditional flow model for p(x | y). The data must be supplied IN THE FORM [y, x] to evaluation/training
        """
        torch.nn.Module.__init__(self)

        self.dim = dim
        self.cond_dim = conditional_dim

        assert len(degrees) == dim
        assert len(conditional_degrees) == conditional_dim

        self.degrees = torch.tensor(conditional_degrees + degrees)
        self.n_layers = layers

        if deg_incr is not None:
            assert len(deg_incr) == dim
            self.deg_incr = cond_deg_incr + deg_incr
        else:
            self.deg_incr = None

        self.constrained = True if deg_incr is None else False

        self.device = device
        self.dtype = dtype

        # Parameters
        self.layers = torch.nn.ModuleList([torch.nn.ParameterList() for _ in range(self.n_layers)])
        for i in range(dim):
            tf_deriv_degrees = self.degrees[:i + 1 + self.cond_dim].clone()
            tf_deriv_degrees[i + self.cond_dim] -= 1

            poly_size = torch.prod(tf_deriv_degrees + 1).item()

            for param_list in self.layers:
                unconstrained_param_mat = torch.nn.Parameter(torch.rand(poly_size, dtype=self.dtype, device=self.device)) 
                param_list.append(unconstrained_param_mat)
        
            if self.deg_incr is not None:

                original_shape = (tf_deriv_degrees + 1).tolist()
                deg_incr_shape = [og_shape + self.deg_incr[j] for j, og_shape in enumerate(original_shape)]
                di_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape, sparse=sparse_di).A
                di_np_sparse = issparse(di_np)

                if di_np_sparse:
                    di_np_coo = di_np.tocoo()
                    values = torch.FloatTensor(di_np_coo.data)
                    indices = torch.LongTensor(np.vstack((di_np_coo.row, di_np_coo.col)))
                    shape = torch.Size(di_np_coo.shape)
                    sparse_di_mat = torch.sparse_coo_tensor(indices=indices, values=values, size=shape).to(dtype=self.dtype, device=self.device)
                    self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                else:
                    dense_di_mat = torch.from_numpy(di_np).to(dtype=self.dtype, device=self.device)
                    n_zeros = torch.sum(dense_di_mat == 0).item()
                    sparsity = n_zeros / dense_di_mat.numel()
                    print(f"DI matrix sparsity {sparsity * 100:.2f}%")
                    if sparsity > 0.7:
                        print("Using sparse matrix for dimension ", i)
                        sparse_di_mat = dense_di_mat.to_sparse_coo()
                        self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                    else:
                        print("Using dense matrix for dimension ", i)
                        self.register_buffer(f"deg_incr_{i}", dense_di_mat)

        self.input_dims = list(range(conditional_dim, conditional_dim + dim))



def train_step(model, x_data, optimizer, hard_constraint = True,
             proj_max_iterations=50,
             proj_tol=1e-2,
             proj_min_thresh=1e-2):

    model.train()
    optimizer.zero_grad()
    
    # Backup parameters
    if not model.constrained:
        param_backup = [[params.detach().clone() for params in layer]  for layer in model.layers]

    loss = model.nll_loss(x_data, hard_constraint=hard_constraint)
    loss.backward()
    optimizer.step()


    if hard_constraint:
        success = model.feasible_projection(max_iterations=proj_max_iterations, tol=proj_tol, min_thresh=proj_min_thresh)

        if not success:
            with torch.no_grad():
                for layer, layer_backup in zip(model.layers, param_backup):
                    for param, backup in zip(layer, layer_backup):
                        param.copy_(backup)

    return loss.item()

def optimize(model, data_loader : DataLoader, optimizer, epochs=100, train_with_hard_constraint = False, 
             proj_max_iterations=50,
             proj_tol=1e-2,
             proj_min_thresh=1e-2,
             log_buffer_size = 20):

    stdout_buffer = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss = train_step(model, x_batch, optimizer, hard_constraint=train_with_hard_constraint)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)
        
        line = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, time: {time.time() - start_time:.3f}"
        stdout_buffer.append(line)
        if len(stdout_buffer) <= log_buffer_size:
            print(line)
        else:
            stdout_buffer.pop(0)
            sys.stdout.write("\033[F" * len(stdout_buffer))
            for l in stdout_buffer:
                sys.stdout.write("\033[K")
                print(l)
    
    # Do a feasible projection at the end of training to make sure the model is a valid distribution
    if not train_with_hard_constraint:
        print("Projecting model to feasible space...")
        success = model.feasible_projection(max_iterations=proj_max_iterations, tol=proj_tol, min_thresh=proj_min_thresh)
        if not success:
            raise RuntimeError("Model projection failed after training")
        else:
            print("Success!")
    

