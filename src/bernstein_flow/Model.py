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

def bernstein_basis_functions(dim : int, degrees : list[int], coefficients : torch.Tensor = None, scale : float = 1.0):
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
    degree_ranges = [range(deg + 1) for deg in degrees]
    multi_indices = list(itertools.product(*degree_ranges))
    
    basis_funcs = []
    for flat_idx, alpha in enumerate(multi_indices):
        B = 1
        for i in range(dim):
            coeff = float(math.comb(degrees[i], alpha[i]))
            B *= coeff * x_syms[i]**alpha[i] * (1 - x_syms[i])**(degrees[i] - alpha[i])
        
        # Incorporate scale/coefficients
        B *= scale
        if coefficients is not None:
            B *= coefficients[flat_idx]

        basis_funcs.append(lambdify(x_syms, B, modules='torch'))
    
    return basis_funcs

def cg_projection(A : torch.Tensor, vec : torch.Tensor):
    sparse = A.is_sparse
    def matvec(x : np.ndarray):
        x_tch = torch.from_numpy(x).to(dtype=A.dtype, device=A.device).unsqueeze(1)
        #product = torch.sparse.mm(A, x_tch)
        if sparse:
            y = torch.sparse.mm(A, x_tch)
            product = torch.sparse.mm(A.t(), y)
        else:
            product = A.t() @ A @ x_tch
        return product.cpu().numpy()
    
    At_b = torch.sparse.mm(A.t(), vec).cpu().numpy() if sparse else torch.mv(A.t(), vec).cpu().numpy()

    lin_op = LinearOperator(shape=(A.shape[1], A.shape[1]), matvec=matvec, dtype=np.float64)
    x_np, info = cg(lin_op, b=At_b)

    return torch.from_numpy(x_np).to(dtype=A.dtype, device=A.device).unsqueeze(1)


class BernsteinFlowModel(torch.nn.Module):
    def __init__(self, dim : int, degrees : list[int], layers : int = 1, deg_incr : list[int] = None, device = None, dtype = torch.float32, sparse_di=True):
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
                deg_incr_shape = [og_shape + deg_incr[i] for og_shape in original_shape]
                di_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape, sparse=sparse_di).A
                print("DI size: ", di_np.shape)
                di_np_sparse = issparse(di_np)


                if di_np_sparse:
                    di_np_coo = di_np.tocoo()
                    values = torch.FloatTensor(di_np_coo.data)
                    indices = torch.LongTensor(np.vstack((di_np_coo.row, di_np_coo.col)))
                    shape = torch.Size(di_np_coo.shape)
                    sparse_di_mat = torch.sparse_coo_tensor(indices=indices, values=values, size=shape).to(dtype=self.dtype, device=self.device)
                    #print("   sparse di mat shape: ", sparse_di_mat.shape)
                    self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                else:
                    dense_di_mat = torch.from_numpy(di_np).to(dtype=self.dtype, device=self.device)
                    n_zeros = torch.sum(dense_di_mat == 0).item()
                    sparsity = n_zeros / dense_di_mat.numel()
                    print(f"DI matrix sparsity {sparsity * 100:.2f}%")
                    #dense_mpsi_mat = torch.from_numpy(np.linalg.pinv(deg_incr_matrix_np)).to(dtype=self.dtype, device=self.device)
                    if sparsity > 0.7:
                        print("Using sparse matrix for dimension ", i)
                        sparse_di_mat = dense_di_mat.to_sparse_coo()
                        self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                    else:
                        print("Using dense matrix for dimension ", i)
                        self.register_buffer(f"deg_incr_{i}", dense_di_mat)

                #self.register_buffer(f"mpsi_{i}", torch.from_numpy(np.linalg.pinv(deg_incr_matrix_np)).to(dtype=self.dtype, device=self.device))
        
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
            assert False, "Not implemented"
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
    
    def get_constrained_coeff_tensor(self, i : int, layer_i : int = 0):

        # Unconstrained
        param_vec = self.layers[layer_i][i]        
        
        if self.constrained:
            # Make each coefficient positive to ensure invertibility
            param_vec = torch.nn.functional.softplus(param_vec)

        #deg_incr_mat = getattr(self, f"deg_incr_{i}")
        #rd_params = deg_incr_mat @ param_vec
        #print("--------- min deg raise: ", torch.min(deg_incr_mat).item())
        #print("--------- min raised deg params: ", torch.min(rd_params).item())


        # Ensure that the antiderivative's range is [0, 1] by ensuring that all coefficients along the antiderivative axis add to the required value
        #print("Min normalizer: ", torch.min(torch.clamp(coeff_tensor.sum(dim=i, keepdim=True), min=1e-6)).item())
        #if torch.min(torch.clamp(coeff_tensor.sum(dim=i, keepdim=True), min=1e-12)).item() < 0.0:
        #    print("NORMALIZER NEGATIVE")
        #    assert False
        input_dim = self.input_dims[i]
        #if self.constrained:

        #print("input dim: ", input_dim)
        # Reshape to coefficient tensor
        tensor_shape = self.degrees[:input_dim+1] + 1
        tensor_shape[input_dim] -= 1
        coeff_tensor = param_vec.reshape(tuple(tensor_shape))
        #print("min normalizer: ", torch.min(coeff_tensor.sum(dim=input_dim, keepdim=True)).item())
        #normalizing_coeffs = self.degrees[input_dim] / coeff_tensor.sum(dim=input_dim, keepdim=True)
        
        
        if self.constrained:
            normalizing_coeffs = self.degrees[input_dim] / coeff_tensor.sum(dim=input_dim, keepdim=True)
        else:
            di = getattr(self, f"deg_incr_{i}")
            raised_deg_param_vec = torch.sparse.mm(di, param_vec.unsqueeze(1)) if di.is_sparse else di @ param_vec.unsqueeze(1)
            tensor_shape = self.degrees[:input_dim+1] + torch.tensor(self.deg_incr[:input_dim+1]) + 1
            tensor_shape[input_dim] -= 1
            coeff_tensor = raised_deg_param_vec.reshape(tuple(tensor_shape))
            normalizing_coeffs = (self.degrees[input_dim] + self.deg_incr[input_dim]) / torch.clamp(coeff_tensor.sum(dim=input_dim, keepdim=False), min=1e-12).unsqueeze(dim=input_dim)

        constrained_coeffs = coeff_tensor * normalizing_coeffs

        #di = getattr(self, f"deg_incr_{i}")
        #raised_deg_param_vec = torch.sparse.mm(di, param_vec.unsqueeze(1))
        #minv = torch.min(raised_deg_param_vec)
        #if minv < 0.0:
        #    print("min raised deg: ", minv)
        #    input("...")

        #else:
        #    # Reshape to raised degree coefficient tensor
        #    #deg_incr_mat = getattr(self, f"deg_incr_{i}")
        #    di = getattr(self, f"deg_incr_{i}")
        #    mpsi = getattr(self, f"mpsi_{i}")
        #    #print("Deg incr mat shape: ", deg_incr_mat.shape, " param vec shape: ", param_vec.shape)
        #    #print("param vec shape: ", param_vec.shape, " di shape: ", di.shape)
        #    raised_deg_param_vec = torch.sparse.mm(di, param_vec.unsqueeze(1))
        #    #raised_deg_param_vec = deg_incr_mat @ param_vec
        #    original_shape = raised_deg_param_vec.shape
        #    tensor_shape = self.degrees[:input_dim+1] + torch.tensor(self.deg_incr[:input_dim+1]) + 1
        #    tensor_shape[input_dim] -= 1
        #    coeff_tensor = raised_deg_param_vec.reshape(tuple(tensor_shape))
        #    constrained_coeffs = (self.degrees[input_dim] + self.deg_incr[input_dim]) * coeff_tensor / torch.clamp(coeff_tensor.sum(dim=input_dim, keepdim=True), min=1e-12)
        #    constrained_coeffs = constrained_coeffs.reshape(original_shape)

        #    #orig_deg_params = mpsi @ constrained_coeffs.reshape(-1)
        #    #orig_deg_params = torch.linalg.lstsq(di, constrained_coeffs.reshape(-1)).solution.squeeze()
        #    orig_deg_params = mpsi @ constrained_coeffs.reshape(-1)

        #    tensor_shape = self.degrees[:input_dim+1] + 1
        #    tensor_shape[input_dim] -= 1
        #    constrained_coeffs = orig_deg_params.reshape(tuple(tensor_shape))

        #constrained_coeffs = self.degrees[i] * torch.nn.functional.softmax(coeff_tensor, dim=i)

        #params = constrained_coeffs.reshape(-1)
        #rd_params = deg_incr_mat @ params
        #print("+++++++++ min raised deg params: ", torch.min(rd_params).item())


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
                tf_deriv_coeffs = self.get_constrained_coeff_tensor(i, layer_i=layer_i).detach().clone()
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
            assert False, "Not implemented"
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
                #print("min params: ", torch.min(params).item())
                raised_deg_params = deg_incr_mat @ params
                #print("min rd params: ", torch.min(raised_deg_params).item())
                violation = torch.clamp(-raised_deg_params, min=0.0)
                #print("violatioon: ", torch.sum(violation))
                penalty += torch.sum(violation**2 + violation)
                #input("...")

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

                    #params_nonpos = torch.any(params < 0)
                    for iter in range(max_iterations):
                        raised_deg_params = torch.sparse.mm(di, params)
                        #print(" -- i: ", i, " Raised deg params min val iter: ", iter, " : ", torch.min(raised_deg_params))
                        if torch.all(raised_deg_params >= min_thresh):
                            self.layers[0][i].copy_(params.squeeze())
                            feasible = True

                            #print("((((((((( min raised deg params b4: ", torch.min(raised_deg_params).item())
                            ##const_params = self.layers[0][i].detach().clone()
                            #const_params = self.get_constrained_coeff_tensor(i, layer_i=0).reshape(-1)
                            #rd_params = deg_incr_mat @ const_params.reshape(-1)
                            #print("))))))))) min raised deg params: ", torch.min(rd_params).item())
                            break

                        # Clamp raised degree parameters, then project back to original degree
                        print(f"Projection iteration {iter + 1} / {max_iterations}. Min value: {torch.min(raised_deg_params)} (clamp: {min_thresh + iter * tol}, thresh: {min_thresh})")
                        raised_deg_params = torch.clamp(raised_deg_params, min=(min_thresh + iter * tol))
                        #params_test = getattr(self, f"mpsi_{i}") @ raised_deg_params
                        #print("raised deg params shape: " , raised_deg_params.shape)
                        params = cg_projection(di, raised_deg_params)
                        #print("params test size: ", params_test, " params size: ", params)
                        #print("err vs mpsi:", torch.max(params_test - params))
                        #input("...")
                        #print("raised deg params: ", raised_deg_params)
                        #print("infeasible params: ", params)
                        #input("...")
                    if not feasible:
                        #params = torch.clamp(params, min=1e-6)
                        #self.layers[0][i].copy_(params)

                        raised_deg_params = torch.sparse.mm(di, params)
                        min_val = torch.min(raised_deg_params).item()
                        print(f"Cound not find feasible projection for transformer {i} after {max_iterations} iterations. Min raised degree param {min_val} is below {min_thresh}")
                        #input("...")
                        return False
        return True
    
    def get_raised_degree_params(self, i : int, layer_i : int = 0):
        if self.constrained:
            raise ValueError("Model was not given degree increase")
        deg_incr_mat = getattr(self, f"deg_incr_{i}")
        with torch.no_grad():
            #params = self.layers[layer_i][i].detach().clone()
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
                deg_incr_shape = [og_shape + self.deg_incr[i] for og_shape in original_shape]
                deg_incr_matrix_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape).A
                di_np = bernstein_raised_degree_tf(original_shape, deg_incr_shape, sparse=sparse_di).A
                print("DI size: ", di_np.shape)
                di_np_sparse = issparse(di_np)

                if di_np_sparse:
                    di_np_coo = di_np.tocoo()
                    values = torch.FloatTensor(di_np_coo.data)
                    indices = torch.LongTensor(np.vstack((di_np_coo.row, di_np_coo.col)))
                    shape = torch.Size(di_np_coo.shape)
                    sparse_di_mat = torch.sparse_coo_tensor(indices=indices, values=values, size=shape).to(dtype=self.dtype, device=self.device)
                    #print("   sparse di mat shape: ", sparse_di_mat.shape)
                    self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                else:
                    dense_di_mat = torch.from_numpy(di_np).to(dtype=self.dtype, device=self.device)
                    n_zeros = torch.sum(dense_di_mat == 0).item()
                    sparsity = n_zeros / dense_di_mat.numel()
                    print(f"DI matrix sparsity {sparsity * 100:.2f}%")
                    #dense_mpsi_mat = torch.from_numpy(np.linalg.pinv(deg_incr_matrix_np)).to(dtype=self.dtype, device=self.device)
                    if sparsity > 0.7:
                        print("Using sparse matrix for dimension ", i)
                        sparse_di_mat = dense_di_mat.to_sparse_coo()
                        self.register_buffer(f"deg_incr_{i}", sparse_di_mat)
                    else:
                        print("Using dense matrix for dimension ", i)
                        self.register_buffer(f"deg_incr_{i}", dense_di_mat)

                #self.register_buffer(f"deg_incr_{i}", torch.from_numpy(deg_incr_matrix_np).to(dtype=self.dtype, device=self.device))
                #self.register_buffer(f"mpsi_{i}", torch.from_numpy(np.linalg.pinv(deg_incr_matrix_np)).to(dtype=self.dtype, device=self.device)) # Left psuedo-inverse
        
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
            #print("x_batch device: ",x_batch.device) 
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
    

