from itertools import product
import math
import torch
import torch.fft
from functools import reduce

Polynomial = torch.Tensor

def binomial(n, k):
    return math.comb(n, k)


def combinations(n, k, dtype=torch.double, device='cpu'):
    n = torch.as_tensor(n, dtype=dtype, device=device)
    k = torch.as_tensor(k, dtype=dtype, device=device)
    
    # Create a mask for valid k values (0 <= k <= n)
    valid_mask = (k >= 0) & (k <= n)
    
    # lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    log_comb = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    
    # Use the mask to handle invalid k values, which should result in 0
    # exp(log_comb) for valid cases, 0 otherwise
    # We round to handle potential floating point inaccuracies since combinations should be integers
    return torch.where(valid_mask, torch.exp(log_comb).round(), torch.zeros_like(log_comb))

def bernstein_to_monomial(bernstein_coeffs: torch.Tensor) -> torch.Tensor:
    if not isinstance(bernstein_coeffs, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    d = bernstein_coeffs.ndim
    monomial_coeffs = bernstein_coeffs.clone()
    dtype = bernstein_coeffs.dtype
    device = bernstein_coeffs.device
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for i in range(d):
        n = monomial_coeffs.shape[i] - 1
        if n < 0:
            continue

        # Build the Bernstein-to-Monomial conversion matrix M
        # The (k, j) entry is C(n, k) * C(k, j) * (-1)^(k-j)
        k_ = torch.arange(n + 1, device=device, dtype=dtype).view(-1, 1) # Row index
        j_ = torch.arange(n + 1, device=device, dtype=dtype).view(1, -1) # Col index
        
        mask = (j_ <= k_).float()
        
        comb_nk = combinations(n, k_, dtype=dtype, device=device)
        comb_kj = combinations(k_, j_, dtype=dtype, device=device)
        
        signs = torch.pow(-1.0, k_ - j_)
        
        M = comb_nk * comb_kj * signs * mask
        M = M.to(dtype)

        # Apply the transformation using einsum for clarity and efficiency
        # e.g., for dim 0 of a 3D tensor: 'abc,ka->kbc'
        subscripts = list(alphabet[:d])
        contract_idx = subscripts[i]
        new_idx = alphabet[d] 
        
        mat_subscripts = f"{new_idx}{contract_idx}"
        
        result_subscripts = list(subscripts)
        result_subscripts[i] = new_idx
        
        einsum_str = f"{''.join(subscripts)},{mat_subscripts}->{''.join(result_subscripts)}"
        
        monomial_coeffs = torch.einsum(einsum_str, monomial_coeffs, M)
        
    return monomial_coeffs

def monomial_to_bernstein(monomial_coeffs: torch.Tensor) -> torch.Tensor:
    if not isinstance(monomial_coeffs, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    d = monomial_coeffs.ndim
    bernstein_coeffs = monomial_coeffs.clone()
    dtype = monomial_coeffs.dtype
    device = monomial_coeffs.device

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(d):
        n = bernstein_coeffs.shape[i] - 1
        if n < 0:
            continue

        # Build the Monomial-to-Bernstein conversion matrix N
        # The (j, k) entry is C(j, k) / C(n, k)
        j_ = torch.arange(n + 1, device=device, dtype=dtype).view(-1, 1) # Row index
        k_ = torch.arange(n + 1, device=device, dtype=dtype).view(1, -1) # Col index

        mask = (k_ <= j_).float()
        
        comb_jk = combinations(j_, k_, dtype=dtype, device=device)
        comb_nk = combinations(n, k_, dtype=dtype, device=device)
        
        # Add a small epsilon to denominator to prevent division by zero
        # This is a safeguard; C(n, k) should be non-zero for k <= n.
        N = (comb_jk / (comb_nk + 1e-12)) * mask
        N = N.to(dtype)

        # Apply the transformation using einsum
        subscripts = list(alphabet[:d])
        contract_idx = subscripts[i]
        new_idx = alphabet[d]
        
        mat_subscripts = f"{new_idx}{contract_idx}"
        
        result_subscripts = list(subscripts)
        result_subscripts[i] = new_idx
        
        einsum_str = f"{''.join(subscripts)},{mat_subscripts}->{''.join(result_subscripts)}"
        
        bernstein_coeffs = torch.einsum(einsum_str, bernstein_coeffs, N)

    return bernstein_coeffs





def create_d_separable_tensor(index_fcn, shape):
    # Compute g values for each dimension
    g_vectors = [torch.tensor([index_fcn(d, i) for i in range(shape[d])], dtype=torch.float32) for d in range(len(shape))]

    # Broadcasting multiplication (outer product)
    result = torch.ones(shape)
    for dim, g_vec in enumerate(g_vectors):
        broadcast_shape = [1] * len(shape)
        broadcast_shape[dim] = shape[dim]
        result *= g_vec.reshape(broadcast_shape)
    
    return result

def integrate_weighted_product(p_list : list[Polynomial], q : Polynomial, bernstein_basis = False):
    """
    Compute the integral over x in the unit hypercube of q(x) * \prod_l p_l(y, x)
    Returns R(y) as a coefficient tensor in y

    Args:
        p_list: list of tensors, each of shape (... y_dims ..., ... x_dims ...)
                with d_y + d_q total dimensions
        q:      tensor of shape (x_1_deg+1, ..., x_dq_deg+1)
        bernstein_basis: assuming p & q are in Bernstein basis, perform operations in the Bernstein basis
    Returns:
        R: tensor of shape (sum of y-degrees + 1)
    """

    d_q = q.ndim                        # number of x dimensions
    d_total = p_list[0].ndim           # total number of dimensions in p
    d_y = d_total - d_q                # number of y dimensions


    for i, p in enumerate(p_list):
        if p.ndim != d_total:
            raise ValueError(f"All p_i must have the same number of dims: {p.ndim} vs {d_total}")

    # Determine output polynomial degree shape in y
    y_shapes = [p.shape[:d_y] for p in p_list]
    out_y_shape = [sum(s[k] - 1 for s in y_shapes) + 1 for k in range(d_y)]

    # Determine x convolution shape
    x_shapes = [p.shape[d_y:] for p in p_list]
    q_shape = q.shape
    S = [sum(s[k] - 1 for s in x_shapes) + q_shape[k] for k in range(d_q)]

    fft_shape = out_y_shape + S
    fft_dims = tuple(range(d_y + d_q))  # full fft over y and x

    if bernstein_basis:
        for k in range(len(p_list)):
            p = p_list[k]

            # Pre weight each bernstein tensor by the binomial coeff
            W_pre = create_d_separable_tensor(lambda d, i : binomial(p.shape[d] - 1, i), p.shape)
            p_list[k] = W_pre * p 
        
        W_pre = create_d_separable_tensor(lambda d, i : binomial(q.shape[d] - 1, i), q.shape)
        q = W_pre * q 

        print("weighted p: ", p_list)
        print("weighted q: ", q)


    # FFT of each padded p_i
    P_fft = []
    for p, y_shape, x_shape in zip(p_list, y_shapes, x_shapes):
        pad = []
        for k in reversed(range(d_q)):
            pad.extend([0, S[k] - x_shape[k]])
        for k in reversed(range(d_y)):
            pad.extend([0, out_y_shape[k] - y_shape[k]])
        p_padded = torch.nn.functional.pad(p, pad)
        P_fft.append(torch.fft.rfftn(p_padded, s=fft_shape, dim=fft_dims))

    # Pad and FFT q
    q = q.view((1,) * d_y + q.shape)
    pad_q = []
    for k in reversed(range(d_q)):
        pad_q.extend([0, S[k] - q_shape[k]])
    for k in reversed(range(d_y)):
        pad_q.extend([0, out_y_shape[k] - 1])  # pad to size
    q_padded = torch.nn.functional.pad(q, pad_q)
    Q_fft = torch.fft.rfftn(q_padded, s=fft_shape, dim=fft_dims)

    # Multiply in frequency domain
    F = Q_fft
    for P in P_fft:
        F = F * P

    # Inverse FFT to get coefficients
    C = torch.fft.irfftn(F, s=fft_shape, dim=fft_dims)

    # Compute integration weights over x
    #grids = torch.meshgrid(*[
    #    torch.arange(S[k], dtype=C.dtype, device=C.device) for k in range(d_q)
    #], indexing='ij')
    #print(grids)
    #W = torch.ones_like(C)
    #for i, g in enumerate(grids):
    #    # broadcast to match last d_q dimensions of C
    #    W *= 1.0 / (g + 1.0)

    # Choose the sum-index post-weighting function based on the basis
    if bernstein_basis:
        all_shapes = [p.shape for p in p_list]
        q_shape_combined = [1 for _ in range(d_y)] + list(q_shape)
        #print("q_shape_combined: ", q_shape_combined)
        S_all = [sum(s[k] - 1 for s in all_shapes) + q_shape_combined[k] for k in range(d_total)]
        #print("S: ", S, " out_y_shapes: ", out_y_shape, " S_all: ", S_all)

        W_denom = create_d_separable_tensor(lambda d, s : 1.0 / binomial(S_all[d] - 1, s), S_all)
        print("W_denom: ", 1.0 / W_denom)
        integral_value = 1.0
        for total_x_deg in S:
            integral_value /= (total_x_deg + 1.0)
        W = integral_value * W_denom
        print("bern computation C unwghtd: \n", (C).round().int())
        print("bern computation C: \n", (C * W_denom).round().int())
        #print("C: ", C)
        #print("C shape: ", C.shape)
        #print("integ value: ", integral_value)
        #print("W: ", W)
        #print("W shape: ", W.shape)
    else:
        print("pwr computation  C: \n", (C).round().int())
        print("pwr computation in bern C: \n", monomial_to_bernstein(C).round().int())
        W_x = create_d_separable_tensor(lambda d, s : 1.0 / (s + 1.0), S)
        for _ in range(d_y):
            W_x.unsqueeze(-1)
        W = W_x.expand(out_y_shape + S)


    # Sum over x dimensions (last d_q dims)
    R = (C * W).sum(dim=tuple(range(-d_q, 0)))
    return R


if __name__ == "__main__":

    #p_pwr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    #p_bern = monomial_to_bernstein(p_pwr)
    #print("Power: ", p_pwr)
    #print("Bernstein: ", p_bern)
    #print("Power: ", bernstein_to_monomial(p_bern))

    ## Bernstein basis
    #p_list = [
    #    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
    #    torch.tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.float32)
    #]

    #q = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

    #r = integrate_weighted_product(p_list, q, bernstein_basis=True)
    #r_pwr = bernstein_to_monomial(r)
    #print("Bernstein basis computation: ", r)

    ## Power basis
    #p_list_pwr = [bernstein_to_monomial(p) for p in p_list]
    #q_pwr = bernstein_to_monomial(q)

    #r_pwr = integrate_weighted_product(p_list_pwr, q_pwr, bernstein_basis=False)
    #print("Power basis computation: ", r_pwr)


    # Bernstein basis
    p_list = [
        torch.tensor([1, 2, 3], dtype=torch.float32),
    ]

    q = torch.tensor([2, 3, 4, 5], dtype=torch.float32)

    r = integrate_weighted_product(p_list, q, bernstein_basis=True)
    r_pwr = bernstein_to_monomial(r)
    print("Bernstein basis computation: ", r)
    print("\n")

    # Power basis
    p_list_pwr = [bernstein_to_monomial(p) for p in p_list]
    q_pwr = bernstein_to_monomial(q)

    r_pwr = integrate_weighted_product(p_list_pwr, q_pwr, bernstein_basis=False)
    print("Power basis computation: ", r_pwr)

