import itertools
import torch
import torch.fft
from functools import reduce
import operator

def compute_weighted_convolution(alpha_list : list, scale_fcns : list, h_fcn):
    """
    alpha_list: list of 1D torch tensors [alpha_1, ..., alpha_k]
    scale_fcns: list of functions [g1, ..., gk] to apply element-wise to alpha_1, ..., alpha_k
    h_fcn: function h(s) applied to index sum s
    """
    k = len(alpha_list)
    assert len(scale_fcns) == k, "Mismatch in number of alpha vectors and g functions"

    # Step 1: Scale each alpha vector by its g_l
    scaled = []
    for alpha, g in zip(alpha_list, scale_fcns):
        indices = torch.arange(len(alpha), device=alpha.device)
        scaled.append(alpha * g(indices))

    # Step 2: Convolve all scaled vectors using FFT
    def fft_convolve(a, b):
        n = len(a) + len(b) - 1
        n_fft = 2 ** (n - 1).bit_length()  # next power of 2
        A = torch.fft.rfft(a, n=n_fft)
        B = torch.fft.rfft(b, n=n_fft)
        return torch.fft.irfft(A * B, n=n_fft)[:n]

    conv_result = scaled[0]
    for vec in scaled[1:]:
        conv_result = fft_convolve(conv_result, vec)
    
    print("conv result: ", conv_result)

    # Step 3: Apply 1/h(s) weighting and dot product
    s_vals = torch.arange(len(conv_result), device=conv_result.device)
    h_vals = h_fcn(s_vals)
    result = torch.sum(conv_result / h_vals)

    return result

#def batched_weighted_convolution(alpha_matrices, g_list, h_fcn):
#    """
#    alpha_matrices: list of matrices alpha_l of shape (n_l, m_l)
#    g_list: list of functions gl(i) to apply element-wise to each row of alpha_l (must be matrix-input compatible)
#    h_fcn: function h(s) applied to sum index
#    Returns: tensor of shape (m_1, ..., m_k)
#    """
#    k = len(alpha_matrices)
#    m_list = [mat.shape[1] for mat in alpha_matrices]
#    n_list = [mat.shape[0] for mat in alpha_matrices]
#
#    # Total output length after convolution
#    total_len = sum(n_list) - (k - 1)
#    n_fft = 2 ** ((total_len - 1).bit_length())
#
#    # Step 1: Scale each alpha_l column by gl(i)
#    fft_list = []
#    for l in range(k):
#        alpha = alpha_matrices[l]  # shape (n_l, m_l)
#        g_vals = g_list[l](torch.arange(alpha.shape[0], device=alpha.device)).unsqueeze(1)
#        scaled = alpha * g_vals  # shape (n_l, m_l)
#
#        # Pad to FFT length and compute FFT for each column
#        padded = torch.zeros((n_fft, alpha.shape[1]), device=alpha.device)
#        padded[:alpha.shape[0]] = scaled
#        fft = torch.fft.rfft(padded, dim=0)  # shape (n_fft//2+1, m_l)
#        fft_list.append(fft)
#
#    # Step 2: Compute all FFT product combinations
#    # We'll broadcast all k dimensions into a tensor of shape (n_fft//2+1, m_1, ..., m_k)
#    fft_grid = fft_list[0].unsqueeze(2)  # shape (freq, m1, 1, 1, ...)
#    for l in range(1, k):
#        shape = [1] * (l + 1) + [fft_list[l].shape[1]] + [1] * (k - l - 1)
#        fft_grid = fft_grid * fft_list[l].reshape((fft_list[l].shape[0],) + tuple(shape[1:]))
#
#    # Step 3: Inverse FFT over freq axis to get convolutions
#    convs = torch.fft.irfft(fft_grid, n=n_fft, dim=0)[:total_len]  # shape (total_len, m1, ..., mk)
#
#    # Step 4: Weight with 1 / h(s) and sum over s
#    s_vals = torch.arange(total_len, device=convs.device).view(-1, *([1] * k))  # shape (total_len, 1, ..., 1)
#    weighted = convs / h_fcn(s_vals)  # shape (total_len, m1, ..., mk)
#    result = weighted.sum(dim=0)     # shape (m1, ..., mk)
#
#    return result

def batched_multidim_weighted_convolution(alpha_tensors, g_list, h_fn):
    """
    alpha_tensors: list of tensors αₗ of shape (n₁, ..., n_d, mₗ)
    g_l_list: list of functions [g₁, ..., g_k], each gₗ: ℕ → ℝ
              used as gₗ(i₁) * gₗ(i₂) * ... * gₗ(i_d)
    h_fn: single function h: ℕ → ℝ
          used as ∏ⱼ h(sⱼ) for summed multi-index s = (s₁, ..., s_d)
    Returns:
        result tensor of shape (m₁, ..., m_k)
    """
    k = len(alpha_tensors)
    shapes = [alpha.shape[:-1] for alpha in alpha_tensors]
    m_list = [alpha.shape[-1] for alpha in alpha_tensors]
    d = len(shapes[0])  # spatial dimensions
    device = alpha_tensors[0].device

    # Determine full shape for convolution result
    full_shape = [sum(s[i] for s in shapes) - (k - 1) for i in range(d)]
    fft_shape = list(full_shape)
    fft_shape[-1] = 2 ** (full_shape[-1] - 1).bit_length()

    fft_tensors = []
    for l in range(k):
        alpha = alpha_tensors[l]
        spatial_shape = alpha.shape[:-1]
        m_l = alpha.shape[-1]
        g_fn = g_list[l]

        # Apply gₗ(i₁) * gₗ(i₂) * ... using broadcasting
        g_vals = torch.ones(spatial_shape, device=device)
        for dim, n in enumerate(spatial_shape):
            idx = torch.arange(n, device=device)
            shape = [1] * d
            shape[dim] = -1
            g_vals *= g_fn(idx).reshape(*shape)

        scaled = alpha * g_vals.unsqueeze(-1)

        # Pad to fft_shape
        pad = [(0, fft_shape[i] - spatial_shape[i]) for i in range(d)]
        pad_flat = [p for pair in reversed(pad) for p in pair]
        padded = torch.nn.functional.pad(scaled, pad=pad_flat)

        fft = torch.fft.rfftn(padded, s=fft_shape, dim=tuple(range(d)))
        fft_tensors.append(fft)

    # Multiply FFTs across all m₁ × ... × m_k combinations
    fft_prod = fft_tensors[0].unsqueeze(-1)  # start with (..., m0, 1, 1, ..., 1)
    for l in range(1, k):
        num_prev_m_dims = fft_prod.ndim - fft_tensors[l].ndim
        fft_l_expanded = fft_tensors[l].unsqueeze(-1)
        for _ in range(num_prev_m_dims - 1):
            fft_l_expanded = fft_l_expanded.unsqueeze(-1)
        fft_prod = fft_prod * fft_l_expanded

    # Inverse FFT to get real-space convolution result
    convolved = torch.fft.irfftn(fft_prod, s=fft_shape, dim=tuple(range(d)))

    # Crop to full_shape
    slices = tuple(slice(0, full_shape[i]) for i in range(d))
    convolved = convolved[slices]

    # h(s₁, ..., s_d) = ∏ h(s_j) for same h across all dims
    h_vals = torch.ones(full_shape, device=device)
    for dim, n in enumerate(full_shape):
        idx = torch.arange(n, device=device)
        shape = [1] * d
        shape[dim] = -1
        h_vals *= h_fn(idx).reshape(*shape)

    #h_vals = h_vals.unsqueeze(-1).expand_as(convolved[..., 0])

    # Get number of polynomial axes in convolved tensor
    num_m_dims = convolved.ndim - len(full_shape)

    # Reshape h_vals to (..., 1, ..., 1) to match convolved
    h_vals = h_vals.reshape(*full_shape, *[1] * num_m_dims)
    result = (convolved / h_vals).sum(dim=tuple(range(d)))  # shape: (m₁, ..., m_k)
    return result

if __name__ == "__main__":
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([3.0, 4.0, 7.0])
    #v3 = torch.tensor([8.0, 9.0, 0.0, 10.0, 11.0])

    scale_fcns = [
        lambda i: 1,
        lambda i: 1,
        #lambda i: 1,
    ]

    # Power basis integration
    h_fcn = lambda s: (s + 1)

    #result = compute_weighted_convolution([v1, v2, v3], scale_fcns, h_fcn)
    result = compute_weighted_convolution([v1, v2], scale_fcns, h_fcn)
    print("result: ", result)


    alpha_1 = torch.tensor([
        [[1, 4, 6], [2, 5 ,7], [3, 6, 9]], 
        [[2, 3, 4], [5, 6 ,7], [8, 9, 10]], 
    ]) # (n1=3, n2=3, m1=2)

    alpha_2 = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]], 
    ]) # (n1=3, n2=3, m1=2)


    g1 = lambda i: torch.ones(i.shape)
    g2 = lambda i: torch.ones(i.shape)
    h_fn = lambda s: s + 1.0 

    result = batched_multidim_weighted_convolution([alpha_1, alpha_2], [g1, g2], h_fn)
    print(result)
