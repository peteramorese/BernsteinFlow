import itertools
import torch
import torch.fft

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

def batched_weighted_convolution(alpha_matrices, g_list, h_fcn):
    """
    alpha_matrices: list of matrices αₗ of shape (n_l, m_l)
    g_list: list of functions gl(i) to apply element-wise to each row of alpha_l (must be matrix-input compatible)
    h_fcn: function h(s) applied to sum index
    Returns: tensor of shape (m_1, ..., m_k)
    """
    k = len(alpha_matrices)
    m_list = [mat.shape[1] for mat in alpha_matrices]
    n_list = [mat.shape[0] for mat in alpha_matrices]

    # Total output length after convolution
    total_len = sum(n_list) - (k - 1)
    n_fft = 2 ** ((total_len - 1).bit_length())

    # Step 1: Scale each alpha_l column by gl(i)
    fft_list = []
    for l in range(k):
        alpha = alpha_matrices[l]  # shape (n_l, m_l)
        g_vals = g_list[l](torch.arange(alpha.shape[0], device=alpha.device)).unsqueeze(1)
        scaled = alpha * g_vals  # shape (n_l, m_l)

        # Pad to FFT length and compute FFT for each column
        padded = torch.zeros((n_fft, alpha.shape[1]), device=alpha.device)
        padded[:alpha.shape[0]] = scaled
        fft = torch.fft.rfft(padded, dim=0)  # shape (n_fft//2+1, m_l)
        fft_list.append(fft)

    # Step 2: Compute all FFT product combinations
    # We'll broadcast all k dimensions into a tensor of shape (n_fft//2+1, m_1, ..., m_k)
    fft_grid = fft_list[0].unsqueeze(2)  # shape (freq, m1, 1, 1, ...)
    for l in range(1, k):
        shape = [1] * (l + 1) + [fft_list[l].shape[1]] + [1] * (k - l - 1)
        fft_grid = fft_grid * fft_list[l].reshape((fft_list[l].shape[0],) + tuple(shape[1:]))

    # Step 3: Inverse FFT over freq axis to get convolutions
    convs = torch.fft.irfft(fft_grid, n=n_fft, dim=0)[:total_len]  # shape (total_len, m1, ..., mk)

    # Step 4: Weight with 1 / h(s) and sum over s
    s_vals = torch.arange(total_len, device=convs.device).view(-1, *([1] * k))  # shape (total_len, 1, ..., 1)
    weighted = convs / h_fcn(s_vals)  # shape (total_len, m1, ..., mk)
    result = weighted.sum(dim=0)     # shape (m1, ..., mk)

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

    V1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).t()
    V2 = torch.tensor([[3.0, 4.0, 7.0], [9.0, 7.0, 5.0]]).t()

    scale_fcns = [
        lambda i: torch.ones(i.shape),
        lambda i: torch.ones(i.shape),
    ]

    result = batched_weighted_convolution([V1, V2], scale_fcns, h_fcn)
    print("batched result: ", result)