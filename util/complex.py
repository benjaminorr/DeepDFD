import torch


def pack(real, imag):
    return torch.stack([real, imag], dim=-1)


def unpack(x):
    return x[..., 0], x[..., 1]


def conj(x):
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def ones(shape, dtype=torch.float32, device=torch.device('cpu')):
    return torch.stack([torch.ones(shape, dtype=dtype, device=device),
                        torch.zeros(shape, dtype=dtype, device=device)], dim=-1)


def eye(K):
    return torch.stack([torch.eye(K), torch.zeros((K, K))], dim=-1)


def abs2(x):
    return x[..., -1] ** 2 + x[..., -2] ** 2


def multiply(x, y):
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    return torch.stack([x_real * y_real - x_imag * y_imag, x_imag * y_real + x_real * y_imag], dim=-1)


def mul_with_func(x, y, func):
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    xr_yr = func(x_real, y_real)
    xr_yi = func(x_real, y_imag)
    xi_yr = func(x_imag, y_real)
    xi_yi = func(x_imag, y_imag)
    real = xr_yr - xi_yi
    imag = xr_yi + xi_yr
    return torch.stack([real, imag], dim=-1)


# torch.rfft/torch.irfft were removed in PyTorch 1.8. These wrappers keep their calling convention
# (signal_ndim, signal_sizes) and the packed [..., 2] real/imag layout on top of torch.fft.
def _fft_dtype(x):
    # cuFFT only supports power-of-two sizes in half precision, so compute in fp32 instead.
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


def rfft(x, signal_ndim):
    dims = tuple(range(-signal_ndim, 0))
    fft = torch.fft.rfftn(x.to(_fft_dtype(x)), dim=dims)
    return pack(fft.real, fft.imag).to(x.dtype)


def irfft(x, signal_ndim, signal_sizes=None):
    dims = tuple(range(-signal_ndim, 0))
    real, imag = unpack(x.to(_fft_dtype(x)))
    return torch.fft.irfftn(torch.complex(real, imag), s=signal_sizes, dim=dims).to(x.dtype)
