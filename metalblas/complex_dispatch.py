"""Complex MetalBLAS dispatch."""
from __future__ import annotations

import torch

from . import dispatch as _api
from . import kernels
from .dispatch import _pk


PROFILE = {
    torch.complex64: ("float2", "float"),
    torch.complex32: ("half2", "half"),
}
REAL_DTYPE = {
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}
DTYPES = frozenset(PROFILE)

_T_NWARPS = 8
_NT_NWARPS = 4


def _inputs(a, b):
    dtype = a.dtype if a.dtype in DTYPES else b.dtype
    a = a.to(dtype) if a.dtype != dtype else a
    b = b.to(dtype) if b.dtype != dtype else b
    return (a.resolve_conj() if a.is_conj() else a,
            b.resolve_conj() if b.is_conj() else b, dtype)


def _gemv(a, b, out=None, epilogue=None):
    M, K = a.shape
    N = b.shape[1]
    transposed = (M == 1 and N >= 1 and a.is_contiguous() and b.is_contiguous()
                  and b.stride(0) == N)
    normal = (N == 1 and M >= 1 and a.is_contiguous() and b.is_contiguous()
              and a.stride(0) == K)
    if not (transposed or normal):
        return None
    fused = epilogue is not None
    c2, real_t = PROFILE[a.dtype]
    if fused:
        bias, br, bc, beta, alpha, bnz, anz = epilogue
    if transposed:
        fn = kernels.cgemv_t(c2, "float2", real_t, 32, _T_NWARPS,
                             epilogue=fused, beta_nz=bnz if fused else True,
                             alpha_nz=anz if fused else True)[0]
        out = out if out is not None else a.new_empty(1, N)
        args = (b, a.view(-1), out.view(-1), _pk(N, K, N, 1))
        size, step, nwarps = N, bc if fused else 0, _T_NWARPS
    else:
        fn = kernels.cgemv_nt(c2, "float2", real_t, _NT_NWARPS,
                              epilogue=fused, beta_nz=bnz if fused else True,
                              alpha_nz=anz if fused else True)[0]
        out = out if out is not None else a.new_empty(M, 1)
        args = (a, b.view(-1), out.view(-1), _pk(M, K, K, 1))
        size, step, nwarps = M, br if fused else 0, _NT_NWARPS
    if fused:
        args += (bias, int(step), float(beta.real), float(beta.imag),
                 float(alpha.real), float(alpha.imag))
    groups = (size + (31 if transposed else nwarps - 1)) // (32 if transposed else nwarps)
    fn(*args, threads=(nwarps * 32 * groups, 1, 1),
       group_size=(nwarps * 32, 1, 1))
    return out


def _products(a, b, product):
    """Split two complex tensors and form their four real matrix products."""
    c2, real_t = PROFILE[a.dtype]
    split = kernels.complex_pack(c2, real_t)[0]
    parts = []
    for value in (a, b):
        real = torch.empty(value.shape, dtype=REAL_DTYPE[a.dtype], device=value.device)
        imag = torch.empty_like(real)
        n = value.numel()
        split(value, real, imag, n, threads=(n, 1, 1), group_size=(256, 1, 1))
        parts.extend((real, imag))
    ar, ai, br, bi = parts
    return (product(ar, br), product(ai, bi), product(ar, bi), product(ai, br)), c2, real_t


def matmul(a, b, out=None, epilogue=None):
    """Complex 2-D matrix multiplication with an optional fused epilogue."""
    a, b, dtype = _inputs(a, b)
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dim() == 2 and b.dim() == 2, "complex matmul currently expects 2-D inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"
    if out is not None:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps", \
            f"out must be ({M}, {N}) {dtype} on mps"
    if epilogue is not None:
        a, b = a.contiguous(), b.contiguous()
    result = _gemv(a, b, out, epilogue)
    if result is not None:
        return result

    products, _, _ = _products(a.contiguous(), b.contiguous(), _api.matmul)
    if out is None:
        out = torch.empty(M, N, dtype=dtype, device=a.device)
    n = M * N
    if epilogue is None:
        combine = kernels.complex_pack(*PROFILE[dtype])[1]
        args = (*products, out, n)
    else:
        bias, br, bc, beta, alpha, bnz, anz = epilogue
        combine = kernels.complex_pack(*PROFILE[dtype], epilogue=True,
                                       beta_nz=bnz, alpha_nz=anz)[1]
        args = (*products, out, n, bias, _pk(N, br, bc, 0), float(beta.real),
                float(beta.imag), float(alpha.real), float(alpha.imag))
    combine(*args, threads=(n, 1, 1), group_size=(256, 1, 1))
    return out


def bmm(a, b, out=None):
    a, b, dtype = _inputs(a, b)
    a, b = a.contiguous(), b.contiguous()
    Bb, M, _ = a.shape
    N = b.shape[2]
    products, c2, real_t = _products(a, b, _api.bmm)
    if out is None:
        out = torch.empty(Bb, M, N, dtype=dtype, device=a.device)
    n = Bb * M * N
    kernels.complex_pack(c2, real_t)[1](
        *products, out, n, threads=(n, 1, 1), group_size=(256, 1, 1))
    return out


def baddbmm(input, a, b, beta, alpha, beta_nz, alpha_nz, out=None):
    dtype = a.dtype
    product = bmm(a, b)
    Bb, M, N = product.shape
    bias = input if input.dtype == dtype else input.to(dtype)
    result = (alpha * product if alpha_nz else
              beta * bias.expand(Bb, M, N).to(dtype) if beta_nz else
              torch.zeros_like(product))
    if alpha_nz and beta_nz:
        result = result + beta * bias.expand(Bb, M, N)
    if out is None:
        return result.contiguous() if not result.is_contiguous() else result
    return out.copy_(result)
