"""Public PyTorch API and dtype routing for MetalBLAS."""
from __future__ import annotations

from math import prod
import sys

import torch

from . import kernels


def _pk(*vals):
    if len(vals) & 1:
        vals += (0,)
    return [(int(vals[i]) & 0xFFFFFFFF) | ((int(vals[i + 1]) & 0xFFFFFFFF) << 32)
            for i in range(0, len(vals), 2)]


def _ceildiv(n, d):
    return (n + d - 1) // d


def _unit_lead(t, rows, cols):
    """Return ``(is_column_major, leading_stride)`` for a simple matrix view."""
    s0, s1 = map(int, t.stride()[-2:])
    if s1 == 1 and s0 >= cols:
        return False, s0
    if s0 == 1 and s1 >= rows:
        return True, s1
    return None


def _matrix_view(t, rows, cols):
    layout = _unit_lead(t, rows, cols)
    return (t, *layout) if layout else (t.contiguous(), False, cols)


def _resolve_inputs(a, b):
    """Describe two 2-D operands without copying simple row/column-major views."""
    assert a.dim() == 2 and b.dim() == 2, "matmul currently expects 2-D inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"
    A, trans_a, lda = _matrix_view(a, M, K)
    B, trans_b, ldb = _matrix_view(b, K, N)
    return A, B, M, N, K, lda, ldb, trans_a, trans_b


def _dispatch_gemv(run_t, run_nt, A, B, M, N, K, lda, ldb,
                   trans_a, trans_b, dtype, out, epilogue=None):
    """Orient a matrix-vector product and its optional bias strides."""
    M, N, K = map(int, (M, N, K))
    row_epi = col_epi = None
    if epilogue is not None:
        bias, br, bc, beta, alpha, beta_nz, alpha_nz = epilogue
        row_epi = bias, br, beta, alpha, beta_nz, alpha_nz
        col_epi = bias, bc, beta, alpha, beta_nz, alpha_nz
    if M == 1:
        if trans_b:
            return run_nt(dtype, B, A.reshape(-1).contiguous(), out,
                          N, K, int(ldb), col_epi)
        x = A.reshape(-1)
        return run_t(dtype, B, x, x.stride(0), out, N, K, int(ldb), col_epi)
    if N == 1:
        if trans_a:
            x = B.reshape(-1)
            return run_t(dtype, A, x, x.stride(0), out, M, K, int(lda), row_epi)
        return run_nt(dtype, A, B.reshape(-1).contiguous(), out,
                      M, K, int(lda), row_epi)
    raise ValueError("gemv backend requires M==1 or N==1")


_OUT_POOL = {}
_OUT_POOL_LIST_CAP = 16
_OUT_POOL_MAX_ELEMS = 1 << 21


def _pooled_out(ref, M, N):
    if M * N > _OUT_POOL_MAX_ELEMS:
        return ref.new_empty(M, N)
    key = (ref.dtype, M, N)
    pool = _OUT_POOL.get(key)
    if pool is None:
        out = ref.new_empty(M, N)
        _OUT_POOL[key] = [out]
        return out
    for i in range(len(pool)):
        if sys.getrefcount(pool[i]) == 2:
            return pool[i]
    out = ref.new_empty(M, N)
    if len(pool) < _OUT_POOL_LIST_CAP:
        pool.append(out)
    return out


def _resolve_bmm_inputs(a, b):
    """Resolve row- or column-major batched views, copying other layouts."""
    _, M, K = a.shape
    _, _, N = b.shape
    A, trans_a, lda = _matrix_view(a, M, K)
    B, trans_b, ldb = _matrix_view(b, K, N)
    return (A, B, int(M), int(N), int(K), int(lda), int(ldb), trans_a, trans_b,
            int(A.stride(0)), int(B.stride(0)))


def _bmm_out(ref, out, Bb, M, N, dtype):
    """Return a row-major output and whether it must be copied back."""
    if out is None:
        return ref.new_empty(Bb, M, N), False
    assert out.shape == (Bb, M, N) and out.dtype == dtype and out.is_mps, \
        f"out must be ({Bb}, {M}, {N}) {dtype} on mps"
    if out.stride(2) == 1 and out.stride(1) >= N:
        return out, False
    return ref.new_empty(Bb, M, N), True


def _bmm_shape(a, b, op):
    assert a.is_mps and b.is_mps, f"{op} expects mps tensors"
    assert a.dim() == 3 and b.dim() == 3, f"{op} expects 3-D inputs"
    assert a.dtype == b.dtype, f"{op} requires both operands the same dtype"
    Bb, M, K = a.shape
    Bb2, K2, N = b.shape
    assert Bb == Bb2 and K == K2, f"{op} shape mismatch: {a.shape} @ {b.shape}"
    return Bb, M, N, K, a.dtype


def _scalars(dtype, beta, alpha):
    cast = int if dtype in _integer.DTYPES else (None if dtype in _complex.DTYPES else float)
    if cast is not None:
        beta, alpha = cast(beta), cast(alpha)
    return beta, alpha, beta != 0, alpha != 0


def _bmm_loop(a, b, out=None, epilogue=None):
    Bb, M, N = a.shape[0], a.shape[1], b.shape[2]
    if out is None:
        out = a.new_empty(Bb, M, N)
    if epilogue is not None:
        input, beta, alpha = epilogue
        bias = (input if input.dtype == a.dtype else input.to(a.dtype)).expand(Bb, M, N)
    for i in range(Bb):
        result = (matmul(a[i], b[i]) if epilogue is None else
                  addmm(bias[i], a[i], b[i], beta=beta, alpha=alpha))
        out[i].copy_(result)
    return out


def _matmul_nd(a, b, out=None):
    """Handle vector promotion and broadcast batches around the 2-D/3-D kernels."""
    assert a.ndim and b.ndim, "matmul operands must be at least 1-D"
    a_vec, b_vec = a.ndim == 1, b.ndim == 1
    if a_vec:
        a = a.unsqueeze(0)
    if b_vec:
        b = b.unsqueeze(-1)
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    if batch:
        size = prod(batch)
        a3 = a.expand(*batch, M, K).reshape(size, M, K)
        b3 = b.expand(*batch, K, N).reshape(size, K, N)
        result = bmm(a3, b3).reshape(*batch, M, N)
    else:
        result = matmul(a, b)
    if a_vec:
        result = result.squeeze(-2)
    if b_vec:
        result = result.squeeze(-1)
    return out.copy_(result) if out is not None else result


def matmul(a: torch.Tensor, b: torch.Tensor, *, backend: str | None = None,
           tile: tuple | None = None, swizzle_log: int | None = None,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``a @ b`` on MPS."""
    if a.ndim != 2 or b.ndim != 2:
        return _matmul_nd(a, b, out)
    if a.is_mps and (a.dtype in _complex.DTYPES or b.dtype in _complex.DTYPES):
        return _complex.matmul(a, b, out)
    if a.is_mps and a.dtype in _integer.DTYPES:
        return _integer.matmul(a, b, out)
    return _float.matmul(a, b, backend=backend, tile=tile,
                         swizzle_log=swizzle_log, out=out)


def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *,
          beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``beta * input + alpha * (mat1 @ mat2)``."""
    assert mat1.is_mps and mat2.is_mps
    assert mat1.dim() == 2 and mat2.dim() == 2, "addmm expects 2-D mat1, mat2"
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2, f"shape mismatch: mat1 is {mat1.shape}, mat2 is {mat2.shape}"
    dtype = mat1.dtype
    assert mat2.dtype == dtype, "addmm requires mat1.dtype == mat2.dtype"
    if out is not None:
        assert out.shape == (M, N) and out.dtype == dtype and out.is_mps
    beta, alpha, bnz, anz = _scalars(dtype, beta, alpha)
    bias = input if input.dtype == dtype else input.to(dtype)
    br, bc = map(int, bias.expand(M, N).stride())
    epilogue = bias, br, bc, beta, alpha, bnz, anz
    if dtype in _complex.DTYPES:
        return _complex.matmul(mat1, mat2, out, epilogue)
    if dtype in _integer.DTYPES:
        return _integer.matmul(mat1, mat2, out, epilogue)
    return _float.addmm(mat1, mat2, out, epilogue)


def bmm(a: torch.Tensor, b: torch.Tensor, *,
        out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute batched 3-D matrix multiplication."""
    Bb, M, N, K, dtype = _bmm_shape(a, b, "bmm")
    if Bb == 0 or M == 0 or N == 0:
        return out if out is not None else a.new_empty(Bb, M, N)
    if K == 0:
        return (out if out is not None else a.new_empty(Bb, M, N)).zero_()
    if dtype in _complex.DTYPES:
        return _complex.bmm(a, b, out)
    if dtype in _integer.DTYPES:
        return _integer.bmm(a, b, out)
    if dtype not in _float.PROFILE or not kernels.has_metal4():
        return _bmm_loop(a, b, out)
    return _float.bmm(a, b, out)


def baddbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, *,
            beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``beta * input + alpha * (batch1 @ batch2)``."""
    Bb, M, N, _, dtype = _bmm_shape(batch1, batch2, "baddbmm")
    beta, alpha, bnz, anz = _scalars(dtype, beta, alpha)
    if dtype in _complex.DTYPES:
        return _complex.baddbmm(input, batch1, batch2, beta, alpha, bnz, anz, out)
    if dtype in _integer.DTYPES:
        return _integer.bmm(batch1, batch2, out, (input, beta, alpha, bnz, anz))
    if not kernels.has_metal4():
        return _bmm_loop(batch1, batch2, out, (input, beta, alpha))
    bias = input if input.dtype == dtype else input.to(dtype)
    sBias, br, bc = map(int, bias.expand(Bb, M, N).stride())
    return _float.bmm(batch1, batch2, out,
                      (bias, br, bc, sBias, beta, alpha, bnz, anz))


def addbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, *,
           beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``beta * input + alpha * sum(batch1 @ batch2)``."""
    Bb, M, N, K, dtype = _bmm_shape(batch1, batch2, "addbmm")
    if Bb == 0 or K == 0:
        bias = input if input.dtype == dtype else input.to(dtype)
        beta = int(beta) if dtype in _integer.DTYPES else beta
        result = ((bias * beta).expand(M, N).contiguous() if beta != 0 else
                  bias.new_zeros(M, N))
        if out is not None:
            assert out.shape == (M, N) and out.dtype == dtype and out.is_mps
            return out.copy_(result)
        return result
    A = batch1.permute(1, 0, 2).reshape(M, Bb * K)
    B = batch2.reshape(Bb * K, N)
    return addmm(input, A, B, beta=beta, alpha=alpha, out=out)


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1-D inner product, matching ``torch.dot``."""
    assert a.dim() == 1 and b.dim() == 1, \
        f"1D tensors expected, but got {a.dim()}D and {b.dim()}D tensors"
    assert a.shape[0] == b.shape[0], \
        f"inconsistent tensor size, expected [{a.shape[0]}] and [{b.shape[0]}]"
    assert a.dtype == b.dtype, \
        f"dot : expected both vectors to have same dtype, got {a.dtype} and {b.dtype}"
    return matmul(a, b)


def vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inner product conjugating the first argument."""
    return dot(a.conj().resolve_conj() if a.is_complex() else a, b)


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the outer product of two vectors."""
    assert a.dim() == 1, f"outer: Expected 1-D argument self, but got {a.dim()}-D"
    assert b.dim() == 1, f"outer: Expected 1-D argument vec2, but got {b.dim()}-D"
    return a.unsqueeze(1) * b.unsqueeze(0)


def mv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Matrix-vector product."""
    assert mat.dim() == 2 and vec.dim() == 1, \
        f"vector + matrix @ vector expected, got {mat.dim()}, {mat.dim()}, {vec.dim()}"
    assert mat.shape[1] == vec.shape[0], \
        f"size mismatch, got mat ({mat.shape[0]}x{mat.shape[1]}), vec ({vec.shape[0]})"
    return matmul(mat, vec)


# Imported after the shared utilities so dtype modules can reuse them during initialization.
from . import complex_dispatch as _complex
from . import float_dispatch as _float
from . import integer_dispatch as _integer


gemm = matmul
ger = outer
