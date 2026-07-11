"""Integer MetalBLAS dispatch."""
from __future__ import annotations

import torch

from . import dispatch as _api
from . import kernels
from .dispatch import (
    _bmm_out, _ceildiv, _dispatch_gemv, _pk, _resolve_bmm_inputs, _resolve_inputs,
)


PROFILE = {
    torch.int8: ("char", "int", "char"),
    torch.uint8: ("uchar", "uint", "uchar"),
    torch.int16: ("short", "int", "short"),
    torch.int32: ("int", "int", "int"),
    torch.int64: ("long", "long", "long"),
}
DTYPES = frozenset(PROFILE)
_BYTES = {torch.int8: 1, torch.uint8: 1, torch.int16: 2,
          torch.int32: 4, torch.int64: 8}
_GEMV_T = {
    torch.int8: (8, 8), torch.uint8: (8, 8), torch.int16: (4, 8),
    torch.int32: (2, 8), torch.int64: (1, 8),
}
_GEMV_NT = {
    torch.int8: (8, 4), torch.uint8: (8, 4), torch.int16: (4, 4),
    torch.int32: (4, 8), torch.int64: (2, 4),
}


def _clamp_vec(vec, ld, offset):
    """Largest power of two no larger than ``vec`` aligned to stride and offset."""
    align = int(ld) | int(offset)
    while vec > 1 and align % vec:
        vec >>= 1
    return vec


def _pick_tile(M, N, dtype):
    """Return ``(BM, BN, BK, TX, TY)`` from the M5 Pro integer sweeps."""
    nbytes = _BYTES[dtype]
    if max(M, N) <= 256:
        return 64, 64, 16, 16, 16
    if M <= 16 and N >= 1024:
        return 16, 64, 16, 16, 16
    if M <= 128 and N >= 1024:
        return 32, 64, 16, 8, 16
    if nbytes == 8:
        return 64, 64, 8, 16, 16
    if nbytes == 1 and M >= 512:
        return 128, 64, 16, 16, 16
    return 64, 64, 16, 16, 16


def _gemv_t(dtype, matrix, x, xs, out, cols, k, ld, epilogue=None):
    vec, nwarps = _GEMV_T[dtype]
    vec = _clamp_vec(vec, ld, matrix.storage_offset())
    fused = epilogue is not None
    beta_nz, alpha_nz = epilogue[-2:] if fused else (True, True)
    fn = kernels.gemv_t(*PROFILE[dtype], 32 * vec, nwarps, vec,
                        epilogue=fused, beta_nz=beta_nz, alpha_nz=alpha_nz)[0]
    args = (matrix, x, out.view(-1), _pk(cols, k, ld, int(xs)))
    if fused:
        bias, step, beta, alpha, _, _ = epilogue
        args += (bias, int(step), beta, alpha)
    groups = _ceildiv(cols, 32 * vec)
    fn(*args, threads=(nwarps * 32 * groups, 1, 1),
       group_size=(nwarps * 32, 1, 1))
    return out


def _gemv_nt(dtype, matrix, x, out, rows, k, ld, epilogue=None):
    vec, nwarps = _GEMV_NT[dtype]
    vec = _clamp_vec(vec, ld, matrix.storage_offset())
    fused = epilogue is not None
    beta_nz, alpha_nz = epilogue[-2:] if fused else (True, True)
    fn = kernels.gemv_nt(*PROFILE[dtype], 1, nwarps, vec,
                         red_tg=dtype is torch.int64, epilogue=fused,
                         beta_nz=beta_nz, alpha_nz=alpha_nz)[0]
    args = (matrix, x, out.view(-1), _pk(rows, k, ld))
    if fused:
        bias, step, beta, alpha, _, _ = epilogue
        args += (bias, int(step), beta, alpha)
    groups = _ceildiv(rows, nwarps)
    fn(*args, threads=(nwarps * 32 * groups, 1, 1),
       group_size=(nwarps * 32, 1, 1))
    return out


def matmul(a, b, out=None, epilogue=None):
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype, "integer matmul requires both operands the same dtype"
    dtype = a.dtype
    A, B, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)
    if out is None:
        out = a.new_empty(M, N)
    else:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps"
    if (M == 1 and N >= 16) or (N == 1 and M >= 16):
        return _dispatch_gemv(_gemv_t, _gemv_nt, A, B, M, N, K, lda, ldb,
                              trans_a, trans_b, dtype, out, epilogue)

    BM, BN, BK, TX, TY = _pick_tile(int(M), int(N), dtype)
    fused = epilogue is not None
    beta_nz, alpha_nz = epilogue[-2:] if fused else (True, True)
    fn = kernels.int_gemm(*PROFILE[dtype], BM, BN, BK, TX, TY, trans_a, trans_b,
                          epilogue=fused, beta_nz=beta_nz, alpha_nz=alpha_nz)[0]
    args = (A, B, out, _pk(M, N, K, lda, ldb, out.stride(0)))
    if fused:
        bias, br, bc, beta, alpha, _, _ = epilogue
        args += (bias, _pk(br, bc), beta, alpha)
    group = TX * TY
    fn(*args, threads=(group * _ceildiv(N, BN), _ceildiv(M, BM), 1),
       group_size=(group, 1, 1))
    return out


def bmm(a, b, out=None, epilogue=None):
    Bb = a.shape[0]
    fused = epilogue is not None
    if fused:
        input, beta, alpha, beta_nz, alpha_nz = epilogue
    if a.shape[1] == 1 or b.shape[2] == 1:
        loop_epi = (input, beta, alpha) if fused else None
        return _api._bmm_loop(a, b, out, loop_epi)

    dtype = a.dtype
    A, B, M, N, K, lda, ldb, trans_a, trans_b, sA, sB = _resolve_bmm_inputs(a, b)
    result, copy_back = _bmm_out(a, out, Bb, M, N, dtype)
    ldc, sC = map(int, (result.stride(1), result.stride(0)))
    BM, BN, BK, TX, TY = _pick_tile(M, N, dtype)
    fn = kernels.int_gemm(*PROFILE[dtype], BM, BN, BK, TX, TY, trans_a, trans_b,
                          epilogue=fused, beta_nz=beta_nz if fused else True,
                          alpha_nz=alpha_nz if fused else True, batched=True)[0]
    args = (A, B, result, _pk(M, N, K, lda, ldb, ldc))
    if fused:
        bias = input if input.dtype == dtype else input.to(dtype)
        sBias, br, bc = map(int, bias.expand(Bb, M, N).stride())
        args += (bias, _pk(br, bc), int(beta), int(alpha), _pk(sA, sB, sC, sBias))
    else:
        args += (_pk(sA, sB, sC, 0),)
    group = TX * TY
    fn(*args, threads=(group * _ceildiv(N, BN), _ceildiv(M, BM), Bb),
       group_size=(group, 1, 1))
    if copy_back:
        out.copy_(result)
        return out
    return result
