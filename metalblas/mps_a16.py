"""Execution of the recovered A16 GEMM schedule.

This entry point intentionally exposes only the A16 GEMM family. Vector routing,
the older NNA14 family and A18 tensor kernels have separate implementations.
The prepared launcher also supports the recovered linked bias epilogue.
"""
from functools import cache

import torch

from . import kernels
from .dispatch import _pk, _resolve_inputs, _resolve_bmm_inputs
from .mps_a14 import edge_fragments, gemm_plan
from .mps_heuristics import device_info

_TYPES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}


@cache
def _launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb, va, vb,
                 batch, bsa, bsb, bsc, dev, macos_major, epilogue=None):
    # Optional validation epilogue: (row stride, column stride, batch stride,
    # beta, alpha). The run callable receives the bias as its fourth argument.
    plan = gemm_plan(M, N, K, dtype, dev=dev, macos_major=macos_major, batch=batch)
    if not plan.kernel.startswith("mmul_kernel_a16_"):
        raise ValueError("A16 execution requires an M3 or newer GPU")
    assert not plan.stage_b
    in_t = _TYPES[dtype]
    # A16 adds a staged-operand stride bound to the encoder's dimension
    # check in EncodeMatrixMultiplyLinkingA14 (26.5.2 0x18faaebfc;
    # 27.2 0x190c778a4), packed into function constant 124 bit 11.
    dim16 = max(M, N, K) < 32767 and (not plan.stage_a or plan.tile_k * lda < 32767)
    # va/vb belong to the shared validation-launch signature. Apple's A16
    # vector loads do not require either leading-stride alignment flag.
    fn, reduce = kernels.a16_gemm(
        in_t, in_t, plan.tile_m, plan.tile_n, plan.tile_k, plan.simd_m, plan.simd_n,
        plan.stage_a, ta, tb, M % plan.tile_m == 0 and N % plan.tile_n == 0,
        K % plan.tile_k == 0, plan.splits, plan.k_per_split, dim16,
        4 if N % 4 == 0 else 1,
        edge_fragments(M, plan.tile_m, plan.simd_m),
        edge_fragments(N, plan.tile_n, plan.simd_n),
        max((plan.tile_k if ta else plan.tile_m) * lda,
            (plan.tile_n if tb else plan.tile_k) * ldb) >= 2**32,
        epilogue=None if epilogue is None else epilogue[-2:],
    )
    group = (plan.threads, 1, 1)
    threads = (plan.grid[0] * plan.threads, *plan.grid[1:])
    out_dims = _pk(M, N, K, lda, ldb, ldc) + [bsa, bsb, bsc]
    if epilogue is not None:
        br, bc, bb = epilogue[:3]
        bias_strides = [bb, br, bc]
    if plan.splits == 1:
        if epilogue is None:
            def run(a, b, out):
                fn(a, b, out, out_dims, threads=threads, group_size=group)
        else:
            def run(a, b, out, bias):
                fn(a, b, out, out_dims, bias, bias_strides, threads=threads, group_size=group)
    else:
        partial_dims = _pk(M, N, K, lda, ldb, N) + [bsa, bsb, plan.splits * M * N]
        rgrid = plan.reduction_grid
        red_threads = (rgrid[0] * 32, rgrid[1] * 32, rgrid[2])

        if epilogue is None:
            def run(a, b, out):
                partial = torch.empty((batch, plan.splits, M, N), device=out.device, dtype=torch.float32)
                fn(a, b, partial, partial_dims, threads=threads, group_size=group)
                reduce(partial, out, out_dims, threads=red_threads, group_size=(32, 32, 1))
        else:
            def run(a, b, out, bias):
                partial = torch.empty((batch, plan.splits, M, N), device=out.device, dtype=torch.float32)
                fn(a, b, partial, partial_dims, threads=threads, group_size=group)
                reduce(partial, out, out_dims, bias, bias_strides,
                       threads=red_threads, group_size=(32, 32, 1))
    return run, plan


def matmul(a, b, *, out=None, dev=None, macos_major=None):
    """Run the recovered A16 GEMM schedule for 2-D or equal-batch 3-D operands.

    This low-level validation entry always uses GEMM, even for shapes for which
    MPS would select its separate vector family. It performs no timing or tuning.
    """
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("Expected two matrices or two equally batched matrix arrays")
    if a.dtype not in _TYPES or b.dtype != a.dtype or not a.is_mps or not b.is_mps:
        raise ValueError("Expected MPS operands of the same floating-point dtype")
    if a.shape[-1] != b.shape[-2] or (a.ndim == 3 and a.shape[0] != b.shape[0]):
        raise ValueError("Incompatible matrix or batch dimensions")
    if a.ndim == 2:
        A, B, M, N, K, lda, ldb, ta, tb = _resolve_inputs(a, b)
        batch, bsa, bsb = 1, 0, 0
        shape = (M, N)
    else:
        A, B, M, N, K, lda, ldb, ta, tb, bsa, bsb = _resolve_bmm_inputs(a, b)
        batch = a.shape[0]
        shape = (batch, M, N)
    if out is None:
        out = a.new_empty(shape)
    if tuple(out.shape) != shape or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("Output shape, dtype and device must match the product")
    if out.stride(-1) != 1 or out.stride(-2) < N:
        raise ValueError("A16 output must have unit column stride and nonoverlapping rows")
    ldc, bsc = out.stride(-2), (out.stride(0) if a.ndim == 3 else 0)
    if min(M, N, batch) == 0:
        return out
    if K == 0:
        return out.zero_()
    va = (lda | A.storage_offset() | bsa) % 4 == 0
    vb = (ldb | B.storage_offset() | bsb) % 2 == 0
    run, _ = _launch_plan(a.dtype, M, N, K, lda, ldb, ldc, ta, tb, va, vb,
                          batch, bsa, bsb, bsc, dev or device_info(), macos_major)
    run(A, B, out)
    return out
