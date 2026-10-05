"""Validation entry for MPSNDArray's source-derived LoRA vector schedule."""
from functools import cache
import platform

import torch

from . import kernels
from .dispatch import _pk, _resolve_inputs
from .mps_a14 import vector_plan
from .mps_heuristics import device_info


@cache
def _launch_plan(M, N, K, lda, ldb, ldc, vec_b, dev, macos_major, epilogue=None):
    plan = vector_plan(M, N, K, torch.float16, dev=dev, trans_b=True, macos_major=macos_major)
    if plan.kernel != "LORADOWN_gemv_kernel":
        raise ValueError("Shape is outside the recovered LoRA route")
    tg_k = plan.threadgroup[0] * 4
    tile_k = tg_k * plan.unroll_k
    fn = kernels.mps_lora(N, plan.row_tile, plan.unroll_k, plan.threadgroup[0] // 32,
                          vec_b, K % 4, min(K // tile_k, 3),
                          (K & ~3) % tile_k != 0, (K & ~3) % tg_k != 0,
                          epilogue=None if epilogue is None else epilogue[-2:])
    dims = _pk(K, lda, ldb, ldc)
    threads = (plan.threadgroup[0], M, 1)
    if epilogue is None:
        def run(a, b, out):
            fn(a, b, out, dims, threads=threads, group_size=plan.threadgroup)
    else:
        br, bc, bb = epilogue[:3]
        bias_strides = [bb, br, bc]
        def run(a, b, out, bias):
            fn(a, b, out, dims, bias, bias_strides, threads=threads, group_size=plan.threadgroup)
    return run, plan


def matmul(a, b, *, out=None, dev=None, macos_major=None):
    """Run unbatched half products on the recovered LoRA schedule, without timing.

    Requires row-major A, column-major B and 2 <= N <= min(M,16), M <= 128.
    Vector routing and default API integration are separate from this entry.
    """
    if (a.ndim != 2 or b.ndim != 2 or not a.is_mps or not b.is_mps
            or a.dtype != torch.float16 or b.dtype != a.dtype):
        raise ValueError("Expected two half-precision matrices on MPS")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    A, B, M, N, K, lda, ldb, ta, tb = _resolve_inputs(a, b)
    if not (2 <= N <= min(M, 16) and M <= 128 and not ta and tb):
        raise ValueError("LoRA requires row-major A, column-major B and 2 <= N <= min(M,16), M <= 128")
    if out is None:
        out = a.new_empty((M, N))
    if tuple(out.shape) != (M, N) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("Output shape, dtype and device must match the product")
    if out.stride(1) != 1 or out.stride(0) < N:
        raise ValueError("LoRA output requires unit column stride and nonoverlapping rows")
    if macos_major is None:
        macos_major = int(platform.mac_ver()[0].split('.')[0])
    run, _ = _launch_plan(M, N, K, lda, ldb, out.stride(0),
                          ldb % 4 == 0, dev or device_info(), macos_major)
    run(A, B, out)
    return out
