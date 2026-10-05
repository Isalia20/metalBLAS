"""Execute the recovered floating-point vector schedules, without tuning."""
from functools import cache
import platform

import torch

from . import kernels, mps_lora
from .dispatch import _pk, _resolve_inputs, _resolve_bmm_inputs
from .mps_a14 import vector_plan
from .mps_heuristics import device_info

_TYPES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}


@cache
def _zero_buffer():
    # MPS AIR redirects invalid vector loads to a shared zero buffer.
    return torch.zeros(4, dtype=torch.float32, device="mps")


@cache
def _launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb, batch, bsa, bsb, bsc, dev, macos_major,
                 epilogue=None):
    plan = vector_plan(M, N, K, dtype, dev=dev, trans_a=ta, trans_b=tb,
                       batch=batch, macos_major=macos_major)
    if plan.kernel == "LORADOWN_gemv_kernel":
        return mps_lora._launch_plan(M, N, K, lda, ldb, ldc, N == 1 or ldb % 4 == 0,
                                     dev, macos_major, epilogue)
    swap = M < N
    W, R = max(M, N), min(M, N)
    flat, tv = (not tb, not ta) if swap else (ta, tb)
    ldx, ldv, bsx, bsv = (ldb, lda, bsb, bsa) if swap else (lda, ldb, bsa, bsb)
    gx, gy, _ = plan.threadgroup
    rows = 4 if R == 1 and K <= 256 and not flat else 1
    k_threads = gy if gy > 1 else gx
    groups = ((K & ~3) + k_threads * 4 - 1) // (k_threads * 4)
    full_groups = K // (k_threads * 4 * plan.unroll_k) * plan.unroll_k
    fn = kernels.mps_vector(_TYPES[dtype], R, gx, gy, flat, tv, swap, rows, plan.unroll_k,
                             groups if groups <= 31 else 0, full_groups if groups <= 31 else 0,
                             (K & ~3) % (k_threads * 4 * plan.unroll_k) != 0, K % 4,
                             epilogue=None if epilogue is None else epilogue[-2:])
    dims = _pk(W, K, ldx, ldv, ldc, groups, full_groups, 0) + [bsx, bsv, bsc]
    threads = tuple(plan.grid[i] * plan.threadgroup[i] for i in range(3))
    zero = _zero_buffer()
    if epilogue is None:
        def run(a, b, out):
            x, v = (b, a) if swap else (a, b)
            fn(x, v, out, dims, zero, threads=threads, group_size=plan.threadgroup)
    else:
        br, bc, bb = epilogue[:3]
        bias_strides = [bb, br, bc]
        def run(a, b, out, bias):
            x, v = (b, a) if swap else (a, b)
            fn(x, v, out, dims, zero, bias, bias_strides,
               threads=threads, group_size=plan.threadgroup)
    return run, plan


def matmul(a, b, *, out=None, dev=None, macos_major=None):
    """Run vector schedules for 2-D or equal-batch 3-D inputs with min(M,N)<=16.

    This validation entry includes LoRA; the public dispatcher separately applies
    MPS's GEMM/vector routing. Transposes, row padding and offsets work.
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
    if min(M, N) > 16:
        raise ValueError("Vector execution requires a dimension at most 16")
    if out is None:
        out = a.new_empty(shape)
    if tuple(out.shape) != shape or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("Output shape, dtype and device must match the product")
    if out.stride(-1) != 1 or out.stride(-2) < N:
        raise ValueError("Vector output requires unit column stride and nonoverlapping rows")
    if min(M, N, batch) == 0:
        return out
    if K == 0:
        return out.zero_()
    # Canonicalize contiguous unit dimensions as PyTorch does before MPSGraph.
    if A.is_contiguous(): ta, lda = False, K
    if B.is_contiguous(): tb, ldb = False, N
    if macos_major is None:
        macos_major = int(platform.mac_ver()[0].split('.')[0])
    bsc = out.stride(0) if a.ndim == 3 else 0
    run, _ = _launch_plan(a.dtype, M, N, K, lda, ldb, out.stride(-2), ta, tb,
                          batch, bsa, bsb, bsc, dev or device_info(), macos_major)
    run(A, B, out)
    return out
