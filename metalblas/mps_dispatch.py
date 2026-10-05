"""Floating-point matmul dispatch from the recovered MPSNDArray encoders."""
from functools import cache
import platform

import torch

from . import mps_a14, mps_a16, mps_heuristics, mps_nna14, mps_submission, mps_vector
from .dispatch import _pk, _pooled_out, _resolve_inputs, _resolve_bmm_inputs


@cache
def _macos_major():
    return int(platform.mac_ver()[0].split('.')[0])


@cache
def _launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb,
                 batch, bsa, bsb, bsc, batched, dev, macos_major, epilogue=None):
    a18 = dev.uses_a18_kernel and dtype is not torch.float32
    if a18:
        if macos_major != 27:
            raise NotImplementedError("A18 entry-point routing is recovered for macOS 27")
        vector = mps_heuristics.vector_kernel_preferred(M, N, K, dtype, dev, ta, tb)
    else:
        vector = mps_a14.vector_preferred(M, N, K, dtype, dev=dev,
                                          trans_a=ta, trans_b=tb, macos_major=macos_major)
    if vector:
        return mps_vector._launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb,
                                       batch, bsa, bsb, bsc, dev, macos_major, epilogue)
    if not a18:
        from . import mps_chunk
        chunk = mps_chunk.schedule(M, N, batch, dev)
        if chunk is not None:
            return mps_chunk.launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb,
                                         batched, dev, macos_major, chunk, epilogue)
        family = mps_nna14 if dev.generation < 18 else mps_a16
        return family._launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb, False, False,
                                    batch, bsa, bsb, bsc, dev, macos_major, epilogue)

    # Import after dispatch.py has finished importing float_dispatch.py.
    from .float_dispatch import _a18_launch
    source, params = mps_heuristics.select_params(M, N, K, ta, tb, dtype, dev, batch)
    if params[15]:
        raise NotImplementedError("A18 Morton rasterization is not ported")
    if params[6] and params[5] > 1:
        from . import mps_a18_global
        run, _ = mps_a18_global.launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb,
                                             batch, bsa, bsb, bsc, batched, params, epilogue)
        return run, mps_heuristics.MPSPlan(source, tuple(params))
    fn, threads, group, dims = _a18_launch(
        dtype, M, N, K, params, ta, tb, lda=lda, ldb=ldb, ldc=ldc,
        batch=batch if batched else None,
        epilogue=None if epilogue is None else (epilogue[-2] != 0, True),
        source_epilogue=None if epilogue is None else epilogue[-2:])

    if epilogue is None:
        def run(a, b, out):
            args = (a, b, out, dims)
            if batched:
                args += (_pk(bsa, bsb, bsc, 0),)
            fn(*args, threads=threads, group_size=group)
    else:
        br, bc, bb, beta, alpha = epilogue
        bias_strides = _pk(br, bc)
        batch_strides = _pk(bsa, bsb, bsc, bb)
        def run(a, b, out, bias):
            args = (a, b, out, dims, bias, bias_strides, beta, alpha)
            if batched:
                args += (batch_strides,)
            fn(*args, threads=threads, group_size=group)

    return run, mps_heuristics.MPSPlan(source, tuple(params))


def matmul(a, b, out=None, epilogue=None):
    """Execute source-derived routes for 2-D or equally batched 3-D operands."""
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("Expected two matrices or two equally batched matrix arrays")
    if a.dtype not in mps_a16._TYPES or b.dtype != a.dtype or not a.is_mps or not b.is_mps:
        raise ValueError("Expected MPS operands of the same floating-point dtype")
    if a.shape[-1] != b.shape[-2] or (a.ndim == 3 and a.shape[0] != b.shape[0]):
        raise ValueError("Incompatible matrix or batch dimensions")
    if epilogue is not None:
        bias = epilogue[0]
        if bias.dtype != a.dtype or bias.device != a.device:
            raise ValueError("Bias dtype and device must match the operands")
    batched = a.ndim == 3
    if batched:
        A, B, M, N, K, lda, ldb, ta, tb, bsa, bsb = _resolve_bmm_inputs(a, b)
        batch, shape = a.shape[0], (a.shape[0], M, N)
    else:
        A, B, M, N, K, lda, ldb, ta, tb = _resolve_inputs(a, b)
        batch, bsa, bsb, shape = 1, 0, 0, (M, N)
    if out is not None and (tuple(out.shape) != shape or out.dtype != a.dtype or out.device != a.device):
        raise ValueError("Output shape, dtype and device must match the product")
    copy_back = out is not None and (out.stride(-1) != 1 or out.stride(-2) < N)
    result = out
    if result is None or copy_back:
        result = a.new_empty(shape) if batched else _pooled_out(a, M, N)
    if min(M, N, batch) == 0:
        return out if copy_back else result
    if K == 0:
        if epilogue is None or epilogue[-2] == 0:
            result.zero_()
        else:
            beta = torch.tensor(epilogue[-2], dtype=a.dtype, device="cpu").item()
            result.copy_(bias.expand(shape)).mul_(beta)
    else:
        # Match PyTorch's contiguous representation of transposed unit dimensions.
        if A.is_contiguous(): ta, lda = False, K
        if B.is_contiguous(): tb, ldb = False, N
        bsc = result.stride(0) if batched else 0
        run, _ = mps_submission.launch_plan(a.dtype, M, N, K, lda, ldb, result.stride(-2), ta, tb,
                              batch, bsa, bsb, bsc, batched, mps_heuristics.device_info(),
                              _macos_major(), None if epilogue is None else epilogue[1:])
        if epilogue is None:
            run(A, B, result)
        else:
            run(A, B, result, bias)
    return out.copy_(result) if copy_back else result


def addbmm(a, b, out, epilogue):
    """MPSGraph's typed batched matmul followed by a linked batch reduction."""
    batch, M, K = a.shape
    N = b.shape[-1]
    bias, br, bc, _, beta, alpha = epilogue
    if bias.dtype != a.dtype or bias.device != a.device:
        raise ValueError("Bias dtype and device must match the operands")
    if out is not None and (tuple(out.shape) != (M, N) or out.dtype != a.dtype or out.device != a.device):
        raise ValueError("Output shape, dtype and device must match the product")
    copy_back = out is not None and (out.stride(-1) != 1 or out.stride(-2) < N)
    result = out
    if result is None or copy_back:
        result = _pooled_out(a, M, N) if batch == 1 else a.new_empty(M, N)
    if min(M, N) == 0:
        return out if copy_back else result
    if batch == 0 or K == 0:
        if beta == 0:
            result.zero_()
        else:
            beta = torch.tensor(beta, dtype=a.dtype, device="cpu").item()
            result.copy_(bias.expand(M, N)).mul_(beta)
    else:
        mps_submission.addbmm(a, b, result, epilogue)
    return out.copy_(result) if copy_back else result
