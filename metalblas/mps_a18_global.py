"""Global K partitions and reduction from MPSNDArray's A18 encoder."""
from dataclasses import dataclass

import torch

from . import kernels
from .dispatch import _ceildiv, _pk


@dataclass(frozen=True)
class GlobalSchedule:
    splits: int
    atomic: bool
    k_per_split: int
    last_k: int
    k_iterations: int
    grid: tuple
    threadgroup: tuple
    partial_elements: int
    counter_elements: int
    initializer_grid: tuple | None
    reduction_grid: tuple | None


def schedule(M, N, K, batch, params):
    unroll_k, um, un, sm, sn, splits, global_split, atomic, uber_m, uber_n = params[:10]
    if not global_split or splits < 2 or not params[13]:
        raise ValueError("Expected a linearized global K-split schedule")
    if params[15]:
        raise NotImplementedError("Morton rasterization is not ported")
    tm, tn = um * sm * 16, un * sn * 16
    tiles_m, tiles_n = _ceildiv(M, tm), _ceildiv(N, tn)
    groups = uber_m * uber_n * _ceildiv(tiles_m, uber_m) * _ceildiv(tiles_n, uber_n) * batch
    k_per_split = K // splits
    return GlobalSchedule(splits, bool(atomic), k_per_split, k_per_split + K % splits,
                           _ceildiv(k_per_split, unroll_k * 16), (groups * splits, 1, 1),
                           (sm * sn * 32, 1, 1), splits * batch * M * N, groups if atomic else 0,
                           (_ceildiv(groups, 32), 1, 1) if atomic else None,
                           None if atomic else (_ceildiv(batch * M * N, 256), 1, 1))


def launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb, batch, bsa, bsb, bsc, batched,
                params, epilogue=None):
    from .float_dispatch import _a18_launch
    plan = schedule(M, N, K, batch, params)
    fn, threads, group, dims = _a18_launch(
        dtype, M, N, K, params, ta, tb, lda=lda, ldb=ldb, ldc=ldc,
        batch=batch if batched else None,
        epilogue=None if epilogue is None else (epilogue[-2] != 0, True),
        source_epilogue=None if epilogue is None else epilogue[-2:])
    assert tuple(t // g for t, g in zip(threads, group)) == plan.grid
    out_t = {torch.float16: 'half', torch.bfloat16: 'bfloat'}[dtype]
    initialize, reduce = kernels.a18_global_helpers(out_t, plan.splits,
                                                    None if epilogue is None else epilogue[-2:])
    counter_count = _pk(plan.counter_elements, 0)
    output_stride = bsc if batched else M * ldc
    reduce_dims = _pk(M, N, batch, ldc, output_stride & 0xffffffff, output_stride >> 32)
    if epilogue is not None:
        br, bc, bb, beta, alpha = epilogue
        bias_strides = _pk(br, bc)
        reduce_bias_strides = _pk(br, bc, bb, 0)
    else:
        bb = 0
    batch_strides = _pk(bsa, bsb, bsc, bb)

    def run(a, b, out, bias=None):
        partials = torch.empty((plan.partial_elements,), dtype=torch.float32, device=a.device)
        args = (a, b, out, dims)
        if epilogue is not None:
            args += (bias, bias_strides, beta, alpha)
        if batched:
            args += (batch_strides,)
        args += (partials,)
        if plan.atomic:
            counters = torch.empty((plan.counter_elements,), dtype=torch.int32, device=a.device)
            initialize(counters, counter_count,
                       threads=(plan.initializer_grid[0] * 32, 1, 1), group_size=(32, 1, 1))
            args += (counters,)
        fn(*args, threads=threads, group_size=group)
        if not plan.atomic:
            args = (partials, out, reduce_dims)
            if epilogue is not None:
                args += (bias, reduce_bias_strides)
            reduce(*args, threads=(plan.reduction_grid[0] * 256, 1, 1), group_size=(256, 1, 1))

    return run, plan
