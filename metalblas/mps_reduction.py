"""Batch-axis sum from MPSNDArray's EncodeReduction and reduction AIR.

The schedule is an axis-2 specialization of the encoders identified in
docs/mps_dispatch_sources.json. No measurements enter this planner.
"""
from dataclasses import dataclass
from functools import cache
from math import prod

import torch

from . import kernels
from .dispatch import _pk


@dataclass(frozen=True)
class ReductionPlan:
    elements_per_thread: tuple
    elements_per_threadgroup: tuple
    threads_per_axis: tuple
    groups_per_axis: tuple
    src_bounds: tuple
    dst_bounds: tuple
    threadgroup: tuple

    @property
    def grid(self):
        x, y, batch, _ = self.groups_per_axis
        return x, batch, y

    @property
    def threadgroup_memory(self):
        return prod(self.threadgroup) * 16  # one float4 per physical thread

    @property
    def parameters(self):
        # MPSNDArrayReductionAxesParameters, including initialValue=0 bits.
        return ((0,) * 4 + self.elements_per_thread + self.elements_per_threadgroup
                + self.threads_per_axis + self.groups_per_axis + self.src_bounds
                + self.dst_bounds + (1,) * 8 + (0,))


def reduction_plans(M, N, batch, *, dev, macos_major, max_threads=1024):
    """Recover up to three passes; batch=1 needs no reduction.

    The encoder queries the minimum maxTotalThreadsPerThreadgroup of its
    four linked pipelines, then rounds down to a power of two. All captured
    M2/M3/M5 pipelines report 1024; callers checking traces can supply it.
    """
    if min(M, N, batch, max_threads) < 1:
        raise ValueError("Reduction planning requires positive dimensions and capacity")
    if macos_major not in (26, 27):
        raise NotImplementedError("Reduction encoders are recovered for macOS 26 and 27")
    base = 32  # ReductionDefaultBehaviorsCtor / ReductionAA10BehaviorsCtor
    while base > max_threads:
        base >>= 1
    max_threads = 1 << (max_threads.bit_length() - 1)
    future = dev.generation >= (25 if macos_major == 27 else 24)
    minimum, target_shift = (64, 6) if future else (32, 4)
    result = []
    for pass_index in range(3):
        if batch == 1:
            break
        limit = 1024 if pass_index == 2 else 128
        tx = 4
        if 1 <= (N & 15) <= 8:
            tx = 1 if 1 <= (N & 7) <= 4 else 2
        remaining = base // tx
        iterations, batch_threads = limit, remaining
        while batch <= iterations * batch_threads // 2:
            if iterations > 1:
                iterations //= 2
            else:
                batch_threads //= 2
        batch_tile = iterations * batch_threads
        tz = remaining // batch_threads
        # At least sixteen elements per thread, when reduced-axis threads
        # can be exchanged for iterations without changing the batch tile.
        while 4 * iterations < 16 and batch_threads > 1:
            batch_threads //= 2
            iterations *= 2
            tz *= 2
        tm = tz
        while tm > M:
            tm //= 2
        threads = [tx, tm, batch_threads, 1]
        elements = [4, 1, iterations, 1]
        tile = [tx * 4, tm, batch_tile, 1]
        src = [N, M, batch, 1]
        grid = [(s + z - 1) // z for s, z in zip(src, tile)]
        # The physical z dimension retains padding after clipping tm to M.
        physical = [tx, batch_threads, tz]
        tgsize = prod(physical)
        budget = limit >> ((prod(elements) - 1).bit_length())
        available = max_threads // tgsize
        if grid[2] != 1 and grid[1] >= dev.cores and grid[2] < available * budget:
            while tile[2] < batch and available * budget > 1:
                if budget > 1:
                    elements[2] *= 2
                    budget //= 2
                else:
                    threads[2] *= 2
                    physical[1] *= 2
                    available //= 2
                tile[2] *= 2
            grid[2] = (batch + tile[2] - 1) // tile[2]
            tgsize = prod(physical)
        original_tgsize = tgsize
        original_groups = prod(grid)
        estimated_groups = original_groups
        # EncodeReduction redistributes nonreduced axes only above 63 cores.
        if dev.cores > 63:
            nonreduced = threads[0] * threads[1]
            if estimated_groups < dev.cores * 8 and nonreduced > 1:
                while tgsize > minimum or elements[2] > 1:
                    if tgsize == minimum:
                        elements[2] //= 2
                        threads[2] *= 2
                        tgsize *= 2
                    axis = 0 if threads[0] > 1 else 1
                    threads[axis] //= 2
                    tile[axis] //= 2
                    grid[axis] = (src[axis] + tile[axis] - 1) // tile[axis]
                    nonreduced //= 2
                    estimated_groups *= 2
                    tgsize //= 2
                    if estimated_groups >= dev.cores * 8 or nonreduced < 2:
                        break
            physical = [threads[0], threads[2], threads[1]]
        if estimated_groups < dev.cores * 8:
            score, target = 1, 8
        else:
            score, target = original_groups, dev.cores << target_shift
        # The binary deliberately retains the pre-redistribution tgsize here.
        score *= original_tgsize >> 5
        while elements[2] > 1 and score < target and tgsize < max_threads:
            elements[2] //= 2
            threads[2] *= 2
            physical[1] *= 2
            tgsize *= 2
            score *= 2
        dst = (N, M, grid[2], 1)
        result.append(ReductionPlan(tuple(elements), tuple(tile), tuple(threads),
                                     tuple(grid), tuple(src), dst, tuple(physical)))
        batch = grid[2]
    if batch != 1:
        raise NotImplementedError("Batch extent exceeds the recovered three-pass reduction")
    return tuple(result)


_TYPES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}


@cache
def _launch_plan(dtype, M, N, batch, ldc, dev, macos_major, beta, alpha):
    plans = reduction_plans(M, N, batch, dev=dev, macos_major=macos_major)
    launches = []
    for i, plan in enumerate(plans):
        final = i == len(plans) - 1
        fn = kernels.mps_batch_reduce(_TYPES[dtype] if i == 0 else "float",
                                       _TYPES[dtype] if final else "float", *plan.threadgroup,
                                       epilogue=(beta, alpha) if final else None)
        if prod(plan.threadgroup) > fn.max_threads_per_threadgroup:
            raise RuntimeError("Ported reduction pipeline cannot execute the MPS threadgroup")
        dims = _pk(N, M, plan.src_bounds[2], plan.elements_per_threadgroup[2],
                   ldc if final else N, 0)
        threads = tuple(g * t for g, t in zip(plan.grid, plan.threadgroup))
        launches.append((fn, dims, threads, plan))
    return tuple(launches)


def reduce_batch(values, out, bias, br, bc, beta, alpha, *, dev, macos_major):
    """Reduce contiguous typed matmul results; apply the epilogue on the last pass."""
    batch, M, N = values.shape
    launches = _launch_plan(values.dtype, M, N, batch, out.stride(0), dev, macos_major,
                             beta, alpha)
    for i, (fn, dims, threads, plan) in enumerate(launches):
        final = i == len(launches) - 1
        dst = out if final else torch.empty((plan.dst_bounds[2], M, N), dtype=torch.float32,
                                            device=values.device)
        args = (values, dst, dims)
        if final:
            args += (bias, [br, bc])
        fn(*args, threads=threads, group_size=plan.threadgroup)
        values = dst
    return out
