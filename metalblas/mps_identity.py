"""Two-dimensional specialization of MPSNDArray's identity encoder and AIR."""
from dataclasses import dataclass
from functools import cache
from math import prod

import torch

from . import kernels
from .dispatch import _ceildiv, _pk


def launch_order(src, dst, shape):
    """MPSNDArrayIdentityAppleBehaviors::getLaunchOrder, not measured tuning."""
    def order(permutation, minimum):
        return tuple(sorted(range(4), key=lambda d: (shape[d] < minimum, permutation[d], d)))

    default = (0, 1, 2, 3)
    source, destination = order(src, 1), order(dst, 1)
    candidate, contiguous = default, source[0]
    large = any(n >= 4 for n in shape)
    if large:
        preferred = src if shape[destination[0]] * 4 < shape[source[0]] else dst
        candidate = order(preferred, 4)
        contiguous = candidate[0]

    def span(order_, first):
        return prod(shape[d] for d in order_[:candidate.index(first)])

    small = (max(shape[candidate[i]] for i in range(3)) <= 32
             or (source[0] > 0 and source[1] > 0)
             or (destination[0] > 0 and destination[1] > 0))
    if not small:
        keep = ((shape[candidate[0]] < 8192 or shape[candidate[1]] < 8192)
                and span(source, source[0]) < 8192
                and span(destination, destination[0]) < 512)
        if not large or keep:
            return (candidate if keep else default), contiguous
    elif not large:
        return default, contiguous
    source, destination = order(src, 4), order(dst, 4)
    x = source[0]
    y = destination[0] if destination[0] != x else source[1]
    return (x, y, *(d for d in source if d not in (x, y))), destination[0]


@dataclass(frozen=True)
class IdentityPlan:
    order: tuple
    transpose: bool
    results: int
    threadgroup: tuple
    grid: tuple

    @property
    def threadgroup_memory(self):
        x = self.threadgroup[0]
        return 4 * x * (x + 1) if self.transpose else 0


def schedule(M, N, src_transposed, dst_transposed, cores, max_threads=1024):
    shape = (N, M, 1, 1)
    src = (1, 0, 2, 3) if src_transposed else (0, 1, 2, 3)
    dst = (1, 0, 2, 3) if dst_transposed else (0, 1, 2, 3)
    order, contiguous = launch_order(src, dst, shape)
    X, Y = (shape[d] for d in order[:2])
    transpose = order[0] != contiguous
    # Both Apple behavior records carry 0x12: NX is 2 below/above the
    # getNXResultsPerThread Y>4096 boundary, but 1 if either extent <128.
    results = 2 if transpose and min(X, Y) >= 128 else 1
    width = 32
    height = max_threads // width if max_threads > 2 * width else 1
    if transpose:
        while width > max(X, 8):
            width //= 2
        height = width // 4 if width > 16 else width
        while width * height > max_threads:
            height //= 2
        while height > max(Y, 4) and height >= 2:
            height //= 2
        tx, ty = width, height
        gx, gy = _ceildiv(X, width * results), _ceildiv(Y, width * results)
    else:
        volume = M * N
        while height > 1 and width * height > 32 and cores * width * height > volume:
            height //= 2
        while width > 1 and width * height > 32 and cores * width * height > volume:
            width //= 2
        tx, ty = min(width, _ceildiv(X, 2) * 2), min(height, _ceildiv(Y, 2) * 2)
        while tx * ty < width:
            ty *= 2
        if X == 1 and Y > 1:
            tx, ty = 1, width * (1 if Y <= width * 8 else height)
        elif X > 1 and Y == 1:
            tx, ty = width * (1 if X <= width * 8 else height), 1
        gx, gy = _ceildiv(X, tx), _ceildiv(Y, ty)
    return IdentityPlan(order, transpose, results, (tx, ty, 1), (gx, gy, 1))


@cache
def launch_plan(dtype, M, N, sr, sc, dr, dc, cores):
    plan = schedule(M, N, sc > sr, dc > dr, cores)
    in_t = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}[dtype]
    fn = kernels.mps_identity(in_t, plan.order[0], plan.transpose,
                              plan.results, plan.threadgroup[0])
    if prod(plan.threadgroup) > fn.max_threads_per_threadgroup:
        raise RuntimeError("Ported identity pipeline cannot execute the MPS threadgroup")
    dims = _pk(M, N) + [sr, sc, dr, dc]
    threads = tuple(g * t for g, t in zip(plan.grid, plan.threadgroup))

    def run(src, dst):
        fn(src, dst, dims, threads=threads, group_size=plan.threadgroup)

    return run, plan
