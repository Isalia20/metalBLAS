"""MPS's GEMM selection for float16 and bfloat16.

Recovered from MPSNDArray (macOS 26 and 27):

- M5: the 17 gemm_a18 parameters, from a tuning table chosen by the GPU core count, else from a low-core or a
  high-core heuristic.
- M1-M4: the A14 GEMM encoder picks the tile, the simdgroup grid and a split of K across threadgroups. The kernel is
  NDArrayMatrixMultiplyNNA14 on M1/M2 and mmul_kernel_a16 on M3/M4.

tests/test_apple_heuristics.py checks both against MPS dispatches traced on M2, M3 Ultra, M4 (macOS 26) and M5 Pro
(macOS 27). Not covered by traces: M1, the macOS 27 K split on M1-M4, the M5 tables for up to 10 and more than 20 GPU
cores, and the M5 branch for 60 or more cores. On M1-M4 only shapes with M > 16 and N > 16 are covered, because MPS
sends the others to its matrix-vector kernels.
"""
from __future__ import annotations

from functools import cache
import json
import os
import platform
import re
import struct
import subprocess
from typing import NamedTuple

import torch


class Device(NamedTuple):
    generation: int  # MPSCore's GPU generation code
    cores: int
    macos: int


def _cdiv(a, b):
    return (a + b - 1) // b


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _ceil_log2(x):
    return (x - 1).bit_length()


def generation(chip: int, cores: int) -> int:
    """MPSCore's generation code for an M<chip> GPU. Newer chips are treated as M5."""
    if chip == 1:
        return 11 + sum(cores >= c for c in (5, 9, 17, 33))
    low, threshold = {2: (16, 10), 3: (18, 11), 4: (20, 11)}.get(chip, (22, 11))
    return low + (cores >= threshold)


@cache
def device() -> Device | None:
    """The GPU of this Mac, or None if it is not an M-series GPU."""
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=2).stdout
        ioreg = subprocess.run(["ioreg", "-rc", "AGXAccelerator", "-d", "1"],
                               capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    chip = re.match(r"Apple M(\d+)", brand)
    cores = re.search(r'"gpu-core-count" = (\d+)', ioreg)
    if chip is None or cores is None:
        return None
    cores = int(cores.group(1))
    return Device(generation(int(chip.group(1)), cores), cores, int(platform.mac_ver()[0].split(".")[0]))


# M5: MPSNDArrayMatMulA18DeviceBehavior

_DTYPE_CODE = {torch.float16: 4, torch.bfloat16: 5}


class A18Params(NamedTuple):
    """MPSMatMulAutoTuningParametersA18. A threadgroup of simd_m x simd_n simdgroups (times k_splits) computes a
    (16 unroll_m simd_m) x (16 unroll_n simd_n) tile, 16 unroll_k elements of K per step."""
    unroll_k: int
    unroll_m: int
    unroll_n: int
    simd_m: int
    simd_n: int
    k_splits: int  # simdgroup groups splitting K in a threadgroup, or threadgroups with global_split
    global_split: int
    global_reduce: int
    uber_m: int  # tiles rasterized together
    uber_n: int
    barrier_iterations: int
    tgmem_size: int
    elastic_barrier: int
    linearize: int  # 1-D grid
    raster_m_bias: int  # uber tiles advance along M first
    morton: int
    barrier_inside: int


@cache
def _tables():
    """Tuning tables keyed by (M, N, K, batch, trans_a, trans_b, dtype code)."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "apple_tables.json")) as f:
        raw = json.load(f)
    return {name: {tuple(r[:7]): A18Params(*r[7:]) for r in raw[name]} for name in "GSC"}


def _a18_low_core(M, N, K):
    long_k = K >= 12288
    log2_n = _clamp(_ceil_log2(N), 4, 7 if long_k else 6)
    log2_m = _clamp(_ceil_log2(M), 4, 8 if long_k else 7)
    log2_sn = min(log2_n - 4, 2) if long_k else 0
    log2_sm = min(log2_m - 4, 3 if long_k else 2)
    unroll_k = _clamp(_cdiv(K, 16), 1, 3 if K > 4096 else 2)
    barrier = 8 if long_k else (12 if K > 4096 else (8 if K > 1024 else 0))
    return A18Params(unroll_k, (1 << log2_m >> log2_sm) >> 4, (1 << log2_n >> log2_sn) >> 4, 1 << log2_sm,
                     1 << log2_sn, 1, 0, 0, 2, 1, barrier, 0, 0, int(long_k), 0, 0, 0)


def _a18_big_gpu_split(M, N, K, cores, trans_a):
    """The 60-or-more-core branch of the high-core heuristic. No such M5 GPU was available, so it is untested."""
    k_threshold = 6000 if trans_a else 12288
    sm, sn = 1, 2
    tiles_m, tiles_n = _cdiv(M, 32), _cdiv(N, 64)
    while tiles_m * tiles_n > 4 * cores:
        if tiles_n < tiles_m:
            if tiles_m < 64 or 2 * sm * sn > 4:
                break
            sm *= 2
        else:
            if tiles_n < 64 or 2 * sm * sn > 4:
                break
            sn *= 2
        tiles_m, tiles_n = _cdiv(M, sm * 32), _cdiv(N, sn * 32)
    tiles = tiles_m * tiles_n
    if K > k_threshold:
        if _cdiv(K, 2) <= k_threshold:
            splits, may_grow = 2, True
        elif _cdiv(K, 4) <= k_threshold:
            splits, may_grow = 4, False
        else:
            return sm, sn, 8, 1, _cdiv(K, 8)
        tiles *= splits
    else:
        splits, may_grow = 1, True
    if tiles < cores and _cdiv(K, 2 * splits) > 511:
        splits *= 2
        base = tiles_m * tiles_n
        if base * splits < cores and may_grow:
            while _cdiv(K, 2 * splits) >= 512:
                splits *= 2
                if splits > 4 or base * splits >= cores:
                    break
        return sm, sn, splits, 1, _cdiv(K, splits)
    if K <= k_threshold:
        return sm, sn, splits, 0, K
    return sm, sn, splits, 1, _cdiv(K, splits)


def _fill_gpu(M, N, um, un, sm, sn, splits, cores):
    """While there are fewer threadgroups than cores, shrink the tile by one step, alternating N (first) and M.
    A step lowers the unroll, or the simdgroup count once the unroll is 1."""
    shrink_m = False
    while _cdiv(M, sm * um * 16) * splits * _cdiv(N, sn * un * 16) < cores:
        if shrink_m:
            if um > 1:
                um -= 1
            elif sm > 1:
                sm -= 1
        elif un > 1:
            un -= 1
        elif sn > 1:
            sn -= 1
        if um == un == sm == sn == 1:
            break
        shrink_m = not shrink_m
    return um, un, sm, sn


def _a18_thin(M, N, K, trans_a, trans_b, cores, unroll_k):
    """M or N at most 16: a tile 16 wide along that side; K is split across simdgroups unless the long side alone
    fills the GPU and its operand is not stored along it."""
    n_thin = N <= M
    raster_m = int(M <= N)
    um, un, sm, sn = (2, 1, 1, 1) if n_thin else (1, 2, 1, 2)
    uber_m, uber_n = (2, 1) if n_thin else (1, 2)
    flat = not trans_a if n_thin else trans_b
    if 128 * cores <= max(M, N) and not flat:
        k_splits = min(2 if K >= 16384 or max(M, N) > 16384 else 1, _cdiv(K, 16 * unroll_k))
        if M < N:
            sm, sn, raster_m = 1, _clamp((N + 15) >> 4, 1, 4), 1
            un = 2 if _cdiv(N, 16 * sn) > 1 else 1
        else:
            sm = _clamp((M + 15) >> 4, 1, 4)
            um = 2 if _cdiv(M, 16 * sm) > 1 else 1
    else:
        k_splits = _clamp(_cdiv(K, 16 * unroll_k), 1, 8)
        if M < N:
            un, sn, sm, raster_m = 1, 2, 1, 1
        else:
            um, sm = 1, 2
    um, un, sm, sn = _fill_gpu(M, N, um, un, sm, sn, 1, cores)
    tiles_m, tiles_n = _cdiv(M, 16 * um * sm), _cdiv(N, 16 * un * sn)
    return A18Params(unroll_k, um, un, sm, sn, k_splits, 0, 0, min(uber_m, tiles_m), min(uber_n, tiles_n), 4, 0, 0, 1,
                     raster_m, 0, 1)


def _a18_high_core(M, N, K, trans_a, trans_b, cores):
    if M <= 16 or N <= 16:
        return _a18_thin(M, N, K, trans_a, trans_b, cores, _clamp(_cdiv(K, 16), 1, 4))
    misaligned = (K | M | N) & 31
    if cores < 60 and ((K > 2048 and K % 32) or (trans_a and misaligned and not trans_b)):
        return _a18_low_core(M, N, K)
    sm, sn, k_splits, global_split, k_per_split = 1, 2, 1, 0, K
    if cores >= 60:
        sm, sn, k_splits, global_split, k_per_split = _a18_big_gpu_split(M, N, K, cores, trans_a)
    um, un, sm, sn = _fill_gpu(M, N, 2, 2, sm, sn, k_splits if global_split else 1, cores)
    tiles_m, tiles_n = _cdiv(M, 16 * um * sm), _cdiv(N, 16 * un * sn)
    raster_m = 1 if M <= N else int(not trans_a)
    return A18Params(_clamp(_cdiv(k_per_split, 16), 1, 4), um, un, sm, sn, k_splits, global_split, global_split,
                     min(2, tiles_m), min(2, tiles_n), 4, 0, 0, 1, raster_m, 0, 1)


def a18_params(M, N, K, dtype, cores, *, batch=1, trans_a=False, trans_b=False) -> A18Params:
    """The gemm_a18 parameters MPS picks on an M5 GPU with ``cores`` GPU cores."""
    table = "G" if cores <= 10 else ("S" if cores <= 20 else "C")
    hit = _tables()[table].get((M, N, K, batch, int(trans_a), int(trans_b), _DTYPE_CODE[dtype]))
    if hit is not None:
        return hit
    if cores < 13:
        return _a18_low_core(M, N, K)
    return _a18_high_core(M, N, K, trans_a, trans_b, cores)



# M1-M4: the A14 GEMM encoder

_F32_TENTH = struct.unpack("<f", struct.pack("<f", 0.1))[0]


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _a14_split_k(M, N, K, dev):
    """Split K until K / (M*N) <= 0.1f: macOS 26 halves the chunk, macOS 27 doubles the split count."""
    area = _f32(M * N)
    deep = _f32(_f32(K) / area) > _F32_TENTH and K > 2048
    splits, chunk = 1, K
    if dev.macos < 27:
        if not deep:
            return 1, K
        while True:
            chunk //= 2
            if _f32(_f32(chunk) / area) <= _F32_TENTH:
                break
    else:
        if deep:
            while _f32(_f32(chunk) / area) > _F32_TENTH:
                splits *= 2
                chunk = _cdiv(K, splits)
        if chunk > 8192:
            while chunk > 8192:
                splits *= 2
                chunk = _cdiv(K, splits)
        elif dev.generation == 17 and dev.cores > 64 and N <= 256 and K == 2048:
            chunk = 512 if N > 127 else 256
        elif not deep:
            return 1, K
    quantum = 32 if dev.generation >= 18 else 16
    chunk = _cdiv(max(chunk, 1), quantum) * quantum
    return _cdiv(K, chunk), chunk


def _a14_square(size, cores, fp16):
    """(bm, bn, bk) for M == N == K on M1/M2."""
    pad32, pad48, pad64 = (-size) % 32, (-size) % 48, (-size) % 64
    if cores <= 64:
        if not fp16:
            bn, bk = 48, 24
            if pad48 and pad32 < pad48:
                if _f32(_f32(_cdiv(size, 64) ** 2) / _f32(cores)) <= 12.0:
                    bn, bk = (32, 32) if pad32 <= pad64 else (64, 16)
                else:
                    bn, bk = (64, 16) if pad64 <= pad32 else (32, 32)
            return 64, bn, bk
        waste64 = ((pad64 * (size + 31)) >> 5) + ((pad32 * (size + 63)) >> 6)
        waste48 = 2 * ((size + 47) * pad48 // 48)
        return (64, 32, 32) if waste64 <= waste48 else (48, 48, 24)
    if fp16:
        return (48, 48, 24) if not pad48 or pad48 <= pad32 else (48, 32, 32)
    return (64, 32, 16) if pad64 + pad32 <= 2 * pad48 else (48, 48, 16)


def a14_params(M, N, K, dtype, dev, *, batch=1):
    """bm, bn, bk, sm, sn, splits and k_chunk that MPS picks on an M1-M4 GPU, in the caller's orientation."""
    gen, cores = dev.generation, dev.cores
    splits, chunk = _a14_split_k(M, N, K, dev)
    if M <= 32 and N >= 512:
        bm, bn, sm, sn = 32, 128, 2, 4
    elif N <= 32:
        bm = 128 if M >= 512 else (64 if M > 32 else 32)
        bn, sm, sn = 32, (4 if M >= 512 else 2), 2
    else:
        bm, bn, sm, sn = (64 if M > 32 else 32), 64, 2, 2
    bk = 16
    if dev.macos >= 27 and gen == 17 and cores > 64:
        bk = 16 if K <= 10240 else 32
        if N <= 256 and K == 2048:
            bm, bn, bk, sm, sn = (64 if N == 256 else 32), 32, 32, 2, 2
    if M == N == K:
        bm, bn, bk = _a14_square(M, cores, dtype is torch.float16)
    else:
        # Halve the tile while the work does not fill 4 tiles per core. The encoder counts the splits twice.
        work = splits * splits * M * N * batch
        if work <= 4 * cores * bm * bn:
            bm //= 2
            shrink_n = True
            while work <= 4 * cores * bm * bn and (bm > 16 or bn > 16):
                if bn > 16 and shrink_n:
                    bn //= 2
                elif bm > 16:
                    bm //= 2
                shrink_n = not shrink_n
            sm = sn = 2
    if gen >= 18:
        sm = sn = 2
        bm, bn, bk = (32, 32, 16) if M == N == K else (bm, bn, 32)
    if splits > 1:
        chunk = _cdiv(chunk, bk) * bk
        splits = _cdiv(K, chunk)
    return bm, bn, bk, sm, sn, splits, chunk
