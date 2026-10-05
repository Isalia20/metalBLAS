"""Apple's MPS matmul tile selection, ported from MetalPerformanceShaders.

This mirrors what ``MPSGraph.matrixMultiplicationWithPrimaryTensor`` does on
Apple silicon for 16-bit GEMM: ``MPSNDArrayMatrixMultiplication`` asks its
per-GPU device behavior (``MPSNDArrayMatMulA18DeviceBehavior`` on M5-class
GPUs) for dispatch parameters.  That lookup is:

1. an exact-shape match in a built-in autotune table chosen by GPU generation
   and core count (``_matmulA18{G,S,C}Table``, ``_matmulA20PTable``), else
2. ``_A18GEMMHeuristicForHighCoreCountGPU`` (more than 12 GPU cores) or
   ``_A18GEMMHeuristicForLowCoreCountGPU`` (12 or fewer).

The result is a 17-word ``MPSMatMulAutoTuningParametersA18``::

    unrollK, unrollM, unrollN, simdM, simdN, kSplits, _, _, uberM, uberN,
    barrierIterations, tgmemSize, elasticBarrier, linearize, rasterMBias,
    morton, barrierInside

The threadgroup tile is ``(unrollM*simdM*16) x (unrollN*simdN*16)`` computed
by ``simdM*simdN`` simdgroups, ``tileK = unrollK*16``, and ``kSplits``
simdgroup groups split K inside the threadgroup.  ``uberM x uberN`` tiles
are rasterized together.

Everything here was recovered from the macOS 27.2 MPSNDArray binary and
validated against 1565 shapes traced through ``MPS_LOG_INFO`` on an M5 Pro,
where the port reproduces every parameter vector exactly.  See
``docs/mps_matmul_heuristics.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import re
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# MPSDataType -> MPSNDArrayA18MatMulKey dtype code.
DTYPE_CODE = {torch.float16: 4, torch.bfloat16: 5, torch.float32: 6}

PARAM_NAMES = ("unrollK", "unrollM", "unrollN", "simdM", "simdN", "kSplits", "f6", "f7",
               "uberM", "uberN", "barrierIterations", "tgmemSize", "elasticBarrier",
               "linearize", "rasterMBias", "morton", "barrierInside")


# Device identification (MPSCore MPSDevice::GetMPSDevice)

# MPS "generation" codes assigned by MPSDevice::GetMPSDevice from the Metal GPU
# family and the GPU core count.  Lower code = low-core variant of a family.
#   Apple7  (M1 / A14):      11 (<5 cores) 12 (<9) 13 (<17) 14 (<33) 15 (<65)
#   Apple8  (M2 / A15, A16): 16 (<10 cores) else 17
#   Apple9  (M3 / A17):      18 (<11 cores) else 19
#   M4 / A18 (private family id): 20 (<11 cores) else 21
#   Apple10 (M5 / A19):      22 (<11 cores) else 23
#   Apple11 (newer):         24
_FAMILY_RULES = (
    # (Metal GPU family id, low-core generation code, high-core generation code, threshold)
    (1011, 24, 24, 0),
    (1010, 22, 23, 11),
    (-1009, 20, 21, 11),
    (1009, 18, 19, 11),
    (1008, 16, 17, 10),
)


def _chip_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True, timeout=2).stdout.strip()
    except Exception:
        return ""


def _gpu_core_count() -> int | None:
    try:
        out = subprocess.run(["ioreg", "-rc", "AGXAccelerator", "-d", "1"],
                             capture_output=True, text=True, check=True, timeout=3).stdout
    except Exception:
        return None
    m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    return int(m.group(1)) if m else None


def _supports_family(family: int) -> bool | None:
    """Query ``-[MTLDevice supportsFamily:]`` through the ObjC runtime."""
    try:
        import ctypes
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.restype = ctypes.c_bool
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return None
        return bool(send(dev, objc.sel_registerName(b"supportsFamily:"), family))
    except Exception:
        return None


@dataclass(frozen=True)
class DeviceInfo:
    chip: str
    cores: int
    generation: int

    @property
    def table(self) -> str | None:
        """Built-in table used by MPSNDArrayMatMulA18DeviceBehavior's constructor."""
        g, c = self.generation, self.cores
        if (g & ~1) == 22:
            return "A18G" if c < 11 else ("A18S" if c < 21 else "A18C")
        if g >= 24:
            return "A20P" if c < 10 else ("A18G" if c < 13 else ("A18S" if c < 21 else "A18C"))
        return None

    @property
    def uses_a18_kernel(self) -> bool:
        """M5-class (Apple10) and newer GPUs run the ``gemm_a18`` kernel.

        M4 and older dispatch ``mmul_kernel_a16`` through the A14 behavior
        (verified by tracing a 10-core M4 and an 80-core M3 Ultra).
        """
        return self.generation >= 22


def _generation_from_name(chip: str, cores: int) -> int:
    m = re.search(r"\bM(\d)\b", chip)
    series = int(m.group(1)) if m else 5
    if series <= 1:
        for thr, gen in ((5, 11), (9, 12), (17, 13), (33, 14), (65, 15)):
            if cores < thr:
                return gen
        return 12
    if series == 2:
        return 16 if cores < 10 else 17
    if series == 3:
        return 18 if cores < 11 else 19
    if series == 4:
        return 20 if cores < 11 else 21
    return 22 if cores < 11 else 23


@lru_cache(maxsize=1)
def device_info() -> DeviceInfo:
    chip = _chip_name()
    cores = _gpu_core_count()
    env_cores = os.environ.get("METALBLAS_GPU_CORES")
    if env_cores:
        cores = int(env_cores)
    if not cores:
        cores = 10
    env_gen = os.environ.get("METALBLAS_MPS_GENERATION")
    if env_gen:
        return DeviceInfo(chip, cores, int(env_gen))
    gen = None
    if sys.platform == "darwin":
        for family, low, high, thr in _FAMILY_RULES:
            ok = _supports_family(family)
            if ok is None:
                break
            if ok:
                gen = low if cores < thr else high
                break
        else:
            if _supports_family(1007):
                gen = _generation_from_name("M1", cores)
    if gen is None:
        gen = _generation_from_name(chip, cores)
    return DeviceInfo(chip, cores, gen)


# Built-in tables

@lru_cache(maxsize=1)
def _tables():
    with open(os.path.join(_HERE, "mps_matmul_tables.json")) as f:
        raw = json.load(f)
    return {k: [(tuple(r[:21]), tuple(r[21:])) for r in v]
            for k, v in raw.items() if not k.startswith("_")}


def make_key(M, N, K, trans_a, trans_b, dtype_code, batch=1):
    perm_a = (1, 0, 2, 3) if trans_a else (0, 1, 2, 3)
    perm_b = (1, 0, 2, 3) if trans_b else (0, 1, 2, 3)
    return (M, N, K, batch, 1) + perm_a + perm_b + (dtype_code, dtype_code, 0, 0, 0, 0, 0, 0)


def table_lookup(table, key):
    """First exact match wins; the shipped tables contain duplicate keys."""
    for k, p in _tables().get(table, ()):
        if k == key:
            return p
    return None


# Heuristics (MPSNDArrayMatMulA18DeviceBehavior)

def _ceil_div(a, b):
    return 0 if b == 0 else (a + b - 1) // b


def _log2_ceil(x):
    return 0 if x < 2 else (x - 1).bit_length()


def _shrink_until_covered(M, N, unroll_m, unroll_n, simd_m, simd_n, cores, splits=1):
    """Shrink the tile until ``tilesM * tilesN * splits >= cores``.

    Alternates between the N side and the M side, reducing ``unroll`` before
    ``simd`` on each side.  Returns the tile and the final tile counts.
    """
    tiles_m = _ceil_div(M, unroll_m * simd_m * 16)
    tiles_n = _ceil_div(N, unroll_n * simd_n * 16)
    shrink_m = False
    while tiles_m * splits * tiles_n < cores:
        if not shrink_m:
            if unroll_n > 1:
                unroll_n -= 1
            elif simd_n > 1:
                simd_n -= 1
        else:
            if unroll_m > 1:
                unroll_m -= 1
            elif simd_m > 1:
                simd_m -= 1
        if unroll_m == unroll_n == simd_m == simd_n == 1:
            tiles_m, tiles_n = _ceil_div(M, 16), _ceil_div(N, 16)
            break
        shrink_m = not shrink_m
        tiles_m = _ceil_div(M, unroll_m * simd_m * 16)
        tiles_n = _ceil_div(N, unroll_n * simd_n * 16)
    return unroll_m, unroll_n, simd_m, simd_n, tiles_m, tiles_n


def a18_low_core(M, N, K, trans_a, trans_b, dtype_code, cores):
    """``_A18GEMMHeuristicForLowCoreCountGPU``: 12 or fewer cores, and the
    delegate for awkward shapes on bigger GPUs."""
    if dtype_code < 4:
        unroll_k_max, uber_m = 4, 2
    elif dtype_code == 6:
        unroll_k_max, uber_m = 0, 4
    else:
        unroll_k_max, uber_m = (3 if K > 4096 else 2), 2
    k12 = K >> 12
    if k12 < 3:
        simd_m_log_max, n_log_cap, m_log_cap = 2, 6, 7
        barrier = 12 if K > 4096 else (8 if K > 1024 else 0)
    else:
        simd_m_log_max, n_log_cap, m_log_cap = 3, 7, 8
        barrier = 8
    n_log = max(min(_log2_ceil(N), n_log_cap), 4)
    m_log = max(min(_log2_ceil(M), m_log_cap), 4)
    n_split_log = min(n_log - 4, 2)
    simd_m_log = min(simd_m_log_max, m_log - 4)
    unroll_k = max(min(unroll_k_max, (K + 15) >> 4), 1)
    simd_n_log = 0 if k12 < 3 else n_split_log
    unroll_m = ((1 << m_log) >> simd_m_log) >> 4
    unroll_n = ((1 << n_log) >> simd_n_log) >> 4
    return [unroll_k, unroll_m, unroll_n, 1 << simd_m_log, 1 << simd_n_log, 1, 0, 0,
            uber_m, 1, barrier, 0, 0, int(k12 > 2), 0, 0, 0]


def a18_high_core(M, N, K, trans_a, trans_b, dtype_code, cores):
    """``_A18GEMMHeuristicForHighCoreCountGPU``: more than 12 GPU cores."""
    if dtype_code == 6:
        raise NotImplementedError("fp32 is encoded by the A14 behavior on these GPUs")
    big = M > 16 and N > 16
    unaligned = ((K | M) & 31) != 0 or (N & 31) != 0
    if cores < 60 and big and ((K > 2048 and (K & 31) != 0)
                               or (trans_a and unaligned and not trans_b)):
        return a18_low_core(M, N, K, trans_a, trans_b, dtype_code, cores)
    unroll_k = max(1, min(4, (K + 15) >> 4))
    if not big:
        n_le_m = N <= M
        mx = max(M, N)
        raster_m_bias = int(M <= N)
        flat_layout = (not trans_a) if n_le_m else trans_b
        unroll_m, uber_n_max, uber_m_max = (2, 1, 2) if n_le_m else (1, 2, 1)
        simd_n = 1 if n_le_m else 2
        unroll_n = simd_n
        simd_m = 1
        if (cores << 7) <= mx and not flat_layout:
            k_splits = 2 if (K >= 16384 or mx > 16384) else 1
            k_splits = max(1, min(k_splits, _ceil_div(K, unroll_k * 16)))
            if M < N:
                simd_n = max(1, min(4, (N + 15) >> 4))
                unroll_n = 2 if _ceil_div(N, simd_n * 16) > 1 else 1
                simd_m = 1
                raster_m_bias = 1
            else:
                simd_m = max(1, min(4, (M + 15) >> 4))
                unroll_m = 2 if _ceil_div(M, simd_m * 16) > 1 else 1
        else:
            k_splits = max(1, min(8, _ceil_div(K, unroll_k * 16)))
            if M < N:
                unroll_n, simd_n, simd_m = 1, 2, 1
                raster_m_bias = 1
            else:
                simd_m = 2
                unroll_m = 1
        unroll_m, unroll_n, simd_m, simd_n, tiles_m, tiles_n = _shrink_until_covered(
            M, N, unroll_m, unroll_n, simd_m, simd_n, cores)
        return [unroll_k, unroll_m, unroll_n, simd_m, simd_n, k_splits, 0, 0,
                min(uber_m_max, tiles_m), min(uber_n_max, tiles_n), 4, 0, 0, 1, raster_m_bias, 0, 1]
    raster_m_bias = 1 if M <= N else int(not trans_a)
    k_splits, global_split, k_per = 1, False, K
    if cores < 60:
        simd_m, simd_n = 1, 2
    else:
        simd_m, simd_n, k_splits, global_split, k_per = _ultra_tile(M, N, K, trans_b, cores)
    unroll_k = max(1, min(4, (k_per + 15) >> 4))
    unroll_m, unroll_n, simd_m, simd_n, tiles_m, tiles_n = _shrink_until_covered(
        M, N, 2, 2, simd_m, simd_n, cores, splits=(k_splits if global_split else 1))
    flag = int(global_split)
    return [unroll_k, unroll_m, unroll_n, simd_m, simd_n, k_splits, flag, flag,
            min(2, tiles_m), min(2, tiles_n), 4, 0, 0, 1, raster_m_bias, 0, 1]


def _ultra_tile(M, N, K, trans_b, cores):
    """60+ core branch of ``_A18GEMMHeuristicForHighCoreCountGPU``.

    Grows the 32x64 tile while there are more than ``4 * cores`` tiles, then
    splits K across threadgroups (``kGlobalSplits``, reduced with atomics by
    ``gemmA18GlobalReduction``) when K is long or the tile count is below the
    core count.  Transliterated from the decompiled binary; not yet validated
    on hardware with 60 or more cores.

    Returns ``(simd_m, simd_n, k_splits, global_split, k_per_split)``.
    """
    k_thr = 6000 if trans_b else 0x3000
    simd_m, simd_n = 1, 2
    tiles_m = (M - 1 + 32) >> 5
    tiles_n = (N + 63) >> 6
    total = tiles_m * tiles_n
    if total > cores * 4:
        while True:
            cand = simd_m * 2 * simd_n
            if tiles_n < tiles_m:
                if tiles_m < 64 or cand > 4:
                    break
                simd_m *= 2
            else:
                if tiles_n < 64 or cand > 4:
                    break
                simd_n *= 2
            tiles_m = _ceil_div(M, simd_m << 5)
            tiles_n = _ceil_div(N, simd_n << 5)
            total = tiles_m * tiles_n
            if total <= cores * 4:
                break
    tile_m, tile_n = simd_m << 5, simd_n << 5
    tiles_m, tiles_n = _ceil_div(M, tile_m), _ceil_div(N, tile_n)
    splits, global_split = 1, False
    few_splits = True        # bVar8: splits still small enough to keep doubling
    take_global = False
    if K > k_thr:
        if (K + 1) >> 1 <= k_thr:
            splits, total = 2, tiles_m * tiles_n * 2
        elif (K + 3) >> 2 <= k_thr:
            splits, total, few_splits = 4, tiles_m * tiles_n * 4, False
        else:
            splits, take_global = 8, True
    else:
        total = tiles_m * tiles_n
    if not take_global:
        if total < cores:
            s2 = splits * 2
            if _ceil_div(K, s2) >= 512:
                grow = tiles_m * tiles_n * s2 < cores and few_splits
                result = s2
                cur = s2
                if grow:
                    while True:
                        s4 = cur * 2
                        if _ceil_div(K, s4) < 512:
                            result = cur
                            break
                        result = s4
                        if cur > 3:
                            break
                        cur = s4
                        if not tiles_m * tiles_n * s4 < cores:
                            break
                splits, take_global = result, True
        if not take_global and K <= k_thr:
            return simd_m, simd_n, splits, False, K
        take_global = True
    return simd_m, simd_n, splits, True, _ceil_div(K, splits)


def vector_kernel_preferred(M, N, K, dtype, dev: DeviceInfo, trans_a=False, trans_b=False) -> bool:
    """``MPSNDArrayIsVectorKernelPreferred`` for row-major operands.

    MPS computes a thin product with its ``NDArrayVectorMatrixMultiply``
    kernels instead of the GEMM when one dimension is tiny.
    """
    if max(M, N) <= 128 and min(M, N) <= 16 and trans_b and not trans_a:
        return True     # early exit for tiny products with a transposed right operand
    if dev.generation > 23:
        return min(M, N) <= 8
    weight = 4 if dtype is torch.float32 else 3
    thr = 9 if (dev.cores << 12) <= M * N * weight else 15
    if dev.generation == 22 and dev.cores > 10:
        if min(M, N) > 4 or K > 4096:
            return False
    if N > thr and M > thr:
        return False
    if N <= thr:
        return True
    return N <= 128


# Public planning API

@dataclass(frozen=True)
class MPSPlan:
    source: str            # "vector", "table", "high-core", "low-core"
    params: tuple          # 17 MPSMatMulAutoTuningParametersA18 words (empty for vector)

    @property
    def tile(self):
        """Threadgroup tile ``(rows, cols)``."""
        p = self.params
        return (p[1] * p[3] * 16, p[2] * p[4] * 16)

    @property
    def simdgroups(self):
        return self.params[3] * self.params[4]

    @property
    def tile_k(self):
        return self.params[0] * 16

    @property
    def k_splits(self):
        return self.params[5]

    @property
    def uber(self):
        return (self.params[8], self.params[9])

    def describe(self):
        if self.source == "vector":
            return "vector-matrix kernel"
        tm, tn = self.tile
        return (f"{self.source}: tile {tm}x{tn} ({self.simdgroups} simdgroups), "
                f"tileK {self.tile_k}, kSplits {self.k_splits}, uber {self.uber[0]}x{self.uber[1]}")


def select_params(M, N, K, trans_a, trans_b, dtype, dev: DeviceInfo | None = None, batch=1):
    """Reproduce ``GetKernelDispatchParametersForKey`` for 16-bit GEMM."""
    dev = dev or device_info()
    code = DTYPE_CODE[dtype]
    table = dev.table
    if table:
        hit = table_lookup(table, make_key(M, N, K, trans_a, trans_b, code, batch))
        if hit:
            return "table", tuple(hit)
    if dev.cores <= 12:
        return "low-core", tuple(a18_low_core(M, N, K, trans_a, trans_b, code, dev.cores))
    return "high-core", tuple(a18_high_core(M, N, K, trans_a, trans_b, code, dev.cores))


@lru_cache(maxsize=4096)
def plan(M, N, K, trans_a=False, trans_b=False, dtype=torch.bfloat16, batch=1) -> MPSPlan:
    """What MPSGraph would dispatch for ``A[MxK] @ B[KxN]`` (``batch`` of them) on this GPU."""
    dev = device_info()
    if dtype not in DTYPE_CODE or dtype is torch.float32:
        raise ValueError("MPS A18 planning covers float16 and bfloat16 only")
    if vector_kernel_preferred(M, N, K, dtype, dev, trans_a, trans_b):
        return MPSPlan("vector", ())
    source, params = select_params(M, N, K, trans_a, trans_b, dtype, dev, batch)
    return MPSPlan(source, params)
