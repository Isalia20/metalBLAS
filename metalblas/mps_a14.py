"""Apple's pre-A18 floating-point GEMM selection.

Transcribed from the GEMM/vector encoders in MPSNDArray on macOS 26.3,
26.5.2 and 27.2. Source identities and addresses are recorded in
``docs/mps_dispatch_sources.json``; the rules are described in
``docs/mps_matmuls.md``. This is a selection model; ``mps_a16.py`` executes
its A16 GEMM schedules; the public dispatcher applies its family routing. All
choices come from those encoders, not timings.
"""
from __future__ import annotations

from dataclasses import dataclass
import platform
import struct

import torch

from .mps_heuristics import DeviceInfo, device_info


def _ceildiv(a, b):
    return (a + b - 1) // b


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def vector_preferred(M, N, K, dtype, *, dev, trans_a=False, trans_b=False,
                     macos_major=26):
    """Vector routing for dense 2-D or equally batched rank-3 operands.

    macOS 26: MPSNDArrayVectorCompatibleWithDims, 0x18faa7e54.
    macOS 27: MPSNDArrayIsVectorKernelPreferred, 0x190c6f378.
    """
    if macos_major == 27 and max(M, N) <= 128 and min(M, N) <= 16 and trans_b and not trans_a:
        return True
    if dev.generation >= 22 and dev.cores > 10 and (macos_major == 26 or dev.generation == 22):
        limit_dim = M if macos_major == 26 else min(M, N)
        if limit_dim > 4 or K > 4096 or (trans_a if M >= N else not trans_b):
            return False
    weight = 4 if dtype is torch.float32 else 3
    threshold = 9 if M * N * weight >= dev.cores * 4096 else 15
    if M > threshold and N > threshold:
        return False
    # The final load in this function indexes B's sizes through its
    # permutation: it is logical N, including for a transposed B. For a
    # non-batched product this last N <= 128 test cannot reject N <= 15.
    return True


def _split_k(M, N, K, generation, cores, macos_major):
    """Initial split count, before rounding each split to the selected K tile."""
    area = _f32(M * N)
    ratio = _f32(_f32(K) / area)
    threshold = _f32(0.1)
    splits, chunk = 1, K
    if macos_major == 26:
        # macOS 26.5.2, 0x18faaf6ac–0x18faaf754: repeated LSR #1,
        # followed by rounding to 16 (Apple7/8) or 32 (Apple9+) elements.
        if ratio <= threshold or K <= 2048:
            return 1, K
        while True:
            chunk //= 2
            if _f32(_f32(chunk) / area) <= threshold:
                break
        quantum = 32 if generation >= 18 else 16
        chunk = _ceildiv(max(chunk, 1), quantum) * quantum
        return _ceildiv(K, chunk), chunk
    special = generation == 17 and cores > 64 and N <= 256 and K == 2048
    if ratio > threshold and K > 2048:
        while _f32(_f32(chunk) / area) > threshold:
            splits *= 2
            chunk = _ceildiv(K, splits)
    if chunk > 8192:
        while chunk > 8192:
            splits *= 2
            chunk = _ceildiv(K, splits)
    elif special:
        chunk = 512 if N > 127 else 256
    elif ratio <= threshold or K <= 2048:
        return 1, K
    quantum = 32 if generation >= 18 else 16
    chunk = _ceildiv(max(chunk, 1), quantum) * quantum
    return _ceildiv(K, chunk), chunk


def _square_tiles(size, cores, half):
    """The square-shape branch, before the Apple9+ override."""
    pad32 = (-size) % 32
    pad64 = (-size) % 64
    pad48 = (-size) % 48
    if cores <= 64:
        if not half:
            n, k = 48, 24
            if pad48 and pad32 < pad48:
                groups = _ceildiv(size, 64) ** 2
                if _f32(_f32(groups) / _f32(cores)) <= 12.0:
                    n, k = (32, 32) if pad32 <= pad64 else (64, 16)
                else:
                    n, k = (64, 16) if pad64 <= pad32 else (32, 32)
            return 64, n, k
        waste64 = ((pad64 * (size + 31)) >> 5) + ((pad32 * (size + 63)) >> 6)
        waste48 = 2 * ((size + 47) * pad48 // 48)
        return (64, 32, 32) if waste64 <= waste48 else (48, 48, 24)
    if half:
        return (48, 48, 24) if not pad48 or size * pad48 <= size * pad32 else (48, 32, 32)
    return (64, 32, 16) if size * (pad64 + pad32) <= 2 * size * pad48 else (48, 48, 16)


def edge_fragments(size, tile, simds):
    """MatProdSxLim/SyLim: fragments per SIMD group in the last partial tile.

    EncodeMatrixMultiplyLinkingA14 sets fcPartialOuterM/N (function constant
    115 bits 7 and 8) only when ``0 < size % tile < tile - 8*simds``; the limit
    is then ``ceil(remainder / (8*simds))`` (bits 9-11 and 12-14). Otherwise the
    full ``tile / (8*simds)`` count applies (26.3 0x18f726a38, 26.5.2
    0x18faaebfc, 27.2 0x190c778a4).
    """
    remainder, step = size % tile, simds * 8
    return (remainder + step - 1) // step if 0 < remainder < tile - step else tile // step


@dataclass(frozen=True)
class A14Plan:
    kernel: str
    tile_m: int
    tile_n: int
    tile_k: int
    simd_m: int
    simd_n: int
    splits: int
    k_per_split: int
    stage_a: bool
    stage_b: bool
    grid: tuple[int, int, int]
    tgmem: tuple[int, int]
    reduction_grid: tuple[int, int, int] | None

    @property
    def threads(self):
        return 32 * self.simd_m * self.simd_n


@dataclass(frozen=True)
class VectorPlan:
    kernel: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    tgmem: tuple[int, int]
    unroll_k: int
    row_tile: int = 1  # LoRA fcTileM: rows loaded together; all SIMDs compute all rows.


# EncodeMatrixVectorMultiply's LORADOWN tables, indexed by narrow size 2..15.
# macOS 26.5.2 __const: 0x18fbb9f48, 0x18fbb9fb8, 0x18fbba028.
_LORA_SIMDS = (1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4)
_LORA_ROWS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 7, 5)
_LORA_UNROLL = (16, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4)


def vector_plan(M, N, K, dtype, *, dev, trans_a=False, trans_b=False, macos_major=26, batch=1):
    """EncodeMatrixVectorMultiply for dense operands, without fused norms.

    The Apple GPU path uses 32 lanes and SIMD reduction. As with gemm_plan,
    the caller must first apply vector routing and PyTorch's own fallbacks.
    """
    narrow, wide = min(M, N), max(M, N)
    if not 1 <= narrow <= 16:
        raise ValueError('The vector encoder requires a dimension in 1..16')
    narrow_n = N <= M
    flat = trans_a if narrow_n else not trans_b
    if dtype is torch.float16 and batch == 1 and wide <= 128 and narrow_n and not trans_a and trans_b:
        sg, rows, unroll = ((_LORA_SIMDS[narrow - 2], _LORA_ROWS[narrow - 2], _LORA_UNROLL[narrow - 2])
                            if 2 <= narrow <= 15 else (1, 1, 16))
        sg <<= K > 8192
        while K < sg * 32 * unroll * 4 and sg * unroll > 1:
            if sg > 1:
                sg //= 2
            else:
                unroll //= 2
        return VectorPlan('LORADOWN_gemv_kernel', (1, wide, 1), (sg * 32, 1, 1), (0, 0), unroll, rows)
    unroll = 1
    if K >= 2048:
        unroll = 8 if narrow < 4 else (4 if narrow < 6 else (2 if narrow == 6 else 1))
        if K % 32 == 0 and K % (unroll * 128):
            # Power of two dividing floor(K/128), from XOR(x-1, x)/CLZ.
            q = K >> 7
            unroll = min(unroll, q & -q)
    if narrow < 8 and flat:
        if wide < dev.cores * 8:
            gx, gy = 1, 32
        elif wide < dev.cores * 16:
            gx, gy = 2, 16
        elif wide < dev.cores * 32:
            gx, gy = 4, 8
        else:
            gx, gy = 8, 4
        groups = _ceildiv(wide, gx * 4)
        max_simds = min(16, 32768 // (32 * narrow * 4 * 4))
        # macOS 27 doubles the coverage target for K <= 256 * cores.
        coverage = dev.cores * (32 if macos_major == 27 and K <= dev.cores * 256 else 16)
        while groups * batch * (gx * gy // 32) < coverage and gx * gy * 2 // 32 <= max_simds and gy * 2 < K:
            gy *= 2
        gy = max(gy, 32 // gx)
        grid, mem = (groups, 1, 1), narrow * gx * gy * 16
    elif not narrow_n and flat:
        gx = gy = 16
        grid, mem = (1, _ceildiv(wide, 16), 1), narrow * 16 * 16 * 4
    else:
        gx, gy = 32, 1
        rows = 4 if narrow == 1 and K <= 256 else 1
        grid, mem = (1, _ceildiv(wide, rows), 1), 0
    group = (narrow - 1) // 4
    name = f'NDArrayVectorMatrixMultiply_{group * 4 + 1}_{group * 4 + 4}'
    return VectorPlan(name, (*grid[:2], batch), (gx, gy, 1), (mem, 0), unroll)


def gemm_plan(M, N, K, dtype=torch.bfloat16, *, dev: DeviceInfo | None = None,
              macos_major: int | None = None, batch=1):
    """Select a dense GEMM or equal-batch BMM after vector/fallback routing.

    Layout does not change this encoder's tile selection for these inputs.
    Vector routing, broadcast/rank-4 batch flattening, large-input chunking,
    and PyTorch's own kernels are outside this low-level function.
    """
    if min(M, N, K, batch) <= 0:
        raise ValueError('GEMM dimensions must be positive')
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('A14 selection supports float16, bfloat16 and float32')
    dev = dev or device_info()
    if macos_major is None:
        macos_major = int(platform.mac_ver()[0].split('.')[0])
    if macos_major not in (26, 27):
        raise ValueError('Selection is recovered for macOS 26 and 27 only')
    gen, cores = dev.generation, dev.cores
    if not 11 <= gen <= 23 or cores <= 0:
        raise ValueError('Expected an M1–M5 MPS generation and positive core count')
    splits, chunk = _split_k(M, N, K, gen, cores, macos_major)
    if M <= 32 and N >= 512:
        bm, bn, sm, sn = 32, 128, 2, 4
    elif N <= 32:
        bm = 128 if M >= 512 else (64 if M > 32 else 32)
        bn, sm, sn = 32, (4 if M >= 512 else 2), 2
    else:
        bm, bn, sm, sn = (64 if M > 32 else 32), 64, 2, 2
    bk = 16
    ultra = macos_major >= 27 and gen == 17 and cores > 64
    if ultra:
        bk = 16 if K <= 10240 else 32
        if N <= 256 and K == 2048:
            bm, bn, bk, sm, sn = (64 if N == 256 else 32), 32, 32, 2, 2
    if M == N == K:
        bm, bn, bk = _square_tiles(M, cores, dtype is torch.float16)
    else:
        # The encoder uses splits twice here, before the final K-tile rounding.
        work = splits * splits * M * N * batch
        capacity = 4 * cores
        if work <= capacity * bm * bn:
            bm //= 2
            shrink_n = True
            while work <= capacity * bm * bn and (bm > 16 or bn > 16):
                if bn > 16 and shrink_n:
                    bn //= 2
                elif bm > 16:
                    bm //= 2
                shrink_n = not shrink_n
            sm = sn = 2
    stage_a, stage_b = True, gen < 18
    if gen >= 18:
        sm = sn = 2
        if M == N == K:
            bm = 32
            if dtype in (torch.float16, torch.bfloat16):
                bn, bk, stage_a = 32, 16, False
            else:
                bn, bk = 64, 32
        else:
            bk = 32
    if splits > 1:
        chunk = _ceildiv(chunk, bk) * bk
        splits = _ceildiv(K, chunk)
    size = 4 if dtype is torch.float32 else 2
    name = {torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float'}[dtype]
    edge = M % bm != 0 or N % bn != 0 or K % bk != 0
    kernel = ('NDArrayMatrixMultiplyNNA14' + ('_EdgeCase' if edge else '') if gen < 18
              else f'mmul_kernel_a16_{name}_{name}_float')
    grid = (_ceildiv(N, bn), _ceildiv(M, bm), splits * batch)
    if ultra:
        grid = (grid[0] * grid[1] * grid[2], 1, 1)
    reduction = (_ceildiv(N, 128 if N % 4 == 0 else 32), _ceildiv(M, 32), batch) if splits > 1 else None
    return A14Plan(kernel, bm, bn, bk, sm, sn, splits, chunk, stage_a, stage_b,
                   grid, (bn * bk * size if stage_b else 0, bm * bk * size if stage_a else 0), reduction)
