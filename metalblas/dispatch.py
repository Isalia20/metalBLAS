"""High-level matmul dispatcher.

`matmul(a, b)` picks the best kernel variant (tile size, GEMV vs GEMM, simd vs
M5 tensor unit, swizzle, etc.) given the input shapes and dtype.
"""
from __future__ import annotations

import math
import torch

from . import kernels


_DTYPE_NAME = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

# Mapping dtype -> (in_t, acc_t, out_t) for the standard "fast" matmul:
# fp32 inputs use fp32 accumulation; fp16 inputs use fp32 accumulation (tensor-core style).
_PROFILE = {
    torch.float32: ("float", "float", "float"),
    torch.float16: ("half",  "float", "half"),
    torch.bfloat16: ("bfloat", "float", "bfloat"),
}


# Scalars are passed directly to the kernel — no tensor allocation needed.


def _threadgroup_bytes(BM, BN, BK, dtype_bytes):
    pad = max(1, 16 // dtype_bytes)
    lda = BK + pad
    ldb = BN + pad
    return (BM * lda + BK * ldb) * dtype_bytes


def _pick_simd_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int, int, int]:
    """Pick (BM, BN, BK, WM, WN) for the simdgroup_matrix kernel.

    Constraint: TM * TN = (BM/(8*WM)) * (BN/(8*WN)) C-fragments per warp must
    stay small (≤16) to fit in registers, otherwise we spill and lose 10x.
    """
    dtype_bytes = 4 if dtype == torch.float32 else 2
    # (BM, BN, BK, WM, WN) — we enumerate sensible tiles and pick the best.
    candidates = [
        (128, 128, 16, 4, 4),  # TM=TN=4, 16 frags/warp, 16 warps
        (128, 128, 32, 4, 4),
        ( 64, 128, 16, 2, 4),  # TM=TN=4, 8 warps
        (128,  64, 16, 4, 2),
        ( 64,  64, 32, 2, 2),  # TM=TN=4, 4 warps
        ( 64,  64, 16, 2, 2),
        ( 32, 128, 16, 1, 4),  # TM=4, TN=4, 4 warps
        (128,  32, 16, 4, 1),
        ( 32,  64, 16, 1, 2),  # TM=TN=4, 2 warps
        ( 64,  32, 16, 2, 1),
        ( 32,  32, 32, 1, 1),  # 4 frags, 1 warp
        ( 32,  32, 16, 1, 1),
        ( 16,  16, 16, 1, 1),  # tiny
    ]
    best = None
    for (BM, BN, BK, WM, WN) in candidates:
        bytes_needed = _threadgroup_bytes(BM, BN, BK, dtype_bytes)
        if bytes_needed > 32 * 1024:
            continue
        if BK % 8 != 0 or BM % (8 * WM) != 0 or BN % (8 * WN) != 0:
            continue
        TM = BM // (8 * WM); TN = BN // (8 * WN)
        if TM * TN > 16:
            continue  # register pressure -> spills
        if WM * WN > 16:
            continue  # threadgroup too big
        # bench-validated heuristic: warps must be at most 8 for small problems
        tiles_m = (M + BM - 1) // BM
        tiles_n = (N + BN - 1) // BN
        total_tiles = tiles_m * tiles_n
        if total_tiles < 4 and BM * BN > 64 * 64:
            continue
        waste = (tiles_m * BM * tiles_n * BN) / max(1, M * N)
        # Prefer big tiles for big problems (better K reuse), small tiles otherwise.
        ops = M * N * K
        if ops > 256 * 1024**2:
            score = (BM * BN, -waste, BK)
        else:
            score = (-abs(BM * BN - max(M, N) * 16), -waste)
        if best is None or score > best[0]:
            best = (score, (BM, BN, BK, WM, WN))
    if best is None:
        return (16, 16, 16, 1, 1)
    return best[1]


def _pick_m5_tensor_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """Pick (BM, BN, NSG) for the m5_tensor_gemm kernel.

    Sweep on M5 Pro (warmup 100, iters 500) — winners by regime:
      - Tiny K (≤ 256) with large M, N: (32, 128, 4) — more tiles amortize
        per-TG fixed cost when each K-loop is short.
      - Non-divisible large (≥ 1024 with M%64 or N%64 != 0): (128, 128, 8)
        — fewer total tiles, big interior, edges absorbed.
      - Deep-K (K ≥ 2*max(M, N)): (64, 128, 4) — wider BN amortizes K-loop.
      - Very small (M, N ≤ 256): (32, 128, 4) — keeps a few large TGs.
      - Small-medium (M, N ≤ 768): (32, 64, 4) — small tile, more TGs.
      - Default for divisible large: (64, 64, 2) — light TG threading.
      - Fallback for awkward shapes: (64, 64, 4) when small, (64, 128, 4) otherwise.

    The kernel handles partial edge tiles natively (cT.store(mC) clips to
    the slice's extents; Metal zero-pads OOB buffer reads).
    """
    m_div_64  = (M % 64  == 0)
    n_div_64  = (N % 64  == 0)
    n_div_128 = (N % 128 == 0)
    m_div_32  = (M % 32  == 0)
    n_div_32  = (N % 32  == 0)

    # Tiny K with big M, N: smaller BM = more tiles in flight.
    if K <= 256 and M >= 1024 and N >= 1024 and m_div_32 and n_div_128:
        return (32, 128, 4)

    # Non-divisible large shapes: (128, 128, 8) amortizes edge waste.
    if (M >= 1024 and N >= 1024
            and not (m_div_64 and n_div_64)
            and (M + 127) // 128 * (N + 127) // 128 >= 8):
        return (128, 128, 8)

    # Deep-K: wider BN amortizes K-loop overhead per tile.
    if K >= 2 * max(M, N) and m_div_64 and n_div_128:
        return (64, 128, 4)

    # Very small: (32, 128, 4) keeps a small number of bigger TGs.
    if M <= 256 and N <= 256 and m_div_32 and n_div_128:
        return (32, 128, 4)

    # Small-medium: (32, 64, 4) — more tiles, NSG=4 keeps cores busy.
    if M <= 768 and N <= 768 and m_div_32 and n_div_32:
        return (32, 64, 4)

    # Default for divisible — empirical winner across most large regimes.
    if m_div_64 and n_div_64:
        return (64, 64, 2)

    # Awkward divisibility.  Small shapes do better with (64, 64, 4); large
    # shapes with (64, 128, 4) — (64, 64, 4) wastes too little tile width.
    if M <= 768 and N <= 768:
        return (64, 64, 4)
    return (64, 128, 4)


def _pick_m5_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int, int, int, bool]:
    """Pick (BM, BN, BK, WM, WN, dbuf) for the m5_gemm kernel (16x32x16 frags).

    Heuristic is hand-tuned from sweeps on M5 Pro.  Falls back to a smaller
    tile if the chosen one would exceed threadgroup memory or register budget.
    `dbuf=True` enables double-buffered K-loop tiles (2x threadgroup mem) to
    overlap global→threadgroup loads with the MMA inner loop — wins on
    bandwidth-bound large K.
    """
    is_lp = (dtype != torch.float32)        # low-precision (fp16 or bf16)
    dtype_bytes = 2 if is_lp else 4
    ops = float(M) * N * K
    aspect = max(M, N) / max(1, min(M, N))

    # Primary candidate based on problem size and dtype.
    # Empirically (sweep on M5 Pro):
    #   - Small problems (≤ 1024³ lp): bigger tile, no dbuf, BK=16/32 is enough.
    #   - Large bf16/fp16 problems: prefer double-buffered (128, 64, 32, 4, 2)
    #     or (64, 128, 32, 2, 4) — the dbuf pipeline overlaps loads with MMA.
    #   - K-dominated low-precision: (64, 128, 32, 2, 4) double-buffered.
    if is_lp:
        if ops <= 128 ** 3:
            primary = (32, 32, 16, 1, 1, False)
        elif ops <= 256 ** 3:
            primary = (64, 64, 32, 2, 2, False)
        elif K >= 2 * max(M, N):
            primary = (64, 128, 64, 2, 4, False)
        else:
            primary = (128, 64, 64, 4, 2, False)
    else:
        if ops <= 256 ** 3:
            primary = (32, 32, 16, 1, 1, False)
        elif ops <= 768 ** 3:
            primary = (128, 128, 16, 4, 4, False)
        elif ops <= 1536 ** 3:
            primary = (128, 64, 32, 4, 2, False)
        elif ops <= 3072 ** 3:
            primary = (64, 64, 16, 2, 2, False)
        else:
            primary = (64, 128, 32, 2, 4, False)

    # Tall-skinny adjustments (extreme aspect)
    if aspect >= 4 and max(M, N) >= 1024:
        if M > N:
            primary = (128, 64, 32, 4, 2, False)
        else:
            primary = (64, 128, 32, 2, 4, False)

    # Validate / fallback chain (try primary, then smaller variants)
    fallbacks = [
        primary,
        (128, 64, 32, 4, 2, True),
        (64, 128, 32, 2, 4, True),
        (128, 64, 64, 4, 2, False),
        (64,  64, 32, 2, 2, False),
        (64,  64, 16, 2, 2, False),
        (32,  64, 16, 1, 2, False),
        (32,  32, 16, 1, 1, False),
        (16,  32, 16, 1, 1, False),
    ]
    for cand in fallbacks:
        BM, BN, BK, WM, WN, dbuf = cand
        if BM % (16 * WM) != 0:    continue
        if BN % (32 * WN) != 0:    continue
        if BK % 16 != 0:           continue
        mult = 2 if dbuf else 1
        if _threadgroup_bytes(BM, BN, BK, dtype_bytes) * mult > 32 * 1024: continue
        TM = BM // (16 * WM); TN = BN // (32 * WN)
        if TM * TN > 8: continue
        if WM * WN > 16: continue
        # Avoid huge wasted output (tiny problem with very big tile)
        tiles_m = (M + BM - 1) // BM
        tiles_n = (N + BN - 1) // BN
        if tiles_m * tiles_n == 0: continue
        if BM > 2 * M or BN > 2 * N: continue
        return (BM, BN, BK, WM, WN, dbuf)
    return (16, 32, 16, 1, 1, False)


def _round_swizzle_log(tiles_m: int, tiles_n: int) -> int:
    # Tiny swizzle to improve L2 locality on large problems.
    if tiles_m * tiles_n < 32:
        return 0
    return 2


def _resolve_inputs(a: torch.Tensor, b: torch.Tensor):
    """Return (A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b)
    such that view + (trans flag) describe how the kernel reads the original
    memory.  We avoid materializing copies."""
    assert a.dim() == 2 and b.dim() == 2, "matmul currently expects 2-D inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"

    # Discover whether each input is row-major (stride: (K,1)) or col-major (stride: (1,M))
    sa = a.stride()
    sb = b.stride()
    # A row-major contig: (K, 1)
    # A col-major (transposed) contig: (1, M)
    if sa[1] == 1 and sa[0] >= K:
        trans_a = False
        lda = sa[0]
        A_view = a
    elif sa[0] == 1 and sa[1] >= M:
        trans_a = True
        lda = sa[1]
        A_view = a
    else:
        A_view = a.contiguous()
        trans_a = False
        lda = K

    if sb[1] == 1 and sb[0] >= N:
        trans_b = False
        ldb = sb[0]
        B_view = b
    elif sb[0] == 1 and sb[1] >= K:
        trans_b = True
        ldb = sb[1]
        B_view = b
    else:
        B_view = b.contiguous()
        trans_b = False
        ldb = N
    return A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b


def matmul(a: torch.Tensor, b: torch.Tensor, *,
           backend: str | None = None,
           tile: tuple | None = None,
           swizzle_log: int | None = None,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute C = A @ B on the MPS device with our hand-tuned Metal kernels.

    Parameters
    ----------
    a, b : torch.Tensor on device 'mps'
    backend : "auto" (default), "simd", "m5", "gemv"
    tile  : (BM, BN, BK, WM, WN) – override the tile heuristic
    swizzle_log : override the swizzle (0, 1, 2 …)
    out   : optional preallocated output of shape (M, N)
    """
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype
    dtype = a.dtype
    if dtype not in _PROFILE:
        raise NotImplementedError(f"dtype {dtype} not supported")

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)

    if out is None:
        out = torch.empty((M, N), dtype=dtype, device="mps")
    else:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps"
    ldc = out.stride(0)

    in_t, acc_t, out_t = _PROFILE[dtype]
    M_, N_, K_ = int(M), int(N), int(K)
    lda_, ldb_, ldc_ = int(lda), int(ldb), int(ldc)

    backend = backend or "auto"

    if backend == "auto":
        is_lp = (dtype != torch.float32)
        # Big-N GEMV (e.g. lm_head 1×32000×4096): the m5_tensor kernel reads
        # the 63 OOB rows as zero (Metal bounds-check) and writes only to the
        # single valid row via cT.store(mC), so a "padded" matmul beats the
        # dedicated gemv kernel on bandwidth-bound shapes.  Empirical crossover
        # is around N=8000 for K=4096.
        if (M == 1 and N >= 8192 and K >= 1024 and is_lp
                and not trans_a and not trans_b):
            backend = "m5_tensor"
        # GEMV fast path for everything else.  N==1 stays on gemv_nt which
        # already achieves bandwidth-peak for that pattern.
        elif M == 1 and N >= 16:
            backend = "gemv"
        elif N == 1 and M >= 16:
            backend = "gemv"
        else:
            # m5_tensor (MPP op.run path) wins on bf16/fp16 thanks to internal
            # load orchestration.  The kernel's cT.store(mC) and Metal's bounds-
            # checked buffer reads handle non-divisible shapes natively, so we
            # don't require tile-aligned M/N — just enough work to fill the GPU.
            tiles_m = (int(M) + 63) // 64
            tiles_n = (int(N) + 63) // 64
            if (is_lp and M >= 64 and N >= 64 and K >= 64
                    and tiles_m * tiles_n >= 8
                    and not trans_a and not trans_b):
                backend = "m5_tensor"
            else:
                backend = "m5"

    if backend == "gemv":
        return _dispatch_gemv(A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, dtype, out)

    if backend == "simd":
        BM, BN, BK, WM, WN = tile if tile else _pick_simd_tile(M_, N_, K_, dtype)
        mn_aligned = (M_ % BM == 0) and (N_ % BN == 0)
        k_aligned  = (K_ % BK == 0)
        tiles_m = (M_ + BM - 1) // BM
        tiles_n = (N_ + BN - 1) // BN
        swz = swizzle_log if swizzle_log is not None else _round_swizzle_log(tiles_m, tiles_n)
        fn, _ = kernels.simd_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN,
                                  trans_a, trans_b, mn_aligned, k_aligned, swz)
        group_size = (WM * WN * 32, 1, 1)
        tile_factor = 1 << swz
        tn_swz = tiles_n * tile_factor
        tm_swz = (tiles_m + tile_factor - 1) // tile_factor
        total = (group_size[0] * tn_swz, tm_swz, 1)
        fn(A_view, B_view, out, M_, N_, K_, lda_, ldb_, ldc_,
           threads=total, group_size=group_size)
        return out

    if backend == "m5_tensor":
        # Tile = (BM, BN, NSG)
        if tile is None:
            BM, BN, NSG = _pick_m5_tensor_tile(M_, N_, K_, dtype)
        else:
            assert len(tile) >= 3
            BM, BN, NSG = tile[0], tile[1], tile[2]
        tiles_m = (M_ + BM - 1) // BM
        tiles_n = (N_ + BN - 1) // BN
        mn_aligned = (M_ % BM == 0) and (N_ % BN == 0)
        # MPP path has its own internal scheduling; an external swizzle
        # hurts it (empirically loses 15-20% TF).  Leave it at 0.
        swz = swizzle_log if swizzle_log is not None else 0
        fn, _ = kernels.m5_tensor_gemm(
            in_t, out_t, BM, BN, NSG, trans_a, trans_b,
            relaxed=True, swizzle_log=swz, mn_aligned=mn_aligned,
        )
        group_size = (NSG * 32, 1, 1)
        tile_factor = 1 << swz
        tn_swz = tiles_n * tile_factor
        tm_swz = (tiles_m + tile_factor - 1) // tile_factor
        total = (group_size[0] * tn_swz, tm_swz, 1)
        fn(A_view, B_view, out, M_, N_, K_, lda_, ldb_, ldc_,
           threads=total, group_size=group_size)
        return out

    if backend == "m5":
        pad = None
        if tile is None:
            BM, BN, BK, WM, WN, dbuf = _pick_m5_tile(M_, N_, K_, dtype)
        elif len(tile) == 7:
            BM, BN, BK, WM, WN, dbuf, pad = tile
        elif len(tile) == 6:
            BM, BN, BK, WM, WN, dbuf = tile
        else:
            BM, BN, BK, WM, WN = tile
            dbuf = False
        mn_aligned = (M_ % BM == 0) and (N_ % BN == 0)
        k_aligned  = (K_ % BK == 0)
        tiles_m = (M_ + BM - 1) // BM
        tiles_n = (N_ + BN - 1) // BN
        swz = swizzle_log if swizzle_log is not None else _round_swizzle_log(tiles_m, tiles_n)
        fn, _ = kernels.m5_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN,
                                trans_a, trans_b, mn_aligned, k_aligned,
                                relaxed=True, swizzle_log=swz, dbuf=dbuf, pad=pad)
        group_size = (WM * WN * 32, 1, 1)
        tile_factor = 1 << swz
        tn_swz = tiles_n * tile_factor
        tm_swz = (tiles_m + tile_factor - 1) // tile_factor
        total = (group_size[0] * tn_swz, tm_swz, 1)
        fn(A_view, B_view, out, M_, N_, K_, lda_, ldb_, ldc_,
           threads=total, group_size=group_size)
        return out

    raise ValueError(f"unknown backend {backend}")


def _dispatch_gemv(A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, dtype, out):
    in_t, acc_t, out_t = _PROFILE[dtype]
    M, N, K = int(M), int(N), int(K)
    lda, ldb = int(lda), int(ldb)
    NWARPS = 4
    ROWS_PER_SG = 1
    BLOCK_N = 32
    tg_size = NWARPS * 32

    def gemv_nt_call(matrix, vec, out_v, rows, k, ld):
        fn, _ = kernels.gemv_nt(in_t, acc_t, out_t, ROWS_PER_SG=ROWS_PER_SG, NWARPS=NWARPS)
        rows_per_tg = NWARPS * ROWS_PER_SG
        n_groups = (rows + rows_per_tg - 1) // rows_per_tg
        fn(matrix, vec, out_v, rows, k, ld,
           threads=(tg_size * n_groups, 1, 1), group_size=(tg_size, 1, 1))

    def gemv_t_call(matrix, vec, out_v, cols, k, ld):
        fn, _ = kernels.gemv_t(in_t, acc_t, out_t, BLOCK_N=BLOCK_N, NWARPS=NWARPS)
        n_groups = (cols + BLOCK_N - 1) // BLOCK_N
        fn(matrix, vec, out_v, cols, k, ld,
           threads=(tg_size * n_groups, 1, 1), group_size=(tg_size, 1, 1))

    if M == 1:
        if trans_b:
            gemv_nt_call(B_view, A_view.view(-1), out.view(-1), N, K, ldb)
        else:
            gemv_t_call(B_view, A_view.view(-1), out.view(-1), N, K, ldb)
        return out
    elif N == 1:
        if trans_a:
            gemv_t_call(A_view, B_view.view(-1), out.view(-1), M, K, lda)
        else:
            gemv_nt_call(A_view, B_view.view(-1), out.view(-1), M, K, lda)
        return out
    else:
        raise ValueError("gemv backend requires M==1 or N==1")


# Convenience alias
gemm = matmul
