"""High-level matmul dispatcher.

`matmul(a, b)` picks the best kernel variant (tile size, GEMV vs GEMM, simd vs
M5 tensor unit, swizzle, etc.) given the input shapes and dtype.
"""
from __future__ import annotations

import math
import os
import sys
import time
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


# --- GEMV hot-path caches -------------------------------------------------
# Small GEMV is CPU-bound: the kernel runs in a few µs, so per-call Python
# (compile-cache lookups, closures, allocations) is what caps the speedup.
# Resolve the kernel handles once per dtype so the hot path is just arithmetic
# plus one enqueue.  gemv_t splits K across NWARPS simdgroups *within* the
# threadgroup (reduced in threadgroup memory) — that fills each core's
# 1024-thread budget even when there are few column-blocks, which is what
# starved small-N GEMV (1×512×4096 was 0.21× with 4 warps × 16 TGs; at 32 warps
# it's >1×, and latency-bound shapes like 1024² hit ~2×).  One dispatch, no
# cross-TG reduction.  We keep two variants because the sweet spot moves with
# the column-block count (see _gemv_pick).
_GEMV_HANDLES: dict = {}

# gemv_t variants we compile, keyed by (VEC, NWARPS).  VEC = columns per lane:
# a warp's 32-lane coalesced read spans 32*VEC consecutive cols, so VEC=2 (bf16)
# covers a full 128-B cache line and VEC=4 covers two — eliminating the half-line
# cross-TG re-fetch that left the K≥4096 band at 0.87-0.97×.  fp32 only needs
# VEC=1 (its 4-byte loads already span a full line), so it keeps the proven
# NWARPS={16,32} split; bf16/fp16 add the vectorized (2,32)/(4,16)/(4,8) tiles.
_GEMV_T_VARIANTS = {
    torch.float32:  [(1, 16), (1, 32)],
    torch.float16:  [(1, 32), (2, 32), (2, 16), (4, 16), (4, 8)],
    torch.bfloat16: [(1, 32), (2, 32), (2, 16), (4, 16), (4, 8)],
}


def _gemv_handles(dtype):
    h = _GEMV_HANDLES.get(dtype)
    if h is None:
        in_t, acc_t, out_t = _PROFILE[dtype]
        gt = {}
        for (vec, nw) in _GEMV_T_VARIANTS[dtype]:
            fn, _ = kernels.gemv_t(in_t, acc_t, out_t, 32 * vec, nw, vec)
            gt[(vec, nw)] = fn
        nt, _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4)
        h = {"gt": gt, "nt": nt}
        _GEMV_HANDLES[dtype] = h
    return h


def _gemv_pick(gh, cols, ldb, dtype, vec_ok=True, k_big=True):
    """Return (gemv_t_fn, threadgroup_size, vec) for a `cols`-wide GEMV.

    bf16/fp16: scale VEC (columns/lane) with N so each warp covers full cache
    lines, while keeping the threadgroup count high enough to fill the 16 cores —
    small N stays at VEC=1 (16 TGs at N=512); the K≥4096 band moves to VEC=2/4.
    NWARPS (the within-TG K-split) shrinks as VEC grows so wide tiles don't
    over-subscribe the reduction (3072×4096 wants VEC=4/NW=8, 1.01× vs 0.89×).

    The widest tile (VEC=4 ⇒ N/128 threadgroups) is gated on `k_big` (K≥2048):
    with a short K there isn't enough per-TG work to hide latency at half the
    core count, so 2048×1024 (16 TGs at VEC=4 = half occupancy → 382 GB/s) wants
    VEC=2 instead (32 TGs, full occupancy → 427 GB/s).

    fp32: VEC=1 (4-byte loads already span a line); keep the proven NWARPS split —
    512<cols≤1024 peaks at 16, else 32.

    Vectorized loads need `ldb % vec == 0` (and a 16-B-aligned base, guaranteed
    for the contiguous fast path); `vec_ok=False` forces VEC=1 for view inputs.
    """
    gt = gh["gt"]
    if dtype is torch.float32:
        ng = (cols + 31) // 32
        if 16 < ng <= 32:
            return gt[(1, 16)], 512, 1
        return gt[(1, 32)], 1024, 1
    if vec_ok and k_big and cols >= 2560:
        vec, nw = 4, 8
    elif vec_ok and k_big and cols >= 1280:
        vec, nw = 4, 16
    elif vec_ok and cols >= 1024:
        vec, nw = 2, 16     # ≥16 TGs already fill the cores; NW=32 just
                            # over-subscribes the K-reduction (1024: 453 vs 444 GB/s)
    elif vec_ok and cols > 512:
        vec, nw = 2, 32     # <16 TGs: more warps to fill cores (768: 416 vs 373)
    else:
        vec, nw = 1, 32
    # Clamp VEC to the column stride's alignment.
    if vec == 4 and (ldb & 3):
        vec, nw = (2, 32) if not (ldb & 1) else (1, 32)
    elif vec == 2 and (ldb & 1):
        vec, nw = 1, 32
    return gt[(vec, nw)], nw * 32, vec


# Fast-path launch plan, memoized by (dtype, N): the kernel handle and the
# thread/group tuples are a pure function of (dtype, N) for contiguous GEMV
# (ldb == N), so we resolve them once and the hot path becomes one dict hit.
_GEMV_PLAN: dict = {}


def _gemv_plan(dtype, N, k_big):
    key = (dtype, N, k_big)
    plan = _GEMV_PLAN.get(key)
    if plan is None:
        gt, tg, vec = _gemv_pick(_gemv_handles(dtype), N, N, dtype, k_big=k_big)
        ng = (N + 32 * vec - 1) // (32 * vec)
        plan = (gt, (tg * ng, 1, 1), (tg, 1, 1))
        _GEMV_PLAN[key] = plan
    return plan


# Memoized m5_tensor launch plan for the contiguous-GEMM fast path, keyed by
# (dtype, M, N, K).  Mirrors the backend=="m5_tensor" block for the untransposed,
# swizzle-0 case (lda=K, ldb=N, ldc=N), so the hot path skips _resolve_inputs,
# the tile pick, and the shader-cache lookup — just a dict hit + enqueue.
_GEMM_PLAN: dict = {}
# The (BM,BN,NSG) actually chosen per (dtype,M,N,K) — the autotuned winner when a
# shape was probed, else the heuristic primary.  Lets diagnostics report the real
# tile (e.g. floor_probe) instead of re-deriving it from the heuristic.
_GEMM_TILE: dict = {}

# --- Runtime tile autotuner ------------------------------------------------
# A static `(M,N,K)→tile` heuristic can't track the few shapes where the best
# tile flips on K-depth and aspect without overfitting (every session's notes
# record a hand-gate that then regressed a neighbour: the tall deep-K guard is
# right at K=2·max but wrong at K≥4·max; the 8:1-aspect 8192×1024×8192 wants a
# thin BM=48 the (64,64,2) default never offers).  So for those *ambiguous*
# regimes the heuristic stops at a short CANDIDATE LIST and we measure on the
# real operands the first time a shape is seen, caching the winner in _GEMM_PLAN
# (followups TODO #2, option b — the general fix that also catches future
# outliers).  This is provably ≥ the heuristic: its pick is always candidate 0
# and we only switch off it for a margin-beating win.  Confident regimes and
# fp32 (which already wins 2–3× everywhere) return a single candidate, so the
# common shapes pay zero probe cost — only a handful of shapes ever probe, and
# only on their first call (amortized to nothing for any repeated-shape
# workload).  Disable with METALBLAS_AUTOTUNE=0.
_AUTOTUNE = os.environ.get("METALBLAS_AUTOTUNE", "1") != "0"
# Only abandon the heuristic primary for a margin-beating win.  Set above the
# run-to-run thermal/scheduler noise floor of the smallest probed kernels (~14µs)
# so a near-tie can't demote the trusted default: 512³'s tiles sit within ~2% and
# (32,64,2) is already its best, whereas 640³'s (16,64,2) is a solid ~5% win — 3%
# keeps the former and captures the latter.  The large deep-K/high-aspect wins are
# all ≥6%, far clear of this.
_AUTOTUNE_MARGIN = 0.03


def _build_m5t_plan(dtype, M, N, K, BM, BN, NSG):
    """Compile one m5_tensor tile and return its (fn, threads, group) launch plan
    for the packed, untransposed, swizzle-0 case (lda=K, ldb=N, ldc=N)."""
    in_t, _, out_t = _PROFILE[dtype]
    mn_aligned = (M % BM == 0) and (N % BN == 0)
    fn, _ = kernels.m5_tensor_gemm(in_t, out_t, BM, BN, NSG, False, False,
                                   relaxed=True, swizzle_log=0, mn_aligned=mn_aligned)
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    return (fn, (NSG * 32 * tiles_n, tiles_m, 1), (NSG * 32, 1, 1))


def _m5_tensor_tile_candidates(M, N, K, dtype):
    """Tiles to consider for (M,N,K,dtype), heuristic primary first.

    A single-element list means "the heuristic is confident, don't probe".  A
    longer list marks an ambiguous regime where the autotuner measures and
    caches the winner.  The primary is exactly `_pick_m5_tensor_tile`, so non-
    autotuned paths (fp32, the general dispatcher) are byte-for-byte unchanged.
    """
    primary = _pick_m5_tensor_tile(M, N, K, dtype)
    # fp32 wins 2–3× on every shape — never worth a probe.
    if dtype == torch.float32:
        return [primary]
    mx = max(M, N)

    # Confidently-handled regimes: padded GEMV, tiny-K, and tiny problems all
    # have a single decisive tile (see _pick_m5_tensor_tile) — no probe.
    if M == 1 or N == 1:
        return [primary]
    if K <= 256 and M >= 1024 and N >= 1024:
        return [primary]
    if mx <= 256:
        return [primary]

    if mx <= 1024:
        # Small/medium — the MPS sweet-spot band.  The best thin tile shifts with
        # size (512³→(32,64,2) at the floor, 640³→(16,64,2) at 0.97× vs the
        # heuristic (32,64,2)'s 0.93×); probe the thin family.
        extra = [(16, 64, 2), (32, 64, 2), (64, 64, 2)]
    else:
        # Large (mx > 1024) — one rule for square, deep-K, and high-aspect alike.
        # The thin-BM tile beats the (64,64,2) default by a margin that GROWS with
        # size (2048³ ~+2%, 6144³/8192³ ~+4%) and the deep-K / high-aspect winners
        # live in the same family ((48,128,4) for square-deep & 8:1-aspect,
        # (64,128,4) for tall-deep).  The crossover is size/aspect/K-specific, so
        # probe instead of gate: a static "(64,64,2) is the broad large winner"
        # silently lost 8192³ (0.95×) and 8192×1024×8192 (0.93×), and every hand
        # gate the session notes added later regressed a neighbour.  The 3% margin
        # keeps (64,64,2) wherever the thin-tile edge is within noise (2048³–4096³,
        # tall low-K like 2048×768×4096), so this never demotes a shape it nailed.
        extra = [(64, 64, 2), (48, 128, 4), (64, 128, 4)]

    cands = [primary]
    for t in extra:
        if t not in cands:
            cands.append(t)
    return cands


def _probe_params(M, N, K):
    """(warmup, iters, reps) for the probe loop, scaled by FLOPs so the one-time
    probe stays bounded while each measurement is trustworthy.  Small/medium
    kernels run in µs, where the per-call sync+Python tail dominates unless we
    run many iters (too few made an early version rank 512³ tiles off noisy ~32µs
    reads of a ~14µs kernel); ms-and-up kernels swamp that tail, so a couple of
    timed iters already give a stable best-of-reps."""
    flops = M * N * K
    if flops <= 2_000_000_000:        # small/medium µs kernels: amortize the tail
        return 20, 80, 8              #   and min-over-many-reps to reject spikes
    if flops <= 50_000_000_000:       # low-ms kernels (≲9ms)
        return 3, 3, 3
    return 1, 1, 3                    # tens-of-ms kernels: one timed iter is stable


def _autotune_m5t(dtype, M, N, K, cands, a, b):
    """Time each candidate tile on the real operands (best-of-reps to suppress
    thermal/scheduler noise) and return the fastest plan.  Candidate 0 is the
    heuristic primary; we keep it unless another beats it by >_AUTOTUNE_MARGIN,
    so a noisy near-tie can never demote the trusted default."""
    warmup, iters, reps = _probe_params(M, N, K)
    o = a.new_empty(M, N)             # private scratch — never the pooled output
    plans = [_build_m5t_plan(dtype, M, N, K, BM, BN, NSG) for (BM, BN, NSG) in cands]

    # Warm EVERY candidate (JIT-compile + ramp the GPU clocks) before timing any
    # of them.  Timing each right after its own cold warmup penalised whichever
    # was measured first — enough to flip the near-tied small-square tiles and
    # bias against the (trusted) primary, which is always candidate 0.
    for (fn, thr, grp) in plans:
        for _ in range(warmup):
            fn(a, b, o, M, N, K, K, N, N, threads=thr, group_size=grp)
    torch.mps.synchronize()

    # Interleave the timed reps so every candidate is measured under the same
    # warm steady state; keep the best (min) per candidate.
    times = [float("inf")] * len(plans)
    for _ in range(reps):
        for j, (fn, thr, grp) in enumerate(plans):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(a, b, o, M, N, K, K, N, N, threads=thr, group_size=grp)
            torch.mps.synchronize()
            times[j] = min(times[j], (time.perf_counter() - t0) / iters)

    i = min(range(len(times)), key=lambda j: times[j])
    if i != 0 and times[i] < times[0] * (1.0 - _AUTOTUNE_MARGIN):
        return plans[i], cands[i]
    return plans[0], cands[0]


def _gemm_plan(dtype, M, N, K, a=None, b=None):
    key = (dtype, M, N, K)
    plan = _GEMM_PLAN.get(key)
    if plan is None:
        cands = _m5_tensor_tile_candidates(M, N, K, dtype)
        if _AUTOTUNE and a is not None and len(cands) > 1:
            plan, tile = _autotune_m5t(dtype, M, N, K, cands, a, b)
        else:
            tile = cands[0]
            plan = _build_m5t_plan(dtype, M, N, K, *tile)
        _GEMM_PLAN[key] = plan
        _GEMM_TILE[key] = tile
    return plan


# Recycled output buffers, keyed by (dtype, M, N).  For ops short enough that the
# output allocation isn't hidden behind the GPU (GEMV at the bandwidth wall, and
# small/medium GEMM where torch.matmul is itself CPU-bound at ~13 µs), torch fuses
# the alloc into its C++ op so it overlaps the kernel, while our per-call new_empty
# leaves a ~0.7-2 µs non-overlapped tail — enough to turn a kernel that beats torch
# into a losing call (512³ 0.97→0.84, 1536×4096 0.99).  We reclaim it by recycling
# a prior output, but ONLY one the caller has provably released: sys.getrefcount ==
# 2 means the buffer is referenced solely by this pool list (+ the getrefcount
# argument), so nothing — not even a view, which keeps a base ref — still aliases
# it.  Anything live reads >2 and forces a fresh allocation, so every returned
# buffer is private, identical semantics to torch.  Large outputs (≥ _CAP_ELEMS)
# skip the pool: there the kernel dwarfs the alloc, and pooling would pin memory.
_OUT_POOL: dict = {}
_OUT_POOL_LIST_CAP = 16
_OUT_POOL_MAX_ELEMS = 1 << 21          # 2 Mi elements (~4 MB bf16) — alloc matters only below this


def _pooled_out(ref, M, N):
    if M * N > _OUT_POOL_MAX_ELEMS:
        return ref.new_empty(M, N)      # long kernel: alloc is already hidden
    key = (ref.dtype, M, N)
    pool = _OUT_POOL.get(key)
    if pool is None:
        o = ref.new_empty(M, N)
        _OUT_POOL[key] = [o]
        return o
    for i in range(len(pool)):
        if sys.getrefcount(pool[i]) == 2:   # held only by this list → free to reuse
            return pool[i]
    o = ref.new_empty(M, N)
    if len(pool) < _OUT_POOL_LIST_CAP:
        pool.append(o)
    return o


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

    Tuned from isolated tile sweeps on M5 Pro (best-of-N timing to suppress
    thermal/scheduler noise).  The governing principle: small problems need
    MANY small tiles to fill the 15 cores and hide latency (small BM, NSG=4);
    large problems already saturate the GPU and prefer the light (64, 64, 2)
    tile for its better occupancy / K-reuse balance.  Deep-K over *large* M, N
    is the one regime where wide-BN (64, 128, 4) wins.

      regime                              tile          speedup vs torch (bf16)
      ----------------------------------  ------------  ------------------------
      M==1 / N==1 (padded GEMV)           (64, 128, 4)  1×32000×4096 → 1.0×
      tiny K (≤256), large M,N            (32, 128, 4)  4096²×64 → 1.03×
      max(M,N) ≤ 256                      (32, 32, 4)   128³→2.2×, 256²×4096→0.98×
      256 < max(M,N) ≤ 768                (32, 64, 2)   512³→0.88×, 333×444×555→0.92×
      deep K (K≥2·max) & max ≥ 1792       (64, 128, 4)  2048²×8192→0.94×, 4096²×11008→0.96×
      max ≥ 4096 & not 64-aligned (lp)    (128, 128, 8) 4097³→1.01×
      large default (bf16/fp16)           (64, 64, 2)   1024³→1.0×, 2047³→1.22×, 4096³→1.0×
      large default (fp32, divisible)     (64, 128, 4)  4096³→2.05×, 2048³→2.38×

    The same tiles serve fp32 (which runs through the M5 tensor unit with
    TF32-relaxed accumulation, identical precision to the manual m5 path) — the
    only dtype splits are the large default (fp32 prefers wide BN) and the
    (128,128,8) edge tile (low-precision only).

    The kernel handles partial edge tiles natively (cT.store(mC) clips to the
    slice's extents; Metal zero-pads OOB buffer reads), so none of these tiles
    require M/N to be a multiple of (BM, BN).
    """
    mx = max(M, N)
    is_lp     = (dtype != torch.float32)   # bf16 / fp16 vs (TF32-relaxed) fp32
    m_div_32  = (M % 32  == 0)
    m_div_64  = (M % 64  == 0)
    n_div_64  = (N % 64  == 0)
    n_div_128 = (N % 128 == 0)

    # Padded GEMV (M==1 with big N routes here): the BM over-read of the single
    # row is free (Metal zeros OOB and the rows cost no bandwidth), and a wide
    # BN streams B efficiently.  Use a THIN BM=16 — a 64-row tile wastes 63 of
    # its rows on MPP work that's thrown away, which cratered N=4096 to 0.70×;
    # BM=16 keeps it bandwidth-bound (1.03× at 4096, 1.09× at 16384, 1.02× at
    # 32000).  The (128,128,8) tile would waste 127 rows/tile and tanks to 0.61×.
    if M == 1 or N == 1:
        return (16, 128, 4)

    # Tiny K over large M, N (small-K attention): the K-loop is so short that
    # per-TG fixed cost dominates, so we want more, smaller tiles.
    if K <= 256 and M >= 1024 and N >= 1024 and m_div_32 and n_div_128:
        return (32, 128, 4)

    # Tiny problems (≤256): (32, 32, 4) keeps enough tiles in flight to fill
    # the cores.  Divisibility-agnostic.  This bucket is checked *before* the
    # deep-K rule on purpose — small-MN deep-K (e.g. 256²×4096) wants the tiny
    # tile too (0.98×), whereas the old deep-K rule gave it (64,128,4) → 0.43×.
    if mx <= 256:
        return (32, 32, 4)

    # Small / medium (256 < max ≤ 1024): a thin BM=32 tile fills the 16 cores
    # with many light tiles.  "Small" must account for tile COUNT and K, not
    # just max(M,N) — the (64,64,2) default makes ceil(M/64)·ceil(N/64) tiles, so
    # use the thin tile when that is too few to fill the cores (n64 < 120, e.g.
    # 512²×8192 → only 64 (64,64,2)-tiles → 0.80×, vs 0.98× thin) OR when K is
    # cube-ish/shallow (K ≤ max — a short K-loop needs many tiles to amortize its
    # per-TG cost, so 896³→0.99, 960³→1.05, 1024³→1.01 want the thin tile that
    # the old `mx ≤ 768` rule denied them at 0.94×).  Otherwise — medium M,N with
    # DEEP K, where (64,64,2) both fills the cores and amortizes the long K-loop
    # efficiently — fall through to it (fixes 768²×4096 0.85→1.04, 768²×8192
    # 0.82→1.08, 704²×4096 1.04→1.18, which the thin tile starves).
    if mx <= 1024:
        n64 = ((M + 63) // 64) * ((N + 63) // 64)
        # Thin when (64,64,2) would either underfill the cores (n64 < 120, e.g.
        # 512²×8192 → 64 tiles) OR the shape is a 64-aligned cube-ish tile
        # (K ≤ max AND M,N % 64 == 0 — 896³→1.00, 960³→1.06).  The 64-alignment
        # guard matters: a *non*-divisible near-cube (1023³) instead loves
        # (64,64,2) (1.16×) — its partial edge tiles ride the static-slice path
        # — so it must fall through, not take the thin tile (which gives 0.94×).
        if n64 < 120 or (K <= mx and m_div_64 and n_div_64):
            return (32, 64, 2)
        # else: medium M,N + deep K, or large non-divisible → (64,64,2) below.

    # Deep K over large M, N: wide BN amortizes the per-tile K-loop.  Only pays
    # off once M, N are large enough to keep the GPU full (≥1792); below that
    # the default (64,64,2) wins (1024²×4096 → 0.98× vs (64,128,4)'s 0.93×).
    if K >= 2 * mx and mx >= 1792 and m_div_64 and n_div_128:
        # bf16/fp16: a THIN BM=48 tile (wide BN=128) is the deep-K sweet spot.
        # The long K-loop wants a wide BN to amortize it, but a 64+ row BM makes
        # too few, too-heavy tiles for these large outputs — BM=48 strikes the
        # balance, giving ~85·N/128 finely-grained tiles that load-balance the
        # 16 cores.  Unifies the whole regime: 4096²×8192 0.96→1.00, 4096²×11008
        # 0.96→1.01, 2048²×8192 0.94→0.99 (also beats the (128,128,8) that the
        # very-deep shapes used to want).  48 = 3·FM(16) so the fragment tiling
        # is exact; M%48 partial row-tiles (≤1 per column) take the edge path.
        if is_lp:
            # ...EXCEPT a moderate, TALL (M>N) deep-K shape (e.g. 2048×768×4096):
            # the wide BN=128 has too few N-tiles while BM=48 over-splits the long
            # M, so the symmetric (64,64,2) balances far better (0.91→1.03×, true
            # at every K depth tested).  Only applies once the wide tile is
            # genuinely backwards — N<M AND the problem is moderate (max≤2048);
            # for larger (max≥4096, e.g. 4096×768×8192) or N≥M (768×2048×4096)
            # the (48,128,4) tile still wins, so those keep it.
            if N < M and mx <= 2048:
                return (64, 64, 2)
            return (48, 128, 4)
        return (64, 128, 4)

    # Very large non-divisible (≥4096, not 64-aligned): bigger tiles + NSG=8
    # amortize edge waste; (64,64,2) starves on edges at this size (4097³).
    # Require min(M,N) ≥ 1024 so thin/GEMV-ish shapes never take this big tile.
    # bf16 ONLY — at 4097³ this is 1.05× for bf16 but only 0.69× for fp16 and
    # 1.54× for fp32 (both far better on the defaults below: 0.82× / 2.0×).
    if dtype == torch.bfloat16 and mx >= 4096 and min(M, N) >= 1024 \
            and not (m_div_64 and n_div_64):
        return (128, 128, 8)

    # Large default.  bf16/fp16: (64,64,2) is the broad winner — light TG
    # threading maximizes resident threadgroups, and it also wins on
    # non-divisible large shapes (1025³→1.04×, 1535³→1.13×, 2047³→1.22×) where
    # the old (128,128,8)/(64,128,4) fallbacks fell to 0.65–0.94×.
    if is_lp:
        return (64, 64, 2)

    # fp32 large: the 4-byte loads make arithmetic intensity matter more, so the
    # wide-BN (64,128,4) tile wins on divisible shapes (4096³→2.05×, 2048³→2.38×,
    # vs (64,64,2)'s 1.92×/2.16×).  Non-divisible fp32 is safer on (64,64,2)
    # (2047³→2.03× vs (64,128,4)'s 1.79×).
    if m_div_64 and n_div_128:
        return (64, 128, 4)
    return (64, 64, 2)


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
    # --- Lean GEMV fast path (checked BEFORE the asserts) -------------------
    # A contiguous row-vector × matrix runs in ~7-12 µs, so for these tiny ops
    # the Python preamble *is* the bottleneck: the asserts + .dim()/.is_contiguous()
    # chains alone cost ~0.9 µs, enough to turn a 1.1× kernel into a 0.93× call.
    # So we gate the whole path on cheap attribute reads (.is_mps, .shape, .ndim)
    # and only fall through to the full preamble (which re-validates) otherwise.
    # bf16/fp16 with N≥4096 prefers padded m5_tensor (handled below), so the
    # fast path is taken only for fp32 or N<4096.  `a.shape[0] == 1` short-circuits
    # GEMM (M>1) immediately.
    sa = a.shape
    if (sa[0] == 1 and backend is None and out is None and swizzle_log is None
            and a.is_mps and len(sa) == 2 and b.ndim == 2):
        dtype = a.dtype
        sb = b.shape
        N = sb[1]
        K = sa[1]
        if (dtype in _PROFILE and N >= 16 and sb[0] == K
                and (dtype is torch.float32 or N < 4096)
                and a.is_contiguous() and b.is_contiguous()):
            # Everything the launch needs (kernel handle + thread/group tuples)
            # is a pure function of (dtype, N), so memoize it: the per-call work
            # is then just one dict hit, the output alloc, and the enqueue.
            # b is (K×N) contiguous ⇒ stride(0)==N, so we pass N as ldb (skips the
            # stride() call).  a (1×K) / o (1×N) are contiguous, so their linear
            # storage is exactly the K-/N-vector the kernel indexes.  b.new_empty
            # (b's dtype/device, positional dims) is ~0.4 µs cheaper than
            # torch.empty((1,N), dtype=…, device="mps") — material at this size.
            gt, thr, grp = _gemv_plan(dtype, N, K >= 2048)
            o = _pooled_out(b, 1, N)
            gt(b, a, o, N, K, N, threads=thr, group_size=grp)
            return o

    # --- Lean GEMM fast path (contiguous, untransposed, m5_tensor regime) -----
    # The common GEMM call (both inputs row-major contiguous, M,N,K ≥ 64) routes
    # to m5_tensor.  Small/medium GEMM is short enough that the general preamble
    # (_resolve_inputs stride analysis + tile pick + shader-cache lookup, ~2 µs)
    # is NOT hidden behind the kernel and torch.matmul is itself CPU-bound at
    # ~13 µs — so memoizing the whole launch by (dtype, M, N, K) is what gets
    # 511³/333×444×555 over parity (their kernels already beat torch's CPU floor).
    # `a.shape[0] > 1` here (M==1 took the GEMV path above); anything not matching
    # (transposed / non-contiguous / sub-64 / overrides) falls through unchanged.
    if (backend is None and out is None and swizzle_log is None
            and a.is_mps and a.ndim == 2 and b.ndim == 2):
        M = sa[0]; K = sa[1]
        sb = b.shape
        N = sb[1]
        dtype = a.dtype
        if (M >= 64 and N >= 64 and K >= 64 and sb[0] == K and dtype in _PROFILE
                and a.is_contiguous() and b.is_contiguous()):
            # Pass a, b so a first-sight ambiguous shape autotunes on the real
            # operands; cached thereafter, so the steady-state hot path is still
            # just a dict hit + the pooled-output alloc + the enqueue.
            fn, thr, grp = _gemm_plan(dtype, M, N, K, a, b)
            o = _pooled_out(a, M, N)
            fn(a, b, o, M, N, K, K, N, N, threads=thr, group_size=grp)
            return o

    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype
    dtype = a.dtype
    if dtype not in _PROFILE:
        raise NotImplementedError(f"dtype {dtype} not supported")

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)

    if out is None:
        # Recycle a released buffer (see _pooled_out): small/medium GEMM is
        # short enough that the alloc tail isn't hidden behind the kernel and
        # torch is CPU-bound, so this is what gets 511³/333×444×555 over parity.
        out = _pooled_out(a, M, N)
    else:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps"
    ldc = out.stride(0)

    in_t, acc_t, out_t = _PROFILE[dtype]
    M_, N_, K_ = int(M), int(N), int(K)
    lda_, ldb_, ldc_ = int(lda), int(ldb), int(ldc)

    backend = backend or "auto"

    if backend == "auto":
        is_lp = (dtype != torch.float32)
        # m5_tensor builds tensor_inline views that assume PACKED storage (row
        # stride == inner extent); it ignores lda/ldb.  So it's only correct when
        # the untransposed operands are densely packed (lda==K, ldb==N).  A
        # strided sub-view (e.g. X[:, :K] with ld>K) is row-major but NOT packed —
        # routing it to m5_tensor reads the wrong elements (max_err ~120).  Such
        # inputs fall back to the manual m5 kernel, which honors lda/ldb.
        packed_ab = (lda_ == K_ and ldb_ == N_)
        # Wide-N GEMV (M==1, N≥4096 ⇒ ≥128 column-blocks already fill the GPU).
        # Here we're bandwidth bound and a "padded" m5_tensor wins for bf16/fp16:
        # it reads the 15 OOB rows as zero (Metal bounds-check) and writes only
        # the single valid row via cT.store(mC).  With the BM=16 tile this is
        # 1.03–1.09× across N (vs the dedicated gemv kernel's 0.93×).  fp32 stays
        # on the dedicated gemv kernel — m5_tensor loses for fp32 above N=4096
        # (4-byte loads already saturate; m5t 8192 → 0.89×, gemv → 1.01×).
        if (M == 1 and N >= 4096 and K >= 256 and is_lp
                and not trans_a and not trans_b and packed_ab):
            backend = "m5_tensor"
        # Everything else GEMV-shaped goes to the dedicated kernel, which fills
        # the GPU on small-N by splitting K across 32 simdgroups (see
        # _gemv_handles / _gemv_t).  fp32 of every N comes here too.
        elif M == 1 and N >= 16:
            backend = "gemv"
        elif N == 1 and M >= 16:
            backend = "gemv"
        else:
            # m5_tensor (MPP op.run path) wins across the whole size range AND
            # every supported dtype thanks to internal load orchestration.  For
            # fp32 it runs the M5 tensor unit with TF32-relaxed accumulation
            # (same precision as the manual m5 path) and beats it everywhere —
            # small fp32 especially (256³: 0.50× → 1.55×).  Its cT.store(mC)
            # plus Metal's bounds-checked reads handle non-divisible AND tiny
            # shapes natively, so we route everything down to 64³ here (64³ runs
            # ~2× vs torch).  Only sub-64 dims and transposed inputs (whose MPP
            # descriptor TRANS_* path isn't wired up) fall back to manual m5.
            if (M >= 64 and N >= 64 and K >= 64
                    and not trans_a and not trans_b and packed_ab):
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


def _gemv_nt(nt, matrix, vec, out_v, rows, k, ld):
    # y[r] = sum_k matrix[r, k] * vec[k] (matrix row-major M×K, one warp/row).
    n_groups = (rows + 3) // 4          # NWARPS=4, ROWS_PER_SG=1
    nt(matrix, vec, out_v, rows, k, ld,
       threads=(128 * n_groups, 1, 1), group_size=(128, 1, 1))


def _gemv_t(gh, matrix, x_vec, out_v, cols, k, ld, dtype):
    # y[c] = sum_k matrix[k, c] * x[k].  The chosen gt splits K across its
    # simdgroups within each threadgroup and reduces in threadgroup memory —
    # one dispatch, full per-core occupancy regardless of cols.  This helper
    # serves transposed / non-contiguous view inputs, so it forces VEC=1
    # (vec>1 needs a 16-B-aligned base, only guaranteed on the contiguous path).
    gt, tg, vec = _gemv_pick(gh, cols, ld, dtype, vec_ok=False, k_big=(k >= 2048))
    n_groups = (cols + 32 * vec - 1) // (32 * vec)
    gt(matrix, x_vec, out_v, cols, k, ld,
       threads=(tg * n_groups, 1, 1), group_size=(tg, 1, 1))


def _dispatch_gemv(A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, dtype, out):
    gh = _gemv_handles(dtype)
    M, N, K = int(M), int(N), int(K)
    nt = gh["nt"]
    yv = out.view(-1)
    if M == 1:
        if trans_b:
            _gemv_nt(nt, B_view, A_view.view(-1), yv, N, K, int(ldb))
        else:
            _gemv_t(gh, B_view, A_view.view(-1), yv, N, K, int(ldb), dtype)
        return out
    elif N == 1:
        if trans_a:
            _gemv_t(gh, A_view, B_view.view(-1), yv, M, K, int(lda), dtype)
        else:
            _gemv_nt(nt, A_view, B_view.view(-1), yv, M, K, int(lda))
        return out
    else:
        raise ValueError("gemv backend requires M==1 or N==1")


# Convenience alias
gemm = matmul
