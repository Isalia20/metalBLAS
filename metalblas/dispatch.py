"""High-level matmul dispatcher.

`matmul(a, b)` picks the best kernel variant (tile, GEMV vs GEMM, simd vs M5
tensor unit, swizzle) from input shapes and dtype.
"""
from __future__ import annotations

import math
import os
import sys
import time
import torch

from . import kernels


def _pk(*vals):
    words = []
    for i in range(0, len(vals), 2):
        lo = int(vals[i]) & 0xFFFFFFFF
        hi = (int(vals[i + 1]) & 0xFFFFFFFF) if i + 1 < len(vals) else 0
        words.append(lo | (hi << 32))
    return words


_DTYPE_NAME = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

# dtype -> (in_t, acc_t, out_t) for the standard "fast" matmul.
# All inputs accumulate in fp32 (tensor-core style).
_PROFILE = {
    torch.float32: ("float", "float", "float"),
    torch.float16: ("half",  "float", "half"),
    torch.bfloat16: ("bfloat", "float", "bfloat"),
}
_COMPLEX = {
    torch.complex64: ("float2", "float"),
    torch.complex32: ("half2",  "half"),
}
_COMPLEX_REAL = {
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

# Apple GPU family detection
# M5 adds a tensor unit; the M5-tuned GEMV heuristic underfills pre-M5 chips, so
# `_gemv_pick` branches on this. Override with METALBLAS_HAS_TENSOR_UNIT=1/0.
def _detect_has_tensor_unit() -> bool:
    env = os.environ.get("METALBLAS_HAS_TENSOR_UNIT")
    if env is not None:
        return env != "0"
    if sys.platform != "darwin":
        return False
    try:
        import re
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True, timeout=2,
        ).stdout.strip()
    except Exception:
        return False
    m = re.match(r"Apple M(\d+)", out)
    return bool(m and int(m.group(1)) >= 5)


_HAS_TENSOR_UNIT = _detect_has_tensor_unit()


# GEMV hot-path caches
# Small GEMV is CPU-bound, so resolve handles once per dtype. gemv_t splits K
# across NWARPS simdgroups per threadgroup to fill cores on few column-blocks.
_GEMV_HANDLES: dict = {}

# gemv_t variants keyed by (VEC, NWARPS). Higher VEC (columns/lane) covers more
# of a 128-B line; fp32 keeps VEC=1, bf16/fp16 add vectorized tiles.
_GEMV_T_VARIANTS = {
    torch.float32:  [(1, 16), (1, 32)],
    torch.float16:  [(1, 32), (2, 32), (2, 16), (4, 16), (4, 8), (8, 8), (8, 16)],
    torch.bfloat16: [(1, 32), (2, 32), (2, 16), (4, 16), (4, 8), (8, 8), (8, 16)],
}


def _gemv_handles(dtype):
    h = _GEMV_HANDLES.get(dtype)
    if h is None:
        in_t, acc_t, out_t = _PROFILE[dtype]
        gt = {}
        for (vec, nw) in _GEMV_T_VARIANTS[dtype]:
            fn, _ = kernels.gemv_t(in_t, acc_t, out_t, 32 * vec, nw, vec)
            gt[(vec, nw)] = fn
        # gemv_nt (N==1, matrix @ vector): VEC>1 reads each row a full cache line at
        # a time. Built pre-M5 (all dtypes, original) and on M5 for low precision
        # (fp32 VEC=1 is already a full 128-B line). The picker gates VEC on the
        # row-stride/offset alignment and K.
        nt = {}
        nt[1], _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 1)
        if not _HAS_TENSOR_UNIT or dtype is not torch.float32:
            nt[2], _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 2)
            nt[4], _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 4)
        h = {"gt": gt, "nt": nt}
        _GEMV_HANDLES[dtype] = h
    return h


def _gemv_nt_pick(nt_dict, rows, k, ld, off=0):
    if 4 not in nt_dict:                 # fp32 (or VEC>1 not built): VEC=1 only
        return nt_dict[1]
    align = int(ld) | int(off)
    if _HAS_TENSOR_UNIT:
        if (align & 3) == 0 and k >= 512:
            return nt_dict[4]
        if (align & 1) == 0 and k >= 512:
            return nt_dict[2]
        return nt_dict[1]
    if (align & 3) == 0 and k >= 64:
        return nt_dict[4]
    if (align & 1) == 0 and k >= 32:
        return nt_dict[2]
    return nt_dict[1]


def _gemv_pick(gh, cols, ldb, dtype, vec_ok=True, k=None, k_big=True):
    """Return (gemv_t_fn, threadgroup_size, vec) for a `cols`-wide GEMV.

    Scales VEC with N for cache-line coverage while keeping enough TGs to fill
    the cores (see inline branches). `vec_ok=False` forces VEC=1 for views.
    """
    gt = gh["gt"]
    if k is not None:
        k_big = (k >= 2048)
    if dtype is torch.float32:
        ng = (cols + 31) // 32
        if not _HAS_TENSOR_UNIT:
            # Pre-M5: NW=32 doubles the simdgroup count to fill the many cores
            # (lifts cols<=2048 15-30%); cols>=4096 already has plenty of TGs.
            return (gt[(1, 16)], 512, 1) if ng >= 128 else (gt[(1, 32)], 1024, 1)
        if 16 < ng <= 32:
            return gt[(1, 16)], 512, 1
        return gt[(1, 32)], 1024, 1
    if not _HAS_TENSOR_UNIT:
        # Pre-M5: the M5 rule's high-VEC/low-NWARPS gives too few TGs to fill the
        # many cores, so target total simdgroup count (ng*nw) in [~1000, ~4000].
        if vec_ok and cols > 12288:
            # Huge N (e.g. lm_head 32000): VEC=8 amortizes launch; VEC=2's 16k+
            # TGs would lose to scheduling overhead.
            vec, nw = 8, 8
        elif vec_ok and cols >= 2560:
            # Mid-to-wide N: VEC=2 keeps ng*nw above ~1000 while still spanning
            # a full 128-B cache line per warp.
            vec, nw = 2, 32
        else:
            # Up to N=2048: VEC=1 gives the most simdgroups (ng*nw=cols) without
            # idling cores, and wins across the small/medium range.
            vec, nw = 1, 32
    elif vec_ok and cols >= 4096 and (k_big or (k is not None and k >= 1024)):
        # Wide N: full-line VEC=8 coalescing pays off once K>=1024, even below the
        # deep-K (k_big) gate - the half-line VEC=2 loses ~8% here (4096x1024 bf16).
        vec = 8
        nw = 16 if (k is not None and k >= 8192) else 8
    elif vec_ok and cols >= 2560 and (k_big or (k is not None and k >= 1024)):
        vec, nw = 4, 8
    elif vec_ok and k_big and cols >= 1280:
        # The 1280..2560 VEC=4 tier stays deep-K-only: at K=1024 these N prefer VEC=2.
        vec, nw = 4, 16
    elif vec_ok and cols >= 1024:
        vec, nw = 2, 16     # >=16 TGs fill the cores; NW=32 over-subscribes the
                            # K-reduction (1024: 453 vs 444 GB/s)
    elif vec_ok and cols > 512:
        vec, nw = 2, 32     # <16 TGs: more warps to fill cores (768: 416 vs 373)
    else:
        vec, nw = 1, 32
    # Clamp VEC to the column stride's alignment.
    if vec == 8 and (ldb & 7):
        vec, nw = (4, 8) if not (ldb & 3) else ((2, 32) if not (ldb & 1) else (1, 32))
    elif vec == 4 and (ldb & 3):
        vec, nw = (2, 32) if not (ldb & 1) else (1, 32)
    elif vec == 2 and (ldb & 1):
        vec, nw = 1, 32
    return gt[(vec, nw)], nw * 32, vec


# Fast-path launch plan, memoized by (dtype, N, K): for contiguous GEMV (ldb == N)
# the handle/thread/group are functions of (dtype, N) and the K tiers, plus the
# precomputed packed (N, K, N, 1) dims buffer (which embeds the exact K).
_GEMV_PLAN: dict = {}
# Symmetric memo for the N==1 (gemv_nt) path: pure function of (dtype, K).
_GEMV_NT_PLAN: dict = {}


def _gemv_nt_plan(dtype, K):
    """Resolve the gemv_nt variant + threadgroup size for the contiguous N==1
    path (lda == K). Returns (fn, group_size, k_hi, k_lda); group size is fixed
    at 128. gemv_nt packs (M, K, lda=K) into a uint4 (x=M, y=K, z=lda); M is the
    runtime arg, so we precompute the K halves (k_hi = K<<32, k_lda = K) and the
    hot path only ORs in M: dims = [M | k_hi, k_lda]."""
    key = (dtype, K)
    plan = _GEMV_NT_PLAN.get(key)
    if plan is None:
        gh = _gemv_handles(dtype)
        fn = _gemv_nt_pick(gh["nt"], 0, K, K)  # rows arg unused by picker
        plan = (fn, (128, 1, 1), K << 32, K)
        _GEMV_NT_PLAN[key] = plan
    return plan


def _gemv_plan(dtype, N, K):
    # Key on the exact K so the packed dims (which embed K) can be memoized; the
    # (fn, thr, grp) pick itself only varies with N and the K>=2048/>=8192 tiers.
    key = (dtype, N, K)
    plan = _GEMV_PLAN.get(key)
    if plan is None:
        gt, tg, vec = _gemv_pick(_gemv_handles(dtype), N, N, dtype, k=K)
        ng = (N + 32 * vec - 1) // (32 * vec)
        # Contiguous M==1: ldb==N, xs==1, so the full (N, K, ldb, xs) packs once.
        plan = (gt, (tg * ng, 1, 1), (tg, 1, 1), _pk(N, K, N, 1))
        _GEMV_PLAN[key] = plan
    return plan


# Memoized mpp_tensor plan for the contiguous-GEMM fast path, keyed by
# (dtype, M, N, K); untransposed swizzle-0 so the hot path is a dict hit + enqueue.
_GEMM_PLAN: dict = {}
# The (BM,BN,NSG) actually chosen per shape (autotuned winner or heuristic
# primary), so diagnostics report the real tile.
_GEMM_TILE: dict = {}

# Runtime tile autotuner
# Ambiguous regimes (best tile flips on K/aspect) emit a candidate list probed
# once and cached; candidate 0 is the heuristic. Disable with METALBLAS_AUTOTUNE=0.
_AUTOTUNE = os.environ.get("METALBLAS_AUTOTUNE", "1") != "0"
# Margin a non-primary must beat the heuristic by, above the ~14us noise floor
# so a near-tie can't demote the default (real wins are all >=6%).
_AUTOTUNE_MARGIN = 0.03
# Tighter margin for tall-narrow regimes, where the win is ~1-2% and flips with
# size; probed with many iters so 1% beats noise (bad tiles lose by 15-25%).
_TALL_NARROW_MARGIN = 0.01


def _build_mppt_plan(dtype, M, N, K, BM, BN, NSG):
    """Compile one mpp_tensor tile and return its (fn, threads, group) launch plan
    for the packed, untransposed, swizzle-0 case (lda=K, ldb=N, ldc=N)."""
    in_t, _, out_t = _PROFILE[dtype]
    mn_aligned = (M % BM == 0) and (N % BN == 0)
    fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, False, False,
                                   relaxed=True, swizzle_log=0, mn_aligned=mn_aligned)
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    return (fn, (NSG * 32 * tiles_n, tiles_m, 1), (NSG * 32, 1, 1))


# Split-K plan (deep-K, few-tile shapes)
# Reused fp32 partial buffers keyed by (M, N, planes); MPS runs in program order
# so one buffer per key is safe to recycle across calls.
_SPLITK_POOL: dict = {}


def _splitk_partial(ref, planes, M, N):
    key = (M, N, planes)
    buf = _SPLITK_POOL.get(key)
    if buf is None:
        buf = ref.new_empty(planes, M, N, dtype=torch.float32)
        _SPLITK_POOL[key] = buf
    return buf


class _SplitKPlan:
    """Two-pass split-K launch. The splitk kernel runs G K-chunks per tile
    (chunk 0 writes C, the rest write fp32 planes); reduce sums planes into C."""

    def __init__(self, splitk_fn, reduce_fn, M, N, K, BM, BN, NSG, G):
        self.splitk_fn = splitk_fn
        self.reduce_fn = reduce_fn
        self.M = M
        self.N = N
        self.K = K
        self.G = G
        self.planes = G - 1
        tiles_m = (M + BM - 1) // BM
        tiles_n = (N + BN - 1) // BN
        self.sk_threads = (NSG * 32 * tiles_n, tiles_m, G)
        self.sk_group = (NSG * 32, 1, 1)
        self.n_elems = M * N
        self.red_threads = (self.n_elems, 1, 1)
        self.red_group = (256, 1, 1)
        self.sk_dims = _pk(M, N, K)
        self.red_dims = _pk(self.n_elems, self.planes)

    def run(self, a, b, o):
        cp = _splitk_partial(o, self.planes, self.M, self.N)
        self.splitk_fn(a, b, o, cp, self.sk_dims,
                       threads=self.sk_threads, group_size=self.sk_group)
        self.reduce_fn(cp, o, self.red_dims,
                       threads=self.red_threads, group_size=self.red_group)


def _build_splitk_plan(dtype, M, N, K, BM, BN, NSG, G):
    in_t, _, out_t = _PROFILE[dtype]
    splitk_fn, reduce_fn = kernels.splitk_gemm(in_t, out_t, BM, BN, NSG, K // G, relaxed=True)
    return _SplitKPlan(splitk_fn, reduce_fn, M, N, K, BM, BN, NSG, G)


def _is_splitk_regime(M, N, K, dtype):
    """Few-output-tile shapes where op.run underfills but long K splits across G
    threadgroups per tile. Gated on deep K, small output, low-precision."""
    return (dtype is not torch.float32 and K >= 2048
            and 64 <= min(M, N) and M * N <= 1_500_000
            and (min(M, N) <= 256 or K >= 8 * max(M, N)))


def _splitk_specs(M, N, K):
    """(BM, BN, NSG, G) deep-K candidates; G must divide K with a 16-aligned chunk.
    The (tile, G) winner shifts with size, so the autotuner picks from this family."""
    specs = []
    for (BM, BN, NSG) in [(128, 32, 2), (64, 64, 2), (32, 64, 2)]:
        for G in (2, 4):
            if K % G == 0 and (K // G) % 16 == 0:
                specs.append((BM, BN, NSG, G))
    return specs


# 1x1-conv plan (very-thin-N)
# matmul2d underfills a 1-fragment-wide N tile; a 1x1 conv (N as output channels)
# schedules thin channels better. Autotuner keeps a conv candidate only when it wins.
class _Conv1x1Plan:
    """1x1-conv GEMM launch for one (dtype, M, N, K, BMW, BNO, NSG)."""

    def __init__(self, fn, M, N, K, BMW, BNO, NSG):
        self.fn = fn
        self.M = M
        self.N = N
        self.K = K
        tiles_o = (N + BNO - 1) // BNO
        tiles_w = (M + BMW - 1) // BMW
        self.threads = (NSG * 32 * tiles_o, tiles_w, 1)
        self.group = (NSG * 32, 1, 1)
        self.dims = _pk(M, N, K)

    def run(self, a, b, o):
        self.fn(a, b, o, self.dims,
                threads=self.threads, group_size=self.group)


def _build_conv_plan(dtype, M, N, K, BMW, BNO, NSG):
    in_t, _, out_t = _PROFILE[dtype]
    fn, _ = kernels.conv1x1_gemm(in_t, out_t, BMW, BNO, NSG, K, relaxed=True)
    return _Conv1x1Plan(fn, M, N, K, BMW, BNO, NSG)


def _is_conv_regime(M, N, K, dtype):
    """Very-thin-N (N<=64, multiple of 32) over wide M: matmul2d's 32-wide fragment
    underfills, so probe the conv path. N>=96 keeps matmul2d (it wins there)."""
    return (dtype is not torch.float32 and N <= 64 and N % 32 == 0
            and M >= 512 and K >= 256)


def _conv_specs(M, N, K):
    """(BMW, BNO, NSG) conv candidates: BNO=N covers all output channels, BMW
    (M-width tile) must divide M. Winner shifts with M and K, so the autotuner picks."""
    specs = []
    for BMW in (16, 32, 48, 64, 96, 128):
        if M % BMW == 0:
            for NSG in (2, 4):
                specs.append((BMW, N, NSG))
    return specs


def _mpp_tensor_tile_candidates(M, N, K, dtype):
    """Return (candidate tiles, autotuner margin) for (M,N,K,dtype).

    A one-element list means the heuristic is confident; a longer list marks an
    ambiguous regime. Candidate 0 is `_pick_mpp_tensor_tile`; margin = win threshold.
    """
    primary = _pick_mpp_tensor_tile(M, N, K, dtype)
    # fp32 wins 2-3x everywhere - never worth a probe.
    if dtype == torch.float32:
        return [primary], _AUTOTUNE_MARGIN
    mx = max(M, N)
    mn = min(M, N)

    # Confident regimes: padded GEMV, tiny-K, and tiny problems each have a
    # single decisive tile - no probe.
    if M == 1 or N == 1:
        return [primary], _AUTOTUNE_MARGIN
    if K <= 256 and M >= 1024 and N >= 1024:
        return [primary], _AUTOTUNE_MARGIN
    if mx <= 256:
        return [primary], _AUTOTUNE_MARGIN

    # Thin min-dim: square default makes too few tiles on the short axis. Offer
    # both tall-narrow orientations plus the NSG/BM ambiguity for the autotuner.
    if mn <= 256 and mx >= 1024:
        extra = [(128, 32, 2), (256, 32, 4), (32, 128, 2), (32, 256, 4), (64, 64, 2),
                 (64, 32, 2), (32, 64, 2)]
        # Thin M (M<<N) wants a SHORT BM so the few rows nearly fill the tile;
        # add the BM=16/32 family when the min dim is tiny.
        if mn <= 48:
            extra = [(16, 64, 2), (32, 64, 2), (32, 128, 4)] + extra
        return _with_primary(primary, extra), _TALL_NARROW_MARGIN

    if mx <= 1024:
        # Very deep K over small M,N: tall-narrow family wins but best NSG flips
        # with tile count, so probe. Gated K>=8*max so cube-ish shapes are untouched.
        if K >= 8 * mx:
            extra = [(128, 32, 4), (128, 32, 2), (192, 32, 2), (32, 64, 2), (64, 64, 2)]
            return _with_primary(primary, extra), _TALL_NARROW_MARGIN
        # Small/medium MPS sweet-spot band. The best thin tile shifts with size,
        # so probe the thin family.
        extra = [(16, 64, 2), (32, 64, 2), (64, 64, 2)]
    else:
        # Large (mx > 1024): thin-BM tile beats (64,64,2) by a size-growing margin,
        # sharing the deep-K/high-aspect family; crossover is shape-specific so probe.
        extra = [(64, 64, 2), (48, 128, 4), (64, 128, 4), (128, 64, 4)]
        # Large non-64-divisible (e.g. 4095^3): big (128,128,8) amortizes edge waste
        # that starves thin tiles. The 3% margin keeps divisible squares on (64,64,2).
        if mx >= 2048 and not (M % 64 == 0 and N % 64 == 0):
            extra.append((128, 128, 8))

    return _with_primary(primary, extra), _AUTOTUNE_MARGIN


def _with_primary(primary, extra):
    cands = [primary]
    for t in extra:
        if t not in cands:
            cands.append(t)
    return cands


def _probe_params(M, N, K):
    """(warmup, iters, reps) scaled by FLOPs: us kernels need many iters to amortize
    the per-call tail; ms-and-up kernels swamp it, so a few iters suffice."""
    flops = M * N * K
    if flops <= 2_000_000_000:        # small/medium us kernels: amortize the tail
        return 20, 80, 8              #   and min-over-many-reps to reject spikes
    if flops <= 50_000_000_000:       # low-ms kernels (<~9ms)
        return 3, 3, 3
    return 2, 3, 3                    # tens-of-ms: one timed iter is too noisy to
                                      # rank tiles a few % apart; 3 iters / min of
                                      # 3 reps stabilizes the ranking.


def _autotune_mppt(dtype, M, N, K, cands, a, b, margin=_AUTOTUNE_MARGIN, sk_specs=(),
                  conv_specs=()):
    """Time each candidate (best-of-reps), returning the fastest (plan, label).
    Candidate 0 (heuristic) is kept unless beaten by >margin; sk/conv_specs add candidates."""
    warmup, iters, reps = _probe_params(M, N, K)
    # A tight margin must resolve a ~1% gap, so spend more iters/reps to push
    # noise below it. Still one-time, and only the few tall-narrow shapes probe.
    if margin < _AUTOTUNE_MARGIN:
        iters = max(iters * 2, 6)
        reps = max(reps, 5)
    o = a.new_empty(M, N)             # private scratch - never the pooled output

    # Each candidate is (cached_plan, run(a,b,o), label): mpp_tensor tiles keep the
    # lean (fn, thr, grp) tuple, split-K uses its plan object.
    mppt_dims = _pk(M, N, K, K, N, N)     # packed dims, shared by all candidate tiles
    cand = []
    for (BM, BN, NSG) in cands:
        fn, thr, grp = _build_mppt_plan(dtype, M, N, K, BM, BN, NSG)
        run = (lambda a, b, o, fn=fn, thr=thr, grp=grp, d=mppt_dims:
               fn(a, b, o, d, threads=thr, group_size=grp))
        cand.append(((fn, thr, grp), run, (BM, BN, NSG)))

    # Add split-K / 1x1-conv candidates, but only those whose result matches the
    # single-pass primary (guards a precision surprise). Checked once.
    if sk_specs or conv_specs:
        ref = a.new_empty(M, N)
        cand[0][1](a, b, ref)
        torch.mps.synchronize()
        scale = ref.abs().max().item() + 1e-6
        for (BM, BN, NSG, G) in sk_specs:
            plan = _build_splitk_plan(dtype, M, N, K, BM, BN, NSG, G)
            plan.run(a, b, o)
            torch.mps.synchronize()
            if (o - ref).abs().max().item() <= 0.02 * scale:
                cand.append((plan, plan.run, (BM, BN, NSG, G)))
        for (BMW, BNO, NSG) in conv_specs:
            plan = _build_conv_plan(dtype, M, N, K, BMW, BNO, NSG)
            plan.run(a, b, o)
            torch.mps.synchronize()
            if (o - ref).abs().max().item() <= 0.02 * scale:
                cand.append((plan, plan.run, (BMW, BNO, NSG, "conv")))

    # Warm EVERY candidate (JIT-compile + ramp GPU clocks) before timing any;
    # warming each right before its own timing biased against whichever ran first.
    for (_, run, _) in cand:
        for _ in range(warmup):
            run(a, b, o)
    torch.mps.synchronize()

    # Interleave the timed reps so every candidate is measured under the same
    # warm steady state; keep the best (min) per candidate.
    times = [float("inf")] * len(cand)
    for _ in range(reps):
        for j, (_, run, _) in enumerate(cand):
            t0 = time.perf_counter()
            for _ in range(iters):
                run(a, b, o)
            torch.mps.synchronize()
            times[j] = min(times[j], (time.perf_counter() - t0) / iters)

    i = min(range(len(times)), key=lambda j: times[j])
    if i != 0 and times[i] < times[0] * (1.0 - margin):
        return cand[i][0], cand[i][2]
    return cand[0][0], cand[0][2]


def _gemm_plan(dtype, M, N, K, a=None, b=None):
    key = (dtype, M, N, K)
    plan = _GEMM_PLAN.get(key)
    if plan is None:
        cands, margin = _mpp_tensor_tile_candidates(M, N, K, dtype)
        sk_specs = ()
        conv_specs = ()
        if _AUTOTUNE and a is not None and _is_splitk_regime(M, N, K, dtype):
            sk_specs = _splitk_specs(M, N, K)
        if _AUTOTUNE and a is not None and _is_conv_regime(M, N, K, dtype):
            conv_specs = _conv_specs(M, N, K)
        if _AUTOTUNE and a is not None and (len(cands) > 1 or sk_specs or conv_specs):
            plan, tile = _autotune_mppt(dtype, M, N, K, cands, a, b, margin,
                                        sk_specs, conv_specs)
        else:
            tile = cands[0]
            plan = _build_mppt_plan(dtype, M, N, K, *tile)
        _GEMM_PLAN[key] = plan
        _GEMM_TILE[key] = tile
    return plan


# Recycled output buffers: short ops' per-call new_empty tail can flip a win to a
# loss. Reuse only a released buffer (getrefcount == 2); large outputs skip the pool.
_OUT_POOL: dict = {}
_OUT_POOL_LIST_CAP = 16
_OUT_POOL_MAX_ELEMS = 1 << 21          # 2 Mi elements (~4 MB bf16) - alloc matters only below this


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

    Constraint: TM*TN C-fragments per warp must stay <=16 to fit registers,
    else we spill and lose 10x.
    """
    dtype_bytes = 4 if dtype == torch.float32 else 2
    # (BM, BN, BK, WM, WN) - we enumerate sensible tiles and pick the best.
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


def _pick_mpp_tensor_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """Pick (BM, BN, NSG) for the mpp_tensor_gemm kernel.

    Tuned from M5 Pro sweeps. Small problems need many small tiles to fill the
    cores; large problems prefer (64,64,2); deep-K large M,N wants (64,128,4).

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

    fp32 uses the same tiles (TF32-relaxed); only the large default and (128,128,8)
    edge tile split on dtype. Partial edge tiles are native, so no divisibility needed.
    """
    mx = max(M, N)
    is_lp     = (dtype != torch.float32)   # bf16 / fp16 vs (TF32-relaxed) fp32
    m_div_32  = (M % 32  == 0)
    m_div_64  = (M % 64  == 0)
    n_div_64  = (N % 64  == 0)
    n_div_128 = (N % 128 == 0)

    # Padded GEMV (M==1 with big N): the BM over-read is free (Metal zeros OOB)
    # and wide BN streams B. THIN BM=16 - a 64-row tile wastes 63 rows.
    if M == 1 or N == 1:
        return (16, 128, 4)

    # Tiny K over large M,N (small-K attention): the K-loop is so short that
    # per-TG cost dominates, so we want more, smaller tiles.
    if K <= 256 and M >= 1024 and N >= 1024 and m_div_32 and n_div_128:
        return (32, 128, 4)

    # Tiny problems (<=256): (32,32,4) keeps enough tiles to fill the cores. Before
    # the deep-K rule on purpose - small-MN deep-K wants it too (0.98x vs 0.43x).
    if mx <= 256:
        return (32, 32, 4)

    # Thin M, wide N (small batched decode): a 64-row tile throws away most rows,
    # so use a short BM. Gated on N being the wide axis so thin-N is untouched.
    if M <= 48 and N >= 1024:
        return (16, 64, 2) if M <= 16 else (32, 64, 2)

    # Small/medium (256 < max <= 1024): the thin BM=32 tile fills the cores;
    # medium M,N with DEEP K falls through to (64,64,2) which amortizes K.
    if mx <= 1024:
        n64 = ((M + 63) // 64) * ((N + 63) // 64)
        # Thin when (64,64,2) underfills (n64 < 120) or a 64-aligned cube-ish tile
        # (K <= max); the alignment guard drops non-divisible near-cubes (1023^3).
        if n64 < 120 or (K <= mx and m_div_64 and n_div_64):
            return (32, 64, 2)
        # else: medium M,N + deep K, or large non-divisible -> (64,64,2) below.

    # Deep K over large M,N: wide BN amortizes the per-tile K-loop. Only pays off
    # once M,N are large enough (>=1792); below that the default (64,64,2) wins.
    if K >= 2 * mx and mx >= 1792 and m_div_64 and n_div_128:
        # bf16/fp16 deep-K sweet spot: wide BN=128 amortizes the long K-loop while
        # THIN BM=48 (=3*FM) load-balances cores with exact tiling.
        if is_lp:
            # EXCEPT moderate TALL (M>N, max<=2048) shapes: wide BN has too few
            # N-tiles and BM=48 over-splits M, so (64,64,2) balances better.
            if N < M and mx <= 2048:
                return (64, 64, 2)
            return (48, 128, 4)
        return (64, 128, 4)

    # Very large non-divisible (>=4096, not 64-aligned): big tiles + NSG=8 amortize
    # edge waste. min(M,N)>=1024 skips thin shapes; bf16 ONLY (fp16/fp32 use defaults).
    if dtype == torch.bfloat16 and mx >= 4096 and min(M, N) >= 1024 \
            and not (m_div_64 and n_div_64):
        return (128, 128, 8)

    # Large default. bf16/fp16: (64,64,2) is the broad winner - light TG threading
    # maximizes resident groups; also wins large non-divisible (old fallbacks: 0.65x).
    if is_lp:
        return (64, 64, 2)

    # fp32 large: 4-byte loads make arithmetic intensity matter, so wide-BN
    # (64,128,4) wins on divisible shapes; non-divisible fp32 is safer on (64,64,2).
    if m_div_64 and n_div_128:
        return (64, 128, 4)
    return (64, 64, 2)


def _pick_mpp_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int, int, int, bool]:
    """Pick (BM, BN, BK, WM, WN, dbuf) for the mpp_gemm kernel (16x32x16 frags).

    Hand-tuned from M5 Pro sweeps; falls back to a smaller tile if over budget.
    `dbuf=True` double-buffers the K-loop to overlap loads with the MMA (large K).
    """
    is_lp = (dtype != torch.float32)        # low-precision (fp16 or bf16)
    dtype_bytes = 2 if is_lp else 4
    ops = float(M) * N * K
    aspect = max(M, N) / max(1, min(M, N))

    # Primary candidate by size and dtype (M5 Pro sweeps): small problems use a
    # bigger no-dbuf tile; large/K-dominated low-precision prefer overlapped tiles.
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
    """Return (A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b) describing
    how the kernel reads the original memory, avoiding materialized copies."""
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


# Complex GEMV + pack kernel handles, resolved once per complex dtype.
_CGEMV_HANDLES: dict = {}
_CGEMV_T_NWARPS = 8     # K-split simdgroups for y = x @ B
_CGEMV_NT_NWARPS = 4    # rows per threadgroup for y = A @ x


def _cgemv_handles(cdt):
    h = _CGEMV_HANDLES.get(cdt)
    if h is None:
        c2, r = _COMPLEX[cdt]
        t, _ = kernels.cgemv_t(c2, "float2", r, 32, _CGEMV_T_NWARPS)
        nt, _ = kernels.cgemv_nt(c2, "float2", r, _CGEMV_NT_NWARPS)
        split, combine = kernels.complex_pack(c2, r)
        h = {"t": t, "nt": nt, "split": split, "combine": combine}
        _CGEMV_HANDLES[cdt] = h
    return h


def _complex_matmul(a: torch.Tensor, b: torch.Tensor,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """Complex C = A @ B: native interleaved-complex GEMV for M==1 / N==1, else
    four real products (ar@br - ai@bi) + i(ar@bi + ai@br) on the tuned backend."""
    cdt = a.dtype if a.dtype in _COMPLEX else b.dtype
    # promote a real operand (torch allows complex @ real); resolve lazy conj() views
    if a.dtype != cdt:
        a = a.to(cdt)
    if b.dtype != cdt:
        b = b.to(cdt)
    if a.is_conj():
        a = a.resolve_conj()
    if b.is_conj():
        b = b.resolve_conj()
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dim() == 2 and b.dim() == 2, "complex matmul currently expects 2-D inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"
    if out is not None:
        assert out.shape == (M, N) and out.dtype == cdt and out.device.type == "mps", \
            f"out must be ({M}, {N}) {cdt} on mps"
    h = _cgemv_handles(cdt)

    # y = x @ B (M==1), B contiguous (ldb == N)
    if M == 1 and a.is_contiguous() and b.is_contiguous() and b.stride(0) == N and N >= 1:
        if out is None:
            out = torch.empty(1, N, dtype=cdt, device=a.device)
        ng = (N + 31) // 32
        h["t"](b, a.view(-1), out.view(-1), _pk(N, K, N, 1),
               threads=(_CGEMV_T_NWARPS * 32 * ng, 1, 1),
               group_size=(_CGEMV_T_NWARPS * 32, 1, 1))
        return out
    # y = A @ x (N==1), A contiguous (lda == K)
    if N == 1 and a.is_contiguous() and b.is_contiguous() and a.stride(0) == K and M >= 1:
        if out is None:
            out = torch.empty(M, 1, dtype=cdt, device=a.device)
        ng = (M + _CGEMV_NT_NWARPS - 1) // _CGEMV_NT_NWARPS
        h["nt"](a, b.view(-1), out.view(-1), _pk(M, K, K, 1),
                threads=(_CGEMV_NT_NWARPS * 32 * ng, 1, 1),
                group_size=(_CGEMV_NT_NWARPS * 32, 1, 1))
        return out

    # GEMM: deinterleave to real planes, four real GEMMs, fold back to complex.
    a = a.contiguous()
    b = b.contiguous()
    rdt = _COMPLEX_REAL[cdt]
    ar = torch.empty(M, K, dtype=rdt, device=a.device)
    ai = torch.empty(M, K, dtype=rdt, device=a.device)
    br = torch.empty(K, N, dtype=rdt, device=b.device)
    bi = torch.empty(K, N, dtype=rdt, device=b.device)
    nA, nB = M * K, K * N
    h["split"](a, ar, ai, nA, threads=(nA, 1, 1), group_size=(256, 1, 1))
    h["split"](b, br, bi, nB, threads=(nB, 1, 1), group_size=(256, 1, 1))
    P = matmul(ar, br)
    Q = matmul(ai, bi)
    S = matmul(ar, bi)
    T = matmul(ai, br)
    if out is None:
        out = torch.empty(M, N, dtype=cdt, device=a.device)
    nC = M * N
    h["combine"](P, Q, S, T, out, nC, threads=(nC, 1, 1), group_size=(256, 1, 1))
    return out


# Integer matmul (no tensor/simdgroup-matrix unit): register-tiled int_gemm + the
# gemv_t/gemv_nt templates. ACC_T >= output width, then truncate -> bit-exact vs
# torch's wrap-on-overflow.
_INT_PROFILE = {
    torch.int8:  ("char",  "int",  "char"),
    torch.uint8: ("uchar", "uint", "uchar"),
    torch.int16: ("short", "int",  "short"),
    torch.int32: ("int",   "int",  "int"),
    torch.int64: ("long",  "long", "long"),
}
_INT_BYTES = {torch.int8: 1, torch.uint8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8}

# GEMV (VEC cols/lane, NWARPS K-split); VEC clamped to alignment at call time.
_IGEMV_T_CFG = {     # M==1: y = x @ B (gemv_t)
    torch.int8: (8, 8), torch.uint8: (8, 8), torch.int16: (4, 8),
    torch.int32: (2, 8), torch.int64: (1, 8),
}
_IGEMV_NT_CFG = {    # N==1: y = A @ x (gemv_nt); int64 reduces via threadgroup mem
    torch.int8: (8, 4), torch.uint8: (8, 4), torch.int16: (4, 4),
    torch.int32: (4, 8), torch.int64: (2, 4),
}


def _int_clamp_vec(vec: int, ld: int, off: int) -> int:
    """Largest power-of-2 <= vec dividing both the row stride and base offset."""
    a = int(ld) | int(off)
    while vec > 1 and (a % vec):
        vec >>= 1
    return vec


def _pick_int_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int, int, int]:
    """(BM, BN, BK, TX, TY) for int_gemm, from M5 Pro sweeps."""
    nbytes = _INT_BYTES[dtype]
    mx = max(M, N)
    if mx <= 256:                              # small: keep enough threadgroups
        return (64, 64, 16, 16, 16)
    if M <= 16 and N >= 1024:                  # M<=16 fills exactly one BM=16 row-tile
        return (16, 64, 16, 16, 16)
    if M <= 128 and N >= 1024:                 # thin-M wide-N: short BM
        return (32, 64, 16, 8, 16)
    if nbytes == 8:                            # int64: shallow BK caps threadgroup mem
        return (64, 64, 8, 16, 16)
    if nbytes == 1 and M >= 512:               # int8/uint8 large: tall BM amortizes
        return (128, 64, 16, 16, 16)
    return (64, 64, 16, 16, 16)


def _int_gemv_t(dtype, matrix, x_vec, xs, out, cols, k, ld):
    # y[c] = sum_k matrix[k,c] * x[k]; matrix read VEC-wide along c, K split over NWARPS.
    in_t, acc_t, out_t = _INT_PROFILE[dtype]
    vec, nw = _IGEMV_T_CFG[dtype]
    vec = _int_clamp_vec(vec, ld, matrix.storage_offset())
    fn, _ = kernels.gemv_t(in_t, acc_t, out_t, 32 * vec, nw, vec)
    ng = (cols + 32 * vec - 1) // (32 * vec)
    fn(matrix, x_vec, out.view(-1), _pk(cols, k, ld, int(xs)),
       threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _int_gemv_nt(dtype, matrix, x_vec, out, rows, k, ld):
    # y[r] = sum_k matrix[r,k] * x[k]; one warp per row (int64 reduces via tgroup mem).
    in_t, acc_t, out_t = _INT_PROFILE[dtype]
    vec, nw = _IGEMV_NT_CFG[dtype]
    vec = _int_clamp_vec(vec, ld, matrix.storage_offset())
    fn, _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, nw, vec, red_tg=(dtype is torch.int64))
    ng = (rows + nw - 1) // nw
    fn(matrix, x_vec, out.view(-1), _pk(rows, k, ld),
       threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _int_matmul(a: torch.Tensor, b: torch.Tensor,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """Integer C = A @ B: native GEMV for rank-1 (M==1 / N==1), else a register-tiled
    integer GEMM. Bit-exact vs torch (same two's-complement wrapping)."""
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype, "integer matmul requires both operands the same dtype"
    dtype = a.dtype
    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)
    if out is None:
        out = a.new_empty(M, N)
    else:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps"
    ldc = out.stride(0)

    # Rank-1 fast paths: gemv_t (y = x @ B), gemv_nt (y = A @ x).
    if M == 1 and N >= 16:
        if not trans_b:
            xv = A_view.reshape(-1)
            return _int_gemv_t(dtype, B_view, xv, xv.stride(0), out, N, K, ldb)
        xv = A_view.reshape(-1).contiguous()
        return _int_gemv_nt(dtype, B_view, xv, out, N, K, ldb)
    elif N == 1 and M >= 16:
        if trans_a:
            xv = B_view.reshape(-1)
            return _int_gemv_t(dtype, A_view, xv, xv.stride(0), out, M, K, lda)
        xv = B_view.reshape(-1).contiguous()
        return _int_gemv_nt(dtype, A_view, xv, out, M, K, lda)

    in_t, acc_t, out_t = _INT_PROFILE[dtype]
    BM, BN, BK, TX, TY = _pick_int_tile(int(M), int(N), int(K), dtype)
    fn, _ = kernels.int_gemm(in_t, acc_t, out_t, BM, BN, BK, TX, TY, trans_a, trans_b)
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    grp = TX * TY
    fn(A_view, B_view, out, _pk(M, N, K, lda, ldb, ldc),
       threads=(grp * tiles_n, tiles_m, 1), group_size=(grp, 1, 1))
    return out


def matmul(a: torch.Tensor, b: torch.Tensor, *,
           backend: str | None = None,
           tile: tuple | None = None,
           swizzle_log: int | None = None,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute C = A @ B on the MPS device with our hand-tuned Metal kernels.

    Parameters
    ----------
    a, b : torch.Tensor on device 'mps'
    backend : "auto" (default), "simd", "mpp", "gemv"
    tile  : (BM, BN, BK, WM, WN) - override the tile heuristic
    swizzle_log : override the swizzle (0, 1, 2 …)
    out   : optional preallocated output of shape (M, N)
    """
    # Complex dtypes route to the dedicated complex path (native GEMV for rank-1
    # problems, decomposed real GEMMs otherwise). backend/tile/swizzle don't apply.
    if a.is_mps and (a.dtype in _COMPLEX or b.dtype in _COMPLEX):
        return _complex_matmul(a, b, out=out)

    # Integer dtypes: dedicated path (backend/tile/swizzle don't apply).
    if a.is_mps and a.dtype in _INT_PROFILE:
        return _int_matmul(a, b, out=out)

    # Lean GEMV fast path (checked BEFORE the asserts)
    # Row-vector x matrix runs in ~7-12 us, so the Python preamble IS the
    # bottleneck. Gate on cheap reads and route contiguous M==1 GEMV to gemv_t.
    sa = a.shape
    if (sa[0] == 1 and backend is None and out is None and swizzle_log is None
            and a.is_mps and len(sa) == 2 and b.ndim == 2):
        dtype = a.dtype
        sb = b.shape
        N = sb[1]
        K = sa[1]
        if (dtype in _PROFILE and N >= 16 and sb[0] == K
                and a.is_contiguous() and b.is_contiguous()):
            # Memoized launch (pure in (dtype, N)): one dict hit + alloc +
            # enqueue. b is (K×N) contiguous so stride(0)==N, passed as ldb.
            gt, thr, grp, dims = _gemv_plan(dtype, N, K)
            o = _pooled_out(b, 1, N)
            gt(b, a, o, dims, threads=thr, group_size=grp)  # xs=1: contiguous a
            return o

    # Lean GEMV fast path: N==1 (matrix × column-vector)
    # Symmetric to M==1: gemv_nt runs in ~7-25 us, so the ~2 us preamble decides
    # parity vs win. Memoize the variant pick by (dtype, K) (ld == K, contiguous).
    if (b.ndim == 2 and b.shape[1] == 1 and backend is None and out is None
            and swizzle_log is None and a.is_mps and a.ndim == 2 and sa[0] > 1):
        dtype = a.dtype
        M = sa[0]
        K = sa[1]
        if (dtype in _PROFILE and M >= 16 and b.shape[0] == K
                and a.is_contiguous() and b.is_contiguous()):
            fn, grp, k_hi, k_lda = _gemv_nt_plan(dtype, K)
            o = _pooled_out(a, M, 1)
            n_groups = (M + 3) // 4
            fn(a, b.view(-1), o.view(-1), [M | k_hi, k_lda],
               threads=(grp[0] * n_groups, 1, 1), group_size=grp)
            return o

    # Lean GEMM fast path (contiguous, untransposed, mpp_tensor regime)
    # Common GEMM routes to mpp_tensor; the ~2 us preamble isn't hidden behind short
    # kernels, so memoize the launch by (dtype,M,N,K). Non-matching inputs fall through.
    if (backend is None and out is None and swizzle_log is None
            and a.is_mps and a.ndim == 2 and b.ndim == 2):
        M = sa[0]; K = sa[1]
        sb = b.shape
        N = sb[1]
        dtype = a.dtype
        # N>=32 and M>=2: thin-N and batched-decode GEMM belong on mpp_tensor
        # (BN=32/short-BM tiles beat manual mpp); autotuner picks the tile.
        if (M >= 2 and N >= 32 and K >= 64 and sb[0] == K and dtype in _PROFILE
                and a.is_contiguous() and b.is_contiguous()
                and kernels.has_metal4()):
            # Pass a, b so a first-sight ambiguous shape autotunes, cached thereafter.
            # Plan is the lean (fn, thr, grp) tuple, or a _SplitKPlan for deep-K.
            plan = _gemm_plan(dtype, M, N, K, a, b)
            o = _pooled_out(a, M, N)
            if type(plan) is tuple:
                fn, thr, grp = plan
                fn(a, b, o, _pk(M, N, K, K, N, N), threads=thr, group_size=grp)
            else:
                plan.run(a, b, o)
            return o

    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype
    dtype = a.dtype
    if dtype not in _PROFILE:
        raise NotImplementedError(f"dtype {dtype} not supported")

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)

    if out is None:
        # Recycle a released buffer (see _pooled_out): for short GEMM the alloc
        # tail isn't hidden behind the kernel, so this gets small shapes to parity.
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
        # mpp_tensor reads any unit-inner-stride operand directly through a strided
        # tensor_inline view (leading dim = lda/ldb), so packed, [::2]-strided and
        # col-major all ride it. _resolve_inputs already contiguified anything with
        # no unit-stride dim, so by here every operand is expressible that way.
        packed_ab = (lda_ == K_ and ldb_ == N_)                  # NN-packed (M==1 fallback below)
        # The mpp_tensor / mpp kernels need Metal 4 cooperative-tensor headers
        # (macOS 26+). When absent, route everything non-GEMV to the simd kernel.
        m4 = kernels.has_metal4()
        # Wide-N M==1 fallback (only when an explicit arg bypassed gemv_t): padded
        # mpp_tensor zeros OOB rows, beating VEC=1 gemv on views. bf16/fp16 only.
        if (m4 and M == 1 and N >= 4096 and K >= 256 and is_lp
                and not trans_a and not trans_b and packed_ab):
            backend = "mpp_tensor"
        # Everything else GEMV-shaped goes to the dedicated kernel, which fills
        # the GPU on small-N by splitting K across 32 simdgroups. fp32 too.
        elif M == 1 and N >= 16:
            backend = "gemv"
        elif N == 1 and M >= 16:
            backend = "gemv"
        elif not m4:
            # macOS < 26: no cooperative-tensor headers. The SIMD-group GEMM is the
            # general fallback (handles transposed / unaligned / tiny shapes too).
            backend = "simd"
        # mpp_tensor (MPP op.run) wins across all sizes/dtypes (fp32 TF32-relaxed);
        # the strided view absorbs transposed + leading-dim ([::2]) operands, so
        # everything above the tile floor rides it. Tiny shapes fall to mpp.
        elif (M >= 2 and N >= 32 and K >= 64):
            backend = "mpp_tensor"
        else:
            backend = "mpp"

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
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=total, group_size=group_size)
        return out

    if backend == "mpp_tensor":
        # Tile = (BM, BN, NSG)
        if tile is None:
            BM, BN, NSG = _pick_mpp_tensor_tile(M_, N_, K_, dtype)
        else:
            assert len(tile) >= 3
            BM, BN, NSG = tile[0], tile[1], tile[2]
        tiles_m = (M_ + BM - 1) // BM
        tiles_n = (N_ + BN - 1) // BN
        mn_aligned = (M_ % BM == 0) and (N_ % BN == 0)
        # MPP path has its own internal scheduling; an external swizzle
        # hurts it (empirically loses 15-20% TF).  Leave it at 0.
        swz = swizzle_log if swizzle_log is not None else 0
        fn, _ = kernels.mpp_tensor_gemm(
            in_t, out_t, BM, BN, NSG, trans_a, trans_b,
            relaxed=True, swizzle_log=swz, mn_aligned=mn_aligned,
        )
        group_size = (NSG * 32, 1, 1)
        tile_factor = 1 << swz
        tn_swz = tiles_n * tile_factor
        tm_swz = (tiles_m + tile_factor - 1) // tile_factor
        total = (group_size[0] * tn_swz, tm_swz, 1)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=total, group_size=group_size)
        return out

    if backend == "mpp":
        pad = None
        if tile is None:
            BM, BN, BK, WM, WN, dbuf = _pick_mpp_tile(M_, N_, K_, dtype)
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
        fn, _ = kernels.mpp_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN,
                                 trans_a, trans_b, mn_aligned, k_aligned,
                                 relaxed=True, swizzle_log=swz, dbuf=dbuf, pad=pad)
        group_size = (WM * WN * 32, 1, 1)
        tile_factor = 1 << swz
        tn_swz = tiles_n * tile_factor
        tm_swz = (tiles_m + tile_factor - 1) // tile_factor
        total = (group_size[0] * tn_swz, tm_swz, 1)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=total, group_size=group_size)
        return out

    raise ValueError(f"unknown backend {backend}")


def _gemv_nt(nt_dict, matrix, vec, out_v, rows, k, ld):
    # y[r] = sum_k matrix[r, k] * vec[k] (matrix row-major M×K, one warp/row).
    n_groups = (rows + 3) // 4          # NWARPS=4, ROWS_PER_SG=1
    fn = _gemv_nt_pick(nt_dict, rows, k, ld, matrix.storage_offset())
    fn(matrix, vec, out_v, _pk(rows, k, ld),
       threads=(128 * n_groups, 1, 1), group_size=(128, 1, 1))


def _gemv_t(gh, matrix, x_vec, xs, out_v, cols, k, ld, dtype):
    # y[c] = sum_k matrix[k, c] * x[k]. gt splits K across simdgroups, reducing in
    # threadgroup memory. The matrix is read VEC-wide, the vector scalar (stride xs).
    # VEC-wide matrix loads need both its row stride AND its base offset VEC-aligned:
    # a VecT at element (off + k*ld + n0) is aligned for all k iff off and ld are.
    # OR-ing them gives the combined low-bit alignment the picker clamps on, while
    # the kernel still gets the real ld. Aligned views (incl. contiguous) now get
    # the full VEC instead of a half-line VEC=1 read.
    align = int(ld) | int(matrix.storage_offset())
    gt, tg, vec = _gemv_pick(gh, cols, align, dtype, vec_ok=True, k=int(k))
    n_groups = (cols + 32 * vec - 1) // (32 * vec)
    gt(matrix, x_vec, out_v, _pk(cols, k, ld, int(xs)),
       threads=(tg * n_groups, 1, 1), group_size=(tg, 1, 1))


def _dispatch_gemv(A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, dtype, out):
    gh = _gemv_handles(dtype)
    M, N, K = int(M), int(N), int(K)
    nt = gh["nt"]
    yv = out.view(-1)
    if M == 1:
        if trans_b:
            _gemv_nt(nt, B_view, A_view.reshape(-1).contiguous(), yv, N, K, int(ldb))
        else:
            xv = A_view.reshape(-1)
            _gemv_t(gh, B_view, xv, xv.stride(0), yv, N, K, int(ldb), dtype)
        return out
    elif N == 1:
        if trans_a:
            xv = B_view.reshape(-1)
            _gemv_t(gh, A_view, xv, xv.stride(0), yv, M, K, int(lda), dtype)
        else:
            _gemv_nt(nt, A_view, B_view.reshape(-1).contiguous(), yv, M, K, int(lda))
        return out
    else:
        raise ValueError("gemv backend requires M==1 or N==1")


# addmm  (C = beta*input + alpha*(A@B))
# The bias+scales are fused into each GEMM/GEMV kernel via EPILOGUE builds, so addmm
# costs ~the same as the bare matmul.
# Backend selection mirrors matmul()'s; only the chosen kernel is rebuilt with the
# epilogue and handed bias + broadcast strides + scalars.
def _addmm_gemv_t(dtype, matrix, xv, xs, out, cols, k, ld, bstep, bias, beta, alpha, bnz, anz):
    # y[c] = sum_k matrix[k,c]*x[k]; bias indexed by the output col -> bstep.
    if dtype in _INT_PROFILE:
        in_t, acc_t, out_t = _INT_PROFILE[dtype]
        vec, nw = _IGEMV_T_CFG[dtype]
        vec = _int_clamp_vec(vec, ld, matrix.storage_offset())
    else:
        in_t, acc_t, out_t = _PROFILE[dtype]
        _, tg, vec = _gemv_pick(_gemv_handles(dtype), cols,
                                int(ld) | matrix.storage_offset(), dtype, k=int(k))
        nw = tg // 32
    fn, _ = kernels.gemv_t(in_t, acc_t, out_t, 32 * vec, nw, vec,
                           epilogue=True, beta_nz=bnz, alpha_nz=anz)
    ng = (cols + 32 * vec - 1) // (32 * vec)
    fn(matrix, xv, out.view(-1), _pk(cols, k, ld, int(xs)), bias, int(bstep), beta, alpha,
       threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _addmm_gemv_nt(dtype, matrix, xv, out, rows, k, ld, bstep, bias, beta, alpha, bnz, anz):
    # y[r] = sum_k matrix[r,k]*x[k]; bias indexed by the output row -> bstep.
    if dtype in _INT_PROFILE:
        in_t, acc_t, out_t = _INT_PROFILE[dtype]
        vec, nw = _IGEMV_NT_CFG[dtype]
        vec = _int_clamp_vec(vec, ld, matrix.storage_offset())
        red_tg = dtype is torch.int64
    else:
        in_t, acc_t, out_t = _PROFILE[dtype]
        nw, red_tg = 4, False
        vec = _gemv_nt_vec(dtype, int(k), int(ld), matrix.storage_offset())
    fn, _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, nw, vec, red_tg=red_tg,
                            epilogue=True, beta_nz=bnz, alpha_nz=anz)
    ng = (rows + nw - 1) // nw
    fn(matrix, xv, out.view(-1), _pk(rows, k, ld), bias, int(bstep), beta, alpha,
       threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _gemv_nt_vec(dtype, k, ld, off):
    """The VEC gemv_nt would pick for (dtype, K, alignment) - mirrors _gemv_nt_pick."""
    if dtype is torch.float32:
        return 1
    align = int(ld) | int(off)
    if _HAS_TENSOR_UNIT:
        if not (align & 3) and k >= 512: return 4
        if not (align & 1) and k >= 512: return 2
        return 1
    if not (align & 3) and k >= 64: return 4
    if not (align & 1) and k >= 32: return 2
    return 1


def _addmm_real(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out):
    """Fused real-dtype addmm: rank-1 -> gemv, else the tiled GEMM for this backend."""
    dtype = mat1.dtype
    is_int = dtype in _INT_PROFILE
    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(mat1, mat2)
    if out is None:
        out = _pooled_out(mat1, M, N)
    ldc = out.stride(0)
    bstride = _pk(int(br), int(bc))               # int2 (row, col) broadcast strides

    # Rank-1 fast paths (bias is 1-D over the output index: col for M==1, row for N==1).
    if M == 1 and N >= 16:
        if not trans_b:
            xv = A_view.reshape(-1)
            return _addmm_gemv_t(dtype, B_view, xv, xv.stride(0), out, N, K, ldb, bc, x, beta, alpha, bnz, anz)
        xv = A_view.reshape(-1).contiguous()
        return _addmm_gemv_nt(dtype, B_view, xv, out, N, K, ldb, bc, x, beta, alpha, bnz, anz)
    if N == 1 and M >= 16:
        if trans_a:
            xv = B_view.reshape(-1)
            return _addmm_gemv_t(dtype, A_view, xv, xv.stride(0), out, M, K, lda, br, x, beta, alpha, bnz, anz)
        xv = B_view.reshape(-1).contiguous()
        return _addmm_gemv_nt(dtype, A_view, xv, out, M, K, lda, br, x, beta, alpha, bnz, anz)

    M_, N_, K_, lda_, ldb_, ldc_ = int(M), int(N), int(K), int(lda), int(ldb), int(ldc)
    dims6 = _pk(M_, N_, K_, lda_, ldb_, ldc_)

    if is_int:
        in_t, acc_t, out_t = _INT_PROFILE[dtype]
        BM, BN, BK, TX, TY = _pick_int_tile(M_, N_, K_, dtype)
        fn, _ = kernels.int_gemm(in_t, acc_t, out_t, BM, BN, BK, TX, TY, trans_a, trans_b,
                                 epilogue=True, beta_nz=bnz, alpha_nz=anz)
        grp = TX * TY
        fn(A_view, B_view, out, dims6, x, bstride, beta, alpha,
           threads=(grp * ((N_ + BN - 1) // BN), (M_ + BM - 1) // BM, 1), group_size=(grp, 1, 1))
        return out

    in_t, acc_t, out_t = _PROFILE[dtype]
    packed_ab = (lda_ == K_ and ldb_ == N_)
    m4 = kernels.has_metal4()
    if m4 and M_ >= 2 and N_ >= 32 and K_ >= 64 and not trans_a and not trans_b and packed_ab:
        # Reuse matmul's autotuner: the winning tile transfers (the epilogue is a
        # cheap store-time change), so awkward shapes get the same fast tile here.
        _gemm_plan(dtype, M_, N_, K_, A_view, B_view)
        tile = _GEMM_TILE[(dtype, M_, N_, K_)]
        # An mpp_tensor tile is a 3-tuple; a 4/5-tuple means the autotuner chose a
        # split-K or conv plan (no epilogue) - fall back to the heuristic tile.
        if len(tile) == 3:
            BM, BN, NSG = tile
        else:
            BM, BN, NSG = _pick_mpp_tensor_tile(M_, N_, K_, dtype)
        mn_aligned = (M_ % BM == 0) and (N_ % BN == 0)
        fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, False, False,
                                       relaxed=True, swizzle_log=0, mn_aligned=mn_aligned,
                                       epilogue=True, beta_nz=bnz, alpha_nz=anz)
        grp = (NSG * 32, 1, 1)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_), x, bstride, float(beta), float(alpha),
           threads=(grp[0] * ((N_ + BN - 1) // BN), (M_ + BM - 1) // BM, 1), group_size=grp)
        return out

    # Transposed / non-packed (m4) or no Metal 4: the manual tiled fallbacks.
    if m4:
        BM, BN, BK, WM, WN, dbuf = _pick_mpp_tile(M_, N_, K_, dtype)
        builder = lambda: kernels.mpp_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN, trans_a, trans_b,
                                          (M_ % BM == 0) and (N_ % BN == 0), K_ % BK == 0,
                                          relaxed=True, swizzle_log=0, dbuf=dbuf,
                                          epilogue=True, beta_nz=bnz, alpha_nz=anz)
    else:
        BM, BN, BK, WM, WN = _pick_simd_tile(M_, N_, K_, dtype)
        builder = lambda: kernels.simd_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN, trans_a, trans_b,
                                            (M_ % BM == 0) and (N_ % BN == 0), K_ % BK == 0,
                                            swizzle_log=0, epilogue=True, beta_nz=bnz, alpha_nz=anz)
    fn, _ = builder()
    grp = (WM * WN * 32, 1, 1)
    fn(A_view, B_view, out, dims6, x, bstride, float(beta), float(alpha),
       threads=(grp[0] * ((N_ + BN - 1) // BN), (M_ + BM - 1) // BM, 1), group_size=grp)
    return out


def _addmm_complex(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out):
    """Complex addmm: four real products folded into one complex_combine pass that
    also applies beta*input + alpha*(.) - no extra sweep beyond the GEMM's own combine."""
    cdt = mat1.dtype
    a = mat1.resolve_conj().contiguous() if mat1.is_conj() else mat1.contiguous()
    b = mat2.resolve_conj().contiguous() if mat2.is_conj() else mat2.contiguous()
    M, K = a.shape
    _, N = b.shape
    c2, r = _COMPLEX[cdt]
    rdt = _COMPLEX_REAL[cdt]

    # Rank-1 fast paths: native interleaved-complex cgemv with bias fused in (~2x the
    # 4-GEMV decomposition). beta/alpha split into re/im (a Python real has imag 0).
    bre, bim = float(beta.real), float(beta.imag)
    are, aim = float(alpha.real), float(alpha.imag)
    if M == 1 and N >= 1:
        out = out if out is not None else torch.empty(1, N, dtype=cdt, device=a.device)
        fn, _ = kernels.cgemv_t(c2, "float2", r, 32, _CGEMV_T_NWARPS,
                                epilogue=True, beta_nz=bnz, alpha_nz=anz)
        ng = (N + 31) // 32
        fn(b, a.view(-1), out.view(-1), _pk(N, K, N, 1),
           x, int(bc), bre, bim, are, aim,
           threads=(_CGEMV_T_NWARPS * 32 * ng, 1, 1),
           group_size=(_CGEMV_T_NWARPS * 32, 1, 1))
        return out
    if N == 1 and M >= 1:
        out = out if out is not None else torch.empty(M, 1, dtype=cdt, device=a.device)
        fn, _ = kernels.cgemv_nt(c2, "float2", r, _CGEMV_NT_NWARPS,
                                 epilogue=True, beta_nz=bnz, alpha_nz=anz)
        ng = (M + _CGEMV_NT_NWARPS - 1) // _CGEMV_NT_NWARPS
        fn(a, b.view(-1), out.view(-1), _pk(M, K, K, 1),
           x, int(br), bre, bim, are, aim,
           threads=(_CGEMV_NT_NWARPS * 32 * ng, 1, 1),
           group_size=(_CGEMV_NT_NWARPS * 32, 1, 1))
        return out

    ar = torch.empty(M, K, dtype=rdt, device=a.device); ai = torch.empty(M, K, dtype=rdt, device=a.device)
    bre = torch.empty(K, N, dtype=rdt, device=b.device); bim = torch.empty(K, N, dtype=rdt, device=b.device)
    split, _ = kernels.complex_pack(c2, r)
    nA, nB = M * K, K * N
    split(a, ar, ai, nA, threads=(nA, 1, 1), group_size=(256, 1, 1))
    split(b, bre, bim, nB, threads=(nB, 1, 1), group_size=(256, 1, 1))
    P = matmul(ar, bre); Q = matmul(ai, bim); S = matmul(ar, bim); T = matmul(ai, bre)
    if out is None:
        out = torch.empty(M, N, dtype=cdt, device=a.device)
    _, combine = kernels.complex_pack(c2, r, epilogue=True, beta_nz=bnz, alpha_nz=anz)
    nC = M * N
    combine(P, Q, S, T, out, nC, x, _pk(int(N), int(br), int(bc), 0),
            float(beta.real), float(beta.imag), float(alpha.real), float(alpha.imag),
            threads=(nC, 1, 1), group_size=(256, 1, 1))
    return out


def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *,
          beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """C = beta*input + alpha*(mat1 @ mat2), matching torch.addmm (2-D only).

    The product runs on the same tuned kernels as matmul, with the bias and scales
    fused into the EPILOGUE build of each kernel.
    """
    assert mat1.is_mps and mat2.is_mps
    assert mat1.dim() == 2 and mat2.dim() == 2, "addmm expects 2-D mat1, mat2"
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2, f"shape mismatch: mat1 is {mat1.shape}, mat2 is {mat2.shape}"
    dtype = mat1.dtype
    assert mat2.dtype == dtype, "addmm requires mat1.dtype == mat2.dtype"
    if out is not None:
        assert out.shape == (M, N) and out.dtype == dtype and out.is_mps

    # Coerce scalars to the kernel's arg type: ints truncate toward zero (as torch);
    # fp needs real floats (an int reinterpreted as float binds ~0); complex splits re/im.
    if dtype in _INT_PROFILE:
        beta, alpha = int(beta), int(alpha)
    elif dtype not in _COMPLEX:
        beta, alpha = float(beta), float(alpha)
    bnz, anz = beta != 0, alpha != 0

    # input -> result dtype, then expand to (M,N): broadcast dims get stride 0, so
    # the kernel reads input[i*br + j*bc] uniformly for any broadcastable shape.
    x = input if input.dtype == dtype else input.to(dtype)
    xe = x.expand(M, N)
    br, bc = (int(s) for s in xe.stride())

    if dtype in _COMPLEX:
        return _addmm_complex(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out)
    return _addmm_real(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out)


def _pick_bmm_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """(BM, BN, NSG) for a batched matmul (autotuner candidate 0 / autotune-off pick).

    Unlike the 2-D pick, the batch already saturates the cores, so we never need tiny
    tiles to make work - a bigger tile (more K-reuse) and NSG=4 (latency hiding across
    the independent batch) win. From M5 Pro batched sweeps."""
    if M == 1 or N == 1:                     # rank-1: reuse the padded-GEMV tile
        return _pick_mpp_tensor_tile(M, N, K, dtype)
    mx, mn = max(M, N), min(M, N)
    if K <= 128 and mx >= 512:               # wide-N / tiny-K (attention scores): wide BN
        return (32, 128, 4)
    if mn <= 32:                             # tiny matrix: match it, don't pad to 64x64
        return (32, 32, 4)
    if mx <= 256:                            # small per-matrix: 64x64x4 (2-D pick goes tiny)
        return (64, 64, 4)
    return (64, 64, 2)                        # medium / large default


# Batched tile candidates probed by the autotuner; the winning tile shifts with
# aspect/K (sweeps), so probe on the real operands and cache by (dtype,M,N,K,trans).
_BMM_TILES = [(64, 64, 2), (64, 64, 4), (32, 64, 2), (32, 128, 4), (64, 128, 4),
              (128, 64, 4), (128, 32, 4), (128, 128, 8)]
_BMM_PLAN: dict = {}


def _bmm_candidates(M, N, K, dtype):
    """Candidate 0 = the heuristic; then the curated batched set, deduped and pruned
    of tiles far larger than the problem (cuts probe cost without losing a winner)."""
    primary = _pick_bmm_tile(M, N, K, dtype)
    cands = [primary]
    for t in _BMM_TILES:
        if t not in cands and t[0] <= 2 * M and t[1] <= 2 * N:
            cands.append(t)
    return cands


def _autotune_bmm(dtype, a, b, Bb, M, N, K, lda, ldb, trans_a, trans_b, sA, sB, cands, margin):
    """Time each candidate's batched launch (best-of-reps), returning (fn, BM, BN, NSG).
    Candidate 0 (heuristic) is kept unless beaten by >margin. Mirrors _autotune_mppt."""
    in_t, _, out_t = _PROFILE[dtype]
    warmup, iters, reps = _probe_params(Bb * M, N, K)
    o = a.new_empty(Bb, M, N)
    sC, ldc = int(o.stride(0)), int(o.stride(1))
    dims, bstr = _pk(M, N, K, lda, ldb, ldc), _pk(sA, sB, sC, 0)
    cand = []
    for (BM, BN, NSG) in cands:
        mn = (M % BM == 0) and (N % BN == 0)
        fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, trans_a, trans_b,
                                        relaxed=True, swizzle_log=0, mn_aligned=mn, batched=True)
        grp = (NSG * 32, 1, 1)
        thr = (grp[0] * ((N + BN - 1) // BN), (M + BM - 1) // BM, Bb)
        run = (lambda fn=fn, thr=thr, grp=grp: fn(a, b, o, dims, bstr, threads=thr, group_size=grp))
        cand.append(((fn, BM, BN, NSG), run))
    for (_, run) in cand:                    # warm every candidate before timing any
        for _ in range(warmup):
            run()
    torch.mps.synchronize()
    times = [float("inf")] * len(cand)
    for _ in range(reps):
        for j, (_, run) in enumerate(cand):
            t0 = time.perf_counter()
            for _ in range(iters):
                run()
            torch.mps.synchronize()
            times[j] = min(times[j], (time.perf_counter() - t0) / iters)
    i = min(range(len(times)), key=lambda j: times[j])
    return cand[i][0] if (i != 0 and times[i] < times[0] * (1.0 - margin)) else cand[0][0]


def _bmm_plan(dtype, M, N, K, lda, ldb, trans_a, trans_b, Bb, a=None, b=None, sA=0, sB=0):
    """Resolve (fn, BM, BN, NSG) for a batched mpp_tensor launch, autotuning the tile
    on first sight of (dtype,M,N,K,trans) and caching the winner."""
    key = (dtype, M, N, K, trans_a, trans_b)
    plan = _BMM_PLAN.get(key)
    if plan is None:
        cands = _bmm_candidates(M, N, K, dtype)
        if _AUTOTUNE and a is not None and len(cands) > 1:
            plan = _autotune_bmm(dtype, a, b, Bb, M, N, K, lda, ldb,
                                 trans_a, trans_b, sA, sB, cands, _AUTOTUNE_MARGIN)
        else:
            BM, BN, NSG = cands[0]
            in_t, _, out_t = _PROFILE[dtype]
            mn = (M % BM == 0) and (N % BN == 0)
            fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, trans_a, trans_b,
                                            relaxed=True, swizzle_log=0, mn_aligned=mn, batched=True)
            plan = (fn, BM, BN, NSG)
        _BMM_PLAN[key] = plan
    return plan


def _resolve_bmm_inputs(a: torch.Tensor, b: torch.Tensor):
    """3-D analogue of _resolve_inputs for (B,M,K) @ (B,K,N). Each matrix needs a
    unit-stride inner dim (row- or col-major) plus a uniform batch stride; anything
    else is contiguified. -> (A,B, M,N,K, lda,ldb, trans_a,trans_b, sA,sB)."""
    _, M, K = a.shape
    _, _, N = b.shape
    sa, sb = a.stride(), b.stride()
    if sa[2] == 1 and sa[1] >= K:
        trans_a, lda, A_view = False, sa[1], a
    elif sa[1] == 1 and sa[2] >= M:
        trans_a, lda, A_view = True, sa[2], a
    else:
        A_view, trans_a, lda = a.contiguous(), False, K
    if sb[2] == 1 and sb[1] >= N:
        trans_b, ldb, B_view = False, sb[1], b
    elif sb[1] == 1 and sb[2] >= K:
        trans_b, ldb, B_view = True, sb[2], b
    else:
        B_view, trans_b, ldb = b.contiguous(), False, N
    return (A_view, B_view, int(M), int(N), int(K), int(lda), int(ldb),
            trans_a, trans_b, int(A_view.stride(0)), int(B_view.stride(0)))


def _bmm_out(ref: torch.Tensor, out, Bb, M, N, dtype):
    """(write_target, needs_copyback). A user `out` whose N dim is unit-stride is
    written in place; otherwise a contiguous scratch is allocated and copied back."""
    if out is not None:
        assert out.shape == (Bb, M, N) and out.dtype == dtype and out.is_mps, \
            f"out must be ({Bb}, {M}, {N}) {dtype} on mps"
        if out.stride(2) == 1 and out.stride(1) >= N:
            return out, False
        return ref.new_empty(Bb, M, N), True
    return ref.new_empty(Bb, M, N), False


def _bmm_loop(a, b, out=None):
    """Per-matrix fallback: loop the 2-D matmul over the batch. Correct for every
    dtype/shape (integer, non-Metal4); not launch-optimal for large B."""
    Bb, M = a.shape[0], a.shape[1]
    N = b.shape[2]
    if out is None:
        out = a.new_empty(Bb, M, N)
    for i in range(Bb):
        out[i].copy_(matmul(a[i], b[i]))
    return out


def _complex_bmm(a, b, out=None):
    """Complex batched GEMM: deinterleave to real planes, four batched real bmm
    (ar@br - ai@bi) + i(ar@bi + ai@br), fold back to interleaved complex."""
    cdt = a.dtype if a.dtype in _COMPLEX else b.dtype
    if a.dtype != cdt:
        a = a.to(cdt)
    if b.dtype != cdt:
        b = b.to(cdt)
    a = a.resolve_conj().contiguous() if a.is_conj() else a.contiguous()
    b = b.resolve_conj().contiguous() if b.is_conj() else b.contiguous()
    Bb, M, K = a.shape
    N = b.shape[2]
    rdt = _COMPLEX_REAL[cdt]
    c2, r = _COMPLEX[cdt]
    ar = torch.empty(Bb, M, K, dtype=rdt, device=a.device)
    ai = torch.empty(Bb, M, K, dtype=rdt, device=a.device)
    br = torch.empty(Bb, K, N, dtype=rdt, device=b.device)
    bi = torch.empty(Bb, K, N, dtype=rdt, device=b.device)
    split, combine = kernels.complex_pack(c2, r)
    nA, nB = Bb * M * K, Bb * K * N
    split(a, ar, ai, nA, threads=(nA, 1, 1), group_size=(256, 1, 1))
    split(b, br, bi, nB, threads=(nB, 1, 1), group_size=(256, 1, 1))
    P, Q, S, T = bmm(ar, br), bmm(ai, bi), bmm(ar, bi), bmm(ai, br)
    if out is None:
        out = torch.empty(Bb, M, N, dtype=cdt, device=a.device)
    nC = Bb * M * N
    combine(P, Q, S, T, out, nC, threads=(nC, 1, 1), group_size=(256, 1, 1))
    return out


def bmm(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Batched C[i] = a[i] @ b[i] for 3-D (B,M,K) @ (B,K,N), matching torch.bmm.

    Real dtypes on Metal 4 run as a single batched mpp_tensor launch; complex
    deinterleaves into four batched real bmm; integer / non-Metal4 loop the 2-D matmul.
    """
    assert a.is_mps and b.is_mps, "bmm expects mps tensors"
    assert a.dim() == 3 and b.dim() == 3, "bmm expects 3-D inputs"
    assert a.dtype == b.dtype, "bmm requires both operands the same dtype"
    Bb, M, K = a.shape
    Bb2, K2, N = b.shape
    assert Bb == Bb2 and K == K2, f"bmm shape mismatch: {a.shape} @ {b.shape}"
    dtype = a.dtype

    if Bb == 0 or M == 0 or N == 0:                  # empty output
        o = out if out is not None else a.new_empty(Bb, M, N)
        return o
    if K == 0:                                       # sum over nothing -> zeros
        o = out if out is not None else a.new_empty(Bb, M, N)
        return o.zero_()
    if dtype in _COMPLEX:
        return _complex_bmm(a, b, out=out)
    if dtype not in _PROFILE or not kernels.has_metal4():
        return _bmm_loop(a, b, out)                  # integer / no cooperative-tensor headers

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, sA, sB = _resolve_bmm_inputs(a, b)
    oc, needs_copy = _bmm_out(a, out, Bb, M, N, dtype)
    ldc, sC = int(oc.stride(1)), int(oc.stride(0))
    fn, BM, BN, NSG = _bmm_plan(dtype, M, N, K, lda, ldb, trans_a, trans_b, Bb,
                                A_view, B_view, sA, sB)
    grp = (NSG * 32, 1, 1)
    threads = (grp[0] * ((N + BN - 1) // BN), (M + BM - 1) // BM, Bb)
    fn(A_view, B_view, oc, _pk(M, N, K, lda, ldb, ldc), _pk(sA, sB, sC, 0),
       threads=threads, group_size=grp)
    if needs_copy:
        out.copy_(oc)
        return out
    return oc


def _baddbmm_loop(input, batch1, batch2, beta, alpha, out=None):
    """Per-matrix fallback for baddbmm (integer / complex / non-Metal4)."""
    Bb, M = batch1.shape[0], batch1.shape[1]
    N = batch2.shape[2]
    x = input if input.dtype == batch1.dtype else input.to(batch1.dtype)
    xe = x.expand(Bb, M, N)
    if out is None:
        out = batch1.new_empty(Bb, M, N)
    for i in range(Bb):
        out[i].copy_(addmm(xe[i], batch1[i], batch2[i], beta=beta, alpha=alpha))
    return out


def baddbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, *,
            beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """C = beta*input + alpha*(batch1 @ batch2), matching torch.baddbmm (3-D).

    `input` is broadcastable to (B,M,N). Real dtypes fuse the bias into the batched
    mpp_tensor epilogue; complex / integer loop the 2-D addmm.
    """
    assert batch1.is_mps and batch2.is_mps, "baddbmm expects mps tensors"
    assert batch1.dim() == 3 and batch2.dim() == 3, "baddbmm expects 3-D batch1, batch2"
    assert batch1.dtype == batch2.dtype, "baddbmm requires batch1.dtype == batch2.dtype"
    Bb, M, K = batch1.shape
    Bb2, K2, N = batch2.shape
    assert Bb == Bb2 and K == K2, f"baddbmm shape mismatch: {batch1.shape} @ {batch2.shape}"
    dtype = batch1.dtype

    # Coerce scalars to the kernel arg type (mirrors addmm).
    if dtype in _INT_PROFILE:
        beta, alpha = int(beta), int(alpha)
    elif dtype not in _COMPLEX:
        beta, alpha = float(beta), float(alpha)
    bnz, anz = beta != 0, alpha != 0

    # Complex / integer / no-Metal4: the batched real epilogue kernel doesn't cover
    # them, so loop the 2-D addmm (which has the tuned complex/int epilogues).
    if dtype in _COMPLEX or dtype in _INT_PROFILE or not kernels.has_metal4():
        return _baddbmm_loop(input, batch1, batch2, beta, alpha, out)

    # input -> (B,M,N) broadcast strides (0 on stretched dims); the buffer is the base
    # tensor, the kernel reads bias[b*sBias + r*br + c*bc].
    x = input if input.dtype == dtype else input.to(dtype)
    xe = x.expand(Bb, M, N)
    sBias, br, bc = (int(s) for s in xe.stride())

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, sA, sB = _resolve_bmm_inputs(batch1, batch2)
    oc, needs_copy = _bmm_out(batch1, out, Bb, M, N, dtype)
    ldc, sC = int(oc.stride(1)), int(oc.stride(0))
    in_t, _, out_t = _PROFILE[dtype]
    # Reuse the (autotuned) bmm tile - the epilogue is a cheap store-time change.
    _, BM, BN, NSG = _bmm_plan(dtype, M, N, K, lda, ldb, trans_a, trans_b, Bb,
                               A_view, B_view, sA, sB)
    mn_aligned = (M % BM == 0) and (N % BN == 0)
    fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, trans_a, trans_b,
                                    relaxed=True, swizzle_log=0, mn_aligned=mn_aligned,
                                    epilogue=True, beta_nz=bnz, alpha_nz=anz, batched=True)
    grp = (NSG * 32, 1, 1)
    threads = (grp[0] * ((N + BN - 1) // BN), (M + BM - 1) // BM, Bb)
    fn(A_view, B_view, oc, _pk(M, N, K, lda, ldb, ldc), x, _pk(br, bc),
       float(beta), float(alpha), _pk(sA, sB, sC, sBias),
       threads=threads, group_size=grp)
    if needs_copy:
        out.copy_(oc)
        return out
    return oc


# Convenience aliases
gemm = matmul
