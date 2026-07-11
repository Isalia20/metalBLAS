"""PyTorch-facing dispatch for the Metal kernels."""
from __future__ import annotations

from math import prod
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


# PyTorch dtype -> (Metal input, accumulator, output).
_PROFILE = {
    torch.float32: ("float", "float", "float"),
    torch.float16: ("half", "float", "half"),
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


# GEMV planning

_GEMV_HANDLES: dict = {}

# (elements per lane, simdgroups per threadgroup)
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
        nt = {}
        nt[1], _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 1)
        if not _HAS_TENSOR_UNIT or dtype is not torch.float32:
            nt[2], _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 2)
            nt[4], _ = kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 4)
        h = {"gt": gt, "nt": nt}
        _GEMV_HANDLES[dtype] = h
    return h


def _gemv_nt_width(vectorized, k, ld, off=0):
    if not vectorized:
        return 1
    align = int(ld) | int(off)
    if _HAS_TENSOR_UNIT:
        if (align & 3) == 0 and k >= 512:
            return 4
        if (align & 1) == 0 and k >= 512:
            return 2
        return 1
    if (align & 3) == 0 and k >= 64:
        return 4
    if (align & 1) == 0 and k >= 32:
        return 2
    return 1


def _gemv_nt_pick(variants, _rows, k, ld, off=0):
    return variants[_gemv_nt_width(4 in variants, k, ld, off)]


def _gemv_pick(gh, cols, ldb, dtype, vec_ok=True, k=None, k_big=True):
    """Return the kernel, threadgroup size, and vector width for GEMV."""
    gt = gh["gt"]
    if k is not None:
        k_big = (k >= 2048)
    if dtype is torch.float32:
        ng = (cols + 31) // 32
        if not _HAS_TENSOR_UNIT:
            return (gt[(1, 16)], 512, 1) if ng >= 128 else (gt[(1, 32)], 1024, 1)
        if 16 < ng <= 32:
            return gt[(1, 16)], 512, 1
        return gt[(1, 32)], 1024, 1
    if not _HAS_TENSOR_UNIT:
        if vec_ok and cols > 12288:
            vec, nw = 8, 8
        elif vec_ok and cols >= 2560:
            vec, nw = 2, 32
        else:
            vec, nw = 1, 32
    elif vec_ok and cols >= 4096 and (k_big or (k is not None and k >= 1024)):
        vec = 8
        nw = 16 if (k is not None and k >= 8192) else 8
    elif vec_ok and cols >= 2560 and (k_big or (k is not None and k >= 1024)):
        vec, nw = 4, 8
    elif vec_ok and k_big and cols >= 1280:
        vec, nw = 4, 16
    elif vec_ok and cols >= 1024:
        vec, nw = 2, 16
    elif vec_ok and cols > 512:
        vec, nw = 2, 32
    else:
        vec, nw = 1, 32
    if vec == 8 and (ldb & 7):
        vec, nw = (4, 8) if not (ldb & 3) else ((2, 32) if not (ldb & 1) else (1, 32))
    elif vec == 4 and (ldb & 3):
        vec, nw = (2, 32) if not (ldb & 1) else (1, 32)
    elif vec == 2 and (ldb & 1):
        vec, nw = 1, 32
    return gt[(vec, nw)], nw * 32, vec


_GEMV_PLAN: dict = {}
_GEMV_NT_PLAN: dict = {}


def _gemv_nt_plan(dtype, K):
    """Cache the contiguous N=1 launch and the constant part of its dimensions."""
    key = (dtype, K)
    plan = _GEMV_NT_PLAN.get(key)
    if plan is None:
        gh = _gemv_handles(dtype)
        fn = _gemv_nt_pick(gh["nt"], 0, K, K)
        plan = (fn, (128, 1, 1), K << 32, K)
        _GEMV_NT_PLAN[key] = plan
    return plan


def _gemv_plan(dtype, N, K):
    key = (dtype, N, K)
    plan = _GEMV_PLAN.get(key)
    if plan is None:
        gt, tg, vec = _gemv_pick(_gemv_handles(dtype), N, N, dtype, k=K)
        ng = (N + 32 * vec - 1) // (32 * vec)
        plan = (gt, (tg * ng, 1, 1), (tg, 1, 1), _pk(N, K, N, 1))
        _GEMV_PLAN[key] = plan
    return plan


# GEMM planning and autotuning

_GEMM_PLAN: dict = {}
_GEMM_TILE: dict = {}

_AUTOTUNE = os.environ.get("METALBLAS_AUTOTUNE", "1") != "0"
_AUTOTUNE_MARGIN = 0.03
_TALL_NARROW_MARGIN = 0.01


def _build_mppt_plan(dtype, M, N, K, BM, BN, NSG):
    """Build a packed, untransposed cooperative-tensor launch."""
    in_t, _, out_t = _PROFILE[dtype]
    mn_aligned = (M % BM == 0) and (N % BN == 0)
    fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, False, False,
                                   relaxed=True, swizzle_log=0, mn_aligned=mn_aligned)
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    return (fn, (NSG * 32 * tiles_n, tiles_m, 1), (NSG * 32, 1, 1))


_SPLITK_POOL: dict = {}


def _splitk_partial(ref, planes, M, N):
    key = (M, N, planes)
    buf = _SPLITK_POOL.get(key)
    if buf is None:
        buf = ref.new_empty(planes, M, N, dtype=torch.float32)
        _SPLITK_POOL[key] = buf
    return buf


class _SplitKPlan:
    __slots__ = (
        "splitk_fn", "reduce_fn", "M", "N", "planes", "sk_threads",
        "sk_group", "red_threads", "red_group", "sk_dims", "red_dims",
    )

    def __init__(self, splitk_fn, reduce_fn, M, N, K, BM, BN, NSG, G):
        self.splitk_fn = splitk_fn
        self.reduce_fn = reduce_fn
        self.M = M
        self.N = N
        self.planes = G - 1
        tiles_m = (M + BM - 1) // BM
        tiles_n = (N + BN - 1) // BN
        self.sk_threads = (NSG * 32 * tiles_n, tiles_m, G)
        self.sk_group = (NSG * 32, 1, 1)
        n_elems = M * N
        self.red_threads = (n_elems, 1, 1)
        self.red_group = (256, 1, 1)
        self.sk_dims = _pk(M, N, K)
        self.red_dims = _pk(n_elems, self.planes)

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
    return (dtype is not torch.float32 and K >= 2048
            and 64 <= min(M, N) and M * N <= 1_500_000
            and (min(M, N) <= 256 or K >= 8 * max(M, N)))


def _splitk_specs(M, N, K):
    return [(BM, BN, NSG, G)
            for BM, BN, NSG in ((128, 32, 2), (64, 64, 2), (32, 64, 2))
            for G in (2, 4) if K % G == 0 and (K // G) % 16 == 0]


class _LaunchPlan:
    __slots__ = ("fn", "threads", "group", "dims")

    def __init__(self, fn, threads, group, dims):
        self.fn = fn
        self.threads = threads
        self.group = group
        self.dims = dims

    def run(self, a, b, o):
        self.fn(a, b, o, self.dims, threads=self.threads, group_size=self.group)


def _build_conv_plan(dtype, M, N, K, BMW, BNO, NSG):
    in_t, _, out_t = _PROFILE[dtype]
    fn, _ = kernels.conv1x1_gemm(in_t, out_t, BMW, BNO, NSG, K)
    threads = (NSG * 32 * ((N + BNO - 1) // BNO), (M + BMW - 1) // BMW, 1)
    return _LaunchPlan(fn, threads, (NSG * 32, 1, 1), _pk(M, N, K))


def _is_conv_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and N <= 64 and N % 32 == 0
            and M >= 512 and K >= 256)


def _conv_specs(M, N, K):
    return [(BMW, N, NSG) for BMW in (16, 32, 48, 64, 96, 128)
            if M % BMW == 0 for NSG in (2, 4)]


def _build_sgpipe_plan(dtype, M, N, K, SGM, SGN, KC, NSGX, NSGY, GK=0):
    in_t, _, out_t = _PROFILE[dtype]
    fn, _ = kernels.sgpipe_gemm(in_t, out_t, SGM, SGN, KC, NSGX, NSGY, GK)
    nsg = NSGX * NSGY
    threads = (nsg * 32 * (N // (NSGX * SGN)), M // (NSGY * SGM), 1)
    return _LaunchPlan(fn, threads, (nsg * 32, 1, 1), _pk(M, N, K))


def _build_flipt_plan(dtype, M, N, K, BM, BN, NSG, KC=0, PFD=0):
    in_t, _, out_t = _PROFILE[dtype]
    fn, _ = kernels.flipt_gemm(in_t, out_t, BM, BN, NSG, KC, PFD)
    threads = (NSG * 32 * ((M + BN - 1) // BN), (N + BM - 1) // BM, 1)
    return _LaunchPlan(fn, threads, (NSG * 32, 1, 1), _pk(M, N, K))


def _is_flipt_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and 128 <= N <= 256 and N % 64 == 0
            and M >= 2048 and M % 64 == 0 and K >= 512)


def _is_sgpipe_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and N in (8, 16, 32, 64, 128)
            and M >= 1024 and K >= 1024 and K % 128 == 0)


def _sgpipe_specs(M, N, K):
    if N == 128:
        cands = [(32, 64, 64, 2, 1), (32, 64, 64, 2, 2)]
    elif N == 64:
        cands = [(16, 64, 64, 1, 2), (32, 64, 64, 1, 2)]
    else:   # N in (8, 16, 32)
        cands = [(16, N, 64, 1, 2), (32, N, 64, 1, 2),
                 (16, N, 64, 1, 4), (32, N, 64, 1, 1)]
    return [spec + (gk,) for spec in cands if M % (spec[4] * spec[0]) == 0
            for gk in ((K, 0) if K <= 8192 else (0,))]


def _flipt_specs(M, N, K, a):
    specs = []
    pf_ok = K % 128 == 0 and int(a.storage_offset()) % 2 == 0
    if pf_ok and N % 128 == 0 and M % 32 == 0:
        specs.append((128, 32, 2, 128, 2))
    if pf_ok and N % 128 == 0 and M % 64 == 0:
        specs.append((128, 64, 4, 128, 2))
    if N % 64 == 0 and M % 64 == 0:
        specs.append((64, 64, 2, 0, 0))
    return specs


_GEMV_BT_MAX_M = 16
_GEMV_BT_TG_BUDGET = 192


class _GemvBtPlan:
    __slots__ = ("fn", "dims", "threads", "group")

    def __init__(self, fn, M, N, K, vec, nwarps, trans_b=False, ldb=None, ldx=None, ldy=None, ncols=1):
        self.fn = fn
        ldb = ldb if ldb is not None else (K if trans_b else N)
        self.dims = _pk(N, K, ldb, ldx if ldx is not None else K, ldy if ldy is not None else N)
        if trans_b:
            ntg = (N + nwarps * ncols - 1) // (nwarps * ncols)
        else:
            ntg = (N + 32 * vec - 1) // (32 * vec)
        self.threads = (nwarps * 32 * ntg, 1, 1)
        self.group = (nwarps * 32, 1, 1)

    def run(self, a, b, o):
        self.fn(b, a, o, self.dims, threads=self.threads, group_size=self.group)


def _build_gemv_bt_plan(dtype, M, N, K, vec, nwarps, trans_b=False, ldb=None, ldx=None, ldy=None,
                        ncols=1, trans_a=False):
    in_t, acc_t, out_t = _PROFILE[dtype]
    fn, _ = kernels.gemv_bt(in_t, acc_t, out_t, M, 32 * vec, nwarps, vec,
                            trans_b=trans_b, NCOLS=ncols, trans_a=trans_a)
    return _GemvBtPlan(fn, M, N, K, vec, nwarps, trans_b, ldb, ldx, ldy, ncols)


def _is_gemv_bt_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and 2 <= M <= _GEMV_BT_MAX_M
            and 16 <= N <= 8192 and K >= 64)


def _largest_pow2_le(x):
    x = int(x)
    return 1 << (x.bit_length() - 1) if x >= 1 else 1


def _gemv_bt_specs(M, N, K, dtype, align, trans_b=False):
    if trans_b:
        v = 8 if K >= 2048 else (4 if K >= 512 else 2)
        while v > 1 and (align % v):
            v >>= 1
        ncol_opts = [1] + ([nc for nc in (2, 4) if M * nc <= 48] if M >= 6 else [])
        return [(v, nw, nc) for nc in ncol_opts
                for nw in ((8, 4) if nc == 1 else (8,))]
    nat = 4 if N >= 4096 else (2 if N >= 256 else 1)
    vecs = []
    for v0 in (nat, 1):
        v = v0
        while v > 1 and (align % v):
            v >>= 1
        while v > 1 and M * v > 32:
            v >>= 1
        if v not in vecs:
            vecs.append(v)
    specs, seen = [], set()
    for vi, v in enumerate(vecs):
        cap = min(32, _GEMV_BT_TG_BUDGET // (M * v), max(1, (K + 31) // 32))
        wants = (cap, min(cap, 8)) if vi == 0 else (cap,)
        for want in wants:
            nw = max(1, min(cap, _largest_pow2_le(want)))
            if (v, nw) not in seen:
                seen.add((v, nw))
                specs.append((v, nw))
    return specs


def _mpp_tensor_tile_candidates(M, N, K, dtype):
    """Return the primary tile followed by candidates worth measuring."""
    primary = _pick_mpp_tensor_tile(M, N, K, dtype)
    if dtype == torch.float32:
        return [primary], _AUTOTUNE_MARGIN
    mx = max(M, N)
    mn = min(M, N)

    if M == 1 or N == 1:
        return [primary], _AUTOTUNE_MARGIN
    if K <= 256 and M >= 1024 and N >= 1024:
        return [primary], _AUTOTUNE_MARGIN
    if mx <= 256:
        return [primary], _AUTOTUNE_MARGIN

    if N < 32 and M >= 1024:
        return [primary], _TALL_NARROW_MARGIN

    if mn <= 256 and mx >= 1024:
        extra = [(128, 32, 2), (256, 32, 4), (32, 128, 2), (32, 256, 4), (64, 64, 2),
                 (64, 32, 2), (32, 64, 2)]
        if mn <= 48:
            extra = [(16, 64, 2), (32, 64, 2), (32, 128, 4)] + extra
        return _with_primary(primary, extra), _TALL_NARROW_MARGIN

    if mx <= 1024:
        if K >= 8 * mx:
            extra = [(128, 32, 4), (128, 32, 2), (192, 32, 2), (32, 64, 2), (64, 64, 2)]
            return _with_primary(primary, extra), _TALL_NARROW_MARGIN
        extra = [(16, 64, 2), (32, 64, 2), (64, 64, 2)]
    else:
        extra = [(64, 64, 2), (48, 128, 4), (64, 128, 4), (128, 64, 4)]
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
    """Scale autotuning work down as kernels get longer."""
    flops = M * N * K
    if flops <= 2_000_000_000:
        return 20, 80, 8
    if flops <= 50_000_000_000:
        return 3, 3, 3
    return 2, 3, 3


def _fastest(runs, warmup, iters, reps, margin):
    """Return the fastest run, keeping the default unless a rival clears margin."""
    for run in runs:
        for _ in range(warmup):
            run()
    torch.mps.synchronize()
    times = [float("inf")] * len(runs)
    for _ in range(reps):
        for i, run in enumerate(runs):
            t0 = time.perf_counter()
            for _ in range(iters):
                run()
            torch.mps.synchronize()
            times[i] = min(times[i], (time.perf_counter() - t0) / iters)
    best = min(range(len(times)), key=times.__getitem__)
    return best if best and times[best] < times[0] * (1.0 - margin) else 0


def _autotune_mppt(dtype, M, N, K, cands, a, b, margin=_AUTOTUNE_MARGIN, sk_specs=(),
                  conv_specs=(), bt_specs=(), flip_specs=(), sgp_specs=()):
    """Return the first candidate or one that beats it by ``margin``."""
    warmup, iters, reps = _probe_params(M, N, K)
    if margin < _AUTOTUNE_MARGIN:
        iters = max(iters * 2, 6)
        reps = max(reps, 5)
    if sgp_specs:
        iters = max(iters, 12)
    o = a.new_empty(M, N)

    mppt_dims = _pk(M, N, K, K, N, N)
    candidates = []
    for (BM, BN, NSG) in cands:
        fn, thr, grp = _build_mppt_plan(dtype, M, N, K, BM, BN, NSG)
        run = (lambda a, b, o, fn=fn, thr=thr, grp=grp, d=mppt_dims:
               fn(a, b, o, d, threads=thr, group_size=grp))
        candidates.append(((fn, thr, grp), run, (BM, BN, NSG)))

    if sk_specs or conv_specs or bt_specs or flip_specs or sgp_specs:
        ref = a.new_empty(M, N)
        candidates[0][1](a, b, ref)
        torch.mps.synchronize()
        scale = ref.abs().max().item() + 1e-6

        def accept(plan, label):
            plan.run(a, b, o)
            torch.mps.synchronize()
            if (o - ref).abs().max().item() <= 0.02 * scale:
                candidates.append((plan, plan.run, label))

        families = ((sgp_specs, _build_sgpipe_plan, "sgp"),
                    (flip_specs, _build_flipt_plan, "flip"),
                    (bt_specs, _build_gemv_bt_plan, "bt"),
                    (sk_specs, _build_splitk_plan, None),
                    (conv_specs, _build_conv_plan, "conv"))
        for specs, build, tag in families:
            for spec in specs:
                accept(build(dtype, M, N, K, *spec), (*spec, tag) if tag else spec)

    i = _fastest([lambda run=run: run(a, b, o) for _, run, _ in candidates],
                 warmup, iters, reps, margin)
    return candidates[i][0], candidates[i][2]


def _gemm_plan(dtype, M, N, K, a=None, b=None):
    key = (dtype, M, N, K)
    plan = _GEMM_PLAN.get(key)
    if plan is None:
        cands, margin = _mpp_tensor_tile_candidates(M, N, K, dtype)
        tune = _AUTOTUNE and a is not None
        sk_specs = _splitk_specs(M, N, K) if tune and _is_splitk_regime(M, N, K, dtype) else ()
        conv_specs = _conv_specs(M, N, K) if tune and _is_conv_regime(M, N, K, dtype) else ()
        flip_specs = _flipt_specs(M, N, K, a) if tune and _is_flipt_regime(M, N, K, dtype) else ()
        sgp_specs = _sgpipe_specs(M, N, K) if tune and _is_sgpipe_regime(M, N, K, dtype) else ()
        bt_specs = ()
        if tune and _is_gemv_bt_regime(M, N, K, dtype):
            align = int(b.stride(0)) | int(b.storage_offset())
            bt_specs = _gemv_bt_specs(M, N, K, dtype, align)
        if tune and (len(cands) > 1 or sk_specs or conv_specs or bt_specs or flip_specs or sgp_specs):
            plan, tile = _autotune_mppt(dtype, M, N, K, cands, a, b, margin,
                                        sk_specs, conv_specs, bt_specs, flip_specs, sgp_specs)
        else:
            tile = cands[0]
            plan = _build_mppt_plan(dtype, M, N, K, *tile)
        _GEMM_PLAN[key] = plan
        _GEMM_TILE[key] = tile
    return plan


_GEMM_TRANS_PLAN: dict = {}
_GEMM_TRANS_TILE: dict = {}


def _build_mpp_trans_plan(dtype, M, N, K, BM, BN, NSG, trans_a, trans_b, lda, ldb):
    in_t, _, out_t = _PROFILE[dtype]
    fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, trans_a, trans_b, relaxed=True,
                                    swizzle_log=0, mn_aligned=(M % BM == 0) and (N % BN == 0))
    threads = (NSG * 32 * ((N + BN - 1) // BN), (M + BM - 1) // BM, 1)
    return _LaunchPlan(fn, threads, (NSG * 32, 1, 1), _pk(M, N, K, lda, ldb, N))


def _autotune_trans(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb):
    BM, BN, NSG = _pick_mpp_tensor_tile(M, N, K, dtype)
    cand = [(_build_mpp_trans_plan(dtype, M, N, K, BM, BN, NSG, trans_a, trans_b, lda, ldb),
             ("mpp_tr", BM, BN, NSG))]
    align = lda | ldb | int(a.storage_offset()) | int(b.storage_offset())
    o, ref = a.new_empty(M, N), a.new_empty(M, N)
    cand[0][0].run(a, b, ref)
    torch.mps.synchronize()
    scale = ref.abs().max().item() + 1e-6
    t0 = time.perf_counter()
    for _ in range(3):
        cand[0][0].run(a, b, o)
    torch.mps.synchronize()
    est = (time.perf_counter() - t0) / 3
    iters = max(3, min(120, int(0.02 / max(est, 1e-7))))
    reps, warmup = 5, min(iters, 12)
    for spec in _gemv_bt_specs(M, N, K, dtype, align, trans_b=trans_b):
        vec, nwarps, ncols = spec if trans_b else (spec[0], spec[1], 1)
        plan = _build_gemv_bt_plan(dtype, M, N, K, vec, nwarps, trans_b=trans_b,
                                   ldb=ldb, ldx=lda, ldy=N, ncols=ncols, trans_a=trans_a)
        plan.run(a, b, o)
        torch.mps.synchronize()
        if (o - ref).abs().max().item() <= 0.02 * scale:
            cand.append((plan, (vec, nwarps, ncols, "bt_tr")))
    i = _fastest([lambda p=p: p.run(a, b, o) for p, _ in cand],
                 warmup, iters, reps, _AUTOTUNE_MARGIN)
    return cand[i]


def _gemm_trans_plan(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb):
    key = (dtype, M, N, K, trans_a, trans_b)
    plan = _GEMM_TRANS_PLAN.get(key)
    if plan is None:
        if _AUTOTUNE:
            plan, tile = _autotune_trans(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb)
        else:
            BM, BN, NSG = _pick_mpp_tensor_tile(M, N, K, dtype)
            plan = _build_mpp_trans_plan(dtype, M, N, K, BM, BN, NSG, trans_a, trans_b, lda, ldb)
            tile = ("mpp_tr", BM, BN, NSG)
        _GEMM_TRANS_PLAN[key] = plan
        _GEMM_TRANS_TILE[key] = tile
    return plan


_GEMV_BT_TB_MAX_N = 262144
_GEMV_BT_TB_MAX_N_BMM = 16384


# Layout and tile helpers

def _unit_lead(t, d0, d1):
    """Return ``(is_column_major, leading_stride)`` for a simple 2-D view."""
    s0, s1 = int(t.stride(-2)), int(t.stride(-1))
    if s1 == 1 and s0 >= d1:
        return (False, s0)
    if s0 == 1 and s1 >= d0:
        return (True, s1)
    return None


def _matrix_view(t, rows, cols):
    layout = _unit_lead(t, rows, cols)
    return (t, *layout) if layout else (t.contiguous(), False, cols)


def _thin_trans_layout(a, b, M, N, K, dtype):
    if (dtype is torch.float32 or not (2 <= M <= _GEMV_BT_MAX_M)
            or not (16 <= N <= _GEMV_BT_TB_MAX_N) or K < 64):
        return None
    la, lb = _unit_lead(a, M, K), _unit_lead(b, K, N)
    if la is None or lb is None or not (la[0] or lb[0]):
        return None
    return (la[0], lb[0], la[1], lb[1])


_OUT_POOL: dict = {}
_OUT_POOL_LIST_CAP = 16
_OUT_POOL_MAX_ELEMS = 1 << 21


def _pooled_out(ref, M, N):
    if M * N > _OUT_POOL_MAX_ELEMS:
        return ref.new_empty(M, N)
    key = (ref.dtype, M, N)
    pool = _OUT_POOL.get(key)
    if pool is None:
        o = ref.new_empty(M, N)
        _OUT_POOL[key] = [o]
        return o
    for i in range(len(pool)):
        if sys.getrefcount(pool[i]) == 2:
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
    """Pick ``(BM, BN, BK, WM, WN)`` for the simdgroup kernel."""
    dtype_bytes = 4 if dtype == torch.float32 else 2
    candidates = [
        (128, 128, 16, 4, 4),
        (128, 128, 32, 4, 4),
        (64, 128, 16, 2, 4),
        (128, 64, 16, 4, 2),
        (64, 64, 32, 2, 2),
        (64, 64, 16, 2, 2),
        (32, 128, 16, 1, 4),
        (128, 32, 16, 4, 1),
        (32, 64, 16, 1, 2),
        (64, 32, 16, 2, 1),
        (32, 32, 32, 1, 1),
        (32, 32, 16, 1, 1),
        (16, 16, 16, 1, 1),
    ]
    best = None
    for (BM, BN, BK, WM, WN) in candidates:
        bytes_needed = _threadgroup_bytes(BM, BN, BK, dtype_bytes)
        if bytes_needed > 32 * 1024:
            continue
        if BK % 8 != 0 or BM % (8 * WM) != 0 or BN % (8 * WN) != 0:
            continue
        TM = BM // (8 * WM)
        TN = BN // (8 * WN)
        if TM * TN > 16:
            continue
        if WM * WN > 16:
            continue
        tiles_m = (M + BM - 1) // BM
        tiles_n = (N + BN - 1) // BN
        total_tiles = tiles_m * tiles_n
        if total_tiles < 4 and BM * BN > 64 * 64:
            continue
        waste = (tiles_m * BM * tiles_n * BN) / max(1, M * N)
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
    """Pick ``(BM, BN, NSG)`` for cooperative-tensor GEMM."""
    mx = max(M, N)
    is_lp = dtype != torch.float32
    m_div_32 = M % 32 == 0
    m_div_64 = M % 64 == 0
    n_div_64 = N % 64 == 0
    n_div_128 = (N % 128 == 0)

    if M == 1 or N == 1:
        return (16, 128, 4)

    if K <= 256 and M >= 1024 and N >= 1024 and m_div_32 and n_div_128:
        return (32, 128, 4)

    if mx <= 256:
        return (32, 32, 4)

    if M <= 48 and N >= 1024:
        return (16, 64, 2) if M <= 16 else (32, 64, 2)

    if mx <= 1024:
        n64 = ((M + 63) // 64) * ((N + 63) // 64)
        if n64 < 120 or (K <= mx and m_div_64 and n_div_64):
            return (32, 64, 2)

    if K >= 2 * mx and mx >= 1792 and m_div_64 and n_div_128:
        if is_lp:
            if N < M and mx <= 2048:
                return (64, 64, 2)
            return (48, 128, 4)
        return (64, 128, 4)

    if dtype == torch.bfloat16 and mx >= 4096 and min(M, N) >= 1024 \
            and not (m_div_64 and n_div_64):
        return (128, 128, 8)

    if is_lp:
        return (64, 64, 2)

    if m_div_64 and n_div_128:
        return (64, 128, 4)
    return (64, 64, 2)


def _pick_mpp_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int, int, int, bool]:
    """Pick ``(BM, BN, BK, WM, WN, double_buffered)`` for MPP GEMM."""
    is_lp = dtype != torch.float32
    dtype_bytes = 2 if is_lp else 4
    ops = float(M) * N * K
    aspect = max(M, N) / max(1, min(M, N))

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

    if aspect >= 4 and max(M, N) >= 1024:
        if M > N:
            primary = (128, 64, 32, 4, 2, False)
        else:
            primary = (64, 128, 32, 2, 4, False)

    fallbacks = [
        primary,
        (128, 64, 32, 4, 2, True),
        (64, 128, 32, 2, 4, True),
        (128, 64, 64, 4, 2, False),
        (64, 64, 32, 2, 2, False),
        (64, 64, 16, 2, 2, False),
        (32, 64, 16, 1, 2, False),
        (32, 32, 16, 1, 1, False),
        (16, 32, 16, 1, 1, False),
    ]
    for cand in fallbacks:
        BM, BN, BK, WM, WN, dbuf = cand
        if BM % (16 * WM) != 0:
            continue
        if BN % (32 * WN) != 0:
            continue
        if BK % 16 != 0:
            continue
        mult = 2 if dbuf else 1
        if _threadgroup_bytes(BM, BN, BK, dtype_bytes) * mult > 32 * 1024:
            continue
        TM = BM // (16 * WM)
        TN = BN // (32 * WN)
        if TM * TN > 8:
            continue
        if WM * WN > 16:
            continue
        tiles_m = (M + BM - 1) // BM
        tiles_n = (N + BN - 1) // BN
        if tiles_m * tiles_n == 0:
            continue
        if BM > 2 * M or BN > 2 * N:
            continue
        return (BM, BN, BK, WM, WN, dbuf)
    return (16, 32, 16, 1, 1, False)


def _round_swizzle_log(tiles_m: int, tiles_n: int) -> int:
    if tiles_m * tiles_n < 32:
        return 0
    return 2


def _swizzled_grid(tiles_m, tiles_n, group, log):
    factor = 1 << log
    return (group * tiles_n * factor, (tiles_m + factor - 1) // factor, 1)


def _resolve_inputs(a: torch.Tensor, b: torch.Tensor):
    """Return (A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b) describing
    how the kernel reads the original memory, avoiding materialized copies."""
    assert a.dim() == 2 and b.dim() == 2, "matmul currently expects 2-D inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"

    sa, sb = a.stride(), b.stride()
    if sa[1] == 1 and sa[0] >= K:
        A_view, trans_a, lda = a, False, sa[0]
    elif sa[0] == 1 and sa[1] >= M:
        A_view, trans_a, lda = a, True, sa[1]
    else:
        A_view, trans_a, lda = a.contiguous(), False, K
    if sb[1] == 1 and sb[0] >= N:
        B_view, trans_b, ldb = b, False, sb[0]
    elif sb[0] == 1 and sb[1] >= K:
        B_view, trans_b, ldb = b, True, sb[1]
    else:
        B_view, trans_b, ldb = b.contiguous(), False, N
    return A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b


_CGEMV_T_NWARPS = 8
_CGEMV_NT_NWARPS = 4


# Complex and integer dispatch

def _complex_inputs(a, b):
    dtype = a.dtype if a.dtype in _COMPLEX else b.dtype
    a = a.to(dtype) if a.dtype != dtype else a
    b = b.to(dtype) if b.dtype != dtype else b
    return (a.resolve_conj() if a.is_conj() else a,
            b.resolve_conj() if b.is_conj() else b, dtype)


def _run_complex_gemv(a, b, out=None, epilogue=None):
    M, K = a.shape
    N = b.shape[1]
    t = (M == 1 and N >= 1 and a.is_contiguous()
         and b.is_contiguous() and b.stride(0) == N)
    nt = (N == 1 and M >= 1 and a.is_contiguous()
          and b.is_contiguous() and a.stride(0) == K)
    if not (t or nt):
        return None
    fused = epilogue is not None
    c2, real_t = _COMPLEX[a.dtype]
    if fused:
        bias, br, bc, beta, alpha, bnz, anz = epilogue
    if t:
        fn = kernels.cgemv_t(c2, "float2", real_t, 32, _CGEMV_T_NWARPS,
                             epilogue=fused, beta_nz=bnz if fused else True,
                             alpha_nz=anz if fused else True)[0]
        out = out if out is not None else a.new_empty(1, N)
        args = (b, a.view(-1), out.view(-1), _pk(N, K, N, 1))
        size, step, nw = N, bc if fused else 0, _CGEMV_T_NWARPS
    else:
        fn = kernels.cgemv_nt(c2, "float2", real_t, _CGEMV_NT_NWARPS,
                              epilogue=fused, beta_nz=bnz if fused else True,
                              alpha_nz=anz if fused else True)[0]
        out = out if out is not None else a.new_empty(M, 1)
        args = (a, b.view(-1), out.view(-1), _pk(M, K, K, 1))
        size, step, nw = M, br if fused else 0, _CGEMV_NT_NWARPS
    if fused:
        args += (bias, int(step), float(beta.real), float(beta.imag),
                 float(alpha.real), float(alpha.imag))
    groups = (size + (31 if t else nw - 1)) // (32 if t else nw)
    fn(*args, threads=(nw * 32 * groups, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _complex_products(a, b, product):
    """Split two complex tensors and form the four real matrix products."""
    c2, real_t = _COMPLEX[a.dtype]
    split, _ = kernels.complex_pack(c2, real_t)
    parts = []
    for z in (a, b):
        real = torch.empty(z.shape, dtype=_COMPLEX_REAL[a.dtype], device=z.device)
        imag = torch.empty_like(real)
        n = z.numel()
        split(z, real, imag, n, threads=(n, 1, 1), group_size=(256, 1, 1))
        parts.extend((real, imag))
    ar, ai, br, bi = parts
    return (product(ar, br), product(ai, bi), product(ar, bi), product(ai, br)), c2, real_t


def _complex_matmul(a: torch.Tensor, b: torch.Tensor,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """Complex C = A @ B: native interleaved-complex GEMV for M==1 / N==1, else
    four real products (ar@br - ai@bi) + i(ar@bi + ai@br) on the tuned backend."""
    a, b, cdt = _complex_inputs(a, b)
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dim() == 2 and b.dim() == 2, "complex matmul currently expects 2-D inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"
    if out is not None:
        assert out.shape == (M, N) and out.dtype == cdt and out.device.type == "mps", \
            f"out must be ({M}, {N}) {cdt} on mps"
    gemv = _run_complex_gemv(a, b, out)
    if gemv is not None:
        return gemv

    products, _, _ = _complex_products(a.contiguous(), b.contiguous(), matmul)
    if out is None:
        out = torch.empty(M, N, dtype=cdt, device=a.device)
    nC = M * N
    combine = kernels.complex_pack(*_COMPLEX[cdt])[1]
    combine(*products, out, nC, threads=(nC, 1, 1), group_size=(256, 1, 1))
    return out


_INT_PROFILE = {
    torch.int8:  ("char",  "int",  "char"),
    torch.uint8: ("uchar", "uint", "uchar"),
    torch.int16: ("short", "int",  "short"),
    torch.int32: ("int",   "int",  "int"),
    torch.int64: ("long",  "long", "long"),
}
_INT_BYTES = {torch.int8: 1, torch.uint8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8}

_IGEMV_T_CFG = {
    torch.int8: (8, 8), torch.uint8: (8, 8), torch.int16: (4, 8),
    torch.int32: (2, 8), torch.int64: (1, 8),
}
_IGEMV_NT_CFG = {
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

    if M == 1 and N >= 16:
        if not trans_b:
            xv = A_view.reshape(-1)
            return _run_gemv_t(dtype, B_view, xv, xv.stride(0), out, N, K, ldb)
        xv = A_view.reshape(-1).contiguous()
        return _run_gemv_nt(dtype, B_view, xv, out, N, K, ldb)
    elif N == 1 and M >= 16:
        if trans_a:
            xv = B_view.reshape(-1)
            return _run_gemv_t(dtype, A_view, xv, xv.stride(0), out, M, K, lda)
        xv = B_view.reshape(-1).contiguous()
        return _run_gemv_nt(dtype, A_view, xv, out, M, K, lda)

    in_t, acc_t, out_t = _INT_PROFILE[dtype]
    BM, BN, BK, TX, TY = _pick_int_tile(int(M), int(N), int(K), dtype)
    fn, _ = kernels.int_gemm(in_t, acc_t, out_t, BM, BN, BK, TX, TY, trans_a, trans_b)
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    grp = TX * TY
    fn(A_view, B_view, out, _pk(M, N, K, lda, ldb, ldc),
       threads=(grp * tiles_n, tiles_m, 1), group_size=(grp, 1, 1))
    return out


def _matmul_nd(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Full torch.matmul for non-2-D operands: 1-D dot/promotion and batched
    broadcasting, built on the 2-D `matmul` and 3-D `bmm` kernels."""
    assert a.ndim and b.ndim, "matmul operands must be at least 1-D"
    a_vec, b_vec = a.ndim == 1, b.ndim == 1
    if a_vec:
        a = a.unsqueeze(0)
    if b_vec:
        b = b.unsqueeze(-1)

    M, K = a.shape[-2], a.shape[-1]
    K2, N = b.shape[-2], b.shape[-1]
    assert K == K2, f"shape mismatch: A is {a.shape}, B is {b.shape}"
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    if not batch:
        r = matmul(a, b)
    else:
        B = prod(batch)
        a3 = a.expand(*batch, M, K).reshape(B, M, K)
        b3 = b.expand(*batch, K, N).reshape(B, K, N)
        r = bmm(a3, b3).reshape(*batch, M, N)
    if a_vec:
        r = r.squeeze(-2)
    if b_vec:
        r = r.squeeze(-1)
    return out.copy_(r) if out is not None else r


# Public matmul

def matmul(a: torch.Tensor, b: torch.Tensor, *,
           backend: str | None = None,
           tile: tuple | None = None,
           swizzle_log: int | None = None,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``a @ b`` on MPS.

    ``backend``, ``tile``, and ``swizzle_log`` only apply to 2-D real inputs.
    """
    if a.ndim != 2 or b.ndim != 2:
        return _matmul_nd(a, b, out=out)

    if a.is_mps and (a.dtype in _COMPLEX or b.dtype in _COMPLEX):
        return _complex_matmul(a, b, out=out)

    if a.is_mps and a.dtype in _INT_PROFILE:
        return _int_matmul(a, b, out=out)

    fast_path = backend is None and out is None and swizzle_log is None and a.is_mps
    M, K = a.shape
    K2, N = b.shape
    dtype = a.dtype
    if fast_path and dtype in _PROFILE and K == K2:
        packed = a.is_contiguous() and b.is_contiguous()
        if packed and M == 1 and N >= 16:
            gt, thr, grp, dims = _gemv_plan(dtype, N, K)
            o = _pooled_out(b, 1, N)
            gt(b, a, o, dims, threads=thr, group_size=grp)
            return o
        if packed and N == 1 and M >= 16:
            fn, grp, k_hi, k_lda = _gemv_nt_plan(dtype, K)
            o = _pooled_out(a, M, 1)
            n_groups = (M + 3) // 4
            fn(a, b.view(-1), o.view(-1), [M | k_hi, k_lda],
               threads=(grp[0] * n_groups, 1, 1), group_size=grp)
            return o
        m4 = kernels.has_metal4()
        if (m4 and packed and M >= 2 and K >= 64
                and (N >= 32 or _is_sgpipe_regime(M, N, K, dtype))):
            plan = _gemm_plan(dtype, M, N, K, a, b)
            o = _pooled_out(a, M, N)
            if type(plan) is tuple:
                fn, thr, grp = plan
                fn(a, b, o, _pk(M, N, K, K, N, N), threads=thr, group_size=grp)
            else:
                plan.run(a, b, o)
            return o
        if m4:
            lay = _thin_trans_layout(a, b, M, N, K, dtype)
            if lay is not None:
                trans_a, trans_b, lda, ldb = lay
                plan = _gemm_trans_plan(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb)
                o = _pooled_out(a, M, N)
                plan.run(a, b, o)
                return o

    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype
    dtype = a.dtype
    if dtype not in _PROFILE:
        raise NotImplementedError(f"dtype {dtype} not supported")

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)

    if out is None:
        out = _pooled_out(a, M, N)
    else:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps"
    ldc = out.stride(0)

    in_t, acc_t, out_t = _PROFILE[dtype]
    M_, N_, K_ = int(M), int(N), int(K)
    lda_, ldb_, ldc_ = int(lda), int(ldb), int(ldc)

    backend = backend or "auto"

    if backend == "auto":
        is_lp = dtype != torch.float32
        packed_ab = lda_ == K_ and ldb_ == N_
        m4 = kernels.has_metal4()
        if (m4 and M == 1 and N >= 4096 and K >= 256 and is_lp
                and not trans_a and not trans_b and packed_ab):
            backend = "mpp_tensor"
        elif M == 1 and N >= 16:
            backend = "gemv"
        elif N == 1 and M >= 16:
            backend = "gemv"
        elif not m4:
            backend = "simd"
        elif M >= 2 and N >= 32 and K >= 64:
            backend = "mpp_tensor"
        else:
            backend = "mpp"

    if backend == "gemv":
        return _dispatch_gemv(A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, dtype, out)

    if backend == "mpp_tensor":
        if tile is None:
            BM, BN, NSG = _pick_mpp_tensor_tile(M_, N_, K_, dtype)
        else:
            assert len(tile) >= 3
            BM, BN, NSG = tile[0], tile[1], tile[2]
        tiles_m = (M_ + BM - 1) // BM
        tiles_n = (N_ + BN - 1) // BN
        mn_aligned = (M_ % BM == 0) and (N_ % BN == 0)
        swz = swizzle_log if swizzle_log is not None else 0
        fn, _ = kernels.mpp_tensor_gemm(
            in_t, out_t, BM, BN, NSG, trans_a, trans_b,
            relaxed=True, swizzle_log=swz, mn_aligned=mn_aligned,
        )
        group_size = (NSG * 32, 1, 1)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=_swizzled_grid(tiles_m, tiles_n, group_size[0], swz),
           group_size=group_size)
        return out

    if backend in ("simd", "mpp"):
        pad = None
        if backend == "simd":
            BM, BN, BK, WM, WN = tile if tile else _pick_simd_tile(M_, N_, K_, dtype)
        else:
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
        k_aligned = K_ % BK == 0
        tiles_m = (M_ + BM - 1) // BM
        tiles_n = (N_ + BN - 1) // BN
        swz = swizzle_log if swizzle_log is not None else _round_swizzle_log(tiles_m, tiles_n)
        if backend == "simd":
            fn, _ = kernels.simd_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN,
                                      trans_a, trans_b, mn_aligned, k_aligned, swz)
        else:
            fn, _ = kernels.mpp_gemm(in_t, acc_t, out_t, BM, BN, BK, WM, WN,
                                     trans_a, trans_b, mn_aligned, k_aligned,
                                     relaxed=True, swizzle_log=swz, dbuf=dbuf, pad=pad)
        group_size = (WM * WN * 32, 1, 1)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=_swizzled_grid(tiles_m, tiles_n, group_size[0], swz),
           group_size=group_size)
        return out

    raise ValueError(f"unknown backend {backend}")


def _run_gemv_t(dtype, matrix, x, xs, out, cols, k, ld, epilogue=None):
    fused = epilogue is not None
    if dtype in _INT_PROFILE:
        profile = _INT_PROFILE[dtype]
        vec, nw = _IGEMV_T_CFG[dtype]
        vec = _int_clamp_vec(vec, ld, matrix.storage_offset())
        fn = None
    else:
        profile = _PROFILE[dtype]
        fn, tg, vec = _gemv_pick(_gemv_handles(dtype), cols,
                                 int(ld) | matrix.storage_offset(), dtype, k=int(k))
        nw = tg // 32
    if fused or fn is None:
        bnz, anz = epilogue[-2:] if fused else (True, True)
        fn = kernels.gemv_t(*profile, 32 * vec, nw, vec,
                            epilogue=fused, beta_nz=bnz, alpha_nz=anz)[0]
    dims = _pk(cols, k, ld, int(xs))
    ng = (cols + 32 * vec - 1) // (32 * vec)
    if fused:
        bias, step, beta, alpha, _, _ = epilogue
        fn(matrix, x, out.view(-1), dims, bias, int(step), beta, alpha,
           threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    else:
        fn(matrix, x, out.view(-1), dims,
           threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _run_gemv_nt(dtype, matrix, x, out, rows, k, ld, epilogue=None):
    fused = epilogue is not None
    if dtype in _INT_PROFILE:
        profile = _INT_PROFILE[dtype]
        vec, nw = _IGEMV_NT_CFG[dtype]
        vec = _int_clamp_vec(vec, ld, matrix.storage_offset())
        red_tg, fn = dtype is torch.int64, None
    else:
        profile, nw, red_tg = _PROFILE[dtype], 4, False
        vec = _gemv_nt_width(dtype is not torch.float32, int(k), int(ld), matrix.storage_offset())
        fn = None if fused else _gemv_nt_pick(
            _gemv_handles(dtype)["nt"], rows, k, ld, matrix.storage_offset())
    if fused or fn is None:
        bnz, anz = epilogue[-2:] if fused else (True, True)
        fn = kernels.gemv_nt(*profile, 1, nw, vec, red_tg=red_tg,
                             epilogue=fused, beta_nz=bnz, alpha_nz=anz)[0]
    dims = _pk(rows, k, ld)
    ng = (rows + nw - 1) // nw
    if fused:
        bias, step, beta, alpha, _, _ = epilogue
        fn(matrix, x, out.view(-1), dims, bias, int(step), beta, alpha,
           threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    else:
        fn(matrix, x, out.view(-1), dims,
           threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _dispatch_gemv(A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b, dtype, out):
    M, N, K = int(M), int(N), int(K)
    if M == 1:
        if trans_b:
            return _run_gemv_nt(dtype, B_view, A_view.reshape(-1).contiguous(), out, N, K, int(ldb))
        else:
            xv = A_view.reshape(-1)
            return _run_gemv_t(dtype, B_view, xv, xv.stride(0), out, N, K, int(ldb))
    elif N == 1:
        if trans_a:
            xv = B_view.reshape(-1)
            return _run_gemv_t(dtype, A_view, xv, xv.stride(0), out, M, K, int(lda))
        else:
            return _run_gemv_nt(dtype, A_view, B_view.reshape(-1).contiguous(), out, M, K, int(lda))
    else:
        raise ValueError("gemv backend requires M==1 or N==1")


# Fused matmul

def _addmm_real(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out):
    dtype = mat1.dtype
    is_int = dtype in _INT_PROFILE
    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(mat1, mat2)
    if out is None:
        out = _pooled_out(mat1, M, N)
    ldc = out.stride(0)
    bstride = _pk(int(br), int(bc))

    if M == 1 and N >= 16:
        if not trans_b:
            xv = A_view.reshape(-1)
            return _run_gemv_t(dtype, B_view, xv, xv.stride(0), out, N, K, ldb,
                               (x, bc, beta, alpha, bnz, anz))
        xv = A_view.reshape(-1).contiguous()
        return _run_gemv_nt(dtype, B_view, xv, out, N, K, ldb,
                            (x, bc, beta, alpha, bnz, anz))
    if N == 1 and M >= 16:
        if trans_a:
            xv = B_view.reshape(-1)
            return _run_gemv_t(dtype, A_view, xv, xv.stride(0), out, M, K, lda,
                               (x, br, beta, alpha, bnz, anz))
        xv = B_view.reshape(-1).contiguous()
        return _run_gemv_nt(dtype, A_view, xv, out, M, K, lda,
                            (x, br, beta, alpha, bnz, anz))

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
        _gemm_plan(dtype, M_, N_, K_, A_view, B_view)
        tile = _GEMM_TILE[(dtype, M_, N_, K_)]
        if tile[-1] == "bt":
            vec, nwarps = tile[0], tile[1]
            fn, _ = kernels.gemv_bt(in_t, acc_t, out_t, M_, 32 * vec, nwarps, vec,
                                    epilogue=True, beta_nz=bnz, alpha_nz=anz)
            ng = (N_ + 32 * vec - 1) // (32 * vec)
            fn(B_view, A_view, out, _pk(N_, K_, ldb_, lda_, ldc_), x, bstride,
               float(beta), float(alpha),
               threads=(nwarps * 32 * ng, 1, 1), group_size=(nwarps * 32, 1, 1))
            return out
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

    if (m4 and (trans_a or trans_b) and ldc_ == N_ and dtype is not torch.float32
            and 2 <= M_ <= _GEMV_BT_MAX_M and 16 <= N_ <= _GEMV_BT_TB_MAX_N and K_ >= 64):
        _gemm_trans_plan(dtype, M_, N_, K_, A_view, B_view, trans_a, trans_b, lda_, ldb_)
        tile = _GEMM_TRANS_TILE[(dtype, M_, N_, K_, trans_a, trans_b)]
        if len(tile) == 4 and tile[3] == "bt_tr":
            vec, nwarps, ncols = tile[0], tile[1], tile[2]
            fn, _ = kernels.gemv_bt(in_t, acc_t, out_t, M_, 32 * vec, nwarps, vec,
                                    epilogue=True, beta_nz=bnz, alpha_nz=anz,
                                    trans_a=trans_a, trans_b=trans_b, NCOLS=ncols)
            ntg = ((N_ + nwarps * ncols - 1) // (nwarps * ncols) if trans_b
                   else (N_ + 32 * vec - 1) // (32 * vec))
            fn(B_view, A_view, out, _pk(N_, K_, ldb_, lda_, ldc_), x, bstride,
               float(beta), float(alpha),
               threads=(nwarps * 32 * ntg, 1, 1), group_size=(nwarps * 32, 1, 1))
        else:
            BM, BN, NSG = tile[1], tile[2], tile[3]
            fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, trans_a, trans_b, relaxed=True,
                                            swizzle_log=0, mn_aligned=(M_ % BM == 0) and (N_ % BN == 0),
                                            epilogue=True, beta_nz=bnz, alpha_nz=anz)
            grp = (NSG * 32, 1, 1)
            fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_), x, bstride,
               float(beta), float(alpha),
               threads=(grp[0] * ((N_ + BN - 1) // BN), (M_ + BM - 1) // BM, 1), group_size=grp)
        return out

    if m4:
        BM, BN, BK, WM, WN, dbuf = _pick_mpp_tile(M_, N_, K_, dtype)
        fn, _ = kernels.mpp_gemm(
            in_t, acc_t, out_t, BM, BN, BK, WM, WN, trans_a, trans_b,
            (M_ % BM == 0) and (N_ % BN == 0), K_ % BK == 0,
            relaxed=True, swizzle_log=0, dbuf=dbuf,
            epilogue=True, beta_nz=bnz, alpha_nz=anz,
        )
    else:
        BM, BN, BK, WM, WN = _pick_simd_tile(M_, N_, K_, dtype)
        fn, _ = kernels.simd_gemm(
            in_t, acc_t, out_t, BM, BN, BK, WM, WN, trans_a, trans_b,
            (M_ % BM == 0) and (N_ % BN == 0), K_ % BK == 0,
            swizzle_log=0, epilogue=True, beta_nz=bnz, alpha_nz=anz,
        )
    grp = (WM * WN * 32, 1, 1)
    fn(A_view, B_view, out, dims6, x, bstride, float(beta), float(alpha),
       threads=(grp[0] * ((N_ + BN - 1) // BN), (M_ + BM - 1) // BM, 1), group_size=grp)
    return out


def _addmm_complex(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out):
    cdt = mat1.dtype
    a = mat1.resolve_conj().contiguous() if mat1.is_conj() else mat1.contiguous()
    b = mat2.resolve_conj().contiguous() if mat2.is_conj() else mat2.contiguous()
    M, K = a.shape
    _, N = b.shape
    gemv = _run_complex_gemv(a, b, out, (x, br, bc, beta, alpha, bnz, anz))
    if gemv is not None:
        return gemv

    c2, r = _COMPLEX[cdt]
    products, _, _ = _complex_products(a, b, matmul)
    if out is None:
        out = torch.empty(M, N, dtype=cdt, device=a.device)
    _, combine = kernels.complex_pack(c2, r, epilogue=True, beta_nz=bnz, alpha_nz=anz)
    nC = M * N
    combine(*products, out, nC, x, _pk(int(N), int(br), int(bc), 0),
            float(beta.real), float(beta.imag), float(alpha.real), float(alpha.imag),
            threads=(nC, 1, 1), group_size=(256, 1, 1))
    return out


def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *,
          beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``beta * input + alpha * (mat1 @ mat2)`` with a fused epilogue."""
    assert mat1.is_mps and mat2.is_mps
    assert mat1.dim() == 2 and mat2.dim() == 2, "addmm expects 2-D mat1, mat2"
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2, f"shape mismatch: mat1 is {mat1.shape}, mat2 is {mat2.shape}"
    dtype = mat1.dtype
    assert mat2.dtype == dtype, "addmm requires mat1.dtype == mat2.dtype"
    if out is not None:
        assert out.shape == (M, N) and out.dtype == dtype and out.is_mps

    if dtype in _INT_PROFILE:
        beta, alpha = int(beta), int(alpha)
    elif dtype not in _COMPLEX:
        beta, alpha = float(beta), float(alpha)
    bnz, anz = beta != 0, alpha != 0

    x = input if input.dtype == dtype else input.to(dtype)
    xe = x.expand(M, N)
    br, bc = (int(s) for s in xe.stride())

    if dtype in _COMPLEX:
        return _addmm_complex(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out)
    return _addmm_real(mat1, mat2, x, br, bc, beta, alpha, bnz, anz, out)


# Batched matmul

def _pick_bmm_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """Pick ``(BM, BN, NSG)`` for batched cooperative-tensor GEMM."""
    if M == 1 or N == 1:
        return _pick_mpp_tensor_tile(M, N, K, dtype)
    mx, mn = max(M, N), min(M, N)
    if K <= 128 and mx >= 512:
        return (32, 128, 4)
    if mn <= 32:
        return (32, 32, 4)
    if mx <= 256:
        return (64, 64, 4)
    return (64, 64, 2)


_BMM_TILES = [(64, 64, 2), (64, 64, 4), (32, 64, 2), (32, 128, 4), (64, 128, 4),
              (128, 64, 4), (128, 32, 4), (128, 128, 8)]
_BMM_PLAN: dict = {}


def _bmm_candidates(M, N, K, dtype):
    primary = _pick_bmm_tile(M, N, K, dtype)
    return [primary] + [t for t in _BMM_TILES
                        if t != primary and t[0] <= 2 * M and t[1] <= 2 * N]


def _autotune_bmm(dtype, a, b, Bb, M, N, K, lda, ldb, trans_a, trans_b, sA, sB, cands, margin,
                  bt_specs=()):
    """Time each batched candidate (best-of-reps), returning the winning plan label
    ("mpp",fn,BM,BN,NSG) | ("bt",vec,nwarps). bt_specs add correctness-guarded gemv_bt."""
    in_t, acc_t, out_t = _PROFILE[dtype]
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
        cand.append((("mpp", fn, BM, BN, NSG), run))
    if bt_specs:
        ref = a.new_empty(Bb, M, N)
        cand[0][1]()
        ref.copy_(o)
        torch.mps.synchronize()
        scale = ref.abs().max().item() + 1e-6
        dims_bt, batch_bt = _pk(N, K, ldb, lda, ldc), _pk(sB, sA, sC, 0)
        kind = "bt_tb" if trans_b else "bt"
        for spec in bt_specs:
            if trans_b:
                vec, nwarps, ncols = spec
                ng = (N + nwarps * ncols - 1) // (nwarps * ncols)
            else:
                vec, nwarps = spec
                ncols = 1
                ng = (N + 32 * vec - 1) // (32 * vec)
            fnbt, _ = kernels.gemv_bt(in_t, acc_t, out_t, M, 32 * vec, nwarps, vec,
                                      batched=True, trans_b=trans_b, NCOLS=ncols)
            thrbt, grpbt = (nwarps * 32 * ng, 1, Bb), (nwarps * 32, 1, 1)
            runbt = (lambda fnbt=fnbt, thrbt=thrbt, grpbt=grpbt:
                     fnbt(b, a, o, dims_bt, batch_bt, threads=thrbt, group_size=grpbt))
            runbt()
            torch.mps.synchronize()
            if (o - ref).abs().max().item() <= 0.02 * scale:
                label = (kind, vec, nwarps, ncols) if trans_b else (kind, vec, nwarps)
                cand.append((label, runbt))
    return cand[_fastest([run for _, run in cand], warmup, iters, reps, margin)][0]


def _bmm_plan(dtype, M, N, K, lda, ldb, trans_a, trans_b, Bb, a=None, b=None, sA=0, sB=0):
    key = (dtype, M, N, K, trans_a, trans_b)
    plan = _BMM_PLAN.get(key)
    if plan is None:
        cands = _bmm_candidates(M, N, K, dtype)
        bt_specs = ()
        if (_AUTOTUNE and a is not None and not trans_a and dtype is not torch.float32
                and 2 <= M <= _GEMV_BT_MAX_M and N >= 16 and K >= 64):
            if not trans_b and N <= 8192:
                align = int(ldb) | int(sB) | int(b.storage_offset())
                bt_specs = _gemv_bt_specs(M, N, K, dtype, align)
            elif trans_b and N <= _GEMV_BT_TB_MAX_N_BMM:
                align = (int(ldb) | int(lda) | int(sB) | int(sA)
                         | int(a.storage_offset()) | int(b.storage_offset()))
                bt_specs = _gemv_bt_specs(M, N, K, dtype, align, trans_b=True)
        if _AUTOTUNE and a is not None and (len(cands) > 1 or bt_specs):
            plan = _autotune_bmm(dtype, a, b, Bb, M, N, K, lda, ldb,
                                 trans_a, trans_b, sA, sB, cands, _AUTOTUNE_MARGIN, bt_specs)
        else:
            BM, BN, NSG = cands[0]
            in_t, _, out_t = _PROFILE[dtype]
            mn = (M % BM == 0) and (N % BN == 0)
            fn, _ = kernels.mpp_tensor_gemm(in_t, out_t, BM, BN, NSG, trans_a, trans_b,
                                            relaxed=True, swizzle_log=0, mn_aligned=mn, batched=True)
            plan = ("mpp", fn, BM, BN, NSG)
        _BMM_PLAN[key] = plan
    return plan


def _resolve_bmm_inputs(a: torch.Tensor, b: torch.Tensor):
    """Resolve row- or column-major batched views, copying other layouts."""
    _, M, K = a.shape
    _, _, N = b.shape
    A_view, trans_a, lda = _matrix_view(a, M, K)
    B_view, trans_b, ldb = _matrix_view(b, K, N)
    return (A_view, B_view, int(M), int(N), int(K), int(lda), int(ldb),
            trans_a, trans_b, int(A_view.stride(0)), int(B_view.stride(0)))


def _bmm_out(ref: torch.Tensor, out, Bb, M, N, dtype):
    """Return a writable output and whether it must be copied back."""
    if out is not None:
        assert out.shape == (Bb, M, N) and out.dtype == dtype and out.is_mps, \
            f"out must be ({Bb}, {M}, {N}) {dtype} on mps"
        if out.stride(2) == 1 and out.stride(1) >= N:
            return out, False
        return ref.new_empty(Bb, M, N), True
    return ref.new_empty(Bb, M, N), False


def _bmm_loop(a, b, out=None):
    Bb, M = a.shape[0], a.shape[1]
    N = b.shape[2]
    if out is None:
        out = a.new_empty(Bb, M, N)
    for i in range(Bb):
        out[i].copy_(matmul(a[i], b[i]))
    return out


def _int_bmm(a, b, dtype, out=None, epilogue=None):
    Bb = a.shape[0]
    fused = epilogue is not None
    if fused:
        input, beta, alpha, bnz, anz = epilogue
    if a.shape[1] == 1 or b.shape[2] == 1:
        return (_baddbmm_loop(input, a, b, beta, alpha, out) if fused
                else _bmm_loop(a, b, out))
    in_t, acc_t, out_t = _INT_PROFILE[dtype]
    A, B, M, N, K, lda, ldb, ta, tb, sA, sB = _resolve_bmm_inputs(a, b)
    o, copy_back = _bmm_out(a, out, Bb, M, N, dtype)
    ldc, sC = int(o.stride(1)), int(o.stride(0))
    BM, BN, BK, TX, TY = _pick_int_tile(M, N, K, dtype)
    fn, _ = kernels.int_gemm(in_t, acc_t, out_t, BM, BN, BK, TX, TY, ta, tb,
                             epilogue=fused, beta_nz=bnz if fused else True,
                             alpha_nz=anz if fused else True, batched=True)
    grp = TX * TY
    threads = (grp * ((N + BN - 1) // BN), (M + BM - 1) // BM, Bb)
    args = (A, B, o, _pk(M, N, K, lda, ldb, ldc))
    if fused:
        x = input if input.dtype == dtype else input.to(dtype)
        sBias, br, bc = map(int, x.expand(Bb, M, N).stride())
        args += (x, _pk(br, bc), int(beta), int(alpha), _pk(sA, sB, sC, sBias))
    else:
        args += (_pk(sA, sB, sC, 0),)
    fn(*args, threads=threads, group_size=(grp, 1, 1))
    if copy_back:
        out.copy_(o)
        return out
    return o


def _complex_baddbmm(input, batch1, batch2, beta, alpha, bnz, anz, out=None):
    cdt = batch1.dtype
    prod = _complex_bmm(batch1, batch2)
    Bb, M, N = prod.shape
    x = input if input.dtype == cdt else input.to(cdt)
    if anz:
        r = alpha * prod
        if bnz:
            r = r + beta * x.expand(Bb, M, N)
    elif bnz:
        r = beta * x.expand(Bb, M, N).to(cdt)
    else:
        r = torch.zeros_like(prod)
    if out is None:
        return r.contiguous() if not r.is_contiguous() else r
    return out.copy_(r)


def _complex_bmm(a, b, out=None):
    a, b, cdt = _complex_inputs(a, b)
    a, b = a.contiguous(), b.contiguous()
    Bb, M, K = a.shape
    N = b.shape[2]
    products, c2, r = _complex_products(a, b, bmm)
    if out is None:
        out = torch.empty(Bb, M, N, dtype=cdt, device=a.device)
    nC = Bb * M * N
    combine = kernels.complex_pack(c2, r)[1]
    combine(*products, out, nC, threads=(nC, 1, 1), group_size=(256, 1, 1))
    return out


def _real_bmm(a, b, out=None, epilogue=None):
    """Launch the selected batched real kernel, optionally with a fused epilogue."""
    dtype, Bb = a.dtype, a.shape[0]
    A, B, M, N, K, lda, ldb, ta, tb, sA, sB = _resolve_bmm_inputs(a, b)
    o, copy_back = _bmm_out(a, out, Bb, M, N, dtype)
    ldc, sC = int(o.stride(1)), int(o.stride(0))
    plan = _bmm_plan(dtype, M, N, K, lda, ldb, ta, tb, Bb, A, B, sA, sB)
    in_t, acc_t, out_t = _PROFILE[dtype]
    fused = epilogue is not None
    if fused:
        bias, br, bc, sBias, beta, alpha, bnz, anz = epilogue

    if plan[0] in ("bt", "bt_tb"):
        tb = plan[0] == "bt_tb"
        vec, nwarps = plan[1:3]
        ncols = plan[3] if tb else 1
        fn, _ = kernels.gemv_bt(in_t, acc_t, out_t, M, 32 * vec, nwarps, vec,
                                epilogue=fused, beta_nz=bnz if fused else True,
                                alpha_nz=anz if fused else True, batched=True,
                                trans_b=tb, NCOLS=ncols)
        ng = ((N + nwarps * ncols - 1) // (nwarps * ncols) if tb
              else (N + 32 * vec - 1) // (32 * vec))
        rows = 1
        args = (B, A, o, _pk(N, K, ldb, lda, ldc))
        if fused:
            args += (bias, _pk(br, bc), float(beta), float(alpha), _pk(sB, sA, sC, sBias))
        else:
            args += (_pk(sB, sA, sC, 0),)
    else:
        _, fn, BM, BN, NSG = plan
        if fused:
            fn, _ = kernels.mpp_tensor_gemm(
                in_t, out_t, BM, BN, NSG, ta, tb, relaxed=True, swizzle_log=0,
                mn_aligned=(M % BM == 0) and (N % BN == 0), epilogue=True,
                beta_nz=bnz, alpha_nz=anz, batched=True,
            )
        ng, nwarps, rows = (N + BN - 1) // BN, NSG, (M + BM - 1) // BM
        args = (A, B, o, _pk(M, N, K, lda, ldb, ldc))
        if fused:
            args += (bias, _pk(br, bc), float(beta), float(alpha), _pk(sA, sB, sC, sBias))
        else:
            args += (_pk(sA, sB, sC, 0),)

    fn(*args, threads=(nwarps * 32 * ng, rows, Bb), group_size=(nwarps * 32, 1, 1))
    if copy_back:
        out.copy_(o)
        return out
    return o


def bmm(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute batched 3-D matrix multiplication."""
    assert a.is_mps and b.is_mps, "bmm expects mps tensors"
    assert a.dim() == 3 and b.dim() == 3, "bmm expects 3-D inputs"
    assert a.dtype == b.dtype, "bmm requires both operands the same dtype"
    Bb, M, K = a.shape
    Bb2, K2, N = b.shape
    assert Bb == Bb2 and K == K2, f"bmm shape mismatch: {a.shape} @ {b.shape}"
    dtype = a.dtype

    if Bb == 0 or M == 0 or N == 0:
        o = out if out is not None else a.new_empty(Bb, M, N)
        return o
    if K == 0:
        o = out if out is not None else a.new_empty(Bb, M, N)
        return o.zero_()
    if dtype in _COMPLEX:
        return _complex_bmm(a, b, out=out)
    if dtype in _INT_PROFILE:
        return _int_bmm(a, b, dtype, out)
    if dtype not in _PROFILE or not kernels.has_metal4():
        return _bmm_loop(a, b, out)

    return _real_bmm(a, b, out)


def _baddbmm_loop(input, batch1, batch2, beta, alpha, out=None):
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
    """Compute ``beta * input + alpha * (batch1 @ batch2)`` for 3-D batches."""
    assert batch1.is_mps and batch2.is_mps, "baddbmm expects mps tensors"
    assert batch1.dim() == 3 and batch2.dim() == 3, "baddbmm expects 3-D batch1, batch2"
    assert batch1.dtype == batch2.dtype, "baddbmm requires batch1.dtype == batch2.dtype"
    Bb, M, K = batch1.shape
    Bb2, K2, N = batch2.shape
    assert Bb == Bb2 and K == K2, f"baddbmm shape mismatch: {batch1.shape} @ {batch2.shape}"
    dtype = batch1.dtype

    if dtype in _INT_PROFILE:
        beta, alpha = int(beta), int(alpha)
    elif dtype not in _COMPLEX:
        beta, alpha = float(beta), float(alpha)
    bnz, anz = beta != 0, alpha != 0

    if dtype in _COMPLEX:
        return _complex_baddbmm(input, batch1, batch2, beta, alpha, bnz, anz, out)
    if dtype in _INT_PROFILE:
        return _int_bmm(batch1, batch2, dtype, out,
                        (input, beta, alpha, bnz, anz))
    if not kernels.has_metal4():
        return _baddbmm_loop(input, batch1, batch2, beta, alpha, out)

    x = input if input.dtype == dtype else input.to(dtype)
    xe = x.expand(Bb, M, N)
    sBias, br, bc = (int(s) for s in xe.stride())

    return _real_bmm(batch1, batch2, out,
                     (x, br, bc, sBias, beta, alpha, bnz, anz))


def addbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, *,
           beta=1, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``beta * input + alpha * sum(batch1 @ batch2)``."""
    assert batch1.is_mps and batch2.is_mps, "addbmm expects mps tensors"
    assert batch1.dim() == 3 and batch2.dim() == 3, "addbmm expects 3-D batch1, batch2"
    assert batch1.dtype == batch2.dtype, "addbmm requires batch1.dtype == batch2.dtype"
    Bb, M, K = batch1.shape
    Bb2, K2, N = batch2.shape
    assert Bb == Bb2 and K == K2, f"addbmm shape mismatch: {batch1.shape} @ {batch2.shape}"
    dtype = batch1.dtype

    if Bb == 0 or K == 0:
        x = input if input.dtype == dtype else input.to(dtype)
        b = int(beta) if dtype in _INT_PROFILE else beta
        r = (x * b).expand(M, N).contiguous() if b != 0 else x.new_zeros(M, N)
        if out is not None:
            assert out.shape == (M, N) and out.dtype == dtype and out.is_mps
            return out.copy_(r)
        return r

    A = batch1.permute(1, 0, 2).reshape(M, Bb * K)
    Bm = batch2.reshape(Bb * K, N)
    return addmm(input, A, Bm, beta=beta, alpha=alpha, out=out)


# Vector operations

def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1-D inner product, matching torch.dot (0-D scalar)."""
    assert a.dim() == 1 and b.dim() == 1, \
        f"1D tensors expected, but got {a.dim()}D and {b.dim()}D tensors"
    assert a.shape[0] == b.shape[0], \
        f"inconsistent tensor size, expected [{a.shape[0]}] and [{b.shape[0]}]"
    assert a.dtype == b.dtype, \
        f"dot : expected both vectors to have same dtype, got {a.dtype} and {b.dtype}"
    return matmul(a, b)


def vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1-D inner product conjugating the FIRST arg, matching torch.vdot."""
    if a.is_complex():
        return dot(a.conj().resolve_conj(), b)
    return dot(a, b)


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outer product a[i]*b[j] -> (M, N), matching torch.outer."""
    assert a.dim() == 1, f"outer: Expected 1-D argument self, but got {a.dim()}-D"
    assert b.dim() == 1, f"outer: Expected 1-D argument vec2, but got {b.dim()}-D"
    return a.unsqueeze(1) * b.unsqueeze(0)


def mv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Matrix-vector product (M,K)@(K,) -> (M,), matching torch.mv."""
    assert mat.dim() == 2 and vec.dim() == 1, \
        f"vector + matrix @ vector expected, got {mat.dim()}, {mat.dim()}, {vec.dim()}"
    assert mat.shape[1] == vec.shape[0], \
        f"size mismatch, got mat ({mat.shape[0]}x{mat.shape[1]}), vec ({vec.shape[0]})"
    return matmul(mat, vec)


gemm = matmul
ger = outer
