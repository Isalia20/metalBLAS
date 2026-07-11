"""Floating-point Metal kernel planning and dispatch."""
from __future__ import annotations

from functools import cache
import os
import sys
import time

import torch

from . import kernels
from .dispatch import (
    _bmm_out,
    _ceildiv,
    _dispatch_gemv,
    _pk,
    _pooled_out,
    _resolve_bmm_inputs,
    _resolve_inputs,
    _unit_lead,
)


# PyTorch dtype -> (Metal input, accumulator, output).
PROFILE = {
    torch.float32: ("float", "float", "float"),
    torch.float16: ("half", "float", "half"),
    torch.bfloat16: ("bfloat", "float", "bfloat"),
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

# (elements per lane, simdgroups per threadgroup)
_GEMV_T_VARIANTS = {
    torch.float32:  [(1, 16), (1, 32)],
    torch.float16:  [(1, 32), (2, 32), (2, 16), (4, 16), (4, 8), (8, 8), (8, 16)],
    torch.bfloat16: [(1, 32), (2, 32), (2, 16), (4, 16), (4, 8), (8, 8), (8, 16)],
}


@cache
def _gemv_handles(dtype):
    in_t, acc_t, out_t = PROFILE[dtype]
    gt = {(vec, nw): kernels.gemv_t(in_t, acc_t, out_t, 32 * vec, nw, vec)[0]
          for vec, nw in _GEMV_T_VARIANTS[dtype]}
    nt = {1: kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, 1)[0]}
    if not _HAS_TENSOR_UNIT or dtype is not torch.float32:
        nt.update({vec: kernels.gemv_nt(in_t, acc_t, out_t, 1, 4, vec)[0]
                   for vec in (2, 4)})
    return {"gt": gt, "nt": nt}


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


def _gemv_nt_pick(variants, k, ld, off=0):
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


@cache
def _gemv_nt_plan(dtype, K):
    """Cache the contiguous N=1 launch and the constant part of its dimensions."""
    fn = _gemv_nt_pick(_gemv_handles(dtype)["nt"], K, K)
    return fn, (128, 1, 1), K << 32, K


@cache
def _gemv_plan(dtype, N, K):
    gt, tg, vec = _gemv_pick(_gemv_handles(dtype), N, N, dtype, k=K)
    ng = _ceildiv(N, 32 * vec)
    return gt, (tg * ng, 1, 1), (tg, 1, 1), _pk(N, K, N, 1)


# GEMM planning and autotuning

_GEMM_PLAN: dict = {}

_AUTOTUNE = os.environ.get("METALBLAS_AUTOTUNE", "1") != "0"
_AUTOTUNE_MARGIN = 0.03
_TALL_NARROW_MARGIN = 0.01


def _mpp_tensor_plan(dtype, M, N, tile, trans_a=False, trans_b=False, *,
                     swizzle=0, batch=None, epilogue=None):
    """Build a cooperative-tensor kernel and its launch geometry."""
    BM, BN, NSG = tile
    in_t, _, out_t = PROFILE[dtype]
    fused = epilogue is not None
    bnz, anz = epilogue if fused else (True, True)
    fn = kernels.mpp_tensor_gemm(
        in_t, out_t, BM, BN, NSG, trans_a, trans_b, relaxed=True,
        swizzle_log=swizzle, mn_aligned=M % BM == 0 and N % BN == 0,
        epilogue=fused, beta_nz=bnz, alpha_nz=anz, batched=batch is not None,
    )[0]
    group = (NSG * 32, 1, 1)
    threads = _swizzled_grid(_ceildiv(M, BM), _ceildiv(N, BN), group[0], swizzle)
    return fn, (*threads[:2], batch if batch is not None else 1), group


def _build_mppt_plan(dtype, M, N, BM, BN, NSG):
    """Build a packed, untransposed cooperative-tensor launch."""
    return _mpp_tensor_plan(dtype, M, N, (BM, BN, NSG))


_SPLITK_POOL: dict = {}


def _splitk_partial(ref, planes, M, N):
    key = (M, N, planes)
    buf = _SPLITK_POOL.get(key)
    if buf is None:
        buf = ref.new_empty(planes, M, N, dtype=torch.float32)
        _SPLITK_POOL[key] = buf
    return buf


def _build_splitk_plan(dtype, M, N, K, BM, BN, NSG, G):
    in_t, _, out_t = PROFILE[dtype]
    splitk_fn, reduce_fn = kernels.splitk_gemm(in_t, out_t, BM, BN, NSG, K // G, relaxed=True)
    planes, n = G - 1, M * N
    sk_threads = (NSG * 32 * _ceildiv(N, BN), _ceildiv(M, BM), G)
    sk_group, sk_dims = (NSG * 32, 1, 1), _pk(M, N, K)
    red_dims = _pk(n, planes)

    def run(a, b, o):
        partial = _splitk_partial(o, planes, M, N)
        splitk_fn(a, b, o, partial, sk_dims, threads=sk_threads, group_size=sk_group)
        reduce_fn(partial, o, red_dims, threads=(n, 1, 1), group_size=(256, 1, 1))
    return run


def _is_splitk_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and K >= 2048
            and 64 <= min(M, N) and M * N <= 1_500_000
            and (min(M, N) <= 256 or K >= 8 * max(M, N)))


def _splitk_specs(K):
    return [(BM, BN, NSG, G)
            for BM, BN, NSG in ((128, 32, 2), (64, 64, 2), (32, 64, 2))
            for G in (2, 4) if K % G == 0 and (K // G) % 16 == 0]


def _launch(fn, threads, group, dims, swap=False):
    if swap:
        def run(a, b, o):
            fn(b, a, o, dims, threads=threads, group_size=group)
    else:
        def run(a, b, o):
            fn(a, b, o, dims, threads=threads, group_size=group)
    return run


def _build_conv_plan(dtype, M, N, K, BMW, BNO, NSG):
    in_t, _, out_t = PROFILE[dtype]
    fn, _ = kernels.conv1x1_gemm(in_t, out_t, BMW, BNO, NSG, K)
    threads = (NSG * 32 * _ceildiv(N, BNO), _ceildiv(M, BMW), 1)
    return _launch(fn, threads, (NSG * 32, 1, 1), _pk(M, N, K))


def _is_conv_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and N <= 64 and N % 32 == 0
            and M >= 512 and K >= 256)


def _conv_specs(M, N):
    return [(BMW, N, NSG) for BMW in (16, 32, 48, 64, 96, 128)
            if M % BMW == 0 for NSG in (2, 4)]


def _build_sgpipe_plan(dtype, M, N, K, SGM, SGN, KC, NSGX, NSGY, GK=0):
    in_t, _, out_t = PROFILE[dtype]
    fn, _ = kernels.sgpipe_gemm(in_t, out_t, SGM, SGN, KC, NSGX, NSGY, GK)
    nsg = NSGX * NSGY
    threads = (nsg * 32 * (N // (NSGX * SGN)), M // (NSGY * SGM), 1)
    return _launch(fn, threads, (nsg * 32, 1, 1), _pk(M, N, K))


def _build_flipt_plan(dtype, M, N, K, BM, BN, NSG, KC=0, PFD=0):
    in_t, _, out_t = PROFILE[dtype]
    fn, _ = kernels.flipt_gemm(in_t, out_t, BM, BN, NSG, KC, PFD)
    threads = (NSG * 32 * _ceildiv(M, BN), _ceildiv(N, BM), 1)
    return _launch(fn, threads, (NSG * 32, 1, 1), _pk(M, N, K))


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


def _gemv_bt_launch(dtype, M, N, K, spec, trans_a=False, trans_b=False, *,
                    lda=None, ldb=None, ldc=None, batch=None, epilogue=None):
    """Build a thin-M GEMM kernel and its launch metadata."""
    vec, nwarps, *rest = spec
    ncols = rest[0] if rest else 1
    in_t, acc_t, out_t = PROFILE[dtype]
    fused = epilogue is not None
    bnz, anz = epilogue if fused else (True, True)
    fn, _ = kernels.gemv_bt(in_t, acc_t, out_t, M, 32 * vec, nwarps, vec,
                            epilogue=fused, beta_nz=bnz, alpha_nz=anz,
                            batched=batch is not None, trans_a=trans_a,
                            trans_b=trans_b, NCOLS=ncols)
    ldb = ldb if ldb is not None else (K if trans_b else N)
    dims = _pk(N, K, ldb, lda if lda is not None else K, ldc if ldc is not None else N)
    block = nwarps * ncols if trans_b else 32 * vec
    group = (nwarps * 32, 1, 1)
    threads = (group[0] * _ceildiv(N, block), 1, batch if batch is not None else 1)
    return fn, threads, group, dims


def _build_gemv_bt_plan(dtype, M, N, K, vec, nwarps, trans_b=False, ldb=None,
                        ldx=None, ldy=None, ncols=1, trans_a=False):
    launch = _gemv_bt_launch(dtype, M, N, K, (vec, nwarps, ncols), trans_a, trans_b,
                             lda=ldx, ldb=ldb, ldc=ldy)
    return _launch(*launch, swap=True)


def _is_gemv_bt_regime(M, N, K, dtype):
    return (dtype is not torch.float32 and 2 <= M <= _GEMV_BT_MAX_M
            and 16 <= N <= 8192 and K >= 64)


def _largest_pow2_le(x):
    x = int(x)
    return 1 << (x.bit_length() - 1) if x >= 1 else 1


def _gemv_bt_specs(M, N, K, align, trans_b=False):
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
    return list(dict.fromkeys((primary, *extra)))


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
        fn, thr, grp = _build_mppt_plan(dtype, M, N, BM, BN, NSG)
        run = (lambda a, b, o, fn=fn, thr=thr, grp=grp, d=mppt_dims:
               fn(a, b, o, d, threads=thr, group_size=grp))
        candidates.append(((fn, thr, grp), run, (BM, BN, NSG)))

    if sk_specs or conv_specs or bt_specs or flip_specs or sgp_specs:
        ref = a.new_empty(M, N)
        candidates[0][1](a, b, ref)
        torch.mps.synchronize()
        scale = ref.abs().max().item() + 1e-6

        def accept(plan, label):
            plan(a, b, o)
            torch.mps.synchronize()
            if (o - ref).abs().max().item() <= 0.02 * scale:
                candidates.append((plan, plan, label))

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
    cached = _GEMM_PLAN.get(key)
    if cached is None:
        cands, margin = _mpp_tensor_tile_candidates(M, N, K, dtype)
        tune = _AUTOTUNE and a is not None
        sk_specs = _splitk_specs(K) if tune and _is_splitk_regime(M, N, K, dtype) else ()
        conv_specs = _conv_specs(M, N) if tune and _is_conv_regime(M, N, K, dtype) else ()
        flip_specs = _flipt_specs(M, N, K, a) if tune and _is_flipt_regime(M, N, K, dtype) else ()
        sgp_specs = _sgpipe_specs(M, N, K) if tune and _is_sgpipe_regime(M, N, K, dtype) else ()
        bt_specs = ()
        if tune and _is_gemv_bt_regime(M, N, K, dtype):
            align = int(b.stride(0)) | int(b.storage_offset())
            bt_specs = _gemv_bt_specs(M, N, K, align)
        if tune and (len(cands) > 1 or sk_specs or conv_specs or bt_specs or flip_specs or sgp_specs):
            plan, tile = _autotune_mppt(dtype, M, N, K, cands, a, b, margin,
                                        sk_specs, conv_specs, bt_specs, flip_specs, sgp_specs)
        else:
            tile = cands[0]
            plan = _build_mppt_plan(dtype, M, N, *tile)
        cached = _GEMM_PLAN[key] = plan, tile
    return cached


_GEMM_TRANS_PLAN: dict = {}


def _build_mpp_trans_plan(dtype, M, N, K, BM, BN, NSG, trans_a, trans_b, lda, ldb):
    fn, threads, group = _mpp_tensor_plan(dtype, M, N, (BM, BN, NSG), trans_a, trans_b)
    return _launch(fn, threads, group, _pk(M, N, K, lda, ldb, N))


def _autotune_trans(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb):
    BM, BN, NSG = _pick_mpp_tensor_tile(M, N, K, dtype)
    cand = [(_build_mpp_trans_plan(dtype, M, N, K, BM, BN, NSG, trans_a, trans_b, lda, ldb),
             ("mpp_tr", BM, BN, NSG))]
    align = lda | ldb | int(a.storage_offset()) | int(b.storage_offset())
    o, ref = a.new_empty(M, N), a.new_empty(M, N)
    cand[0][0](a, b, ref)
    torch.mps.synchronize()
    scale = ref.abs().max().item() + 1e-6
    t0 = time.perf_counter()
    for _ in range(3):
        cand[0][0](a, b, o)
    torch.mps.synchronize()
    est = (time.perf_counter() - t0) / 3
    iters = max(3, min(120, int(0.02 / max(est, 1e-7))))
    reps, warmup = 5, min(iters, 12)
    for spec in _gemv_bt_specs(M, N, K, align, trans_b=trans_b):
        vec, nwarps, ncols = spec if trans_b else (spec[0], spec[1], 1)
        plan = _build_gemv_bt_plan(dtype, M, N, K, vec, nwarps, trans_b=trans_b,
                                   ldb=ldb, ldx=lda, ldy=N, ncols=ncols, trans_a=trans_a)
        plan(a, b, o)
        torch.mps.synchronize()
        if (o - ref).abs().max().item() <= 0.02 * scale:
            cand.append((plan, (vec, nwarps, ncols, "bt_tr")))
    i = _fastest([lambda p=p: p(a, b, o) for p, _ in cand],
                 warmup, iters, reps, _AUTOTUNE_MARGIN)
    return cand[i]


def _gemm_trans_plan(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb):
    key = (dtype, M, N, K, trans_a, trans_b)
    cached = _GEMM_TRANS_PLAN.get(key)
    if cached is None:
        if _AUTOTUNE:
            plan, tile = _autotune_trans(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb)
        else:
            BM, BN, NSG = _pick_mpp_tensor_tile(M, N, K, dtype)
            plan = _build_mpp_trans_plan(dtype, M, N, K, BM, BN, NSG, trans_a, trans_b, lda, ldb)
            tile = ("mpp_tr", BM, BN, NSG)
        cached = _GEMM_TRANS_PLAN[key] = plan, tile
    return cached


_GEMV_BT_TB_MAX_N = 262144
_GEMV_BT_TB_MAX_N_BMM = 16384


# Layout and tile helpers

def _thin_trans_layout(a, b, M, N, K, dtype):
    if (dtype is torch.float32 or not (2 <= M <= _GEMV_BT_MAX_M)
            or not (16 <= N <= _GEMV_BT_TB_MAX_N) or K < 64):
        return None
    la, lb = _unit_lead(a, M, K), _unit_lead(b, K, N)
    if la is None or lb is None or not (la[0] or lb[0]):
        return None
    return (la[0], lb[0], la[1], lb[1])


def _threadgroup_bytes(BM, BN, BK, dtype_bytes):
    pad = max(1, 16 // dtype_bytes)
    lda = BK + pad
    ldb = BN + pad
    return (BM * lda + BK * ldb) * dtype_bytes


def _pick_simd_tile(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[int, int, int, int, int]:
    """Pick ``(BM, BN, BK, WM, WN)`` for the simdgroup kernel."""
    dtype_bytes = 4 if dtype == torch.float32 else 2
    candidates = [
        (128, 128, 16, 4, 4), (128, 128, 32, 4, 4),
        (64, 128, 16, 2, 4), (128, 64, 16, 4, 2),
        (64, 64, 32, 2, 2), (64, 64, 16, 2, 2),
        (32, 128, 16, 1, 4), (128, 32, 16, 4, 1),
        (32, 64, 16, 1, 2), (64, 32, 16, 2, 1),
        (32, 32, 32, 1, 1), (32, 32, 16, 1, 1),
        (16, 16, 16, 1, 1),
    ]
    best = None
    for (BM, BN, BK, WM, WN) in candidates:
        bytes_needed = _threadgroup_bytes(BM, BN, BK, dtype_bytes)
        if (bytes_needed > 32 * 1024 or BK % 8 or BM % (8 * WM) or BN % (8 * WN)
                or (BM // (8 * WM)) * (BN // (8 * WN)) > 16 or WM * WN > 16):
            continue
        tiles_m, tiles_n = _ceildiv(M, BM), _ceildiv(N, BN)
        total_tiles = tiles_m * tiles_n
        if total_tiles < 4 and BM * BN > 64 * 64:
            continue
        waste = (tiles_m * BM * tiles_n * BN) / max(1, M * N)
        ops = M * N * K
        score = ((BM * BN, -waste, BK) if ops > 256 * 1024**2 else
                 (-abs(BM * BN - max(M, N) * 16), -waste))
        if best is None or score > best[0]:
            best = (score, (BM, BN, BK, WM, WN))
    return best[1] if best is not None else (16, 16, 16, 1, 1)


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
        primary = ((128, 64, 32, 4, 2, False) if M > N else
                   (64, 128, 32, 2, 4, False))

    fallbacks = [
        primary, (128, 64, 32, 4, 2, True), (64, 128, 32, 2, 4, True),
        (128, 64, 64, 4, 2, False), (64, 64, 32, 2, 2, False),
        (64, 64, 16, 2, 2, False), (32, 64, 16, 1, 2, False),
        (32, 32, 16, 1, 1, False),
        (16, 32, 16, 1, 1, False),
    ]
    for cand in fallbacks:
        BM, BN, BK, WM, WN, dbuf = cand
        if BM % (16 * WM) or BN % (32 * WN) or BK % 16:
            continue
        mult = 2 if dbuf else 1
        tiles_m, tiles_n = _ceildiv(M, BM), _ceildiv(N, BN)
        if (_threadgroup_bytes(BM, BN, BK, dtype_bytes) * mult > 32 * 1024
                or (BM // (16 * WM)) * (BN // (32 * WN)) > 8 or WM * WN > 16
                or not tiles_m * tiles_n or BM > 2 * M or BN > 2 * N):
            continue
        return (BM, BN, BK, WM, WN, dbuf)
    return (16, 32, 16, 1, 1, False)


def _round_swizzle_log(tiles_m: int, tiles_n: int) -> int:
    return 0 if tiles_m * tiles_n < 32 else 2


def _swizzled_grid(tiles_m, tiles_n, group, log):
    factor = 1 << log
    return (group * tiles_n * factor, (tiles_m + factor - 1) // factor, 1)


def _classic_plan(backend, dtype, M, N, K, tile, trans_a, trans_b, swizzle,
                  epilogue=None):
    """Build an MPP or simdgroup GEMM kernel and its launch geometry."""
    BM, BN, BK, WM, WN = tile[:5]
    dbuf = tile[5] if len(tile) > 5 else False
    pad = tile[6] if len(tile) > 6 else None
    in_t, acc_t, out_t = PROFILE[dtype]
    fused = epilogue is not None
    bnz, anz = epilogue if fused else (True, True)
    common = (in_t, acc_t, out_t, BM, BN, BK, WM, WN, trans_a, trans_b,
              M % BM == 0 and N % BN == 0, K % BK == 0)
    if backend == "mpp":
        fn = kernels.mpp_gemm(*common, relaxed=True, swizzle_log=swizzle,
                              dbuf=dbuf, pad=pad, epilogue=fused,
                              beta_nz=bnz, alpha_nz=anz)[0]
    else:
        fn = kernels.simd_gemm(*common, swizzle_log=swizzle, epilogue=fused,
                               beta_nz=bnz, alpha_nz=anz)[0]
    group = (WM * WN * 32, 1, 1)
    threads = _swizzled_grid(_ceildiv(M, BM), _ceildiv(N, BN), group[0], swizzle)
    return fn, threads, group


# Public floating-point matmul

def matmul(a, b, *, backend=None, tile=None, swizzle_log=None, out=None):
    """Compute a 2-D floating-point matrix product."""
    fast_path = backend is None and out is None and swizzle_log is None and a.is_mps
    M, K = a.shape
    K2, N = b.shape
    dtype = a.dtype
    if fast_path and dtype in PROFILE and K == K2:
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
            plan, _ = _gemm_plan(dtype, M, N, K, a, b)
            o = _pooled_out(a, M, N)
            if type(plan) is tuple:
                fn, thr, grp = plan
                fn(a, b, o, _pk(M, N, K, K, N, N), threads=thr, group_size=grp)
            else:
                plan(a, b, o)
            return o
        if m4:
            lay = _thin_trans_layout(a, b, M, N, K, dtype)
            if lay is not None:
                trans_a, trans_b, lda, ldb = lay
                plan, _ = _gemm_trans_plan(dtype, M, N, K, a, b, trans_a, trans_b, lda, ldb)
                o = _pooled_out(a, M, N)
                plan(a, b, o)
                return o

    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == b.dtype
    dtype = a.dtype
    if dtype not in PROFILE:
        raise NotImplementedError(f"dtype {dtype} not supported")

    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(a, b)

    if out is None:
        out = _pooled_out(a, M, N)
    else:
        assert out.shape == (M, N) and out.dtype == dtype and out.device.type == "mps"
    ldc = out.stride(0)

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
        return _dispatch_gemv(_run_gemv_t, _run_gemv_nt, A_view, B_view,
                              M, N, K, lda, ldb, trans_a, trans_b, dtype, out)

    if backend == "mpp_tensor":
        if tile is not None:
            assert len(tile) >= 3
        chosen = _pick_mpp_tensor_tile(M_, N_, K_, dtype) if tile is None else tile[:3]
        swz = swizzle_log if swizzle_log is not None else 0
        fn, threads, group = _mpp_tensor_plan(dtype, M_, N_, chosen, trans_a, trans_b,
                                               swizzle=swz)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=threads, group_size=group)
        return out

    if backend in ("simd", "mpp"):
        if tile is None:
            chosen = (_pick_simd_tile(M_, N_, K_, dtype) if backend == "simd"
                      else _pick_mpp_tile(M_, N_, K_, dtype))
        else:
            chosen = tile
        BM, BN = chosen[:2]
        tiles_m, tiles_n = _ceildiv(M_, BM), _ceildiv(N_, BN)
        swz = swizzle_log if swizzle_log is not None else _round_swizzle_log(tiles_m, tiles_n)
        fn, threads, group = _classic_plan(backend, dtype, M_, N_, K_, chosen,
                                            trans_a, trans_b, swz)
        fn(A_view, B_view, out, _pk(M_, N_, K_, lda_, ldb_, ldc_),
           threads=threads, group_size=group)
        return out

    raise ValueError(f"unknown backend {backend}")


def _run_gemv_t(dtype, matrix, x, xs, out, cols, k, ld, epilogue=None):
    fused = epilogue is not None
    profile = PROFILE[dtype]
    fn, tg, vec = _gemv_pick(_gemv_handles(dtype), cols,
                             int(ld) | matrix.storage_offset(), dtype, k=int(k))
    nw = tg // 32
    if fused:
        bnz, anz = epilogue[-2:]
        fn = kernels.gemv_t(*profile, 32 * vec, nw, vec,
                            epilogue=fused, beta_nz=bnz, alpha_nz=anz)[0]
    dims = _pk(cols, k, ld, int(xs))
    ng = _ceildiv(cols, 32 * vec)
    args = (matrix, x, out.view(-1), dims)
    if fused:
        bias, step, beta, alpha, _, _ = epilogue
        args += (bias, int(step), beta, alpha)
    fn(*args, threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


def _run_gemv_nt(dtype, matrix, x, out, rows, k, ld, epilogue=None):
    fused = epilogue is not None
    profile, nw = PROFILE[dtype], 4
    vec = _gemv_nt_width(dtype is not torch.float32, int(k), int(ld), matrix.storage_offset())
    fn = None if fused else _gemv_nt_pick(
        _gemv_handles(dtype)["nt"], k, ld, matrix.storage_offset())
    if fused:
        bnz, anz = epilogue[-2:]
        fn = kernels.gemv_nt(*profile, 1, nw, vec, red_tg=False,
                             epilogue=fused, beta_nz=bnz, alpha_nz=anz)[0]
    dims = _pk(rows, k, ld)
    ng = _ceildiv(rows, nw)
    args = (matrix, x, out.view(-1), dims)
    if fused:
        bias, step, beta, alpha, _, _ = epilogue
        args += (bias, int(step), beta, alpha)
    fn(*args, threads=(nw * 32 * ng, 1, 1), group_size=(nw * 32, 1, 1))
    return out


# Fused matmul

def addmm(mat1, mat2, out, epilogue):
    x, br, bc, beta, alpha, bnz, anz = epilogue
    dtype = mat1.dtype
    A_view, B_view, M, N, K, lda, ldb, trans_a, trans_b = _resolve_inputs(mat1, mat2)
    if out is None:
        out = _pooled_out(mat1, M, N)
    ldc = out.stride(0)

    if (M == 1 and N >= 16) or (N == 1 and M >= 16):
        return _dispatch_gemv(
            _run_gemv_t, _run_gemv_nt, A_view, B_view, M, N, K, lda, ldb,
            trans_a, trans_b, dtype, out, (x, br, bc, beta, alpha, bnz, anz))

    M_, N_, K_, lda_, ldb_, ldc_ = int(M), int(N), int(K), int(lda), int(ldb), int(ldc)
    dims6 = _pk(M_, N_, K_, lda_, ldb_, ldc_)
    bstride = _pk(br, bc)

    packed_ab = (lda_ == K_ and ldb_ == N_)
    m4 = kernels.has_metal4()
    choice = None
    if m4 and M_ >= 2 and N_ >= 32 and K_ >= 64 and not trans_a and not trans_b and packed_ab:
        _, tile = _gemm_plan(dtype, M_, N_, K_, A_view, B_view)
        if tile[-1] == "bt":
            choice = "bt", tile[:2]
        else:
            choice = "mpp", (tile if len(tile) == 3 else
                              _pick_mpp_tensor_tile(M_, N_, K_, dtype))
    elif (m4 and (trans_a or trans_b) and ldc_ == N_ and dtype is not torch.float32
          and 2 <= M_ <= _GEMV_BT_MAX_M and 16 <= N_ <= _GEMV_BT_TB_MAX_N and K_ >= 64):
        _, tile = _gemm_trans_plan(dtype, M_, N_, K_, A_view, B_view,
                                   trans_a, trans_b, lda_, ldb_)
        if len(tile) == 4 and tile[3] == "bt_tr":
            choice = "bt", tile[:3]
        else:
            choice = "mpp", tile[1:]

    if choice is not None:
        kind, spec = choice
        if kind == "bt":
            fn, threads, group, dims = _gemv_bt_launch(
                dtype, M_, N_, K_, spec, trans_a, trans_b, lda=lda_, ldb=ldb_,
                ldc=ldc_, epilogue=(bnz, anz),
            )
            args = (B_view, A_view, out, dims)
        else:
            fn, threads, group = _mpp_tensor_plan(
                dtype, M_, N_, spec, trans_a, trans_b, epilogue=(bnz, anz))
            args = (A_view, B_view, out, dims6)
        fn(*args, x, bstride, float(beta), float(alpha),
           threads=threads, group_size=group)
        return out

    backend = "mpp" if m4 else "simd"
    tile = (_pick_mpp_tile(M_, N_, K_, dtype) if m4
            else _pick_simd_tile(M_, N_, K_, dtype))
    fn, threads, grp = _classic_plan(backend, dtype, M_, N_, K_, tile,
                                      trans_a, trans_b, 0, (bnz, anz))
    fn(A_view, B_view, out, dims6, x, bstride, float(beta), float(alpha),
       threads=threads, group_size=grp)
    return out


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
    warmup, iters, reps = _probe_params(Bb * M, N, K)
    o = a.new_empty(Bb, M, N)
    sC, ldc = int(o.stride(0)), int(o.stride(1))
    dims, bstr = _pk(M, N, K, lda, ldb, ldc), _pk(sA, sB, sC, 0)
    cand = []
    for (BM, BN, NSG) in cands:
        fn, thr, grp = _mpp_tensor_plan(dtype, M, N, (BM, BN, NSG),
                                         trans_a, trans_b, batch=Bb)
        run = (lambda fn=fn, thr=thr, grp=grp: fn(a, b, o, dims, bstr, threads=thr, group_size=grp))
        cand.append((("mpp", fn, BM, BN, NSG), run))
    if bt_specs:
        ref = a.new_empty(Bb, M, N)
        cand[0][1]()
        ref.copy_(o)
        torch.mps.synchronize()
        scale = ref.abs().max().item() + 1e-6
        batch_bt = _pk(sB, sA, sC, 0)
        kind = "bt_tb" if trans_b else "bt"
        for spec in bt_specs:
            fnbt, thrbt, grpbt, dims_bt = _gemv_bt_launch(
                dtype, M, N, K, spec, trans_b=trans_b, lda=lda, ldb=ldb,
                ldc=ldc, batch=Bb,
            )
            runbt = (lambda fnbt=fnbt, thrbt=thrbt, grpbt=grpbt:
                     fnbt(b, a, o, dims_bt, batch_bt, threads=thrbt, group_size=grpbt))
            runbt()
            torch.mps.synchronize()
            if (o - ref).abs().max().item() <= 0.02 * scale:
                cand.append(((kind, *spec), runbt))
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
                bt_specs = _gemv_bt_specs(M, N, K, align)
            elif trans_b and N <= _GEMV_BT_TB_MAX_N_BMM:
                align = (int(ldb) | int(lda) | int(sB) | int(sA)
                         | int(a.storage_offset()) | int(b.storage_offset()))
                bt_specs = _gemv_bt_specs(M, N, K, align, trans_b=True)
        if _AUTOTUNE and a is not None and (len(cands) > 1 or bt_specs):
            plan = _autotune_bmm(dtype, a, b, Bb, M, N, K, lda, ldb,
                                 trans_a, trans_b, sA, sB, cands, _AUTOTUNE_MARGIN, bt_specs)
        else:
            BM, BN, NSG = cands[0]
            fn = _mpp_tensor_plan(dtype, M, N, (BM, BN, NSG), trans_a, trans_b,
                                  batch=Bb)[0]
            plan = ("mpp", fn, BM, BN, NSG)
        _BMM_PLAN[key] = plan
    return plan


def bmm(a, b, out=None, epilogue=None):
    """Launch the selected batched real kernel, optionally with a fused epilogue."""
    dtype, Bb = a.dtype, a.shape[0]
    A, B, M, N, K, lda, ldb, ta, tb, sA, sB = _resolve_bmm_inputs(a, b)
    o, copy_back = _bmm_out(a, out, Bb, M, N, dtype)
    ldc, sC = int(o.stride(1)), int(o.stride(0))
    plan = _bmm_plan(dtype, M, N, K, lda, ldb, ta, tb, Bb, A, B, sA, sB)
    fused = epilogue is not None
    if fused:
        bias, br, bc, sBias, beta, alpha, bnz, anz = epilogue

    if plan[0] in ("bt", "bt_tb"):
        tb = plan[0] == "bt_tb"
        fn, threads, group, dims = _gemv_bt_launch(
            dtype, M, N, K, plan[1:], trans_b=tb, lda=lda, ldb=ldb, ldc=ldc,
            batch=Bb, epilogue=(bnz, anz) if fused else None,
        )
        args = (B, A, o, dims)
        if fused:
            args += (bias, _pk(br, bc), float(beta), float(alpha), _pk(sB, sA, sC, sBias))
        else:
            args += (_pk(sB, sA, sC, 0),)
    else:
        _, fn, BM, BN, NSG = plan
        if fused:
            fn, threads, group = _mpp_tensor_plan(
                dtype, M, N, (BM, BN, NSG), ta, tb, batch=Bb,
                epilogue=(bnz, anz),
            )
        else:
            group = (NSG * 32, 1, 1)
            threads = (group[0] * _ceildiv(N, BN), _ceildiv(M, BM), Bb)
        args = (A, B, o, _pk(M, N, K, lda, ldb, ldc))
        if fused:
            args += (bias, _pk(br, bc), float(beta), float(alpha), _pk(sA, sB, sC, sBias))
        else:
            args += (_pk(sA, sB, sC, 0),)

    fn(*args, threads=threads, group_size=group)
    if copy_back:
        out.copy_(o)
        return out
    return o
