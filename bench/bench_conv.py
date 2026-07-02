import os
import sys
import time
import argparse
import statistics

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import metalblas


def _time_pair(fa, fb, args, kwargs, iters, trials, warmup):
    t_start = time.perf_counter()
    for _ in range(warmup):
        fa(*args, **kwargs)
        fb(*args, **kwargs)
    torch.mps.synchronize()
    best_a = best_b = float('inf')
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            fa(*args, **kwargs)
        torch.mps.synchronize()
        best_a = min(best_a, (time.perf_counter() - t0) / iters)
        t0 = time.perf_counter()
        for _ in range(iters):
            fb(*args, **kwargs)
        torch.mps.synchronize()
        best_b = min(best_b, (time.perf_counter() - t0) / iters)
    return best_a, best_b, time.perf_counter() - t_start


def _iters_for(flops):
    if flops < 2e9:
        return 100, 10, 50
    if flops < 5e10:
        return 20, 6, 10
    if flops < 1e12:
        return 5, 4, 3
    return 2, 3, 2


CONV2D_SHAPES = [
    ("rn50-l1",     2, 8, 64, 56, 56, 64, 3, 1, 1, 1, 1),
    ("rn50-l2",     2, 8, 128, 28, 28, 128, 3, 1, 1, 1, 1),
    ("rn50-l3",     2, 8, 256, 14, 14, 256, 3, 1, 1, 1, 1),
    ("rn50-l4",     2, 8, 512, 7, 7, 512, 3, 1, 1, 1, 1),
    ("rn-down2",    2, 8, 64, 56, 56, 128, 3, 2, 1, 1, 1),
    ("rn-down4",    2, 8, 256, 14, 14, 512, 3, 2, 1, 1, 1),
    ("rn-1x1-a",    2, 8, 64, 56, 56, 256, 1, 1, 0, 1, 1),
    ("rn-1x1-b",    2, 8, 256, 56, 56, 64, 1, 1, 0, 1, 1),
    ("rn-1x1-c",    2, 8, 2048, 7, 7, 512, 1, 1, 0, 1, 1),
    ("stem-7x7",    2, 8, 3, 224, 224, 64, 7, 2, 3, 1, 1),
    ("vgg-hires",   2, 1, 64, 224, 224, 64, 3, 1, 1, 1, 1),
    ("sd-mid",      2, 2, 320, 64, 64, 320, 3, 1, 1, 1, 1),
    ("sd-deep",     2, 2, 1280, 16, 16, 1280, 3, 1, 1, 1, 1),
    ("sd-in",       2, 2, 4, 64, 64, 320, 3, 1, 1, 1, 1),
    ("n1-l1",       2, 1, 64, 56, 56, 64, 3, 1, 1, 1, 1),
    ("n1-l3",       2, 1, 256, 14, 14, 256, 3, 1, 1, 1, 1),
    ("dilated-d2",  2, 4, 256, 33, 33, 256, 3, 1, 2, 2, 1),
    ("resnext-g32", 2, 8, 256, 14, 14, 256, 3, 1, 1, 1, 32),
    ("mbnet-dw",    2, 8, 144, 56, 56, 144, 3, 1, 1, 1, 144),
    ("mbnet-dw2",   2, 8, 384, 14, 14, 384, 3, 1, 1, 1, 384),
]

CONV2D_CL_SHAPES = [
    ("cl-rn50-l1",  2, 8, 64, 56, 56, 64, 3, 1, 1, 1, 1, True),
    ("cl-rn50-l3",  2, 8, 256, 14, 14, 256, 3, 1, 1, 1, 1, True),
    ("cl-sd-mid",   2, 2, 320, 64, 64, 320, 3, 1, 1, 1, 1, True),
    ("cl-1x1-b",    2, 8, 256, 56, 56, 64, 1, 1, 0, 1, 1, True),
    ("cl-stem",     2, 8, 3, 224, 224, 64, 7, 2, 3, 1, 1, True),
]

CONV1D_SHAPES = [
    ("whisper-c1",  1, 8, 80, 3000, 384, 3, 1, 1, 1, 1),
    ("whisper-c2",  1, 8, 384, 3000, 384, 3, 2, 1, 1, 1),
    ("wav-mid",     1, 4, 512, 4096, 512, 3, 1, 1, 1, 1),
    ("wav-k5",      1, 4, 1024, 8192, 1024, 5, 1, 2, 1, 1),
    ("short-seq",   1, 32, 256, 128, 256, 3, 1, 1, 1, 1),
    ("mamba-dw",    1, 8, 2048, 2048, 2048, 4, 1, 3, 1, 2048),
    ("tcn-d4",      1, 4, 256, 4096, 256, 3, 1, 4, 4, 1),
]

CONV3D_SHAPES = [
    ("vid-rn-l1",   3, 2, 64, 8, 56, 56, 64, 3, 1, 1, 1, 1),
    ("vid-rn-l3",   3, 2, 256, 4, 14, 14, 256, 3, 1, 1, 1, 1),
    ("vae3d",       3, 1, 128, 17, 64, 64, 128, 3, 1, 1, 1, 1),
    ("vid-down",    3, 2, 128, 8, 28, 28, 256, 3, 2, 1, 1, 1),
    ("vid-k133",    3, 2, 256, 8, 14, 14, 256, (1, 3, 3), 1, (0, 1, 1), 1, 1),
]

GROUPS = {
    "conv2d": CONV2D_SHAPES,
    "conv2d_cl": CONV2D_CL_SHAPES,
    "conv1d": CONV1D_SHAPES,
    "conv3d": CONV3D_SHAPES,
}


def _shape_str(shape):
    nd = shape[1]
    N, C = shape[2], shape[3]
    sp = shape[4:4 + nd]
    O, k, s, p, d, g = shape[4 + nd:10 + nd]
    cl = shape[10 + nd] if len(shape) > 10 + nd else False

    def fmt(v):
        return "x".join(str(x) for x in v) if isinstance(v, (tuple, list)) else str(v)

    dims = "x".join(str(v) for v in (N, C, *sp))
    parts = [f"k{fmt(k)}"]
    if (p if isinstance(p, int) else max(p)) != 0:
        parts.append(f"p{fmt(p)}")
    if (s if isinstance(s, int) else max(s)) != 1:
        parts.append(f"s{fmt(s)}")
    if (d if isinstance(d, int) else max(d)) != 1:
        parts.append(f"d{fmt(d)}")
    if g != 1:
        parts.append(f"g{g}")
    if cl:
        parts.append("CL")
    return f"{dims}->{O}", " ".join(parts)


def _mk(shape, dtype):
    label, nd = shape[0], shape[1]
    N, C = shape[2], shape[3]
    sp = shape[4:4 + nd]
    O, k, s, p, d, g = shape[4 + nd:10 + nd]
    cl = shape[10 + nd] if len(shape) > 10 + nd else False
    torch.manual_seed(0)
    x = torch.randn(N, C, *sp, dtype=dtype, device="mps")
    if cl:
        mf = torch.channels_last if nd == 2 else torch.channels_last_3d
        x = x.to(memory_format=mf)
    kt = (k,) * nd if isinstance(k, int) else k
    wt = torch.randn(O, C // g, *kt, dtype=dtype, device="mps") * 0.05
    bias = torch.randn(O, dtype=dtype, device="mps")
    ksp = dict(stride=s, padding=p, dilation=d, groups=g)
    tfn = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
    mfn = (metalblas.conv1d, metalblas.conv2d, metalblas.conv3d)[nd - 1]
    ospatial = []
    st = (s,) * nd if isinstance(s, int) else s
    pt = (p,) * nd if isinstance(p, int) else p
    dt = (d,) * nd if isinstance(d, int) else d
    for i in range(nd):
        ospatial.append((sp[i] + 2 * pt[i] - dt[i] * (kt[i] - 1) - 1) // st[i] + 1)
    flops = 2.0 * N * O * (C // g)
    for v in ospatial:
        flops *= v
    for v in kt:
        flops *= v
    return label, tfn, mfn, (x, wt, bias), ksp, flops


def _mem_bytes(shape, dtype):
    nd = shape[1]
    N, C = shape[2], shape[3]
    sp = shape[4:4 + nd]
    O, k, s, p, d, g = shape[4 + nd:10 + nd]
    st = (s,) * nd if isinstance(s, int) else s
    pt = (p,) * nd if isinstance(p, int) else p
    dt_ = (d,) * nd if isinstance(d, int) else d
    kt = (k,) * nd if isinstance(k, int) else k
    osp = [(sp[i] + 2 * pt[i] - dt_[i] * (kt[i] - 1) - 1) // st[i] + 1 for i in range(nd)]
    elt = 4 if dtype == torch.float32 else 2
    x = N * C
    y = N * O
    for i in range(nd):
        x *= sp[i]
        y *= osp[i]
    return (3 * x + 2 * y) * elt


def run(shapes, dtype, label="", cool=0.0, check=False, mem_cap=6e9):
    print(f"\n=== {label}  dtype={dtype} ===")
    print(f"{'N x C x spatial -> O':>20} {'params':<14} | {'torch ms':>9} {'TFLOPS':>7} | "
          f"{'mb ms':>9} {'TFLOPS':>7} | speedup")
    print("-" * 92)
    speedups, rows = [], []
    for shape in shapes:
        dims, params = _shape_str(shape)
        if _mem_bytes(shape, dtype) > mem_cap:
            print(f"{dims:>20} {params:<14} | skipped (exceeds memory cap)")
            continue
        name, tfn, mfn, args, ksp, flops = _mk(shape, dtype)
        iters, trials, warmup = _iters_for(flops)
        if check:
            ref = tfn(args[0].float().cpu(), args[1].float().cpu(),
                      args[2].float().cpu(), **ksp)
            got = mfn(*args, **ksp).float().cpu()
            rel = ((got - ref).abs() / ref.abs().clamp(min=1.0)).max().item()
            assert rel < 0.06, f"{name}: rel={rel}"
        try:
            t_t, t_m, busy = _time_pair(tfn, mfn, args, ksp, iters, trials, warmup)
            if cool > 0:
                time.sleep(cool * busy)
            ratio = t_t / t_m
            speedups.append(ratio)
            line = (f"{dims:>20} {params:<14} | {t_t*1e3:>9.3f} {flops/t_t/1e12:>7.2f} | "
                    f"{t_m*1e3:>9.3f} {flops/t_m/1e12:>7.2f} | {ratio:>5.2f}x")
        except Exception as e:
            line = f"{dims:>20} {params:<14} | FAILED: {e}"
        rows.append(line)
        print(line)
        del args
        if _mem_bytes(shape, dtype) > 1e9:
            torch.mps.empty_cache()
    if speedups:
        print(f"Median: {statistics.median(speedups):.2f}x  Mean: {statistics.mean(speedups):.2f}x  "
              f"Min: {min(speedups):.2f}x  Max: {max(speedups):.2f}x")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--group", default="all",
                    help="comma list of: conv2d, conv2d_cl, conv1d, conv3d")
    ap.add_argument("--cool", type=float, default=0.0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--batches", default=None,
                    help="comma list (e.g. 1,2,4,8,16,32): re-run every shape at "
                         "each batch size (memory-capped points are skipped)")
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    groups = list(GROUPS) if args.group == "all" else args.group.split(",")
    batches = [int(v) for v in args.batches.split(",")] if args.batches else None
    for gname in groups:
        shapes = GROUPS[gname]
        if batches:
            seen = set()
            shapes = [s for s in
                      ((s[0], s[1], b, *s[3:]) for s in GROUPS[gname] for b in batches)
                      if not (s[1:] in seen or seen.add(s[1:]))]
        run(shapes, dt, label=gname + (" batch-sweep" if batches else ""),
            cool=args.cool, check=args.check)
