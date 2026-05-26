"""Benchmark metalblas.matmul vs torch.matmul on MPS."""
import os
import sys
import time
import argparse
from typing import List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metalblas


def bench(M, N, K, dtype, fn, iters=100, trials=10, warmup=50):
    torch.manual_seed(0)
    a = torch.randn(M, K, device='mps', dtype=dtype)
    b = torch.randn(K, N, device='mps', dtype=dtype)
    for _ in range(warmup):
        c = fn(a, b)
    torch.mps.synchronize()
    best = float('inf')
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            c = fn(a, b)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        sec = (t1 - t0) / iters
        if sec < best:
            best = sec
    flops = 2.0 * M * N * K
    tflops = flops / best / 1e12
    return best, tflops


def run(shapes: List[Tuple[int, int, int]], dtype: torch.dtype, *, label: str = "", cool: float = 0.0):
    print(f"\n=== {label}  dtype={dtype} ===")
    print(f"{'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | {'mb ms':>9} {'TFLOPS':>7} | speedup")
    print("-" * 80)
    speedups = []
    for (M, N, K) in shapes:
        if cool > 0:
            time.sleep(cool)
        torch_s, torch_t = bench(M, N, K, dtype, torch.matmul)
        try:
            if cool > 0:
                time.sleep(cool)
            mb_s, mb_t = bench(M, N, K, dtype, metalblas.matmul)
            ratio = torch_s / mb_s
            speedups.append(ratio)
            print(f"{M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | "
                  f"{mb_s*1e3:>9.3f} {mb_t:>7.2f} | {ratio:>5.2f}x")
        except Exception as e:
            print(f"{M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | FAILED: {e}")
    if speedups:
        import statistics
        print(f"Median speedup: {statistics.median(speedups):.2f}x  "
              f"Mean: {statistics.mean(speedups):.2f}x  "
              f"Min: {min(speedups):.2f}x  Max: {max(speedups):.2f}x")


SHAPES = {
    "square": [(s, s, s) for s in [128, 256, 512, 1024, 2048, 4096]],
    "tall": [(4096, 1024, 1024), (1024, 4096, 1024), (8192, 1024, 1024),
             (2048, 2048, 8192)],
    "attn": [(4096, 4096, 64), (4096, 4096, 128), (4096, 4096, 1024)],
    "gemv": [(1, 4096, 4096), (4096, 1, 4096), (1, 1024, 1024),
             (1, 32000, 4096)],
    "llm": [(2048, 14336, 4096), (4096, 4096, 11008), (4096, 11008, 4096)],
    "odd": [(257, 257, 257), (1023, 1023, 1023), (4097, 4097, 4097),
            (511, 511, 511), (333, 444, 555)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="all", choices=["fp32", "fp16", "bf16", "all"])
    p.add_argument("--group", default="all",
                   choices=list(SHAPES.keys()) + ["all"])
    p.add_argument("--cool", type=float, default=0.0,
                   help="Seconds to sleep before each bench() call to let the GPU cool")
    args = p.parse_args()

    dtypes = []
    if args.dtype in ("fp32", "all"): dtypes.append(torch.float32)
    if args.dtype in ("fp16", "all"): dtypes.append(torch.float16)
    if args.dtype in ("bf16", "all"): dtypes.append(torch.bfloat16)

    groups = list(SHAPES.keys()) if args.group == "all" else [args.group]

    for dt in dtypes:
        for g in groups:
            run(SHAPES[g], dt, label=g, cool=args.cool)


if __name__ == "__main__":
    main()
