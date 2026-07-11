"""Benchmark metalblas.bmm vs torch.bmm on MPS."""
import os
import sys
import argparse
import statistics
from typing import List, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import metalblas
from bench_matmul import _rand, _time_call


def bench(B, M, N, K, dtype, fn, iters=50, trials=8, warmup=20):
    torch.manual_seed(0)
    a = _rand(B, M, K, dtype=dtype)
    b = _rand(B, K, N, dtype=dtype)
    best, _ = _time_call(fn, a, b, iters, trials, warmup)
    flops = 2.0 * B * M * N * K * (4.0 if dtype.is_complex else 1.0)
    return best, flops / best / 1e12


def run(shapes: List[Tuple[int, int, int, int]], dtype, *, label="", cool=0.0):
    print(f"\n=== {label}  dtype={dtype} ===")
    print(f"{'B':>4} {'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | "
          f"{'mb ms':>9} {'TFLOPS':>7} | speedup")
    print("-" * 84)
    speedups = []
    for (B, M, N, K) in shapes:
        if cool > 0:
            time.sleep(cool)
        torch_s, torch_t = bench(B, M, N, K, dtype, torch.bmm)
        try:
            if cool > 0:
                time.sleep(cool)
            mb_s, mb_t = bench(B, M, N, K, dtype, metalblas.bmm)
            ratio = torch_s / mb_s
            speedups.append(ratio)
            print(f"{B:>4} {M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | "
                  f"{mb_s*1e3:>9.3f} {mb_t:>7.2f} | {ratio:>5.2f}x")
        except Exception as e:
            print(f"{B:>4} {M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | FAILED: {e}")
    if speedups:
        print(f"Median: {statistics.median(speedups):.2f}x  Mean: {statistics.mean(speedups):.2f}x  "
              f"Min: {min(speedups):.2f}x  Max: {max(speedups):.2f}x")


# (B, M, N, K)
SHAPES = {
    # attention: scores = Q@Kᵀ (B*H, S, S, D), output = A@V (B*H, S, D, S)
    "attn": [(96, 512, 512, 64), (96, 512, 64, 512), (192, 512, 512, 64), (192, 512, 64, 512),
             (32, 1024, 1024, 128), (32, 1024, 128, 1024), (48, 2048, 2048, 64)],
    # general square-ish batched
    "square": [(64, 128, 128, 128), (64, 256, 256, 256), (32, 512, 512, 512),
               (16, 1024, 1024, 1024), (8, 256, 256, 256), (256, 128, 128, 128)],
    # thin batched (small M or N per matrix)
    "thin": [(128, 64, 4096, 256), (128, 4096, 64, 256), (256, 32, 2048, 512),
             (64, 16, 4096, 512)],
    # many small (launch-overhead regime)
    "small": [(512, 64, 64, 64), (1024, 32, 32, 64), (2048, 64, 64, 32)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16", "all"])
    p.add_argument("--group", default="all", choices=list(SHAPES.keys()) + ["all"])
    p.add_argument("--cool", type=float, default=0.0)
    args = p.parse_args()
    dtypes = {"fp32": [torch.float32], "fp16": [torch.float16], "bf16": [torch.bfloat16],
              "all": [torch.bfloat16, torch.float16, torch.float32]}[args.dtype]
    groups = list(SHAPES.keys()) if args.group == "all" else [args.group]
    for dt in dtypes:
        for g in groups:
            run(SHAPES[g], dt, label=g, cool=args.cool)


if __name__ == "__main__":
    main()
