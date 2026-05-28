"""Validate and benchmark the autotuned simd_gemm fallback (the macOS<26 path).

Forces has_metal4()->False so metalblas.matmul takes the simd path regardless of
the host OS. Columns: static = heuristic tile, single buffer; auto = autotuned
tile+dbuf; torch = Apple MPS; gain = auto/static; vs-MPS = auto/torch.

vs-MPS is only meaningful on GPUs without a tensor unit (M4 and earlier), where
torch.matmul is also simdgroup-class. On M5+ torch uses the tensor unit, so vs-MPS
is not a fair comparison there.
"""
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import metalblas
from metalblas import kernels, dispatch

# Pretend matmul2d / Metal 4 does not exist -> take the Sequoia simd path.
kernels.has_metal4 = lambda: False


def params(M, N, K):
    f = M * N * K
    if f < 2e9:
        return 50, 20, 50
    if f < 5e10:
        return 10, 10, 30
    return 4, 8, 10


SHAPES = [
    (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
    (4096, 1024, 1024), (8192, 1024, 1024), (2048, 2048, 8192),
    (2048, 14336, 4096), (4096, 11008, 4096),
    (1023, 1023, 1023), (4097, 4097, 4097),
]


def static_call(a, b):
    # Original behaviour: heuristic tile, single buffer (dbuf off).
    t = dispatch._pick_simd_tile(a.shape[0], b.shape[1], a.shape[1], a.dtype)
    return metalblas.matmul(a, b, backend="simd", tile=(*t, 0))


def _best(fn, a, b, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(a, b)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_pair(M, N, K, iters, trials, warmup):
    """Time static / auto / torch INTERLEAVED so thermal drift hits all equally."""
    a = torch.randn(M, K, device="mps", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="mps", dtype=torch.bfloat16)
    ref = (a.float() @ b.float())
    err = (metalblas.matmul(a, b).float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    for _ in range(warmup):
        static_call(a, b)
        metalblas.matmul(a, b)
        torch.matmul(a, b)
    torch.mps.synchronize()
    bs = ba = bt = float("inf")
    for _ in range(trials):
        bs = min(bs, _best(static_call, a, b, iters))
        ba = min(ba, _best(metalblas.matmul, a, b, iters))
        bt = min(bt, _best(torch.matmul, a, b, iters))
    f = 2.0 * M * N * K
    return f / bs / 1e12, f / ba / 1e12, f / bt / 1e12, err


def main():
    print(f"{'M':>5}{'N':>6}{'K':>6} | {'static':>7} {'auto':>7} {'torch':>7} (TFLOPS)"
          f" | {'gain':>5} {'vs-MPS':>6} | tile picked")
    print("-" * 92)
    torch.manual_seed(0)
    ok = True
    for (M, N, K) in SHAPES:
        it, tr, wu = params(M, N, K)
        st, at, tt, err = bench_pair(M, N, K, it, tr, wu)
        if err > 2e-2:
            ok = False
        tile = dispatch._SIMD_TILE.get((torch.bfloat16, M, N, K))
        tag = "" if err <= 2e-2 else f" !ERR={err:.1e}"
        print(f"{M:>5}{N:>6}{K:>6} | {st:>7.2f} {at:>7.2f} {tt:>7.2f}         "
              f" | {at/st:>4.2f}x {at/tt:>5.2f}x | {tile}{tag}")
    print("\ncorrectness:", "ALL OK" if ok else "FAILURES")


if __name__ == "__main__":
    main()
