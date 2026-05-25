"""Correctness tests against torch.matmul on MPS.

Run with:  python tests/test_basic.py
"""
import os
import sys
import torch

# Allow running from project root or anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metalblas import matmul as mb_matmul


def check(M, N, K, dtype, backend="auto", tile=None, atol=None):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device='mps')
    b = torch.randn(K, N, dtype=dtype, device='mps')
    if atol is None:
        # Loose tolerances – m5 backend uses TF32-like relaxed precision for fp32.
        if dtype == torch.float32:
            atol = max(0.1, 5e-3 * K**0.5) if backend == "m5" else max(1e-3, 1e-5 * K)
        elif dtype == torch.bfloat16:
            # bf16 has only 7 mantissa bits → errors grow ~ K * 2^-7
            atol = max(5e-1, 3e-2 * K**0.5)
        else:
            atol = max(5e-2, 1e-2 * K**0.5)
    c = mb_matmul(a, b, backend=backend, tile=tile)
    ref = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
    err = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    rel = err / (ref.to(torch.float32).abs() + 1e-6)
    ok = err.max().item() <= atol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {dtype} {backend:5s} {M:5d}x{N:5d}x{K:5d} "
          f"max_err={err.max().item():.3e} mean_err={err.mean().item():.3e} "
          f"atol={atol:.3e}")
    return ok


def check_transposed(M, N, K, dtype, backend="auto"):
    """Test with transposed A and B (views into col-major memory)."""
    torch.manual_seed(0)
    if dtype == torch.float32:
        atol = max(0.1, 5e-3 * K**0.5)
    elif dtype == torch.bfloat16:
        atol = max(5e-1, 3e-2 * K**0.5)
    else:
        atol = max(5e-2, 1e-2 * K**0.5)
    # A is M x K but stored as K x M with .T view
    a_raw = torch.randn(K, M, dtype=dtype, device='mps')
    a = a_raw.t()  # M x K view, stride (1, M)
    # B is K x N but stored as N x K with .T view
    b_raw = torch.randn(N, K, dtype=dtype, device='mps')
    b = b_raw.t()  # K x N view, stride (1, K)
    c = mb_matmul(a, b)
    ref = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
    err = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    ok = err.max().item() <= atol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {dtype} trans-AB {M:5d}x{N:5d}x{K:5d} "
          f"max_err={err.max().item():.3e} mean_err={err.mean().item():.3e}")
    return ok


def main():
    print("=== fp32, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256), (33, 33, 33)]:
        check(*shape, dtype=torch.float32, backend="simd")
    print("=== fp16, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.float16, backend="simd")
    print("=== fp32, m5 backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.float32, backend="m5")
    print("=== fp16, m5 backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.float16, backend="m5")
    print("=== bf16, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.bfloat16, backend="simd")
    print("=== bf16, m5 backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.bfloat16, backend="m5")
    print("=== Transposed inputs ===")
    for shape in [(128, 128, 128), (513, 257, 129), (1024, 1024, 256)]:
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            check_transposed(*shape, dtype=dt)
    print("=== GEMV ===")
    for shape in [(1, 4096, 4096), (4096, 1, 4096), (1, 1024, 1024), (1024, 1, 1024)]:
        check(*shape, dtype=torch.float32, backend="gemv")
        check(*shape, dtype=torch.float16, backend="gemv")
        check(*shape, dtype=torch.bfloat16, backend="gemv")
    print("=== GEMV transposed ===")
    for shape in [(1, 4096, 4096), (4096, 1, 4096)]:
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            check_transposed(*shape, dtype=dt)


if __name__ == "__main__":
    main()
