"""Correctness tests against torch.matmul on MPS.

Run with:  python tests/test_basic.py
"""
import os
import sys
import torch

# Allow running from project root or anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metalblas import matmul as mb_matmul
from metalblas.kernels import has_metal4


def check(M, N, K, dtype, backend="auto", tile=None, atol=None):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device='mps')
    b = torch.randn(K, N, dtype=dtype, device='mps')
    if atol is None:
        # Loose tolerances - m5 backend uses TF32-like relaxed precision for fp32.
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


def check_gemv_strided_vec(M, N, K, dtype):
    """Regression for the strided-GEMV-vector bug: the vector operand is a
    non-contiguous sub-view (a sliced column/row off a wider buffer). The kernel
    reads the vector as unit-stride, so _dispatch_gemv must contiguify it while
    leaving the matrix strided. Compares against a reference on the SAME views."""
    torch.manual_seed(0)
    if dtype == torch.float32:
        atol = max(0.1, 5e-3 * K**0.5)
    elif dtype == torch.bfloat16:
        atol = max(5e-1, 3e-2 * K**0.5)
    else:
        atol = max(5e-2, 1e-2 * K**0.5)
    if N == 1:                       # matrix @ strided column vector
        a = torch.randn(M, K, dtype=dtype, device='mps')
        b = torch.randn(K, 2, dtype=dtype, device='mps')[:, :1]   # (K,1) stride (2,1)
        tag = "vecB"
    else:                            # strided row vector @ matrix  (M == 1)
        a = torch.randn(K, 2, dtype=dtype, device='mps')[:, :1].t()  # (1,K) stride (1,2)
        b = torch.randn(K, N, dtype=dtype, device='mps')
        tag = "vecA"
    assert not (a.is_contiguous() and b.is_contiguous()), "vector should be strided"
    c = mb_matmul(a, b, backend="gemv")
    ref = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
    err = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    ok = err.max().item() <= atol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {dtype} gemv-{tag} {M:5d}x{N:5d}x{K:5d} "
          f"max_err={err.max().item():.3e} atol={atol:.3e}")
    return ok


def check_complex(M, N, K, dtype, layout="rm", rtol=None):
    """
    Complex matmul vs a full-precision CPU reference (relative max-error).

    complex64 GEMM rides the TF32-relaxed fp32 backend (rel ~1e-3); the native
    complex GEMV path accumulates in fp32 (rel ~1e-5)
    """
    torch.manual_seed(0)
    if rtol is None:
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
    a = torch.randn(M, K, dtype=dtype, device='mps')
    b = torch.randn(K, N, dtype=dtype, device='mps')
    if layout == "tr":            # transposed (col-major) A view
        a = torch.randn(K, M, dtype=dtype, device='mps').t()
    elif layout == "conj":        # lazy conjugate view (must be resolved)
        a = a.conj()
    c = mb_matmul(a, b)
    hp = torch.complex64
    ref = (a.cpu().to(hp) @ b.cpu().to(hp))
    err = (c.cpu().to(hp) - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    ok = rel <= rtol and c.dtype == dtype
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:9s} {layout:4s} {M:5d}x{N:5d}x{K:5d} "
          f"max_err={err:.3e} rel={rel:.3e} rtol={rtol:.1e}")
    return ok


def main():
    print("=== fp32, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256), (33, 33, 33)]:
        check(*shape, dtype=torch.float32, backend="simd")
    print("=== fp16, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.float16, backend="simd")
    # m5 / m5_tensor backends need Metal 4 cooperative-tensor headers (macOS 26+).
    m4 = has_metal4()
    if m4:
        print("=== fp32, m5 backend ===")
        for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
            check(*shape, dtype=torch.float32, backend="m5")
        print("=== fp16, m5 backend ===")
        for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
            check(*shape, dtype=torch.float16, backend="m5")
    else:
        print("=== m5 backend: SKIPPED (Metal 4 / macOS 26+ not available) ===")
    print("=== bf16, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.bfloat16, backend="simd")
    if m4:
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
    print("=== GEMV strided vector ===")
    for shape in [(1, 1024, 256), (1, 4096, 1024), (64, 1, 256), (1024, 1, 1024)]:
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            check_gemv_strided_vec(*shape, dtype=dt)

    print("=== complex64 GEMM ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512),
                  (1024, 1024, 1024), (513, 257, 129), (333, 444, 555), (2, 64, 128)]:
        check_complex(*shape, dtype=torch.complex64)
    print("=== complex64 GEMV (M==1 / N==1) ===")
    for shape in [(1, 4096, 4096), (4096, 1, 4096), (1, 1024, 1024),
                  (1024, 1, 1024), (1, 1, 512), (1, 17, 33)]:
        check_complex(*shape, dtype=torch.complex64)
    print("=== complex64 transposed / conj views ===")
    for shape in [(256, 256, 256), (513, 257, 129)]:
        check_complex(*shape, dtype=torch.complex64, layout="tr")
        check_complex(*shape, dtype=torch.complex64, layout="conj")
    print("=== complex32 (chalf) ===")
    for shape in [(256, 256, 256), (512, 512, 512), (1, 2048, 2048), (2048, 1, 2048)]:
        check_complex(*shape, dtype=torch.complex32)


if __name__ == "__main__":
    main()
