"""Correctness tests against torch.matmul on MPS.

Run with:  python tests/test_basic.py
"""
import os
import sys
import torch

# Allow running from project root or anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metalblas
from metalblas import matmul as mb_matmul
from metalblas.kernels import has_metal4


def check(M, N, K, dtype, backend="auto", tile=None, atol=None):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device='mps')
    b = torch.randn(K, N, dtype=dtype, device='mps')
    if atol is None:
        # Loose tolerances - mpp backend uses TF32-like relaxed precision for fp32.
        if dtype == torch.float32:
            atol = max(0.1, 5e-3 * K**0.5) if backend == "mpp" else max(1e-3, 1e-5 * K)
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


_INT_BITS = {torch.int8: 8, torch.uint8: 8, torch.int16: 16, torch.int32: 32}


def _int_ref(a, b, dtype):
    """Exact integer reference: accumulate in int64 on CPU, then truncate to the
    output width (two's-complement wrap) - matches torch's integer matmul exactly."""
    r = a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)
    if dtype == torch.int64:
        return r                       # int64 already wrapped mod 2^64 on CPU
    mod = 1 << _INT_BITS[dtype]
    r = r % mod
    if dtype != torch.uint8:           # signed: fold high half to negative
        r = torch.where(r >= (mod >> 1), r - mod, r)
    return r.to(dtype)


def _int_rand(*shape, dtype, lim=40):
    if dtype == torch.uint8:
        return torch.randint(0, 2 * lim, shape, device='mps', dtype=dtype)
    info = torch.iinfo(dtype)
    return torch.randint(max(info.min, -lim), min(info.max, lim), shape, device='mps', dtype=dtype)


def check_int(M, N, K, dtype, layout="rm"):
    """Integer matmul must be BIT-EXACT vs torch (no precision tradeoff: ACC>=output
    width + truncate is identical to torch's wrap-on-overflow)."""
    torch.manual_seed(0)
    a = _int_rand(M, K, dtype=dtype)
    b = _int_rand(K, N, dtype=dtype)
    if layout == "tr":            # transposed (col-major) A view
        a = _int_rand(K, M, dtype=dtype).t()
    c = mb_matmul(a, b)
    ref = _int_ref(a, b, dtype)
    ok = torch.equal(c.cpu(), ref) and c.dtype == dtype
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:6s} {layout:3s} {M:5d}x{N:5d}x{K:5d} "
          f"{'bit-exact' if ok else 'MISMATCH'}")
    return ok


def _bias(bshape, dtype):
    if dtype in (torch.complex64, torch.complex32) or dtype.is_floating_point:
        return torch.randn(bshape, dtype=dtype, device='mps')
    return _int_rand(*bshape, dtype=dtype) if bshape else _int_rand(1, dtype=dtype).reshape(())


def check_addmm(M, N, K, dtype, bshape, beta=1, alpha=1):
    """metalblas.addmm vs torch.addmm. Bit-exact for ints; |alpha|-scaled atol for
    fp (the product rides the TF32-relaxed / bf16 backend); relative for complex."""
    torch.manual_seed(0)
    is_int = dtype in _INT_BITS or dtype == torch.int64
    if dtype.is_floating_point or dtype.is_complex:
        a = torch.randn(M, K, dtype=dtype, device='mps')
        b = torch.randn(K, N, dtype=dtype, device='mps')
    else:
        a = _int_rand(M, K, dtype=dtype)
        b = _int_rand(K, N, dtype=dtype)
    inp = _bias(bshape, dtype)
    got = metalblas.addmm(inp, a, b, beta=beta, alpha=alpha)
    if is_int:
        ref = torch.addmm(inp, a, b, beta=beta, alpha=alpha)
        ok = torch.equal(got.cpu(), ref.cpu()) and got.dtype == dtype
        metric = "bit-exact" if ok else "MISMATCH"
    elif dtype.is_complex:
        # torch's complex (esp. chalf) addmm on MPS is unreliable, so compare to a
        # high-precision CPU reference: beta*input + alpha*(A@B) in complex64.
        hp = torch.complex64
        prod = a.cpu().to(hp) @ b.cpu().to(hp)
        ref = (alpha * prod if alpha != 0 else torch.zeros_like(prod))
        if beta != 0:
            ref = ref + beta * inp.cpu().to(hp).expand(M, N)
        err = (got.cpu().to(hp) - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-9)
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
        ok = rel <= rtol and got.dtype == dtype
        metric = f"rel={rel:.2e}"
    else:
        ref = torch.addmm(inp, a, b, beta=beta, alpha=alpha)
        sc = abs(alpha) + abs(beta)
        base = (max(0.1, 5e-3 * K**0.5) if dtype == torch.float32
                else max(5e-1, 3e-2 * K**0.5) if dtype == torch.bfloat16
                else max(5e-2, 1e-2 * K**0.5))
        atol = sc * base
        err = (got.float() - ref.float()).abs().max().item()
        ok = err <= atol and got.dtype == dtype
        metric = f"err={err:.2e} atol={atol:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:9s} {str(bshape):8s} b={beta} a={alpha} "
          f"{M:4d}x{N:4d}x{K:4d} {metric}")
    return ok


def check_addmm_beta0_nan(M, N, K, dtype):
    """beta==0 must drop `input` entirely - a NaN/Inf bias cannot leak into C."""
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device='mps')
    b = torch.randn(K, N, dtype=dtype, device='mps')
    inp = torch.full((M, N), float('nan'), dtype=dtype, device='mps')
    got = metalblas.addmm(inp, a, b, beta=0, alpha=1)
    ok = not bool(torch.isnan(got).any().item())
    print(f"  [{'OK' if ok else 'FAIL'}] {str(dtype).split('.')[-1]:9s} beta=0 NaN-bias "
          f"{M}x{N}x{K} {'no NaN leaked' if ok else 'NaN LEAKED'}")
    return ok


def main():
    print("=== fp32, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256), (33, 33, 33)]:
        check(*shape, dtype=torch.float32, backend="simd")
    print("=== fp16, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.float16, backend="simd")
    # mpp / mpp_tensor backends need Metal 4 cooperative-tensor headers (macOS 26+).
    m4 = has_metal4()
    if m4:
        print("=== fp32, mpp backend ===")
        for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
            check(*shape, dtype=torch.float32, backend="mpp")
        print("=== fp16, mpp backend ===")
        for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
            check(*shape, dtype=torch.float16, backend="mpp")
    else:
        print("=== mpp backend: SKIPPED (Metal 4 / macOS 26+ not available) ===")
    print("=== bf16, simd backend ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
        check(*shape, dtype=torch.bfloat16, backend="simd")
    if m4:
        print("=== bf16, mpp backend ===")
        for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (513, 257, 129), (1024, 1024, 256)]:
            check(*shape, dtype=torch.bfloat16, backend="mpp")
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

    int_dtypes = [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]
    print("=== integer GEMM (bit-exact) ===")
    for dt in int_dtypes:
        for shape in [(64, 64, 64), (256, 256, 256), (513, 257, 129), (333, 444, 555),
                      (1024, 1024, 1024), (96, 4096, 512), (33, 33, 33)]:
            check_int(*shape, dtype=dt)
    print("=== integer GEMV (M==1 / N==1) ===")
    for dt in int_dtypes:
        for shape in [(1, 4096, 4096), (4096, 1, 4096), (1, 1024, 257), (300, 1, 1024)]:
            check_int(*shape, dtype=dt)
    print("=== integer transposed views ===")
    for dt in int_dtypes:
        check_int(256, 256, 256, dtype=dt, layout="tr")

    # addmm: C = beta*input + alpha*(A@B), matching torch.addmm.
    addmm_dtypes = [torch.float32, torch.float16, torch.bfloat16,
                    torch.complex64, torch.complex32] + int_dtypes
    print("=== addmm: bias broadcast shapes ===")
    for dt in addmm_dtypes:
        M, N, K = 128, 96, 256
        for bshape in [(M, N), (1, N), (M, 1), (N,), (1,), ()]:
            check_addmm(M, N, K, dt, bshape)
    print("=== addmm: unaligned edge tiles (mpp_tensor VALIDATE path) ===")
    # M%BM / N%BN != 0 routes interior tiles through the static store and the
    # final row/col strip through the dynamic per-element edge store; exercise both
    # with every bias broadcast so the epilogue index math is checked on partials.
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        for (M, N, K) in [(130, 100, 200), (257, 257, 257), (333, 444, 555)]:
            for bshape in [(M, N), (1, N), (M, 1), (N,), (1,), ()]:
                check_addmm(M, N, K, dt, bshape)
    print("=== addmm: beta/alpha scaling ===")
    # torch.addmm only accepts real beta/alpha (even for complex tensors).
    for dt in addmm_dtypes:
        for (beta, alpha) in [(2, 3), (0, 1), (1, 0), (0, 0)]:
            check_addmm(128, 96, 256, dt, (96,), beta=beta, alpha=alpha)
    print("=== addmm: GEMV-shaped (M==1 / N==1) ===")
    for dt in addmm_dtypes:
        check_addmm(1, 4096, 4096, dt, (4096,))
        check_addmm(4096, 1, 4096, dt, (1,))
    print("=== addmm: beta=0 drops NaN bias ===")
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        check_addmm_beta0_nan(128, 96, 256, dt)


if __name__ == "__main__":
    main()
