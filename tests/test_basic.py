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


def check_bmm(B, M, N, K, dtype, layout="rm"):
    """metalblas.bmm vs torch.bmm (3-D batched). Bit-exact for ints; atol for fp;
    relative for complex. layout 'tr' makes both operands col-major (transposed view)."""
    torch.manual_seed(0)
    is_int = dtype in _INT_BITS or dtype == torch.int64
    rand = (lambda *s: _int_rand(*s, dtype=dtype)) if is_int else \
           (lambda *s: torch.randn(*s, dtype=dtype, device='mps'))
    if layout == "tr":            # col-major matrices (e.g. attention Q@Kᵀ)
        a = rand(B, K, M).transpose(-2, -1)
        b = rand(B, N, K).transpose(-2, -1)
    else:
        a = rand(B, M, K)
        b = rand(B, K, N)
    got = metalblas.bmm(a, b)
    if is_int:
        ref = _int_ref(a, b, dtype)        # int64 @ then wrap (== torch's overflow)
        ok = torch.equal(got.cpu(), ref) and got.dtype == dtype
        metric = "bit-exact" if ok else "MISMATCH"
    elif dtype.is_complex:
        hp = torch.complex64
        ref = a.cpu().to(hp) @ b.cpu().to(hp)
        err = (got.cpu().to(hp) - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-9)
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
        ok = rel <= rtol and got.dtype == dtype
        metric = f"rel={rel:.2e}"
    else:
        ref = torch.bmm(a.float(), b.float()).to(dtype)
        atol = (max(0.1, 5e-3 * K**0.5) if dtype == torch.float32
                else max(5e-1, 3e-2 * K**0.5) if dtype == torch.bfloat16
                else max(5e-2, 1e-2 * K**0.5))
        err = (got.float() - ref.float()).abs().max().item()
        ok = err <= atol and got.dtype == dtype
        metric = f"err={err:.2e} atol={atol:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:9s} {layout:3s} B={B:4d} {M:4d}x{N:4d}x{K:4d} {metric}")
    return ok


def _baddbmm_int_ref(inp, a, b, beta, alpha, dtype):
    """Exact int baddbmm: beta*input + alpha*(a@b) in int64, then wrap to output width."""
    prod = a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)
    Bb, M, N = prod.shape
    r = alpha * prod + beta * inp.cpu().to(torch.int64).expand(Bb, M, N)
    if dtype == torch.int64:
        return r
    mod = 1 << _INT_BITS[dtype]
    r = r % mod
    if dtype != torch.uint8:
        r = torch.where(r >= (mod >> 1), r - mod, r)
    return r.to(dtype)


def check_baddbmm(B, M, N, K, dtype, bshape, beta=1, alpha=1):
    """metalblas.baddbmm vs torch.baddbmm: C = beta*input + alpha*(b1 @ b2)."""
    torch.manual_seed(0)
    is_int = dtype in _INT_BITS or dtype == torch.int64
    if dtype.is_floating_point or dtype.is_complex:
        a = torch.randn(B, M, K, dtype=dtype, device='mps')
        b = torch.randn(B, K, N, dtype=dtype, device='mps')
    else:
        a = _int_rand(B, M, K, dtype=dtype)
        b = _int_rand(B, K, N, dtype=dtype)
    inp = _bias(bshape, dtype)
    got = metalblas.baddbmm(inp, a, b, beta=beta, alpha=alpha)
    if is_int:
        ref = _baddbmm_int_ref(inp, a, b, beta, alpha, dtype)
        ok = torch.equal(got.cpu(), ref) and got.dtype == dtype
        metric = "bit-exact" if ok else "MISMATCH"
    elif dtype.is_complex:
        hp = torch.complex64
        prod = a.cpu().to(hp) @ b.cpu().to(hp)
        ref = (alpha * prod if alpha != 0 else torch.zeros_like(prod))
        if beta != 0:
            ref = ref + beta * inp.cpu().to(hp).expand(B, M, N)
        err = (got.cpu().to(hp) - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-9)
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
        ok = rel <= rtol and got.dtype == dtype
        metric = f"rel={rel:.2e}"
    else:
        ref = torch.baddbmm(inp.float(), a.float(), b.float(), beta=beta, alpha=alpha).to(dtype)
        sc = abs(alpha) + abs(beta)
        base = (max(0.1, 5e-3 * K**0.5) if dtype == torch.float32
                else max(5e-1, 3e-2 * K**0.5) if dtype == torch.bfloat16
                else max(5e-2, 1e-2 * K**0.5))
        err = (got.float() - ref.float()).abs().max().item()
        ok = err <= sc * base and got.dtype == dtype
        metric = f"err={err:.2e} atol={sc*base:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:9s} {str(bshape):9s} b={beta} a={alpha} "
          f"B={B} {M}x{N}x{K} {metric}")
    return ok


def _addbmm_int_ref(inp, a, b, beta, alpha, dtype):
    """Exact int addbmm: beta*input + alpha*Σᵢ(a[i]@b[i]) in int64, wrap to width.
    torch.addbmm errors on int/mps, so build the reference on CPU."""
    prod = (a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)).sum(0)   # (M,N)
    M, N = prod.shape
    r = alpha * prod + beta * inp.cpu().to(torch.int64).expand(M, N)
    if dtype == torch.int64:
        return r
    mod = 1 << _INT_BITS[dtype]
    r = r % mod
    if dtype != torch.uint8:
        r = torch.where(r >= (mod >> 1), r - mod, r)
    return r.to(dtype)


def check_addbmm(B, M, N, K, dtype, bshape, beta=1, alpha=1):
    """metalblas.addbmm vs torch.addbmm: C = beta*input + alpha*Σᵢ(b1[i]@b2[i]) (2-D)."""
    torch.manual_seed(0)
    is_int = dtype in _INT_BITS or dtype == torch.int64
    if dtype.is_floating_point or dtype.is_complex:
        a = torch.randn(B, M, K, dtype=dtype, device='mps')
        b = torch.randn(B, K, N, dtype=dtype, device='mps')
    else:
        a = _int_rand(B, M, K, dtype=dtype)
        b = _int_rand(B, K, N, dtype=dtype)
    inp = _bias(bshape, dtype)
    got = metalblas.addbmm(inp, a, b, beta=beta, alpha=alpha)
    shape_ok = tuple(got.shape) == (M, N) and got.dtype == dtype
    if is_int:                                  # torch.addbmm errors on int/mps -> CPU ref
        ref = _addbmm_int_ref(inp, a, b, beta, alpha, dtype)
        ok = shape_ok and torch.equal(got.cpu(), ref)
        metric = "bit-exact" if ok else "MISMATCH"
    elif dtype.is_complex:
        hp = torch.complex64
        prod = (a.cpu().to(hp) @ b.cpu().to(hp)).sum(0)
        ref = (alpha * prod if alpha != 0 else torch.zeros_like(prod))
        if beta != 0:
            ref = ref + beta * inp.cpu().to(hp).expand(M, N)
        err = (got.cpu().to(hp) - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-9)
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
        ok = shape_ok and rel <= rtol
        metric = f"rel={rel:.2e}"
    else:
        ref = torch.addbmm(inp.float(), a.float(), b.float(), beta=beta, alpha=alpha).to(dtype)
        sc = abs(alpha) + abs(beta)
        base = (max(0.1, 5e-3 * (B * K)**0.5) if dtype == torch.float32
                else max(5e-1, 3e-2 * (B * K)**0.5) if dtype == torch.bfloat16
                else max(5e-2, 1e-2 * (B * K)**0.5))
        err = (got.float() - ref.float()).abs().max().item()
        ok = shape_ok and err <= sc * base
        metric = f"err={err:.2e} atol={sc*base:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:9s} {str(bshape):9s} b={beta} a={alpha} "
          f"B={B} {M}x{N}x{K} {metric}")
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


def check_matmul_nd(sa, sb, dtype):
    """metalblas.matmul vs torch.matmul for N-D / 1-D operands (drop-in semantics).
    Bit-exact for ints; atol for fp; relative for complex. K is the contracted dim."""
    torch.manual_seed(0)
    is_int = dtype in _INT_BITS or dtype == torch.int64
    rand = (lambda s: _int_rand(*s, dtype=dtype)) if is_int else \
           (lambda s: torch.randn(*s, dtype=dtype, device='mps'))
    a = rand(sa)
    b = rand(sb)
    K = sa[-1] if len(sa) >= 2 else sa[0]
    got = mb_matmul(a, b)
    ref = torch.matmul(a, b)
    shape_ok = tuple(got.shape) == tuple(ref.shape) and got.dtype == dtype
    if is_int:
        hp = torch.matmul(a.cpu().to(torch.int64), b.cpu().to(torch.int64))
        if dtype != torch.int64:
            mod = 1 << _INT_BITS[dtype]
            hp = hp % mod
            if dtype != torch.uint8:
                hp = torch.where(hp >= (mod >> 1), hp - mod, hp)
        ok = shape_ok and torch.equal(got.cpu(), hp.to(dtype))
        metric = "bit-exact" if ok else "MISMATCH"
    elif dtype.is_complex:
        hp = torch.complex64
        rref = torch.matmul(a.cpu().to(hp), b.cpu().to(hp))
        err = (got.cpu().to(hp) - rref).abs().max().item()
        rel = err / (rref.abs().max().item() + 1e-9)
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
        ok = shape_ok and rel <= rtol
        metric = f"rel={rel:.2e}"
    else:
        atol = (max(0.1, 5e-3 * K**0.5) if dtype == torch.float32
                else max(5e-1, 3e-2 * K**0.5) if dtype == torch.bfloat16
                else max(5e-2, 1e-2 * K**0.5))
        err = (got.float() - ref.float()).abs().max().item()
        ok = shape_ok and err <= atol
        metric = f"err={err:.2e} atol={atol:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {str(dtype).split('.')[-1]:9s} {str(sa):16s} @ {str(sb):16s} "
          f"-> {str(tuple(got.shape)):14s} {metric}")
    return ok


# Vector / rank-1 ops (dot / vdot / outer / mv) vs torch.
_IS_INT = lambda dt: dt in _INT_BITS or dt == torch.int64


def _vrand(shape, dtype):
    return _int_rand(*shape, dtype=dtype) if _IS_INT(dtype) else \
        torch.randn(*shape, dtype=dtype, device='mps')


def _vcheck(name, got, ref, dtype, tag):
    """Shared pass/print: ints bit-exact, complex relative, fp atol (K-scaled)."""
    shape_ok = tuple(got.shape) == tuple(ref.shape) and got.dtype == dtype
    K = max(got.shape[-1] if got.ndim else 1, 1)
    if _IS_INT(dtype):
        ok = shape_ok and torch.equal(got.cpu(), ref.cpu())
        metric = "bit-exact" if ok else "MISMATCH"
    elif dtype.is_complex:
        hp = torch.complex64
        err = (got.cpu().to(hp) - ref.cpu().to(hp)).abs().max().item()
        rel = err / (ref.cpu().to(hp).abs().max().item() + 1e-9)
        rtol = 3e-2 if dtype == torch.complex32 else 5e-3
        ok = shape_ok and rel <= rtol
        metric = f"rel={rel:.2e} rtol={rtol:.1e}"
    else:
        atol = (max(0.1, 5e-3 * K**0.5) if dtype == torch.float32
                else max(5e-1, 3e-2 * K**0.5) if dtype == torch.bfloat16
                else max(5e-2, 1e-2 * K**0.5))
        err = (got.float() - ref.float()).abs().max().item()
        ok = shape_ok and err <= atol
        metric = f"err={err:.2e} atol={atol:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name:6s} {str(dtype).split('.')[-1]:9s} {tag:14s} {metric}")
    return ok


def check_dot(K, dtype):
    torch.manual_seed(0)
    a, b = _vrand((K,), dtype), _vrand((K,), dtype)
    return _vcheck("dot", metalblas.dot(a, b), torch.dot(a, b), dtype, f"K={K}")


def check_vdot(K, dtype):
    """vdot conjugates the FIRST arg; complex inputs have nonzero imaginary parts."""
    torch.manual_seed(1)
    a, b = _vrand((K,), dtype), _vrand((K,), dtype)
    got, ref = metalblas.vdot(a, b), torch.vdot(a, b)
    ok = _vcheck("vdot", got, ref, dtype, f"K={K}")
    if dtype.is_complex:                # vdot must DIFFER from dot (conj effect)
        hp = torch.complex64
        same = (metalblas.dot(a, b).cpu().to(hp) - ref.cpu().to(hp)).abs().max().item()
        ok = ok and same > 1e-3        # un-conjugated dot is measurably different
    return ok


def check_outer(M, N, dtype):
    torch.manual_seed(0)
    a, b = _vrand((M,), dtype), _vrand((N,), dtype)
    got = metalblas.outer(a, b)
    ok = _vcheck("outer", got, torch.outer(a, b), dtype, f"{M}x{N}")
    return ok and torch.equal(metalblas.ger(a, b).cpu().to(torch.complex64) if dtype.is_complex
                              else metalblas.ger(a, b).cpu(),
                              got.cpu().to(torch.complex64) if dtype.is_complex else got.cpu())


def check_mv(M, K, dtype):
    torch.manual_seed(0)
    mat, vec = _vrand((M, K), dtype), _vrand((K,), dtype)
    return _vcheck("mv", metalblas.mv(mat, vec), torch.mv(mat, vec), dtype, f"{M}x{K}")


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

    # bmm / baddbmm: batched 3-D GEMM, matching torch.bmm / torch.baddbmm.
    bmm_dtypes = addmm_dtypes
    print("=== bmm (batched) ===")
    for dt in bmm_dtypes:
        # square, thin-M/N, attention-shaped, many-small (launch regime), partial-edge.
        for (B, M, N, K) in [(8, 128, 128, 128), (32, 256, 256, 256), (4, 512, 512, 512),
                             (96, 512, 512, 64), (16, 64, 4096, 512), (128, 4096, 64, 256),
                             (512, 64, 64, 64), (2048, 64, 64, 32), (3, 130, 100, 200)]:
            check_bmm(B, M, N, K, dt)
    print("=== bmm: rank-1 batched (M==1 / N==1) ===")
    for dt in bmm_dtypes:
        check_bmm(8, 1, 512, 256, dt)
        check_bmm(8, 512, 1, 256, dt)
    print("=== bmm: transposed batched views (col-major matrices) ===")
    for dt in bmm_dtypes:
        check_bmm(8, 256, 256, 256, dt, layout="tr")
    print("=== baddbmm: bias broadcast shapes ===")
    for dt in bmm_dtypes:
        B, M, N, K = 4, 128, 96, 256
        for bshape in [(B, M, N), (B, 1, N), (B, M, 1), (1, M, N), (M, N), (N,), (1,), ()]:
            check_baddbmm(B, M, N, K, dt, bshape)
    print("=== baddbmm: beta/alpha scaling ===")
    for dt in bmm_dtypes:
        for (beta, alpha) in [(2, 3), (0, 1), (1, 0), (0, 0)]:
            check_baddbmm(4, 128, 96, 256, dt, (96,), beta=beta, alpha=alpha)
    print("=== baddbmm: partial-edge tiles ===")
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        for (M, N, K) in [(130, 100, 200), (257, 129, 257)]:
            for bshape in [(4, M, N), (N,), ()]:
                check_baddbmm(4, M, N, K, dt, bshape)

    # addbmm: 2-D C = beta*input + alpha*Σᵢ(b1[i]@b2[i]); batch REDUCED (vs baddbmm).
    print("=== addbmm (batch-reduced) ===")
    for dt in bmm_dtypes:
        B, M, N, K = 8, 128, 96, 256
        for bshape in [(M, N), (1, N), (M, 1), (N,), (1,), ()]:
            check_addbmm(B, M, N, K, dt, bshape)
    print("=== addbmm: shapes / beta-alpha ===")
    for dt in bmm_dtypes:
        for (B, M, N, K) in [(32, 64, 64, 64), (4, 256, 256, 512)]:
            check_addbmm(B, M, N, K, dt, (N,))
        for (beta, alpha) in [(1, 1), (2, 3), (0, 1), (1, 0)]:
            check_addbmm(8, 128, 96, 256, dt, (96,), beta=beta, alpha=alpha)

    # batched int/complex under load: exercise the single batched int_gemm launch and
    # the batched complex baddbmm path at large B. Bit-exact (int) / relative (complex).
    print("=== batched int/complex (large B) ===")
    for dt in [torch.int8, torch.int32, torch.int64]:
        for (B, M, N, K) in [(256, 128, 128, 128), (512, 64, 64, 64)]:
            check_bmm(B, M, N, K, dt)
            check_baddbmm(B, M, N, K, dt, (N,))
    check_baddbmm(64, 256, 256, 128, torch.complex64, (256,))
    check_baddbmm(64, 256, 256, 128, torch.complex64, (64, 256, 256), beta=2, alpha=3)

    # matmul N-D / 1-D: drop-in torch.matmul (1-D dot/promotion + batched broadcast).
    print("=== matmul N-D / 1-D ===")
    nd_dtypes = [torch.float32, torch.bfloat16, torch.float16,
                 torch.complex64, torch.int32, torch.int8, torch.int64]
    nd_cases = [
        ((64,), (64,)),                       # 1-D @ 1-D -> scalar
        ((32,), (32, 48)),                    # 1-D @ 2-D
        ((40, 32), (32,)),                    # 2-D @ 1-D
        ((32,), (8, 32, 48)),                 # 1-D @ 3-D
        ((6, 40, 32), (32,)),                 # 3-D @ 1-D
        ((8, 40, 32), (8, 32, 48)),           # (B,M,K)@(B,K,N)
        ((8, 40, 32), (32, 48)),              # (B,M,K)@(K,N)
        ((40, 32), (8, 32, 48)),              # (M,K)@(B,K,N)
        ((2, 3, 40, 32), (2, 3, 32, 48)),     # 4-D @ 4-D
        ((3, 1, 64, 32), (1, 5, 32, 48)),     # true broadcast -> (3,5,64,48)
    ]
    for dt in nd_dtypes:
        for (sa, sb) in nd_cases:
            check_matmul_nd(sa, sb, dt)

    # vector ops (dot / vdot / outer / mv): drop-in torch.dot/vdot/outer/mv.
    print("=== vector ops (dot / vdot / outer / mv) ===")
    vec_dtypes = [torch.float32, torch.bfloat16, torch.float16,
                  torch.complex64, torch.complex32,
                  torch.int8, torch.int32, torch.int64]
    for dt in vec_dtypes:
        for K in (4096, 7, 1):
            check_dot(K, dt)
            check_vdot(K, dt)          # asserts conj-of-first for complex
        check_outer(128, 96, dt)
        check_mv(256, 512, dt)


if __name__ == "__main__":
    main()
