"""Correctness checks against PyTorch on MPS."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metalblas
from metalblas.kernels import has_metal4


INTS = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
FLOATS = (torch.float32, torch.float16, torch.bfloat16)
COMPLEX = (torch.complex64, torch.complex32)
ALL = FLOATS + COMPLEX + INTS
_BITS = {torch.int8: 8, torch.uint8: 8, torch.int16: 16, torch.int32: 32}
_checks = 0


def _rand(shape, dtype):
    if dtype in INTS:
        lo = 0 if dtype is torch.uint8 else -4
        return torch.randint(lo, 4, shape, dtype=dtype, device="mps")
    return torch.randn(shape, dtype=dtype, device="mps")


def _wrap(x, dtype):
    if dtype is torch.int64:
        return x
    mod = 1 << _BITS[dtype]
    x = x % mod
    if dtype is not torch.uint8:
        x = torch.where(x >= mod // 2, x - mod, x)
    return x.to(dtype)


def _product(a, b):
    if a.dtype in INTS:
        return torch.matmul(a.cpu().long(), b.cpu().long())
    if a.dtype in COMPLEX:
        return torch.matmul(a.cpu().cfloat(), b.cpu().cfloat())
    return torch.matmul(a.float(), b.float())


def _atol(dtype, k, strict=False):
    if dtype is torch.float32:
        return max(1e-3, 1e-5 * k) if strict else max(0.1, 5e-3 * k**0.5)
    if dtype is torch.bfloat16:
        return max(0.5, 3e-2 * k**0.5)
    return max(5e-2, 1e-2 * k**0.5)


def _check(got, ref, dtype, k, tag, scale=1, strict=False):
    global _checks
    assert got.dtype == dtype and got.shape == ref.shape, \
        f"{tag}: got {got.shape} {got.dtype}, expected {ref.shape} {dtype}"
    if dtype in INTS:
        expected = _wrap(ref, dtype)
        assert torch.equal(got.cpu(), expected), f"{tag}: integer mismatch"
    elif dtype in COMPLEX:
        got, ref = got.cpu().cfloat(), ref.cfloat()
        rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        limit = 3e-2 if dtype is torch.complex32 else 5e-3
        assert rel <= limit, f"{tag}: relative error {rel:.3e} > {limit:.3e}"
    else:
        error = (got.float() - ref.to(dtype).float()).abs().max().item()
        limit = scale * _atol(dtype, k, strict)
        assert error <= limit, f"{tag}: absolute error {error:.3e} > {limit:.3e}"
    _checks += 1


def _matrices(M, N, K, dtype, layout="rm"):
    a = _rand((K, M), dtype).t() if layout in ("ta", "tr") else _rand((M, K), dtype)
    b = _rand((N, K), dtype).t() if layout == "tr" else _rand((K, N), dtype)
    if layout == "conj":
        a = a.conj()
    elif layout == "strided":
        if N == 1:
            b = _rand((K, 2), dtype)[:, :1]
        else:
            a = _rand((K, 2), dtype)[:, :1].t()
    return a, b


def check_mm(M, N, K, dtype, backend=None, layout="rm"):
    torch.manual_seed(0)
    a, b = _matrices(M, N, K, dtype, layout)
    kwargs = {"backend": backend} if backend else {}
    got = metalblas.matmul(a, b, **kwargs)
    _check(got, _product(a, b), dtype, K,
           f"matmul {dtype} {layout} {backend or 'auto'} {M}x{N}x{K}",
           strict=backend not in (None, "mpp"))


def check_bmm(B, M, N, K, dtype, layout="rm"):
    torch.manual_seed(0)
    if layout == "tr":
        a = _rand((B, K, M), dtype).transpose(-2, -1)
        b = _rand((B, N, K), dtype).transpose(-2, -1)
    else:
        a, b = _rand((B, M, K), dtype), _rand((B, K, N), dtype)
    _check(metalblas.bmm(a, b), _product(a, b), dtype, K,
           f"bmm {dtype} {layout} {B}x{M}x{N}x{K}")


def _bias(shape, dtype):
    return _rand(shape, dtype)


def _fused_ref(input, a, b, beta, alpha, reduce):
    product = _product(a, b)
    if reduce:
        product = product.sum(0)
    if a.dtype in INTS:
        bias = input.cpu().long()
    elif a.dtype in COMPLEX:
        bias = input.cpu().cfloat()
    else:
        bias = input.float()
    result = alpha * product if alpha != 0 else torch.zeros_like(product)
    return result + beta * bias.expand(product.shape) if beta != 0 else result


def check_fused(op, shape, dtype, bshape, beta=1, alpha=1):
    torch.manual_seed(0)
    if op == "addmm":
        M, N, K = shape
        a, b = _rand((M, K), dtype), _rand((K, N), dtype)
    else:
        B, M, N, K = shape
        a, b = _rand((B, M, K), dtype), _rand((B, K, N), dtype)
    input = _bias(bshape, dtype)
    got = getattr(metalblas, op)(input, a, b, beta=beta, alpha=alpha)
    ref = _fused_ref(input, a, b, beta, alpha, op == "addbmm")
    _check(got, ref, dtype, K * (shape[0] if op == "addbmm" else 1),
           f"{op} {dtype} {shape} bias={bshape} beta={beta} alpha={alpha}",
           scale=abs(alpha) + abs(beta))


def check_nd(a_shape, b_shape, dtype):
    torch.manual_seed(0)
    a, b = _rand(a_shape, dtype), _rand(b_shape, dtype)
    k = a_shape[0] if len(a_shape) == 1 else a_shape[-1]
    _check(metalblas.matmul(a, b), _product(a, b), dtype, k,
           f"matmul {dtype} {a_shape} @ {b_shape}")


def check_vectors(dtype):
    for k in (4096, 7, 1):
        a, b = _rand((k,), dtype), _rand((k,), dtype)
        _check(metalblas.dot(a, b), _product(a, b), dtype, k, f"dot {dtype} {k}")
        ref = torch.vdot(a.cpu().cfloat(), b.cpu().cfloat()) if dtype in COMPLEX else _product(a, b)
        _check(metalblas.vdot(a, b), ref, dtype, k, f"vdot {dtype} {k}")
    a, b = _rand((128,), dtype), _rand((96,), dtype)
    if dtype in INTS:
        ref = a.cpu().long().unsqueeze(1) * b.cpu().long().unsqueeze(0)
    elif dtype in COMPLEX:
        ref = torch.outer(a.cpu().cfloat(), b.cpu().cfloat())
    else:
        ref = torch.outer(a.float(), b.float())
    _check(metalblas.outer(a, b), ref, dtype, 1, f"outer {dtype}")
    a, b = _rand((256, 512), dtype), _rand((512,), dtype)
    _check(metalblas.mv(a, b), _product(a, b), dtype, 512, f"mv {dtype}")


def main():
    base = [(64, 64, 64), (128, 128, 128), (256, 256, 256),
            (513, 257, 129), (1024, 1024, 256)]
    for dtype in FLOATS:
        for backend in (("simd", "mpp") if has_metal4() else ("simd",)):
            for shape in base + ([(33, 33, 33)] if dtype is torch.float32 and backend == "simd" else []):
                check_mm(*shape, dtype, backend)
        for shape in [(128, 128, 128), (513, 257, 129), (1024, 1024, 256),
                      (1, 4096, 4096), (4096, 1, 4096)]:
            check_mm(*shape, dtype, layout="tr")
        for shape in [(1, 4096, 4096), (4096, 1, 4096),
                      (1, 1024, 1024), (1024, 1, 1024)]:
            check_mm(*shape, dtype, backend="gemv")
        for shape in [(1, 1024, 256), (1, 4096, 1024),
                      (64, 1, 256), (1024, 1, 1024)]:
            check_mm(*shape, dtype, backend="gemv", layout="strided")

    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512),
                  (1024, 1024, 1024), (513, 257, 129), (333, 444, 555), (2, 64, 128),
                  (1, 4096, 4096), (4096, 1, 4096), (1, 1024, 1024),
                  (1024, 1, 1024), (1, 1, 512), (1, 17, 33)]:
        check_mm(*shape, torch.complex64)
    for shape in [(256, 256, 256), (513, 257, 129)]:
        check_mm(*shape, torch.complex64, layout="ta")
        check_mm(*shape, torch.complex64, layout="conj")
    for shape in [(256, 256, 256), (512, 512, 512),
                  (1, 2048, 2048), (2048, 1, 2048)]:
        check_mm(*shape, torch.complex32)

    for dtype in INTS:
        for shape in [(64, 64, 64), (256, 256, 256), (513, 257, 129), (333, 444, 555),
                      (1024, 1024, 1024), (96, 4096, 512), (33, 33, 33),
                      (1, 4096, 4096), (4096, 1, 4096), (1, 1024, 257), (300, 1, 1024)]:
            check_mm(*shape, dtype)
        check_mm(256, 256, 256, dtype, layout="ta")

    for dtype in ALL:
        for bshape in [(128, 96), (1, 96), (128, 1), (96,), (1,), ()]:
            check_fused("addmm", (128, 96, 256), dtype, bshape)
        for beta, alpha in [(2, 3), (0, 1), (1, 0), (0, 0)]:
            check_fused("addmm", (128, 96, 256), dtype, (96,), beta, alpha)
        check_fused("addmm", (1, 4096, 4096), dtype, (4096,))
        check_fused("addmm", (4096, 1, 4096), dtype, (1,))

    for dtype in FLOATS:
        for M, N, K in [(130, 100, 200), (257, 257, 257), (333, 444, 555)]:
            for bshape in [(M, N), (1, N), (M, 1), (N,), (1,), ()]:
                check_fused("addmm", (M, N, K), dtype, bshape)
        a, b = _rand((128, 256), dtype), _rand((256, 96), dtype)
        got = metalblas.addmm(torch.full((128, 96), float("nan"), dtype=dtype, device="mps"),
                              a, b, beta=0)
        assert not torch.isnan(got).any().item(), f"addmm beta=0 leaked NaN for {dtype}"

    bmm_shapes = [(8, 128, 128, 128), (32, 256, 256, 256), (4, 512, 512, 512),
                  (96, 512, 512, 64), (16, 64, 4096, 512), (128, 4096, 64, 256),
                  (512, 64, 64, 64), (2048, 64, 64, 32), (3, 130, 100, 200),
                  (8, 1, 512, 256), (8, 512, 1, 256)]
    for dtype in ALL:
        for shape in bmm_shapes:
            check_bmm(*shape, dtype)
        check_bmm(8, 256, 256, 256, dtype, "tr")
        for bshape in [(4, 128, 96), (4, 1, 96), (4, 128, 1),
                       (1, 128, 96), (128, 96), (96,), (1,), ()]:
            check_fused("baddbmm", (4, 128, 96, 256), dtype, bshape)
        for beta, alpha in [(2, 3), (0, 1), (1, 0), (0, 0)]:
            check_fused("baddbmm", (4, 128, 96, 256), dtype, (96,), beta, alpha)
        for bshape in [(128, 96), (1, 96), (128, 1), (96,), (1,), ()]:
            check_fused("addbmm", (8, 128, 96, 256), dtype, bshape)
        for shape in [(32, 64, 64, 64), (4, 256, 256, 512)]:
            check_fused("addbmm", shape, dtype, (shape[2],))
        for beta, alpha in [(1, 1), (2, 3), (0, 1), (1, 0)]:
            check_fused("addbmm", (8, 128, 96, 256), dtype, (96,), beta, alpha)

    for dtype in FLOATS:
        for M, N, K in [(130, 100, 200), (257, 129, 257)]:
            for bshape in [(4, M, N), (N,), ()]:
                check_fused("baddbmm", (4, M, N, K), dtype, bshape)

    nd_cases = [((64,), (64,)), ((32,), (32, 48)), ((40, 32), (32,)),
                ((32,), (8, 32, 48)), ((6, 40, 32), (32,)),
                ((8, 40, 32), (8, 32, 48)), ((8, 40, 32), (32, 48)),
                ((40, 32), (8, 32, 48)), ((2, 3, 40, 32), (2, 3, 32, 48)),
                ((3, 1, 64, 32), (1, 5, 32, 48))]
    for dtype in (torch.float32, torch.bfloat16, torch.float16,
                  torch.complex64, torch.int32, torch.int8, torch.int64):
        for shapes in nd_cases:
            check_nd(*shapes, dtype)
    for dtype in (torch.float32, torch.bfloat16, torch.float16,
                  torch.complex64, torch.complex32, torch.int8, torch.int32, torch.int64):
        check_vectors(dtype)

    torch.mps.synchronize()
    print(f"{_checks} checks passed")


if __name__ == "__main__":
    main()
