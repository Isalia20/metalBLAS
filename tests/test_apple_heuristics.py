"""Check MPS's GEMM selection (metalblas/apple_heuristics.py) against traced MPS dispatches, and the GEMMs that run
on it against PyTorch."""
import gzip
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metalblas
from metalblas import apple_heuristics as ah
from metalblas import float_dispatch as fd
from metalblas import kernels

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, 4: torch.float16, 5: torch.bfloat16}


def _load(name):
    with gzip.open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def check_a18():
    """unrollK..globalKSplits of gemm_a18 calls recorded with MPS_LOG_INFO on an M5 Pro (16 GPU cores), some with
    the tuning tables disabled."""
    data = _load("apple_a18_m5pro.json.gz")
    cores = data["device"]["cores"]
    bad = []
    for M, N, K, batch, ta, tb, dt, tables, *expected in data["rows"]:
        if tables:
            got = ah.a18_params(M, N, K, DTYPES[dt], cores, batch=batch, trans_a=bool(ta), trans_b=bool(tb))
        else:
            got = ah._a18_high_core(M, N, K, bool(ta), bool(tb), cores)
        if list(got[:7]) != expected:
            bad.append(f"{M}x{N}x{K} batch={batch} ta={ta} tb={tb} dt={dt} tables={tables}: {list(got)} != {expected}")
    assert not bad, f"{len(bad)}/{len(data['rows'])} A18 mismatches:\n" + "\n".join(bad[:20])
    return len(data["rows"])


def check_a18_routing():
    """All 17 gemm_a18 parameters of mm/bmm calls traced on an M5 Pro (rows without parameters ran on MPS's
    matrix-vector kernels)."""
    data = _load("apple_a18_m5pro_routing.json.gz")
    cores = data["device"]["cores"]
    rows = [r for r in data["rows"] if len(r) > 7]
    bad = []
    for M, N, K, batch, ta, tb, dt, *expected in rows:
        got = list(ah.a18_params(M, N, K, DTYPES[dt], cores, batch=batch, trans_a=bool(ta), trans_b=bool(tb)))
        if got != expected:
            bad.append(f"{M}x{N}x{K} batch={batch} ta={ta} tb={tb} dt={dt}: {got} != {expected}")
    assert not bad, f"{len(bad)}/{len(rows)} A18 mismatches:\n" + "\n".join(bad[:20])
    return len(rows)


def check_a14():
    """Kernel, grid, threadgroup size, staging memory and reduction pass of fp16/bf16 GEMMs traced on M2, M3 Ultra
    and M4 (macOS 26)."""
    data = _load("apple_a14_macos26.json.gz")
    checked = 0
    for d in data["devices"]:
        dev = ah.Device(d["generation"], d["cores"], d["macos"])
        bad = []
        for dt, M, N, K, batch, kernel, grid, tg, mem_b, mem_a, reductions in d["rows"]:
            dtype = DTYPES[dt]
            bm, bn, bk, sm, sn, splits, _ = ah.a14_params(M, N, K, dtype, dev, batch=batch)
            if dev.generation < 18:
                name = "NDArrayMatrixMultiplyNNA14" + ("_EdgeCase" if M % bm or N % bn or K % bk else "")
            else:
                t = "half" if dtype is torch.float16 else "bfloat"
                name = f"mmul_kernel_a16_{t}_{t}_float"
            # B is staged in threadgroup memory before Apple9; A unless the square shape skips it on Apple9+.
            stage_b, stage_a = dev.generation < 18, not (dev.generation >= 18 and M == N == K)
            got = (name, [-(-N // bn), -(-M // bm), splits * batch], 32 * sm * sn,
                   bn * bk * 2 if stage_b else 0, bm * bk * 2 if stage_a else 0, int(splits > 1))
            if got != (kernel, grid, tg, mem_b, mem_a, reductions):
                bad.append(f"{dt} {M}x{N}x{K} batch={batch}: {got} != {(kernel, grid, tg, mem_b, mem_a, reductions)}")
        assert not bad, f"{d['chip']}: {len(bad)}/{len(d['rows'])} A14 mismatches:\n" + "\n".join(bad[:20])
        checked += len(d["rows"])
    return checked


def check_generations():
    cases = {(1, 8): 12, (1, 16): 13, (1, 32): 14, (1, 64): 15, (2, 10): 17, (2, 8): 16, (3, 10): 18, (3, 40): 19,
             (4, 10): 20, (4, 16): 21, (5, 10): 22, (5, 16): 23, (5, 40): 23, (6, 16): 23}
    for (chip, cores), gen in cases.items():
        assert ah.generation(chip, cores) == gen, (chip, cores, ah.generation(chip, cores), gen)


GPUS = {"M1": ah.Device(12, 8, 26), "M2": ah.Device(17, 10, 26), "M3 Ultra": ah.Device(19, 80, 26),
        "M4": ah.Device(20, 10, 27), "M5": ah.Device(22, 10, 27), "M5 Max": ah.Device(23, 40, 27),
        "M5 80-core": ah.Device(23, 80, 27)}
# Square (48x48 tiles on M1/M2), deep K (K split before M5 and on 60+ cores), table shapes, edges, thin M and N,
# matrix-vector shapes, and a single K step with edges.
SHAPES = [(1000, 1000, 1000), (64, 100, 4096), (300, 257, 200), (2000, 1536, 768), (17, 33, 65), (513, 129, 3000),
          (2000, 768, 2304), (12, 256, 1024), (256, 12, 1024), (40, 4096, 40), (128, 128, 20000), (1, 300, 200),
          (300, 1, 200), (3, 5, 1000), (1000, 7, 64), (300, 264, 64), (77, 136, 32), (24, 1024, 1000), (17, 2048, 256),
          (30, 600, 130), (2600, 20, 384), (4100, 32, 520), (20, 4096, 192), (96, 2080, 384), (1536, 2000, 512)]


def _operand(rows, cols, dtype, transposed, batch=(), pad=0):
    t = torch.randn(*batch, *((cols, rows + pad) if transposed else (rows, cols + pad)), device="mps", dtype=dtype)
    t = t[..., :rows] if transposed else t[..., :cols]
    return t.transpose(-2, -1) if transposed else t


def _close(got, ref, dtype, tag):
    err = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    limit = 1e-2 if dtype is torch.bfloat16 else 2e-3
    assert err <= limit, f"{tag}: relative error {err:.2e} > {limit:.0e}"


def check_gemm():
    """matmul, addmm, bmm and baddbmm on the selection of this GPU and of other M-series GPUs, for every layout,
    padded rows, in-place addmm and broadcast biases."""
    if not (kernels.has_metal4() and ah.device()):
        return 0
    detect, checks = ah.device, 0
    try:
        for gpu, dev in [("this GPU", ah.device())] + list(GPUS.items()):
            ah.device = lambda dev=dev: dev
            fd._apple_generation.cache_clear()
            fd.MPS_VECTOR_DTYPES = None
            fd._apple_plan.cache_clear()
            fd._APPLE_FAST.clear()
            fd._APPLE_BMM_FAST.clear()
            for dtype, (M, N, K) in itertools.product((torch.float16, torch.bfloat16), SHAPES):
                torch.manual_seed(0)
                for ta, tb, pad in itertools.product((False, True), (False, True), (0, 3)):
                    a, b = _operand(M, K, dtype, ta, pad=pad), _operand(K, N, dtype, tb, pad=pad)
                    ref, bias = a.float() @ b.float(), torch.randn(N, device="mps", dtype=dtype)
                    tag = f"{gpu} {dtype} {M}x{N}x{K} ta={ta} tb={tb} pad={pad}"
                    _close(metalblas.matmul(a, b), ref, dtype, "matmul " + tag)
                    _close(metalblas.addmm(bias, a, b, beta=0.5, alpha=2.0), 2 * ref + 0.5 * bias.float(), dtype,
                           "addmm " + tag)
                    checks += 2
                a, b = _operand(M, K, dtype, False), _operand(K, N, dtype, False)
                c = torch.randn(M, N, device="mps", dtype=dtype)
                ref = 0.5 * c.float() + 2 * (a.float() @ b.float())
                _close(metalblas.addmm(c, a, b, beta=0.5, alpha=2.0, out=c), ref, dtype, f"in-place addmm {gpu} {M}x{N}x{K}")
                a, b = _operand(M, K, dtype, False, (3,)), _operand(K, N, dtype, True, (3,))
                ref, bias = a.float() @ b.float(), torch.randn(3, M, N, device="mps", dtype=dtype)
                tag = f"{gpu} {dtype} 3x{M}x{N}x{K}"
                _close(metalblas.bmm(a, b), ref, dtype, "bmm " + tag)
                _close(metalblas.baddbmm(bias, a, b, beta=0.5, alpha=2.0), 2 * ref + 0.5 * bias.float(), dtype,
                       "baddbmm " + tag)
                checks += 3
            assert fd._apple_plan.cache_info().currsize > 0 or not fd._APPLE, gpu
    finally:
        ah.device = detect
        fd._apple_generation.cache_clear()
        fd.MPS_VECTOR_DTYPES = None
        fd._apple_plan.cache_clear()
        fd._APPLE_FAST.clear()
        fd._APPLE_BMM_FAST.clear()
    return checks


def main():
    check_generations()
    n18 = check_a18()
    nr = check_a18_routing()
    n14 = check_a14()
    print(f"A18: {n18} recorded selections and {nr} traced parameter sets match; A14: {n14} traced dispatches match")
    print(f"{check_gemm()} GEMM checks passed")


if __name__ == "__main__":
    main()
