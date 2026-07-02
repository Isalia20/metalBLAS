import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metalblas

FAILS = []
PASSES = [0]


def _check(label, got, ref, out_fmt=None, tol=0.06):
    g = got.float().cpu()
    rel = ((g - ref).abs() / ref.abs().clamp(min=1.0)).max().item()
    nan = torch.isnan(g).sum().item()
    ok = rel < tol and nan == 0
    if out_fmt is not None and not got.is_contiguous(memory_format=out_fmt):
        ok = False
        label += " [bad memory format]"
    if ok:
        PASSES[0] += 1
    else:
        FAILS.append(f"{label} (rel={rel:.4f}, nan={nan})")
        print(f"FAIL {label}: rel={rel:.4f} nan={nan}")


def t2d(label, NB=2, C=32, H=17, W=19, O=48, k=3, s=1, p=1, d=1, groups=1,
        bias=True, dtype=torch.bfloat16, channels_last=False, unbatched=False):
    torch.manual_seed(0)
    x = torch.randn(NB, C, H, W, dtype=dtype, device="mps")
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    if unbatched:
        x = x[0]
    kh, kw = (k, k) if isinstance(k, int) else k
    wt = torch.randn(O, C // groups, kh, kw, dtype=dtype, device="mps") * 0.1
    b = torch.randn(O, dtype=dtype, device="mps") if bias else None
    ref = F.conv2d(x.float().cpu(), wt.float().cpu(),
                   b.float().cpu() if bias else None,
                   stride=s, padding=p, dilation=d, groups=groups)
    got = metalblas.conv2d(x, wt, b, stride=s, padding=p, dilation=d, groups=groups)
    fmt = None if unbatched else (
        torch.channels_last if channels_last else torch.contiguous_format)
    _check(label, got, ref, fmt)


def t1d(label, NB=2, C=32, L=100, O=48, k=3, s=1, p=1, d=1, groups=1,
        bias=True, dtype=torch.bfloat16):
    torch.manual_seed(0)
    x = torch.randn(NB, C, L, dtype=dtype, device="mps")
    wt = torch.randn(O, C // groups, k, dtype=dtype, device="mps") * 0.1
    b = torch.randn(O, dtype=dtype, device="mps") if bias else None
    ref = F.conv1d(x.float().cpu(), wt.float().cpu(),
                   b.float().cpu() if bias else None,
                   stride=s, padding=p, dilation=d, groups=groups)
    got = metalblas.conv1d(x, wt, b, stride=s, padding=p, dilation=d, groups=groups)
    _check(label, got, ref, torch.contiguous_format)


def t3d(label, NB=2, C=16, D=6, H=13, W=15, O=24, k=3, s=1, p=1, d=1, groups=1,
        bias=True, dtype=torch.bfloat16, channels_last=False):
    torch.manual_seed(0)
    x = torch.randn(NB, C, D, H, W, dtype=dtype, device="mps")
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    kt = (k, k, k) if isinstance(k, int) else k
    wt = torch.randn(O, C // groups, *kt, dtype=dtype, device="mps") * 0.1
    b = torch.randn(O, dtype=dtype, device="mps") if bias else None
    ref = F.conv3d(x.float().cpu(), wt.float().cpu(),
                   b.float().cpu() if bias else None,
                   stride=s, padding=p, dilation=d, groups=groups)
    got = metalblas.conv3d(x, wt, b, stride=s, padding=p, dilation=d, groups=groups)
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    _check(label, got, ref, fmt)


def main():
    for dtype in (torch.bfloat16, torch.float32):
        tag = {torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]

        t2d(f"{tag} 2d 3x3 p1 bias", dtype=dtype)
        t2d(f"{tag} 2d 3x3 nobias", bias=False, dtype=dtype)
        t2d(f"{tag} 2d 3x3 s2", s=2, dtype=dtype)
        t2d(f"{tag} 2d s(2,1) asym", s=(2, 1), dtype=dtype)
        t2d(f"{tag} 2d 5x3 p(2,1)", k=(5, 3), p=(2, 1), dtype=dtype)
        t2d(f"{tag} 2d d2 p2", d=2, p=2, dtype=dtype)
        t2d(f"{tag} 2d 1x1 (GEMM route)", k=1, p=0, dtype=dtype)
        t2d(f"{tag} 2d 1x1 nobias", k=1, p=0, bias=False, dtype=dtype)
        t2d(f"{tag} 2d 1x1 s2", k=1, p=0, s=2, dtype=dtype)
        t2d(f"{tag} 2d 7x7 s2 stem", C=3, O=64, H=64, W=64, k=7, s=2, p=3, dtype=dtype)
        t2d(f"{tag} 2d groups2", groups=2, dtype=dtype)
        t2d(f"{tag} 2d groups8", groups=8, C=64, O=64, dtype=dtype)
        t2d(f"{tag} 2d depthwise", groups=32, C=32, O=32, dtype=dtype)
        t2d(f"{tag} 2d depthwise mult2", groups=32, C=32, O=64, dtype=dtype)
        t2d(f"{tag} 2d depthwise s2", groups=32, C=32, O=32, s=2, dtype=dtype)
        t2d(f"{tag} 2d same-pad", p="same", dtype=dtype)
        t2d(f"{tag} 2d same-pad even-k", k=4, p="same", dtype=dtype)
        t2d(f"{tag} 2d same-pad k2", k=2, p="same", dtype=dtype)
        t2d(f"{tag} 2d valid-pad", p="valid", dtype=dtype)
        t2d(f"{tag} 2d overpad p2", p=2, dtype=dtype)

        t2d(f"{tag} 2d channels_last", channels_last=True, dtype=dtype)
        t2d(f"{tag} 2d chlast 1x1", channels_last=True, k=1, p=0, dtype=dtype)
        t2d(f"{tag} 2d chlast depthwise", channels_last=True, groups=32, C=32,
            O=32, dtype=dtype)
        t2d(f"{tag} 2d unbatched", unbatched=True, dtype=dtype)
        t2d(f"{tag} 2d tiny", H=4, W=4, C=8, O=8, dtype=dtype)
        t2d(f"{tag} 2d W=1", W=1, dtype=dtype)

        t1d(f"{tag} 1d k3 p1", dtype=dtype)
        t1d(f"{tag} 1d k5 s2", k=5, s=2, p=2, dtype=dtype)
        t1d(f"{tag} 1d k9 d4", k=9, d=4, p=16, dtype=dtype)
        t1d(f"{tag} 1d long L=20000", L=20000, C=16, O=16, dtype=dtype)
        t1d(f"{tag} 1d groups4", groups=4, C=64, O=64, dtype=dtype)
        t1d(f"{tag} 1d depthwise k4 (mamba)", groups=64, C=64, O=64, k=4, p=3,
            dtype=dtype)

        t3d(f"{tag} 3d 3x3x3 p1", dtype=dtype)
        t3d(f"{tag} 3d s2", s=2, dtype=dtype)
        t3d(f"{tag} 3d k(1,3,3)", k=(1, 3, 3), p=(0, 1, 1), dtype=dtype)
        t3d(f"{tag} 3d channels_last_3d", channels_last=True, dtype=dtype)
        t3d(f"{tag} 3d groups2", groups=2, dtype=dtype)

    t2d("fp16 2d 3x3 (fallback)", dtype=torch.float16)


    x = torch.randn(2, 64, 20, 20, dtype=torch.bfloat16, device="mps")[:, ::2]
    wt = torch.randn(48, 32, 3, 3, dtype=torch.bfloat16, device="mps") * 0.1
    ref = F.conv2d(x.float().cpu(), wt.float().cpu(), padding=1)
    _check("bf16 2d strided-channels", metalblas.conv2d(x, wt, padding=1), ref)

    print(f"\n{PASSES[0]} passed, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print("  ", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
