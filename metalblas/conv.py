from __future__ import annotations

import os
import time
import torch

from . import kernels
from .dispatch import _pk, _PROFILE, matmul, addmm, bmm, baddbmm

_AUTOTUNE = os.environ.get("METALBLAS_AUTOTUNE", "1") != "0"
_AUTOTUNE_MARGIN = 0.03

_SRC_HINT = 16384

_CONV_PLAN: dict = {}
_CONV_TILE: dict = {}


_TORCH_ROUTE: set = set()


def _fast_key(nd, input, weight, bias, stride, padding, dilation, groups):
    t = lambda v: v if isinstance(v, (int, str)) else tuple(v)
    return (nd, input.dtype, tuple(input.shape), tuple(input.stride()),
            tuple(weight.shape), t(stride), t(padding), t(dilation), groups,
            bias is not None)


def _ntuple(v, n):
    if isinstance(v, (tuple, list)):
        if len(v) != n:
            raise ValueError(f"expected {n}-tuple, got {v}")
        return tuple(int(x) for x in v)
    return (int(v),) * n


def _same_padding(kernel, dilation, stride, n):
    for s in stride:
        if s != 1:
            raise ValueError("padding='same' requires stride 1")


    return tuple((dilation[i] * (kernel[i] - 1)) // 2 for i in range(n))


class _ConvPlan:

    def __init__(self, fn, dims, o_tiles, w_tiles, h_tiles, zmul, NSG, has_bias):
        self.fn = fn
        self.dims = dims
        self.threads = (NSG * 32 * o_tiles, w_tiles, h_tiles * zmul)
        self.group = (NSG * 32, 1, 1)
        self.has_bias = has_bias

    def run(self, act, wts, out, bias=None):
        if self.has_bias:
            self.fn(act, wts, out, self.dims, bias,
                    threads=self.threads, group_size=self.group)
        else:
            self.fn(act, wts, out, self.dims,
                    threads=self.threads, group_size=self.group)


def _build_plan(dtype, tile, khw, sxy, dxy, shape, groups, has_bias, d_nchw,
                conv3d_p=None, srcw=_SRC_HINT, srch=_SRC_HINT):
    BO, BW, BH, NSG = tile
    KH, KW = khw
    SY, SX = sxy
    DY, DX = dxy
    C, H, W, O, HO, WO, NB, PADY, PADX = shape[:9]
    D, DO, PADZ = shape[9:] if len(shape) > 9 else (1, 1, 0)
    CG, OG = C // groups, O // groups
    OGT = (OG + BO - 1) // BO
    kd, sz, dz = conv3d_p if conv3d_p else (1, 1, 1)
    in_t, _, out_t = _PROFILE[dtype]
    srcc = CG if CG <= 64 else -1
    fn, _ = kernels.conv2d_mpp(
        in_t, out_t, BO, BW, BH, NSG, KH, KW, SX, SY, DX, DY,
        relaxed=True, bias=has_bias, d_nchw=d_nchw, grouped=groups > 1,
        conv3d=conv3d_p is not None, KD=kd, SZ=sz, DZ=dz,
        srcw=srcw, srch=srch, srcc=srcc)
    dims = _pk(C, H, W, O, HO, WO, NB, PADX, PADY, CG, OG, OGT, D, DO, PADZ, 0)
    w_tiles = (WO + BW - 1) // BW
    h_tiles = (HO + BH - 1) // BH
    return _ConvPlan(fn, dims, OGT * groups, w_tiles, h_tiles, NB * DO, NSG, has_bias)


class _Conv1dBwPlan:

    def __init__(self, fn, dims, o_tiles, z_tiles, NSG, has_bias):
        self.fn = fn
        self.dims = dims
        self.threads = (NSG * 32 * o_tiles, 1, z_tiles)
        self.group = (NSG * 32, 1, 1)
        self.has_bias = has_bias

    def run(self, act, wts, out, bias=None):
        if self.has_bias:
            self.fn(act, wts, out, self.dims, bias,
                    threads=self.threads, group_size=self.group)
        else:
            self.fn(act, wts, out, self.dims,
                    threads=self.threads, group_size=self.group)


def _build_bw_plan(dtype, tile, k, s, d, shape, has_bias):
    BO, BW, BH, NSG = tile
    C, L, NB, O, LO, PAD = shape
    in_t, _, out_t = _PROFILE[dtype]
    fn, _ = kernels.conv1d_bw(in_t, out_t, BO, BW, BH, NSG, k, s, d,
                              relaxed=True, bias=has_bias)
    dims = _pk(C, L, NB, O, LO, PAD)
    o_t = (O + BO - 1) // BO
    z_t = ((LO + BW - 1) // BW) * ((NB + BH - 1) // BH)
    return _Conv1dBwPlan(fn, dims, o_t, z_t, NSG, has_bias)


def _bw_tile_candidates(NB, O):
    BO = 64 if O >= 48 else 32
    if NB >= 8:
        tiles = [(32, 4), (16, 8), (32, 8), (16, 4)]
    elif NB >= 4:
        tiles = [(32, 4), (64, 4), (16, 4)]
    else:
        tiles = [(64, 2), (32, 2), (128, 2)]
    return [("bw", BO, bw, bh, 4) for (bw, bh) in tiles]


class _Conv1dSgPlan:

    family = "sg"

    def __init__(self, fn, fix, dims, n_tiles, m_tiles, NB, NSG, pad, O, has_bias):
        self.fn = fn
        self.fix = fix
        self.dims = dims
        self.threads = (NSG * 32 * n_tiles, m_tiles, NB)
        self.group = (NSG * 32, 1, 1)
        self.fix_threads = (max(pad, 1), O, NB)
        self.fix_group = (max(min(pad, 32), 1), 8, 1)
        self.pad = pad
        self.has_bias = has_bias

    def run(self, x, wts, out, bias=None):
        args = [x, wts, out, self.dims] + ([bias] if self.has_bias else [])
        self.fn(*args, threads=self.threads, group_size=self.group)
        if self.pad > 0:
            self.fix(*args, threads=self.fix_threads, group_size=self.fix_group)


def _build_sg_plan(dtype, tile, k, d, shape, has_bias):
    BM, BN, NSG = tile
    C, L, NB, O, LO, PAD = shape
    in_t, _, out_t = _PROFILE[dtype]
    fn, fix = kernels.conv1d_sg(in_t, out_t, BM, BN, NSG, k, d,
                                relaxed=True, bias=has_bias)
    dims = _pk(C, L, O, LO, PAD, 0)
    n_t = (LO + BN - 1) // BN
    m_t = (O + BM - 1) // BM
    return _Conv1dSgPlan(fn, fix, dims, n_t, m_t, NB, NSG, PAD, O, has_bias)


_SG_TILES = [("sg", 128, 64, 4), ("sg", 64, 64, 2), ("sg", 64, 128, 4),
             ("sg", 64, 64, 4)]


def _tile_candidates(O, HO, WO, C, groups):
    OG = O // groups
    cands = []

    def add(BO, BW, BH, NSG):
        t = (BO, BW, BH, NSG)
        if t not in cands:
            cands.append(t)


    if WO == 1:
        BOp = min(64, max(32, OG))
        for BH in (64, 32, 128):
            for NSG in (4, 2):
                add(BOp, 1, BH, NSG)
        return cands


    if WO <= 8 and HO > 4:
        add(min(64, max(32, OG)), 8, 8, 4)
    elif HO <= 4:
        add(min(64, max(32, OG)), 32 if WO >= 32 else 16, 4, 4)
    else:
        add(min(64, max(32, OG)), 16, 8, 4)
    add(64, 16, 4, 4)
    add(64, 8, 8, 4)
    add(32, 16, 8, 4)
    add(64, 16, 8, 4)
    if OG >= 128:
        add(128, 16, 4, 4)
        add(128, 8, 8, 4)
    if OG <= 32:
        add(32, 8, 8, 2)
    return cands


def _probe_iters(flops):
    if flops <= 2_000_000_000:
        return 30, 60, 6
    if flops <= 50_000_000_000:
        return 10, 4, 8
    return 4, 1, 8


_OURS_MARGIN = 0.06


def _autotune_conv(built, flops, torch_probe=None):
    warmup, iters, reps = _probe_iters(flops)
    runs = [r for (_, r, _) in built]
    if torch_probe is not None:
        runs.append(torch_probe)
    for r in runs:
        for _ in range(warmup):
            r()
    torch.mps.synchronize()
    if torch_probe is not None:
        t0 = time.perf_counter()
        torch_probe()
        torch.mps.synchronize()
        t1 = time.perf_counter() - t0
        p0 = time.perf_counter()
        built[0][1]()
        torch.mps.synchronize()
        if t1 < 2.0 * (time.perf_counter() - p0):
            deadline = time.perf_counter() + 1.5
            for _ in range(150):
                torch_probe()
                if time.perf_counter() > deadline:
                    break
            torch.mps.synchronize()
    times = [float("inf")] * len(runs)
    for _ in range(reps):
        for j, r in enumerate(runs):
            t0 = time.perf_counter()
            for _ in range(iters):
                r()
            torch.mps.synchronize()
            times[j] = min(times[j], (time.perf_counter() - t0) / iters)
    if torch_probe is not None:
        best_ours = min(times[:len(built)])
        if best_ours >= times[-1] * (1.0 - _OURS_MARGIN):
            return "torch", "torch"
        times = times[:len(built)]
    i = min(range(len(times)), key=lambda j: times[j])
    if i != 0 and times[i] >= times[0] * (1.0 - _AUTOTUNE_MARGIN):
        i = 0
    return built[i][0], built[i][2]


def _to_nhwc(x):
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return x.permute(0, 2, 3, 1)
    if x.dim() == 5 and x.is_contiguous(memory_format=torch.channels_last_3d):
        return x.permute(0, 2, 3, 4, 1)
    xc = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    if C <= 8:
        perm = (0, *range(2, x.dim()), 1)
        return xc.permute(*perm).contiguous()
    X = 1
    for s in x.shape[2:]:
        X *= s
    out = torch.empty((N, *x.shape[2:], C), dtype=x.dtype, device=x.device)
    elt = _PROFILE[x.dtype][0]
    TC, TX = (16, 64) if C <= 16 else (32, 32)
    fn, _ = kernels.nchw_to_nhwc(elt, TC, TX, 256,
                                 vecr=X % 2 == 0, vecw=C % 2 == 0)
    fn(xc, out, _pk(C, X),
       threads=(256 * ((X + TX - 1) // TX), (C + TC - 1) // TC, N),
       group_size=(256, 1, 1))
    return out


_WCACHE: dict = {}
_WCACHE_CAP = 256


def _weights_transform(weight, perm):
    key = (id(weight), perm)
    hit = _WCACHE.get(key)
    if hit is not None:
        src, ver, wt = hit
        if src is weight and ver == weight._version:
            return wt
    wt = weight.permute(*perm).contiguous()
    if len(_WCACHE) >= _WCACHE_CAP:
        _WCACHE.pop(next(iter(_WCACHE)))
    _WCACHE[key] = (weight, weight._version, wt)
    return wt


def _out_channels_last(x):
    if x.dim() == 4:
        return x.is_contiguous(memory_format=torch.channels_last) and \
            not x.is_contiguous()
    if x.dim() == 5:
        return x.is_contiguous(memory_format=torch.channels_last_3d) and \
            not x.is_contiguous()
    return False


def _dw_conv(input, weight, bias, khw, sxy, dxy, pxy, shape):
    NB, C, H, W, O, HO, WO = shape
    KH, KW = khw
    SY, SX = sxy
    DY, DX = dxy
    PADY, PADX = pxy
    if input.dim() == 3:
        H, W, HO, WO = W, H, WO, HO
        KH, KW = KW, KH
        SY, SX = SX, SY
        DY, DX = DX, DY
        PADY, PADX = PADX, PADY
    dtype = input.dtype
    elt = _PROFILE[dtype][0]
    nd = input.dim() - 2
    channels_last = _out_channels_last(input)
    b = bias.contiguous() if bias is not None else None
    OPT = 4
    fn, _ = kernels.dw_conv(elt, KH, KW, SX, SY, DX, DY,
                            bias=b is not None, nhwc=channels_last, OPT=OPT)
    if channels_last:
        act = input.permute(0, 2, 3, 1)
        wts = _weights_transform(weight, (2, 3, 1, 0))
        out = torch.empty((NB, O, HO, WO), dtype=dtype, device=input.device,
                          memory_format=torch.channels_last)
        out_mem = out.permute(0, 2, 3, 1)
        threads = (O, WO, ((HO + OPT - 1) // OPT) * NB)
        group = (min(64, max(32, O)), 4, 1) if O >= 32 else (O, 8, 4)
    else:
        act = input.contiguous()
        wts = weight.contiguous()
        oshape = (NB, O, HO, WO) if nd == 2 else (NB, O, WO)
        out = torch.empty(oshape, dtype=dtype, device=input.device)
        out_mem = out
        threads = ((WO + OPT - 1) // OPT, HO, NB * O)
        group = (256, 1, 1) if HO == 1 else (32, 8, 1)
    dims = _pk(C, H, W, O, HO, WO, NB, PADX, PADY, O // C)
    args = [act, wts, out_mem, dims] + ([b] if b is not None else [])
    fn(*args, threads=threads, group_size=group)
    return out


def _conv_1x1_gemm(nd, input, weight, bias, NB, C, O, ospatial):
    X = 1
    for s in ospatial:
        X *= s
    if _out_channels_last(input):
        x2 = input.permute(0, *range(2, input.dim()), 1).reshape(NB * X, C)
        w2 = weight.reshape(O, C).t()
        y = addmm(bias, x2, w2) if bias is not None else matmul(x2, w2)
        y = y.view(NB, *ospatial, O)
        return y.permute(0, nd + 1, *range(1, nd + 1))
    xc = input.contiguous().view(NB, C, X)
    w3 = weight.reshape(1, O, C).expand(NB, O, C)
    if bias is not None:
        y = baddbmm(bias.view(1, O, 1), w3, xc)
    else:
        y = bmm(w3, xc)
    return y.view(NB, O, *ospatial)


def _conv_nd(nd, input, weight, bias, stride, padding, dilation, groups,
             fkey=None):
    if input.dim() == nd + 1:
        out = _conv_nd(nd, input.unsqueeze(0), weight, bias, stride, padding,
                       dilation, groups)
        return out.squeeze(0)

    stride = _ntuple(stride, nd)
    dilation = _ntuple(dilation, nd)
    same_pad = padding == "same"
    if isinstance(padding, str):
        if padding == "valid":
            padding = (0,) * nd
        elif padding == "same":
            padding = _same_padding(weight.shape[2:], dilation, stride, nd)
        else:
            raise ValueError(f"unknown padding {padding!r}")
    else:
        padding = _ntuple(padding, nd)

    NB, C = input.shape[0], input.shape[1]
    O = weight.shape[0]
    kdims = tuple(weight.shape[2:])
    ispatial = tuple(input.shape[2:])
    if same_pad:
        ospatial = ispatial
    else:
        ospatial = tuple(
            (ispatial[i] + 2 * padding[i] - dilation[i] * (kdims[i] - 1) - 1)
            // stride[i] + 1
            for i in range(nd))

    dtype = input.dtype
    fallback = (
        dtype not in _PROFILE
        or dtype is torch.float16
        or input.device.type != "mps"
        or C % groups or O % groups
        or any(s <= 0 for s in ospatial)
    )
    if fallback:
        import torch.nn.functional as F
        f = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
        return f(input, weight, bias, stride, padding, dilation, groups)


    if (groups == 1 and all(k == 1 for k in kdims) and all(s == 1 for s in stride)
            and all(p == 0 for p in padding)
            and not (_out_channels_last(input) and O < 96)):
        return _conv_1x1_gemm(nd, input, weight, bias, NB, C, O, ospatial)


    if nd == 1:
        (L,), (LO,), (k,) = ispatial, ospatial, kdims
        H, W, HO, WO = L, 1, LO, 1
        KH, KW = k, 1
        SY, SX = stride[0], 1
        DY, DX = dilation[0], 1
        PADY, PADX = padding[0], 0
        conv3d_p, D, DO, PADZ = None, 1, 1, 0
    elif nd == 2:
        (H, W), (HO, WO) = ispatial, ospatial
        KH, KW = kdims
        SY, SX = stride
        DY, DX = dilation
        PADY, PADX = padding
        conv3d_p, D, DO, PADZ = None, 1, 1, 0
    else:
        (D, H, W), (DO, HO, WO) = ispatial, ospatial
        KD, KH, KW = kdims
        SZ, SY, SX = stride
        DZ, DY, DX = dilation
        PADZ, PADY, PADX = padding
        conv3d_p = (KD, SZ, DZ)


    if groups == C and nd <= 2 and weight.shape[1] == 1:
        return _dw_conv(input, weight, bias, (KH, KW), (SY, SX), (DY, DX),
                        (PADY, PADX), (NB, C, H, W, O, HO, WO))

    srcw = W if W > _SRC_HINT else _SRC_HINT
    srch = H if H > _SRC_HINT else _SRC_HINT

    channels_last = _out_channels_last(input)
    d_nchw = not channels_last
    key = (dtype, nd, C, H, W, O, HO, WO, NB, kdims, stride, padding, dilation,
           groups, bias is not None, d_nchw, D)
    plan = _CONV_PLAN.get(key)
    if plan == "torch":
        if fkey is not None:
            _TORCH_ROUTE.add(fkey)
        import torch.nn.functional as F
        tf = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
        return tf(input, weight, bias, stride, padding, dilation, groups)
    if plan is not None and getattr(plan, "family", None) == "sg":
        out = torch.empty((NB, O, *ospatial), dtype=dtype, device=input.device)
        plan.run(input.contiguous(), _weights_transform(weight, (2, 0, 1)), out,
                 bias.contiguous() if bias is not None else None)
        return out

    act = _to_nhwc(input)
    if nd == 1:
        w_hwio = _weights_transform(weight, (2, 1, 0))
    elif nd == 2:
        w_hwio = _weights_transform(weight, (2, 3, 1, 0))
    else:
        w_hwio = _weights_transform(weight, (2, 3, 4, 1, 0))
    b = bias.contiguous() if bias is not None else None

    if channels_last:
        mf = torch.channels_last if nd == 2 else torch.channels_last_3d
        out = torch.empty((NB, O, *ospatial), dtype=dtype, device=input.device,
                          memory_format=mf)
        out_mem = out.permute(0, 2, 3, 1) if nd == 2 else out.permute(0, 2, 3, 4, 1)
    else:
        out = torch.empty((NB, O, *ospatial), dtype=dtype, device=input.device)
        out_mem = out

    shape = (C, H, W, O, HO, WO, NB, PADY, PADX, D, DO, PADZ)
    if plan is None:
        sg_ok = (nd == 1 and groups == 1 and stride[0] == 1
                 and padding[0] < 64 and kdims[0] >= 2)
        xr = input.contiguous() if sg_ok else None
        w_sg = _weights_transform(weight, (2, 0, 1)) if sg_ok else None

        def build(tile):
            if tile[0] == "sg":
                return _build_sg_plan(dtype, tile[1:], KH, DY,
                                      (C, H, NB, O, HO, padding[0]), b is not None)
            if tile[0] == "bw":
                return _build_bw_plan(dtype, tile[1:], KH, SY, DY,
                                      (C, H, NB, O, HO, padding[0]), b is not None)
            return _build_plan(dtype, tile, (KH, KW), (SY, SX), (DY, DX), shape,
                               groups, b is not None, d_nchw, conv3d_p, srcw, srch)

        def runner(p):
            if getattr(p, "family", None) == "sg":
                return lambda: p.run(xr, w_sg, out_mem, b)
            if d_nchw:
                return lambda: p.run(_to_nhwc(input), w_hwio, out_mem, b)
            return lambda: p.run(act, w_hwio, out_mem, b)

        cands = _tile_candidates(O, HO, WO, C, groups)
        if nd == 1 and groups == 1 and NB >= 2:
            cands = _bw_tile_candidates(NB, O) + cands
        if sg_ok:
            cands = _SG_TILES + cands
        if _AUTOTUNE and len(cands) > 1:
            flops = 2.0 * NB * DO * HO * WO * O * (C // groups) * 1
            for k_ in kdims:
                flops *= k_
            import torch.nn.functional as F
            tf = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
            torch_probe = (lambda: tf(input, weight, bias, stride, padding,
                                      dilation, groups))
            built = [(p, runner(p), t) for t in cands for p in (build(t),)]
            plan, tile = _autotune_conv(built, flops, torch_probe)
        else:
            plan, tile = build(cands[0]), cands[0]
        _CONV_PLAN[key] = plan
        _CONV_TILE[key] = tile
    if plan == "torch":
        if fkey is not None:
            _TORCH_ROUTE.add(fkey)
        import torch.nn.functional as F
        tf = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
        return tf(input, weight, bias, stride, padding, dilation, groups)
    if getattr(plan, "family", None) == "sg":
        plan.run(input.contiguous(), _weights_transform(weight, (2, 0, 1)),
                 out_mem, b)
        return out
    plan.run(act, w_hwio, out_mem, b)
    return out


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    fkey = _fast_key(1, input, weight, bias, stride, padding, dilation, groups)
    if fkey in _TORCH_ROUTE:
        import torch.nn.functional as F
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)
    return _conv_nd(1, input, weight, bias, stride, padding, dilation, groups, fkey)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    fkey = _fast_key(2, input, weight, bias, stride, padding, dilation, groups)
    if fkey in _TORCH_ROUTE:
        import torch.nn.functional as F
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return _conv_nd(2, input, weight, bias, stride, padding, dilation, groups, fkey)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    fkey = _fast_key(3, input, weight, bias, stride, padding, dilation, groups)
    if fkey in _TORCH_ROUTE:
        import torch.nn.functional as F
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)
    return _conv_nd(3, input, weight, bias, stride, padding, dilation, groups, fkey)
