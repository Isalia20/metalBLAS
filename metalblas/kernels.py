"""Loads and compiles the metalBLAS Metal kernels from shaders/.

Per call: inline the binder's local #includes, enable one family via
-DMB_BUILD_<NAME>, substitute its __PARAM__ tile tokens, compile (cached).
"""
from __future__ import annotations

import os
import re
import functools

import torch

_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
_LOCAL_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.M)


@functools.lru_cache(maxsize=None)
def _read(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _inline_includes(path: str) -> str:
    """Inline local #include "x" (recursively); leave system #include <...> for the compiler."""
    base = os.path.dirname(path)
    return _LOCAL_INCLUDE.sub(
        lambda m: _inline_includes(os.path.join(base, m.group(1))),
        _read(path),
    )


@functools.lru_cache(maxsize=1)
def _binder_source() -> str:
    return _inline_includes(os.path.join(_SHADER_DIR, "metalblas.metal"))


def _subst(src: str, **kw) -> str:
    for k, v in kw.items():
        src = src.replace("__" + k + "__", str(v))
    return src


def _build(build_flag: str, **params) -> str:
    """Assemble one kernel's source: enable its build flag, inline the shaders, substitute params."""
    return _subst(f"#define {build_flag} 1\n" + _binder_source(), **params)


@functools.lru_cache(maxsize=None)
def _compile(src: str):
    return torch.mps.compile_shader(src)


@functools.lru_cache(maxsize=1)
def has_metal4() -> bool:
    """True if the Metal 4 cooperative-tensor headers are available (macOS 26+).

    The m5_tensor / m5_gemm / splitk / conv1x1 kernels #include
    <metal_cooperative_tensor> and use mpp::tensor_ops, which only ship with
    Metal 4. On older macOS, compile_shader fails with "file not found". Probe
    once with a minimal kernel and cache the result so dispatch can route to the
    simd/gemv fallbacks instead of emitting a cryptic compile error.
    """
    probe = "#include <metal_cooperative_tensor>\n[[kernel]] void _mb_metal4_probe() {}\n"
    try:
        torch.mps.compile_shader(probe)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def simd_gemm(in_t: str, acc_t: str, out_t: str,
              BM: int, BN: int, BK: int, WM: int, WN: int,
              trans_a: bool, trans_b: bool,
              mn_aligned: bool, k_aligned: bool,
              swizzle_log: int = 0):
    src = _build(
        "MB_BUILD_SIMD_GEMM",
        IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
        BM=BM, BN=BN, BK=BK, WM=WM, WN=WN,
        TRANS_A=int(trans_a), TRANS_B=int(trans_b),
        MN_ALIGNED=int(mn_aligned), K_ALIGNED=int(k_aligned),
        OUT_IS_ACC=int(out_t == acc_t),
        SWIZZLE_LOG=swizzle_log,
    )
    lib = _compile(src)
    return lib.simd_gemm, src


@functools.lru_cache(maxsize=None)
def m5_gemm(in_t: str, acc_t: str, out_t: str,
            BM: int, BN: int, BK: int, WM: int, WN: int,
            trans_a: bool, trans_b: bool,
            mn_aligned: bool, k_aligned: bool,
            relaxed: bool = True,
            swizzle_log: int = 0,
            dbuf: bool = False,
            pad: int | None = None):
    # pad defaults to 16/sizeof(IN_T) for VecF alignment (0 is OK when BK/BN are VEC-aligned).
    if pad is None:
        in_bytes = 4 if in_t == "float" else 2
        pad = 16 // in_bytes
    src = _build(
        "MB_BUILD_M5_GEMM",
        IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
        BM=BM, BN=BN, BK=BK, WM=WM, WN=WN,
        TRANS_A=int(trans_a), TRANS_B=int(trans_b),
        MN_ALIGNED=int(mn_aligned), K_ALIGNED=int(k_aligned),
        RELAXED=("true" if relaxed else "false"),
        SWIZZLE_LOG=swizzle_log,
        DBUF=int(dbuf),
        PAD=int(pad),
    )
    lib = _compile(src)
    return lib.m5_gemm, src


@functools.lru_cache(maxsize=None)
def m5_tensor_gemm(in_t: str, out_t: str,
                   BM: int, BN: int, NSG: int,
                   trans_a: bool, trans_b: bool,
                   relaxed: bool = True,
                   swizzle_log: int = 0,
                   mn_aligned: bool = False):
    # Static-extent slices only for non-transposed (the orientation auto-dispatch routes here).
    static_slice = (not trans_a) and (not trans_b)
    src = _build(
        "MB_BUILD_M5_TENSOR",
        IN_T=in_t, OUT_T=out_t,
        BM=BM, BN=BN, NSG=NSG,
        TRANS_A=("true" if trans_a else "false"),
        TRANS_B=("true" if trans_b else "false"),
        RELAXED=("true" if relaxed else "false"),
        SWIZZLE_LOG=swizzle_log,
        MN_ALIGNED=int(mn_aligned),
        STATIC_SLICE=int(static_slice),
    )
    lib = _compile(src)
    return lib.m5_tensor_gemm, src


@functools.lru_cache(maxsize=None)
def splitk_gemm(in_t: str, out_t: str, BM: int, BN: int, NSG: int, KCHUNK: int,
                relaxed: bool = True):
    """Split-K m5_tensor GEMM -> (splitk_fn, reduce_fn). KCHUNK must divide K (caller guarantees)."""
    src = _build(
        "MB_BUILD_SPLITK",
        IN_T=in_t, OUT_T=out_t,
        BM=BM, BN=BN, NSG=NSG, KCHUNK=KCHUNK,
        RELAXED=("true" if relaxed else "false"),
    )
    lib = _compile(src)
    return lib.splitk_gemm, lib.splitk_reduce


@functools.lru_cache(maxsize=None)
def conv1x1_gemm(in_t: str, out_t: str, BMW: int, BNO: int, NSG: int, K: int,
                 relaxed: bool = True):
    """1x1-conv GEMM for very-thin-N (shaders/conv1x1.h). K is baked into the descriptor, so it's per-K."""
    src = _build(
        "MB_BUILD_CONV1X1",
        IN_T=in_t, OUT_T=out_t,
        BMW=BMW, BNO=BNO, NSG=NSG, KCONST=K,
    )
    return _compile(src).conv1x1_gemm, src


@functools.lru_cache(maxsize=None)
def gemv_nt(in_t: str, acc_t: str, out_t: str, ROWS_PER_SG: int = 1, NWARPS: int = 4,
            VEC: int = 1, red_tg: bool = False):
    # red_tg: reduce via threadgroup mem instead of simd_sum (int64: no simd_sum(long)).
    src = _build("MB_BUILD_GEMV_NT", IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                 ROWS_PER_SG=ROWS_PER_SG, NWARPS=NWARPS, VEC=VEC, RED_TG=int(red_tg))
    return _compile(src).gemv_nt, src


@functools.lru_cache(maxsize=None)
def gemv_t(in_t: str, acc_t: str, out_t: str, BLOCK_N: int = 32, NWARPS: int = 4,
           VEC: int = 1):
    # Each lane owns VEC columns, so a threadgroup spans BLOCK_N == 32*VEC cols.
    assert BLOCK_N == 32 * VEC, f"BLOCK_N ({BLOCK_N}) must equal 32*VEC ({32*VEC})"
    src = _build("MB_BUILD_GEMV_T", IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                 BLOCK_N=BLOCK_N, NWARPS=NWARPS, VEC=VEC)
    return _compile(src).gemv_t, src


@functools.lru_cache(maxsize=None)
def cgemv_t(c2_t: str, acc2_t: str, r_t: str, BLOCK_N: int = 32, NWARPS: int = 8):
    src = _build("MB_BUILD_CGEMV_T", C2=c2_t, ACC2=acc2_t, R=r_t,
                 BLOCK_N=BLOCK_N, NWARPS=NWARPS)
    return _compile(src).cgemv_t, src


@functools.lru_cache(maxsize=None)
def cgemv_nt(c2_t: str, acc2_t: str, r_t: str, NWARPS: int = 4):
    src = _build("MB_BUILD_CGEMV_NT", C2=c2_t, ACC2=acc2_t, R=r_t, NWARPS=NWARPS)
    return _compile(src).cgemv_nt, src


@functools.lru_cache(maxsize=None)
def complex_pack(c2_t: str, r_t: str):
    """-> (split_fn, combine_fn) for the given complex element type (float2/half2)."""
    src = _build("MB_BUILD_COMPLEX_PACK", C2=c2_t, R=r_t)
    lib = _compile(src)
    return lib.complex_split, lib.complex_combine


@functools.lru_cache(maxsize=None)
def int_gemm(in_t: str, acc_t: str, out_t: str, BM: int, BN: int, BK: int,
             TX: int, TY: int, trans_a: bool, trans_b: bool):
    """Register-tiled integer GEMM (simdgroup_matrix / the tensor unit are float-only)."""
    src = _build("MB_BUILD_INT_GEMM", IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                 BM=BM, BN=BN, BK=BK, TX=TX, TY=TY,
                 TRANS_A=int(trans_a), TRANS_B=int(trans_b))
    return _compile(src).int_gemm, src
