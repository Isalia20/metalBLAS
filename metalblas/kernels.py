"""Assemble and cache Metal kernel variants."""
from __future__ import annotations

from functools import lru_cache
import os
import re

import torch

_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
_LOCAL_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.M)


@lru_cache(maxsize=None)
def _read(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _inline_includes(path: str) -> str:
    """Inline local includes and leave system includes for Metal."""
    base = os.path.dirname(path)
    return _LOCAL_INCLUDE.sub(
        lambda m: _inline_includes(os.path.join(base, m.group(1))),
        _read(path),
    )


@lru_cache(maxsize=1)
def _binder_source() -> str:
    return _inline_includes(os.path.join(_SHADER_DIR, "metalblas.metal"))


def _subst(src: str, **kw) -> str:
    for k, v in kw.items():
        src = src.replace("__" + k + "__", str(v))
    return src


def _build(build_flag: str, *, defines=None, **params) -> str:
    """Enable one shader family and substitute its compile-time parameters."""
    prelude = f"#define {build_flag} 1\n"
    if defines:
        prelude += "".join(f"#define {k} {int(v)}\n" for k, v in defines.items())
    return _subst(prelude + _binder_source(), **params)


def _defines(epilogue=False, beta_nz=True, alpha_nz=True, **flags):
    defines = ({"EPILOGUE": 1, "BETA_NZ": int(beta_nz), "ALPHA_NZ": int(alpha_nz)}
               if epilogue else {})
    defines.update({name: 1 for name, enabled in flags.items() if enabled})
    return defines or None


@lru_cache(maxsize=None)
def _compile(src: str):
    return torch.mps.compile_shader(src)


def _kernel(name, build_flag, *, defines=None, **params):
    src = _build(build_flag, defines=defines, **params)
    return getattr(_compile(src), name), src


@lru_cache(maxsize=1)
def has_metal4() -> bool:
    """Return whether Metal cooperative tensors can be compiled."""
    probe = "#include <metal_cooperative_tensor>\n[[kernel]] void _mb_metal4_probe() {}\n"
    try:
        torch.mps.compile_shader(probe)
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def simd_gemm(in_t: str, acc_t: str, out_t: str,
              BM: int, BN: int, BK: int, WM: int, WN: int,
              trans_a: bool, trans_b: bool,
              mn_aligned: bool, k_aligned: bool,
              swizzle_log: int = 0,
              epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    return _kernel(
        "simd_gemm", "MB_BUILD_SIMD_GEMM",
        defines=_defines(epilogue, beta_nz, alpha_nz),
        IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
        BM=BM, BN=BN, BK=BK, WM=WM, WN=WN,
        TRANS_A=int(trans_a), TRANS_B=int(trans_b),
        MN_ALIGNED=int(mn_aligned), K_ALIGNED=int(k_aligned),
        OUT_IS_ACC=int(out_t == acc_t),
        SWIZZLE_LOG=swizzle_log,
    )


@lru_cache(maxsize=None)
def mpp_gemm(in_t: str, acc_t: str, out_t: str,
            BM: int, BN: int, BK: int, WM: int, WN: int,
            trans_a: bool, trans_b: bool,
            mn_aligned: bool, k_aligned: bool,
            relaxed: bool = True,
            swizzle_log: int = 0,
            dbuf: bool = False,
            pad: int | None = None,
            epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    if pad is None:
        in_bytes = 4 if in_t == "float" else 2
        pad = 16 // in_bytes
    return _kernel(
        "mpp_gemm", "MB_BUILD_MPP_GEMM",
        defines=_defines(epilogue, beta_nz, alpha_nz),
        IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
        BM=BM, BN=BN, BK=BK, WM=WM, WN=WN,
        TRANS_A=int(trans_a), TRANS_B=int(trans_b),
        MN_ALIGNED=int(mn_aligned), K_ALIGNED=int(k_aligned),
        RELAXED=("true" if relaxed else "false"),
        SWIZZLE_LOG=swizzle_log,
        DBUF=int(dbuf),
        PAD=int(pad),
    )


@lru_cache(maxsize=None)
def mpp_tensor_gemm(in_t: str, out_t: str,
                   BM: int, BN: int, NSG: int,
                   trans_a: bool, trans_b: bool,
                   relaxed: bool = True,
                   swizzle_log: int = 0,
                   mn_aligned: bool = False,
                   epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True,
                   batched: bool = False):
    static_slice = (not trans_a) and (not trans_b)
    defines = _defines(epilogue, beta_nz, alpha_nz, BATCHED=batched)
    return _kernel(
        "mpp_tensor_gemm", "MB_BUILD_MPP_TENSOR",
        defines=defines,
        IN_T=in_t, OUT_T=out_t,
        BM=BM, BN=BN, NSG=NSG,
        TRANS_A=("true" if trans_a else "false"),
        TRANS_B=("true" if trans_b else "false"),
        RELAXED=("true" if relaxed else "false"),
        SWIZZLE_LOG=swizzle_log,
        MN_ALIGNED=int(mn_aligned),
        STATIC_SLICE=int(static_slice),
    )


@lru_cache(maxsize=None)
def splitk_gemm(in_t: str, out_t: str, BM: int, BN: int, NSG: int, KCHUNK: int,
                relaxed: bool = True):
    src = _build(
        "MB_BUILD_SPLITK",
        IN_T=in_t, OUT_T=out_t,
        BM=BM, BN=BN, NSG=NSG, KCHUNK=KCHUNK,
        RELAXED=("true" if relaxed else "false"),
    )
    lib = _compile(src)
    return lib.splitk_gemm, lib.splitk_reduce


@lru_cache(maxsize=None)
def conv1x1_gemm(in_t: str, out_t: str, BMW: int, BNO: int, NSG: int, K: int):
    return _kernel(
        "conv1x1_gemm", "MB_BUILD_CONV1X1",
        IN_T=in_t, OUT_T=out_t,
        BMW=BMW, BNO=BNO, NSG=NSG, KCONST=K,
    )


@lru_cache(maxsize=None)
def sgpipe_gemm(in_t: str, out_t: str, SGM: int, SGN: int, KC: int,
                NSGX: int, NSGY: int, GK: int = 0, GM: int = 0, GN: int = 0):
    return _kernel(
        "sgpipe_gemm", "MB_BUILD_MPP_SGPIPE",
        IN_T=in_t, OUT_T=out_t,
        SGM=SGM, SGN=SGN, KC=KC, NSGX=NSGX, NSGY=NSGY, GK=GK, GM=GM, GN=GN,
    )


@lru_cache(maxsize=None)
def flipt_gemm(in_t: str, out_t: str, BM: int, BN: int, NSG: int,
               KC: int = 0, PFD: int = 0):
    return _kernel(
        "flipt_gemm", "MB_BUILD_FLIPT",
        IN_T=in_t, OUT_T=out_t,
        BM=BM, BN=BN, NSG=NSG, KC=KC, PFD=PFD,
    )


@lru_cache(maxsize=None)
def gemv_nt(in_t: str, acc_t: str, out_t: str, ROWS_PER_SG: int = 1, NWARPS: int = 4,
            VEC: int = 1, red_tg: bool = False,
            epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    return _kernel("gemv_nt", "MB_BUILD_GEMV_NT", defines=_defines(epilogue, beta_nz, alpha_nz),
                   IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                   ROWS_PER_SG=ROWS_PER_SG, NWARPS=NWARPS, VEC=VEC, RED_TG=int(red_tg))


@lru_cache(maxsize=None)
def gemv_t(in_t: str, acc_t: str, out_t: str, BLOCK_N: int = 32, NWARPS: int = 4,
           VEC: int = 1,
           epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    assert BLOCK_N == 32 * VEC, f"BLOCK_N ({BLOCK_N}) must equal 32*VEC ({32*VEC})"
    return _kernel("gemv_t", "MB_BUILD_GEMV_T", defines=_defines(epilogue, beta_nz, alpha_nz),
                   IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                   BLOCK_N=BLOCK_N, NWARPS=NWARPS, VEC=VEC)


@lru_cache(maxsize=None)
def gemv_bt(in_t: str, acc_t: str, out_t: str, MROWS: int,
            BLOCK_N: int = 64, NWARPS: int = 8, VEC: int = 2,
            epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True,
            batched: bool = False, trans_b: bool = False, NCOLS: int = 1, trans_a: bool = False):
    assert BLOCK_N == 32 * VEC, f"BLOCK_N ({BLOCK_N}) must equal 32*VEC ({32*VEC})"
    defines = _defines(epilogue, beta_nz, alpha_nz,
                       BATCHED=batched, TRANS_B=trans_b, TRANS_A=trans_a)
    return _kernel("gemv_bt", "MB_BUILD_GEMV_BT", defines=defines,
                   IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                   MROWS=MROWS, BLOCK_N=BLOCK_N, NWARPS=NWARPS, VEC=VEC, NCOLS=NCOLS)


@lru_cache(maxsize=None)
def cgemv_t(c2_t: str, acc2_t: str, r_t: str, BLOCK_N: int = 32, NWARPS: int = 8,
            epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    return _kernel("cgemv_t", "MB_BUILD_CGEMV_T", defines=_defines(epilogue, beta_nz, alpha_nz),
                   C2=c2_t, ACC2=acc2_t, R=r_t, BLOCK_N=BLOCK_N, NWARPS=NWARPS)


@lru_cache(maxsize=None)
def cgemv_nt(c2_t: str, acc2_t: str, r_t: str, NWARPS: int = 4,
             epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    return _kernel("cgemv_nt", "MB_BUILD_CGEMV_NT", defines=_defines(epilogue, beta_nz, alpha_nz),
                   C2=c2_t, ACC2=acc2_t, R=r_t, NWARPS=NWARPS)


@lru_cache(maxsize=None)
def complex_pack(c2_t: str, r_t: str,
                 epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True):
    src = _build("MB_BUILD_COMPLEX_PACK", defines=_defines(epilogue, beta_nz, alpha_nz),
                 C2=c2_t, R=r_t)
    lib = _compile(src)
    return lib.complex_split, lib.complex_combine


@lru_cache(maxsize=None)
def int_gemm(in_t: str, acc_t: str, out_t: str, BM: int, BN: int, BK: int,
             TX: int, TY: int, trans_a: bool, trans_b: bool,
             epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True,
             batched: bool = False):
    defines = _defines(epilogue, beta_nz, alpha_nz, BATCHED=batched)
    return _kernel("int_gemm", "MB_BUILD_INT_GEMM", defines=defines,
                   IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                   BM=BM, BN=BN, BK=BK, TX=TX, TY=TY,
                   TRANS_A=int(trans_a), TRANS_B=int(trans_b))
