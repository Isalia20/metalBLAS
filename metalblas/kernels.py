"""Assemble and cache Metal kernel variants."""
from __future__ import annotations

from functools import lru_cache
import os
import re
import struct

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


def _mps_epilogue_defines(in_t, epilogue):
    if epilogue is None:
        return None
    beta, alpha = epilogue
    dtype = {"float": torch.float32, "half": torch.float16, "bfloat": torch.bfloat16}[in_t]

    def bits(value):
        # MPSGraph's constant nodes round to the input dtype before the
        # MPSCore multiplication_f callback consumes the value as float.
        value = torch.tensor(value, dtype=dtype, device="cpu").item()
        return struct.unpack("<I", struct.pack("<f", value))[0]

    return {"MB_MPS_EPILOGUE": 1, "MB_MPS_ALPHA_BITS": bits(alpha),
            "MB_MPS_BETA_BITS": bits(beta), "MB_MPS_BETA_NZ": beta != 0}


@lru_cache(maxsize=None)
def _compile(src: str):
    from .mps_submission import Library
    return Library(torch.mps.compile_shader(src))


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
def a18_gemm(in_t: str, out_t: str,
             UM: int, UN: int, SM: int, SN: int, KS: int, KC: int,
             BIT: int, PACE: int, RW: int, LIN: int,
             trans_a: bool, trans_b: bool,
             mn_aligned: bool, k_aligned: bool,
             epilogue: bool = False, beta_nz: bool = True, alpha_nz: bool = True,
             batched: bool = False, source_epilogue=None,
             global_splits: int = 1, global_atomic: bool = False):
    """GEMM with the structure of MPSNDArray's gemm_a18 (see shaders/a18_gemm.h)."""
    defines = _defines(epilogue, beta_nz, alpha_nz, BATCHED=batched) or {}
    defines.update(_mps_epilogue_defines(in_t, source_epilogue) or {})
    defines.update(GLOBAL_SPLITS=global_splits, GLOBAL_ATOMIC=int(global_atomic))
    return _kernel(
        "a18_gemm", "MB_BUILD_A18",
        defines=defines,
        IN_T=in_t, OUT_T=out_t,
        UM=UM, UN=UN, SM=SM, SN=SN, KS=KS, KC=KC, BIT=BIT, PACE=PACE, RW=RW, LIN=LIN,
        TRANS_A=int(trans_a), TRANS_B=int(trans_b),
        MN_ALIGNED=int(mn_aligned), K_ALIGNED=int(k_aligned),
    )


@lru_cache(maxsize=None)
def a18_global_helpers(out_t, splits, epilogue=None):
    source = _build("MB_BUILD_A18_GLOBAL", defines=_mps_epilogue_defines(out_t, epilogue),
                    OUT_T=out_t, GLOBAL_SPLITS=splits)
    library = _compile(source)
    return library.a18_global_initialize, library.a18_global_reduce


@lru_cache(maxsize=None)
def a16_gemm(in_t: str, out_t: str, BM: int, BN: int, BK: int, SM: int, SN: int,
             stage_a: bool, trans_a: bool, trans_b: bool,
             mn_aligned: bool, k_aligned: bool, splits: int, kchunk: int,
             dim16: bool, reduce_vec: int, edge_tm: int, edge_tn: int,
             wide_index: bool = False, epilogue=None):
    """A16 fragment/staging schedule selected by MPSNDArray's A14 encoder."""
    src = _build(
        "MB_BUILD_A16", defines=_mps_epilogue_defines(in_t, epilogue),
        IN_T=in_t, OUT_T="float" if splits > 1 else out_t,
        FINAL_T=out_t, BM=BM, BN=BN, BK=BK, SM=SM, SN=SN,
        STAGE_A=int(stage_a), TRANS_A=int(trans_a), TRANS_B=int(trans_b),
        MN_ALIGNED=int(mn_aligned), K_ALIGNED=int(k_aligned),
        SPLITS=splits, KCHUNK=kchunk, DIM16=int(dim16),
        REDUCE_VEC=reduce_vec, EDGE_TM=edge_tm, EDGE_TN=edge_tn,
        INDEX_T="ulong" if wide_index else "uint",
    )
    lib = _compile(src)
    return lib.a16_gemm, lib.a16_reduce


@lru_cache(maxsize=None)
def nna14_gemm(in_t: str, out_t: str, BM: int, BN: int, BK: int, SM: int, SN: int,
               trans_a: bool, trans_b: bool, mn_aligned: bool,
               splits: int, kchunk: int, linear: bool, reduce_vec: int, dim16: bool,
               edge_tm: int, edge_tn: int, k_tail: bool, wide_index: bool, epilogue=None):
    """NNA14 async-copy/fragment schedule selected by MPSNDArray's A14 encoder."""
    src = _build(
        "MB_BUILD_NNA14", defines=_mps_epilogue_defines(in_t, epilogue),
        IN_T=in_t, OUT_T="float" if splits > 1 else out_t,
        FINAL_T=out_t, BM=BM, BN=BN, BK=BK, SM=SM, SN=SN,
        TRANS_A=int(trans_a), TRANS_B=int(trans_b), MN_ALIGNED=int(mn_aligned),
        SPLITS=splits, KCHUNK=kchunk, LINEAR=int(linear), REDUCE_VEC=reduce_vec,
        DIM16=int(dim16), EDGE_TM=edge_tm, EDGE_TN=edge_tn, K_TAIL=int(k_tail),
        INDEX_T="ulong" if wide_index else "uint",
    )
    lib = _compile(src)
    return lib.nna14_gemm, lib.nna14_reduce


@lru_cache(maxsize=None)
def mps_lora(rows: int, row_tile: int, unroll_k: int, nsimd: int, vec_b: bool,
             k_mod4: int, k_loops: int, tiled_tail: bool, tg_tail: bool, epilogue=None):
    """LoRA row-prefetch and K-reduction schedule from Apple's vector encoder."""
    return _kernel(
        "mps_lora", "MB_BUILD_MPS_LORA", defines=_mps_epilogue_defines("half", epilogue),
        ROWS=rows, ROW_TILE=row_tile,
        UNROLL_K=unroll_k, NSIMD=nsimd, VEC_B=int(vec_b), K_MOD4=k_mod4,
        K_LOOPS=k_loops, TILED_TAIL=int(tiled_tail), TG_TAIL=int(tg_tail),
    )[0]


@lru_cache(maxsize=None)
def mps_vector(in_t, r, gx, gy, flat, trans_b, trans_out, rows, unroll_k,
                groups, full_groups, partial, k_mod4, epilogue=None):
    """Ordinary MPS vector schedules, including the TransA and 16x16 paths."""
    return _kernel(
        "mps_vector", "MB_BUILD_MPS_VECTOR", defines=_mps_epilogue_defines(in_t, epilogue),
        IN_T=in_t,
        LOAD4_T={"float": "float4", "half": "packed_half4", "bfloat": "bfloat4"}[in_t],
        R=r, GX=gx, GY=gy, FLAT=int(flat), TRANS_B=int(trans_b), TRANS_OUT=int(trans_out),
        ROWS=rows, UNROLL_K=unroll_k, GROUPS=groups, FULL_GROUPS=full_groups,
        PARTIAL=int(partial), K_MOD4=k_mod4,
    )[0]


@lru_cache(maxsize=None)
def mps_batch_reduce(in_t, out_t, gx, gy, gz, epilogue=None):
    return _kernel("mps_batch_reduce", "MB_BUILD_MPS_REDUCTION",
                   defines=_mps_epilogue_defines(out_t, epilogue),
                   IN_T=in_t, OUT_T=out_t, GX=gx, GY=gy, GZ=gz)[0]


@lru_cache(maxsize=None)
def mps_identity(in_t, x_map, transpose, results, gx):
    return _kernel("mps_identity", "MB_BUILD_MPS_IDENTITY", IN_T=in_t,
                   X_MAP=x_map, TRANSPOSE_COPY=int(transpose),
                   COPY_NR=results, COPY_GX=gx)[0]


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
