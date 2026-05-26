"""Metal shader sources for high-performance matmul on Apple Silicon."""
from __future__ import annotations

import functools
import torch


# ---------------------------------------------------------------------------
# simdgroup_matrix tiled GEMM (static threadgroup memory)
# ---------------------------------------------------------------------------
SIMD_GEMM_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define BM          __BM__
#define BN          __BN__
#define BK          __BK__
#define WM          __WM__
#define WN          __WN__
#define TRANS_A     __TRANS_A__
#define TRANS_B     __TRANS_B__
#define MN_ALIGNED  __MN_ALIGNED__
#define K_ALIGNED   __K_ALIGNED__
#define SWIZZLE_LOG __SWIZZLE_LOG__
#define OUT_IS_ACC  __OUT_IS_ACC__

#define SG_SIZE     32
#define WARPS       (WM * WN)
#define TGP_SIZE    (WARPS * SG_SIZE)

constant constexpr int WT_M = BM / WM;
constant constexpr int WT_N = BN / WN;
constant constexpr int TM   = WT_M / 8;
constant constexpr int TN   = WT_N / 8;

constant constexpr int PAD_A = 16 / sizeof(IN_T);
constant constexpr int PAD_B = 16 / sizeof(IN_T);
constant constexpr int LDA_TGP = BK + PAD_A;
constant constexpr int LDB_TGP = BN + PAD_B;

// Use a 16-byte load granularity – chooses VEC so VecF is always 16 bytes.
constant constexpr int VEC = 16 / sizeof(IN_T);
constant constexpr int A_TCOLS = BK / VEC;
constant constexpr int A_ROW_STEP = TGP_SIZE / A_TCOLS;
constant constexpr int B_TCOLS = BN / VEC;
constant constexpr int B_ROW_STEP = TGP_SIZE / B_TCOLS;

struct alignas(16) VecF { IN_T v[VEC]; };

static_assert(BM % (8 * WM) == 0, "BM must be a multiple of 8*WM");
static_assert(BN % (8 * WN) == 0, "BN must be a multiple of 8*WN");
static_assert(BK % 8 == 0,        "BK must be a multiple of 8");
static_assert(BK % VEC == 0,      "BK must be divisible by VEC");
static_assert(BN % VEC == 0,      "BN must be divisible by VEC");

static inline void load_A_tile(threadgroup IN_T   *As,
                               const device IN_T  *A,
                               int                 lda,
                               int                 M,
                               int                 K,
                               int                 a_row0,
                               int                 a_col0,
                               int                 tid,
                               int                 kbound)
{
    int local_row0 = tid / A_TCOLS;
    int local_col0 = (tid % A_TCOLS) * VEC;
    #pragma unroll
    for (int r = 0; r < BM; r += A_ROW_STEP) {
        int rl = local_row0 + r;
        if (rl >= BM) break;
        VecF acc;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc.v[i] = (IN_T)0;
#if TRANS_A
        // A is stored K x M (lda = stride along K).  Element (m, k) is A[k*lda + m].
        int gm = a_row0 + rl;
        bool m_ok = gm < M;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int gk = a_col0 + local_col0 + i;
            bool k_ok = gk < kbound;
            acc.v[i] = (m_ok && k_ok) ? A[gk * lda + gm] : (IN_T)0;
        }
#else
        bool m_ok = (a_row0 + rl) < M;
        int gc_k0 = a_col0 + local_col0;
        bool k_full = (gc_k0 + VEC) <= kbound;
        if (m_ok && k_full) {
            acc = *((const device VecF*)(&A[(a_row0 + rl) * lda + gc_k0]));
        } else {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gk = gc_k0 + i;
                bool ok = m_ok && (gk < kbound);
                acc.v[i] = ok ? A[(a_row0 + rl) * lda + gk] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&As[rl * LDA_TGP + local_col0])) = acc;
    }
}

static inline void load_B_tile(threadgroup IN_T   *Bs,
                               const device IN_T  *B,
                               int                 ldb,
                               int                 N,
                               int                 K,
                               int                 b_row0,
                               int                 b_col0,
                               int                 tid,
                               int                 kbound)
{
    int local_row0 = tid / B_TCOLS;
    int local_col0 = (tid % B_TCOLS) * VEC;
    int n_global   = b_col0 + local_col0;
    #pragma unroll
    for (int r = 0; r < BK; r += B_ROW_STEP) {
        int rl = local_row0 + r;
        if (rl >= BK) break;
        VecF acc;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc.v[i] = (IN_T)0;
        int gk = b_row0 + rl;
#if TRANS_B
        bool k_ok = gk < kbound;
        if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i;
                bool ok = gn < N;
                acc.v[i] = ok ? B[gn * ldb + gk] : (IN_T)0;
            }
        }
#else
        bool k_ok = gk < kbound;
        bool n_full = (n_global + VEC) <= N;
        if (k_ok && n_full) {
            acc = *((const device VecF*)(&B[gk * ldb + n_global]));
        } else if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i;
                bool ok = gn < N;
                acc.v[i] = ok ? B[gk * ldb + gn] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&Bs[rl * LDB_TGP + local_col0])) = acc;
    }
}

kernel void simd_gemm(
    device const IN_T   *A           [[buffer(0)]],
    device const IN_T   *B           [[buffer(1)]],
    device       OUT_T  *C           [[buffer(2)]],
    constant int& gM                 [[buffer(3)]],
    constant int& gN                 [[buffer(4)]],
    constant int& gK                 [[buffer(5)]],
    constant int& gLda               [[buffer(6)]],
    constant int& gLdb               [[buffer(7)]],
    constant int& gLdc               [[buffer(8)]],
    uint3        tgid                [[threadgroup_position_in_grid]],
    uint         sgid                [[simdgroup_index_in_threadgroup]],
    uint         lane                [[thread_index_in_simdgroup]])
{
    threadgroup IN_T As[BM * LDA_TGP];
    threadgroup IN_T Bs[BK * LDB_TGP];

    int tid = int(sgid) * SG_SIZE + int(lane);

    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_block = tgy * BM;
    int n_block = tgx * BN;

    int warp_row = int(sgid) / WN;
    int warp_col = int(sgid) % WN;
    int warp_m   = warp_row * WT_M;
    int warp_n   = warp_col * WT_N;

    simdgroup_matrix<ACC_T, 8, 8> Cfrag[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            Cfrag[i][j] = simdgroup_matrix<ACC_T, 8, 8>(0);

    int k_tiles_full = gK / BK;
    int k_tail       = gK - k_tiles_full * BK;

    for (int kt = 0; kt < k_tiles_full; ++kt) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, kt * BK, tid, (kt + 1) * BK);
        load_B_tile(Bs, B, gLdb, gN, gK, kt * BK, n_block, tid, (kt + 1) * BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<IN_T, 8, 8> Afrag[TM];
            simdgroup_matrix<IN_T, 8, 8> Bfrag[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                simdgroup_load(Afrag[i],
                               &As[(warp_m + i * 8) * LDA_TGP + kk],
                               LDA_TGP);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                simdgroup_load(Bfrag[j],
                               &Bs[kk * LDB_TGP + warp_n + j * 8],
                               LDB_TGP);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
    }

#if !K_ALIGNED
    if (k_tail > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < BM * LDA_TGP; i += TGP_SIZE) As[i] = (IN_T)0;
        for (int i = tid; i < BK * LDB_TGP; i += TGP_SIZE) Bs[i] = (IN_T)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, k_tiles_full * BK, tid, gK);
        load_B_tile(Bs, B, gLdb, gN, gK, k_tiles_full * BK, n_block, tid, gK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<IN_T, 8, 8> Afrag[TM];
            simdgroup_matrix<IN_T, 8, 8> Bfrag[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA_TGP + kk], LDA_TGP);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                simdgroup_load(Bfrag[j], &Bs[kk * LDB_TGP + warp_n + j * 8], LDB_TGP);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
    }
#endif

    // Compute the (row, col) each lane owns within an 8x8 simdgroup_matrix.
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = m_block + warp_m + i * 8 + fm;
            int col = n_block + warp_n + j * 8 + fn;
#if MN_ALIGNED
            int row0 = m_block + warp_m + i * 8;
            int col0 = n_block + warp_n + j * 8;
#if OUT_IS_ACC
            simdgroup_store(Cfrag[i][j], &C[row0 * gLdc + col0], gLdc);
#else
            simdgroup_matrix<OUT_T, 8, 8> Cout;
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                Cout.thread_elements()[kk] = (OUT_T)Cfrag[i][j].thread_elements()[kk];
            simdgroup_store(Cout, &C[row0 * gLdc + col0], gLdc);
#endif
            (void)row; (void)col;
#else
            ACC_T te0 = Cfrag[i][j].thread_elements()[0];
            ACC_T te1 = Cfrag[i][j].thread_elements()[1];
            int rr = row;
            int cc0 = col + 0;
            int cc1 = col + 1;
            if (rr < gM && cc0 < gN) C[rr * gLdc + cc0] = OUT_T(te0);
            if (rr < gM && cc1 < gN) C[rr * gLdc + cc1] = OUT_T(te1);
#endif
        }
}
"""


# ---------------------------------------------------------------------------
# M5 tensor-ops GEMM (matmul2d 16x32x16 cooperative tensor fragments)
# ---------------------------------------------------------------------------
# Per-thread C accumulator is stored as a plain array of ACC_T's; cooperative
# tensors are constructed only inside the K loop so that arrays of them are
# never declared (Metal can't form arrays of `[[sizeas(...)]]` types).
M5_GEMM_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define BM          __BM__
#define BN          __BN__
#define BK          __BK__
#define WM          __WM__
#define WN          __WN__
#define TRANS_A     __TRANS_A__
#define TRANS_B     __TRANS_B__
#define MN_ALIGNED  __MN_ALIGNED__
#define K_ALIGNED   __K_ALIGNED__
#define SWIZZLE_LOG __SWIZZLE_LOG__
#define RELAXED     __RELAXED__
#define DBUF        __DBUF__

#define SG_SIZE     32
#define WARPS       (WM * WN)
#define TGP_SIZE    (WARPS * SG_SIZE)
#define NBUF        (DBUF ? 2 : 1)

constant constexpr int WT_M = BM / WM;
constant constexpr int WT_N = BN / WN;
constant constexpr int FM   = 16;
constant constexpr int FN   = 32;
constant constexpr int FK   = 16;
constant constexpr int TM   = WT_M / FM;
constant constexpr int TN   = WT_N / FN;

// Per-thread element counts for each fragment (16x16 = 8 per thread; 16x32 = 16 per thread)
constant constexpr int A_ELEM_PER_THR = (FM * FK) / 32;     // 8
constant constexpr int B_ELEM_PER_THR = (FK * FN) / 32;     // 16
constant constexpr int C_ELEM_PER_THR = (FM * FN) / 32;     // 16

// PAD avoids bank conflicts on threadgroup access AND ensures 16-byte alignment
// for the vectorized VecF loads.  When BK / BN are already multiples of VEC,
// PAD=0 still works for alignment — but might cause bank conflicts.
// __PAD__ is supplied by the Python wrapper; default behaviour is 16/sizeof(IN_T).
constant constexpr int PAD_A = __PAD__;
constant constexpr int PAD_B = __PAD__;
constant constexpr int LDA_TGP = BK + PAD_A;
constant constexpr int LDB_TGP = BN + PAD_B;

constant constexpr int VEC = 16 / sizeof(IN_T);
constant constexpr int A_TCOLS = BK / VEC;
constant constexpr int A_ROW_STEP = TGP_SIZE / A_TCOLS;
constant constexpr int B_TCOLS = BN / VEC;
constant constexpr int B_ROW_STEP = TGP_SIZE / B_TCOLS;

struct alignas(16) VecF { IN_T v[VEC]; };

static_assert(BM % (FM * WM) == 0, "BM must be multiple of 16*WM");
static_assert(BN % (FN * WN) == 0, "BN must be multiple of 32*WN");
static_assert(BK % FK == 0,         "BK must be multiple of 16");
static_assert(BK % VEC == 0,        "BK must be divisible by VEC");
static_assert(BN % VEC == 0,        "BN must be divisible by VEC");

static inline void load_A_tile(threadgroup IN_T   *As,
                               const device IN_T *A,
                               int lda, int M, int K,
                               int a_row0, int a_col0,
                               int tid, int kbound) {
    int local_row0 = tid / A_TCOLS;
    int local_col0 = (tid % A_TCOLS) * VEC;
    #pragma unroll
    for (int r = 0; r < BM; r += A_ROW_STEP) {
        int rl = local_row0 + r; if (rl >= BM) break;
        VecF acc;
        #pragma unroll
        for (int i=0;i<VEC;++i) acc.v[i] = (IN_T)0;
#if TRANS_A
        // A is stored K x M (lda = stride along K).  Element (m, k) is A[k*lda + m].
        int gm = a_row0 + rl;
        bool m_ok = gm < M;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int gk = a_col0 + local_col0 + i;
            bool k_ok = gk < kbound;
            acc.v[i] = (m_ok && k_ok) ? A[gk * lda + gm] : (IN_T)0;
        }
#else
        bool m_ok = (a_row0 + rl) < M;
        int gc_k0 = a_col0 + local_col0;
        bool k_full = (gc_k0 + VEC) <= kbound;
        if (m_ok && k_full) {
            acc = *((const device VecF*)(&A[(a_row0 + rl) * lda + gc_k0]));
        } else {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gk = gc_k0 + i; bool ok = m_ok && (gk < kbound);
                acc.v[i] = ok ? A[(a_row0 + rl) * lda + gk] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&As[rl * LDA_TGP + local_col0])) = acc;
    }
}

static inline void load_B_tile(threadgroup IN_T *Bs,
                               const device IN_T *B,
                               int ldb, int N, int K,
                               int b_row0, int b_col0,
                               int tid, int kbound) {
    int local_row0 = tid / B_TCOLS;
    int local_col0 = (tid % B_TCOLS) * VEC;
    int n_global   = b_col0 + local_col0;
    #pragma unroll
    for (int r = 0; r < BK; r += B_ROW_STEP) {
        int rl = local_row0 + r; if (rl >= BK) break;
        VecF acc;
        #pragma unroll
        for (int i=0;i<VEC;++i) acc.v[i] = (IN_T)0;
        int gk = b_row0 + rl;
#if TRANS_B
        bool k_ok = gk < kbound;
        if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i; bool ok = gn < N;
                acc.v[i] = ok ? B[gn * ldb + gk] : (IN_T)0;
            }
        }
#else
        bool k_ok = gk < kbound;
        bool n_full = (n_global + VEC) <= N;
        if (k_ok && n_full) {
            acc = *((const device VecF*)(&B[gk * ldb + n_global]));
        } else if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i; bool ok = gn < N;
                acc.v[i] = ok ? B[gk * ldb + gn] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&Bs[rl * LDB_TGP + local_col0])) = acc;
    }
}

kernel void m5_gemm(
    device const IN_T   *A           [[buffer(0)]],
    device const IN_T   *B           [[buffer(1)]],
    device       OUT_T  *C           [[buffer(2)]],
    constant int& gM                 [[buffer(3)]],
    constant int& gN                 [[buffer(4)]],
    constant int& gK                 [[buffer(5)]],
    constant int& gLda               [[buffer(6)]],
    constant int& gLdb               [[buffer(7)]],
    constant int& gLdc               [[buffer(8)]],
    uint3        tgid                [[threadgroup_position_in_grid]],
    uint         sgid                [[simdgroup_index_in_threadgroup]],
    uint         lane                [[thread_index_in_simdgroup]])
{
    // Double-buffered (NBUF=2) tiles when DBUF==1, single buffer otherwise.
    threadgroup IN_T As[NBUF * BM * LDA_TGP];
    threadgroup IN_T Bs[NBUF * BK * LDB_TGP];

    int tid = int(sgid) * SG_SIZE + int(lane);

    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_block = tgy * BM;
    int n_block = tgx * BN;

    int warp_row = int(sgid) / WN;
    int warp_col = int(sgid) % WN;
    int warp_m   = warp_row * WT_M;
    int warp_n   = warp_col * WT_N;

    constexpr auto desc = matmul2d_descriptor(
        FM, FN, FK, false, false, RELAXED,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    // --- Pre-compute per-thread multidim index → (row, col) maps ONCE ---
    // We index A by (m, k) [m ∈ 0..15, k ∈ 0..15]; B by (n, k) [n ∈ 0..31, k ∈ 0..15];
    // C by (n, m).  Cooperative-tensor::get_multidimensional_index() returns
    // [fastest, slowest], hence the order [k, m] / [n, k] / [n, m].
    short a_off[A_ELEM_PER_THR];   // = idx_m * LDA_TGP + idx_k (within frag)
    short b_off[B_ELEM_PER_THR];
    short c_om[C_ELEM_PER_THR];    // store row offset of each ct_c element
    short c_on[C_ELEM_PER_THR];    // store col offset
    {
        auto ct_a_proto = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
        int e = 0;
        for (auto it = ct_a_proto.begin(); it != ct_a_proto.end(); ++it, ++e) {
            auto idx = it.get_multidimensional_index();
            a_off[e] = short(idx[1]) * short(LDA_TGP) + short(idx[0]);
        }
        auto ct_b_proto = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
        e = 0;
        for (auto it = ct_b_proto.begin(); it != ct_b_proto.end(); ++it, ++e) {
            auto idx = it.get_multidimensional_index();
            b_off[e] = short(idx[1]) * short(LDB_TGP) + short(idx[0]);
        }
        auto ct_c_proto = op.get_destination_cooperative_tensor<decltype(ct_a_proto), decltype(ct_b_proto), ACC_T>();
        e = 0;
        for (auto it = ct_c_proto.begin(); it != ct_c_proto.end(); ++it, ++e) {
            auto idx = it.get_multidimensional_index();
            c_om[e] = short(idx[1]);
            c_on[e] = short(idx[0]);
        }
    }

    // Per-thread accumulator storage
    ACC_T Cacc[TM * TN * C_ELEM_PER_THR];
    #pragma unroll
    for (int i = 0; i < TM * TN * C_ELEM_PER_THR; ++i) Cacc[i] = (ACC_T)0;

    // Pre-allocate input fragment storage for the inner K-iter (reused across i, j).
    IN_T Astg[TM * A_ELEM_PER_THR];
    IN_T Bstg[TN * B_ELEM_PER_THR];

    int k_tiles_full = gK / BK;
    int k_tail       = gK - k_tiles_full * BK;

#if DBUF
    // --- Double-buffered K-loop ---
    // Issue the next K-tile's threadgroup loads while the current MMA runs.
    // Buffer A occupies bytes [cur * BM * LDA_TGP, (cur+1) * BM * LDA_TGP).
    if (k_tiles_full > 0) {
        load_A_tile(As, A, gLda, gM, gK, m_block, 0, tid, BK);
        load_B_tile(Bs, B, gLdb, gN, gK, 0, n_block, tid, BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kt = 0; kt < k_tiles_full; ++kt) {
        int cur_a = (kt & 1) * (BM * LDA_TGP);
        int cur_b = (kt & 1) * (BK * LDB_TGP);
        if (kt + 1 < k_tiles_full) {
            int nxt_a = ((kt + 1) & 1) * (BM * LDA_TGP);
            int nxt_b = ((kt + 1) & 1) * (BK * LDB_TGP);
            load_A_tile(As + nxt_a, A, gLda, gM, gK, m_block, (kt + 1) * BK, tid, (kt + 2) * BK);
            load_B_tile(Bs + nxt_b, B, gLdb, gN, gK, (kt + 1) * BK, n_block, tid, (kt + 2) * BK);
        }
        threadgroup IN_T *As_use = As + cur_a;
        threadgroup IN_T *Bs_use = Bs + cur_b;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += FK) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int base_row = warp_m + i * FM;
                threadgroup IN_T *src = &As_use[base_row * LDA_TGP + kk];
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    Astg[i * A_ELEM_PER_THR + e] = src[a_off[e]];
                }
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int base_col = warp_n + j * FN;
                threadgroup IN_T *src = &Bs_use[kk * LDB_TGP + base_col];
                #pragma unroll
                for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                    Bstg[j * B_ELEM_PER_THR + e] = src[b_off[e]];
                }
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                auto ct_a = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    ct_a[e] = Astg[i * A_ELEM_PER_THR + e];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    auto ct_b = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                    #pragma unroll
                    for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                        ct_b[e] = Bstg[j * B_ELEM_PER_THR + e];
                    }
                    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), ACC_T>();
                    int frag_off = (i * TN + j) * C_ELEM_PER_THR;
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) ct_c[e] = Cacc[frag_off + e];
                    op.run(ct_a, ct_b, ct_c);
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) Cacc[frag_off + e] = ct_c[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#else
    for (int kt = 0; kt < k_tiles_full; ++kt) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, kt * BK, tid, (kt + 1) * BK);
        load_B_tile(Bs, B, gLdb, gN, gK, kt * BK, n_block, tid, (kt + 1) * BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += FK) {
            // Load all A fragments for this kk (TM of them).
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int base_row = warp_m + i * FM;
                threadgroup IN_T *src = &As[base_row * LDA_TGP + kk];
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    Astg[i * A_ELEM_PER_THR + e] = src[a_off[e]];
                }
            }
            // Load all B fragments for this kk (TN of them).
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int base_col = warp_n + j * FN;
                threadgroup IN_T *src = &Bs[kk * LDB_TGP + base_col];
                #pragma unroll
                for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                    Bstg[j * B_ELEM_PER_THR + e] = src[b_off[e]];
                }
            }
            // Outer-product MMA loop.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                auto ct_a = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    ct_a[e] = Astg[i * A_ELEM_PER_THR + e];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    auto ct_b = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                    #pragma unroll
                    for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                        ct_b[e] = Bstg[j * B_ELEM_PER_THR + e];
                    }
                    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), ACC_T>();
                    int frag_off = (i * TN + j) * C_ELEM_PER_THR;
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) ct_c[e] = Cacc[frag_off + e];
                    op.run(ct_a, ct_b, ct_c);
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) Cacc[frag_off + e] = ct_c[e];
                }
            }
        }
    }
#endif

#if !K_ALIGNED
    if (k_tail > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < BM * LDA_TGP; i += TGP_SIZE) As[i] = (IN_T)0;
        for (int i = tid; i < BK * LDB_TGP; i += TGP_SIZE) Bs[i] = (IN_T)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, k_tiles_full * BK, tid, gK);
        load_B_tile(Bs, B, gLdb, gN, gK, k_tiles_full * BK, n_block, tid, gK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += FK) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int base_row = warp_m + i * FM;
                threadgroup IN_T *src = &As[base_row * LDA_TGP + kk];
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    Astg[i * A_ELEM_PER_THR + e] = src[a_off[e]];
                }
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int base_col = warp_n + j * FN;
                threadgroup IN_T *src = &Bs[kk * LDB_TGP + base_col];
                #pragma unroll
                for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                    Bstg[j * B_ELEM_PER_THR + e] = src[b_off[e]];
                }
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                auto ct_a = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) ct_a[e] = Astg[i * A_ELEM_PER_THR + e];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    auto ct_b = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                    #pragma unroll
                    for (int e = 0; e < B_ELEM_PER_THR; ++e) ct_b[e] = Bstg[j * B_ELEM_PER_THR + e];
                    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), ACC_T>();
                    int frag_off = (i * TN + j) * C_ELEM_PER_THR;
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) ct_c[e] = Cacc[frag_off + e];
                    op.run(ct_a, ct_b, ct_c);
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) Cacc[frag_off + e] = ct_c[e];
                }
            }
        }
    }
#endif

    // Store accumulators to C using the pre-computed (row, col) offsets.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int base_row = m_block + warp_m + i * FM;
            int base_col = n_block + warp_n + j * FN;
            int frag_off = (i * TN + j) * C_ELEM_PER_THR;
            #pragma unroll
            for (int e = 0; e < C_ELEM_PER_THR; ++e) {
                int r = base_row + c_om[e];
                int c = base_col + c_on[e];
#if MN_ALIGNED
                C[r * gLdc + c] = (OUT_T)Cacc[frag_off + e];
#else
                if (r < gM && c < gN)
                    C[r * gLdc + c] = (OUT_T)Cacc[frag_off + e];
#endif
            }
        }
    }
}
"""


# ---------------------------------------------------------------------------
# M5 tensor-view GEMM (uses mpp::matmul2d with tensor_inline views, no
# manual threadgroup management).  Each TG does op.run for one (BM x BN)
# output tile, sweeping the full K dim in a single op.run call.  The MPP
# runtime handles loads and accumulation internally — empirically wins on
# large bf16/fp16 GEMM where load bandwidth is the bottleneck.
# ---------------------------------------------------------------------------
M5_TENSOR_GEMM_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T        __IN_T__
#define OUT_T       __OUT_T__
#define BM          __BM__
#define BN          __BN__
#define NSG         __NSG__
#define TRANS_A     __TRANS_A__
#define TRANS_B     __TRANS_B__
#define RELAXED     __RELAXED__
#define SWIZZLE_LOG __SWIZZLE_LOG__
#define MN_ALIGNED  __MN_ALIGNED__
#define STATIC_SLICE __STATIC_SLICE__

kernel void m5_tensor_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant int& gM   [[buffer(3)]],
    constant int& gN   [[buffer(4)]],
    constant int& gK   [[buffer(5)]],
    constant int& gLda [[buffer(6)]],
    constant int& gLdb [[buffer(7)]],
    constant int& gLdc [[buffer(8)]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    // Construct tensor views from raw pointers.  Extent order is
    // (innermost, outermost) — i.e. (cols, rows) for row-major storage.
    // For row-major A (gM × gK, stride (gLda, 1)) extents are (gK, gM).
    // For transposed A (stored as gK × gM, stride (1, gLda)) we still
    // expose extents (gK, gM) but tell the matmul descriptor it's transposed.
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(gK, gM));
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(gN, gK));
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_extent, TRANS_A, TRANS_B, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    // Swizzle TG ids for L2 reuse.
    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_off = tgy * BM;
    int n_off = tgx * BN;

#if STATIC_SLICE
    // Non-transposed fast path.  Give matmul2d *static-extent* tile slices so
    // it knows each interior tile is exactly BM x BN and fully in-bounds; it
    // then drops the per-tile edge/predication logic the dynamic slice() path
    // emits unconditionally.  Apple's MPPTensorOpsMatMul2d.h recommends this
    // "inside threadgroup" fast path (it calls it static_slice; the real
    // spelling is the templated slice<E0,E1>).  Measured: a broad win — 2048³
    // 0.97→1.00, 4097³ 0.98→1.07, 1023³ 1.15→1.20, with no precision change
    // (matmul2d still accumulates the K-reduction in fp32 internally).
  #if !MN_ALIGNED
    // Partial edge tiles (only exist when M%BM or N%BN != 0): fall back to a
    // dynamic slice with the per-element validity mask.  Interior tiles — the
    // vast majority for any large shape — still take the static path below.
    bool inside = (m_off + BM <= gM) && (n_off + BN <= gN);
    if (!inside) {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(n_off, 0);
        auto mC = tC.slice(n_off, m_off);
        auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT_f32);
        auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
        for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
            if (cT_f32.is_valid_element(i)) cT_out[i] = (OUT_T)cT_f32[i];
        cT_out.store(mC);
        return;
    }
  #endif
    auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
    auto mB = tB.slice<BN, dynamic_extent>(n_off, 0);
    auto mC = tC.slice<BN, BM>(n_off, m_off);
    auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT_f32);
    auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
    for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
        cT_out[i] = (OUT_T)cT_f32[i];
    cT_out.store(mC);
#else
    // Transposed operands keep the dynamic-slice path: the static slice<>
    // extent order would need the transposed layout, and this orientation is
    // off the hot path (auto-dispatch routes transposed inputs to m5_gemm).
    auto mA = tA.slice(0, m_off);
    auto mB = tB.slice(n_off, 0);
    auto mC = tC.slice(n_off, m_off);
    auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT_f32);
    auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
  #if MN_ALIGNED
    for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
        cT_out[i] = (OUT_T)cT_f32[i];
  #else
    for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
        if (cT_f32.is_valid_element(i)) cT_out[i] = (OUT_T)cT_f32[i];
  #endif
    cT_out.store(mC);
#endif
}
"""


# ---------------------------------------------------------------------------
# Split-K m5_tensor GEMM (for deep-K, few-tile shapes).
#
# A single op.run over the full K underfills the GPU when M,N are small (few
# output tiles) but K is huge: each of the handful of tiles carries a very long
# serial K-reduction.  Splitting K across G threadgroups per output tile gives
# G x the tiles, filling the cores; the G partial sums are then reduced.  We
# avoid the atomic-accumulation contention that sank an earlier prototype by
# writing each chunk's partial to its OWN buffer (no contention) and summing in
# a cheap second pass: chunk g==0 writes C directly (bf16), chunks g>=1 write
# fp32 planes of Cp, and `reduce` does C += sum(planes).  Wins the deep-K band
# (512x512x32768 0.96->0.99, 256x256x32768 1.9x where MPSGraph underfills,
# 2048x2048x16384 1.04x); the dispatcher only routes deep-K here (see
# _is_splitk_regime) and the autotuner confirms it beats the single-pass tile.
# ---------------------------------------------------------------------------
SPLITK_GEMM_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T    __IN_T__
#define OUT_T   __OUT_T__
#define BM      __BM__
#define BN      __BN__
#define NSG     __NSG__
#define KCHUNK  __KCHUNK__
#define RELAXED __RELAXED__

// tgid.z = K-chunk index g.  g==0 writes C (OUT_T) directly; g>=1 writes fp32
// partial plane (g-1) of Cp.  Each tile's K range is [g*KCHUNK, (g+1)*KCHUNK).
kernel void splitk_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    device float  *Cp  [[buffer(3)]],
    constant int& gM   [[buffer(4)]],
    constant int& gN   [[buffer(5)]],
    constant int& gK   [[buffer(6)]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    int g  = int(tgid.z);
    int k0 = g * KCHUNK;
    if (k0 >= gK) return;
    int tiles_n = (gN + BN - 1) / BN;
    int tgx = int(tgid.x);
    int tgy = int(tgid.y);
    if (tgx >= tiles_n) return;
    int m_off = tgy * BM;
    int n_off = tgx * BN;

    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(gK, gM));
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(gN, gK));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, KCHUNK, false, false, RELAXED, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    // KCHUNK divides gK, so every chunk is a full static-extent K slice.
    auto mA = tA.slice<KCHUNK, BM>(k0, m_off);
    auto mB = tB.slice<BN, KCHUNK>(n_off, k0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);

    if (g == 0) {
        tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM));
        auto mC = tC.slice<BN, BM>(n_off, m_off);
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
        cO.store(mC);
    } else {
        device float *Cg = Cp + (size_t)(g - 1) * gM * gN;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tCp(Cg, dextents<int32_t, 2>(gN, gM));
        auto mC = tCp.slice<BN, BM>(n_off, m_off);
        cT.store(mC);
    }
}

// C[i] += sum_{p<Gm1} Cp[p, i].  C already holds chunk 0 (bf16); accumulate the
// remaining fp32 planes.  One thread per output element.
kernel void splitk_reduce(
    device const float *Cp [[buffer(0)]],
    device OUT_T *C        [[buffer(1)]],
    constant int& n        [[buffer(2)]],
    constant int& Gm1      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int i = int(gid);
    if (i >= n) return;
    float s = (float)C[i];
    size_t off = (size_t)i;
    for (int p = 0; p < Gm1; ++p) { s += Cp[off]; off += (size_t)n; }
    C[i] = (OUT_T)s;
}
"""


# ---------------------------------------------------------------------------
# 1x1 convolution as GEMM (very-thin-N path).
#
# Disassembling MPSGraph's bf16 GEMM (gemm_a18) showed it drives the M5 tensor
# coprocessor through the PRIVATE air.simdgroup_matrix_16x16x16 MMA (a 16-wide
# N-fragment), which the public matmul2d op (16x32x16, FN=32) and MSL builtins
# (8x8 only) cannot emit -- so matmul2d underfills on a 1-fragment-wide N tile
# (N<=64: ~0.5-0.8x).  convolution2d is the OTHER public coprocessor entry; a
# 1x1 conv is exactly C=A@B with M as spatial width and N as output channels.
# Empirically conv2d schedules thin output-channels better than matmul2d does
# thin-N for N<=64 (e.g. 4096x32x4096 0.53->0.74x, 8192x32x4096 0.76->0.86x;
# marginal at N=64).  N>=96 still prefers matmul2d, so the autotuner only adds
# conv as a candidate for N<=64 and keeps it only when it actually wins.
#
# Layouts map with no data movement: A (MxK row-major) = activation NHWC
# [1,1,M,K]; B (KxN) = weights HWIO [1,1,K,N]; C (MxN) = dest NHWO [1,1,M,N].
# K (input channels) must be a compile-time constant in the conv descriptor, so
# the kernel is specialized per K (cached).
# ---------------------------------------------------------------------------
CONV1X1_GEMM_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T   __IN_T__
#define OUT_T  __OUT_T__
#define BMW    __BMW__
#define BNO    __BNO__
#define NSG    __NSG__
#define KCONST __KCONST__

kernel void conv1x1_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant int& gM   [[buffer(3)]],
    constant int& gN   [[buffer(4)]],
    constant int& gK   [[buffer(5)]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tA(A, dextents<int32_t, 4>(gK, gM, 1, 1));
    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tW(B, dextents<int32_t, 4>(gN, gK, 1, 1));
    tensor<device OUT_T, dextents<int32_t, 4>, tensor_inline> tC(C, dextents<int32_t, 4>(gN, gM, 1, 1));

    constexpr auto desc = convolution2d_descriptor(
        int4(BNO, BMW, 1, 1), int4(KCONST, BMW, 1, 1), int2(1, 1),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, true, convolution2d_descriptor::mode::multiply);
    convolution2d<desc, execution_simdgroups<NSG>> op;

    int tiles_o = (gN + BNO - 1) / BNO;
    int tiles_w = (gM + BMW - 1) / BMW;
    if (int(tgid.x) >= tiles_o || int(tgid.y) >= tiles_w) return;
    int o_off = int(tgid.x) * BNO;
    int w_off = int(tgid.y) * BMW;

    auto mA = tA.slice(0, w_off, 0, 0);
    auto mW = tW.slice(o_off, 0, 0, 0);
    auto mC = tC.slice(o_off, w_off, 0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
    op.run(mA, mW, cT);
    auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mW), OUT_T>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
    cO.store(mC);
}
"""


# ---------------------------------------------------------------------------
# GEMV kernels
# ---------------------------------------------------------------------------

GEMV_NT_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define ROWS_PER_SG __ROWS_PER_SG__
#define NWARPS      __NWARPS__

// y = A @ x, A is M x K row-major.
//   - Each threadgroup handles NWARPS * ROWS_PER_SG rows.
//   - One warp computes ROWS_PER_SG rows: lanes split K, then simd_sum.
//   - Reads are coalesced (lanes in a warp read consecutive K elements of a row).
kernel void gemv_nt(
    device const IN_T   *A   [[buffer(0)]],
    device const IN_T   *x   [[buffer(1)]],
    device       OUT_T  *y   [[buffer(2)]],
    constant int& gM         [[buffer(3)]],
    constant int& gK         [[buffer(4)]],
    constant int& gLda       [[buffer(5)]],
    uint3        tgid        [[threadgroup_position_in_grid]],
    uint         sgid        [[simdgroup_index_in_threadgroup]],
    uint         lane        [[thread_index_in_simdgroup]])
{
    const int rows_per_tg = NWARPS * ROWS_PER_SG;
    int row0_tg = int(tgid.x) * rows_per_tg;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_SG; ++r) {
        int row = row0_tg + int(sgid) * ROWS_PER_SG + r;
        if (row >= gM) return;
        const device IN_T *Arow = &A[row * gLda];
        ACC_T acc = (ACC_T)0;
        int k = int(lane);
        // 4-way unrolled
        for (; k + 4*32 <= gK; k += 4*32) {
            acc += (ACC_T)Arow[k +   0] * (ACC_T)x[k +   0];
            acc += (ACC_T)Arow[k +  32] * (ACC_T)x[k +  32];
            acc += (ACC_T)Arow[k +  64] * (ACC_T)x[k +  64];
            acc += (ACC_T)Arow[k +  96] * (ACC_T)x[k +  96];
        }
        for (; k < gK; k += 32) {
            acc += (ACC_T)Arow[k] * (ACC_T)x[k];
        }
        acc = simd_sum(acc);
        if (lane == 0) y[row] = (OUT_T)acc;
    }
}
"""

GEMV_T_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define BLOCK_N     __BLOCK_N__
#define NWARPS      __NWARPS__
#define VEC         __VEC__

// Memory-bandwidth bound GEMV with coalesced, cache-line-wide reads.
//   B is K x N row-major; we compute y[n] = sum_k B[k, n] * x[k].
//   - Each lane owns VEC consecutive columns; a warp's 32 lanes read
//     32*VEC = BLOCK_N consecutive elements per k.  With VEC=2 (bf16) that is
//     a full 128-B cache line, so each line is consumed by exactly one
//     threadgroup — no half-line waste / cross-TG re-fetch (the residual that
//     left the K>=4096 band 0.87-0.97x).  VEC=1 reproduces the old 32-col,
//     half-line layout (best when N is small and we need every TG for occupancy).
//   - NWARPS simdgroups split the K dimension; partials reduced in tgroup mem.
//   - Reads at fixed k are CONSECUTIVE across lanes (perfect coalescing).
struct alignas(sizeof(IN_T) * VEC) VecT { IN_T v[VEC]; };

kernel void gemv_t(
    device const IN_T   *B   [[buffer(0)]],
    device const IN_T   *x   [[buffer(1)]],
    device       OUT_T  *y   [[buffer(2)]],
    constant int& gN         [[buffer(3)]],
    constant int& gK         [[buffer(4)]],
    constant int& gLdb       [[buffer(5)]],
    uint3        tgid        [[threadgroup_position_in_grid]],
    uint         sgid        [[simdgroup_index_in_threadgroup]],
    uint         lane        [[thread_index_in_simdgroup]])
{
    static_assert(BLOCK_N == 32 * VEC, "BLOCK_N must equal 32*VEC");
    threadgroup ACC_T partials[NWARPS][BLOCK_N];

    int col0 = int(tgid.x) * BLOCK_N;
    int n0   = col0 + int(lane) * VEC;     // first column this lane owns

    // Distribute K across warps: warp `sgid` handles k in [start, end) striding 1.
    int k_per_warp = (gK + NWARPS - 1) / NWARPS;
    int k_start    = int(sgid) * k_per_warp;
    int k_end      = min(gK, k_start + k_per_warp);

    ACC_T acc[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = (ACC_T)0;

    bool full = (n0 + VEC) <= gN;
    if (full) {
        // Vectorized full-line path: one aligned VEC-wide load per k, 4x unrolled.
        int k = k_start;
        for (; k + 4 <= k_end; k += 4) {
            VecT b0 = *((const device VecT*)(&B[(k+0) * gLdb + n0]));
            VecT b1 = *((const device VecT*)(&B[(k+1) * gLdb + n0]));
            VecT b2 = *((const device VecT*)(&B[(k+2) * gLdb + n0]));
            VecT b3 = *((const device VecT*)(&B[(k+3) * gLdb + n0]));
            ACC_T x0 = (ACC_T)x[k+0], x1 = (ACC_T)x[k+1];
            ACC_T x2 = (ACC_T)x[k+2], x3 = (ACC_T)x[k+3];
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                acc[i] += (ACC_T)b0.v[i] * x0;
                acc[i] += (ACC_T)b1.v[i] * x1;
                acc[i] += (ACC_T)b2.v[i] * x2;
                acc[i] += (ACC_T)b3.v[i] * x3;
            }
        }
        for (; k < k_end; ++k) {
            VecT bv = *((const device VecT*)(&B[k * gLdb + n0]));
            ACC_T xk = (ACC_T)x[k];
            #pragma unroll
            for (int i = 0; i < VEC; ++i) acc[i] += (ACC_T)bv.v[i] * xk;
        }
    } else {
        // Edge block: scalar with per-column bounds (handles non-VEC-aligned N).
        for (int k = k_start; k < k_end; ++k) {
            ACC_T xk = (ACC_T)x[k];
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int n = n0 + i;
                if (n < gN) acc[i] += (ACC_T)B[k * gLdb + n] * xk;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < VEC; ++i) partials[sgid][int(lane) * VEC + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp aggregates the partials; each lane writes its VEC columns.
    if (sgid == 0) {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int c = int(lane) * VEC + i;
            ACC_T s = (ACC_T)0;
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) s += partials[w][c];
            int n = col0 + c;
            if (n < gN) y[n] = (OUT_T)s;
        }
    }
}
"""


# ---------------------------------------------------------------------------
# Compilation helpers / cache
# ---------------------------------------------------------------------------

def _subst(src: str, **kw) -> str:
    out = src
    for k, v in kw.items():
        out = out.replace("__" + k + "__", str(v))
    return out


@functools.lru_cache(maxsize=None)
def _compile(src: str):
    return torch.mps.compile_shader(src)


@functools.lru_cache(maxsize=None)
def simd_gemm(in_t: str, acc_t: str, out_t: str,
              BM: int, BN: int, BK: int, WM: int, WN: int,
              trans_a: bool, trans_b: bool,
              mn_aligned: bool, k_aligned: bool,
              swizzle_log: int = 0):
    src = _subst(
        SIMD_GEMM_SRC,
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
    # default pad = 16/sizeof(IN_T) (for VecF alignment); zero is OK when
    # BK / BN are already aligned multiples of VEC.
    if pad is None:
        in_bytes = 4 if in_t == "float" else 2
        pad = 16 // in_bytes
    src = _subst(
        M5_GEMM_SRC,
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
    # Static-extent tile slices are only correct/used for the non-transposed
    # orientation (the only one auto-dispatch routes here); transposed keeps
    # the dynamic-slice path.
    static_slice = (not trans_a) and (not trans_b)
    src = _subst(
        M5_TENSOR_GEMM_SRC,
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
    """Split-K m5_tensor GEMM.  Returns (splitk_fn, reduce_fn).  KCHUNK must
    divide K (the caller guarantees this); the per-chunk K slice is static."""
    src = _subst(
        SPLITK_GEMM_SRC,
        IN_T=in_t, OUT_T=out_t,
        BM=BM, BN=BN, NSG=NSG, KCHUNK=KCHUNK,
        RELAXED=("true" if relaxed else "false"),
    )
    lib = _compile(src)
    return lib.splitk_gemm, lib.splitk_reduce


@functools.lru_cache(maxsize=None)
def conv1x1_gemm(in_t: str, out_t: str, BMW: int, BNO: int, NSG: int, K: int,
                 relaxed: bool = True):
    """1x1-conv GEMM for very-thin-N (see CONV1X1_GEMM_SRC).  K (input channels)
    is baked into the conv descriptor, so the kernel is specialized per K."""
    src = _subst(
        CONV1X1_GEMM_SRC,
        IN_T=in_t, OUT_T=out_t,
        BMW=BMW, BNO=BNO, NSG=NSG, KCONST=K,
    )
    return _compile(src).conv1x1_gemm, src


@functools.lru_cache(maxsize=None)
def gemv_nt(in_t: str, acc_t: str, out_t: str, ROWS_PER_SG: int = 1, NWARPS: int = 4):
    src = _subst(GEMV_NT_SRC, IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                 ROWS_PER_SG=ROWS_PER_SG, NWARPS=NWARPS)
    return _compile(src).gemv_nt, src


@functools.lru_cache(maxsize=None)
def gemv_t(in_t: str, acc_t: str, out_t: str, BLOCK_N: int = 32, NWARPS: int = 4,
           VEC: int = 1):
    # Each lane owns VEC columns, so a threadgroup spans BLOCK_N == 32*VEC cols.
    assert BLOCK_N == 32 * VEC, f"BLOCK_N ({BLOCK_N}) must equal 32*VEC ({32*VEC})"
    src = _subst(GEMV_T_SRC, IN_T=in_t, ACC_T=acc_t, OUT_T=out_t,
                 BLOCK_N=BLOCK_N, NWARPS=NWARPS, VEC=VEC)
    return _compile(src).gemv_t, src
