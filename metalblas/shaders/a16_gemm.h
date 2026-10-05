// A16 GEMM structure recovered from MPSNDArray's mmul_kernel_a16_* AIR.
// Dispatch parameters come from mps_a14.py; see docs/mps_matmuls.md.
//
// Two details of that AIR matter for speed, not just for structure:
//   * Device fragments are addressed from per-thread pointers created once
//     per tile and advanced by fixed strides (the AIR's pointer phis), not
//     recomputed as row * stride + column for every load.
//   * fcPartialOuterM/N with MatProdSxLim/SyLim: the last tile row/column
//     only loads, multiplies and stores the fragments that hold valid
//     rows/columns (EDGE_TM/EDGE_TN, from mps_a14.edge_fragments).
#ifdef MB_BUILD_A16
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "mps_epilogue.h"

#define IN_T __IN_T__
#define OUT_T __OUT_T__
#define BM __BM__
#define BN __BN__
#define BK __BK__
#define SM __SM__
#define SN __SN__
#define STAGE_A __STAGE_A__
#define TRANS_A __TRANS_A__
#define TRANS_B __TRANS_B__
#define MN_ALIGNED __MN_ALIGNED__
#define K_ALIGNED __K_ALIGNED__
#define SPLITS __SPLITS__
#define KCHUNK __KCHUNK__
#define DIM16 __DIM16__
#define REDUCE_VEC __REDUCE_VEC__
#define INDEX_T __INDEX_T__
#define EDGE_TM __EDGE_TM__
#define EDGE_TN __EDGE_TN__

constant constexpr uint TM = BM / (8 * SM);
constant constexpr uint TN = BN / (8 * SN);
constant constexpr uint NTHREADS = 32 * SM * SN;

// Batch strides are in elements; unlike matrix dimensions they may exceed 32 bits.
struct MBA16Dims {
    uint M, N, K, lda, ldb, ldc;
    ulong bsa, bsb, bsc;
};

template<bool EDGE_M, bool EDGE_K>
static inline IN_T a16_load_a(const device IN_T* a, uint m, uint k,
                              constant MBA16Dims& p, uint rows, uint count_k) {
    if ((EDGE_M && m >= rows) || (EDGE_K && k >= count_k)) return IN_T(0);
    return a[TRANS_A ? INDEX_T(k) * p.lda + m : INDEX_T(m) * p.lda + k];
}

template<bool EDGE_N, bool EDGE_K>
static inline IN_T a16_load_b(const device IN_T* b, uint k, uint n,
                              constant MBA16Dims& p, uint cols, uint count_k) {
    if ((EDGE_N && n >= cols) || (EDGE_K && k >= count_k)) return IN_T(0);
    return b[TRANS_B ? INDEX_T(n) * p.ldb + k : INDEX_T(k) * p.ldb + n];
}

#if STAGE_A
template<bool EDGE_M, bool EDGE_K>
static inline void a16_stage_a(threadgroup IN_T* tile, const device IN_T* source,
                               constant MBA16Dims& p, uint rows, uint count_k, uint tid) {
    // Preserve A's physical orientation, copying four elements per lane.
    // AIR uses four-element loads even for an odd leading stride, and
    // exactly BM*BK threadgroup elements, without bank padding.
    constexpr uint width = TRANS_A ? BM : BK;
    constexpr uint height = TRANS_A ? BK : BM;
    constexpr uint row_step = NTHREADS / (width / 4);
    ushort first_row = tid / (width / 4), col = (tid % (width / 4)) * 4;
    // The caller supplies this thread's first source element; it persists
    // across K steps and advances with the tile.
    INDEX_T source_step = INDEX_T(row_step) * p.lda;
    // The loop counter starts at zero in AIR, independently of this
    // thread's first row. Its trip count is fixed by the selected tile.
    #pragma clang loop unroll(full)
    for (ushort row_base = 0; row_base < height; row_base += row_step) {
        ushort row = first_row + row_base;
        uint m = TRANS_A ? col : row;
        uint k = TRANS_A ? row : col;
        vec<IN_T, 4> v;
        if ((!EDGE_M || m + (TRANS_A ? 3 : 0) < rows)
                  && (!EDGE_K || k + (TRANS_A ? 0 : 3) < count_k)) {
            v = *reinterpret_cast<const device vec<IN_T, 4>*>(source);
        } else {
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i) {
                bool valid = (!EDGE_M || m + (TRANS_A ? i : 0) < rows)
                          && (!EDGE_K || k + (TRANS_A ? 0 : i) < count_k);
                v[i] = valid ? source[i] : IN_T(0);
            }
        }
        *reinterpret_cast<threadgroup vec<IN_T, 4>*>(tile + row * width + col) = v;
        source += source_step;
    }
}
#endif

// Apple separates complete tiles from the M/N boundary and the K tail.
// Its AIR marks the fragment loops llvm.loop.unroll.full, not unroll.enable.
template<bool EDGE_M, bool EDGE_N, bool EDGE_K, uint FM, uint FN>
static inline __attribute__((always_inline)) void a16_step(
    const device IN_T* A, const device IN_T* B, constant MBA16Dims& p,
#if STAGE_A
    threadgroup IN_T* tile_a,
#endif
    thread simdgroup_matrix<float, 8, 8> (&acc)[TN][TM],
    const device IN_T* ap, const device IN_T* bp, const device IN_T* cp,
    uint rows, uint cols, uint count_k, ushort sg, ushort lane) {
    uint sm = sg / SN, sn = sg % SN;
    uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
    uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
#if STAGE_A
    a16_stage_a<EDGE_M, EDGE_K>(tile_a, cp, p, rows, count_k, uint(sg) * 32 + lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    #pragma clang loop unroll(full)
    for (uint kk = 0; kk < BK; kk += 8) {
        if (EDGE_K && kk >= count_k) break;
        simdgroup_matrix<IN_T, 8, 8> af[TM], bf[TN];
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < FN; ++j) {
            uint n = (j * SN + sn) * 8 + fn;
            uint k = kk + fm;
            if (!EDGE_N && !EDGE_K) {
                // Fixed offsets from the per-thread pointer, as in the AIR's pointer phis.
                if (!TRANS_B) {
                    vec<IN_T, 2> v = *reinterpret_cast<const device vec<IN_T, 2>*>(
                        bp + INDEX_T(kk) * p.ldb + j * SN * 8);
                    bf[j].thread_elements()[0] = v[0];
                    bf[j].thread_elements()[1] = v[1];
                } else {
                    const device IN_T* q = bp + INDEX_T(j * SN * 8) * p.ldb + kk;
                    bf[j].thread_elements()[0] = q[0];
                    bf[j].thread_elements()[1] = q[p.ldb];
                }
            } else if (!TRANS_B && (!EDGE_N || n + 1 < cols)
                                 && (!EDGE_K || k < count_k)) {
                vec<IN_T, 2> v =
                    *reinterpret_cast<const device vec<IN_T, 2>*>(B + INDEX_T(k) * p.ldb + n);
                bf[j].thread_elements()[0] = v[0];
                bf[j].thread_elements()[1] = v[1];
            } else {
                bf[j].thread_elements()[0] = a16_load_b<EDGE_N, EDGE_K>(B, k, n, p, cols, count_k);
                bf[j].thread_elements()[1] = a16_load_b<EDGE_N, EDGE_K>(B, k, n + 1, p, cols, count_k);
            }
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < FM; ++i) {
            // Adjacent fragments belong to successive SIMD groups.
            uint lm = (i * SM + sm) * 8 + fm;
#if STAGE_A
            uint off = TRANS_A ? (kk + fn) * BM + lm : lm * BK + kk + fn;
            af[i].thread_elements()[0] = tile_a[off];
            af[i].thread_elements()[1] = tile_a[off + (TRANS_A ? BM : 1)];
#else
            // The unstaged AIR path loads adjacent fragment elements as a
            // pair, including on odd leading strides. Separate scalar loads
            // are not combined by the public Metal compiler here.
            uint k = kk + fn;
            if (!EDGE_M && !EDGE_K) {
                if (!TRANS_A) {
                    vec<IN_T, 2> v = *reinterpret_cast<const device vec<IN_T, 2>*>(
                        ap + INDEX_T(i * SM * 8) * p.lda + kk);
                    af[i].thread_elements()[0] = v[0];
                    af[i].thread_elements()[1] = v[1];
                } else {
                    const device IN_T* q = ap + INDEX_T(kk) * p.lda + i * SM * 8;
                    af[i].thread_elements()[0] = q[0];
                    af[i].thread_elements()[1] = q[p.lda];
                }
            } else if (!TRANS_A && (!EDGE_M || lm < rows)
                                 && (!EDGE_K || k + 1 < count_k)) {
                vec<IN_T, 2> v =
                    *reinterpret_cast<const device vec<IN_T, 2>*>(A + INDEX_T(lm) * p.lda + k);
                af[i].thread_elements()[0] = v[0];
                af[i].thread_elements()[1] = v[1];
            } else {
                af[i].thread_elements()[0] = a16_load_a<EDGE_M, EDGE_K>(A, lm, k, p, rows, count_k);
                af[i].thread_elements()[1] = a16_load_a<EDGE_M, EDGE_K>(A, lm, k + 1, p, rows, count_k);
            }
#endif
        }
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < FN; ++j)
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < FM; ++i)
                simdgroup_multiply_accumulate(acc[j][i], af[i], bf[j], acc[j][i]);
    }
#if STAGE_A
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
}

template<bool EDGE_M, bool EDGE_N, uint FM, uint FN>
static inline __attribute__((always_inline)) void a16_tile(
    const device IN_T* A, const device IN_T* B, device OUT_T* C,
    constant MBA16Dims& p,
#if STAGE_A
    threadgroup IN_T* tile_a,
#endif
    uint3 group, ushort sg, ushort lane
#if MB_MPS_EPILOGUE && SPLITS == 1
    , const device IN_T* bias, constant ulong* bstride
#endif
    ) {
    uint batch = group.z / SPLITS, split = group.z % SPLITS;
    A += ulong(batch) * p.bsa;
    B += ulong(batch) * p.bsb;
    C += ulong(batch) * p.bsc + ulong(split) * p.M * p.ldc;
    uint m0 = group.y * BM, n0 = group.x * BN;
    uint kstart = split * KCHUNK, kend = min(kstart + KCHUNK, p.K);
    A += TRANS_A ? ulong(kstart) * p.lda + m0 : ulong(m0) * p.lda + kstart;
    B += TRANS_B ? ulong(n0) * p.ldb + kstart : ulong(kstart) * p.ldb + n0;
    uint rows = p.M - m0, cols = p.N - n0;
    uint sm = sg / SN, sn = sg % SN;
    uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
    uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
    simdgroup_matrix<float, 8, 8> acc[TN][TM];
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < TN; ++j)
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < TM; ++i)
            acc[j][i] = simdgroup_matrix<float, 8, 8>(0);

    const device IN_T* ap = A + (TRANS_A ? INDEX_T(fn) * p.lda + sm * 8 + fm
                                         : INDEX_T(sm * 8 + fm) * p.lda + fn);
    const device IN_T* bp = B + (TRANS_B ? INDEX_T(sn * 8 + fn) * p.ldb + fm
                                         : INDEX_T(fm) * p.ldb + sn * 8 + fn);
    const device IN_T* cp = A;
#if STAGE_A
    {
        // Copy source: isDim16b narrows only this initial per-thread offset;
        // the AIR advances later rows and K steps with full-width strides.
        constexpr uint width = TRANS_A ? BM : BK;
        uint tid = uint(sg) * 32 + lane;
        INDEX_T origin = INDEX_T(tid / (width / 4)) * p.lda + (tid % (width / 4)) * 4;
        cp = A + (DIM16 ? INDEX_T(ushort(origin)) : origin);
    }
#endif
    const INDEX_T a_step = TRANS_A ? INDEX_T(BK) * p.lda : BK;
    const INDEX_T b_step = TRANS_B ? BK : INDEX_T(BK) * p.ldb;
    uint full_end = kend - kend % BK;
    for (uint k0 = kstart; k0 < full_end; k0 += BK) {
        a16_step<EDGE_M, EDGE_N, false, FM, FN>(A, B, p,
#if STAGE_A
            tile_a,
#endif
            acc, ap, bp, cp, rows, cols, BK, sg, lane);
        A += a_step; B += b_step; ap += a_step; bp += b_step; cp += a_step;
    }
    if (!K_ALIGNED && full_end < kend) {
        a16_step<EDGE_M, EDGE_N, true, FM, FN>(A, B, p,
#if STAGE_A
            tile_a,
#endif
            acc, ap, bp, cp, rows, cols, kend - full_end, sg, lane);
    }
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < FN; ++j) {
        uint n = n0 + (j * SN + sn) * 8 + fn;
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < FM; ++i) {
            uint m = m0 + (i * SM + sm) * 8 + fm;
            if (!EDGE_M || m < p.M) {
#if MB_MPS_EPILOGUE && SPLITS == 1
                ulong bi = ulong(batch) * bstride[0] + ulong(m) * bstride[1] + ulong(n) * bstride[2];
                if (!EDGE_N || n < p.N)
                    C[ulong(m) * p.ldc + n] = mb_mps_epilogue(acc[j][i].thread_elements()[0], bias, bi);
                if (!EDGE_N || n + 1 < p.N)
                    C[ulong(m) * p.ldc + n + 1] = mb_mps_epilogue(acc[j][i].thread_elements()[1], bias, bi + bstride[2]);
#else
                if (!EDGE_N || n < p.N)
                    C[ulong(m) * p.ldc + n] = OUT_T(acc[j][i].thread_elements()[0]);
                if (!EDGE_N || n + 1 < p.N)
                    C[ulong(m) * p.ldc + n + 1] = OUT_T(acc[j][i].thread_elements()[1]);
#endif
            }
        }
    }
}

kernel void a16_gemm(
    const device IN_T* A [[buffer(0)]],
    const device IN_T* B [[buffer(1)]],
    device OUT_T* C [[buffer(2)]],
    constant MBA16Dims& p [[buffer(3)]],
#if MB_MPS_EPILOGUE && SPLITS == 1
    const device IN_T* bias [[buffer(4)]], constant ulong* bstride [[buffer(5)]],
#endif
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
#if STAGE_A
    threadgroup IN_T tile_a[BM * BK];
#define A16_TILE_ARGS A, B, C, p, tile_a, group, sg, lane
#else
#define A16_TILE_ARGS A, B, C, p, group, sg, lane
#endif
#if MB_MPS_EPILOGUE && SPLITS == 1
#define A16_EPILOGUE_ARGS , bias, bstride
#else
#define A16_EPILOGUE_ARGS
#endif
    bool edge_m = !MN_ALIGNED && (group.y + 1) * BM > p.M;
    bool edge_n = !MN_ALIGNED && (group.x + 1) * BN > p.N;
    // fcPartialOuterM/N with MatProdSxLim/SyLim: the last tile row/column only
    // loads and multiplies fragments that contain valid rows/columns.
    if (!edge_m && !edge_n) a16_tile<false, false, TM, TN>(A16_TILE_ARGS A16_EPILOGUE_ARGS);
    else if (!edge_m) a16_tile<false, true, TM, EDGE_TN>(A16_TILE_ARGS A16_EPILOGUE_ARGS);
    else if (!edge_n) a16_tile<true, false, EDGE_TM, TN>(A16_TILE_ARGS A16_EPILOGUE_ARGS);
    else a16_tile<true, true, EDGE_TM, EDGE_TN>(A16_TILE_ARGS A16_EPILOGUE_ARGS);
#undef A16_TILE_ARGS
#undef A16_EPILOGUE_ARGS
}

// MPSNDArrayMatrixMultiplyReduction: 32x32 threads, four adjacent columns
// per thread when N is divisible by four. All split planes remain float32.
kernel void a16_reduce(
    const device float* partial [[buffer(0)]],
    device __FINAL_T__* C [[buffer(1)]],
    constant MBA16Dims& p [[buffer(2)]],
#if MB_MPS_EPILOGUE
    const device IN_T* bias [[buffer(3)]], constant ulong* bstride [[buffer(4)]],
#endif
    uint3 tid [[thread_position_in_grid]]) {
    uint m = tid.y, n = tid.x * REDUCE_VEC;
    if (m >= p.M || n >= p.N) return;
    ulong plane = ulong(p.M) * p.N;
    ulong src = ulong(tid.z) * SPLITS * plane + ulong(m) * p.N + n;
    float sum[REDUCE_VEC] = {};
    for (uint s = 0; s < SPLITS; ++s) {
        #pragma clang loop unroll(full)
        for (uint v = 0; v < REDUCE_VEC; ++v) sum[v] += partial[src + s * plane + v];
    }
    ulong dst = ulong(tid.z) * p.bsc + ulong(m) * p.ldc + n;
    #pragma clang loop unroll(full)
    for (uint v = 0; v < REDUCE_VEC; ++v) {
#if MB_MPS_EPILOGUE
        ulong bi = ulong(tid.z) * bstride[0] + ulong(m) * bstride[1] + ulong(n + v) * bstride[2];
        C[dst + v] = mb_mps_epilogue(sum[v], bias, bi);
#else
        C[dst + v] = __FINAL_T__(sum[v]);
#endif
    }
}
#endif
