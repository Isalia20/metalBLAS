// NNA14 GEMM structure recovered from MPSNDArray's AIR (M1/M2 family).
// Tile sizes and splits are selected exclusively by mps_a14.py.
#ifdef MB_BUILD_NNA14
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
#define TRANS_A __TRANS_A__
#define TRANS_B __TRANS_B__
#define MN_ALIGNED __MN_ALIGNED__
#define SPLITS __SPLITS__
#define KCHUNK __KCHUNK__
#define LINEAR __LINEAR__
#define REDUCE_VEC __REDUCE_VEC__
#define DIM16 __DIM16__
#define EDGE_TM __EDGE_TM__
#define EDGE_TN __EDGE_TN__
#define K_TAIL __K_TAIL__
#define INDEX_T __INDEX_T__

constant constexpr uint TM = BM / (8 * SM);
constant constexpr uint TN = BN / (8 * SN);

struct MBNNA14Dims {
    uint M, N, K, lda, ldb, ldc;
    ulong bsa, bsb, bsc;
};

// Private AIR interfaces present in Apple's shader. The leading \01 is
// LLVM's unmangled-symbol marker; these declarations need runtime validation
// on each supported compiler/OS. They are not a public Metal API.
struct MBNNA14Event;
extern "C" thread MBNNA14Event* nna14_copy(
    ulong, ulong, threadgroup void*, ulong, ulong, ulong2,
    const device void*, ulong, ulong, ulong2, long2, int)
    __asm("\01air.simdgroup_async_copy_2d.p3i8.p1i8");
extern "C" void nna14_wait(int, thread MBNNA14Event* thread*)
    __asm("\01air.wait_simdgroup_events");

template<uint FM, uint FN, bool PARTIAL_K>
static inline __attribute__((always_inline)) void nna14_step(
    const device IN_T* A, const device IN_T* B,
    constant MBNNA14Dims& p, threadgroup IN_T* tile_a, threadgroup IN_T* tile_b,
    thread simdgroup_matrix<float, 8, 8> (&acc)[FN][FM],
    uint rows, uint cols, uint k0, uint kend, ushort sg, ushort lane) {
    uint sm = sg / SN, sn = sg % SN;
    uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
    uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
    // The AIR narrows each SIMD group's tile origin to 16 bits, then adds
    // constant fragment offsets to that base pointer inside the unrolled loop.
    ushort a_origin = TRANS_A ? fn * BM + sm * 8 + fm : (sm * 8 + fm) * BK + fn;
    ushort b_origin = TRANS_B ? (sn * 8 + fn) * BK + fm : fm * BN + sn * 8 + fn;
    const threadgroup IN_T* a_fragments = tile_a + a_origin;
    const threadgroup IN_T* b_fragments = tile_b + b_origin;
    if (sg == 0) {
        // Apple issues B then A from one SIMD group, preserving physical
        // orientation, waits for both events, then synchronizes the group.
        // The source extents provide zero fill at M/N and K boundaries.
#if DIM16
        long origin_k = short(k0);
        uint source_k = p.K;
        const device IN_T* source_a = A;
        const device IN_T* source_b = B;
#else
        // Apple's isDim16b branch uses a signed 16-bit copy origin below
        // 32767 on every dimension. Larger inputs advance the pointers.
        long origin_k = 0;
        uint source_k = kend - k0;
        const device IN_T* source_a = A + (TRANS_A ? INDEX_T(k0) * p.lda : k0);
        const device IN_T* source_b = B + (TRANS_B ? k0 : INDEX_T(k0) * p.ldb);
#endif
        thread MBNNA14Event* eb = nna14_copy(
            sizeof(IN_T), alignof(IN_T), tile_b, TRANS_B ? BK : BN, 1,
            TRANS_B ? ulong2(BK, BN) : ulong2(BN, BK), source_b, p.ldb, 1,
            TRANS_B ? ulong2(source_k, cols) : ulong2(cols, source_k),
            TRANS_B ? long2(origin_k, 0) : long2(0, origin_k), 0);
        thread MBNNA14Event* ea = nna14_copy(
            sizeof(IN_T), alignof(IN_T), tile_a, TRANS_A ? BM : BK, 1,
            TRANS_A ? ulong2(BM, BK) : ulong2(BK, BM), source_a, p.lda, 1,
            TRANS_A ? ulong2(rows, source_k) : ulong2(source_k, rows),
            TRANS_A ? long2(0, origin_k) : long2(origin_k, 0), 0);
        nna14_wait(1, &eb);
        nna14_wait(1, &ea);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort kk = 0; kk < BK; kk += 8) {
        // The separate AIR tail performs ceil(remainder/8) MMA steps.
        if (PARTIAL_K && kk >= kend - k0) break;
        simdgroup_matrix<IN_T, 8, 8> af[FM], bf[FN];
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < FN; ++j) {
            uint off = TRANS_B ? j * SN * 8 * BK + kk : kk * BN + j * SN * 8;
            bf[j].thread_elements()[0] = b_fragments[off];
            bf[j].thread_elements()[1] = b_fragments[off + (TRANS_B ? BK : 1)];
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < FM; ++i) {
            uint off = TRANS_A ? kk * BM + i * SM * 8 : i * SM * 8 * BK + kk;
            af[i].thread_elements()[0] = a_fragments[off];
            af[i].thread_elements()[1] = a_fragments[off + (TRANS_A ? BM : 1)];
        }
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < FN; ++j)
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < FM; ++i)
                simdgroup_multiply_accumulate(acc[j][i], af[i], bf[j], acc[j][i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<uint FM, uint FN>
static inline __attribute__((always_inline)) void nna14_tile(
    const device IN_T* A, const device IN_T* B, device OUT_T* C,
    constant MBNNA14Dims& p, threadgroup IN_T* tile_a, threadgroup IN_T* tile_b,
    uint3 group, ushort sg, ushort lane
#if MB_MPS_EPILOGUE && SPLITS == 1
    , const device IN_T* bias, constant ulong* bstride
#endif
    ) {
    uint batch = group.z / SPLITS, split = group.z % SPLITS;
    uint m0 = group.y * BM, n0 = group.x * BN;
    uint kstart = split * KCHUNK;
    uint kend = SPLITS > 1 ? min(kstart + KCHUNK, p.K) : p.K;
#if MN_ALIGNED && !K_TAIL
    // The aligned AIR passes the original M/N extents to the copy; every
    // destination tile fits. Only EdgeCase clips them to its tile boundary.
    uint rows = p.M, cols = p.N;
#else
    uint rows = min(uint(BM), p.M - m0), cols = min(uint(BN), p.N - n0);
#endif
    A += INDEX_T(batch) * INDEX_T(p.bsa) + (TRANS_A ? m0 : INDEX_T(m0) * p.lda);
    B += INDEX_T(batch) * INDEX_T(p.bsb) + (TRANS_B ? INDEX_T(n0) * p.ldb : n0);
    C += INDEX_T(batch) * INDEX_T(p.bsc) + INDEX_T(split) * p.M * p.ldc;
    simdgroup_matrix<float, 8, 8> acc[FN][FM];
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < FN; ++j)
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < FM; ++i)
            acc[j][i] = simdgroup_matrix<float, 8, 8>(0);

    uint full_end = K_TAIL ? kend - kend % BK : kend;
    for (uint k0 = kstart; k0 < full_end; k0 += BK)
        nna14_step<FM, FN, false>(A, B, p, tile_a, tile_b, acc,
                                  rows, cols, k0, kend, sg, lane);
#if K_TAIL
    if (full_end < kend)
        nna14_step<FM, FN, true>(A, B, p, tile_a, tile_b, acc,
                                 rows, cols, full_end, kend, sg, lane);
#endif
    uint sm = sg / SN, sn = sg % SN;
    uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
    uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < FN; ++j) {
        uint n = n0 + (j * SN + sn) * 8 + fn;
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < FM; ++i) {
            uint m = m0 + (i * SM + sm) * 8 + fm;
            if (MN_ALIGNED || m < p.M) {
#if MB_MPS_EPILOGUE && SPLITS == 1
                ulong bi = ulong(batch) * bstride[0] + ulong(m) * bstride[1] + ulong(n) * bstride[2];
                if (MN_ALIGNED || n < p.N)
                    C[INDEX_T(m) * p.ldc + n] = mb_mps_epilogue(acc[j][i].thread_elements()[0], bias, bi);
                if (MN_ALIGNED || n + 1 < p.N)
                    C[INDEX_T(m) * p.ldc + n + 1] = mb_mps_epilogue(acc[j][i].thread_elements()[1], bias, bi + bstride[2]);
#else
                if (MN_ALIGNED || n < p.N)
                    C[INDEX_T(m) * p.ldc + n] = OUT_T(acc[j][i].thread_elements()[0]);
                if (MN_ALIGNED || n + 1 < p.N)
                    C[INDEX_T(m) * p.ldc + n + 1] = OUT_T(acc[j][i].thread_elements()[1]);
#endif
            }
        }
    }
}

kernel void nna14_gemm(
    const device IN_T* A [[buffer(0)]],
    const device IN_T* B [[buffer(1)]],
    device OUT_T* C [[buffer(2)]],
    constant MBNNA14Dims& p [[buffer(3)]],
#if MB_MPS_EPILOGUE && SPLITS == 1
    const device IN_T* bias [[buffer(4)]], constant ulong* bstride [[buffer(5)]],
#endif
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
#if LINEAR
    uint nx = (p.N + BN - 1) / BN, ny = (p.M + BM - 1) / BM;
    uint q = group.x / nx;
    group = uint3(group.x % nx, q % ny, q / ny);
#endif
    threadgroup IN_T tile_a[BM * BK], tile_b[BN * BK];
    // NNA14_EdgeCase limits fragment rows/columns on the last M/N tile.
    bool edge_m = EDGE_TM < TM && (group.y + 1) * BM > p.M;
    bool edge_n = EDGE_TN < TN && (group.x + 1) * BN > p.N;
#define NNA14_TILE_ARGS A, B, C, p, tile_a, tile_b, group, sg, lane
#if MB_MPS_EPILOGUE && SPLITS == 1
#define NNA14_EPILOGUE_ARGS , bias, bstride
#else
#define NNA14_EPILOGUE_ARGS
#endif
    if (!edge_m && !edge_n) nna14_tile<TM, TN>(NNA14_TILE_ARGS NNA14_EPILOGUE_ARGS);
    else if (!edge_m) nna14_tile<TM, EDGE_TN>(NNA14_TILE_ARGS NNA14_EPILOGUE_ARGS);
    else if (!edge_n) nna14_tile<EDGE_TM, TN>(NNA14_TILE_ARGS NNA14_EPILOGUE_ARGS);
    else nna14_tile<EDGE_TM, EDGE_TN>(NNA14_TILE_ARGS NNA14_EPILOGUE_ARGS);
#undef NNA14_TILE_ARGS
#undef NNA14_EPILOGUE_ARGS
}

// Same MPSNDArrayMatrixMultiplyReduction schedule as the A16 family.
kernel void nna14_reduce(
    const device float* partial [[buffer(0)]],
    device __FINAL_T__* C [[buffer(1)]],
    constant MBNNA14Dims& p [[buffer(2)]],
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
