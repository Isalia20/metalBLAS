// GEMM with the work decomposition of MPSNDArray's `gemm_a18` kernel.
//
// Apple's kernel is driven by MPSMatMulAutoTuningParametersA18 (see
// metalblas/mps_heuristics.py and docs/mps_matmul_heuristics.md).  This kernel
// takes the same parameters as compile-time constants and reproduces the same
// structure, so Apple's per-GPU selection applies to it unchanged:
//
//   * every simdgroup owns a (UM*16) x (UN*16) output block and multiplies it
//     by KC-wide K chunks straight from device memory (no threadgroup staging);
//     the A chunk is loaded into registers first and B streams through the
//     multiply, so all of a chunk's loads are in flight before its MACs;
//   * a threadgroup is SM x SN simdgroups side by side, times KS groups that
//     split K round-robin ("local K splits") and are reduced through
//     threadgroup memory at the end;
//   * threadgroups are numbered linearly and walk the output in uber tiles of
//     subTilesM x subTilesN tiles, M-first when RW is set (Apple's rasterMBias);
//   * the simdgroups of a threadgroup are kept within BIT K iterations of each
//     other so their shared loads hit in L1: gemm_a18's elastic barrier
//     (arrival counters) or, with PACE 1, its hard threadgroup barrier.
//
// The multiply itself is MPP's matmul2d on one simdgroup, which is the public
// path to the 16x16x16 simdgroup matrix unit that gemm_a18 uses directly.
#ifdef MB_BUILD_A18
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#include "mps_epilogue.h"

#define IN_T        __IN_T__
#define OUT_T       __OUT_T__
#define UM          __UM__        // unrollM: 16-row fragments per simdgroup
#define UN          __UN__        // unrollN: 16-column fragments per simdgroup
#define SM          __SM__        // simdM: simdgroups along M in a threadgroup
#define SN          __SN__        // simdN: simdgroups along N in a threadgroup
#define KS          __KS__        // kSplits: simdgroup groups splitting K
#define KC          __KC__        // tileKPerKSplit = unrollK * 16
#define BIT         __BIT__       // barrierIterations (0: no pacing)
#define PACE        __PACE__      // 1: hard barrier every BIT iterations, 2: elastic
#define RW          __RW__        // rasterMBias: uber tiles advance along M first
#define LIN         __LIN__       // linearize: 1-D threadgroup grid
#define TRANS_A     __TRANS_A__
#define TRANS_B     __TRANS_B__
#define MN_ALIGNED  __MN_ALIGNED__   // M % (UM*16*SM) == 0 && N % (UN*16*SN) == 0
#define K_ALIGNED   __K_ALIGNED__    // K % KC == 0

#define TSM    (UM * 16)          // tileMPerSimdM
#define TSN    (UN * 16)          // tileNPerSimdN
#define TILE_M (TSM * SM)
#define TILE_N (TSN * SN)
#define NSG    (SM * SN * KS)
static_assert(NSG <= 32, "threadgroup exceeds 1024 threads");
// MPP accepts K of 16, 32 or 64 for a register-staged operand, so a KC chunk
// is NSUB sub-chunks of KD.
#define KD     ((KC % 64 == 0) ? 64 : ((KC % 32 == 0) ? 32 : 16))
#define NSUB   (KC / KD)
static_assert(KC % KD == 0 && NSUB >= 1 && NSUB <= 4, "unrollK must be 1..4 or a multiple of 4 up to 16");

#ifndef BATCHED
#define BATCHED 0
#endif
#if EPILOGUE
#define MB_BATCH_BUF 8
#else
#define MB_BATCH_BUF 4
#endif
#if BATCHED && EPILOGUE
#define MB_BBAT _bbat
#else
#define MB_BBAT 0
#endif
#ifndef GLOBAL_SPLITS
#define GLOBAL_SPLITS 1
#endif
#ifndef GLOBAL_ATOMIC
#define GLOBAL_ATOMIC 0
#endif
#if BATCHED
#define MB_GLOBAL_BUF (MB_BATCH_BUF + 1)
#else
#define MB_GLOBAL_BUF MB_BATCH_BUF
#endif

struct MBA18Params {
    int M, N, K, lda, ldb, ldc;
    int subTilesN, subTilesM;       // tiles per uber tile (Apple: uberN, uberM)
    int uberTilesN, uberTilesM;     // uber tiles along N and M
    int kIters;                     // ceil(ceil(K / KC) / KS), uniform across simdgroups
    int batchCount;
};

typedef tensor<device IN_T, dextents<int32_t, 2>, tensor_inline>  mb_dev_in_t;
typedef tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> mb_dev_out_t;

kernel void a18_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant MBA18Params& p [[buffer(3)]],
#if EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride  [[buffer(5)]],
    constant float& beta     [[buffer(6)]],
    constant float& alpha    [[buffer(7)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(MB_BATCH_BUF)]],   // (sA, sB, sC, sBias) per-batch element strides
#endif
#if GLOBAL_SPLITS > 1
    device coherent(device) float *partials [[buffer(MB_GLOBAL_BUF)]],
  #if GLOBAL_ATOMIC
    device coherent(device) atomic_uint *counters [[buffer(MB_GLOBAL_BUF + 1)]],
  #endif
#endif
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    // --- threadgroup index -> (uber tile, sub tile, batch), as gemm_a18 ---
    uint subN, subM, uN, uM, bidx;
#if LIN
    uint t = tgid.x;
    subN = t % uint(p.subTilesN);  t /= uint(p.subTilesN);
    subM = t % uint(p.subTilesM);  t /= uint(p.subTilesM);
  #if RW
    uM = t % uint(p.uberTilesM);   t /= uint(p.uberTilesM);
    uN = t % uint(p.uberTilesN);   t /= uint(p.uberTilesN);
  #else
    uN = t % uint(p.uberTilesN);   t /= uint(p.uberTilesN);
    uM = t % uint(p.uberTilesM);   t /= uint(p.uberTilesM);
  #endif
    const uint globalSplit = t % GLOBAL_SPLITS;
    bidx = t / GLOBAL_SPLITS;
#else
    subN = tgid.x % uint(p.subTilesN);
    subM = tgid.x / uint(p.subTilesN);
    uint t = tgid.y;
    uN = t % uint(p.uberTilesN);   t /= uint(p.uberTilesN);
    uM = t;
    bidx = tgid.z;
    const uint globalSplit = 0;
#endif
    const int n_tile = int(uN * uint(p.subTilesN) + subN);
    const int m_tile = int(uM * uint(p.subTilesM) + subM);
    if (n_tile * TILE_N >= p.N || m_tile * TILE_M >= p.M) return;   // threadgroup-uniform
#if BATCHED
    A += (int64_t)bidx * (int64_t)batch.x;
    B += (int64_t)bidx * (int64_t)batch.y;
    C += (int64_t)bidx * (int64_t)batch.z;
  #if EPILOGUE
    int _bbat = (int)bidx * batch.w;
  #endif
#else
    (void)bidx;
#endif

    // --- simdgroup -> (K split group, position in the SM x SN array) ---
    const uint ks  = sgid / (SM * SN);
    const uint pos = sgid % (SM * SN);
    int m_off = m_tile * TILE_M + int(pos % SM) * TSM;
    int n_off = n_tile * TILE_N + int(pos / SM) * TSN;

    const int gM = p.M, gN = p.N;
#if GLOBAL_SPLITS > 1
    const int partK = p.K / GLOBAL_SPLITS;
    const int kStart = int(globalSplit) * partK;
    const int gK = globalSplit == GLOBAL_SPLITS - 1 ? p.K - kStart : partK;
    A += TRANS_A ? int64_t(kStart) * p.lda : kStart;
    B += TRANS_B ? kStart : int64_t(kStart) * p.ldb;
#else
    const int gK = p.K;
#endif
#if MN_ALIGNED
    const bool inside = true, active = true;
#else
    // Simdgroup-uniform.  A block with no valid element still joins the
    // barriers below.  A block that straddles the edge is shifted back inside
    // when the matrix is at least one block wide: it recomputes a few rows or
    // columns the neighbouring block also produces, with identical values, and
    // keeps the unmasked fast path.  Only matrices thinner than a block take
    // the bounds-checked path.
    const bool active = (m_off < gM) && (n_off < gN);
#if GLOBAL_SPLITS == 1
    if (m_off + TSM > gM && gM >= TSM) m_off = gM - TSM;
    if (n_off + TSN > gN && gN >= TSN) n_off = gN - TSN;
#endif
    const bool inside = (m_off + TSM <= gM) && (n_off + TSN <= gN);
#endif

    const array<int32_t, 2> sA_ = {1, p.lda};
    const array<int32_t, 2> sB_ = {1, p.ldb};
    // Interior work uses static slices (no bounds checks).  Edge work uses
    // views whose origin is the block and whose extents are exactly the valid
    // region, so MPP masks the rest.  K tails of interior blocks keep static
    // row/column extents and only the K extent is dynamic.
#if TRANS_A
    mb_dev_in_t tA(A, dextents<int32_t, 2>(gM, gK), sA_);
  #define MB_A_SUB(k0)         tA.slice<TSM, KD>(m_off, k0)
  #define MB_A_VIEW(k0, kend)  mb_dev_in_t(A + m_off + (int64_t)(k0) * p.lda, dextents<int32_t, 2>(gM - m_off, (kend) - (k0)), sA_)
  #define MB_A_TAIL(vA, kb)    vA.slice<TSM, dynamic_extent>(m_off, kb)
  #define MB_A_TAIL_EXT(ke)    dextents<int32_t, 2>(gM, ke)
#else
    mb_dev_in_t tA(A, dextents<int32_t, 2>(gK, gM), sA_);
  #define MB_A_SUB(k0)         tA.slice<KD, TSM>(k0, m_off)
  #define MB_A_VIEW(k0, kend)  mb_dev_in_t(A + (k0) + (int64_t)m_off * p.lda, dextents<int32_t, 2>((kend) - (k0), gM - m_off), sA_)
  #define MB_A_TAIL(vA, kb)    vA.slice<dynamic_extent, TSM>(kb, m_off)
  #define MB_A_TAIL_EXT(ke)    dextents<int32_t, 2>(ke, gM)
#endif
#if TRANS_B
    mb_dev_in_t tB(B, dextents<int32_t, 2>(gK, gN), sB_);
  #define MB_B_SUB(k0)         tB.slice<KD, TSN>(k0, n_off)
  #define MB_B_VIEW(k0, kend)  mb_dev_in_t(B + (k0) + (int64_t)n_off * p.ldb, dextents<int32_t, 2>((kend) - (k0), gN - n_off), sB_)
  #define MB_B_TAIL(vB, kb)    vB.slice<dynamic_extent, TSN>(kb, n_off)
  #define MB_B_TAIL_EXT(ke)    dextents<int32_t, 2>(ke, gN)
#else
    mb_dev_in_t tB(B, dextents<int32_t, 2>(gN, gK), sB_);
  #define MB_B_SUB(k0)         tB.slice<TSN, KD>(n_off, k0)
  #define MB_B_VIEW(k0, kend)  mb_dev_in_t(B + n_off + (int64_t)(k0) * p.ldb, dextents<int32_t, 2>(gN - n_off, (kend) - (k0)), sB_)
  #define MB_B_TAIL(vB, kb)    vB.slice<TSN, dynamic_extent>(n_off, kb)
  #define MB_B_TAIL_EXT(ke)    dextents<int32_t, 2>(gN, ke)
#endif
    mb_dev_out_t tC(C, dextents<int32_t, 2>(gN, gM), array<int32_t, 2>{1, p.ldc});

    constexpr auto desc = matmul2d_descriptor(
        TSM, TSN, KD, TRANS_A, TRANS_B, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    // The destination layout depends only on the element types, so one
    // accumulator serves the static (interior) and dynamic (edge/tail) runs.
    auto cT = op.get_destination_cooperative_tensor<
        decltype(MB_A_SUB(0)), decltype(MB_B_SUB(0)), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

#if K_ALIGNED
  #define MB_FULL(k0) ((k0) < gK)
#else
  #define MB_FULL(k0) ((k0) + KC <= gK)
#endif
    // Partial chunks run at most KD wide at a time: MPP mishandles a dynamic K
    // extent larger than the op's K tile.
#define MB_TAIL_RUN(k0) do {                                                   \
        const int kend_ = min((k0) + KC, gK);                                 \
        for (int kb_ = (k0); kb_ < kend_; kb_ += KD) {                        \
            const int ke_ = min(kb_ + KD, kend_);                             \
            mb_dev_in_t vA_(A, MB_A_TAIL_EXT(ke_), sA_);                      \
            mb_dev_in_t vB_(B, MB_B_TAIL_EXT(ke_), sB_);                      \
            auto sA_t = MB_A_TAIL(vA_, kb_);                                  \
            auto sB_t = MB_B_TAIL(vB_, kb_);                                  \
            op.run(sA_t, sB_t, cT);                                           \
        }                                                                     \
    } while (0)
#define MB_EDGE_RUN(k0) do {                                                   \
        const int kend_ = min((k0) + KC, gK);                                 \
        for (int kb_ = (k0); kb_ < kend_; kb_ += KD) {                        \
            const int ke_ = min(kb_ + KD, kend_);                             \
            mb_dev_in_t vA_ = MB_A_VIEW(kb_, ke_);                            \
            mb_dev_in_t vB_ = MB_B_VIEW(kb_, ke_);                            \
            op.run(vA_, vB_, cT);                                             \
        }                                                                     \
    } while (0)

    // Register-staged A: the NSUB sub-chunks of a KC chunk are loaded, then
    // each is multiplied with B streamed by MPP.  The iteration count is
    // uniform across the threadgroup so the pacing and the K-split reduction
    // barrier are reached by every simdgroup.
    auto cA0 = op.get_left_input_cooperative_tensor<IN_T, IN_T, float>();
#if NSUB > 1
    auto cA1 = op.get_left_input_cooperative_tensor<IN_T, IN_T, float>();
#endif
#if NSUB > 2
    auto cA2 = op.get_left_input_cooperative_tensor<IN_T, IN_T, float>();
#endif
#if NSUB > 3
    auto cA3 = op.get_left_input_cooperative_tensor<IN_T, IN_T, float>();
#endif
#if PACE == 2 && BIT > 0
    // gemm_a18's elastic barrier: per-iteration arrival counters; a simdgroup
    // may start iteration `it` once every simdgroup has finished iteration
    // `it - BIT`.  Counters live in a ring and only ever grow, so slot (j % R)
    // reads NSG * (j / R + 1) once iteration j is complete everywhere.
    #define MB_RING 16
    threadgroup atomic_int mb_cnt[MB_RING];
    if (sgid == 0 && lane < MB_RING) atomic_store_explicit(&mb_cnt[lane], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    const int kIters = GLOBAL_SPLITS > 1 ? (gK + KC * KS - 1) / (KC * KS) : p.kIters;
    for (int it = 0; it < kIters; ++it) {
#if PACE == 2 && BIT > 0
        if (it >= BIT) {
            const int j = it - BIT;
            const int need = NSG * (j / MB_RING + 1);
            while (atomic_load_explicit(&mb_cnt[j % MB_RING], memory_order_relaxed) < need) { }
        }
#elif PACE == 1 && BIT > 0
        if ((it % BIT) == 0)
            threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        const int k0 = (it * KS + int(ks)) * KC;
        const bool work = active && MB_FULL(k0) && inside;
        if (work) {
            cA0.load(MB_A_SUB(k0));
#if NSUB > 1
            cA1.load(MB_A_SUB(k0 + KD));
#endif
#if NSUB > 2
            cA2.load(MB_A_SUB(k0 + 2 * KD));
#endif
#if NSUB > 3
            cA3.load(MB_A_SUB(k0 + 3 * KD));
#endif
            { auto sB = MB_B_SUB(k0);          op.run(cA0, sB, cT); }
#if NSUB > 1
            { auto sB = MB_B_SUB(k0 + KD);     op.run(cA1, sB, cT); }
#endif
#if NSUB > 2
            { auto sB = MB_B_SUB(k0 + 2 * KD); op.run(cA2, sB, cT); }
#endif
#if NSUB > 3
            { auto sB = MB_B_SUB(k0 + 3 * KD); op.run(cA3, sB, cT); }
#endif
        } else if (active && k0 < gK) {
            if (inside) MB_TAIL_RUN(k0); else MB_EDGE_RUN(k0);
        }
#if PACE == 2 && BIT > 0
        if (lane == 0) atomic_fetch_add_explicit(&mb_cnt[it % MB_RING], 1, memory_order_relaxed);
#endif
    }

#if KS > 1
    // Local K-split reduction: groups 1..KS-1 park their accumulators in
    // threadgroup memory, group 0 adds them and stores the tile.  Slots are
    // limited to 16 KB, so large tiles reduce in several waves.
  #define RED_SLOT      (SM * SN * TSM * TSN)              // floats per K-split group
  #define RED_SLOTS_RAW (4096 / RED_SLOT)
  #if RED_SLOTS_RAW < 1
    #define RED_SLOTS 1
  #elif RED_SLOTS_RAW >= (KS - 1)
    #define RED_SLOTS (KS - 1)
  #else
    #define RED_SLOTS RED_SLOTS_RAW
  #endif
    threadgroup float red[RED_SLOTS * RED_SLOT];
    typedef tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> mb_tg_t;
    for (int w = 0; w < KS - 1; w += RED_SLOTS) {
        const int g = int(ks) - 1 - w;                       // my slot in this wave
        if (ks != 0 && active && g >= 0 && g < RED_SLOTS) {
            mb_tg_t tR(red + (g * (SM * SN) + int(pos)) * (TSM * TSN),
                       dextents<int32_t, 2>(TSN, TSM), array<int32_t, 2>{1, TSN});
            cT.store(tR);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks == 0 && active) {
            const int n = min(RED_SLOTS, KS - 1 - w);
            for (int sIdx = 0; sIdx < n; ++sIdx) {
                mb_tg_t tR(red + (sIdx * (SM * SN) + int(pos)) * (TSM * TSN),
                           dextents<int32_t, 2>(TSN, TSM), array<int32_t, 2>{1, TSN});
                auto part = op.get_destination_cooperative_tensor<
                    decltype(MB_A_SUB(0)), decltype(MB_B_SUB(0)), float>();
                part.load(tR);
                #pragma unroll
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] += part[i];
            }
        }
        if (w + RED_SLOTS < KS - 1)
            threadgroup_barrier(mem_flags::mem_threadgroup);   // slots are reused
    }
    if (ks != 0 || !active) return;
#elif GLOBAL_SPLITS == 1
    if (!active) return;
#endif

#if GLOBAL_SPLITS > 1
    const ulong matrixSize = ulong(gM) * gN;
    const ulong splitStride = ulong(p.batchCount) * matrixSize;
    const ulong matrixOffset = ulong(bidx) * matrixSize;
    // Every split stores a disjoint float32 partial before any conversion or
    // linked epilogue. Global partitions use floor(K / splits), with all
    // remaining elements in the last partition.
    for (uint16_t e = 0; e < cT.get_capacity(); ++e) {
        auto idx = cT.get_multidimensional_index(e);
        int r = m_off + int(idx[1]), c = n_off + int(idx[0]);
        if (active && r < gM && c < gN)
            partials[ulong(globalSplit) * splitStride + matrixOffset + ulong(r) * gN + c] = cT[e];
    }
  #if GLOBAL_ATOMIC
    // AIR uses coherent stores, device fences, and a relaxed CAS counter.
    // The last completed threadgroup reduces splits in ascending order.
    threadgroup bool doGlobalReduction;
    if (sgid == 0 && lane == 0) doGlobalReduction = false;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (sgid == 0 && lane == 0) {
        const uint tilesM = (gM + TILE_M - 1) / TILE_M;
        const uint tilesN = (gN + TILE_N - 1) / TILE_N;
        const ulong counter = (ulong(bidx) * tilesM + m_tile) * tilesN + n_tile;
        uint expected = 0;
        while (!atomic_compare_exchange_weak_explicit(counters + counter, &expected, expected + 1,
                                                       memory_order_relaxed, memory_order_relaxed)) {}
        doGlobalReduction = expected == GLOBAL_SPLITS - 1;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!doGlobalReduction) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (!active) return;
    for (uint16_t e = 0; e < cT.get_capacity(); ++e) {
        auto idx = cT.get_multidimensional_index(e);
        int r = m_off + int(idx[1]), c = n_off + int(idx[0]);
        float sum = 0.0f;
        if (r < gM && c < gN) {
            const ulong offset = matrixOffset + ulong(r) * gN + c;
            for (uint split = 0; split < GLOBAL_SPLITS; ++split)
                sum += partials[ulong(split) * splitStride + offset];
        }
        cT[e] = sum;
    }
  #else
    return;
  #endif
#endif

    auto cO = op.get_destination_cooperative_tensor<
        decltype(MB_A_SUB(0)), decltype(MB_B_SUB(0)), OUT_T>();
#if EPILOGUE
    #pragma unroll
    for (uint16_t e = 0; e < cT.get_capacity(); ++e) {
        auto idx = cT.get_multidimensional_index(e);
        int r = m_off + (int)idx[1], c = n_off + (int)idx[0];
        bool ok = inside || (r < gM && c < gN);
#if MB_MPS_EPILOGUE
        cO[e] = ok ? mb_mps_epilogue(cT[e], bias,
                         ulong(MB_BBAT) + ulong(r) * bstride.x + ulong(c) * bstride.y)
                   : (OUT_T)0;
#else
        cO[e] = ok ? mb_epi<OUT_T, float, float>(
                         cT[e], bias, MB_BBAT + r * bstride.x + c * bstride.y, beta, alpha)
                   : (OUT_T)0;
#endif
    }
#else
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
#endif
    if (inside) {
        auto mC = tC.slice<TSN, TSM>(n_off, m_off);
        cO.store(mC);
    } else {
        mb_dev_out_t vC(C + n_off + (int64_t)m_off * p.ldc,
                        dextents<int32_t, 2>(gN - n_off, gM - m_off), array<int32_t, 2>{1, p.ldc});
        cO.store(vC);                          // clipped to the valid region
    }
}
#endif  // MB_BUILD_A18
