// apple_gemm.h - MPP GEMM driven by the tile, raster and K split MPS picks for the shape (see ../apple_heuristics.py).
#ifdef MB_BUILD_APPLE_GEMM
#include <metal_stdlib>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T       __IN_T__
#define OUT_T      __OUT_T__
#define TM         __TM__          // rows per simdgroup
#define TN         __TN__          // columns per simdgroup
#define SM         __SM__          // simdgroups along M
#define SN         __SN__          // simdgroups along N
#define KS         __KS__          // simdgroup groups splitting K inside a threadgroup
#define KC         __KC__          // K per step (one matmul2d call)
#define STAGE      __STAGE__       // operand loaded into registers first: 0 none, 1 A, 2 B, 3 A two steps at a time
#define PACE       __PACE__        // every BIT steps 1: simdgroups wait for those over LEAD steps behind, 2: a barrier
#define BIT        __BIT__
#define LEAD       2
#define RW         __RW__          // uber tiles advance along M first
#define LIN        __LIN__         // 1-D grid
#define SUB_M      __SUB_M__       // tiles per uber tile along M and N
#define SUB_N      __SUB_N__
#define WHOLE_K    __WHOLE_K__     // one matmul2d call over all of K
#define COOP       __COOP__        // with WHOLE_K: one call by all simdgroups on the BM x BN tile
#define TRANS_A    __TRANS_A__
#define TRANS_B    __TRANS_B__
#define MN_ALIGNED __MN_ALIGNED__
#define K_ALIGNED  __K_ALIGNED__
#define STORE_TG   __STORE_TG__    // store the tile through threadgroup memory (one K step, 16-byte aligned rows of C)

#ifndef BATCHED
#define BATCHED 0
#endif
#ifndef BATCH_ROWS
#define BATCH_ROWS 0   // batch.x and batch.y give rows per plane of A and B rather than elements
#endif
#ifndef SPLITK
#define SPLITK 0
#endif
#define BM (TM * SM)
#define BN (TN * SN)
#define NSG (SM * SN * KS)
#define AG_EPILOGUE (EPILOGUE && !SPLITK)
#if SPLITK
#define AG_C_T float
#else
#define AG_C_T OUT_T
#endif
#if AG_EPILOGUE
#define AG_BATCH_BUF 8
#else
#define AG_BATCH_BUF 4
#endif

// Grid geometry: tiles per uber tile, tiles, steps per simdgroup, and the global K split.
struct AppleGemmDims { int M, N, K, lda, ldb, ldc, sub_n, sub_m, tiles_n, tiles_m, k_iters, k_chunk, splits, batch,
                       line_mul, line_shift, plane_mul, plane_shift; };

typedef tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> ag_in_t;
typedef tensor<device AG_C_T, dextents<int32_t, 2>, tensor_inline> ag_out_t;

kernel void apple_gemm(
    device IN_T *A [[buffer(0)]],
    device IN_T *B [[buffer(1)]],
    device AG_C_T *C [[buffer(2)]],
    constant AppleGemmDims& d [[buffer(3)]],
#if AG_EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride  [[buffer(5)]],
    constant float& beta     [[buffer(6)]],
    constant float& alpha    [[buffer(7)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(AG_BATCH_BUF)]],   // (sA, sB, sC, sBias) per-batch element strides
#endif
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    // Threadgroup -> tile, as MPS walks them: N-first inside an uber tile, uber tiles M- or N-first, then split, batch.
#if LIN
    // One threadgroup per tile: uber tiles at the far edges are clipped to the tiles left. Planes and lines of uber
    // tiles divide by the host's multiply-shift constants (n / d == (mulhi(n, mul) + n) >> shift for n < 2^31).
  #define AG_DIV(n, f) ((mulhi((n), uint(d.f##_mul)) + (n)) >> uint(d.f##_shift))
    uint t = tgid.x;
    const uint tiles_m = uint(d.tiles_m), tiles_n = uint(d.tiles_n), plane = AG_DIV(t, plane);
    t -= plane * tiles_m * tiles_n;
  #if RW
    const uint u_n = AG_DIV(t, line);
    t -= u_n * tiles_m * uint(SUB_N);
    const uint w = min(uint(SUB_N), tiles_n - u_n * uint(SUB_N)), u_m = t / (uint(SUB_M) * w);
    t -= u_m * uint(SUB_M) * w;
    const uint sub_n = t % w, sub_m = t / w;
  #else
    const uint u_m = AG_DIV(t, line);
    t -= u_m * tiles_n * uint(SUB_M);
    const uint h = min(uint(SUB_M), tiles_m - u_m * uint(SUB_M)), u_n = t / (uint(SUB_N) * h);
    t -= u_n * uint(SUB_N) * h;
    const uint w = min(uint(SUB_N), tiles_n - u_n * uint(SUB_N)), sub_n = t % w, sub_m = t / w;
  #endif
    t = plane;
#else
    const uint ubers_n = (uint(d.tiles_n) + uint(SUB_N) - 1) / uint(SUB_N);
    const uint sub_n = tgid.x % uint(SUB_N), sub_m = tgid.x / uint(SUB_N);
    const uint u_n = tgid.y % ubers_n, u_m = tgid.y / ubers_n;
    uint t = tgid.z;
#endif
    const int m_tile = int(u_m * uint(SUB_M) + sub_m), n_tile = int(u_n * uint(SUB_N) + sub_n);
    if (m_tile * BM >= d.M || n_tile * BN >= d.N) return;
    const int gM = d.M, gN = d.N;
    int gK = d.K;
#if SPLITK
    const int split = int(t % uint(d.splits));
    t /= uint(d.splits);
    const int k_begin = split * d.k_chunk;
    gK = split == d.splits - 1 ? gK - k_begin : d.k_chunk;   // the last split takes the remainder
    A += TRANS_A ? int64_t(k_begin) * d.lda : k_begin;
    B += TRANS_B ? k_begin : int64_t(k_begin) * d.ldb;
    C += (int64_t(t) * d.splits + split) * int64_t(gM) * gN;
    const int ldc = gN;
#else
    const int ldc = d.ldc;
#endif
#if BATCH_ROWS
    // Planes starting on whole rows are reached through A's and B's row coordinates: each operand pointer moved per
    // threadgroup cost about 0.3% on M5, even with one plane.
    const int ra = int(t) * batch.x, rb = int(t) * batch.y;
#else
    constexpr int ra = 0, rb = 0;
  #if BATCHED
    A += int64_t(t) * batch.x;
    B += int64_t(t) * batch.y;
  #endif
#endif
#if BATCHED && !SPLITK
    C += int64_t(t) * batch.z;
#endif
#if AG_EPILOGUE && BATCHED
    const int bias_off = int(t) * batch.w;
#elif AG_EPILOGUE
    const int bias_off = 0;
#endif

    const ushort ks = sgid / (SM * SN), pos = sgid % (SM * SN);
#if COOP
    int m_off = m_tile * BM, n_off = n_tile * BN;
    constexpr int OM = BM, ON = BN;
#else
    int m_off = m_tile * BM + int(pos % SM) * TM, n_off = n_tile * BN + int(pos / SM) * TN;
    constexpr int OM = TM, ON = TN;
#endif
#if WHOLE_K && !COOP && KS == 1 && !STORE_TG && !MN_ALIGNED && (SM % 2 == 0 || SN % 2 == 0)
    // A tile whose rows (columns) end within its first half: the simdgroups of the empty half take the second half
    // of K for the blocks of the full half (edge-free products leave the edge paths and their memory out).
    const int eks_r = int(pos % SM), eks_c = int(pos / SM);
    const bool eks_m = SM % 2 == 0 && gM - m_tile * BM <= BM / 2;
    const bool eks_n = !eks_m && SN % 2 == 0 && gN - n_tile * BN <= BN / 2, eks = eks_m || eks_n;
    const bool eks_partner = eks_m ? eks_r >= SM / 2 : eks_n && eks_c >= SN / 2;
    if (eks_partner) {
        if (eks_m)
            m_off -= SM / 2 * TM;
        else
            n_off -= SN / 2 * TN;
    }
#else
    constexpr bool eks = false, eks_partner = false;
#endif
#if MN_ALIGNED
    const bool active = true, in_m = true, in_n = true;
#else
    // A block straddling the edge is shifted back inside when the matrix is wide enough; it recomputes a few values,
    // which an epilogue reading the output (in-place addmm) must not see twice, and leaves the threadgroup's tile.
    const bool active = m_off < gM && n_off < gN;
  #if !AG_EPILOGUE && !STORE_TG
    if (m_off + OM > gM && gM >= OM) m_off = gM - OM;
    if (n_off + ON > gN && gN >= ON) n_off = gN - ON;
  #endif
    const bool in_m = m_off + OM <= gM, in_n = n_off + ON <= gN;
#endif
    const bool inside = in_m && in_n;
#if !PACE && KS == 1 && !COOP && !STORE_TG
    if (!active && !eks)
        return;   // no threadgroup barrier follows
#endif

    // Tensor extents are (columns, rows) of the stored matrices.
    using Ext = dextents<int32_t, 2>;
    const array<int32_t, 2> sa = {1, d.lda}, sb = {1, d.ldb};
    ag_in_t tA(A, TRANS_A ? Ext(gM, ra + gK) : Ext(gK, ra + gM), sa);
    ag_in_t tB(B, TRANS_B ? Ext(gK, rb + gN) : Ext(gN, rb + gK), sb);
    ag_out_t tC(C, Ext(gN, gM), array<int32_t, 2>{1, ldc});
#if TRANS_A
  #define A_FULL(k0, w) tA.template slice<OM, w>(m_off, ra + (k0))
  #define A_ROWS(k0, ke) ag_in_t(A + m_off + int64_t(ra + (k0)) * d.lda, Ext(gM - m_off, (ke) - (k0)), sa)
#else
  #define A_FULL(k0, w) tA.template slice<w, OM>(k0, ra + m_off)
  #define A_ROWS(k0, ke) ag_in_t(A + (k0) + int64_t(ra + m_off) * d.lda, Ext((ke) - (k0), gM - m_off), sa)
#endif
#if TRANS_B
  #define B_FULL(k0, w) tB.template slice<w, ON>(k0, rb + n_off)
  #define B_COLS(k0, ke) ag_in_t(B + (k0) + int64_t(rb + n_off) * d.ldb, Ext((ke) - (k0), gN - n_off), sb)
#else
  #define B_FULL(k0, w) tB.template slice<ON, w>(n_off, rb + (k0))
  #define B_COLS(k0, ke) ag_in_t(B + n_off + int64_t(rb + (k0)) * d.ldb, Ext(gN - n_off, (ke) - (k0)), sb)
#endif
#if TRANS_A
  #define A_SLICE(k0, ke, h) ag_in_t(A + int64_t(ra + (k0)) * d.lda, Ext(gM, (ke) - (k0)), sa).template slice<h, dynamic_extent>(m_off, 0)
  #define A_FIX(k0, w, h) tA.template slice<h, w>(m_off, ra + (k0))
#else
  #define A_SLICE(k0, ke, h) ag_in_t(A + (k0) + int64_t(ra) * d.lda, Ext((ke) - (k0), gM), sa).template slice<dynamic_extent, h>(0, m_off)
  #define A_FIX(k0, w, h) tA.template slice<w, h>(k0, ra + m_off)
#endif
#if TRANS_B
  #define B_SLICE(k0, ke, w) ag_in_t(B + (k0) + int64_t(rb) * d.ldb, Ext((ke) - (k0), gN), sb).template slice<dynamic_extent, w>(0, n_off)
  #define B_FIX(k0, kw, w) tB.template slice<kw, w>(k0, rb + n_off)
#else
  #define B_SLICE(k0, ke, w) ag_in_t(B + int64_t(rb + (k0)) * d.ldb, Ext(gN, (ke) - (k0)), sb).template slice<w, dynamic_extent>(n_off, 0)
  #define B_FIX(k0, kw, w) tB.template slice<w, kw>(n_off, rb + (k0))
#endif
// acc = the product of an R x C block over K range [k0, ke), from static slices when the block is inside the matrix.
// MPP's whole-K call pays two 32-chunks for a range ending mid-chunk: it takes the 32-aligned part, and calls of 16
// (MPP's smallest static K) take the rest.
#define AG_RUN(op_, acc_, R, C, EXEC, k0_, ke_, full) { \
    const int ag_lo = (k0_), ag_hi = (ke_), ag_mid = ag_lo + ((ag_hi - ag_lo) & ~31); \
    if (ag_mid > ag_lo && (full)) { \
        auto sA = A_SLICE(ag_lo, ag_mid, R); \
        auto sB = B_SLICE(ag_lo, ag_mid, C); \
        op_.run(sA, sB, acc_); \
    } else if (ag_mid > ag_lo) { \
        ag_in_t vA = A_ROWS(ag_lo, ag_mid); \
        ag_in_t vB = B_COLS(ag_lo, ag_mid); \
        op_.run(vA, vB, acc_); \
    } else { \
        for (uint16_t i = 0; i < acc_.get_capacity(); ++i) \
            acc_[i] = 0.0f; \
    } \
    if (ag_mid < ag_hi) { \
        constexpr auto ag_dt = matmul2d_descriptor(R, C, 16, TRANS_A, TRANS_B, true, matmul2d_descriptor::mode::multiply_accumulate); \
        matmul2d<ag_dt, EXEC> ag_ot; \
        for (int k = ag_mid; k < ag_hi; k += 16) { \
            if ((full) && k + 16 <= ag_hi) { \
                auto sA = A_FIX(k, 16, R); \
                auto sB = B_FIX(k, 16, C); \
                ag_ot.run(sA, sB, acc_); \
            } else { \
                ag_in_t vA = A_ROWS(k, min(k + 16, ag_hi)); \
                ag_in_t vB = B_COLS(k, min(k + 16, ag_hi)); \
                ag_ot.run(vA, vB, acc_); \
            } \
        } \
    } }

#if KS > 1
    // Partial sums of the simdgroups splitting K (the K-split reduction and edge tiles).
    constexpr int SLOT = SM * SN * TM * TN;
    constexpr int SLOTS = 4096 / SLOT < 1 ? 1 : (4096 / SLOT < KS - 1 ? 4096 / SLOT : KS - 1);
    threadgroup float red[SLOTS * SLOT];
    using TgExt = tensor<threadgroup float, Ext, tensor_inline>;
#endif
#if WHOLE_K
  #define AG_KW dynamic_extent
  #define AG_EXEC conditional_t<COOP, execution_simdgroups<SM * SN>, execution_simdgroup>
    constexpr auto desc = matmul2d_descriptor(OM, ON, static_cast<int>(dynamic_extent), TRANS_A, TRANS_B, true);
    matmul2d<desc, AG_EXEC> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(A_FULL(0, AG_KW)), decltype(B_FULL(0, AG_KW)), float>();
  #if !COOP && KS == 1 && !STORE_TG && !MN_ALIGNED && SM == 2 && TM % 16 == 0 && (TM % 32 == 0 || TN % 16 == 0)
    if (eks_m && gM - m_tile * BM <= TM / 2) {
        // The tile's rows fit half a block: blocks half as tall, the two simdgroups of a column each taking half of K.
        constexpr auto desc_h = matmul2d_descriptor(TM / 2, TN, static_cast<int>(dynamic_extent), TRANS_A, TRANS_B, true);
        matmul2d<desc_h, execution_simdgroup> op_h;
        auto acc_h = op_h.template get_destination_cooperative_tensor<ag_in_t, ag_in_t, float>();
        auto part_h = op_h.template get_destination_cooperative_tensor<ag_in_t, ag_in_t, float>();
    #if AG_EPILOGUE
        m_off = m_tile * BM;   // the rows above belong to another tile: an in-place epilogue must not see them twice
    #else
        m_off = max(0, gM - TM / 2);   // ends at the last row (recomputing a few of the tile above)
    #endif
        const int mid = (gK / 2) & ~31, k_lo = eks_partner ? mid : 0, k_hi = eks_partner ? gK : mid;
        const bool live = n_off < gN && m_off < gM;
        if (live && k_hi > k_lo)
            AG_RUN(op_h, acc_h, TM / 2, TN, execution_simdgroup, k_lo, k_hi, false)
        else {
            #pragma unroll
            for (uint16_t i = 0; i < acc_h.get_capacity(); ++i)
                acc_h[i] = 0.0f;
        }
        threadgroup float half_part[SN * TM / 2 * TN];
        tensor<threadgroup float, Ext, tensor_inline> part_t(half_part + eks_c * TM / 2 * TN, Ext(TN, TM / 2), array<int32_t, 2>{1, TN});
        if (eks_partner && live)
            acc_h.store(part_t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (eks_partner || !live)
            return;
        part_h.load(part_t);
    #if SPLITK
        #pragma unroll
        for (uint16_t i = 0; i < acc_h.get_capacity(); ++i)
            acc_h[i] += part_h[i];
        acc_h.store(ag_out_t(C + n_off + int64_t(m_off) * ldc, Ext(gN - n_off, gM - m_off), array<int32_t, 2>{1, ldc}));
    #else
        auto res_h = op_h.template get_destination_cooperative_tensor<ag_in_t, ag_in_t, OUT_T>();
        #pragma unroll
        for (uint16_t i = 0; i < acc_h.get_capacity(); ++i) {
      #if AG_EPILOGUE
            if (!acc_h.is_valid_element(i))
                continue;
            auto idx = acc_h.get_multidimensional_index(i);
            res_h[i] = mb_epi<OUT_T, float, float>(acc_h[i] + part_h[i], bias,
                                                   bias_off + (m_off + int(idx[1])) * bstride.x + (n_off + int(idx[0])) * bstride.y, beta, alpha);
      #else
            res_h[i] = static_cast<OUT_T>(acc_h[i] + part_h[i]);
      #endif
        }
        res_h.store(ag_out_t(C + n_off + int64_t(m_off) * ldc, Ext(gN - n_off, gM - m_off), array<int32_t, 2>{1, ldc}));
    #endif
        return;
    }
  #endif
  #if KS == 4 && SM == 1 && SN == 1 && TM % 32 == 0 && TN % 32 == 0 && !MN_ALIGNED
    if (gM - m_tile * BM <= BM / 2 || gN - n_tile * BN <= BN / 2) {
        // A tile whose rows (columns) end within its first half takes Apple's 2x2 quadrant blocks instead: the
        // simdgroups of the quadrants past the edge split K with those of the quadrants inside.
        constexpr auto desc_q = matmul2d_descriptor(TM / 2, TN / 2, static_cast<int>(dynamic_extent), TRANS_A, TRANS_B, true);
        matmul2d<desc_q, execution_simdgroup> op_q;
        auto acc_q = op_q.template get_destination_cooperative_tensor<ag_in_t, ag_in_t, float>();
        const int gr = gM - m_tile * BM <= TM / 2 ? 1 : 2, gc = gN - n_tile * BN <= TN / 2 ? 1 : 2, parts = 4 / (gr * gc);
        const int r = int(ks % 2) % gr, c = int(ks / 2) % gc, j = int(ks % 2) / gr + int(ks / 2) / gc * (2 / gr);
        m_off = m_tile * BM + r * (TM / 2);
        n_off = n_tile * BN + c * (TN / 2);
    #if !AG_EPILOGUE
        if (m_off + TM / 2 > gM && gM >= TM / 2) m_off = gM - TM / 2;
        if (n_off + TN / 2 > gN && gN >= TN / 2) n_off = gN - TN / 2;
    #endif
        const int chunks = gK / 32, base = chunks / parts, extra = chunks % parts;
        const int k_lo = 32 * (j * base + min(j, extra)), k_hi = j == parts - 1 ? gK : k_lo + 32 * (base + (j < extra));
        if (k_hi > k_lo)
            AG_RUN(op_q, acc_q, TM / 2, TN / 2, execution_simdgroup, k_lo, k_hi, false)
        else {
            #pragma unroll
            for (uint16_t i = 0; i < acc_q.get_capacity(); ++i)
                acc_q[i] = 0.0f;
        }
        if (parts > 1) {
            // Part j of a quadrant is simdgroup (r + gr (j % (2 / gr))) + 2 (c + gc (j / (2 / gr))).
            constexpr int Q = TM / 2 * TN / 2;
            if (j)
                acc_q.store(TgExt(red + ks * Q, Ext(TN / 2, TM / 2), array<int32_t, 2>{1, TN / 2}));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (j)
                return;
            auto part = op_q.template get_destination_cooperative_tensor<ag_in_t, ag_in_t, float>();
            for (int p = 1; p < parts; ++p) {
                const int src = r + gr * (p % (2 / gr)) + 2 * (c + gc * (p / (2 / gr)));
                part.load(TgExt(red + src * Q, Ext(TN / 2, TM / 2), array<int32_t, 2>{1, TN / 2}));
                #pragma unroll
                for (uint16_t i = 0; i < acc_q.get_capacity(); ++i)
                    acc_q[i] += part[i];
            }
        }
        ag_out_t tQ(C + n_off + int64_t(m_off) * ldc, Ext(gN - n_off, gM - m_off), array<int32_t, 2>{1, ldc});
    #if SPLITK
        acc_q.store(tQ);
    #else
        auto res_q = op_q.template get_destination_cooperative_tensor<ag_in_t, ag_in_t, OUT_T>();
        #pragma unroll
        for (uint16_t i = 0; i < acc_q.get_capacity(); ++i) {
      #if AG_EPILOGUE
            if (!acc_q.is_valid_element(i))
                continue;
            auto idx = acc_q.get_multidimensional_index(i);
            res_q[i] = mb_epi<OUT_T, float, float>(acc_q[i], bias, bias_off + (m_off + int(idx[1])) * bstride.x + (n_off + int(idx[0])) * bstride.y,
                                                   beta, alpha);
      #else
            res_q[i] = static_cast<OUT_T>(acc_q[i]);
      #endif
        }
        res_q.store(tQ);
    #endif
        return;
    }
  #endif
    if (KS > 1 || eks) {
        // This simdgroup's part of K in whole 32-chunks: its group's even share (the last group also takes the
        // tail), or one half in an edge tile.
        const int chunks = gK / 32, base = chunks / KS, extra = chunks % KS, half_k = (gK / 2) & ~31;
        const int k_lo = KS > 1 ? 32 * (int(ks) * base + min(int(ks), extra)) : (eks_partner ? half_k : 0);
        const int k_hi = KS > 1 ? (int(ks) == KS - 1 ? gK : k_lo + 32 * (base + (int(ks) < extra))) : (eks_partner ? gK : half_k);
        if (active && k_hi > k_lo)
            AG_RUN(op, acc, OM, ON, AG_EXEC, k_lo, k_hi, inside)
        else {
            #pragma unroll
            for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                acc[i] = 0.0f;
        }
    } else if (active)
        AG_RUN(op, acc, OM, ON, AG_EXEC, 0, gK, inside)
  #if !COOP && KS == 1 && !STORE_TG && !MN_ALIGNED && (SM % 2 == 0 || SN % 2 == 0)
    if (eks) {
        // The partner leaves its sum in threadgroup memory for the simdgroup that owns the block.
        threadgroup float eks_part[SM * SN / 2 * TM * TN];
        using PartExt = tensor<threadgroup float, Ext, tensor_inline>;
        const int slot = eks_m ? eks_c * (SM / 2) + eks_r % (SM / 2) : eks_c % (SN / 2) * SM + eks_r;
        PartExt part_t(eks_part + slot * TM * TN, Ext(TN, TM), array<int32_t, 2>{1, TN});
        if (eks_partner && active)
            acc.store(part_t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (eks_partner || !active)
            return;
        auto part = op.template get_destination_cooperative_tensor<decltype(A_FULL(0, AG_KW)), decltype(B_FULL(0, AG_KW)), float>();
        part.load(part_t);
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            acc[i] += part[i];
    }
  #endif
#else
  #define AG_KW KC
    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, TRANS_A, TRANS_B, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(A_FULL(0, KC)), decltype(B_FULL(0, KC)), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        acc[i] = 0.0f;
  #if STAGE == 1
    auto rA = op.template get_left_input_cooperative_tensor<IN_T, IN_T, float>();
  #elif STAGE == 2
    auto rB = op.template get_right_input_cooperative_tensor<IN_T, IN_T, float>();
  #endif
  #if PACE == 1
    // Each simdgroup publishes its finished steps; every BIT steps it waits until every simdgroup is at most LEAD steps
    // behind. A count read before its owner clears it only skips or prolongs a wait, so no barrier orders the clears.
    threadgroup atomic_int done[NSG];
    const bool paced = d.k_iters > LEAD;
    if (paced && lane == 0)
        atomic_store_explicit(&done[sgid], 0, memory_order_relaxed);
  #endif
  #if STAGE == 3
    #if PACE || KS > 1
      #error "STAGE 3 runs its own unpaced K loop"
    #endif
    // A streams through two register tiles: the load for step it + 1 is in flight while step it multiplies.
    if (active && in_m && K_ALIGNED && d.k_iters > 1) {
        auto r0 = op.template get_left_input_cooperative_tensor<IN_T, IN_T, float>();
        auto r1 = op.template get_left_input_cooperative_tensor<IN_T, IN_T, float>();
    #define AG_STEP(r, k0) { if (in_n) { auto sB = B_FULL(k0, KC); op.run(r, sB, acc); } \
                             else { ag_in_t vB = B_COLS(k0, (k0) + KC); op.run(r, vB, acc); } }
        const int n = d.k_iters;
        r0.load(A_FULL(0, KC));
        int it = 0;
        for (; it + 1 < n; it += 2) {
            r1.load(A_FULL((it + 1) * KC, KC));
            AG_STEP(r0, it * KC);
            if (it + 2 < n)
                r0.load(A_FULL((it + 2) * KC, KC));
            AG_STEP(r1, (it + 1) * KC);
        }
        if (it < n)
            AG_STEP(r0, it * KC);
    #undef AG_STEP
    } else
  #endif
    for (int it = 0; it < d.k_iters; ++it) {
  #if PACE == 1
        if (it >= LEAD && it % BIT == 0) {
            if (lane < NSG)
                while (atomic_load_explicit(&done[lane], memory_order_relaxed) <= it - LEAD) {}
            simdgroup_barrier(mem_flags::mem_none);
        }
  #elif PACE == 2
        if (it && it % BIT == 0)
            threadgroup_barrier(mem_flags::mem_none);
  #endif
        const int k0 = (it * KS + ks) * KC;
        if (active && inside && (K_ALIGNED ? k0 < gK : k0 + KC <= gK)) {
  #if STAGE == 1
            rA.load(A_FULL(k0, KC));
            auto sB = B_FULL(k0, KC);
            op.run(rA, sB, acc);
  #elif STAGE == 2
            rB.load(B_FULL(k0, KC));
            auto sA = A_FULL(k0, KC);
            op.run(sA, rB, acc);
  #else
            auto sA = A_FULL(k0, KC);
            auto sB = B_FULL(k0, KC);
            op.run(sA, sB, acc);
  #endif
        } else if (active && in_m && !in_n && k0 + KC <= gK) {
            // Only N runs past the edge: bounds-check B alone (staged from its bounds-checked view).
            ag_in_t vB = B_COLS(k0, k0 + KC);
  #if STAGE == 1
            rA.load(A_FULL(k0, KC));
            op.run(rA, vB, acc);
  #elif STAGE == 2
            rB.load(vB);
            auto sA = A_FULL(k0, KC);
            op.run(sA, rB, acc);
  #else
            auto sA = A_FULL(k0, KC);
            op.run(sA, vB, acc);
  #endif
        } else if (active && !in_m && in_n && k0 + KC <= gK) {
            ag_in_t vA = A_ROWS(k0, k0 + KC);
  #if STAGE == 2
            rB.load(B_FULL(k0, KC));
            op.run(vA, rB, acc);
  #elif STAGE == 1
            rA.load(vA);
            auto sB = B_FULL(k0, KC);
            op.run(rA, sB, acc);
  #else
            auto sB = B_FULL(k0, KC);
            op.run(vA, sB, acc);
  #endif
        } else if (active && k0 < gK) {
            // K tail or a block thinner than the simdgroup tile: one bounds-checked call.
            ag_in_t vA = A_ROWS(k0, min(k0 + KC, gK));
            ag_in_t vB = B_COLS(k0, min(k0 + KC, gK));
            op.run(vA, vB, acc);
        }
  #if PACE == 1
        if (paced && lane == 0)
            atomic_store_explicit(&done[sgid], it + 1, memory_order_relaxed);
  #endif
    }
#endif
#if KS > 1
    // Groups 1..KS-1 leave their partial sums in threadgroup memory (16 KB at a time); group 0 adds them.
    for (int w = 0; w < KS - 1; w += SLOTS) {
        const int slot = int(ks) - 1 - w;
        if (ks != 0 && slot >= 0 && slot < SLOTS)
            acc.store(TgExt(red + (slot * SM * SN + pos) * TM * TN, Ext(TN, TM), array<int32_t, 2>{1, TN}));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks == 0) {
            for (int s = 0; s < min(SLOTS, KS - 1 - w); ++s) {
                auto part = op.template get_destination_cooperative_tensor<decltype(A_FULL(0, AG_KW)), decltype(B_FULL(0, AG_KW)), float>();
                part.load(TgExt(red + (s * SM * SN + pos) * TM * TN, Ext(TN, TM), array<int32_t, 2>{1, TN}));
                #pragma unroll
                for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                    acc[i] += part[i];
            }
        }
        if (w + SLOTS < KS - 1)
            threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ks != 0)
        return;
#endif
#if STORE_TG
    // A single K step leaves the store to dominate: the tile goes out through threadgroup memory in whole rows.
    threadgroup OUT_T ctile[BM * BN];
#else
    if (!active)
        return;
#endif
    if (active) {
#if SPLITK
  #define AG_RES acc   // a copy of a cooperative tensor does not compile on macOS 26
#else
        auto res = op.template get_destination_cooperative_tensor<decltype(A_FULL(0, AG_KW)), decltype(B_FULL(0, AG_KW)), OUT_T>();
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
  #if AG_EPILOGUE
            if (!inside && !acc.is_valid_element(i))
                continue;
            auto idx = acc.get_multidimensional_index(i);
            res[i] = mb_epi<OUT_T, float, float>(acc[i], bias, bias_off + (m_off + int(idx[1])) * bstride.x + (n_off + int(idx[0])) * bstride.y,
                                                 beta, alpha);
  #else
            res[i] = static_cast<OUT_T>(acc[i]);
  #endif
        }
  #define AG_RES res
#endif
#if STORE_TG
        AG_RES.store(tensor<threadgroup OUT_T, Ext, tensor_inline>(ctile + (m_off - m_tile * BM) * BN + n_off - n_tile * BN, Ext(ON, OM),
                                                                   array<int32_t, 2>{1, BN}));
#else
        if (inside)
            AG_RES.store(tC.template slice<ON, OM>(n_off, m_off));
        else
            AG_RES.store(ag_out_t(C + n_off + int64_t(m_off) * ldc, Ext(gN - n_off, gM - m_off), array<int32_t, 2>{1, ldc}));
#endif
#undef AG_RES
    }
#if STORE_TG
    threadgroup_barrier(mem_flags::mem_threadgroup);
    struct alignas(16) Row8 { OUT_T v[8]; };
    const int m0 = m_tile * BM, n0 = n_tile * BN, rows = min(BM, gM - m0), cols = min(BN, gN - n0);
    for (int i = (int(sgid) * 32 + int(lane)) * 8; i < rows * BN; i += NSG * 32 * 8) {
        const int r = i / BN, c = i % BN;
        device OUT_T *dst = C + int64_t(m0 + r) * ldc + n0 + c;
        if (c + 8 <= cols)
            *(device Row8 *)dst = *(threadgroup Row8 *)(ctile + i);
        else
            for (int j = 0; j < cols - c; ++j)
                dst[j] = ctile[i + j];
    }
#endif
}

#if SPLITK
// C = epilogue(sum of the split planes). Grid (N, M, batch).
kernel void apple_gemm_reduce(
    device const float *P [[buffer(0)]],
    device OUT_T *C       [[buffer(1)]],
    constant AppleGemmDims& d [[buffer(2)]],
#if EPILOGUE
    device const OUT_T *bias [[buffer(3)]],
    constant int2&  bstride  [[buffer(4)]],
    constant float& beta     [[buffer(5)]],
    constant float& alpha    [[buffer(6)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(EPILOGUE ? 7 : 3)]],
#endif
    uint3 gid [[thread_position_in_grid]])
{
    const int n = int(gid.x), m = int(gid.y), z = int(gid.z);
    if (n >= d.N || m >= d.M) return;
    const size_t plane = size_t(d.M) * d.N;
    P += size_t(z) * d.splits * plane + size_t(m) * d.N + n;
    // Four running sums keep four loads in flight.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= d.splits; i += 4) {
        s0 += P[i * plane];
        s1 += P[(i + 1) * plane];
        s2 += P[(i + 2) * plane];
        s3 += P[(i + 3) * plane];
    }
    for (; i < d.splits; ++i)
        s0 += P[i * plane];
    const float s = (s0 + s1) + (s2 + s3);
    int64_t off = int64_t(m) * d.ldc + n;
#if BATCHED
    off += int64_t(z) * batch.z;
#endif
#if EPILOGUE
  #if BATCHED
    const int bias_off = z * batch.w;
  #else
    const int bias_off = 0;
  #endif
    C[off] = mb_epi<OUT_T, float, float>(s, bias, bias_off + m * bstride.x + n * bstride.y, beta, alpha);
#else
    C[off] = static_cast<OUT_T>(s);
#endif
}
#endif  // SPLITK
#endif  // MB_BUILD_APPLE_GEMM
