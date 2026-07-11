// Cooperative-tensor GEMM.
#ifdef MB_BUILD_MPP_TENSOR
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

// Store a tile with an optional fused epilogue and edge validation.
#if EPILOGUE
#define MB_STORE_TILE(VALIDATE) do {                                              \
    _Pragma("unroll")                                                            \
    for (uint16_t _e = 0; _e < cT_f32.get_capacity(); ++_e)                      \
        if (!(VALIDATE) || cT_f32.is_valid_element(_e)) {                         \
            auto _idx = cT_f32.get_multidimensional_index(_e);                    \
            int _r = m_off + (int)_idx[1], _c = n_off + (int)_idx[0];             \
            cT_out[_e] = mb_epi<OUT_T, float, float>(                             \
                cT_f32[_e], bias, MB_BBAT + _r * bstride.x + _c * bstride.y, beta, alpha); \
        }                                                                         \
} while (0)
#else
#define MB_STORE_TILE(VALIDATE) do {                                              \
    for (uint16_t _i = 0; _i < cT_f32.get_capacity(); ++_i)                      \
        if (!(VALIDATE) || cT_f32.is_valid_element(_i))                           \
            cT_out[_i] = (OUT_T)cT_f32[_i];                                        \
} while (0)
#endif
struct MBTensorDims { int M, N, K, lda, ldb, ldc; };

kernel void mpp_tensor_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant MBTensorDims& gP [[buffer(3)]],   // (M, N, K, lda, ldb, ldc)
#if EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride  [[buffer(5)]],
    constant float& beta     [[buffer(6)]],
    constant float& alpha    [[buffer(7)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(MB_BATCH_BUF)]],   // (sA, sB, sC, sBias) per-batch element strides
#endif
    uint3 tgid         [[threadgroup_position_in_grid]])
{
#if BATCHED
    A += (int64_t)tgid.z * (int64_t)batch.x;
    B += (int64_t)tgid.z * (int64_t)batch.y;
    C += (int64_t)tgid.z * (int64_t)batch.z;
  #if EPILOGUE
    int _bbat = (int)tgid.z * batch.w;
  #endif
#endif
    int gM = gP.M, gN = gP.N, gK = gP.K;
    auto eA = TRANS_A ? dextents<int32_t, 2>(gM, gK) : dextents<int32_t, 2>(gK, gM);
    auto eB = TRANS_B ? dextents<int32_t, 2>(gK, gN) : dextents<int32_t, 2>(gN, gK);
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, eA, array<int32_t, 2>{1, gP.lda});
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, eB, array<int32_t, 2>{1, gP.ldb});
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM), array<int32_t, 2>{1, gP.ldc});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_extent, TRANS_A, TRANS_B, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_off = tgy * BM;
    int n_off = tgx * BN;

#if STATIC_SLICE
    // Interior tiles use static extents; edge tiles use validated dynamic slices.
  #if !MN_ALIGNED
    bool inside = (m_off + BM <= gM) && (n_off + BN <= gN);
    if (!inside) {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(n_off, 0);
        auto mC = tC.slice(n_off, m_off);
        auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT_f32);
        auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
        MB_STORE_TILE(1);
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
    MB_STORE_TILE(0);
    cT_out.store(mC);
#else
    auto mA = TRANS_A ? tA.slice(m_off, 0) : tA.slice(0, m_off);
    auto mB = TRANS_B ? tB.slice(0, n_off) : tB.slice(n_off, 0);
    auto mC = tC.slice(n_off, m_off);
    auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT_f32);
    auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
  #if MN_ALIGNED
    MB_STORE_TILE(0);
  #else
    MB_STORE_TILE(1);
  #endif
    cT_out.store(mC);
#endif
}
#endif  // MB_BUILD_MPP_TENSOR
