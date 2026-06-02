// m5_tensor.h - matmul2d tensor-view GEMM - the primary backend.
#ifdef MB_BUILD_M5_TENSOR
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
    constant int4&  gP [[buffer(3)]],   // packed (gM, gN, gK); lda/ldb/ldc unused (packed storage)
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    int gM = gP.x, gN = gP.y, gK = gP.z;
    // Tensor views from raw pointers; extent order is (cols, rows) for row-major.
    // Transposed A keeps extents (gK, gM) but flags transposed in the descriptor.
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(gK, gM));
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(gN, gK));
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_extent, TRANS_A, TRANS_B, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    // Swizzle threadgroup ids for L2 reuse.
    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_off = tgy * BM;
    int n_off = tgx * BN;

#if STATIC_SLICE
    // Non-transposed fast path: static-extent slices mark each interior tile exactly
    // BM x BN and in-bounds, dropping dynamic-slice edge predication (still fp32 accum).
  #if !MN_ALIGNED
    // Partial edge tiles (M%BM or N%BN != 0) fall back to a dynamic slice with
    // the per-element validity mask; interior tiles take the static path below.
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
    // Transposed operands keep the dynamic-slice path; it is off the hot path
    // (auto-dispatch routes transposed inputs to m5_gemm).
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
#endif  // MB_BUILD_M5_TENSOR
