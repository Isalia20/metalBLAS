// mpp_pf.h - thin-N matmul2d with software-prefetched A chunks (packed, aligned shapes).
#ifdef MB_BUILD_MPP_PF
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T  __IN_T__
#define OUT_T __OUT_T__
#define BM    __BM__
#define BN    __BN__
#define NSG   __NSG__
#define KC    __KC__
#define PFD   __PFD__

// Touching A's next K-chunk (one word per 128-B line) overlaps its DRAM fetch with the
// current chunk's MMA; op.run alone leaves the thin-N stream ~15% under-overlapped.
kernel void mppf_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant int4& gP  [[buffer(3)]],   // packed (gM, gN, gK)
    uint3 tgid         [[threadgroup_position_in_grid]],
    uint  tid          [[thread_index_in_threadgroup]])
{
    int gM = gP.x, gN = gP.y, gK = gP.z;
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(gK, gM));
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(gN, gK));
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM));
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, KC, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    int m_off = int(tgid.y) * BM;
    int n_off = int(tgid.x) * BN;
    auto mA0 = tA.slice<KC, BM>(0, m_off);
    auto mB0 = tB.slice<BN, KC>(n_off, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

    const device uint *A32 = (const device uint *)A;   // caller guarantees 4-B alignment
    uint acc = 0;
    const int lpr = (KC * 2) / 128;                    // 128-B lines per chunk row
    for (int k0 = 0; k0 < gK; k0 += KC) {
        int kp = k0 + PFD * KC;
        if (kp < gK) {
            for (int s = int(tid); s < BM * lpr; s += NSG * 32) {
                int r = s / lpr, c = s % lpr;
                acc |= A32[((m_off + r) * gK + kp + c * 64) >> 1];
            }
        }
        auto mA = tA.slice<KC, BM>(k0, m_off);
        auto mB = tB.slice<BN, KC>(n_off, k0);
        op.run(mA, mB, cT);
    }
    auto mC = tC.slice<BN, BM>(n_off, m_off);
    auto cO = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), OUT_T>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
    cO.store(mC);
    // Opaque never-true guard (gM >= 1 always) so the prefetch loads can't be DCE'd.
    if (acc == 0x7F800001u && gM == -1) C[0] = (OUT_T)0;
}
#endif  // MB_BUILD_MPP_PF
