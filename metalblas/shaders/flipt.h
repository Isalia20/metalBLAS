// flipt.h - thin-N GEMM as thin-M: C^T = B^T A^T tiles, transposed store via TG memory.
#ifdef MB_BUILD_FLIPT
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T  __IN_T__
#define OUT_T __OUT_T__
#define BM    __BM__      // tile over N (the thin dim of C)
#define BN    __BN__      // tile over M (the long dim of C)
#define NSG   __NSG__
#define KC    __KC__      // 0: one op.run over all K; >0: KC-chunked accumulate
#define PFD   __PFD__     // KC>0 only: prefetch distance in chunks (0 = off)
#define LDT   (BN + 2)    // pad dodges TG bank conflicts on the transposed read-out

// Streams A better in the flipped orientation (A becomes the transposed RHS).
// Packed shapes with N % BM == 0 and M % BN == 0 only; caller guarantees.
kernel void flipt_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant int4& gP  [[buffer(3)]],   // packed (gM, gN, gK)
    uint3 tgid         [[threadgroup_position_in_grid]],
    uint  tid          [[thread_index_in_threadgroup]])
{
    int gM = gP.x, gN = gP.y, gK = gP.z;
    int M2 = gN, N2 = gM;               // flipped problem: D (M2 x N2) = C^T
    threadgroup OUT_T tg[BM * LDT];
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(B, dextents<int32_t, 2>(M2, gK), array<int32_t, 2>{1, gN});
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(A, dextents<int32_t, 2>(gK, N2), array<int32_t, 2>{1, gK});
    int m_off = int(tgid.y) * BM;       // over M2 = N
    int n_off = int(tgid.x) * BN;       // over N2 = M

#if KC > 0
    // Chunked accumulate: touching A's next K-chunk (one word per 128-B line)
    // overlaps its DRAM fetch with the current chunk's MMA (same trick as mpp_pf).
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, KC, true, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA0 = tA.slice<BM, KC>(m_off, 0);
    auto mB0 = tB.slice<KC, BN>(0, n_off);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;
    const device uint *S32 = (const device uint *)A;   // caller guarantees 4-B alignment
    uint acc = 0;
    const int lpr = (KC * 2) / 128;                    // 128-B lines per chunk row
    for (int k0 = 0; k0 < gK; k0 += KC) {
#if PFD > 0
        int kp = k0 + PFD * KC;
        if (kp < gK) {
            for (int s = int(tid); s < BN * lpr; s += NSG * 32) {
                int r = s / lpr, c = s % lpr;
                acc |= S32[((n_off + r) * gK + kp + c * 64) >> 1];
            }
        }
#endif
        auto mA = tA.slice<BM, KC>(m_off, k0);
        auto mB = tB.slice<KC, BN>(k0, n_off);
        op.run(mA, mB, cT);
    }
#else
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_extent, true, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA0 = tA.slice(m_off, 0);
    auto mB0 = tB.slice(0, n_off);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
    op.run(mA0, mB0, cT);
#endif

    // D-tile -> TG memory (element (n2,m2) at m2*LDT + n2), then C rows written coalesced.
    auto tT = tensor<threadgroup OUT_T, dextents<int32_t, 2>, tensor_inline>(
        tg, dextents<int32_t, 2>(BN, BM), array<int32_t, 2>{1, LDT});
    auto cO = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), OUT_T>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
    cO.store(tT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < BM * BN; e += NSG * 32) {
        int r = e / BM, c = e % BM;     // C[n_off + r, m_off + c]
        C[(n_off + r) * gN + (m_off + c)] = tg[c * LDT + r];
    }
#if KC > 0 && PFD > 0
    // Opaque never-true guard (gM >= 1 always) so the prefetch loads can't be DCE'd.
    if (acc == 0x7F800001u && gM == -1) C[0] = (OUT_T)0;
#endif
}
#endif  // MB_BUILD_FLIPT
