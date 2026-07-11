// Thin-N GEMM with one register-staged matmul2d operation per simdgroup.
#ifdef MB_BUILD_MPP_SGPIPE
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T  __IN_T__
#define OUT_T __OUT_T__
#define SGM   __SGM__     // per-SG tile rows (over M)
#define SGN   __SGN__     // per-SG tile cols (over N)
#define KC    __KC__      // K-chunk per run (input coop tensors need static K)
#define NSGX  __NSGX__    // SG grid in TG: columns (over N)
#define NSGY  __NSGY__    // SG grid in TG: rows (over M)
#define GK    __GK__      // 0: K from dims buffer; >0: baked, k-loop unrolls
#define GM    __GM__      // 0: M from dims buffer; >0: baked
#define GN    __GN__      // 0: N from dims buffer; >0: baked (folds B/C stride math)

// The caller supplies packed shapes divisible by the tile and 2*KC.
kernel void sgpipe_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant int4& gP  [[buffer(3)]],   // packed (gM, gN, gK)
    uint3 tgid         [[threadgroup_position_in_grid]],
    uint  sgid         [[simdgroup_index_in_threadgroup]])
{
    const int gM = GM > 0 ? GM : gP.x;
    const int gN = GN > 0 ? GN : gP.y;
    const int gK = GK > 0 ? GK : gP.z;
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(gN, gK));
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM));

    int m_off = (int(tgid.y) * NSGY + int(sgid) / NSGX) * SGM;
    int n_off = (int(tgid.x) * NSGX + int(sgid) % NSGX) * SGN;

    constexpr auto mdesc = matmul2d_descriptor(
        SGM, SGN, KC, false, false, true, matmul2d_descriptor::mode::multiply);
    constexpr auto adesc = matmul2d_descriptor(
        SGM, SGN, KC, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<mdesc, execution_simdgroup> opm;   // chunk 0: no zero-init / acc read
    matmul2d<adesc, execution_simdgroup> op;

    device IN_T *Abase = A + m_off * gK;
    const auto chunk = [&](int k0) {            // slices are tensor_offset; load() wants
        return tensor<device IN_T, dextents<int32_t, 2>, tensor_inline>(   // tensor_inline
            Abase + k0, dextents<int32_t, 2>(KC, SGM), array<int32_t, 2>{1, gK});
    };
    auto cA0 = op.get_left_input_cooperative_tensor<IN_T, IN_T, float>();
    auto mB0 = tB.slice<SGN, KC>(n_off, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(cA0), decltype(mB0), float>();

    auto t0 = chunk(0);
    cA0.load(t0);
    opm.run(cA0, mB0, cT);
#if GK > 0
    #pragma unroll
#endif
    for (int k0 = KC; k0 < gK; k0 += KC) {
        auto tk = chunk(k0);
        cA0.load(tk);
        auto mB = tB.slice<SGN, KC>(n_off, k0);
        op.run(cA0, mB, cT);
    }

    auto cO = op.get_destination_cooperative_tensor<decltype(cA0), decltype(mB0), OUT_T>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
    auto mC = tC.slice<SGN, SGM>(n_off, m_off);
    cO.store(mC);
}
#endif  // MB_BUILD_MPP_SGPIPE
