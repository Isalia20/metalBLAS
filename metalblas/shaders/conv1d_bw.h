

#ifdef MB_BUILD_CONV1D_BW
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T    __IN_T__
#define OUT_T   __OUT_T__
#define BO      __BO__
#define BW      __BW__
#define BH      __BH__
#define NSG     __NSG__
#define K       __K__
#define S       __S__
#define DIL     __DIL__
#define RELAXED __RELAXED__

#ifndef BIAS
#define BIAS 0
#endif

struct MBConv1dBwDims { int C, L, NB, O, LO, PAD; };

kernel void conv1d_bw(
    device IN_T   *act  [[buffer(0)]],
    device IN_T   *wts  [[buffer(1)]],
    device OUT_T  *dst  [[buffer(2)]],
    constant MBConv1dBwDims& gP [[buffer(3)]],
#if BIAS
    device const OUT_T *bias [[buffer(4)]],
#endif
    uint3 tgid          [[threadgroup_position_in_grid]])
{
    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tA(
        act, dextents<int32_t, 4>(gP.C, gP.L, gP.NB, 1),
        array<int32_t, 4>{1, gP.C, gP.L * gP.C, gP.NB * gP.L * gP.C});
    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tW(
        wts, dextents<int32_t, 4>(gP.O, gP.C, K, 1));

    constexpr auto desc = convolution2d_descriptor(
        int4(BO, BW, BH, 1), int4(-1, 1 << 20, 16384, 1), int2(K, 1),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(S, 1), int2(DIL, 1), 1, RELAXED,
        convolution2d_descriptor::mode::multiply_accumulate);
    convolution2d<desc, execution_simdgroups<NSG>> op;

    int lo_tiles = (gP.LO + BW - 1) / BW;
    int o_off  = int(tgid.x) * BO;
    int lo_off = (int(tgid.z) % lo_tiles) * BW;
    int n_off  = (int(tgid.z) / lo_tiles) * BH;
    op.set_offsets(int2(lo_off * S - gP.PAD + (K / 2) * DIL, n_off));

    auto mA = tA.slice(0, 0, 0, 0);
    auto mW = tW.slice(o_off, 0, 0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;
    op.run(mA, mW, cT);

    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        auto idx = cT.get_multidimensional_index(i);
        int o  = o_off + (int)idx[0];
        int lo = lo_off + (int)idx[1];
        int n  = n_off + (int)idx[2];
        if (o >= gP.O || lo >= gP.LO || n >= gP.NB) continue;
        float v = cT[i];
#if BIAS
        v += (float)bias[o];
#endif
        dst[((int64_t)n * gP.O + o) * gP.LO + lo] = (OUT_T)v;
    }
}
#endif
