

#ifdef MB_BUILD_CONV2D
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
#define KH      __KH__
#define KW      __KW__
#define SX      __SX__
#define SY      __SY__
#define DX      __DX__
#define DY      __DY__
#define RELAXED __RELAXED__

#ifndef BIAS
#define BIAS 0
#endif
#ifndef D_NCHW
#define D_NCHW 0
#endif
#ifndef GROUPED
#define GROUPED 0
#endif
#ifndef CONV3D
#define CONV3D 0
#endif

#ifndef KD
#define KD 1
#endif
#ifndef SZ
#define SZ 1
#endif
#ifndef DZ
#define DZ 1
#endif

struct MBConvDims {
    int C, H, W, O;
    int HO, WO, NB;
    int PADX, PADY;
    int CG, OG, OGT;
    int D, DO, PADZ;
};

kernel void conv2d_mpp(
    device IN_T   *act  [[buffer(0)]],
    device IN_T   *wts  [[buffer(1)]],
    device OUT_T  *dst  [[buffer(2)]],
    constant MBConvDims& gP [[buffer(3)]],
#if BIAS
    device const OUT_T *bias [[buffer(4)]],
#endif
    uint3 tgid          [[threadgroup_position_in_grid]])
{
    int h_tiles = (gP.HO + BH - 1) / BH;
#if GROUPED
    int g     = int(tgid.x) / gP.OGT;
    int o_off = (int(tgid.x) % gP.OGT) * BO + g * gP.OG;
    int o_end = g * gP.OG + gP.OG;
    int c0    = g * gP.CG;
#else
    int o_off = int(tgid.x) * BO;
    int c0    = 0;
#endif
    int wo_off = int(tgid.y) * BW;
    int zi     = int(tgid.z);
    int ho_off = (zi % h_tiles) * BH;
    zi /= h_tiles;
#if CONV3D
    int dd = zi % gP.DO;
    int nb = zi / gP.DO;
#else
    int nb = zi;
#endif

    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tWt(
        wts, dextents<int32_t, 4>(gP.O, gP.CG, KW, KH * KD));

    constexpr auto desc = convolution2d_descriptor(

        int4(BO, BW, BH, 1), int4(__SRCC__, __SRCW__, __SRCH__, 1), int2(KW, KH),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(SX, SY), int2(DX, DY), 1, RELAXED,
        convolution2d_descriptor::mode::multiply_accumulate);
    convolution2d<desc, execution_simdgroups<NSG>> op;
    op.set_offsets(int2(wo_off * SX - gP.PADX + (KW / 2) * DX,
                        ho_off * SY - gP.PADY + (KH / 2) * DY));

    auto mW0 = tWt.slice(o_off, 0, 0, 0);

    int64_t plane = (int64_t)gP.H * gP.W * gP.C;
    device IN_T *actn = act + (int64_t)nb * gP.D * plane;
#if GROUPED

    #define MB_ACT_VIEW(P) tensor<device IN_T, dextents<int32_t, 4>, tensor_inline>( \
        (P), dextents<int32_t, 4>(c0 + gP.CG, gP.W, gP.H, 1),                        \
        array<int32_t, 4>{1, gP.C, gP.C * gP.W, gP.C * gP.W * gP.H})
#else
    #define MB_ACT_VIEW(P) tensor<device IN_T, dextents<int32_t, 4>, tensor_inline>( \
        (P), dextents<int32_t, 4>(gP.C, gP.W, gP.H, 1))
#endif
    auto tA0 = MB_ACT_VIEW(actn);
    auto mA0 = tA0.slice(c0, 0, 0, 0);

    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mW0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

#if CONV3D
    for (int kd = 0; kd < KD; ++kd) {
        int di = dd * SZ - gP.PADZ + kd * DZ;
        if (di < 0 || di >= gP.D) continue;
        auto tA = MB_ACT_VIEW(actn + di * plane);
        auto mA = tA.slice(c0, 0, 0, 0);
        auto mW = tWt.slice(o_off, 0, 0, kd * KH);
        op.run(mA, mW, cT);
    }
#else
    op.run(mA0, mW0, cT);
#endif

#if GROUPED || D_NCHW

  #if !GROUPED
    int o_end = gP.O;
  #endif
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        auto idx = cT.get_multidimensional_index(i);
        int o = o_off + (int)idx[0];
        int x = wo_off + (int)idx[1];
        int y = ho_off + (int)idx[2];
        if (o >= o_end || x >= gP.WO || y >= gP.HO) continue;
        float v = cT[i];
  #if BIAS
        v += (float)bias[o];
  #endif
  #if CONV3D
    #if D_NCHW
        int64_t di = ((((int64_t)nb * gP.O + o) * gP.DO + dd) * gP.HO + y) * gP.WO + x;
    #else
        int64_t di = ((((int64_t)nb * gP.DO + dd) * gP.HO + y) * gP.WO + x) * gP.O + o;
    #endif
  #else
    #if D_NCHW
        int64_t di = (((int64_t)nb * gP.O + o) * gP.HO + y) * gP.WO + x;
    #else
        int64_t di = (((int64_t)nb * gP.HO + y) * gP.WO + x) * gP.O + o;
    #endif
  #endif
        dst[di] = (OUT_T)v;
    }
#else

  #if CONV3D
    device OUT_T *dstn = dst + ((int64_t)nb * gP.DO + dd) * gP.HO * gP.WO * gP.O;
    tensor<device OUT_T, dextents<int32_t, 4>, tensor_inline> tD(
        dstn, dextents<int32_t, 4>(gP.O, gP.WO, gP.HO, 1));
    auto mD = tD.slice(o_off, wo_off, ho_off, 0);
  #else
    tensor<device OUT_T, dextents<int32_t, 4>, tensor_inline> tD(
        dst, dextents<int32_t, 4>(gP.O, gP.WO, gP.HO, gP.NB));
    auto mD = tD.slice(o_off, wo_off, ho_off, nb);
  #endif
    auto cO = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mW0), OUT_T>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        float v = cT[i];
  #if BIAS
        v += (float)bias[o_off + (int)cT.get_multidimensional_index(i)[0]];
  #endif
        cO[i] = (OUT_T)v;
    }
    cO.store(mD);
#endif
}
#endif
