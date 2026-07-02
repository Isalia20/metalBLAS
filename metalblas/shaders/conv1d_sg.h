

#ifdef MB_BUILD_CONV1D_SG
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T    __IN_T__
#define OUT_T   __OUT_T__
#define BM      __BM__
#define BN      __BN__
#define NSG     __NSG__
#define K       __K__
#define DIL     __DIL__
#define RELAXED __RELAXED__

#ifndef BIAS
#define BIAS 0
#endif

struct MBConv1dSgDims { int C, L, O, LO, PAD; };

kernel void conv1d_sg(
    device IN_T   *x    [[buffer(0)]],
    device IN_T   *wts  [[buffer(1)]],
    device OUT_T  *dst  [[buffer(2)]],
    constant MBConv1dSgDims& gP [[buffer(3)]],
#if BIAS
    device const OUT_T *bias [[buffer(4)]],
#endif
    uint3 tgid          [[threadgroup_position_in_grid]])
{
    device IN_T  *xn = x + (int64_t)tgid.z * gP.C * gP.L;
    device OUT_T *dn = dst + (int64_t)tgid.z * gP.O * gP.LO;
    int m_off  = int(tgid.y) * BM;
    int n_base = max(int(tgid.x) * BN, gP.PAD);
    if (n_base >= gP.LO) return;

    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(
        xn, dextents<int32_t, 2>(gP.L, gP.C), array<int32_t, 2>{1, gP.L});
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(
        dn, dextents<int32_t, 2>(gP.LO, gP.O), array<int32_t, 2>{1, gP.LO});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_extent, false, false, RELAXED,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA0(
        wts, dextents<int32_t, 2>(gP.C, gP.O), array<int32_t, 2>{1, gP.C});
    auto mA0 = tA0.slice(0, m_off);
    auto mB0 = tB.slice(n_base, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;

    for (int kl = 0; kl < K; ++kl) {
        tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(
            wts + (int64_t)kl * gP.O * gP.C,
            dextents<int32_t, 2>(gP.C, gP.O), array<int32_t, 2>{1, gP.C});
        auto mA = tA.slice(0, m_off);

        auto mB = tB.slice(n_base + kl * DIL - gP.PAD, 0);
        op.run(mA, mB, cT);
    }
    auto mC = tC.slice(n_base, m_off);
    auto cO = op.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), OUT_T>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        float v = cT[i];
#if BIAS
        v += (float)bias[m_off + (int)cT.get_multidimensional_index(i)[1]];
#endif
        cO[i] = (OUT_T)v;
    }
    cO.store(mC);
}

kernel void conv1d_sgfix(
    device IN_T   *x    [[buffer(0)]],
    device IN_T   *wts  [[buffer(1)]],
    device OUT_T  *dst  [[buffer(2)]],
    constant MBConv1dSgDims& gP [[buffer(3)]],
#if BIAS
    device const OUT_T *bias [[buffer(4)]],
#endif
    uint3 gid           [[thread_position_in_grid]])
{
    int lo = int(gid.x);
    int o  = int(gid.y);
    int n  = int(gid.z);
    if (lo >= gP.PAD || lo >= gP.LO || o >= gP.O) return;
    device IN_T *xn = x + (int64_t)n * gP.C * gP.L;
    float acc = 0.0f;
#if BIAS
    acc = (float)bias[o];
#endif
    for (int kl = 0; kl < K; ++kl) {
        int xi = lo + kl * DIL - gP.PAD;
        if (xi < 0 || xi >= gP.L) continue;
        device IN_T *wp = wts + ((int64_t)kl * gP.O + o) * gP.C;
        for (int c = 0; c < gP.C; ++c)
            acc += (float)wp[c] * (float)xn[(int64_t)c * gP.L + xi];
    }
    dst[((int64_t)n * gP.O + o) * gP.LO + lo] = (OUT_T)acc;
}
#endif
