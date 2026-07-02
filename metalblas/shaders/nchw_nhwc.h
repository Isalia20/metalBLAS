

#ifdef MB_BUILD_NCHW_NHWC
#include <metal_stdlib>
using namespace metal;

#define ELT_T __ELT_T__
#define TC    __TC__
#define TX    __TX__
#define NTH   __NTH__
#define VECR  __VECR__
#define VECW  __VECW__

kernel void nchw_to_nhwc(
    device const ELT_T *src [[buffer(0)]],
    device ELT_T       *dst [[buffer(1)]],
    constant int2&      dims [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup ELT_T tile[TC][TX + 1];
    int C = dims.x, X = dims.y;
    int c0 = int(tgid.y) * TC;
    int x0 = int(tgid.x) * TX;
    device const ELT_T *s = src + (int64_t)tgid.z * C * X;
    device ELT_T       *d = dst + (int64_t)tgid.z * C * X;

#if VECR

    for (int i = tid; i < TC * (TX / 2); i += NTH) {
        int x = (i % (TX / 2)) * 2, c = i / (TX / 2);
        int gc = c0 + c, gx = x0 + x;
        if (gc >= C || gx >= X) continue;
        int64_t off = (int64_t)gc * X + gx;
        if (gx + 1 < X) {
            vec<ELT_T, 2> v = *(device const vec<ELT_T, 2> *)(s + off);
            tile[c][x] = v.x;
            tile[c][x + 1] = v.y;
        } else {
            tile[c][x] = s[off];
        }
    }
#else
    for (int i = tid; i < TC * TX; i += NTH) {
        int x = i % TX, c = i / TX;
        int gc = c0 + c, gx = x0 + x;
        if (gc < C && gx < X)
            tile[c][x] = s[(int64_t)gc * X + gx];
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if VECW

    for (int i = tid; i < (TC / 2) * TX; i += NTH) {
        int c = (i % (TC / 2)) * 2, x = i / (TC / 2);
        int gc = c0 + c, gx = x0 + x;
        if (gx >= X || gc >= C) continue;
        int64_t off = (int64_t)gx * C + gc;
        if (gc + 1 < C) {
            *(device vec<ELT_T, 2> *)(d + off) =
                vec<ELT_T, 2>(tile[c][x], tile[c + 1][x]);
        } else {
            d[off] = tile[c][x];
        }
    }
#else
    for (int i = tid; i < TC * TX; i += NTH) {
        int c = i % TC, x = i / TC;
        int gc = c0 + c, gx = x0 + x;
        if (gc < C && gx < X)
            d[(int64_t)gx * C + gc] = tile[c][x];
    }
#endif
}
#endif
