

#ifdef MB_BUILD_DWCONV
#include <metal_stdlib>
using namespace metal;

#define ELT_T __ELT_T__
#define KH    __KH__
#define KW    __KW__
#define SX    __SX__
#define SY    __SY__
#define DX    __DX__
#define DY    __DY__
#define OPT   __OPT__

#ifndef BIAS
#define BIAS 0
#endif
#ifndef NHWC
#define NHWC 0
#endif

struct MBDwDims {
    int C, H, W, O;
    int HO, WO, NB;
    int PADX, PADY, MULT;
};

kernel void dw_conv(
    device const ELT_T *act [[buffer(0)]],
    device const ELT_T *wts [[buffer(1)]],
    device ELT_T       *dst [[buffer(2)]],
    constant MBDwDims&  gP  [[buffer(3)]],
#if BIAS
    device const ELT_T *bias [[buffer(4)]],
#endif
    uint3 gid [[thread_position_in_grid]])
{
    float acc[OPT];
#if BIAS
    #define MB_ACC_INIT(B) { for (int j = 0; j < OPT; ++j) acc[j] = (B); }
#else
    #define MB_ACC_INIT(B) { for (int j = 0; j < OPT; ++j) acc[j] = 0.0f; }
#endif

#if NHWC

    int o  = int(gid.x);
    int x  = int(gid.y);
    int yt = int(gid.z) % ((gP.HO + OPT - 1) / OPT);
    int nb = int(gid.z) / ((gP.HO + OPT - 1) / OPT);
    int y0 = yt * OPT;
    if (o >= gP.O || x >= gP.WO || nb >= gP.NB) return;
    int c = o / gP.MULT;
#if BIAS
    float bv = (float)bias[o];
    MB_ACC_INIT(bv)
#else
    MB_ACC_INIT(0)
#endif
    device const ELT_T *a = act + (int64_t)nb * gP.H * gP.W * gP.C;
    int ix0 = x * SX - gP.PADX;
#if SY == 1

    #pragma unroll
    for (int kx = 0; kx < KW; ++kx) {
        int ix = ix0 + kx * DX;
        if (ix < 0 || ix >= gP.W) continue;
        float v[OPT + (KH - 1) * DY];
        #pragma unroll
        for (int t = 0; t < OPT + (KH - 1) * DY; ++t) {
            int iy = y0 - gP.PADY + t;
            v[t] = (iy < 0 || iy >= gP.H) ? 0.0f
                 : (float)a[((int64_t)iy * gP.W + ix) * gP.C + c];
        }
        #pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            float wv = (float)wts[(ky * KW + kx) * gP.O + o];
            #pragma unroll
            for (int j = 0; j < OPT; ++j)
                acc[j] += v[j + ky * DY] * wv;
        }
    }
#else
    #pragma unroll
    for (int kx = 0; kx < KW; ++kx) {
        int ix = ix0 + kx * DX;
        if (ix < 0 || ix >= gP.W) continue;
        #pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            float wv = (float)wts[(ky * KW + kx) * gP.O + o];
            #pragma unroll
            for (int j = 0; j < OPT; ++j) {
                int iy = (y0 + j) * SY - gP.PADY + ky * DY;
                if (iy < 0 || iy >= gP.H) continue;
                acc[j] += (float)a[((int64_t)iy * gP.W + ix) * gP.C + c] * wv;
            }
        }
    }
#endif
    #pragma unroll
    for (int j = 0; j < OPT; ++j) {
        int y = y0 + j;
        if (y < gP.HO)
            dst[(((int64_t)nb * gP.HO + y) * gP.WO + x) * gP.O + o] = (ELT_T)acc[j];
    }
#else

    int xt = int(gid.x);
    int y  = int(gid.y);
    int o  = int(gid.z) % gP.O;
    int nb = int(gid.z) / gP.O;
    int x0 = xt * OPT;
    if (y >= gP.HO || x0 >= gP.WO || nb >= gP.NB) return;
    int c = o / gP.MULT;
#if BIAS
    float bv = (float)bias[o];
    MB_ACC_INIT(bv)
#else
    MB_ACC_INIT(0)
#endif
    device const ELT_T *a = act + ((int64_t)nb * gP.C + c) * gP.H * gP.W;
    device const ELT_T *w = wts + o * KH * KW;
    #pragma unroll
    for (int ky = 0; ky < KH; ++ky) {
        int iy = y * SY - gP.PADY + ky * DY;
        if (iy < 0 || iy >= gP.H) continue;
        device const ELT_T *row = a + (int64_t)iy * gP.W;
#if SX == 1

        float v[OPT + (KW - 1) * DX];
        #pragma unroll
        for (int t = 0; t < OPT + (KW - 1) * DX; ++t) {
            int ix = x0 - gP.PADX + t;
            v[t] = (ix < 0 || ix >= gP.W) ? 0.0f : (float)row[ix];
        }
        #pragma unroll
        for (int kx = 0; kx < KW; ++kx) {
            float wv = (float)w[ky * KW + kx];
            #pragma unroll
            for (int j = 0; j < OPT; ++j)
                acc[j] += v[j + kx * DX] * wv;
        }
#else
        #pragma unroll
        for (int kx = 0; kx < KW; ++kx) {
            float wv = (float)w[ky * KW + kx];
            #pragma unroll
            for (int j = 0; j < OPT; ++j) {
                int ix = (x0 + j) * SX - gP.PADX + kx * DX;
                if (ix < 0 || ix >= gP.W) continue;
                acc[j] += (float)row[ix] * wv;
            }
        }
#endif
    }
    device ELT_T *drow = dst + (((int64_t)nb * gP.O + o) * gP.HO + y) * gP.WO;
    #pragma unroll
    for (int j = 0; j < OPT; ++j)
        if (x0 + j < gP.WO) drow[x0 + j] = (ELT_T)acc[j];
#endif
}
#endif
