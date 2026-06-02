// cgemv_nt.h - complex GEMV  y = A @ x  (A is M x K row-major complex).
// Each simdgroup owns one row; lanes stride K (coalesced C2 row loads) then simd_sum
// the real and imag parts
#ifdef MB_BUILD_CGEMV_NT
#include <metal_stdlib>
using namespace metal;

#define C2     __C2__       // interleaved complex element: float2 / half2
#define ACC2   __ACC2__     // accumulator (float2)
#define R      __R__        // output real-component scalar
#define NWARPS __NWARPS__   // rows (simdgroups) per threadgroup

kernel void cgemv_nt(
    device const C2 *A  [[buffer(0)]],
    device const C2 *x  [[buffer(1)]],
    device       C2 *y  [[buffer(2)]],
    constant int4&  gP  [[buffer(3)]],   // packed (gM, gK, gLda, gXs)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    int gM = gP.x, gK = gP.y, gLda = gP.z, gXs = gP.w;
    int row = int(tgid.x) * NWARPS + int(sgid);
    if (row >= gM) return;

    const device C2 *Arow = A + (size_t)row * gLda;
    ACC2 acc = ACC2(0);
    for (int k = int(lane); k < gK; k += 32) {
        C2 a  = Arow[k];
        C2 xk = x[k * gXs];
        acc.x += (float)a.x * (float)xk.x - (float)a.y * (float)xk.y;
        acc.y += (float)a.x * (float)xk.y + (float)a.y * (float)xk.x;
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    if (lane == 0) y[row] = C2((R)acc.x, (R)acc.y);
}
#endif  // MB_BUILD_CGEMV_NT
