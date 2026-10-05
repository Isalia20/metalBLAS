// reduce_multiple_passes_axes_add AIR, specialized to the addbmm batch axis.
#ifdef MB_BUILD_MPS_REDUCTION
#include <metal_stdlib>
using namespace metal;

#include "mps_epilogue.h"

#define IN_T __IN_T__
#define OUT_T __OUT_T__
#define GX __GX__
#define GY __GY__
#define GZ __GZ__

struct MBBatchReductionDims {
    uint N, M, batch, batch_tile, ldc, unused;
};

kernel void mps_batch_reduce(
    const device IN_T* src [[buffer(0)]], device OUT_T* dst [[buffer(1)]],
    constant MBBatchReductionDims& p [[buffer(2)]],
#if MB_MPS_EPILOGUE
    const device OUT_T* bias [[buffer(3)]], constant ulong* bstride [[buffer(4)]],
#endif
    uint3 group [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]]) {
    uint base_n = group.x * (4 * GX);
    bool full_n = min(p.N - base_n, uint(4 * GX)) == 4 * GX;
    uint first_n = base_n + (full_n ? t.x * 4 : t.x);
    uint step_n = full_n ? 1 : GX;
    uint first_b = group.y * p.batch_tile + t.y;
    uint end_b = min(first_b + p.batch_tile, p.batch);
    float4 acc = 0;
    // AIR decomposes gid.z through srcBounds, masking a leftover quotient.
    // For a single nonreduced row axis this is exactly gid.z < M, even when
    // the physical z dimension exceeds threadsPerAxis[1].
    if (gid.z < p.M) {
        for (uint b = first_b; b < end_b; b += GY) {
            const device IN_T* row = src + ulong(b) * p.M * p.N + ulong(gid.z) * p.N;
            for (ushort c = 0; c < 4; ++c) {
                uint n = first_n + c * step_n;
                if (n < p.N) acc[c] += float(row[n]);
            }
        }
    }
    // AIR stores float4s in [z][x][y] order and halves the y extent at
    // each barrier. This order also applies to float intermediate passes.
    threadgroup float4 partial[GX * GY * GZ];
    uint slot = (t.z * GX + t.x) * GY + t.y;
    partial[slot] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort s = GY / 2; s; s /= 2) {
        if (t.y < s) partial[slot] += partial[slot + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t.y == 0 && gid.z < p.M) {
        float4 sum = partial[slot];
        device OUT_T* row = dst + ulong(group.y) * p.M * p.ldc + ulong(gid.z) * p.ldc;
        for (ushort c = 0; c < 4; ++c) {
            uint n = first_n + c * step_n;
            if (n < p.N) {
#if MB_MPS_EPILOGUE
                row[n] = mb_mps_epilogue(sum[c], bias, ulong(gid.z) * bstride[0] + ulong(n) * bstride[1]);
#else
                row[n] = OUT_T(sum[c]);
#endif
            }
        }
    }
}
#endif
