#ifdef MB_BUILD_MPS_IDENTITY
#include <metal_stdlib>
using namespace metal;

struct MPSIdentityDims {
    uint m, n;
    ulong sr, sc, dr, dc;
};

// Axis-2/3 extents are one for the chunk encoder's matrix copies.
inline uint2 mb_identity_coord(uint x, uint y) {
    return __X_MAP__ == 0 ? uint2(x, y) : uint2(y, x);
}

kernel void mps_identity(
    device const __IN_T__ *src [[buffer(0)]],
    device __IN_T__ *dst [[buffer(1)]],
    constant MPSIdentityDims &p [[buffer(2)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 size [[threads_per_threadgroup]]) {
#if __TRANSPOSE_COPY__
    // ndArrayIdentityTranspose uses a padded float tile, including for
    // half and bfloat inputs, and visits NR x NR subtiles in this order.
    threadgroup float tile[__COPY_GX__ * (__COPY_GX__ + 1)];
    #pragma clang loop unroll(full)
    for (uint sy = 0; sy < __COPY_NR__; ++sy) {
        #pragma clang loop unroll(full)
        for (uint sx = 0; sx < __COPY_NR__; ++sx) {
            uint2 base = (group * __COPY_NR__ + uint2(sx, sy)) * size.x;
            for (uint j = tid.y; j < size.x; j += size.y) {
                uint2 c = mb_identity_coord(base.x + tid.x, base.y + j);
                float v = 0.0f;
                if (c.x < p.n && c.y < p.m)
                    v = float(src[c.y * p.sr + c.x * p.sc]);
                tile[j * (size.x + 1) + tid.x] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = tid.y; j < size.x; j += size.y) {
                uint2 c = mb_identity_coord(base.x + j, base.y + tid.x);
                if (c.x < p.n && c.y < p.m)
                    dst[c.y * p.dr + c.x * p.dc] = __IN_T__(tile[tid.x * (size.x + 1) + j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
#else
    uint2 c = mb_identity_coord(group.x * size.x + tid.x, group.y * size.y + tid.y);
    if (c.x < p.n && c.y < p.m)
        dst[c.y * p.dr + c.x * p.dc] = src[c.y * p.sr + c.x * p.sc];
#endif
}
#endif
