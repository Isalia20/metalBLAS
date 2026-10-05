// Ordinary NDArrayVectorMatrixMultiply schedules recovered from MPSNDArray AIR.
#ifdef MB_BUILD_MPS_VECTOR
#include <metal_stdlib>
using namespace metal;

#include "mps_epilogue.h"

#if MB_MPS_EPILOGUE
#define MB_VECTOR_EPILOGUE_ARGS , bias, bstride, group.z
#else
#define MB_VECTOR_EPILOGUE_ARGS
#endif

#define IN_T __IN_T__
#define LOAD4_T __LOAD4_T__
#define R __R__
#define GX __GX__
#define GY __GY__
#define FLAT __FLAT__
#define TRANS_B __TRANS_B__
#define TRANS_OUT __TRANS_OUT__
#define ROWS __ROWS__
#define UNROLL_K __UNROLL_K__
#define GROUPS __GROUPS__
#define FULL_GROUPS __FULL_GROUPS__
#define PARTIAL __PARTIAL__
#define K_MOD4 __K_MOD4__

// X is logically W x K, V is K x R. Transposing the result handles M < N.
struct MBVectorDims {
    uint W, K, ldx, ldv, ldc, groups, full_groups, unused;
    ulong bsx, bsv, bsc;
};

static inline float vector_x(const device IN_T* x, uint w, uint k, constant MBVectorDims& p) {
    return float(x[FLAT ? k * p.ldx + w : w * p.ldx + k]);
}
static inline float vector_v(const device IN_T* v, uint k, uint r, constant MBVectorDims& p) {
    return float(v[TRANS_B ? r * p.ldv + k : k * p.ldv + r]);
}
static inline void vector_store(device IN_T* c, float value, uint w, uint r,
                                 constant MBVectorDims& p
#if MB_MPS_EPILOGUE
                                 , const device IN_T* bias, constant ulong* bstride, uint batch
#endif
                                 ) {
    device IN_T* row = c + (TRANS_OUT ? r * p.ldc : w * p.ldc);
#if MB_MPS_EPILOGUE
    ulong bi = ulong(batch) * bstride[0] + ulong(TRANS_OUT ? r : w) * bstride[1]
             + ulong(TRANS_OUT ? w : r) * bstride[2];
    row[TRANS_OUT ? w : r] = mb_mps_epilogue(value, bias, bi);
#else
    row[TRANS_OUT ? w : r] = IN_T(value);
#endif
}

#if FLAT && R < 8
// TemplateTransA: one K element per y lane, four adjacent W outputs per x.
// This template does not consume the encoder's ordinary-dot K-unroll field.
template<bool EDGE_W>
static inline void vector_trans_a_step(const device IN_T* x, const device IN_T* v,
                                        thread float4 (&acc)[R], uint w,
                                        constant MBVectorDims& p) {
    float4 a = 0;
    if (!EDGE_W || w + 3 < p.W) a = float4(*reinterpret_cast<const device LOAD4_T*>(x));
    else {
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c)
            if (w + c < p.W) a[c] = float(x[c]);
    }
    float b[R];
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) b[r] = float(v[TRANS_B ? r * p.ldv : r]);
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) acc[r] += a * b[r];
}

template<bool EDGE_W>
static inline void vector_trans_a_body(const device IN_T* x, const device IN_T* v,
                                        thread float4 (&acc)[R], uint w, uint y,
                                        constant MBVectorDims& p) {
    if (EDGE_W && w >= p.W) return;
    x += y * p.ldx;
    x += w;
    v += TRANS_B ? y : y * p.ldv;
    // AIR advances row pointers between complete strips. Its full and partial
    // W threadgroups use separate K loops, so full groups need no lane masks.
    uint x_step = GY * p.ldx, v_step = TRANS_B ? GY : GY * p.ldv;
    for (uint strip = 0; strip < p.K / GY; ++strip) {
        vector_trans_a_step<EDGE_W>(x, v, acc, w, p);
        x += x_step;
        v += v_step;
    }
    if (y < p.K % GY) vector_trans_a_step<EDGE_W>(x, v, acc, w, p);
}

kernel void mps_vector(
    const device IN_T* X [[buffer(0)]], const device IN_T* V [[buffer(1)]],
    device IN_T* C [[buffer(2)]], constant MBVectorDims& p [[buffer(3)]],
    const device IN_T* zero [[buffer(4)]],
#if MB_MPS_EPILOGUE
    const device IN_T* bias [[buffer(5)]], constant ulong* bstride [[buffer(6)]],
#endif
    uint3 group [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]]) {
    X += group.z * p.bsx;
    V += group.z * p.bsv;
    C += group.z * p.bsc;
    uint w = (group.x * GX + t.x) * 4;
    float4 acc[R];
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) acc[r] = 0;
    if ((group.x + 1) * GX * 4 <= p.W) vector_trans_a_body<false>(X, V, acc, w, t.y, p);
    else vector_trans_a_body<true>(X, V, acc, w, t.y, p);
    // [component][narrow row][x lane][y lane], without padding.
    threadgroup float partial[4 * R * GX * GY];
    for (ushort c = 0; c < 4; ++c)
        for (ushort r = 0; r < R; ++r)
            partial[((c * R + r) * GX + t.x) * GY + t.y] = acc[r][c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tid = t.y * GX + t.x;
#if GY >= 32
    // Apple-device AIR uses this branch when kNumThreadsY >= 32 and SIMD
    // reduction is supported. Each SIMD owns output slots, reading the
    // shared K rows in 32-element strips; only lane zero writes the result.
    uint lane = tid % 32;
    for (ushort q = tid / 32; q < 4 * R * GX; q += GX * GY / 32) {
        uint column = group.x * GX * 4 + (q % GX) * 4 + q / (R * GX);
        if (column >= p.W) continue;
        float sum = 0;
        for (ushort s = 0; s < GY; s += 32)
            sum += simd_sum(partial[q * GY + s + lane]);
        if (lane == 0) vector_store(C, sum, column, (q / GX) % R, p MB_VECTOR_EPILOGUE_ARGS);
    }
#else
    uint y = tid % GY;
    for (ushort s = GY / 2; s; s /= 2) {
        if (y < s)
            for (ushort q = tid / GY; q < 4 * R * GX; q += GX)
                partial[q * GY + y] += partial[q * GY + y + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (ushort q = tid; q < 4 * R * GX; q += GX * GY) {
        uint column = group.x * GX * 4 + (q % GX) * 4 + q / (R * GX);
        if (column < p.W) vector_store(C, partial[q * GY], column, (q / GX) % R, p MB_VECTOR_EPILOGUE_ARGS);
    }
#endif
}
#else
// Ordinary template: dot(float4, float4) accumulates into a scalar per output.
template<bool EDGE_K>
static inline void vector_dot_step(const device IN_T* xp, const device IN_T* vp,
                                    thread float (&acc)[R][ROWS], uint w, uint k,
                                    uint4 x_offsets, uint4 v_offsets,
                                    constant MBVectorDims& p, const device IN_T* zero) {
    bool valid_k = !EDGE_K || k < (p.K & ~3u);
    // AIR forms the row pointer before adding the component offsets. Keep
    // the constant narrow-row increments outside 32-bit index arithmetic,
    // allowing the compiler to combine adjacent rows into vector loads.
    // AIR prefetches in the input type, then converts each dot's operands.
    // In particular, bfloat/half vectors stay narrow while other rows load.
    vec<IN_T, 4> a[ROWS], b[R];
    #pragma clang loop unroll(full)
    for (ushort y = 0; y < ROWS; ++y) {
        bool valid_x = valid_k && ((ROWS == 1 && GY == 1) || w + y < p.W);
        if (!FLAT) {
            const device IN_T* row = valid_x ? xp + y * p.ldx : zero;
            a[y] = vec<IN_T, 4>(*reinterpret_cast<const device LOAD4_T*>(row));
        } else {
            const device IN_T* row = xp + y;
            a[y] = vec<IN_T, 4>(*(valid_x ? row + x_offsets.x : zero),
                                *(valid_x ? row + x_offsets.y : zero),
                                *(valid_x ? row + x_offsets.z : zero),
                                *(valid_x ? row + x_offsets.w : zero));
        }
    }
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) {
        if (TRANS_B) {
            const device IN_T* row = valid_k ? vp + r * p.ldv : zero;
            b[r] = vec<IN_T, 4>(*reinterpret_cast<const device LOAD4_T*>(row));
        } else {
            const device IN_T* row = vp + r;
            b[r] = vec<IN_T, 4>(*(valid_k ? row + v_offsets.x : zero),
                                *(valid_k ? row + v_offsets.y : zero),
                                *(valid_k ? row + v_offsets.z : zero),
                                *(valid_k ? row + v_offsets.w : zero));
        }
    }
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r)
        #pragma clang loop unroll(full)
        for (ushort y = 0; y < ROWS; ++y) acc[r][y] += dot(float4(a[y]), float4(b[r]));
}

kernel void mps_vector(
    const device IN_T* X [[buffer(0)]], const device IN_T* V [[buffer(1)]],
    device IN_T* C [[buffer(2)]], constant MBVectorDims& p [[buffer(3)]],
    const device IN_T* zero [[buffer(4)]],
#if MB_MPS_EPILOGUE
    const device IN_T* bias [[buffer(5)]], constant ulong* bstride [[buffer(6)]],
#endif
    uint3 group [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]]) {
    X += group.z * p.bsx;
    V += group.z * p.bsv;
    C += group.z * p.bsc;
    constexpr bool tiled = GY > 1;
    constexpr uint k_threads = tiled ? GY : GX;
    uint w = tiled ? group.y * GX + t.x : group.y * ROWS;
    uint lane_k = tiled ? t.y : t.x;
    float acc[R][ROWS];
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r)
        #pragma clang loop unroll(full)
        for (ushort y = 0; y < ROWS; ++y) acc[r][y] = 0;
    uint full_groups = GROUPS ? FULL_GROUPS : p.full_groups;
    uint groups = GROUPS ? GROUPS : p.groups;
    const device IN_T* xp = X + (FLAT ? w : w * p.ldx);
    xp += FLAT ? lane_k * 4 * p.ldx : lane_k * 4;
    const device IN_T* vp = V + (TRANS_B ? lane_k * 4 : lane_k * 4 * p.ldv);
    uint4 x_offsets = uint4(0, p.ldx, 2 * p.ldx, 3 * p.ldx);
    uint4 v_offsets = uint4(0, p.ldv, 2 * p.ldv, 3 * p.ldv);
    // AIR advances pointers along contiguous K and 32-bit component offsets
    // along strided K. Preserve both instead of rebuilding K*stride each step.
    constexpr uint step = 4 * k_threads;
    uint block = 0;
    for (; block < full_groups; block += UNROLL_K)
        #pragma clang loop unroll(full)
        for (ushort u = 0; u < UNROLL_K; ++u) {
            vector_dot_step<false>(xp, vp, acc, w, ((block + u) * k_threads + lane_k) * 4,
                                   x_offsets, v_offsets, p, zero);
            if (FLAT) x_offsets += step * p.ldx;
            else xp += step;
            if (TRANS_B) vp += step;
            else v_offsets += step * p.ldv;
        }
#if PARTIAL
    for (; block < groups; ++block) {
        vector_dot_step<true>(xp, vp, acc, w, (block * k_threads + lane_k) * 4,
                              x_offsets, v_offsets, p, zero);
        if (FLAT) x_offsets += step * p.ldx;
        else xp += step;
        if (TRANS_B) vp += step;
        else v_offsets += step * p.ldv;
    }
#endif
#if GY > 1
    threadgroup float partial[R * GX * GY];
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) partial[(r * GX + t.x) * GY + t.y] = acc[r][0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort s = GY / 2; s; s /= 2) {
        #pragma clang loop unroll(full)
        for (ushort r = 0; r < R; ++r)
            if (t.y < s) partial[(r * GX + t.x) * GY + t.y] += partial[(r * GX + t.x) * GY + t.y + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) acc[r][0] = partial[(r * GX + t.x) * GY];
#else
    #pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r)
        #pragma clang loop unroll(full)
        for (ushort y = 0; y < ROWS; ++y) acc[r][y] = simd_sum(acc[r][y]);
#endif
    if (lane_k == 0) {
        #pragma clang loop unroll(full)
        for (ushort r = 0; r < R; ++r)
            #pragma clang loop unroll(full)
            for (ushort y = 0; y < ROWS; ++y) {
                if (w + y >= p.W) continue;
                float sum = acc[r][y];
                // AIR handles scalar tails after the reduction, in reverse K order.
                #pragma clang loop unroll(full)
                for (ushort tail = K_MOD4; tail > 0; --tail) {
                    uint k = (p.K & ~3u) + tail - 1;
                    sum += vector_x(X, w + y, k, p) * vector_v(V, k, r, p);
                }
                vector_store(C, sum, w + y, r, p MB_VECTOR_EPILOGUE_ARGS);
            }
    }
}
#endif
#endif
