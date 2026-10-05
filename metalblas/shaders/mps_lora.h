// LORADOWN_gemv_kernel schedule recovered from MPSNDArray's AIR.
#ifdef MB_BUILD_MPS_LORA
#include <metal_stdlib>
using namespace metal;

#include "mps_epilogue.h"

#define ROWS __ROWS__
#define ROW_TILE __ROW_TILE__
#define UNROLL_K __UNROLL_K__
#define NSIMD __NSIMD__
#define VEC_B __VEC_B__
#define K_MOD4 __K_MOD4__
#define K_LOOPS __K_LOOPS__
#define TILED_TAIL __TILED_TAIL__
#define TG_TAIL __TG_TAIL__

struct MBLoRADims { uint K, lda, ldb, ldc; };

static inline half4 lora_load_b(const device half* b, uint offset) {
    if (VEC_B) return *reinterpret_cast<const device packed_half4*>(b + offset);
    // The non-divisible stride path is a single half4 load in Apple's AIR,
    // including its vector alignment, rather than four scalar loads.
    return *reinterpret_cast<const device half4*>(b + offset);
}

kernel void mps_lora(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant MBLoRADims& p [[buffer(3)]],
#if MB_MPS_EPILOGUE
    const device half* bias [[buffer(4)]], constant ulong* bstride [[buffer(5)]],
#endif
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint threads = NSIMD * 32;
    A += group.y * p.lda;
    float4 acc[16];
    float sum[16];
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < ROWS; ++n) { acc[n] = 0; sum[n] = 0; }

    uint vectors = p.K / 4;
    uint base = 0;
    // The encoder specializes 0/1/2 iterations; 3 means a runtime count.
    uint iterations = K_LOOPS < 3 ? K_LOOPS : vectors / (threads * UNROLL_K);
    // The row tile limits live prefetched data. It does not divide output
    // rows among SIMD groups: every group accumulates all ROWS rows.
    for (uint i = 0; i < iterations; ++i, base += threads * UNROLL_K) {
        half4 a[16];
        #pragma clang loop unroll(full)
        for (ushort u = 0; u < UNROLL_K; ++u)
            a[u] = *reinterpret_cast<const device packed_half4*>(A + (base + tid + u * threads) * 4);
        #pragma clang loop unroll(full)
        for (ushort n0 = 0; n0 < ROWS; n0 += ROW_TILE) {
            half4 b[15][16];
            #pragma clang loop unroll(full)
            for (ushort u = 0; u < UNROLL_K; ++u)
                #pragma clang loop unroll(full)
                for (ushort n = 0; n < ROW_TILE; ++n)
                    b[n][u] = lora_load_b(B, (n0 + n) * p.ldb + (base + tid + u * threads) * 4);
            #pragma clang loop unroll(full)
            for (ushort u = 0; u < UNROLL_K; ++u)
                #pragma clang loop unroll(full)
                for (ushort n = 0; n < ROW_TILE; ++n)
                    acc[n0 + n] += float4(b[n][u]) * float4(a[u]);
        }
    }
    // First consume complete threadgroup strips, then the partial strip.
#if TILED_TAIL
    for (; base + threads <= vectors; base += threads) {
        float4 a = float4(*reinterpret_cast<const device packed_half4*>(A + (base + tid) * 4));
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < ROWS; ++n)
            acc[n] += float4(lora_load_b(B, n * p.ldb + (base + tid) * 4)) * a;
    }
#endif
#if TG_TAIL
    if (base + tid < vectors) {
        float4 a = float4(*reinterpret_cast<const device packed_half4*>(A + (base + tid) * 4));
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < ROWS; ++n)
            acc[n] += float4(lora_load_b(B, n * p.ldb + (base + tid) * 4)) * a;
    }
#endif
#if K_MOD4
    if (tid < K_MOD4) {
        uint k = p.K - K_MOD4 + tid;
        half a = A[k];
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < ROWS; ++n)
            // AIR's scalar remainder multiplies in half before extending.
            sum[n] += float(half(B[n * p.ldb + k] * a));
    }
#endif
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < ROWS; ++n)
        sum[n] += ((acc[n].x + acc[n].y) + acc[n].z) + acc[n].w;
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < ROWS; ++n) sum[n] = simd_sum(sum[n]);

#if NSIMD > 1
    threadgroup float partial[8][16];
    if (lane < ROWS) partial[sg][lane] = sum[lane];
    for (ushort s = NSIMD / 2; s; s /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < s)
            #pragma clang loop unroll(full)
            for (ushort n = 0; n < ROWS; ++n)
                partial[tid][n] += partial[tid + s][n];
    }
#endif
    if (tid == 0)
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < ROWS; ++n) {
#if NSIMD > 1
            float value = partial[0][n];
#else
            float value = sum[n];
#endif
#if MB_MPS_EPILOGUE
            C[group.y * p.ldc + n] = mb_mps_epilogue(value, bias,
                                        ulong(group.y) * bstride[1] + ulong(n) * bstride[2]);
#else
            C[group.y * p.ldc + n] = half(value);
#endif
        }
}
#endif
