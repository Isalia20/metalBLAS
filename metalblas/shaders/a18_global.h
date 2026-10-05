// Helpers from MPSNDArray's GEMMGlobalKSplits_Initializer and GEMMGlobalReduction AIR.
#ifdef MB_BUILD_A18_GLOBAL
#include <metal_stdlib>
using namespace metal;
#define OUT_T __OUT_T__
#define GLOBAL_SPLITS __GLOBAL_SPLITS__
#include "mps_epilogue.h"

kernel void a18_global_initialize(device uint *counters [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
    if (tid < count) counters[tid] = 0;
}

struct A18GlobalReduceParams {
    uint M, N, batch, ldc;
    ulong outputBatchStride;
};

kernel void a18_global_reduce(device const float *partials [[buffer(0)]],
                              device OUT_T *output [[buffer(1)]],
                              constant A18GlobalReduceParams& p [[buffer(2)]],
#if MB_MPS_EPILOGUE
                              device const OUT_T *bias [[buffer(3)]],
                              constant uint4& biasStride [[buffer(4)]],
#endif
                              uint tid [[thread_position_in_grid]]) {
    const ulong size = ulong(p.M) * p.N;
    const ulong splitStride = size * p.batch;
    if (tid >= splitStride) return;
    float sum = 0.0f;
    for (uint split = 0; split < GLOBAL_SPLITS; ++split)
        sum += partials[ulong(split) * splitStride + tid];
    const uint col = tid % p.N;
    const uint row = (tid / p.N) % p.M;
    const uint batch = tid / size;
#if MB_MPS_EPILOGUE
    const OUT_T value = mb_mps_epilogue(sum, bias,
        ulong(batch) * biasStride.z + ulong(row) * biasStride.x + ulong(col) * biasStride.y);
#else
    const OUT_T value = OUT_T(sum);
#endif
    output[ulong(batch) * p.outputBatchStride + ulong(row) * p.ldc + col] = value;
}
#endif
