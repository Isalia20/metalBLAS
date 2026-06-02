// metalblas.metal - binder. The loader (../kernels.py) enables one #ifdef-guarded family
// via -DMB_BUILD_<NAME>, so gemv/simd builds skip the MetalPerformancePrimitives headers.

#include "simd_gemm.h"
#include "m5_gemm.h"
#include "m5_tensor.h"
#include "splitk.h"
#include "conv1x1.h"
#include "gemv_nt.h"
#include "gemv_t.h"
#include "cgemv_nt.h"
#include "cgemv_t.h"
#include "complex_pack.h"
