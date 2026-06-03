// metalblas.metal - binder. The loader (../kernels.py) enables one #ifdef-guarded family
// via -DMB_BUILD_<NAME>, so gemv/simd builds skip the MetalPerformancePrimitives headers.

#include "mb_epi.h"
#include "simd_gemm.h"
#include "mpp_gemm.h"
#include "mpp_tensor.h"
#include "splitk.h"
#include "conv1x1.h"
#include "gemv_nt.h"
#include "gemv_t.h"
#include "cgemv_nt.h"
#include "cgemv_t.h"
#include "complex_pack.h"
#include "int_gemm.h"
