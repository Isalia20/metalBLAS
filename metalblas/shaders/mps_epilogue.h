// MPSCore addition_f/multiplication_f/write_f and MPSGraph's linked DAG.
// Source identities and captures are recorded in docs/mps_dispatch_sources.json.
#if MB_MPS_EPILOGUE
template<typename T>
static inline T mb_mps_epilogue(float value, const device T* bias, ulong index) {
    // The linked arithmetic modules disable fast math and use separate
    // float multiplies/adds. Convert to the output dtype only at the store.
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    float result = value * as_type<float>(uint(MB_MPS_ALPHA_BITS));
#if MB_MPS_BETA_NZ
    float term = float(bias[index]) * as_type<float>(uint(MB_MPS_BETA_BITS));
    result = result + term;
#endif
    return T(result);
}
#endif
