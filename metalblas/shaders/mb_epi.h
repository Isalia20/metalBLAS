// Shared fused addmm epilogue.
#ifndef MB_EPI_H
#define MB_EPI_H

#ifndef EPILOGUE
#define EPILOGUE 0
#endif
#ifndef BETA_NZ
#define BETA_NZ 1
#endif
#ifndef ALPHA_NZ
#define ALPHA_NZ 1
#endif

#if EPILOGUE
template <typename O, typename A, typename S>
inline O mb_epi(A acc, device const O *bias, int bidx, S beta, S alpha) {
    S v = (S)0;
#if ALPHA_NZ
    v += alpha * (S)acc;
#endif
#if BETA_NZ
    v += beta * (S)bias[bidx];
#endif
    return (O)v;
}
#endif

#endif  // MB_EPI_H
