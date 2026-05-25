# metalBLAS

Hand-tuned Metal Shading Language matmul kernels for Apple Silicon, callable
from PyTorch on `mps`.  Built on top of `torch.mps.compile_shader` (PyTorch
2.12+) – every kernel is a templated heredoc string that gets JIT-compiled by
the Metal driver.

## Highlights

* **Three complementary kernel families**
  - `m5_tensor` – uses `mpp::tensor_ops::matmul2d` with `tensor_inline` device
    views and `execution_simdgroups<4>`.  Each TG hands a (BM × BN) output
    tile and the full K dim to one `op.run` call — the MPP runtime handles
    loads internally.  Wins on large bf16 / fp16 GEMM where it matches or
    beats `torch.matmul`.
  - `m5_gemm` – manual threadgroup-tiled GEMM using `mpp::tensor_ops::matmul2d`
    with `16×32×16` cooperative-tensor fragments.  Used as a fallback for
    shapes the MPP path can't slot into, and as the primary backend for
    fp32 (where we beat torch by ~ 1.4-1.8×).
  - `simd_gemm` – portable tiled GEMM using `simdgroup_matrix<T, 8, 8>` MMA,
    works on anything with Metal 3 simdgroup matrix intrinsics.
* **`gemv_nt` / `gemv_t`** – bandwidth-bound rank-1 fast paths with coalesced
  loads.
* **Auto-dispatch** picks the right backend / tile from the shape and dtype.
  bf16/fp16 large GEMM goes to `m5_tensor`; fp32, small problems, GEMV, and
  non-divisible odd shapes go to the manual path.

## Benchmarks (M5 Pro, 15 GPU cores, macOS 26.4.1)

Speedups against `torch.matmul` on MPS, median over each shape group.
Bench: warmup 50, iters 200.

### bfloat16 — the dtype that matters for inference / training

| Group  | Median speedup | Best speedup |
| ------ | -------------- | ------------ |
| square | **1.00×**      | 2.85× (512³) |
| tall   | **0.98×**      | 1.03×        |
| attn   | **1.01×**      | **1.04×**    |
| llm    | **1.00×**      | **1.21×**    |
| gemv   | 0.96×          | 1.02×        |
| odd    | 0.96×          | **1.07×**    |

Concrete numbers (m5_tensor backend, auto-picked tile per shape):

```
   shape              torch              metalBLAS         speedup
4096³                5.41 ms 25.4 TF | 5.42 ms 25.4 TF      1.00×
2048³                0.67 ms 25.5 TF | 0.69 ms 24.8 TF      0.98×
1024³                0.09 ms 23.7 TF | 0.09 ms 23.7 TF      1.00×
512³                 0.04 ms  6.6 TF | 0.02 ms 14.6 TF      2.29×
256³                 0.01 ms  2.6 TF | 0.01 ms  2.5 TF      ~1×  (noisy)
4097³                7.54 ms 18.3 TF | 7.26 ms 19.0 TF      1.04×
2048×14336×4096      9.30 ms 25.9 TF | 9.32 ms 25.8 TF      1.00×
4096×4096×11008     15.51 ms 23.8 TF | 15.90 ms 23.2 TF     0.98×
4096×11008×4096     14.71 ms 25.1 TF | 14.63 ms 25.3 TF     1.01×
8192×1024×1024       0.66 ms 25.9 TF | 0.67 ms 25.5 TF      0.99×
4096×4096×128        0.18 ms 24.3 TF | 0.18 ms 24.1 TF      0.99×
4096×4096×64         0.15 ms 14.5 TF | 0.15 ms 14.5 TF      1.00×
gemv 1×32000×4096    0.99 ms        | 1.00 ms              0.99×
gemv 4096×1×4096     0.11 ms        | 0.12 ms              0.97×
```

### float32

| Group  | Median speedup | Best speedup |
| ------ | -------------- | ------------ |
| square | **1.51×**      | 2.25×        |
| tall   | **1.77×**      | 1.95×        |
| attn   | **1.60×**      | 2.10×        |
| llm    | **1.63×**      | 1.69×        |
| odd    | **1.31×**      | 2.12×        |
| gemv   | 1.01×          | 1.07×        |

We beat torch by ≈ 1.4–2.0× on fp32 — torch's MPS fp32 path appears to use
strict fp32 while we use the M5 tensor unit with TF32-style relaxed precision
(still > 4× more precise than bf16).

### float16

Practically identical to bf16 — same auto-dispatch path picks `m5_tensor`
for large problems.  **1.00× llm median**, 0.99× attn, 0.98× tall.

## Usage

```python
import torch
from metalblas import matmul

a = torch.randn(2048, 4096, device="mps", dtype=torch.bfloat16)
b = torch.randn(4096, 2048, device="mps", dtype=torch.bfloat16)
c = matmul(a, b)   # auto picks the best backend / tile
```

You can also override the backend and tile:

```python
c = matmul(a, b, backend="m5_tensor", tile=(64, 128, 4))         # (BM, BN, NSG)
c = matmul(a, b, backend="m5",        tile=(128, 64, 64, 4, 2))  # (BM, BN, BK, WM, WN)
c = matmul(a, b, backend="simd")
c = matmul(a, b, backend="gemv")                                 # for rank-1 problems
```

Supported dtypes: `bfloat16`, `float16`, `float32`.  Transposed inputs
are handled by detecting strides – an `.t()` view of a contiguous tensor
is recognised and dispatched to the right kernel specialisation without
copying.

## How `m5_tensor` works

The MPP `matmul2d` op accepts `tensor<device T, dextents<int32_t, 2>>`
operands directly and runs cooperatively across `execution_simdgroups<N>`
simdgroups in the threadgroup.  Critical details that made it work through
PyTorch's `compile_shader`:

```cpp
// Const-qualified element type breaks the cooperative-tensor static_asserts.
tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA(
    const_cast<device bfloat*>(A), dextents<int32_t, 2>(K, M));

// Descriptor with dynamic_extent K — one op.run does the full K-loop.
constexpr auto desc = matmul2d_descriptor(
    BM, BN, dynamic_extent, false, false, false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroups<NSG>> op;

// Edge tiles handled natively: slice(n_off, m_off) returns a sub-tensor with
// extents clipped to what's left of the original; Metal returns zero for
// OOB buffer reads; cT.store(mC) only writes valid positions.
auto mA = tA.slice(0, m_off);
auto mB = tB.slice(n_off, 0);

// Compute in fp32, convert to OUT_T for store.  When the kernel is compiled
// for MN_ALIGNED shapes we skip the is_valid_element check in the convert
// loop — every cT slot maps to a valid output position so no branch is needed.
auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
op.run(mA, mB, cT_f32);

auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
    if (cT_f32.is_valid_element(i)) cT_out[i] = (OUT_T)cT_f32[i];
cT_out.store(tC.slice(n_off, m_off));
```

The dispatch grid is plain `(tiles_n, tiles_m, 1)`.  No external swizzle:
MPP has its own internal scheduling and an extra swizzle slows it down by
~ 15-20 %.

### Tile picker by regime (M5 Pro)

The picker chooses `(BM, BN, NSG)` from the empirical sweep results.  Different
shape regimes prefer different tile sizes:

| Regime                                       | Tile           | Why                                        |
| -------------------------------------------- | -------------- | ------------------------------------------ |
| Tiny K (≤ 256) over large M, N               | (32, 128, 4)   | More tiles in flight amortize K-loop cost  |
| Non-divisible large (≥ 1024, edge tiles)     | (128, 128, 8)  | Fewer total tiles, big interior            |
| Deep-K (K ≥ 2·max(M, N))                     | (64, 128, 4)   | Wider BN amortizes the per-tile K-loop     |
| Very small (M, N ≤ 256)                      | (32, 128, 4)   | A few bigger TGs beat many tiny ones       |
| Small-medium (M, N ≤ 768)                    | (32, 64, 4)    | More tiles, NSG=4 keeps cores busy         |
| Default divisible (≥ 1024)                   | (64, 64, 2)    | Light TG threading + good K-reuse balance  |

The dispatcher also routes big-N GEMV (M=1, N ≥ 8192) through the m5_tensor
kernel — Metal's OOB-zero-reads make the over-fetched rows free, and the
runtime's bandwidth handling beats the dedicated GEMV kernel above N ≈ 8000.

## Repo layout

```
metalblas/
  __init__.py       exposes `matmul`
  kernels.py        Metal shader source + JIT cache
  dispatch.py       Python dispatch, tile picker
bench/
  bench_matmul.py   benchmark vs torch.matmul on MPS
tests/
  test_basic.py     correctness tests vs the math reference
```

## Running

```bash
python tests/test_basic.py
python bench/bench_matmul.py --dtype bf16
python bench/bench_matmul.py --dtype fp32 --group llm
python bench/bench_matmul.py                  # full sweep across dtypes/groups
```
