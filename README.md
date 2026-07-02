<p align="center">
  <img src="assets/metalblas-logo.svg" alt="metalBLAS" width="520">
</p>

Hand-tuned Metal matmul kernels for Apple Silicon, callable from PyTorch on `mps`. Every kernel is a templated MSL string JIT-compiled at runtime via `torch.mps.compile_shader` (requires PyTorch 2.12+ and an MPS device).

`matmul` matches or beats `torch.matmul` on bf16/fp16, runs **2-3x faster on fp32** (M-series tensor unit, TF32-relaxed precision), and adds **complex64 / complex32** support that beats torch's complex matmul on every config (**~2x** on typical GEMM, **up to ~5-6x** on large / LLM shapes and rank-1 GEMV). It also covers every **integer** dtype torch supports on MPS (`int8`/`uint8`/`int16`/`int32`/`int64`) **~1.4-3x** on GEMM and **up to ~12x** on rank-1 GEMV.

## Usage

```python
import torch
from metalblas import matmul

a = torch.randn(2048, 4096, device="mps", dtype=torch.bfloat16)
b = torch.randn(4096, 2048, device="mps", dtype=torch.bfloat16)
c = matmul(a, b)   # auto-picks backend + tile
```

Supported dtypes: `bfloat16`, `float16`, `float32`, `complex64`, `complex32`, and the integer types `int8` / `uint8` / `int16` / `int32` / `int64`. Complex and integer inputs are auto-detected, so the same `matmul(a, b)` call handles them (including `conj()` / transposed views):

```python
a = torch.randn(1024, 1024, device="mps", dtype=torch.complex64)
b = torch.randn(1024, 1024, device="mps", dtype=torch.complex64)
c = matmul(a, b)   # complex GEMM, ~2x torch.matmul

a = torch.randint(-8, 8, (1024, 1024), device="mps", dtype=torch.int32)
b = torch.randint(-8, 8, (1024, 1024), device="mps", dtype=torch.int32)
c = matmul(a, b)   # integer GEMM, ~3x torch.matmul
```

`addmm` is the fused BLAS GEMM `C = beta*input + alpha*(mat1 @ mat2)` that
`nn.Linear` lowers to. The bias and scales are fused into the GEMM/GEMV kernels, so it
runs at ~parity with `matmul` for every dtype above and matches `torch.addmm` (broadcast
bias, `beta`/`alpha` scaling, the `beta==0` drop-the-bias rule, bit-exact for integers):

```python
from metalblas import addmm

x = torch.randn(2048, 4096, device="mps", dtype=torch.bfloat16)
w = torch.randn(4096, 1024, device="mps", dtype=torch.bfloat16)
bias = torch.randn(1024, device="mps", dtype=torch.bfloat16)
y = addmm(bias, x, w)                 # == nn.Linear(x, w, bias)
y = addmm(bias, x, w, beta=0.5, alpha=2.0)
```

`conv1d` / `conv2d` / `conv3d` mirror `torch.nn.functional` (stride, padding
incl. `'same'`/`'valid'`, dilation, groups, bias, NCHW + channels_last, batched
or not). bf16 and fp32 run on custom kernels; fp16 currently falls back to torch:

```python
from metalblas import conv2d

x = torch.randn(8, 256, 14, 14, device="mps", dtype=torch.bfloat16)
w = torch.randn(256, 256, 3, 3, device="mps", dtype=torch.bfloat16)
y = conv2d(x, w, bias, padding=1)     # ~1.3x torch; conv3d runs ~10x
```

You can override the backend and tile for the real dtypes if needed:

```python
c = matmul(a, b, backend="mpp_tensor", tile=(64, 128, 4))  # (BM, BN, NSG)
c = matmul(a, b, backend="gemv") # rank-1 problems
```

## How it works

Dispatch picks a kernel from shape and dtype:

- **`mpp_tensor`** - the primary path for nearly everything. Uses Apple's
  `mpp::tensor_ops::matmul2d` on the tensor unit, with static-extent tile slices
  so interior tiles skip per-tile edge predication. A strided `tensor_inline`
  view (leading dim `lda`/`ldb` + the descriptor's transpose flags) lets the same
  path consume col-major and `[::2]`-strided operands directly, with no copy.
- **`flipt`** - thin-N overlap variant probed by the autotuner. `matmul2d`
  alone leaves the streamed A operand ~15% under-overlapped with the MMA there;
  `flipt` computes `C^T = B^T A^T` (the flipped RHS streams A near its MMA
  roof; N=128..256) as a full-N-width tile with a chunked A prefetch,
  transposing through threadgroup memory on store.
- **`mpp_sgpipe`** - for the narrowest streamed band (N=64..128 over long M),
  even the prefetch variants cap ~8% under MPS (the TG-scope op under-overlaps
  its operand staging). Here each simdgroup runs its own `matmul2d` and loads
  its A K-chunk into the op's input cooperative tensor (registers) itself,
  with K baked into the kernel so the chunk loop fully unrolls. Single-buffered
  on purpose: a second buffer, a touch-prefetch, or a streamer SG all measure
  slower (register cliff / issue-slot theft).
- **`gemv_nt` / `gemv_t`** - bandwidth-bound rank-1 fast paths (M=1 / N=1) with
  cache-line-wide coalesced loads.
- **`mpp_gemm` / `simd_gemm`** - threadgroup-tiled fallbacks for sub-tile-floor
  shapes (and the general path when Metal 4 is unavailable).
- **complex** - `cgemv_t` / `cgemv_nt` are native interleaved-complex GEMV
  kernels (read the matrix once as `float2`/`half2`, fp32 accumulate). Complex
  GEMM deinterleaves into real planes and runs four real products
  `(ar@br - ai@bi) + i(ar@bi + ai@br)`, folding them back to interleaved complex in one fused pass (`complex_pack`). complex64 thus rides the TF32-relaxed fp32 path and inherits its speed.
- **integer** - `simdgroup_matrix` and the tensor unit are float-only, so integers
  ride a register-tiled GEMM (`int_gemm`) plus the existing `gemv_t` / `gemv_nt`
  kernels. Accumulating at the output width or wider (32-bit for the 8-/16-bit
  types) and truncating to the output width is bit-identical to torch's
  wrap-on-overflow (two's-complement add/mul are mod 2^w), so there's no precision
  tradeoff - integer matmul matches torch exactly.
- **addmm** - `C = beta*input + alpha*(A@B)`. Each backend builds an `EPILOGUE`
  variant (a shared `mb_epi` store helper) that applies the broadcast bias and
  `beta`/`alpha` scaling, so addmm costs ~the same as the bare matmul. (Complex folds
  the bias into the four-product `complex_combine` pass it already does.)
  `beta`/`alpha == 0` are compiled out so a skipped operand's NaN/Inf can't leak into
  `C`, matching torch's drop-the-operand rule. fp32 ~2x and integer ~3x vs
  `torch.addmm`; bf16/fp16 ~parity.

- **conv** - `mpp::tensor_ops::convolution2d` on the tensor unit, NHWC-native
  with the tile + kernel/stride/dilation baked per build and shape dims left
  runtime (tensor extents carry the bounds). channels_last tensors are already
  NHWC memory and run zero-copy; NCHW inputs pay one tiled-transpose pass in
  and scatter straight back to NCHW from the cooperative store (bias fused).
  1x1 stride-1 convs route to the GEMM stack (batched broadcast-weights `bmm`
  for NCHW - no layout work at all), stride-1 conv1d to a shift-GEMM kernel
  (k accumulated column-shifted `matmul2d` runs on raw NCL memory - no layout
  transforms either direction; a tiny pass fixes the left-pad columns),
  depthwise (`groups == C`) to a direct bandwidth-bound kernel (~2.5-6x torch),
  and conv3d runs an in-kernel accumulating depth loop over
  `mode::multiply_accumulate` (~10x torch, whose MPSGraph conv3d is very slow).
  Transformed weights are cached across calls keyed on tensor identity +
  version + layout. The autotuner also probes torch itself (with deep warmup -
  MPSGraph keeps tiering up over its first ~100 calls) and keeps a shape on our
  kernels only when they win by >6%; near-ties route to torch, so parity is the
  floor and our kernels only own shapes they solidly win.

A runtime autotuner probes a short tile-candidate list on the real operands the
first time it sees a bf16/fp16 shape and caches the winner (disable with
`METALBLAS_AUTOTUNE=0`). The tile-picker logic lives in `metalblas/dispatch.py`
(matmul) and `metalblas/conv.py` (conv).

## Benchmarks

Speedup vs `torch.matmul` on MPS (M5 Pro, 15 GPU cores, macOS 26.4.1; isolated
best-of-N):

| dtype       | result                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| bf16 / fp16 | parity-or-better across large / LLM / deep-K GEMM; up to ~2.8x on small shapes |
| fp32        | 2-3x across the board (median 2.5x square, 2.1x LLM)                   |
| complex64   | square/medium GEMM ~1.3-3.3x (median ~2.1x); large 4096³ ~5.6x and LLM ~5.8x (torch's complex GEMM scales poorly); GEMV ~2.0-4.7x (native interleaved kernels, ~280 GB/s vs torch's ~138) |
| int8 / uint8 | square GEMM ~1.8-2.1x (median); int16 ~2.2x, int32 ~3.0x, int64 ~1.4x; thin/LLM-shaped ~1.3-3.1x; rank-1 GEMV ~2.5-12x (coalesced loads vs torch's bandwidth-starved path). |

Speedup vs `torch.nn.functional.conv*` on the same machine (`bench/bench_conv.py`,
ResNet/SD-UNet/MobileNet/Whisper/Mamba/video shape families, every shape swept
over batch 1/2/4/8/16/32). Shapes the kernels don't beat MPSGraph by >6% on
route to torch, so parity is the floor:

| op          | bf16                          | fp32                          |
| ----------- | ----------------------------- | ----------------------------- |
| conv2d      | median 1.68x (1x1s ~2-3.5x, depthwise up to 6.5x) | median 1.98x |
| conv2d (channels_last) | median 1.25x            | median 1.71x                  |
| conv1d      | median 1.33x (Mamba depthwise ~5x) | median 2.73x            |
| conv3d      | median 10.9x (MPSGraph conv3d is very slow) | median 8.1x     |

Per-shape tables: [`perf_benchs/`](perf_benchs/), conv: [`perf_benchs/M5_Pro_conv.md`](perf_benchs/M5_Pro_conv.md).

## Layout

```
metalblas/   kernels.py (MSL source + JIT cache), dispatch.py (matmul dispatch),
             conv.py (conv dispatch), shaders/ (MSL kernel templates)
bench/       bench_matmul.py, bench_conv.py
tests/       test_basic.py, test_conv.py
```

## Running

Install the package and its single dependency (PyTorch 2.12+):

```bash
uv venv
pip install -e .
```

```bash
python tests/test_basic.py
python tests/test_conv.py
python bench/bench_conv.py --dtype bf16                    # conv1d/2d/3d vs torch
python bench/bench_matmul.py --dtype bf16
python bench/bench_matmul.py --dtype fp32 --group llm
python bench/bench_matmul.py --dtype c64 --group square    # complex64
python bench/bench_matmul.py --dtype i32 --group square    # int32 (also i8/u8/i16/i64)
python bench/bench_matmul.py --report          # write perf_benchs/<chip>.md for your Mac

# Layout sweep: same shapes, awkward operand strides (row/col-major, sliced, offset)
python bench/bench_matmul.py --layout all                     # all layouts, bf16
python bench/bench_matmul.py --layout cm_cm,sliced --check    # subset + correctness probe
python bench/bench_matmul.py --report --layout all            # report incl. layout section
```

Layouts: `rm_rm` (row-major, the packed fast path), `rm_cm` / `cm_rm` / `cm_cm`
(col-major views — descriptor transpose flags), `sliced` (`[::2]` rows: leading
dim `lda=2K`), `offset` (`[1:, 1:]`: nonzero storage offset). All of these read
in place on `mpp_tensor` via a strided `tensor_inline` view (no copy); only
`colsl` (`[:, ::2]`, unit stride on neither dim) still needs a contiguous copy.
Each prints a `mb layout tax` line — throughput vs the packed `rm_rm` baseline.

## License

MIT - see [LICENSE](LICENSE).
