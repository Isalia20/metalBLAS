<p align="center">
  <img src="assets/metalblas-logo.svg" alt="metalBLAS" width="520">
</p>

Hand-tuned Metal matmul kernels for Apple Silicon, callable from PyTorch on `mps`. Every kernel is a templated MSL string JIT-compiled at runtime via `torch.mps.compile_shader` (requires PyTorch 2.12+ and an MPS device).

`matmul` matches or beats `torch.matmul` on bf16/fp16 and runs **2-3x faster on fp32** (M-series tensor unit, TF32-relaxed precision) across the benchmark suite.

## Usage

```python
import torch
from metalblas import matmul

a = torch.randn(2048, 4096, device="mps", dtype=torch.bfloat16)
b = torch.randn(4096, 2048, device="mps", dtype=torch.bfloat16)
c = matmul(a, b)   # auto-picks backend + tile
```

Supported dtypes: `bfloat16`, `float16`, `float32`. You can override
the backend and tile if needed:

```python
c = matmul(a, b, backend="m5_tensor", tile=(64, 128, 4))  # (BM, BN, NSG)
c = matmul(a, b, backend="gemv") # rank-1 problems
```

## How it works

Dispatch picks a kernel from shape and dtype:

- **`m5_tensor`** - the primary path for nearly everything. Uses Apple's
  `mpp::tensor_ops::matmul2d` on the tensor unit, with static-extent tile slices
  so interior tiles skip per-tile edge predication.
- **`gemv_nt` / `gemv_t`** - bandwidth-bound rank-1 fast paths (M=1 / N=1) with
  cache-line-wide coalesced loads.
- **`m5_gemm` / `simd_gemm`** - threadgroup-tiled fallbacks for sub-64 dims,
  transposed or non-packed inputs.

A runtime autotuner probes a short tile-candidate list on the real operands the
first time it sees a bf16/fp16 shape and caches the winner (disable with
`METALBLAS_AUTOTUNE=0`). The tile-picker logic lives in `metalblas/dispatch.py`.

## Benchmarks

Speedup vs `torch.matmul` on MPS (M5 Pro, 15 GPU cores, macOS 26.4.1; isolated
best-of-N):

| dtype       | result                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| bf16 / fp16 | parity-or-better across large / LLM / deep-K GEMM; up to ~2.8x on small shapes |
| fp32        | 2-3x across the board (median 2.5x square, 2.1x LLM)                   |

Per-shape tables: [`perf_benchs/`](perf_benchs/).

## Layout

```
metalblas/   kernels.py (MSL source + JIT cache), dispatch.py (dispatch + tile picker)
bench/       bench_matmul.py
tests/       test_basic.py
```

## Running

```bash
python tests/test_basic.py
python bench/bench_matmul.py --dtype bf16
python bench/bench_matmul.py --dtype fp32 --group llm
python bench/bench_matmul.py --report          # write perf_benchs/<chip>.md for your Mac
```

## License

MIT - see [LICENSE](LICENSE).
