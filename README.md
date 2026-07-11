<p align="center">
  <img src="assets/metalblas-logo.svg" alt="metalBLAS" width="520">
</p>

Metal matmul kernels for Apple Silicon, exposed as PyTorch operations on `mps`.
Kernels are compiled on demand with `torch.mps.compile_shader`, so metalBLAS
requires PyTorch 2.12 or newer and an MPS device.

## Usage

```python
import torch
from metalblas import matmul

a = torch.randn(2048, 4096, device="mps", dtype=torch.bfloat16)
b = torch.randn(4096, 2048, device="mps", dtype=torch.bfloat16)
c = matmul(a, b)
```

The same API supports `bfloat16`, `float16`, `complex64`, `complex32`, and the
integer types available in PyTorch on MPS: `int8`, `uint8`, `int16`, `int32`,
and `int64`. Transposed, conjugated, batched, and broadcast operands are handled
without changing the call site.

Fused and batched operations are also available:

```python
from metalblas import addmm, bmm

bias = torch.randn(2048, device="mps", dtype=torch.bfloat16)
y = addmm(bias, a, b, beta=0.5, alpha=2.0)

batch_a = torch.randn(8, 512, 256, device="mps", dtype=torch.float16)
batch_b = torch.randn(8, 256, 512, device="mps", dtype=torch.float16)
batch_c = bmm(batch_a, batch_b)
```

`matmul` chooses a backend and tile automatically. Advanced callers can
override them for real-valued 2-D inputs:

```python
c = matmul(a, b, backend="mpp_tensor", tile=(64, 128, 4))
c = matmul(a, b, backend="gemv")
```

## Dispatch

The main paths are:

- `mpp_tensor` for Metal 4 cooperative-tensor GEMM.
- `gemv_t`, `gemv_nt`, and `gemv_bt` for rank-1 and thin-matrix products.
- `mpp_gemm` and `simd_gemm` for smaller shapes and systems without Metal 4.
- Native complex GEMV plus four real products for complex GEMM.
- Register-tiled kernels for integer GEMM and GEMV.

For ambiguous reduced-precision shapes, a small runtime autotuner measures a
curated set of candidates once and caches the winner. Set
`METALBLAS_AUTOTUNE=0` to use the static tile heuristics.

## Benchmarks

Hardware-specific reduced-precision results are in
[`perf_benchs/`](perf_benchs/). Re-run them on the target machine before making
performance decisions; kernel rankings vary by chip, OS, shape, and layout.

## Development

```bash
uv venv
pip install -e .

python tests/test_basic.py
python bench/bench_matmul.py --dtype bf16
python bench/bench_matmul.py --dtype c64 --group square
python bench/bench_matmul.py --dtype i32 --group square
python bench/bench_bmm.py --dtype bf16
```

Layout benchmarks accept `rm_rm`, `rm_cm`, `cm_rm`, `cm_cm`, `sliced`,
`colsl`, and `offset`:

```bash
python bench/bench_matmul.py --layout all
python bench/bench_matmul.py --layout cm_cm,sliced --check
python bench/bench_matmul.py --report --layout all
```

## Repository layout

```text
metalblas/   Python dispatch and Metal shaders
bench/       matmul and bmm benchmarks
tests/       correctness checks against PyTorch
perf_benchs/ per-machine benchmark reports
```

## License

MIT; see [LICENSE](LICENSE).
