<p align="center">
  <img src="assets/metalblas-logo.svg" alt="metalBLAS" width="520">
</p>

Metal matmul kernels for Apple Silicon, exposed as PyTorch operations on `mps`.

## Usage

```python
import torch
from metalblas import matmul

a = torch.randn(2048, 4096, device="mps", dtype=torch.bfloat16)
b = torch.randn(4096, 2048, device="mps", dtype=torch.bfloat16)
c = matmul(a, b)
```

The same API supports `float32`, `bfloat16`, `float16`, `complex64`, `complex32`, and the
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

Floating-point `matmul`, `bmm`, `addmm`, `baddbmm` and `addbmm` use the dispatch rules recovered from
Apple's MPSNDArray binaries, including vector/GEMM routing, tiles, split K
and launch schedules:

- NNA14 GEMM on M1/M2.
- A16 GEMM on M3/M4 and for `float32` on M5.
- A18 GEMM for `float16`/`bfloat16` on M5.
- The recovered ordinary-vector and LoRA schedules where MPS selects them.

No benchmark selects these parameters. Runtime autotuning is disabled, and
the earlier measured M5 engine overrides have been removed. The default
`METALBLAS_TILE_POLICY=mps` and `mps-strict` use the same source-derived rules;
`legacy` retains the previous hand-written static picker for explicit
comparisons. `METALBLAS_AUTOTUNE` does not enable tuning.

The floating-point path also uses the recovered MPSGraph submission
thresholds. Its small native stream bridge is compiled against the installed
PyTorch on first use and cached locally; an Xcode C++ toolchain is required.
Encoder workspaces use MPSCore's private command-buffer cache. The `addbmm`
graph intermediate follows the recovered compiler-descriptor flag: private
temporary storage or a shared placement heap. The bridge depends on the
recorded OS and PyTorch interfaces.

The recovered profiles cover macOS 26/27 for A14/A16 and macOS 27 for A18.
Captured MPS dispatches match these rules on an M2 (macOS 26.3), M3 Ultra
(26.5.2), M4 (26.6) and M5 Pro (27.2); the M4's 26.6 frameworks are
code-identical to 26.5.2. M1 and other core counts use the same binary
branches with their own core count; [`docs/mps_dispatch_numbers.md`](docs/mps_dispatch_numbers.md)
tabulates the resulting numbers for every M1–M5 GPU
(`python tools/mps_dispatch_numbers.py`). Performance parity is close but
not complete on every shape; see the inventory below. A18
global K splits are ported; natural selection on 60+ core A18 devices remains
unvalidated. Morton rasterization raises `NotImplementedError` in the default
path. Bias epilogues follow MPSGraph's linked float arithmetic and
input-dtype scalar constants. `addbmm` follows the recovered batched matmul
and batch-reduction passes, including intermediate dtype rounding. Complex
and integer products retain their existing kernels.
See the [M1–M5 source and validation inventory](docs/mps_matmuls.md) and the
[A18 details](docs/mps_matmul_heuristics.md) for provenance, coverage and
measured limitations.

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
metalblas/   Python dispatch, MPS-derived tile heuristics, and Metal shaders
bench/       matmul and bmm benchmarks
tests/       correctness checks against PyTorch
tools/       MPS probing and Metal dispatch tracing helpers
docs/        reverse-engineering notes on Apple's matmul heuristics
perf_benchs/ per-machine benchmark reports
```

## License

MIT; see [LICENSE](LICENSE).
