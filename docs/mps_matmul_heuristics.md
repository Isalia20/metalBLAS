# How MPSGraph picks matmul kernels on Apple silicon

Reverse-engineering notes for the matmul dispatch inside
`MetalPerformanceShaders.framework` on macOS 27.2 (build 26B5086k), done on
an Apple M5 Pro (16 GPU cores). The goal was to replace metalBLAS's runtime
autotuner with the static per-GPU heuristics Apple ships, so the dispatch can
be upstreamed to PyTorch.

The selection rules below were read from the binaries in the dyld shared cache
(`MPSNDArray`, `MPSCore`, `MetalPerformanceShadersGraph`) and verified on the
live system by tracing Metal dispatches and MPS's own parameter log. The
Python port in `metalblas/mps_heuristics.py` reproduces every one of the
1565 traced parameter vectors on this machine. Binary identities and table
addresses are now recorded in [`mps_dispatch_sources.json`](mps_dispatch_sources.json).
The M1–M4 and M5 fp32 recovery is in [`mps_matmuls.md`](mps_matmuls.md).
The measured engine overrides described historically in section 5.1 have
been removed. They are not part of these recovered Apple rules.

## 1. The call chain

```
torch.mm / addmm / bmm (bf16, f16)
  -> MPSGraph matrixMultiplicationWithPrimaryTensor:secondaryTensor:
  -> MPSGraphMatrixMultiplicationOp (MLIR mps.MatMul)
  -> GPU::MatMulOpHandler (MPSGraph runtime)
  -> MPSNDArrayMatrixMultiplication (MPSNDArray.framework)
  -> per-GPU "device behavior" object
       MPSNDArrayMatMulA14DeviceBehavior   older GPUs, fp32
       MPSNDArrayMatMulA18DeviceBehavior   M5-class GPUs      <- M5 Pro
       MPSNDArrayMatMulA19DeviceBehavior   variant with its own low-core heuristic
       MPSNDArrayMatMulCommonDeviceBehavior
  -> GetKernelDispatchParametersForKey -> 17 dispatch parameters
  -> kernel `gemm_a18` from MPSNDArray's default.metallib
```

MPSGraph itself contains no tile heuristics and never sets autotuning
parameters on the kernel (verified by hooking
`-[MPSNDArrayMatrixMultiplication setAutoTuningParameters:]`). All decisions
happen in MPSNDArray.

Kernels involved on an M5 Pro for 2-D products:

| Case | Kernel |
|---|---|
| bf16/f16 GEMM | `gemm_a18` (one pipeline state for every shape; tiles are runtime parameters) |
| bf16/f16, one tiny dimension | `NDArrayVectorMatrixMultiply_{1_4,5_8,9_12,13_16}` |
| fp32 | `mmul_kernel_a16_float_float_float` (+ `NDArrayMatrixMultiplyReduction` for split-K) via the A14 behavior |

## 2. Device identification (MPSCore)

`MPSDevice::GetMPSDevice` assigns an MPS "generation" code from the Metal
GPU family and the IORegistry `gpu-core-count`:

| Metal family | GPUs | generation code |
|---|---|---|
| Apple7 (1007) | M1, A14 | 11 (<5 cores), 12 (<9), 13 (<17), 14 (<33), 15 (<65) |
| Apple8 (1008) | M2, A15/A16 | 16 (<10 cores), 17 |
| Apple9 (1009) | M3, A17 | 18 (<11 cores), 19 |
| private id -1009 | M4, A18 | 20 (<11 cores), 21 |
| Apple10 (1010) | M5, A19 | 22 (<11 cores), 23 |
| Apple11 (1011) | newer | 24 |

The code sits at offset `0x5c0` of the `MPSDevice` object and the core count
at `0x5c8`. On this machine: generation 23, 16 cores. `MPS_DEVICE_SET=<code>`
overrides it.

## 3. Parameter selection (`GetKernelDispatchParametersForKey`)

Order of precedence in `MPSNDArrayMatMulA18DeviceBehavior`:

1. Explicit parameters on the kernel (`setAutoTuningParameters:` or the
   `MPS_MATMUL_*` environment overrides below). Never set by MPSGraph.
2. Exact match of the 21-word key in a table loaded from
   `MPS_MATMUL_A18_PARAM_TABLE_PATH`.
3. Exact match in a built-in table picked in the constructor:

   | generation | cores | table | entries |
   |---|---|---|---|
   | 22, 23 (M5 class) | < 11 | `_matmulA18GTable` | 118 |
   | 22, 23 | 11–20 | `_matmulA18STable` | 164 |
   | 22, 23 | >= 21 | `_matmulA18CTable` | 118 |
   | >= 24 | < 10 | `_matmulA20PTable` | 127 |
   | >= 24 | 10–12 / 13–20 / >= 21 | G / S / C tables | |
   | <= 21 | any | none (the behavior is not selected on M4 and older Macs, see 5.2) | |

   `MPS_DISABLE_AUTO_TUNE_TABLES=1` turns this off. Duplicate keys exist and
   the first match wins. The tables are shipped unchanged in
   `metalblas/mps_matmul_tables.json`.
4. The heuristic: `_A18GEMMHeuristicForLowCoreCountGPU` when the GPU has
   12 or fewer cores, else `_A18GEMMHeuristicForHighCoreCountGPU`.

The key (`MPSNDArrayA18MatMulKey`) is
`{M, N, K, batch, 1, permA[4], permB[4], dtypeA, dtypeB, quant[6]}` where a
transposed operand has permutation `(1,0,2,3)`, and the dtype codes are
0 int8, 1 uint8, 2 fp8 e4m3, 3 fp8 e5m2, 4 float16, 5 bfloat16, 6 float32,
7 int4, 8 uint4, 9 int2, 10 uint2.

The result (`MPSMatMulAutoTuningParametersA18`, names recovered from the
`MPS_MATMUL_*` environment parser) is:

| index | field | meaning |
|---|---|---|
| 0 | unrollK | tileK = 16·unrollK |
| 1 | unrollM | rows per simdgroup = 16·unrollM |
| 2 | unrollN | cols per simdgroup = 16·unrollN |
| 3 | simdM | simdgroups along M |
| 4 | simdN | simdgroups along N |
| 5 | kSplits | number of local or global K partitions |
| 6 | global K split | 0: local simdgroup partitions; 1: partitions across threadgroups |
| 7 | global reduction mode | 1: completion counter and last-threadgroup reduction; 0: separate `gemmA18GlobalReduction` dispatch |
| 8, 9 | uberM, uberN | threadgroup tiles rasterized together |
| 10 | barrierIterations | K iterations between barriers |
| 11 | tgmemSize | |
| 12 | elasticBarrier | |
| 13 | linearize | |
| 14 | rasterMBias | |
| 15 | morton | |
| 16 | barrierInside | |

Threadgroup tile = `(unrollM·simdM·16) x (unrollN·simdN·16)` with
`simdM·simdN` simdgroups. The parameters are passed in the 400-byte
`MPSNDArrayA18MatrixMultiplicationParams` buffer; the pipeline state does not
change with shape.

Environment overrides read by MPS: `MPS_MATMUL_{UNROLLK,UNROLLM,UNROLLN,
SIMDM,SIMDN,KSPLITS,UBERM,UBERN,BARRIER_ITERATIONS,BARRIER_INSIDE,TGMEM_SIZE,
ELASTIC_BARRIER,LINEARIZE,RASTER_MBIAS,MORTON}`, and `MPS_LOG_INFO=1
MPS_LOG_AUTO_TUNE=1` prints
`MatMul A18 kernel running key with params: {{key}, {params}}` per encode.

### 3.1 High-core heuristic (13–59 cores: M5 Pro/Max)

For 8/16-bit types with both `M > 16` and `N > 16`:

- Delegate to the low-core heuristic when `K > 2048 and K % 32 != 0`, or
  when A is transposed, B is not, and any of M, N, K is not a multiple of 32.
- Otherwise start from a 32x64 tile (unrollM 2, unrollN 2, simdM 1,
  simdN 2, tileK 64, uber 2x2) and shrink it until
  `ceil(M/tileM)·ceil(N/tileN) >= cores`, alternating sides starting with N
  and reducing `unroll` before `simd`: 32x64 -> 32x32 -> 16x32 -> 16x16.
  `unrollK = clamp(ceil(K/16), 1, 4)`, kSplits 1, barrierIterations 4,
  linearize 1, rasterMBias = 1 if `M <= N` else `not transA`, barrierInside 1.

When one dimension is at most 16 (thin shapes):

- If `max(M,N) >= 128·cores` and the thin operand's layout is not the
  "flat" one, use a 16x(up to 64) or (up to 64)x16 tile with kSplits 1 or 2.
- Otherwise split K inside the threadgroup: `kSplits = clamp(ceil(K/tileK),
  1, 8)` with a 16x16 tile (plus a second simdgroup along the long side),
  then the same shrink loop.

Observed on the M5 Pro (bf16, row-major, K = 1024):

| region | params | tile |
|---|---|---|
| M,N in 17..200 (few tiles) | 4,1,1,1,1,1 … uber 2x2 | 16x16, 1 simdgroup |
| M,N in 33..384 | 4,2,1,1,2,1 | 32x32, 2 simdgroups |
| both large | 4,2,2,1,2,1 … uber 2x2 | 32x64, 2 simdgroups |
| M <= 16, N < 500 | 4,1,1,1,1,8 | 16x16, kSplits 8 |
| M <= 16, 500 <= N < 2048 | 4,1,1,1,2,8 | 16x32, kSplits 8 |
| M <= 16, N >= 2048 | 4,1,2,1,4,1 | 16x128, 4 simdgroups |
| N = 16, M >= 500 | 4,1,1,2,1,8 | 32x16, kSplits 8 |
| K = 16/32/48 | unrollK 1/2/3 | tileK = K |

### 3.2 Low-core heuristic (<= 12 cores, and the delegate above)

Direct formulas, no shrink loop:

```
unrollK = clamp(min(K > 4096 ? 3 : 2, ceil(K/16)), 1)      (16-bit types)
mLog = clamp(ceil(log2 M), 4, K>>12 < 3 ? 7 : 8)
nLog = clamp(ceil(log2 N), 4, K>>12 < 3 ? 6 : 7)
simdM = 2 ** min(K>>12 < 3 ? 2 : 3, mLog - 4)
simdN = K>>12 < 3 ? 1 : 2 ** min(nLog - 4, 2)
unrollM = (2**mLog / simdM) / 16,   unrollN = (2**nLog / simdN) / 16
uberM 2, uberN 1, barrierIterations = K > 4096 ? 12 : K > 1024 ? 8 : 0
linearize = (K>>12 > 2), everything else 0
```

This is what produced the 128x64 / 4-simdgroup, tileK 48 config MPS uses
for 4097x4097x4097 and the 128x64 / tileK 32 config for transposed-A
2000x768x2304.

### 3.3 Ultra branch (60+ cores)

The high-core heuristic grows the tile instead of shrinking it while
`tiles > 4·cores` (simdM or simdN doubles on the side with more tiles, up to
4 simdgroups, only while that side has at least 64 tiles), then splits K
across threadgroups when `K > 12288` (`6000` with transposed B) or when the
tile count is below the core count and each split keeps at least 512 of K
(2, 4 or 8 splits). Those splits set fields 6 and 7 and use a completion
counter: the last threadgroup sums float32 partials in split order. The
selector is transliterated from the decompiled code; its natural selection
on a 60+ core A18 device remains unvalidated. Execution and encoder checks
replay the recovered 80-core schedules through explicit MPS parameters on
M5 Pro, as described in section 5.3.

### 3.4 Vector kernel routing (`MPSNDArrayIsVectorKernelPreferred`)

Before any of the above, MPS sends thin products to its vector-matrix
kernels. With `T = 9` if `M·N·3 >= 4096·cores` (weight 4 for fp32) else
`15`:

- both `M > T` and `N > T`: GEMM;
- `N <= T`: vector kernel;
- `M <= T`: vector kernel if `N <= 128`, else GEMM;
- before all of that, `max(M,N) <= 128` with `min(M,N) <= 16` and a
  transposed right operand always takes the vector kernel.

Generation 24+ GPUs use `min(M,N) <= 8`; generation 22 with more than 10
cores additionally requires `min(M,N) <= 4` and `K <= 4096`.

## 4. fp32

On M5-class GPUs the A18 behavior rejects fp32 and falls back to
`MPSNDArrayMatMulA14DeviceBehavior`, which dispatches
`mmul_kernel_a16_float_float_float` in 128-thread groups with 64-wide N
tiles (M tile 32 or 64) and real split-K through
`NDArrayMatrixMultiplyReduction` for small outputs with long K
(e.g. 128x128x16384 splits 16 ways). Its selection lives in
`EncodeMatrixMultiplyLinkingA14`. Its selection is now ported in
`metalblas/mps_a14.py`, with separate macOS 26/27 profiles and dispatch
validation described in [the M1–M5 inventory](mps_matmuls.md). The A16 execution
port now serves default plain fp32 matmul/bmm; `metalblas.mps_a16.matmul`
also exposes it directly for validation without vector/GEMM routing.

## 5. How metalBLAS uses this

`metalblas/mps_heuristics.py` reproduces steps 2–4 (`device_info`,
`select_params`, `vector_kernel_preferred`, `plan`). The default plain
floating-point dispatcher in `metalblas/mps_dispatch.py` connects those
decisions to the execution ports:

- the vector-kernel regime uses the recovered ordinary-vector and LoRA
  encoders and AIR schedules in `mps_a14.py`, `mps_vector.py` and `mps_lora.py`;
- other M5 f16/bf16 products run `shaders/a18_gemm.h`, a kernel built to the structure of
  `gemm_a18` so that the 17 `MPSMatMulAutoTuningParametersA18` words apply to
  it unchanged. There is no tile remapping and no measured constant in the
  tile choice.

| parameter | in `gemm_a18` | in `a18_gemm.h` |
|---|---|---|
| `unrollK` | `tileKPerKSplit = 16 * unrollK` | the K chunk `KC`; loaded as 64/32/16-wide sub-chunks, the widths MPP accepts for a register-staged operand |
| `unrollM`, `unrollN` | 16x16 fragments per simdgroup | the per-simdgroup block `(16*unrollM) x (16*unrollN)`, one MPP `matmul2d` op per simdgroup |
| `simdM`, `simdN` | simdgroups side by side | same; threadgroup = `32 * simdM * simdN * localKSplits` threads; `localKSplits=1` for global partitions |
| `kSplits` (`f6 = f7 = 0`) | local K split, reduced through threadgroup memory | same: chunks round-robin over the K-split groups, reduction in waves of 16 KB |
| `f6`/`f7` (global K split, Ultra) | coherent float32 partials and an integer completion counter, or a separate reduction dispatch | both modes ported in `mps_a18_global.py` and `shaders/a18_global.h`; details in section 5.3 |
| `uberM`, `uberN`, `linearize`, `rasterMBias` | threadgroup index decode (magic divisions) | same decode with plain divisions: sub tile N-fastest inside an uber tile, uber tiles M-first when `rasterMBias` is set, 1-D grid when `linearize` is set, otherwise `(sub tiles, uber tiles, batch)` |
| `morton` | Z-order sub tiles | not ported; absent from the M1–M5 heuristics and all A18G/S/C rows; one A20P row for generation 24+ sets it |
| `barrierIterations`, `barrierInside`, `elasticBarrier` | simdgroup pacing | the source-selected hard/elastic mode and interval; no timing-derived occupancy suppression or environment override |
| `tgmemSize` | threadgroup allocation | sized from the tile |
| M/N edges | per-fragment masks | local partitions shift a straddling block back inside when possible; global partitions mask partial-buffer writes so output tiles remain disjoint |
| K tail | masked loads | bounds-checked views of at most one sub-chunk |

The multiply-accumulate is not one-to-one with Apple's kernel: it
is MPP's `matmul2d` on one simdgroup (the public route to the 16x16x16
simdgroup matrix unit; `gemm_a18` calls the private builtin directly), with
the A chunk loaded into registers before B streams through the multiply so
that a chunk's loads are issued together as in `gemm_a18`. Boundary loads
also differ as detailed in the table. Source-selected pacing now applies
regardless of the grid's threadgroups-per-core ratio.

### 5.1 Historical engine overrides (removed)

Earlier measurements found two regimes slower on this engine with Apple's
parameters than with the repo's older kernels (section 7.3): operand rows that do not start
on 16-byte boundaries (odd leading dimensions), where every per-simdgroup
load touches two cache lines, and thin N, where Apple's two simdgroups split
the 64 columns and leave too few simdgroups for the GPU. For those the
previous default policy ran the older kernels instead (`mpp_tensor` 64x64 or 16x64,
`sgpipe_gemm`), with the tiles the old autotuner converged on. Those measured
constants were removed with `_engine_override`. The table below
documents the old behavior so the historical measurements remain interpretable;
none of these rules selects the default plain matmul/bmm path now.

| regime | kernel |
|---|---|
| `lda`, `ldb` or the storage offset not a multiple of 8 elements, `min(M,N) > 512` | `mpp_tensor` 64x64, 2 simdgroups |
| same, `max(M,N) <= 512` with more than 4% padding of M to 32 rows | `mpp_tensor` 16x64, 2 simdgroups |
| same, smaller, otherwise | `mpp_tensor` 32x64, 2 simdgroups |
| `N` 64 or 128, `M >= 1024`, `K >= 1024`, `K % 128 == 0` | `sgpipe_gemm` (16,64,64,1,2) / (32,64,64,2,1) |
| `N == 256`, `M >= 2048`, `K >= 2048` | `mpp_tensor` 64x64, 2 simdgroups |
| `64 < M < 128`, `N >= 2048`, `K >= 2048` | `mpp_tensor` 32x128, 2 simdgroups |

`METALBLAS_TILE_POLICY=mps` and `mps-strict` now select the same recovered
rules. `legacy` retains the previous hand-written static picker for explicit
comparisons. Runtime autotuning is disabled; `METALBLAS_AUTOTUNE=1` no longer
enables it. `addmm` and `baddbmm` now use these recovered routes with the
linked MPSGraph bias epilogue. `addbmm` also uses the recovered separate
batch reduction. See the [epilogue evidence](mps_matmuls.md#linked-bias-epilogues)
and [batch-reduction rules](mps_matmuls.md#batch-reduction-for-addbmm).

### 5.2 GPUs the A18 execution backend does not cover

MPS only runs the A18 kernel on M5-class (Apple10) and newer GPUs
(generation 22 and up). Tracing the Metal dispatches of `torch.matmul` on
older Macs shows two more kernel families, neither of which logs A18
parameters (`tools/mps_probe.py` sees no `MatMul A18 kernel` line for any
shape):

| GPU | macOS | MPS matmul kernel | threadgroup |
|---|---|---|---|
| M2, 10 cores (generation 17) | 26.3 | `NDArrayMatrixMultiplyNNA14`, `NDArrayMatrixMultiplyNNA14_EdgeCase` for partial tiles | 128 threads, 64x64 tiles (2048³ on a 32x32 grid) |
| M3 Ultra, 80 cores (generation 19) | 26.5 | `mmul_kernel_a16_{bfloat,half,float}_*_float` | 128 threads, 32x32 tiles |
| M4, 10 cores (generation 20) | 26.6 | `mmul_kernel_a16_*` | 128 threads, 32x32 tiles |
| M5 Pro, 16 cores (generation 23) | 27.2 | `gemm_a18` | per section 3 |

The A18 behavior's constructor has a no-table branch for generations 20–21,
but on the M4 the behavior is not selected at all. The A14/a16 selection is
now recovered in `metalblas/mps_a14.py`, with binary provenance and trace
validation in [the M1–M5 inventory](mps_matmuls.md). The NNA14 and A16 ports
now serve default plain matmul/bmm on these GPUs, including ordinary-vector
and LoRA routing. They follow the corresponding 8x8 SIMD-group matrix
structure and real split K through a separate reduction. The old measuring
autotuner is enabled only in an isolated comparison snapshot. Hardware
coverage and performance parity are still incomplete; the inventory records
the current results and limitations.

### 5.3 Global K partitions

[`mps_a18_global.py`](../metalblas/mps_a18_global.py) implements both global
reduction modes from `EncodeArrayMultiply` at `0x190c70408` and the
`gemm_a18`, `gemmA18InitializeAtomicCounterToZero`, and
`gemmA18GlobalReduction` AIR modules in macOS 27.2 build 26B5086k. Their
offsets and hashes are in [`mps_dispatch_sources.json`](mps_dispatch_sources.json).

For `S=kSplits` with field 6 set, `localKSplits=1`. Each global partition
starts at `split * floor(K/S)`; the last receives the remainder. The encoder
records `ceil(floor(K/S)/(16*unrollK))` K iterations. Let
`TM=16*unrollM*simdM`, `TN=16*unrollN*simdN`, and
`P=uberM*uberN*ceil(ceil(M/TM)/uberM)*ceil(ceil(N/TN)/uberN)*batch`.
The linearized GEMM grid has `P*S` threadgroups of `32*simdM*simdN` threads,
including inactive edge tiles. Its partial buffer contains `S*batch*M*N`
float32 elements in `[split,batch,M,N]` order.

With field 7 set, a 32-thread initializer zeros `P` integer counters. GEMM
writes coherent partials, executes device/threadgroup barriers and device
sequentially consistent fences, and increments a per-output-tile counter
with relaxed compare-and-exchange. The last threadgroup sums all partitions
in ascending order, applies the linked epilogue, and converts to the output
type. With field 7 clear, a separate reduction uses 256-thread groups and
`ceil(batch*M*N/256)` groups, with the same sum and epilogue order. The port
bounds initializer stores to the allocated counters.

The [two explicit-parameter fixtures](../tests/fixtures/mps_a18) contain
64 captures per mode: f16/bf16, NN/NT/TN/TT, batch 1/3, and four shapes
covering 2/4/8 partitions, partial output tiles, and K remainders. All logged
keys and 17 parameters match the requested schedules. Offline checks match
the GEMM/helper grids, threadgroups, K partition fields, and scratch-buffer
sizes. Separate random-input comparisons match native MPS bitwise in all
256 plain/fused cases. GPU tests cover another 176 exact CPU comparisons,
including padded operands, bias broadcasting, and output views.

These are encoder and execution checks on 16-core M5 Pro with fixed
parameters from the recovered 80-core branch, with only field 7 varied to
exercise the other reduction mode. The harness sets the native iteration
to `-1`; it performs no search. They do not validate natural selection on
60+ core A18 hardware or establish performance parity. No timing-derived
threshold or table row was added. The separate implementation patch,
capture harness, logs, and validation sources are retained in
[`a18_global_2026-10-05.tar.gz`](../perf_benchs/mps_dispatch/a18_global_2026-10-05.tar.gz).

## 6. Reproducing

- `tools/mps_probe.py`: sweeps shapes through `torch.matmul` with MPS's
  parameter log enabled and checks the port; exits non-zero on mismatch.
- `tools/mbtrace.m`: a Metal shim (load with `ctypes.CDLL` before torch)
  that logs every pipeline state, function constant, bound buffer and
  dispatch as JSON lines, which is how the kernel names and grid sizes
  above were captured.
- The dylibs were extracted from the dyld shared cache with Xcode's
  `dsc_extractor.bundle`, disassembled with `objdump --macho` and
  decompiled with Ghidra 11.4.2; the tables were read from
  `__TEXT,__const` at the `_matmul*Table` symbols (152 bytes per entry).

## 7. Historical results on M5 Pro

These measurements predate the source-only default integration and pacing
correction. They describe the earlier implementation, including its measured
engine overrides. Current public-API comparisons are recorded in
[the M1–M5 inventory](mps_matmuls.md).

bf16 TFLOPS, M5 Pro (16 cores), macOS 27.2, torch 2.12; "MPS" is
`torch.matmul` through MPSGraph. Interleaved best-of-8 timings in one process.

### 7.1 Getting the kernel to Apple's speed with Apple's parameters

Apple's parameters for 4096³ (and most large aligned shapes) are a 32x64
threadgroup tile of two simdgroups, 64-wide K chunks, 2x2 uber tiles walked
M-first on a linear grid, and a barrier every 4 K iterations. Fed to the
previous `mpp_tensor` kernel they lost 5–50% (its cooperative MPP op owns the
K loop and the 2-D grid walks the hardware's order); fed to a kernel that
mirrors `gemm_a18` they reach MPS:

| kernel structure, 32x64/2 simdgroups, Apple's raster | 4096³ | 2048x2048x8192 |
|---|---|---|
| MPS (`gemm_a18`) | 25.5 | 24.7 |
| one MPP op per simdgroup, operands as device slices per 64-K chunk | 19.8 | 19.1 |
| one MPP op per threadgroup (`execution_simdgroups<2>`, MPP owns the K loop) | 20.9 | 20.2 |
| per simdgroup, A chunk register-staged, hard barrier every 4 iterations | 23.6 | 23.5 |
| same, no pacing | 25.5 | 25.0 |
| same, elastic pacing (arrival counters, lead <= 4) — shipped | 25.3 | 24.5 |

The pacing choice is shape dependent with MPP's loads: no pacing is best on
4096³ but drops 2048x14336x4096 to 21.5 and 4096x4096x11008 to 17.4, the hard
barrier recovers those (24.9, 23.3) at a 5% cost on squares, and the elastic
scheme gets both (25.6, 24.6). Apple's own kernel behaves the other way
round: with `MPS_MATMUL_*` overrides, its hard barrier is worth 1.4% over no
pacing and its elastic mode loses 3%.

The threadgroup order matters as much for Apple's kernel as for this one.
`gemm_a18` with its parameters overridden one at a time (4096³): default
25.6; `rasterMBias=0` (N-first) 19.5; no uber tiles 17.9; no uber tiles and
N-first 14.4; `linearize=0` 19.7; 4x4 uber tiles 25.5; 64x64 tile via
`unrollM=4` 22.0 or via `simdM=2` 24.3; a single simdgroup per threadgroup
20.9.

Edges: shifting a straddling block back inside instead of masking it took
257³ from 3.6 to 4.2, 333x444x555 from 10.0 to 12.1 and 511³ from 14.8 to
16.8 TFLOPS. Odd leading dimensions (1023³, 4097³) still cost both kernels
about 20% against the aligned shape: MPS 16.5 vs 20.5 at 1023³, this kernel
19.0 vs 24.3.

#### Pacing placement under Apple's parameters (2026-10-05)

The shipped kernel now follows Apple's recovered pacing (no measured elastic
exception), which leaves aligned large GEMMs at about 0.94–0.96x MPS on the
M5 Pro (e.g. 4096³ f16 0.94–0.95x, 2048x14336x4096 0.96x, 4096x4096x1024
0.98–0.99x; bitwise-equal outputs). In the 27.2 `gemm_a18` AIR, a nonzero
`barrierIterations` with `barrierInside=1` puts `air.wg.barrier(2, 5, 1)` after
the iteration's operand loads and before its multiply-accumulates; with
`barrierInside=0` the same test sits at the end of the iteration. MSL's
`threadgroup_barrier(mem_threadgroup)` emits the identical intrinsic. With
MPP operands, matching the placement does not help. At 4096³ f16, against
0.945x for the shipped top-of-iteration barrier:

| variant (same parameters and grid) | 4096³ | 2048x14336x4096 | 2048x2048x8192 |
|---|---:|---:|---:|
| barrier at the top of the iteration (shipped) | 0.940–0.948x | 0.961–0.962x | 0.947–0.953x |
| barrier after the A-chunk loads, B streamed by MPP | 0.896x | 0.916x | 0.876x |
| A and B register-staged (MPP requires K 16/32), barrier after loads | 0.645x | 0.700x | 0.632x |
| same staging, barrier at the top | 0.925x | 0.963x | 0.922x |
| execution-only barrier (`mem_none`), top | 0.942x | 0.962x | 0.947x |
| A chunk software-pipelined one iteration ahead | 0.602x | 0.594x | 0.618x |

None of these changes a selected parameter; they are implementation
experiments with Apple's `barrierIterations=4`, `barrierInside=1`,
`elasticBarrier=0` schedule. The remaining gap appears to be MPP's
cooperative load/MAC code generation, since the 16x16x16 matrix intrinsic
`gemm_a18` calls is not available to MSL directly.

### 7.2 `bench/bench_matmul.py` shapes, default policy: MPS, static, old autotuner

`static` was the default dispatcher (Apple's decision plus the engine
overrides of section 5.1, no measurement at run time); `autotuned` was
`METALBLAS_AUTOTUNE=1`, the previous measuring autotuner over the previous
kernels, timed in the same process and interleaved (best of 14 trials of
50 ms bursts, 1 s pause between shapes).

#### bf16

| group | M | N | K | torch TFLOPS | autotuned TFLOPS | static TFLOPS | static vs torch | static vs autotune |
|---|---|---|---|---|---|---|---|---|
| square | 128 | 128 | 128 | 0.24 | 0.89 | 0.88 | 3.71x | 0.99x |
| square | 256 | 256 | 256 | 1.72 | 4.72 | 4.77 | 2.77x | 1.01x |
| square | 512 | 512 | 512 | 10.44 | 17.60 | 17.83 | 1.71x | 1.01x |
| square | 1024 | 1024 | 1024 | 19.92 | 22.87 | 22.71 | 1.14x | 0.99x |
| square | 2048 | 2048 | 2048 | 23.60 | 23.35 | 23.60 | 1.00x | 1.01x |
| square | 4096 | 4096 | 4096 | 25.58 | 25.57 | 25.33 | 0.99x | 0.99x |
| tall | 4096 | 1024 | 1024 | 24.62 | 25.54 | 25.13 | 1.02x | 0.98x |
| tall | 1024 | 4096 | 1024 | 24.51 | 24.69 | 25.03 | 1.02x | 1.01x |
| tall | 8192 | 1024 | 1024 | 25.42 | 25.64 | 25.58 | 1.01x | 1.00x |
| tall | 2048 | 2048 | 8192 | 24.88 | 24.76 | 24.56 | 0.99x | 0.99x |
| attn | 4096 | 4096 | 64 | 16.00 | 17.65 | 17.66 | 1.10x | 1.00x |
| attn | 4096 | 4096 | 128 | 22.85 | 24.70 | 24.66 | 1.08x | 1.00x |
| attn | 4096 | 4096 | 1024 | 25.70 | 25.79 | 25.74 | 1.00x | 1.00x |
| gemv | 1 | 4096 | 4096 | 0.28 | 0.33 | 0.33 | 1.18x | 1.00x |
| gemv | 4096 | 1 | 4096 | 0.29 | 0.32 | 0.32 | 1.11x | 1.00x |
| gemv | 1 | 1024 | 1024 | 0.10 | 0.30 | 0.30 | 2.92x | 1.00x |
| gemv | 1 | 32000 | 4096 | 0.27 | 0.28 | 0.28 | 1.07x | 1.00x |
| llm | 2048 | 14336 | 4096 | 25.93 | 25.97 | 25.63 | 0.99x | 0.99x |
| llm | 4096 | 4096 | 11008 | 24.69 | 25.07 | 24.96 | 1.01x | 1.00x |
| llm | 4096 | 11008 | 4096 | 25.75 | 25.84 | 25.51 | 0.99x | 0.99x |
| thin_m | 96 | 4096 | 4096 | 18.58 | 19.39 | 19.35 | 1.04x | 1.00x |
| thin_m | 128 | 4096 | 4096 | 22.83 | 24.69 | 24.17 | 1.06x | 0.98x |
| thin_m | 192 | 4096 | 4096 | 23.92 | 22.88 | 24.85 | 1.04x | 1.09x |
| thin_m | 256 | 4096 | 4096 | 24.81 | 25.33 | 25.67 | 1.03x | 1.01x |
| thin_m | 128 | 8192 | 4096 | 23.18 | 23.08 | 23.44 | 1.01x | 1.02x |
| thin_n | 4096 | 128 | 4096 | 22.89 | 25.07 | 25.02 | 1.09x | 1.00x |
| thin_n | 4096 | 256 | 4096 | 24.05 | 24.84 | 24.79 | 1.03x | 1.00x |
| thin_n | 8192 | 128 | 4096 | 24.48 | 24.80 | 24.98 | 1.02x | 1.01x |
| thin_n | 4096 | 64 | 4096 | 14.93 | 18.08 | 18.05 | 1.21x | 1.00x |
| thin_n | 4096 | 32 | 4096 | 7.88 | 9.38 | 9.42 | 1.19x | 1.00x |
| thin_n | 4096 | 16 | 4096 | 4.24 | 4.74 | 4.75 | 1.12x | 1.00x |
| thin_n | 4096 | 8 | 4096 | 1.60 | 2.42 | 2.41 | 1.51x | 1.00x |
| odd | 257 | 257 | 257 | 1.49 | 3.52 | 3.44 | 2.31x | 0.98x |
| odd | 1023 | 1023 | 1023 | 16.73 | 22.00 | 22.04 | 1.32x | 1.00x |
| odd | 4097 | 4097 | 4097 | 21.90 | 22.12 | 22.26 | 1.02x | 1.01x |
| odd | 511 | 511 | 511 | 9.37 | 16.24 | 16.19 | 1.73x | 1.00x |
| odd | 333 | 444 | 555 | 6.01 | 12.36 | 12.42 | 2.07x | 1.00x |

Median static/autotune 0.999x (mean 1.001, min 0.98, max 1.09); median static/torch 1.07x (min 0.99, max 3.71).

#### float16

| group | M | N | K | torch TFLOPS | autotuned TFLOPS | static TFLOPS | static vs torch | static vs autotune |
|---|---|---|---|---|---|---|---|---|
| square | 128 | 128 | 128 | 0.22 | 0.64 | 0.81 | 3.68x | 1.28x |
| square | 256 | 256 | 256 | 1.77 | 4.71 | 4.77 | 2.70x | 1.01x |
| square | 512 | 512 | 512 | 9.82 | 17.75 | 17.99 | 1.83x | 1.01x |
| square | 1024 | 1024 | 1024 | 20.75 | 24.21 | 24.09 | 1.16x | 1.00x |
| square | 2048 | 2048 | 2048 | 24.96 | 25.53 | 25.20 | 1.01x | 0.99x |
| square | 4096 | 4096 | 4096 | 25.62 | 25.65 | 25.33 | 0.99x | 0.99x |
| tall | 4096 | 1024 | 1024 | 24.63 | 25.26 | 25.12 | 1.02x | 0.99x |
| tall | 1024 | 4096 | 1024 | 24.54 | 24.80 | 25.05 | 1.02x | 1.01x |
| tall | 8192 | 1024 | 1024 | 25.41 | 25.58 | 25.59 | 1.01x | 1.00x |
| tall | 2048 | 2048 | 8192 | 24.81 | 24.79 | 24.54 | 0.99x | 0.99x |
| attn | 4096 | 4096 | 64 | 15.74 | 17.40 | 17.40 | 1.11x | 1.00x |
| attn | 4096 | 4096 | 128 | 22.46 | 24.68 | 24.69 | 1.10x | 1.00x |
| attn | 4096 | 4096 | 1024 | 25.70 | 25.80 | 25.72 | 1.00x | 1.00x |
| gemv | 1 | 4096 | 4096 | 0.27 | 0.32 | 0.32 | 1.18x | 1.00x |
| gemv | 4096 | 1 | 4096 | 0.07 | 0.32 | 0.32 | 4.52x | 1.00x |
| gemv | 1 | 1024 | 1024 | 0.09 | 0.24 | 0.24 | 2.69x | 1.00x |
| gemv | 1 | 32000 | 4096 | 0.26 | 0.28 | 0.28 | 1.08x | 1.00x |
| llm | 2048 | 14336 | 4096 | 25.85 | 25.76 | 25.53 | 0.99x | 0.99x |
| llm | 4096 | 4096 | 11008 | 24.54 | 24.83 | 24.66 | 1.01x | 0.99x |
| llm | 4096 | 11008 | 4096 | 24.58 | 23.98 | 23.75 | 0.97x | 0.99x |
| thin_m | 96 | 4096 | 4096 | 17.65 | 18.54 | 18.45 | 1.05x | 1.00x |
| thin_m | 128 | 4096 | 4096 | 21.36 | 23.17 | 22.81 | 1.07x | 0.98x |
| thin_m | 192 | 4096 | 4096 | 22.67 | 22.33 | 24.40 | 1.08x | 1.09x |
| thin_m | 256 | 4096 | 4096 | 23.06 | 23.93 | 24.73 | 1.07x | 1.03x |
| thin_m | 128 | 8192 | 4096 | 22.64 | 22.50 | 23.24 | 1.03x | 1.03x |
| thin_n | 4096 | 128 | 4096 | 22.51 | 24.56 | 24.84 | 1.10x | 1.01x |
| thin_n | 4096 | 256 | 4096 | 22.95 | 24.31 | 23.99 | 1.05x | 0.99x |
| thin_n | 8192 | 128 | 4096 | 23.00 | 24.60 | 24.59 | 1.07x | 1.00x |
| thin_n | 4096 | 64 | 4096 | 14.83 | 17.99 | 18.01 | 1.21x | 1.00x |
| thin_n | 4096 | 32 | 4096 | 7.48 | 8.97 | 9.04 | 1.21x | 1.01x |
| thin_n | 4096 | 16 | 4096 | 4.37 | 4.86 | 4.87 | 1.11x | 1.00x |
| thin_n | 4096 | 8 | 4096 | 1.60 | 2.34 | 2.32 | 1.45x | 0.99x |
| odd | 257 | 257 | 257 | 1.61 | 4.03 | 4.17 | 2.60x | 1.04x |
| odd | 1023 | 1023 | 1023 | 16.69 | 21.90 | 21.77 | 1.30x | 0.99x |
| odd | 4097 | 4097 | 4097 | 21.72 | 21.89 | 22.30 | 1.03x | 1.02x |
| odd | 511 | 511 | 511 | 8.91 | 16.18 | 16.17 | 1.82x | 1.00x |
| odd | 333 | 444 | 555 | 5.96 | 12.42 | 12.43 | 2.09x | 1.00x |

Median static/autotune 1.000x (mean 1.012, min 0.98, max 1.28); median static/torch 1.08x (min 0.97, max 4.52).


### 7.3 Apple's decision alone (`METALBLAS_TILE_POLICY=mps-strict`)

The same measurement with the overrides disabled, i.e. every GEMM-regime
shape on `a18_gemm` with Apple's parameters (best of 8 trials of 30 ms bursts):

#### bf16

| group | M | N | K | torch TFLOPS | autotuned TFLOPS | static TFLOPS | static vs torch | static vs autotune |
|---|---|---|---|---|---|---|---|---|
| square | 128 | 128 | 128 | 0.24 | 0.86 | 0.86 | 3.61x | 1.00x |
| square | 256 | 256 | 256 | 1.67 | 4.62 | 4.56 | 2.73x | 0.99x |
| square | 512 | 512 | 512 | 10.28 | 17.34 | 17.64 | 1.72x | 1.02x |
| square | 1024 | 1024 | 1024 | 20.46 | 24.12 | 23.97 | 1.17x | 0.99x |
| square | 2048 | 2048 | 2048 | 24.99 | 24.74 | 25.10 | 1.00x | 1.01x |
| square | 4096 | 4096 | 4096 | 25.54 | 25.47 | 25.26 | 0.99x | 0.99x |
| tall | 4096 | 1024 | 1024 | 24.50 | 25.42 | 25.02 | 1.02x | 0.98x |
| tall | 1024 | 4096 | 1024 | 24.43 | 25.31 | 24.96 | 1.02x | 0.99x |
| tall | 8192 | 1024 | 1024 | 25.28 | 25.34 | 25.42 | 1.01x | 1.00x |
| tall | 2048 | 2048 | 8192 | 24.68 | 24.54 | 24.49 | 0.99x | 1.00x |
| attn | 4096 | 4096 | 64 | 15.73 | 17.80 | 17.68 | 1.12x | 0.99x |
| attn | 4096 | 4096 | 128 | 22.50 | 24.56 | 24.55 | 1.09x | 1.00x |
| attn | 4096 | 4096 | 1024 | 25.55 | 25.65 | 25.57 | 1.00x | 1.00x |
| gemv | 1 | 4096 | 4096 | 0.27 | 0.31 | 0.31 | 1.18x | 1.00x |
| gemv | 4096 | 1 | 4096 | 0.29 | 0.32 | 0.32 | 1.10x | 1.00x |
| gemv | 1 | 1024 | 1024 | 0.12 | 0.34 | 0.34 | 2.92x | 1.00x |
| gemv | 1 | 32000 | 4096 | 0.26 | 0.28 | 0.28 | 1.07x | 1.00x |
| llm | 2048 | 14336 | 4096 | 25.89 | 25.90 | 25.53 | 0.99x | 0.99x |
| llm | 4096 | 4096 | 11008 | 24.51 | 24.85 | 24.69 | 1.01x | 0.99x |
| llm | 4096 | 11008 | 4096 | 25.56 | 25.71 | 25.45 | 1.00x | 0.99x |
| thin_m | 96 | 4096 | 4096 | 18.21 | 19.16 | 18.68 | 1.03x | 0.97x |
| thin_m | 128 | 4096 | 4096 | 22.55 | 24.49 | 23.93 | 1.06x | 0.98x |
| thin_m | 192 | 4096 | 4096 | 24.12 | 22.74 | 24.74 | 1.03x | 1.09x |
| thin_m | 256 | 4096 | 4096 | 24.89 | 25.20 | 25.54 | 1.03x | 1.01x |
| thin_m | 128 | 8192 | 4096 | 23.11 | 23.00 | 23.37 | 1.01x | 1.02x |
| thin_n | 4096 | 128 | 4096 | 22.68 | 24.97 | 24.26 | 1.07x | 0.97x |
| thin_n | 4096 | 256 | 4096 | 24.28 | 25.07 | 24.71 | 1.02x | 0.99x |
| thin_n | 8192 | 128 | 4096 | 24.37 | 24.93 | 24.81 | 1.02x | 1.00x |
| thin_n | 4096 | 64 | 4096 | 14.42 | 17.37 | 16.56 | 1.15x | 0.95x |
| thin_n | 4096 | 32 | 4096 | 7.77 | 9.26 | 9.30 | 1.20x | 1.00x |
| thin_n | 4096 | 16 | 4096 | 4.22 | 4.69 | 4.76 | 1.13x | 1.02x |
| thin_n | 4096 | 8 | 4096 | 1.59 | 2.42 | 2.41 | 1.52x | 1.00x |
| odd | 257 | 257 | 257 | 1.60 | 4.05 | 3.76 | 2.35x | 0.93x |
| odd | 1023 | 1023 | 1023 | 16.59 | 21.98 | 18.93 | 1.14x | 0.86x |
| odd | 4097 | 4097 | 4097 | 21.93 | 23.16 | 20.40 | 0.93x | 0.88x |
| odd | 511 | 511 | 511 | 8.62 | 15.92 | 15.87 | 1.84x | 1.00x |
| odd | 333 | 444 | 555 | 6.32 | 12.07 | 11.13 | 1.76x | 0.92x |

Median static/autotune 0.995x (mean 0.987, min 0.86, max 1.09); median static/torch 1.07x (min 0.93, max 3.61).

#### float16

| group | M | N | K | torch TFLOPS | autotuned TFLOPS | static TFLOPS | static vs torch | static vs autotune |
|---|---|---|---|---|---|---|---|---|
| square | 128 | 128 | 128 | 0.25 | 0.82 | 0.86 | 3.47x | 1.04x |
| square | 256 | 256 | 256 | 1.63 | 4.47 | 4.59 | 2.81x | 1.03x |
| square | 512 | 512 | 512 | 10.06 | 17.23 | 17.47 | 1.74x | 1.01x |
| square | 1024 | 1024 | 1024 | 20.26 | 24.10 | 23.97 | 1.18x | 0.99x |
| square | 2048 | 2048 | 2048 | 24.86 | 25.30 | 25.11 | 1.01x | 0.99x |
| square | 4096 | 4096 | 4096 | 24.71 | 24.33 | 24.37 | 0.99x | 1.00x |
| tall | 4096 | 1024 | 1024 | 23.57 | 23.98 | 23.97 | 1.02x | 1.00x |
| tall | 1024 | 4096 | 1024 | 24.17 | 24.24 | 23.81 | 0.99x | 0.98x |
| tall | 8192 | 1024 | 1024 | 24.81 | 24.49 | 24.76 | 1.00x | 1.01x |
| tall | 2048 | 2048 | 8192 | 23.95 | 23.10 | 23.19 | 0.97x | 1.00x |
| attn | 4096 | 4096 | 64 | 15.55 | 17.29 | 17.23 | 1.11x | 1.00x |
| attn | 4096 | 4096 | 128 | 21.36 | 22.46 | 23.07 | 1.08x | 1.03x |
| attn | 4096 | 4096 | 1024 | 23.64 | 23.08 | 23.37 | 0.99x | 1.01x |
| gemv | 1 | 4096 | 4096 | 0.26 | 0.31 | 0.31 | 1.19x | 1.00x |
| gemv | 4096 | 1 | 4096 | 0.07 | 0.31 | 0.31 | 4.47x | 1.00x |
| gemv | 1 | 1024 | 1024 | 0.11 | 0.33 | 0.33 | 2.97x | 1.00x |
| gemv | 1 | 32000 | 4096 | 0.26 | 0.28 | 0.28 | 1.07x | 1.00x |
| llm | 2048 | 14336 | 4096 | 23.33 | 23.41 | 22.89 | 0.98x | 0.98x |
| llm | 4096 | 4096 | 11008 | 22.54 | 22.42 | 22.12 | 0.98x | 0.99x |
| llm | 4096 | 11008 | 4096 | 22.33 | 22.37 | 21.88 | 0.98x | 0.98x |
| thin_m | 96 | 4096 | 4096 | 17.28 | 18.79 | 17.52 | 1.01x | 0.93x |
| thin_m | 128 | 4096 | 4096 | 22.18 | 23.03 | 22.94 | 1.03x | 1.00x |
| thin_m | 192 | 4096 | 4096 | 21.70 | 21.15 | 22.50 | 1.04x | 1.06x |
| thin_m | 256 | 4096 | 4096 | 23.24 | 22.53 | 23.18 | 1.00x | 1.03x |
| thin_m | 128 | 8192 | 4096 | 21.92 | 21.62 | 22.26 | 1.02x | 1.03x |
| thin_n | 4096 | 128 | 4096 | 21.80 | 23.01 | 22.95 | 1.05x | 1.00x |
| thin_n | 4096 | 256 | 4096 | 22.76 | 22.36 | 22.87 | 1.00x | 1.02x |
| thin_n | 8192 | 128 | 4096 | 22.30 | 22.68 | 22.60 | 1.01x | 1.00x |
| thin_n | 4096 | 64 | 4096 | 14.67 | 17.94 | 17.14 | 1.17x | 0.96x |
| thin_n | 4096 | 32 | 4096 | 7.99 | 9.18 | 9.28 | 1.16x | 1.01x |
| thin_n | 4096 | 16 | 4096 | 4.28 | 4.81 | 4.85 | 1.13x | 1.01x |
| thin_n | 4096 | 8 | 4096 | 1.58 | 2.38 | 2.38 | 1.50x | 1.00x |
| odd | 257 | 257 | 257 | 1.58 | 3.69 | 3.73 | 2.36x | 1.01x |
| odd | 1023 | 1023 | 1023 | 16.77 | 21.65 | 18.95 | 1.13x | 0.88x |
| odd | 4097 | 4097 | 4097 | 20.75 | 20.95 | 19.36 | 0.93x | 0.92x |
| odd | 511 | 511 | 511 | 8.78 | 15.89 | 15.90 | 1.81x | 1.00x |
| odd | 333 | 444 | 555 | 5.93 | 11.99 | 11.09 | 1.87x | 0.93x |

Median static/autotune 1.000x (mean 0.995, min 0.88, max 1.06); median static/torch 1.05x (min 0.93, max 4.47).

### 7.4 Why those two regimes need the older kernels

With Apple's decision alone (7.3) every shape runs within noise of MPS or
faster (0.93–3.6x) and within 1–2% of the old autotuner, except for two
families where Apple's parameters are a worse fit for this kernel than the
autotuner's measured pick:

- odd leading dimensions (1023³ 0.86x, 4097³ 0.88x, 333x444x555 0.92x,
  257³ 0.93x bf16): isolating them with strided views of aligned matrices
  shows the whole cost is row alignment (an odd `ldb` alone costs 11%, an
  odd `lda` 2%; the M/N edge and the K tail cost nothing). Unaligned rows
  touch two cache lines per row segment and only a wider threadgroup tile
  amortizes that. MPS pays the same price (1023³: 16.6 vs 20.5 TFLOPS at
  1024³) and is still 14% slower than this kernel there.
- thin N (4096x64x4096 0.95x, 4096x128 0.97x): Apple stops shrinking the
  tile once the tile count covers the cores and splits the 64 columns between
  the two simdgroups; `sgpipe_gemm` splits M instead and runs twice as many
  simdgroups. MPS is 15% slower than this kernel there.

Tried within Apple's parameters, without success: register-staging B instead
of A or both operands (MPP allows both only for 16- or 32-wide blocks; 1023³
19.3 and 19.1 vs 19.0), 16- and 32-wide sub-chunks, K-loop unrolling and a
compile-time iteration count (all within noise), splitting the same 32x64
tile along M so the two simdgroups share B (1023³ 17.3, 4096³ 20.2), and
mapping a whole uber tile onto one threadgroup so its sub tiles share L1
(1023³ +6%, 2048x2048x8192 +2%, but 4096x128 -6%, 256x4096 -4%,
4096x4096x128 -4%). The overrides of section 5.1 are what remains.

The 1–2% differences left in 7.2 are measurement noise: with the same kernel
on both sides (257³, 4096x4096x64/128, 4096x64 with the override) the
interleaved ratio still moves between 0.988x and 1.011x from run to run, and
a shape measured 0.972x in one run (2048x14336x4096) measured 1.05x in the
next.
