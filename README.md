# metalBLAS

Hand-tuned Metal Shading Language matmul kernels for Apple Silicon, callable
from PyTorch on `mps`.  Built on top of `torch.mps.compile_shader` (PyTorch
2.12+) – every kernel is a templated heredoc string that gets JIT-compiled by
the Metal driver.

## Highlights

* **Three complementary kernel families**
  - `m5_tensor` – uses `mpp::tensor_ops::matmul2d` with `tensor_inline` device
    views and `execution_simdgroups<N>`.  Each TG hands a (BM × BN) output
    tile and the full K dim to one `op.run` call — the MPP runtime handles
    loads internally.  Interior tiles use **static-extent slices**
    (`slice<BN,BM>(…)`) so `matmul2d` skips per-tile edge predication (the big
    Session-6 win); only the ≤1 partial tile per column falls back to a dynamic
    slice + validity mask.  This is the primary backend for **every dtype and
    every size**: bf16/fp16 (matches or beats torch), fp32 (2–3× via the M5
    tensor unit at TF32-relaxed precision), and small / non-divisible shapes (the
    kernel clips partial edge tiles natively, so it needs no tile alignment).
  - `m5_gemm` – manual threadgroup-tiled GEMM with `16×32×16` cooperative-tensor
    fragments.  Now only a fallback for sub-64 dims and transposed inputs.
  - `simd_gemm` – portable tiled GEMM using `simdgroup_matrix<T, 8, 8>` MMA,
    works on anything with Metal 3 simdgroup matrix intrinsics.  Also the
    strict-fp32 path if you override the backend.
* **`gemv_nt` / `gemv_t`** – bandwidth-bound rank-1 fast paths with coalesced,
  **cache-line-wide** loads.  `gemv_t` (the `y = x @ B` case) splits the K
  dimension across simdgroups within a threadgroup and reduces in threadgroup
  memory, so it fills all 16 cores even with few output columns; and each lane
  reads a `VEC`-wide vector of B's columns so a 32-lane warp's coalesced read
  spans a full 128-B cache line (`VEC=2` bf16) or two (`VEC=4`) — closing the
  mid-N K≥4096 band that used to sit at 0.87–0.96× (now 1.01–1.15×).  fp32 keeps
  `VEC=1` (4-byte loads already fill a line).
* **Auto-dispatch + runtime autotuner.**  Dispatch picks the backend from shape
  and dtype: almost everything (≥ 64³, untransposed, packed) goes to `m5_tensor`;
  GEMV (M=1 / N=1) gets the bandwidth kernels.  For the *tile*, a dtype/size-aware
  heuristic emits a short **candidate list**, and on the first call for a new
  bf16/fp16 shape an **autotuner probes those tiles on the real operands and
  caches the winner** — the only robust way to track the size/aspect/K crossover
  where the thin-BM `(48,128,4)` tile overtakes the `(64,64,2)` default (it does
  so by +2% at 2048³ up to +4% at 8192³, and owns the deep-K / high-aspect family
  too).  The heuristic's pick is always candidate 0 and a 3% margin guards it, so
  the autotuner is provably ≥ the old static gates and can't overfit; confident
  regimes and all of fp32 return a single candidate and never probe.  The probe is
  one-time per shape (amortized in ~16 calls); disable with `METALBLAS_AUTOTUNE=0`.
  Both GEMV and contiguous GEMM then take a **lean fast path** that memoizes the
  launch by shape and recycles output buffers from a refcount-gated pool — at
  these sizes `torch.matmul` is itself CPU-bound (~10–14 µs), so this per-call
  overhead is what decides the small/medium shapes.  Big-N bf16/fp16 GEMV (lm_head)
  is routed through a padded `m5_tensor`; sub-64 dims, transposed, and **non-packed**
  (`ld>K`) inputs fall back to `m5_gemm` (which honors arbitrary strides).

## Benchmarks (M5 Pro, 15 GPU cores, macOS 26.4.1)

Speedups against `torch.matmul` on MPS.  **Numbers below are isolated
best-of-6 timings** — the canonical `bench_matmul.py` runs all groups
back-to-back and reads tiny (<20 µs) shapes 2–4× slow from thermal/scheduler
noise, so the per-op truth comes from `bench/sweep.py`.

### bfloat16 / float16 — the inference / training dtypes

Large / LLM / deep-K GEMM is now at **parity-or-better across the board**.
Session 7's **runtime autotuner** closed the last tail: instead of a static
`(M,N,K)→tile` gate (which kept regressing neighbours and silently lost 8192³ at
0.95×), the heuristic emits a candidate list and the autotuner probes it on the
real operands the first time it sees a shape, caching the winner.  This tracks the
size/aspect/K crossover where the thin-BM `(48,128,4)` tile overtakes the
`(64,64,2)` default — fixing `8192×1024×8192` 0.93→1.00×, `2048×768×16384`
0.92→1.00×, `640³` 0.95→1.00×, and the big squares `6144³`/`8192³` 0.95–0.98→1.00×
— while a 3% margin keeps `(64,64,2)` wherever its edge is within noise.  Only two
shapes remain below parity, both genuine ceilings within ~3%: **512³** (a Python
*dispatch* tail — the kernel floor is ~parity) and **512²×16384** (compute-bound
vs Apple's closed GEMM; no tile or split-K beats it).  Numbers are isolated
best-of-N; small shapes confirmed one-per-process to avoid thermal contamination.

```
   shape              bf16     note  (bf16, Session 7)
128³                 2.8×
256³                 1.7×
511³                 1.00×
512³                 0.94×    dispatch ceiling — kernel floor ~parity, ~0.5µs Python tail
640³                 1.00×    was 0.95× — autotuner picks (16,64,2)
1023³                1.18×
1024³                1.02×
2048³                0.98×    within ±3% noise band — autotuner keeps (64,64,2)
4096³                1.00×
4097³                1.19×
6144³                1.02×    was 0.98× — autotuner picks (48,128,4)
8192³                1.00×    was 0.95× (silent loser past the old table)
8192×1024×8192       1.00×    was 0.93× — high-aspect, (48,128,4)
512×512×16384        0.97×    compute ceiling (K=32·max) — no tile/split-K beats it
2048×768×16384       1.00×    was 0.92× — tall deep-K, (64,128,4)
2048×768×4096        1.04×    tall deep-K (low K) — autotuner keeps (64,64,2)
2048×2048×8192       1.00×    deep-K square, (48,128,4)
2048×14336×4096      0.99×    llm gate_proj
4096×4096×11008      1.02×    llm down_proj — (48,128,4)
4096×11008×4096      1.00×    llm up_proj
```

GEMV (M=1), bf16 — steady-state (warm, repeated-shape).  The mid-N K≥4096 band
that used to sit at 0.87–0.96× is now a win across the board, via vectorized
cache-line-wide column loads + a memoized launch / output-pool fast path:

```
   shape (1×N×K)       bf16     note
1×256×256            3.0×     launch-bound (torch CPU-capped ~11µs)
1×512×512            2.8×
1×1024×1024          2.0×     was 1.69× (VEC=2, NW=16)
1×512×1024           1.9×
1×768×2048           1.5×
1×512×4096           1.10×    was 0.93× (VEC=1, full-occupancy)
1×768×4096           1.10×    was 0.94× (VEC=2)
1×1024×4096          1.02×    was 0.95× (VEC=2, NW=16)
1×1280×4096          1.08×    was 0.93× (VEC=4)
1×1536×4096          1.03×    was 0.96× (VEC=4)
1×2048×4096          1.05×    was 1.10×
1×3072×4096          1.01×    was 0.88× (VEC=4, NW=8)
1×2048×1024          1.15×    was 0.90× (K<2048 ⇒ VEC=2 for occupancy)
1×4096×4096          1.02×    padded m5_tensor (BM=16)
1×32000×4096         1.02×    padded m5_tensor (lm_head)
```

### float32 — biggest change this round

fp32 now runs through `m5_tensor` (M5 tensor unit, TF32-relaxed precision —
same as the old manual path, validated bit-for-bit) instead of the manual
`m5_gemm`.  It beats torch's strict-fp32 MPS path by **2–3× across the board**,
and the small-shape losses are gone:

| Group  | Median (was → now) | Best  |
| ------ | ------------------ | ----- |
| square | 1.51× → **2.54×**  | 3.38× (256³) |
| tall   | 1.79× → **2.49×**  | 2.64× |
| attn   | 1.60× → **2.32×**  | 2.67× |
| llm    | 1.63× → **2.07×**  | 2.12× |
| odd    | 1.31× → **1.76×**  | 2.18× |
| gemv   | 1.04× → **1.5×**   | 1.92× (1×512×4096) |

```
   shape              torch              metalBLAS         speedup
256³                 0.029 ms  1.2 TF | 0.009 ms  3.9 TF    3.38×   (was 0.46×)
512³                 0.049 ms  5.3 TF | 0.017 ms 15.1 TF    2.84×
1024³                0.359 ms  6.0 TF | 0.135 ms 16.0 TF    2.66×
4096³               22.86 ms   6.0 TF | 11.01 ms 12.4 TF    2.08×
2048×14336×4096     40.5 ms    5.9 TF | 19.1 ms  12.6 TF    2.12×
4096×4096×11008     62.7 ms    5.9 TF | 32.1 ms  11.5 TF    1.95×
```

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

// STATIC-EXTENT slices for interior tiles (Session 6's main win).  The templated
// slice<E0,E1>(...) (Apple's header calls it static_slice, but that name doesn't
// compile — slice<…> is the real method) gives matmul2d the tile's exact BM×BN
// extents at compile time, so it drops the per-tile edge/predication logic the
// dynamic slice() emits on EVERY tile.  When MN_ALIGNED every tile is interior;
// otherwise an `inside` test sends the ≤1 partial tile per column to the dynamic
// path below.  Strict win — 1024³ 0.93→1.02, 4097³ 0.87→1.12.
auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
auto mB = tB.slice<BN, dynamic_extent>(n_off, 0);
auto mC = tC.slice<BN, BM>(n_off, m_off);

// Compute in fp32 (matmul2d accumulates the K-reduction in fp32 internally
// regardless of OUT_T), convert to OUT_T for store.  Interior tiles need no
// is_valid_element check — every cT slot maps to a valid output position.
auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
op.run(mA, mB, cT_f32);

auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
    cT_out[i] = (OUT_T)cT_f32[i];
cT_out.store(mC);
// (edge tiles: the dynamic tA.slice(0,m_off) + per-element is_valid_element mask)
```

The dispatch grid is plain `(tiles_n, tiles_m, 1)`.  No external swizzle:
MPP has its own internal scheduling and an extra swizzle slows it down by
~ 15-20 %.

### Tile picker by regime (M5 Pro)

The picker chooses `(BM, BN, NSG)` from isolated sweep results.  The governing
principle: **fill the 16 cores with enough tiles, then prefer the most efficient
tile that still does** — "small" must be judged by tile COUNT and K, not just
max(M,N).  Checked top-to-bottom:

> **Note (Session 7):** for bf16/fp16 this table now produces the *primary* tile
> (candidate 0) of a short list; the runtime autotuner probes the list on first
> sight of a shape and caches the winner (see the auto-dispatch highlight above).
> So the deep-K / high-aspect / large-square rows below — exactly the ambiguous
> ones the static gates kept mis-calling — are resolved by measurement at run
> time; a 3% margin keeps this primary wherever a candidate's edge is within
> noise.  fp32 and the confident small rows still use the table value directly.

| Regime                                  | Tile           | Why                                          |
| --------------------------------------- | -------------- | -------------------------------------------- |
| M=1 / N=1 (padded GEMV)                 | (16, 128, 4)   | thin BM: a 64-row tile wastes 63 rows of MPP work on a 1-row problem (cratered N=4096 to 0.70×) |
| Tiny K (≤ 256) over large M, N          | (32, 128, 4)   | short K-loop → more tiles amortize per-TG cost |
| max(M, N) ≤ 256                         | (32, 32, 4)    | many tiles to fill the cores                 |
| max ≤ 1024 **and** (`ceil(M/64)·ceil(N/64) < 120` **or** K ≤ max) | (32, 64, 2) | thin BM when (64,64,2) makes too few tiles to fill the cores (512²×8192) or K is cube-ish (896³→1.00, 960³→1.06) |
| Deep-K (K ≥ 2·max) & max ≥ 1792, **lp**, tall (N<max≤2048) | (64, 64, 2) | wide-BN tile is backwards for a moderate M>N deep-K shape (2048×768×4096 0.93→1.03) |
| Deep-K (K ≥ 2·max) & max ≥ 1792, **lp** | (48, 128, 4)   | thin BM, wide BN: many finely-grained tiles load-balance the long K-loop (4096²×11008 0.96→1.02, 768²×8192 →1.09) |
| Deep-K (K ≥ 2·max) & max ≥ 1792, **fp32**| (64, 128, 4)  | fp32 already 2.2×; keeps the wide-BN tile     |
| max ≥ 4096, non-64-aligned, **bf16**    | (128, 128, 8)  | edge-waste amortization (bf16 only)          |
| Large default, **fp32** & divisible     | (64, 128, 4)   | 4-byte loads → arithmetic intensity → wide BN |
| Everything else (the big default)       | (64, 64, 2)    | broad winner, incl. non-divisible large (2047³ 1.36×, 4097³ 1.20×) |

The tiny / small-medium buckets are checked *before* deep-K on purpose: a
small-MN deep-K shape (e.g. 256²×4096) wants the tiny tile (0.99×), not the
wide-BN one (which gave 0.43×).  The deep-K tile is dtype-specific: bf16/fp16
take the thin `(48,128,4)` (a 64-row tile makes too few/too-heavy tiles for a
large deep-K output — `(128,128,8)` craters 4096²×11008 to 0.82×), fp32 keeps
`(64,128,4)`.  `(128,128,8)` otherwise helps only bf16, and fp32 large prefers
the wider `(64,128,4)`.

The dispatcher also routes wide bf16/fp16 GEMV (M=1, N ≥ 4096) through the
m5_tensor kernel with this BM=16 tile — Metal's OOB-zero-reads make the
over-fetched rows free, and the runtime's bandwidth handling beats the dedicated
GEMV kernel here (1.02–1.09×).  fp32 stays on `gemv_t` at every N (m5_tensor
loses for fp32 above N=4096 — 4-byte loads already saturate).

## Repo layout

```
metalblas/
  __init__.py       exposes `matmul`
  kernels.py        Metal shader source + JIT cache
  dispatch.py       Python dispatch, tile picker
bench/
  bench_matmul.py   benchmark vs torch.matmul on MPS (canonical groups)
  sweep.py          isolated best-of-N GEMM tile sweeper (de-noised per-op timing)
  gemv_probe.py     GEMV bandwidth probe (achieved GB/s vs torch + pure-read ceiling)
  gemv_nwarps_proto.py  GEMV NWARPS (within-TG K-split) sweeper
tests/
  test_basic.py     correctness tests vs the math reference
```

## Running

```bash
python tests/test_basic.py
python bench/bench_matmul.py --dtype bf16
python bench/bench_matmul.py --dtype fp32 --group llm
python bench/bench_matmul.py                  # full sweep across dtypes/groups

# Tune / verify a specific shape against torch (reliable for tiny shapes):
python bench/sweep.py --dtype fp32 --shapes "512x512x512;257x257x257" --reps 6
python bench/sweep.py --dtype bf16 --shapes "2048x2048x8192" \
    --tiles "64,128,4;128,128,8;64,64,2"     # compare specific (BM,BN,NSG) tiles

# GEMV (M=1): achieved bandwidth vs torch, and NWARPS K-split tuning:
python bench/gemv_probe.py --dtype bf16 --shapes "1x512x4096;1x1024x1024" --reps 12
python bench/gemv_nwarps_proto.py --dtype bf16 --shapes "1x1024x4096" --nwarps "8,16,32"
```
