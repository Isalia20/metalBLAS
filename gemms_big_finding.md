# matmul2d: own the loads, hand registers to the matrix unit

Note written by Claude (2026-11-01) from the conv2d investigation, for the future GEMM migration.
Source of the numbers: `agent_space/conv2d_bench/opus_attempt.md` (the kernel that closed the
bf16 3x3 gap against MPSGraph) and `agent_space/conv2d_bench/sd_losers_analysis.md` (the dead ends
before it). All measurements are M5 Pro, macOS 27.2, bf16 inputs, fp32 accumulation.

## The finding

MPP's `matmul2d` has two kinds of input operand, and they lower to completely different code:

| operand kind | what `op.run` compiles to | loads |
|---|---|---|
| device / threadgroup tensor (`tensor<...>` built from a pointer + extents) | `__loweringlib.internal.626` / `.630`: matrix instructions plus the op's own load code | `llvm.agx3.load.with.emask.global` v4i16 / v2i16: 4-8 bytes per lane, an `agx3.edgecheck` per load, `air.get_stride/extent_global_tensor` per entry |
| cooperative tensor (`op.get_left/right_input_cooperative_tensor<...>()`, register resident) | `__loweringlib.internal.634`: nothing but `air.simdgroup_matrix_16x16x16_multiply_accumulate` (8 bf16 sites + an 8x8 fallback), no loads, no stride queries | none: the kernel wrote the operands itself |

So the 16x16x16 matrix instruction is reachable from public Metal source with loads written by hand.
Fill both cooperative operands from your own loads, then `op.run(a, b, acc)`. The accumulator was
always cooperative, so the output side does not change. (`__loweringlib` members: extract with
`agent_space/conv2d_bench/air/metallib2.py` from
`/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Resources/libTensorOps.rtlib`,
an `ar` of bitcode; disassemble with `clang -S -emit-llvm -x ir`.)

Why it matters: on a compute-bound tile the tensor-operand lowering's load path is the whole gap.
Measured on the Stable-Diffusion 3x3 shapes (C128 1024px, C128 512px to 256, C96 512px to 256, C64 512px):

| | TFLOPS | vs MPSGraph |
|---|---:|---:|
| register-only `op.run` loop, no loads (the matrix-unit ceiling) | 26.1-26.2 | |
| MPSGraph's own `conv2d_a20` implicit GEMM | 25.2-26.0 | 1.00x |
| branch fixed 3x3 tile, device-tensor operands | 24.5-25.2 | 0.95-1.00x |
| same kernel structure, cooperative operands, loads deleted (ablation) | 26.2-26.4 | above 1.0x everywhere |
| final `conv2d_cooperative` (own 16-byte loads) | 26.1-26.3 | 1.007-1.070x |

The first-cut cooperative kernel with naive hand loads (8-byte weight loads, element-wise input loads)
was 1.35-1.45x SLOWER than the device-tensor tile. Owning the loads only pays once the loads are good.

## Constraints on cooperative inputs (MPPTensorOpsMatMul2dImpl.h, SDK 26.5, around line 4204)

- Both inputs cooperative requires `execution_simdgroup` (one simdgroup, 32 lanes). No
  `execution_simdgroups<N>` cooperation, no threadgroup-level tiles from the op.
- K must be static (no `dynamic_extent`).
- M, N, K each 16 or 32, at least one of them 32. So the largest run is 32x32x32; the kernel composes
  bigger tiles from runs itself.
- The three element types declared on the cooperative tensors must match the types passed to `run`
  (`"Input types must match cooperative tensor types"`). Mixed-type products need the accumulator
  operand types chosen with `conditional_t` on both sides.
- The register budget is yours: cooperative inputs live next to the accumulator. One 32x32 float
  accumulator per simdgroup (32 registers) plus 16-deep operands (8 registers each) was the sweet spot;
  a second 32x32 accumulator per simdgroup lost 20%, a 2x2 accumulator grid ran at 25.5 TFLOPS, 4x2 and
  larger grids collapsed to 3.5-5.9 (spills), explicit double buffering of the operands spilled and ran
  at 8-9 TFLOPS.

## The lane layout (probed, not documented)

Left, right and destination cooperative tensors of `matmul2d<desc(32,32,32), execution_simdgroup>`
share one 16x16 fragment layout. Lane t, element e = 0..7 of a fragment:

```
dim0 = 4*(t & 1) + 8*((t >> 3) & 1) + (e & 3)
dim1 = ((t >> 1) & 3) + 4*((t >> 4) & 1) + 8*(e >> 2)
```

Fragments are ordered dim0-fragment first: element index i = e + 8*(f0 + 2*f1). dim0 is MPP's unit-stride
dimension: K for the left input (row-major A[M][K]), N for the right input (row-major B[K][N]) and N for
the destination (row-major C[M][N]), with both transpose flags false. Each lane therefore owns runs of
4 contiguous dim0 elements at two dim1 rows (r and r+8) per fragment. That 4-run is exactly why the
tensor-operand lowering loads 8 bytes per lane: it is the longest contiguous piece in the natural layout.

Probe script: `agent_space/conv2d_bench/opus/layout_probe.py M N K TL TR NSG` dumps
`get_multidimensional_index` for every lane and element through `torch.mps.compile_shader`, which
compiles MPP kernels at runtime through the same `compileLibrary` path as the built-in libraries (same
language version and math mode). Use it to re-derive the layout for any descriptor and to check it on a
new OS or GPU. The MSL spec (2.22.3) calls the layout device specific and the MPP header calls it
implementation defined, so any kernel that hard-codes it needs a guard: a test with exact small-integer
data that fails on a layout change, or a one-time device probe that disables the route on mismatch.

## The recipe that reached the ceiling

1. **Make every lane's elements contiguous so loads and stores are 16 bytes.** A permutation of the
   index inside a 32-wide block is free as long as it is applied consistently to both operands that share
   the dimension: K to left and right, N to right and destination, M to left and destination. The product
   does not care which physical column sits in which logical slot. With the permutation
   `physical = 8*g + j` for lane group `g = (t & 1) + 2*((t >> 3) & 1)`, a lane's two 4-runs along
   dim0 (4g..4g+3 and 16+4g..16+4g+3) become one 8-run: one 16-byte bf16 load per dim1 row, one 16-byte
   store per output row.
2. **Pack the reused operand in exact lane order.** For conv the weights are small and reused by every
   tile, so a tiny per-call dispatch (0.01-0.02 ms) writes each 32x16 block in the order the lanes
   consume it: two 16-byte loads per lane, no index arithmetic in the hot loop. For a GEMM the same
   applies to whichever operand is re-read across tiles, if a one-time repack is cheaper than the
   strided reads it removes.
3. **Reuse loaded rows in registers.** Conv loads each input row once per (kernel row, 16-channel block)
   and builds the three horizontal taps by register shifts. The equivalent for a GEMM is keeping a K
   slice of one operand live across the runs that need it.
4. **No bounds logic on interior tiles.** The lowering edge-checks every load because it cannot know.
   Branch once per tile on whether it is interior; the edge path can be slow.
5. **Keep the hot loop compiler-friendly.** Things that silently cost 10% to 2x: a runtime-indexed
   private array (spilled, 13 TFLOPS), `#pragma unroll` over a loop whose body holds `op.run` (11%
   slower; `unroll(disable)` was neutral), a fully unrolled K loop (slower), passing the cooperative
   operands into a helper that calls `op.run` (results silently dropped). Write the taps / K steps out
   explicitly with the run in the kernel body.

## Where it does not help

- **Algorithmic wins beat load wins.** At 256+ input channels the branch's Winograd routes run the
  equivalent of 32-36 TFLOPS by doing fewer multiplies; the cooperative direct kernel is 0.20-0.81x
  there. A 26 TFLOPS ceiling on dense multiplies is a hard limit for any direct formulation.
- **Tiling is now yours.** Rows narrower than the tile width leave simdgroups idle (0.44-0.97x on
  widths below 128); the lowering's generic path degrades more gracefully there.
- **Only the last few percent on an already compute-bound tile** (device-tensor path was 94-97% of
  peak). The bigger wins over the branch (1.04-1.41x) came on shapes where the branch's existing route
  was further from the ceiling.
- The MPP `convolution2d` op is a different beast: NHWC only (strided NCHW views return zeros, MSL 4.1
  table 7.8 and `get_stride(0) == 1`), 0.98-1.01x of MPSGraph on channels-last. Not a GEMM building block.

## Minimal demo: `agent_space/gemm_coop_demo.py`

One file, no rebuild (JIT through `torch.mps.compile_shader`), ~250 lines, plain bf16 GEMM
C[M][N] = A[M][K] @ B[K][N] with fp32 accumulation. Three kernels on the same 64x64-per-threadgroup grid:

- `gemm_tensor`: the production pattern from the branch (device-tensor operands, dynamic K, one
  `op.run` per tile, `execution_simdgroups<4>`, `out.store`).
- `gemm_coop_naive`: cooperative operands filled element by element through
  `get_multidimensional_index` (self-describing, 2-byte loads). Shows the API; always correct.
- `gemm_coop`: cooperative operands filled with 16-byte loads using the probed layout and the
  intra-block renaming, one 32x32 tile per simdgroup, K in steps of 32, 16-byte stores.

It checks all three against a CPU float reference (exact small-integer data: zero differing elements;
randn: within bf16 rounding), prints a GPU health probe, then times them interleaved against `torch.mm`.
If a future OS or GPU changes the cooperative layout, the exact-integer check on `gemm_coop` fails
while `gemm_coop_naive` still passes: that is the layout guard.

```
source .venv/bin/activate && python agent_space/gemm_coop_demo.py            # numerics + timing
python agent_space/gemm_coop_demo.py --no-time                                # numerics only
python agent_space/gemm_coop_demo.py --sizes 2048,2048,2048 --rounds 4 --calls 20
```

Measured 2026-11-01 (M5 Pro, macOS 27.2, sustained ms, best of 4 interleaved rounds of 20 calls,
probe 742 us; log in `agent_space/gemm_coop_demo.log`):

| size | torch.mm (MPSGraph) | gemm_tensor | gemm_coop_naive | gemm_coop | coop vs mm | coop vs tensor |
|---|---:|---:|---:|---:|---:|---:|
| 2048x2048x2048 | 0.693 (24.8 T) | 0.684 (25.1 T) | 0.725 (23.7 T) | 0.673 (25.5 T) | 1.030x | 1.017x |
| 4096x4096x1024 | 1.340 (25.6 T) | 1.458 (23.6 T) | 1.408 (24.4 T) | 1.330 (25.8 T) | 1.008x | 1.096x |
| 1024x1024x4096 | 0.357 (24.0 T) | 0.391 (22.0 T) | 0.390 (22.0 T) | 0.346 (24.8 T) | 1.031x | 1.129x |
| 512x512x512 | 0.031 (8.7 T) | 0.019 (14.4 T) | 0.042 (6.4 T) | 0.018 (14.8 T) | 1.711x | 1.033x |

So an untuned 32x32-per-simdgroup cooperative GEMM already sits at 25.5-25.8 TFLOPS and is 1-3% ahead of
MPSGraph's matmul on the large shapes, and 10-13% ahead of the tensor-operand pattern where that pattern
drops to 22-24 TFLOPS. The naive element-wise fill is only 4-10% behind at large K (its 2-byte loads
are latency-hidden by the matrix work) but 2.3x slower at 512^3. Nothing in `gemm_coop` is tuned: no
software pipelining, no threadgroup staging, no larger per-simdgroup tile, no split-K; those are the
knobs for the real migration.

One numerical note: renaming K changes the order in which the matrix unit sums the 32 products of a
run, so `gemm_coop` differs from `gemm_tensor` / `gemm_coop_naive` by fp32 rounding (max rel err
5e-3 vs 4e-4 at 128x192x256 against the float reference, i.e. about one bf16 ulp on a few near-zero
outputs; the exact-integer tests are bit-identical). Both orders are valid; it only matters for tests
that compare bitwise against a fixed path.

## For the GEMM migration specifically

- Start with the probe and the register-only ceiling test (`agent_space/conv2d_bench/opus/mxu_peak.py`)
  on the target GPU; the numbers above are M5 Pro only. On a chip where the tensor-operand lowering
  already reaches the ceiling there is nothing to gain.
- A plain row-major GEMM already has contiguous dim0 for all three operands, so step 1 is only the
  intra-block permutation plus 16-byte loads. The question is whether the extra register pressure of
  hand loads (two operands per run live in registers) still leaves room for enough tiles in flight.
- Compare against `addmm`'s current MPS path and MPSGraph on the same sustained, interleaved protocol
  (`agent_space/conv2d_bench/sd_bench.py` style: warm-up, 40 back-to-back calls, GPU probe healthy
  before every run; the isolated first-call numbers are noise).
- Reference implementation: `agent_space/conv2d_bench/opus/coop_kernel_final.metal` (kernel + weight
  pack, 187 lines) and the integration patch `agent_space/conv2d_bench/opus/cooperative.patch`.
  Experiment harnesses: `opus/coop_conv.py`, `opus/coop4.py`, `opus/hand_ab.py` (JIT A/B of kernel
  variants without a rebuild).
