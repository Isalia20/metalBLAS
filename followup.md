# Follow-up: bf16 GEMM shapes still under MPSGraph

Shapes where `metalblas.matmul` is still under MPSGraph after Session 10 (see
`optimizations_done.md`).  All bf16 (bf16 in, fp32 accumulate, bf16 out).
Measured with `agent_space/bench_followup.py` (interleaved best-of-N, M5 Pro,
macOS 26.4.1).

## ROOT CAUSE (Session 10 disassembly): a private 16x16x16 MMA intrinsic

torch.matmul (MPSGraph) runs kernel `gemm_a18` (MPSNDArray.framework metallib).
Disassembled it (`metal-objdump`, plus a DYLD-injected Metal pipeline spy to
capture the live dispatch).  Findings:

- It uses **`air.simdgroup_matrix_16x16x16_multiply_accumulate`** -- a 16x16x16
  bf16->fp32 MMA, the M5 tensor coprocessor's NATIVE op.  It does NOT use
  `mpp::tensor_ops::matmul2d` at all.
- Tile **(BM=64, BN=32, NSG=2)**, used uniformly for thin-N AND the square.

The 16x16x16 intrinsic is **not reachable from the public Metal API** we have:
- MSL `simdgroup_matrix` is hardcoded to 8x8 (no `__metal_simdgroup_matrix_16x16`
  builtin); 8x8 is the ALU path (~6 TF), not the coprocessor.
- asm-label injection of the air intrinsic fails (Metal lexer rejects dots; asm
  labels resolve to link symbols, not frontend-lowered intrinsics).
- `torch.mps` only accepts MSL source (`_mps_compileShader`) -- no way to load a
  hand-authored AIR metallib that uses the intrinsic.

`matmul2d` (our path) is the only public coprocessor entry: its 16x**32**x16
fragment hits ~25 TF on the square (BN=64 = 2 N-fragments) but caps ~22 TF at
N=128 / ~13 TF at N=64 because a 1-fragment-wide N tile underfills.  MPS's
16x16x16 (FN=16) gets 2x the N-fragments at the same N -> 25 vs 22.  **So the
thin-N gap is a private Apple intrinsic, not a tiling/scheduling trick.**

## Regressing shapes (after Session 10)

| shape (MxNxK)   | speedup | category |
|-----------------|--------:|----------|
| 4096x128x8192   | ~0.83x  | thin-N, large M -- matmul2d 16x32x16 ceiling |
| 8192x128x4096   | ~0.83x  | thin-N, large M |
| 4096x64x4096    | ~0.84x  | very-thin-N (was 0.76; conv2d path helped) |
| 2048x128x8192   | ~0.82-0.86x | thin-N, medium M (split-K) |
| 4096x128x11008  | ~0.86x  | thin-N, large M |
| 4096x128x4096   | ~0.94x  | thin-N, large M |
| 4095x4095x4095  | ~0.96x  | large non-divisible (edge case, deprioritized) |
| 512x512x16384   | ~0.95-0.98x | deep-K (near parity; noisy) |

All remaining sub-0.99 thin-N shapes are at the **matmul2d 16x32x16 ceiling**
described above.  Per user decision (Session 10): accept this ceiling -- it is a
private-intrinsic gap, not a missed kernel opportunity.

## Session 10 wins shipped (genuine metal, no MPSGraph fallback)

- **Very-thin-N N=32 was catastrophic (0.5x) -- FIXED.** N=32 fell through the
  `N>=64` dispatch gate to the slow manual `m5` kernel (0.38-0.89x).  Lowered the
  gate to **N>=32** so N=32/48 use the tensor-unit `m5_tensor` autotuner instead:
  256x32x256 0.46->1.18x, 1024x32x1024 0.89->2.50x, 4096x32x4096 0.52->0.78x,
  8192x32x4096 0.65->0.78x.
- **conv2d-1x1 path for very-thin-N** (N<=64, multiple of 32).  convolution2d is
  the OTHER public coprocessor entry; a 1x1 conv == GEMM with M as spatial width
  and N as output channels, and it schedules thin output-channels a little better
  than matmul2d does thin-N (4096x64x4096 0.79->0.84, N=32 ties m5_tensor).  Added
  as an autotuner candidate gated to N<=64, so it is used ONLY where it wins and
  never regresses N>=96 (which keeps matmul2d).  It narrows the thin-N gap but
  does not close it (the private 16x16x16 intrinsic still wins).

## What was tried for large-M thin-N (Sessions 8-10), all at/under the ceiling

Every (BM,BN,NSG) matmul2d tile (sweep caps ~22 TF N=128, ~13 N=64); very tall BM;
multi-subtile-per-threadgroup ILP (worse, ~18-20); broad split-K incl. the good
tiles + main-pass-only (worse for large M); destination paths (fp32/bf16/direct,
<1% diff); single-simdgroup matmul2d (~21); occupancy / tiny-BM (worse); 1x1
conv2d (helps only N<=64); double-buffered threadgroup staging (~10 TF); the
transpose/C^T trick (0.917x); BN=16 (14 TF, wastes half the FN=32 frag).  The
disassembly explains why none reach 25: they all drive matmul2d's 16x32x16, while
MPS uses the private 16x16x16.

## 4095^3 (edge case)

4095 (odd) has no fragment-aligned divisor, so partial edge tiles on the slow
dynamic-slice path are unavoidable.  Our ALIGNED 4096-padded matmul alone beats
MPSGraph's 4095^3, but the pad-copy negates it.  Deprioritized per user.
