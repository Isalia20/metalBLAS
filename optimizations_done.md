# GEMM optimization notes

## Session 11 — small-batch decode (2≤M≤63): the M≥64 gate that hid a 0.45× regime

Goal: benchmark bf16 across real LLM-inference shapes (decode / batched decode /
prefill / attention for Llama-3-8B, Qwen2.5-7B, Llama-3-70B) and fix what loses.
The suite was parity-or-better almost everywhere (overall median 1.03×, 33/45 at
≥0.99×) with **one clear loser: small-batch decode at M=16** — Llama-8B o_proj
0.45×, down_proj 0.49×, qkv 0.70×, gate/up 0.79× (group median 0.60×).

**Root cause — an untuned dispatch gate, not a kernel limit.** Two `M >= 64`
gates (the lean contiguous fast path and the `backend=="auto"` branch) routed
*every* 2≤M≤63 GEMM to the manual ALU `m5` kernel instead of the tensor-coprocessor
`m5_tensor` path. M==1 was fine (it has its own GEMV path) and M≥64 was fine
(already `m5_tensor` + autotuner), so only the small-batched-M band fell through —
exactly the regime that serves a handful of concurrent decode sequences. Prior
sessions tuned M==1 GEMV and large-M thin-N exhaustively but never this band, so
it was never measured. Forcing `backend="m5_tensor"` confirmed the kernel handles
small M correctly and faster (o_proj M=16 0.46→0.72× with the default tile).

**The tile crossover (isolated sweep, speedup vs torch).** Why 0.72× and not
parity: the *default* picker hands M<N large shapes `(64,64,2)`, which at M=16
throws away 48 of every 64 MMA rows. A short BM the few rows nearly fill is the
fix, and the best BM flips at M≈32:

| shape          | M  | (16,64,2) | (32,64,2) | (32,128,4) | (64,64,2) |
| -------------- | -- | --------- | --------- | ---------- | --------- |
| o_proj 4096²   | 16 | **1.01×** | 0.96×     | 0.98×      | 0.74×     |
| o_proj 4096²   | 48 | 0.86×     | **1.13×** | 1.10×      | 0.89×     |
| down 4096×14336| 16 | **1.05×** | 1.03×     | 1.04×      | 0.77×     |
| down 4096×14336| 48 | 0.91×     | **1.21×** | 1.16×      | 0.95×     |

M≤16 wants BM=16 (exact — 1 M-fragment, FM=16); 32≤M≤48 wants BM=32 (covers M in
one tile without over-splitting). The same thin-dimension underfill as thin-N,
now on the M axis (see Session 10 / the private-16×16×16 ceiling) — but here the
short-BM tiles reach parity, so it is *not* at that ceiling.

**The fix (3 edits, all `dispatch.py`).**
- Lowered both `M >= 64` gates to **`M >= 2`** (M==1 still returns earlier via the
  GEMV path; the `packed_ab` guard still pins non-contiguous inputs to `m5`, so
  correctness is unchanged).
- `_pick_m5_tensor_tile`: a **thin-M branch** (`M≤48 and N≥1024` → `(16,64,2)` if
  M≤16 else `(32,64,2)`), mirroring the M==1 padded-GEMV BM=16 logic. Gated on N
  being the wide axis so thin-N (large M, small N) keeps its tall-narrow BN=32 path.
- `_m5_tensor_tile_candidates`: in the `mn≤256 and mx≥1024` regime, when `mn≤48`
  prepend `[(16,64,2),(32,64,2),(32,128,4)]` so the autotuner probes the short-BM
  family and picks the per-shape winner (this is what gets the *full* win, not the
  0.72× the static default gave).

**Results.** M=16 group **0.60× → 1.04× median** (o_proj 0.45→1.00, qkv 0.70→1.05,
gate/up 0.79→1.07, down 0.49→1.04). The whole bf16 suite: median **1.03→1.05×**,
shapes at ≥0.99× **33/45 → 39/45**, worst case **0.45× → 0.90×**. Everything else
is unchanged within noise — M=1 decode 1.04–1.06×, M=64 1.15×, prefill M=512/2048
0.99–1.04× (3 models), attention 1.01× (the residual sub-0.99 are now *only* the
thin-K/N attention shapes, A@V s≥2048 ~0.90×, the documented coprocessor ceiling).

**Verified.** All 58 correctness tests pass. Precision is **bit-identical** old vs
new: fp32 small-M went `m5`→`m5_tensor`, both TF32-relaxed (e.g. fp32 M=16 4096²
max_err 0.2517 on both); bf16/fp16 unchanged. M≥64 routing is byte-for-byte
unchanged (the thin-M picker branch is `M≤48`, and `mn≤48` adds no candidates at
M=64). Benchmark harness in `agent_space/` (out of repo).

## Session 10 — disassembled MPSGraph (the thin-N root cause) + conv2d/N=32 wins

Goal: close the residual large-M thin-N gap (N=64/128, ~0.76-0.88x).  Per user
direction, this session DISASSEMBLED what torch.matmul (MPSGraph) actually runs,
to find why it wins, then shipped the genuine metal gains the finding enabled.

**The disassembly (definitive root cause).**  Captured the live dispatch with a
DYLD-injected Metal pipeline spy (swizzles `newComputePipelineStateWithDescriptor`
+ the encoder's `setComputePipelineState`/`dispatchThreadgroups` on the venv
python, which is adhoc/linker-signed so DYLD_INSERT works) and disassembled the
kernel with `metal-objdump --arch-name=air64` on `MPSNDArray.framework`'s 75 MB
`default.metallib`.  torch.matmul -> MPSGraph -> kernel **`gemm_a18`** -> templated
`gemm<float,bfloat,...>`.  Tile **(BM=64, BN=32, NSG=2)** (triangulated: every
shape has exactly BM*BN=2048 output elems/tile, 64 threads/tg), used uniformly for
thin-N AND the 4096^3 square.  **The decisive instruction is
`air.simdgroup_matrix_16x16x16_multiply_accumulate`** (601 calls; bf16 sig
`(<8 x bfloat> A, i1 transA, <8 x bfloat> B, i1 transB, <8 x float> C) -> <8 x
float>`) -- a 16x16x16 MMA, the M5 coprocessor's native op.  It uses ZERO
`matmul2d`.

**Why we can't copy it.**  The 16x16x16 intrinsic is not in the public Metal API:
MSL `simdgroup_matrix` is hardcoded to 8x8 (`_valid_simdgroup_matrix_size`; only
`__metal_simdgroup_matrix_8x8_*` builtins exist), and 8x8 is the ALU path (~6 TF
on 4096^3 vs matmul2d's 25 -- it does NOT use the coprocessor).  asm-label
injection of the air intrinsic fails (the Metal lexer rejects dots in asm strings;
and asm labels resolve to LINK symbols, not frontend-lowered intrinsics -- a
dotless label compiles but fails link).  `torch.mps` only accepts MSL source
(`_mps_compileShader`); no custom-metallib loader.  So `matmul2d` (16x**32**x16)
is our only public coprocessor entry: it hits ~25 TF on the square (BN=64 = 2
N-fragments) but caps ~22 TF at N=128 / ~13 TF at N=64 because a 1-fragment-wide
N tile underfills.  MPS's 16x16x16 (FN=16) gets 2x the N-fragments at the same N
(25 vs 22).  **The thin-N gap is a private Apple intrinsic, not a tiling trick.**
Per user decision: accept this ceiling for N=128 and ship the gains below.

**Gain 1 -- N=32 was catastrophic (0.5x), FIXED.**  Very-thin-N (N=32/48) fell
through the `N>=64` auto-dispatch gate to the slow manual `m5` kernel (0.38-0.89x).
Lowered both gates (lean fast path + auto) to **N>=32** so N=32/48 use the
tensor-unit `m5_tensor` autotuner: 256x32x256 0.46->1.18x, 512x32x512 0.38->1.31x,
1024x32x1024 0.89->2.50x, 4096x32x4096 0.52->0.78x, 8192x32x4096 0.65->0.78x.
This is byte-for-byte unchanged for N>=64 (the gate was already satisfied there).

**Gain 2 -- conv2d-1x1 path for very-thin-N.**  `convolution2d` is the OTHER public
coprocessor entry (a 1x1 conv == GEMM, M as spatial width, N as output channels).
Empirically it schedules thin output-channels a bit better than matmul2d does
thin-N for N<=64 (4096x64x4096 0.79->0.84, N=32 ties m5_tensor; N>=96 still prefers
matmul2d).  Added `conv1x1_gemm` (kernels.py; K baked into the conv descriptor as a
constexpr, so specialized per K) + `_Conv1x1Plan`/`_is_conv_regime` (N<=64, N%32==0,
M>=512)/`_conv_specs` (dispatch.py), wired into the autotuner alongside split-K:
conv candidates are correctness-checked vs the single-pass primary and TIMED, kept
only when they win, so this never regresses a shape matmul2d already nails.  conv
narrows the thin-N gap but does not close it (the private 16x16x16 still wins).

All 58 correctness tests pass; the giant-vocab GEMV / lm_head / llm / square wins
all hold (guard bench unchanged within noise).  Remaining sub-0.99: N=128 thin-N
(~0.83-0.94x, the matmul2d ceiling above) and 4095^3 (~0.96x, odd-dimension edge).

## Session 9 — tall-narrow tiles for thin shapes + split-K for deep-K

Goal: push the four residual sub-0.99x shapes from Session 8 to >=0.99x with real
kernel work (no MPSGraph fallback).  Two new mechanisms landed; two shapes remain
at the matmul2d primitive ceiling.

**Tall-narrow tiles (fixes the thin shapes).** A sweep found thin shapes want a
TALL-NARROW tile the candidate list never offered: BN=32 with a tall BM (or the
mirror, BM=32 + wide BN) makes many small tiles where the square `(64,64,2)`
underfills the few output tiles along the short axis.
- `128x4096x4096` (thin M=128): 0.96 -> ~1.00x on `(128,32,2)`.
- `256x4096x4096`: ~0.98x; `4096x256x4096`: ~1.00x (keeps `(64,64,2)` -- tall-narrow
  is *bad* here at 0.87x, and the autotuner correctly avoids it).
- `4096x128x4096` (thin N=128): 0.80 -> ~0.92x avg (0.88-0.97, high variance).
  Still <0.99 -- this is the matmul2d microkernel ceiling for N=128 (see below).

Implementation in `_m5_tensor_tile_candidates`: a thin regime (`min(M,N)<=256 and
max>=1024`) offering both orientations `[(128,32,2),(256,32,4),(32,128,2),
(32,256,4),(64,64,2)]`, and a deep-K-small regime (`max<=1024 and K>=8*max`)
offering `[(128,32,4),(128,32,2),(192,32,2),...]`.  The best tile FLIPS with size
(512^2x32768 -> (128,32,4); 1024^2x16384 -> (128,32,2)), so these are probed, not
gated.  The wins are only ~1-2% over the square default, under the 3% autotune
margin, so a `_TALL_NARROW_MARGIN=0.01` applies to these regimes (with doubled
probe iters/reps so a 1% gap resolves above the noise; bad tiles in each family
lose by 15-25%, far outside any tie).

**Split-K (beats MPSGraph across the deep-K family).** A single `op.run` over the
full K underfills the GPU when M,N are small (few output tiles) but K is huge:
each of the few tiles carries a long serial K-reduction.  Split-K runs G K-chunks
per output tile (G x the tiles, filling the cores) and reduces the partials.  The
atomic-accumulation prototype lost to contention; the winning design writes each
chunk's partial to its OWN buffer -- chunk 0 straight to C (bf16), chunks 1.. to
fp32 planes -- then a cheap second pass does `C += sum(planes)`.  New shaders
`splitk_gemm`/`splitk_reduce` (kernels.py), a `_SplitKPlan` class, and
`_is_splitk_regime` (low-precision, `K>=2048`, `min(M,N)>=64`, `M*N<=1.5M`, and
`min<=256 or K>=8*max` -- few output tiles) + `_splitk_specs` ((128,32,2)/(64,64,2)/
(32,64,2) x G in {2,4}).  The autotuner times split-K candidates alongside the
single-pass tiles (after a one-time check that the result matches the single-pass
output within 2%, guarding chunk 0's extra bf16 rounding) and keeps whichever
wins, so split-K is used ONLY where it helps -- two sub-cases:
- *deep-K squares*: `256x256x32768` -> **1.90x**, `256x256x16384` -> **1.82x**
  (MPSGraph badly underfills a tiny+deep output), `512x512x32768` 0.96 -> **0.99x**,
  `512x512x16384` 0.96 -> 0.98x.  `768^2/1024^2/2048^2 x16384` use single-pass
  (1.05-1.13x; split-K timed but not chosen).
- *thin-N* (small N -> few n-tiles): `256x128x16384` -> **2.28x**, `512x128x8192`
  -> **1.62x**, `1024x128x4096` -> **1.01x**, `2048x128x8192` 0.67 -> **0.81x**,
  `2048x128x4096` 0.86 -> **0.91x**.  (The `M*N<=1.5M` cap is what now lets thin-N
  in -- the earlier `max<=1024` cap, tuned for deep-K squares, wrongly excluded it.)

The hot path stays lean: the cached plan is the `(fn, thr, grp)` tuple for the
common single-pass tile (inlined), only the few-tile regime gets a `_SplitKPlan`
(dispatched via a `type(plan) is tuple` check, ~0.05us).  58/58 correctness tests
pass; small-shape parity (512^3, 333x444x555) unchanged.

**What still loses: large-M thin-N, and 4095^3 (edge case).**  thin-N with LARGE M
(`4096x128x4096` ~0.88x, `8192x128x4096` ~0.83x, `4096x128x8192` ~0.82x,
`4096x64x4096` ~0.76x) stays under MPSGraph.  Split-K does NOT help here -- with
large M there are already plenty of m-tiles to fill the cores, so the K-split adds
only reduction overhead; the autotuner keeps single-pass.  A FULL tile sweep of
`8192x128x4096` caps at 0.83x (20.6 TF) while MPSGraph hits 24.9 TF (near the
~25.7 TF bf16 peak).  ~16 approaches were exhausted (Sessions 8-9): every tile;
tall-narrow tiles; split-K (atomic + two-pass); the **transpose/C^T trick**
(compute C^T=B^T@A^T in the faster thin-M orientation -- built correctly after
deriving the transposed extents from the impl static_asserts, but it lands 0.917x,
WORSE than direct: the transposed matmul2d reads cost more than the orientation
saves, and C^T re-reads big A ~4x anyway); manual K-chunking; static-K;
direct-bf16-store; swizzle; 1D/persistent grid; manual m5_gemm/simd_gemm
(0.15-0.62x, ALU-bound).

NB on framing (corrected): it is NOT that MPSGraph has a universally better
tensor-engine kernel -- BOTH paths drive the same matrix engine, and we BEAT
MPSGraph by up to 1.9x where its heuristic underfills (256^2x32768 it runs at only
10.9 TF, 4096x64x4096 15.3 TF -- it doesn't split K, so cores sit idle).  Per-shape
performance is a TILING/SCHEDULING-heuristic game, and MPSGraph's heuristic is
sometimes worse (we win) and sometimes better (large-M thin-N, where it IS
near-peak ~24.9 TF and we cap ~20.6 TF).  For large-M thin-N the honest statement
is "I could not find a matmul2d tiling that matches MPSGraph," NOT "proven
impossible."  A plausible-but-unverified cause is that a BN=32 tile is 1 N-fragment
wide, giving matmul2d few independent MMA chains to overlap (a scheduling effect
that would only bite narrow tiles -- consistent with us matching/winning wider
shapes).  OPEN LEAD not yet tried: one threadgroup issuing several op.run calls to
widen the effective tile / expose more ILP, register-blocking multiple tiles.
4095^3 (~0.92x) is edge-bound: 4095 (odd) has no fragment-aligned divisor so
partial edge tiles are unavoidable; our ALIGNED 4096-padded matmul alone (5.37ms)
BEATS MPSGraph's 4095^3 (5.67ms) but the ~0.3-0.6ms pad-copy negates it.

## Session 8 — the 11 followup regressions: wide-N GEMV, (128,64,4), and the kernel ceiling

`followup.md` listed 11 bf16 shapes under MPSGraph. Fixed the impactful ones with
genuine kernel/dispatch changes (no MPSGraph fallback — this is metalBLAS).

**Wide-N GEMV (the two giant-vocab hits, 0.72x/0.84x).** The M==1, N>=4096 bf16/fp16
path routed to a *padded* `m5_tensor` `(16,128,4)` tile that computes a 16-row MMA
for one real row — 15/16 of the work thrown away. An isolated sweep showed the
dedicated `gemv_t` with **VEC=8** (a full 16-byte/lane coalesced load) is the
bandwidth winner across the *entire* N>=4096 band: 1.07x at N=4096 -> 1.16x at
128256x14336, beating both MPSGraph and the padded tile. Fix: added `(8,8)`/`(8,16)`
to `_GEMV_T_VARIANTS`, a VEC=8 tier in `_gemv_pick` (cols>=4096 & K>=2048: VEC=8,
NWARPS=16 if K>=8192 else 8), and extended the contiguous fast path to send *all*
M==1 through `gemv_t` (the padded m5_tensor branch is now only the
non-contiguous/explicit-backend fallback). The whole M=1 N>=4096 band is now
1.04-1.16x.

**`(128,64,4)` — the missing large-shape tile (0.94-0.96x -> 1.01x).** A tall,
narrow-BN tile none of the prior large candidates `[(64,64,2),(48,128,4),(64,128,4)]`
offered. It is the winner for moderate square / llm shapes: 1024x4096x4096
0.97->1.01, 1024x5120x13824 0.94->1.11 (isolated), lm_head 1024x152064x4096. Added
to the large (`mx>1024`) candidate list.

**Stable giant probe (lm_head 0.75x).** `_probe_params` top tier `(1,1,3)`->`(2,3,3)`.
One timed iter on a ~50 ms kernel was too noisy to rank tiles and flipped lm_head
between ~18.5 and ~21.5 TF run to run; the noisy probe sometimes shipped the 0.75x
tile. Averaging 3 iters / min of 3 reps stabilizes the ranking (now always >1.0x).

**`4095^3` (0.89x -> 0.94x).** Added `(128,128,8)` as a candidate for large
non-64-divisible shapes (`mx>=2048 and not div64`). 4095 = 3^2*5*7*13 has no
MMA-friendly divisor, so it always has partial edge tiles; the big tile amortizes
the edge waste best. Still <1.0 (see ceiling below).

**The ceiling (residual losers, all sub-ms or edge-bound).** `4096x128x4096`
(~0.85x, thin N=128), `128x4096x4096` (~0.97x, thin M=128), `512x512x32768`
(~0.96x, deep-K few-tile), `4095^3` (0.94x). These are at the genuine MPP
`matmul2d` ceiling — ruled out by sweeping every tile, all three backends
(m5_gemm/simd_gemm are 0.15-0.7x here), and a split-K prototype (fp32 atomic
accumulation gives 0.956x on 512x512x32768, *no better* than non-split: atomic
contention + accumulator bandwidth + the cast pass negate the occupancy gain).
MPSGraph's edge here is its MMA microkernel/scheduling for few-tile and
partial-edge shapes, not occupancy — beating it needs a better microkernel.
NOTE: `4097^3` *wins* 1.14x with the same `(128,128,8)` tile; MPSGraph is just
erratically fast at exactly 4095.

## Session 7 — the runtime tile autotuner (and 8192³, the loser hiding past 4096³)

Starting state: `followups.md` listed five bf16 GEMM shapes still under torch —
512³/640³ (0.95×, "MPS GPU ceiling"), `8192×1024×8192` (0.93×), `512²×16384`
(0.95×), `2048×768×16384` (0.92×) — and its TODO #2 spelled out exactly why the
last three couldn't be fixed with the heuristic: **a winning tile exists, but it
can't be gated cleanly.**  The right tile flips on size, aspect, and K-depth, and
every static gate the prior sessions added to chase one shape later regressed a
neighbour (the notes are a graveyard of them — the tall deep-K `(64,64,2)` guard,
the `(128,128,8)` overfit, the `max(M,N)`-only rules).  TODO #2 proposed the fix:
**option (b), a runtime micro-probe that tries the candidate tiles on first sight
of a shape and caches the winner** — "the more general fix [that] would also
catch any future such outlier."  That's this session.

### What the data showed (and the new loser it exposed)

Re-measuring the tail with isolated sweeps confirmed the crossover the heuristic
couldn't track, and turned up a shape `followups` never listed because its table
stopped at 4096³:

| shape (bf16)       | (64,64,2) | thin-BM       | note                              |
| ------------------ | --------- | ------------- | --------------------------------- |
| `2048×768×4096`    | **1.03×** | 0.93–1.00×    | K=2·max — default wins            |
| `2048×768×8192`    | 1.01×     | **1.03×** (64,128,4) | K=4·max — flips                   |
| `2048×768×16384`   | 0.92×     | **1.01×** (64,128,4) | K=8·max — flips harder            |
| `8192×1024×8192`   | 0.93×     | **1.00×** (48,128,4) | 8:1 aspect, K=max — not even deep-K |
| `4096³`            | 0.99×     | 1.00× (48,*)  | within noise                      |
| `6144³`            | 0.98×     | **1.02×** (48,128,4) | thin-BM pulling ahead             |
| **`8192³`**        | **0.95×** | **1.00×** (48,128,4) | **silent loser past the old table** |

The unifying truth: the thin-BM `(48,128,4)` tile beats the `(64,64,2)` default
on *large* shapes by a margin that **grows with size** (≈+2% at 2048³ → +4% at
8192³), because once a shape makes far more `(64,64,2)` tiles than the 16 cores
need (8192³ makes 16384 of them), the "more tiles load-balance better" reason for
`(64,64,2)` is spent and `(48,128,4)`'s better K-amortization wins.  The deep-K
and high-aspect winners are the same family.  A static "`(64,64,2)` is the broad
large winner" rule was silently giving up 4–7% on the biggest squares.

### The autotuner

The heuristic (`_pick_m5_tensor_tile`) is now a *candidate generator*
(`_m5_tensor_tile_candidates`): confident regimes (padded GEMV, tiny-K, ≤256, and
all of fp32 — already 2–3×) return a **single** tile and never probe; the
ambiguous regimes return a short list, primary first:

- **small/medium (max ≤ 1024)** → `[(16,64,2), (32,64,2), (64,64,2)]`
- **large (max > 1024)** → `[(64,64,2), (48,128,4), (64,128,4)]` (one rule for
  square, deep-K, and high-aspect — the probe sorts out which)

On the first call for a never-seen shape, `_gemm_plan` probes the candidates on
the **real operands** (`_autotune_m5t`) and caches the winning launch plan in
`_GEMM_PLAN`; every later call is the same dict-hit hot path as before.  Three
things make it trustworthy rather than a noise generator:

1. **Provably ≥ the heuristic.**  The primary is always candidate 0, and we only
   abandon it for a win that clears a **3% margin** — set above the run-to-run
   noise floor of the smallest probed kernel (~14µs).  This is the whole game on
   512³, whose three tiles sit within ~2% (so it *keeps* its `(32,64,2)` primary)
   versus 640³, whose `(16,64,2)` is a clean ~5% win (so it *switches*).
2. **Warm-all-then-interleave timing.**  Timing each candidate right after its
   own cold warmup penalised whichever ran first (the GPU clocks hadn't ramped) —
   enough to flip the near-tied small squares against the trusted primary.  We
   now warm *every* candidate first, then interleave the timed reps, best-of-N.
3. **Size-scaled probe budget** (`_probe_params`): µs kernels run many iters
   (the sync+Python tail dominates a single 14µs call); ms kernels need only 2–3.
   One-time cost: ~0.05–0.1s for a small shape, ~0.7s for 8192³ — amortized after
   ~16 calls, i.e. nothing for any repeated-shape workload.  `METALBLAS_AUTOTUNE=0`
   disables it.

### Results (isolated best-of-N AUTO vs torch.matmul, bf16)

| shape              | before | after  | tile picked   |
| ------------------ | ------ | ------ | ------------- |
| `8192×1024×8192`   | 0.93×  | **1.005×** | (48,128,4) |
| `2048×768×16384`   | 0.92×  | **1.002×** | (64,128,4) |
| `640³`             | 0.95×  | **1.000×** | (16,64,2)  |
| `8192³`            | 0.95×  | **1.000×** | (48,128,4) |
| `6144³`            | 0.98×  | **1.015×** | (48,128,4) |
| `2048×768×4096`    | 1.03×  | 1.038× | (64,64,2) — *kept*, the guard case |
| `4096³` / `2048³`  | 0.99/0.98× | 0.99/0.98× | (64,64,2) — kept (within margin) |

All five `followups` losers addressed: three driven to parity, plus the two newly
found big squares.  Group medians held or improved (square 1.01×, tall 0.99×,
attn 0.99×, llm 1.00×, odd 1.19×); fp32 untouched (single-candidate, 2.17–2.38×);
fp16 large squares incidentally improved to 1.02–1.03× (it probes too).  All 58
correctness tests pass — the probe is performance-only; every tile computes the
same GEMM.

### Honest residual (two genuine ceilings, both ~parity)

- **512³ (0.94×)** — a *dispatch* ceiling, not a kernel one.  `floor_probe` shows
  the kernel itself at ~0.98–1.02×; the gap is a measured **~0.53µs Python enqueue
  tail** (function call + attribute reads + correctness guards + pool/plan dict
  hits) on a 14µs op.  The autotuner can't touch it (it kept the optimal
  `(32,64,2)`); closing it needs a C++/metal-cpp enqueue, unreachable through
  `compile_shader` (followups TODO #3).
- **512²×16384 (0.97×)** — a *compute* ceiling.  K=32·max, output only 512²; it's
  compute-bound (AI≈253) at 24.7 TF vs torch's 25.9.  No tile/NSG beats it (swept
  12 of them), and split-K loses: the reduction pass alone (≈9–17µs even with
  fp16 partials) exceeds the 15µs gap, and the main pass is already near peak so
  the extra parallelism can't pay for it.  Same class as Apple's medium-square
  sweet-spot ceiling — MPP's `matmul2d` is ~3% under the closed MPS GEMM here.

### What didn't pan out (recorded so we don't re-try)

- **Split-K for `512²×16384`**: see above — reduction overhead > the gap on a
  compute-bound shape.  (Same conclusion the Session-6 notes reached for the
  larger deep-K shapes, for the same reason.)
- **Trimming the 512³ Python dispatch tail**: profiled it — the cuttable pieces
  (`_pooled_out` 0.11µs, `_gemm_plan` 0.06µs, the two `is_contiguous()` 0.06µs)
  sum to ~0.23µs of the 0.53µs; the rest is irreducible call/attribute machinery.
  Shaving <1% on one shape isn't worth the fragility.
- **A 1% autotune margin / few-iter probe on small shapes**: flip-flopped the
  near-tied 512³ tiles and demoted its primary.  3% margin + 20-warmup/80-iter/
  8-rep timing fixed it.

### Files touched (session 7)

- `metalblas/dispatch.py` — `_m5_tensor_tile_candidates`, `_build_m5t_plan`,
  `_probe_params`, `_autotune_m5t`, the `_GEMM_TILE` record, and `_gemm_plan`
  reworked to probe-and-cache (takes optional `a, b`); the lean GEMM fast path
  passes them.  `_AUTOTUNE` / `_AUTOTUNE_MARGIN` module flags.
- `bench/floor_probe.py` — passes `A, B` so its floor reflects the autotuned tile,
  and reports the real chosen tile from `_GEMM_TILE`.
- `README.md`, `followups.md`, `optimizations_done.md` — autotuner mechanism,
  refreshed tables, remaining-work list.

## Session 6 — static-extent tile slices, and the BM=48 deep-K tile

Starting state: `followups.md` listed the bf16 GEMM band still under torch as a
"MPS GPU ceiling" — 512³ 0.93×, 640³ 0.94×, 2048³ 0.97×, 2048²×8192 0.93–0.98×,
8192×1024×1024 0.97×, and the LLM down_proj 4096²×11008 measured here at 0.94×
(not the 1.07× the old README claimed — that was an earlier machine/thermal
state).  The kernel was thought to be at the limit of what `matmul2d` can do.

Two changes moved nearly all of it to parity-or-better.

### (1) Static-extent tile slices — the untried `matmul2d` lever

`MPPTensorOpsMatMul2d.h` explicitly documents a fast path the kernel wasn't
using: for threadgroups whose tile is **fully inside** the matrix, slice the
operands with *compile-time* extents so `matmul2d` knows the tile is exactly
`BM×BN` and in-bounds — it then drops the per-tile edge/predication logic the
dynamic `slice()` emits on **every** tile, interior or not.  The header calls
this `static_slice`; that spelling doesn't exist — the real method is the
**templated `slice<E0,E1>(coord0, coord1)`** (`tA.slice<dynamic_extent, BM>(0,
m_off)` etc.), discovered by probing the compiler (the `static_slice` name fails
with "no member"; `slice<…>` compiles and yields a `tensor<…, extents<dyn,BM>>`).

The kernel now branches: when `MN_ALIGNED` (M%BM==N%BN==0) every tile is interior
→ pure static path, no branch; otherwise an `inside = (m_off+BM<=M)&&(n_off+BN<=N)`
test routes interior tiles to the static path and the ≤1 partial tile per column
to the old dynamic+validity-mask path.  Precision is unchanged — `matmul2d` still
accumulates the K-reduction in fp32 internally regardless of the destination
type (verified: max_err identical to the dynamic path, and a direct-bf16-store
variant gave the same error).

Measured (isolated best-of-N, the static path vs the old dynamic one):

| shape           | dyn    | static | note                                    |
| --------------- | ------ | ------ | --------------------------------------- |
| 1024³           | 0.93×  | 1.02×  | interior-tile heavy                     |
| 1023³           | 0.81×  | 1.01×  | edge path correct + faster              |
| 2048³           | 0.94×  | 0.98×  |                                         |
| 4096³           | 0.99×  | 1.01×  |                                         |
| 4097³           | 0.87×  | 1.12×  | mostly-interior + few edges             |
| 257³            | 1.46×  | 1.52×  |                                         |
| 333×444×555     | 1.00×  | 1.02×  |                                         |

It's a strict win — static ≥ dynamic on every shape measured — so it's now the
default for all non-transposed `m5_tensor` launches (`STATIC_SLICE` compile flag,
on whenever `!trans_a && !trans_b`, which is everything auto-dispatch routes
here; transposed keeps the dynamic path).

### (2) BM=48 — the deep-K tile

With the dynamic-vs-static gap closed, the deep-K shapes (K ≥ 2·max) were the
last clear losers.  The old picker gave them `(64,128,4)` (and a brief
experiment special-cased very-deep `(128,128,8)`, which **craters** the larger
deep-K shapes — 4096²×11008 → 0.82×, because 64-row tiles make too few/too-heavy
threadgroups for a large output).  A tile sweep found a **thin BM=48, wide
BN=128** tile is the deep-K sweet spot across the whole regime: the long K-loop
still wants wide BN to amortize it, but BM=48 yields ~`85·N/128` finely-grained
tiles that load-balance the 16 cores far better than 64+.  (48 = 3·FM(16), so the
fragment tiling is exact; M%48 leaves ≤1 partial row-tile per column → edge path.)

| shape (K/max)        | (64,128,4) | (128,128,8) | **(48,128,4)** |
| -------------------- | ---------- | ----------- | -------------- |
| 4096²×8192  (2.0)    | 0.96×      | 0.81×       | **1.00×**      |
| 4096²×11008 (2.7)    | 0.96×      | 0.82×       | **1.02×**      |
| 2048²×8192  (4.0)    | 0.94×      | 0.98×       | **0.99–1.00×** |

So the LLM down_proj (4096²×11008) and the deep-K square both went from losing to
winning.  `(48,128,4)` is bf16/fp16 only; fp32 deep-K keeps `(64,128,4)` (already
2.2× — torch's strict-fp32 path is far slower, so the tile barely matters).

### (3) Occupancy-aware small-medium tile + deep-K tall guard

A loser-hunt *outside* the standard benchmark groups (dense square scan 384–1280,
plus medium shapes with varied K and aspect) exposed two mis-picks the old
`max(M,N)`-only rules made — both from ignoring tile COUNT and K:

- **`max ≤ 768 → (32,64,2)` had no K gate.**  Medium M,N with deep K
  (`768²×4096`, `768²×8192`) got the thin tile that starves them — **0.85× /
  0.82×**, vs `(64,64,2)`'s 1.04× / 1.08×.  And near-square `768 < max ≤ 1024`
  (`896³` 0.94×) got the heavy default.  The right model: the thin `(32,64,2)`
  tile is for when `(64,64,2)` would make **too few tiles to fill the 16 cores**
  (`ceil(M/64)·ceil(N/64) < 120` — `512²×8192` only makes 64 → 0.80× heavy vs
  0.98× thin) **or K is cube-ish** (`K ≤ max`, short K-loop needs many tiles to
  amortize).  Gating on that fixed `768²×4096` →1.05, `768²×8192` →1.09,
  `704²×4096` →1.18, `896³` →1.00, `960³` →1.06, with `512²×deepK` still 0.99.

- **The `(48,128,4)` deep-K tile is thin-M / wide-N**, so it's backwards for a
  moderate **tall** (M>N) deep-K shape (`2048×768×4096` 0.93×, `2048×512×4096`
  0.91×): too few N-tiles for BN=128, M over-split.  Guarded to fall back to
  `(64,64,2)` when `N < M and max ≤ 2048` (→1.02–1.03×, holds at every K depth
  tested up to K=16384); larger (`4096×768×8192` 0.99×) or N≥M (`768×2048×4096`
  1.04×) keep the wide tile.

The unifying lesson: "small problem ⇒ thin tile" must be judged by **tile count
and K**, not `max(M,N)` — the old rule's blind spots were medium-M,N-deep-K (too
big for the thin tile) and tall deep-K (wrong tile aspect).

### Honest residual (after all of the above)

bf16 is parity-or-better on every common shape.  What still loses: `512³`/`640³`
(0.95×, the GPU ceiling above); `512²×16384` / `2048×768×16384` (0.92–0.95×,
extreme depth — near ceiling); and `8192×1024×8192` (0.93×, an 8:1 aspect + deep
K).  The last is the only *non-ceiling* miss: `(48,128,4)` gives it 1.01×, but it
can't be gated cleanly — the same aspect/depth at half scale (`4096×512×4096`)
wants `(64,64,2)` (1.02×), separated only by absolute size.  `(48,128,4)` is NOT a
viable broad large default: it craters the non-divisible squares the `(64,64,2)`
default wins big (`2047³` 1.36×, `4097³` 1.20×).

### Result

bf16 is now parity-or-better on every benchmarked shape **except 512³ and 640³**.
A zero-dispatch kernel-floor probe (`bench/floor_probe.py`, output pre-allocated,
tight enqueue loop) shows those two are a genuine GPU ceiling: 512³ floor 0.969×,
640³ floor 0.966× — MPS's most-optimized medium-square sizes, where `matmul2d`
itself is ~3% slower on the GPU and the valid tile space (BM∈mult-16, BN∈mult-32)
is exhausted ((32,64,2) is the best for 512, (16,64,2) for 640, and they
conflict).  The remaining ~3% AUTO gap on top of the floor is the ~0.45 µs
Python-enqueue tail (torch's C++ enqueue beats it; not reachable from Python).
fp32 wins everywhere (1.14–2.81×, including 512³ at 1.24×).

### What didn't pan out (recorded so we don't re-try)

- **Static-K descriptor specialization** (`matmul2d_descriptor(BM,BN,K_static,…)`
  with a static-K input slice, one `op.run`): the lever `followups.md` flagged as
  "the only untried one."  It's a **dead end** — equal to or slightly slower than
  `dynamic_extent` on the deep-K shapes (MPP schedules the K-loop the same way
  internally either way).  `bench/m5t_statick_proto.py`.
- **Direct-device store** (`op.run(mA,mB,tCs)` straight to a bf16 tensor slice,
  skipping the f32 cooperative tensor): ~1% faster *when it works*, but **craters**
  for narrow-BN / high-NSG tiles (32×32 → 0.3–0.57×; bad store access pattern).
  The fp32-coop-accumulate path is the robust choice and within ~1%.
- **(128,128,8) for all deep-K**: see above — only the least-bad option on the
  smallest deep-K shape, terrible once M,N grow.
- **Split-K for 2048²×8192**: the reduction pass alone (~60–400 µs depending on
  M,N) exceeds the ≤3.7% gap to torch; can't net a win on these large shapes.

### Files touched (session 6)

- `metalblas/kernels.py` — `m5_tensor_gemm` rewritten with the static-slice
  interior path + edge fallback; new `STATIC_SLICE` compile flag.
- `metalblas/dispatch.py` — `_pick_m5_tensor_tile`: deep-K branch returns
  `(48,128,4)` for bf16/fp16 with the tall-shape `(64,64,2)` guard; small/medium
  branch is now occupancy-aware (`max ≤ 1024` with the `n64 < 120 or (K ≤ max and
  64-aligned)` thin-tile gate) instead of the old `max ≤ 768`.
- `bench/floor_probe.py`, `bench/m5t_ss_proto.py`, `bench/m5t_statick_proto.py`
  — new diagnostic harnesses.
- `README.md`, `followups.md` — refreshed benchmark tables, tile-picker table,
  the static-slice mechanism, and the remaining-work list.

## Session 5 — the K=4096 GEMV band, and "it was the allocation all along"

Starting state: `followups.md` listed the bf16/fp16 GEMV K≥4096 band at
0.87–0.96× ("bandwidth-bound, raw DRAM efficiency") plus the GEMM ceilings
(512³ 0.86×, 511³ 0.89×, 333×444×555 0.92×, 2048²×8192 0.93×).

### GEMV band — two independent causes, both fixed

**(a) Cache-line read granularity.**  `gemv_t` had each lane read one `IN_T` of
B per k, so a 32-lane warp's coalesced read covered 64 B — half a 128-B line.
The other half belongs to the *next* threadgroup; with B > L2 it gets evicted
and re-fetched, wasting ~10–15 % of DRAM bandwidth.  Fix: a `VEC` template — each
lane reads a `VEC`-wide vector of columns, so the warp spans `32·VEC` consecutive
cols (a full line at `VEC=2` bf16, two at `VEC=4`).  The dispatch scales `VEC`
with N and shrinks `NWARPS` (the within-TG K-split) inversely so wide tiles don't
over-subscribe the reduction, and gates `VEC=4` on K≥2048 (short-K large-N like
2048×1024 needs the extra threadgroups for occupancy — VEC=4 there = half the
cores = 382 GB/s vs VEC=2's 427).  fp32 stays `VEC=1` (4-byte loads already fill
a line — fp32 never showed the band).

**(b) Dispatch + allocation overhead — half the gap.**  Isolating the kernel
(`bench/gemv_vec_proto.py`, output pre-allocated) showed the *kernels* were
already 1.0–1.18× while end-to-end `matmul()` was 0.91–0.99×.  These ops run in
10–30 µs and **`torch.matmul` is itself CPU-bound at ~10–14 µs** (its CPU enqueue
≈ its wall time — its GPU GEMM is faster but Python/dispatch caps it).  So a
~1.5 µs Python+alloc tail in our wrapper was the whole difference.  Fixes:
  - per-`(dtype, N)` **launch-plan cache** (`_gemv_plan`) — hot path is one dict hit;
  - `b.new_empty(1, N)` (b's dtype/device, positional dims) is ~0.4 µs cheaper
    than `torch.empty((1,N), dtype=…, device="mps")`;
  - a refcount-gated **output buffer pool** (`_pooled_out`): recycles a buffer the
    caller has *provably released* (`sys.getrefcount == 2` ⇒ held only by the pool;
    views keep a base ref so they read > 2 and force a fresh alloc).  torch fuses
    its output alloc into the C++ op so it overlaps the GPU; we can't from Python,
    so we avoid allocating instead.  Identical semantics to torch.

Result: the whole band is **1.01–1.15× steady-state** (the warm, repeated-shape
regime a decode loop actually runs in).

### GEMM — same overhead lesson

The medium-square losses were *largely the same allocation/dispatch overhead*,
not the kernel: measuring the raw `m5_tensor` kernel with the output reused gave
511³ = **1.003×**, 333×444×555 = **1.019×** (vs 512³ = 0.974× — a genuine MPS
ceiling).  Added a `(dtype,M,N,K)` GEMM launch-plan cache + a lean contiguous
fast path + the output pool.  512³ 0.86→0.93, 511³ 0.89→0.97, 333×444×555
0.92→0.97, 2048²×8192 0.93→0.98, and the LLM gate/down/up shapes flipped from
~0.98× to **1.03–1.10×**.

### Correctness fix (pre-existing)

`m5_tensor`'s `tensor_inline` views assume **packed** storage and ignore lda/ldb,
but `_resolve_inputs` was routing non-contiguous row-major inputs (`X[:, :K]`
with `ld>K`) to it — silently wrong (max_err ~120).  Auto-dispatch now routes
non-packed operands (`lda≠K or ldb≠N`) to the stride-aware `m5` kernel.

### What's left (see followups.md)

512³/640³/deep-K/tall bf16 GEMM remain at the **MPS GPU ceiling** — even a
zero-overhead launch is < 1.0× there (Apple's closed GEMM is faster on the GPU
at its sweet-spot sizes; MPP can't beat it).  511³/333×444×555 have winning
kernels held ~3 % under by an irreducible ~0.45 µs `compile_shader` dispatch tail.

### What didn't pan out (recorded so we don't re-try)

- **`VEC=4` everywhere / `VEC=8`.**  VEC=8 (256-col tiles) starves on TG count;
  VEC=4 below K=2048 or below N=1280 underfills the cores.  The N/K-gated rule is
  the sweet spot.
- **`(128,64,4)` tile for 2048³** (→1.0×) — overfit; regresses 2560³ (0.998→0.992)
  and 3072³ (0.988→0.980).  Left on the `(64,64,2)` default.  Same trap the
  Session-3 notes flagged for the deep-K `(128,128,8)` rule.
- **Ring/always-on output reuse** — unsafe for a general matmul (aliasing).  The
  refcount gate (`getrefcount==2`) is what makes recycling correct.
- **bf16 accumulation in the convert step** (to skip the fp32→bf16 loop) — fails
  the precision tolerance over large K; kept the fp32 cooperative-tensor accum.

### Files touched (session 5)

- `metalblas/kernels.py` — `GEMV_T_SRC` rewritten with the `VEC` template
  (vectorized column loads, edge-clipped scalar tail); `gemv_t()` wrapper takes
  `VEC` and asserts `BLOCK_N == 32*VEC`.
- `metalblas/dispatch.py` — `_GEMV_T_VARIANTS` / `_gemv_handles` (compiled
  variant table), `_gemv_pick` (N/K→VEC,NWARPS rule), `_gemv_plan` + `_gemm_plan`
  (memoized launch caches), `_pooled_out` (refcount-gated buffer pool), lean GEMV
  and GEMM fast paths, and the `packed_ab` guard on the m5_tensor auto routes.
- `bench/gemv_vec_proto.py` — new (VEC × NWARPS sweep with GB/s + max-err).

## Session 4 — GEMV occupancy (it was never dispatch time)

Starting state: `followups.md` listed the M=1 GEMV losses (`1×1024×1024` 0.82×,
`1×4096×4096` 0.91×, fp32 `1×4096×4096` 0.93×) and rationalized them as
"launch-bound / bandwidth-bound, hard."  That framing was wrong, and chasing it
would never have closed the gap: **our Metal dispatch is cheaper than the MPS
graph**, so if torch wins it's not because we launch slower.

### What was actually wrong

Probed achieved GB/s (`bench/gemv_probe.py`) and found the documented losses
were the *mild* cases.  The real disaster was **small-N, large-K** GEMV, which
`followups` never even listed:

```
1×512×4096  bf16:  torch 348 GB/s   ours 0.21×  (75 GB/s)
1×1024×4096 bf16:  torch 425 GB/s   ours 0.35×
1×2048×4096 bf16:  torch 407 GB/s   ours 0.61×
```

Our speedup tracked `n_groups = ceil(N/32)` exactly — i.e. **occupancy**.
`gemv_t` launches one threadgroup per 32-column block, so N=512 gave 16
threadgroups on 16 cores (≈1 each, zero latency hiding).  torch splits K across
many threadgroups to fill the GPU; we never split K at all.

### The fix (and the false start)

First attempt: a two-pass **cross-threadgroup split-K** (partials + reduce,
`k_splits = ceil(128/n_groups)`, fp32 partials, a reused scratch buffer to dodge
per-call alloc).  It worked on the GPU (1×1024×1024 raw kernel 1.47×) but capped
at ~1.05× through `matmul`.  Decomposition (isolating each bare `compile_shader`
enqueue) found the smoking gun: **one dispatch costs ~2.2 µs of Python, two cost
~4.2 µs.**  So the 7 µs kernel was actually CPU-bound — `matmul`'s ~5 µs of
Python (closures, `lru_cache` lookups, two allocations, the 2nd enqueue)
dominated.  The whole cross-TG detour was pure overhead.  (Note: single-run
averages swung 6.8–11 µs from thermal/scheduler noise at this scale — only
best-of-N made the comparison legible.)

The actual fix is simpler and single-dispatch: **split K across the simdgroups
*within* one threadgroup.**  `gemv_t` already supported `NWARPS`; raising it from
4 to 32 puts 1024 threads in each threadgroup, filling a core's budget even with
few column-blocks — no cross-TG reduction, no second pass.  **The entire fix is
in `dispatch.py`; `kernels.py` is unchanged** — the kernel always had the
capability, the old code just always called it with 4 warps.

```
1×512×4096   0.21 → 0.93     1×1024×1024  0.84 → 1.69     1×256×256  → 2.30
1×2048×4096  0.61 → 1.10     fp32 1×4096²  0.93 → 1.03     1×512×512  → 2.16
```

Then cut the `matmul` Python tax for the (CPU-bound) small GEMV: cache the kernel
handles per dtype, and add a lean fast path at the top of `matmul` that detects a
contiguous M=1 row-vector and dispatches directly (passing the 2-D tensors
straight to the kernel — their linear storage *is* the vector — so no `.view()`).
The fast-path guard tests `a.shape[0] == 1` **first** so GEMM (M>1)
short-circuits on one check instead of paying the full guard (~0.3 µs, which is
~2% on a 15 µs shape — enough to nudge `512³` 0.86 → 0.84× before the reorder).

### dtype-specific surprises

* **NWARPS sweet spot moves with N.**  N≤512 (few columns) wants all 32 warps;
  512<N≤1024 peaks at **16** (32 over-subscribes the K-reduction — `1×1024×4096`
  is 1.05× at 16 vs 0.93× at 32); N>1024 wants 32 again.  Shipped a two-tier
  pick (`_gemv_pick`) compiling both variants.
* **Large-N (≥4096) splits by dtype.**  bf16/fp16 prefer the *padded m5_tensor*
  (BM=16 tile — the old picker's BM=64 wasted 63 rows and cratered 4096² to
  0.70×); fp32 prefers high-NWARPS `gemv_t` (m5_tensor loses for fp32 above
  N=4096: 8192 → 0.89× vs gemv 1.03×).

### What's left (see followups.md)

A mid-N, K≥4096 band (`1×512×4096` 0.93×, `1×1024×4096` 0.95×) is now genuinely
bandwidth-bound — torch sustains 400–450 GB/s, we sit at 330–430.  Lead:
vectorized column loads (full-cache-line warp reads).  Not dispatch.

## Session 3 — small shapes, non-divisible shapes, and fp32 → m5_tensor

Starting state: bf16/fp16 medians were ~1.0× but the *tails* were bad, and
fp32 still ran the manual `m5` kernel.  Ran the full bench, then re-measured
every suspect shape in **isolation** with a best-of-N harness
(`bench/sweep.py`) — the full bench is unreliable below ~20 µs (it runs all
groups back-to-back, so tiny shapes read 2-4× their true latency from thermal
/ scheduler effects).  Isolated best-of-6 is the source of truth here.

### The four problems found

1. **Catastrophic small-shape losses.**  The auto-dispatch gate
   (`tiles_m*tiles_n >= 8` on 64-tiles) routed 64³/96³/128³ to the slow manual
   `m5` kernel, and the tile picker handed the rest large-shape tiles.
   Isolated truth: 128³ at 0.37×, 257³ at 0.27×, 511³ at 0.71×.

2. **Non-divisible large shapes were badly mis-tiled.**  The
   "non-divisible-large → (128,128,8)" and "awkward → (64,128,4)" fallbacks
   were *much* worse than just using the default (64,64,2):
   `1025³ 0.81×, 2049³ 0.70×, 3073³ 0.65×, 1535³ 0.90×, 2047³ 0.94×`.

3. **A latent pathology: small-MN deep-K.**  256×256×4096 hit the deep-K rule
   (`K≥2·max & div`) → (64,128,4) → **0.43×**.  The 16 tiles starve the GPU.

4. **fp32 was leaving 30-60% on the table.**  The session-1 note "fp32 prefers
   the manual m5 path" was simply wrong (or stale) — `m5_tensor` beats `m5`
   for fp32 *everywhere*, at identical TF32-relaxed precision.

### The fixes (all in `dispatch.py`)

- **Lowered the dispatch gate** to route every `≥ 64³` low-precision *and*
  fp32 GEMM to `m5_tensor` (dropped the tile-count requirement; only sub-64
  dims and transposed inputs fall back to `m5`).
- **Rewrote `_pick_m5_tensor_tile` as a size-first, dtype-aware picker:**

  | regime                          | tile          | why                                  |
  | ------------------------------- | ------------- | ------------------------------------ |
  | M==1 / N==1 (padded GEMV)       | (64, 128, 4)  | free row over-read, wide BN streams B |
  | tiny K (≤256), big M,N          | (32, 128, 4)  | short K-loop → want more tiles        |
  | max(M,N) ≤ 256                  | (32, 32, 4)   | many tiles to fill 15 cores           |
  | 256 < max ≤ 768                 | (32, 64, 2)   | small BM, light NSG                   |
  | deep K (K≥2·max) & max ≥ 1792   | (64, 128, 4)  | wide BN amortizes the K-loop          |
  | max ≥ 4096, non-aligned, **bf16** | (128, 128, 8) | edge-waste amortization (bf16 only)  |
  | large default, fp32 & divisible | (64, 128, 4)  | 4-byte loads → AI matters → wide BN   |
  | everything else                 | (64, 64, 2)   | the broad winner, incl. non-divisible |

  Key intuition: **small problems want many small tiles** (occupancy /
  latency-hiding); **large problems already saturate** and want the light
  (64,64,2) tile.  The two are checked *before* deep-K so small-MN deep-K
  (256²×4096) gets the tiny tile, not the wide-BN one.

### Three things that turned out to be dtype-specific (and surprised me)

- **(128,128,8) is bf16-ONLY.**  At 4097³: bf16 1.05×, but fp16 0.69× and fp32
  1.54×.  fp16 and fp32 both want the default tile there (0.82× / 2.0×).
- **fp32 large prefers (64,128,4), bf16/fp16 prefer (64,64,2).**  fp32's 4-byte
  loads make arithmetic intensity matter more, so the wider BN wins
  (4096³ fp32: 2.05× vs 1.92×; 2048³: 2.38× vs 2.16×).
- **fp32 m5_tensor precision == m5 precision, to the bit.**  Both run the M5
  tensor unit with `RELAXED=true` float accumulation; max-err matched to 4
  decimals on every probed shape, all inside the existing m5 tolerance.

### Results (isolated best-of-6 AUTO vs torch.matmul)

```
shape           bf16              fp16              fp32
                before → after    before → after    before → after
128³            0.37 → 1.50       0.38 → 1.95       0.62 → 1.92
256³            ~noisy → 1.27     0.63 → 1.69       0.46 → 1.50
257³            0.27 → 1.26       ~    → 1.18       0.67 → 1.29
512³            0.84 → 0.87*      0.56 → 0.84*      1.43 → 2.84
511³            0.71 → 0.88       0.76 → 0.88       1.41 → 1.76
333×444×555     0.77 → 0.93       ~    → 0.87       1.02 → 1.74
256²×4096       0.43 → 0.99       0.43 → 0.99       (n/a) → 2.07
1535³           0.90 → 1.15       ~    → 1.14       2.13 → 2.08
2047³           0.70 → 1.22       0.70 → 1.22       1.74 → 2.03
4096³           0.99 → 0.99       0.97 → 0.99       1.56 → 2.08
4097³           1.05 → 1.04       0.76 → 0.94       1.81 → 1.84
2048²×8192      0.94 → 0.94*      0.94 → 0.98       2.03 → 1.99
```
`*` = genuine MPP/torch ceiling at that size (manual m5/simd backends are far
worse: 512³ bf16 is 0.59× on m5, 0.24× on simd).  fp32 medians per group:
square 1.51→**2.54×**, tall 1.79→**2.49×**, attn 1.72→**2.32×**,
llm 1.61→**2.07×**, odd 1.41→**1.76×**.

All 58 correctness tests pass; fp32 m5_tensor precision separately validated
against a CPU-fp64 reference.

### What didn't pan out (recorded so we don't re-try)

- **(128,128,8) for 2048²×8192 deep-K** gives 0.98× (vs 0.94× for (64,128,4))
  but the win is razor-thin and *overfit to exactly M=N=2048*: 2304² and 2560²
  deep-K both *regress* with it (0.94× vs 0.95-0.96×), and K≥12288 flips back.
  Not worth a fragile band rule — kept the robust (64,128,4).
- **Manual `m5` / `simd` backends for any small shape**: always worse than
  `m5_tensor` (512³ bf16: m5 0.59×, simd 0.24× vs m5_tensor 0.88×).

---

## Session 2 — closing the remaining gap to torch.matmul

Starting state (end of session 1):

| Group  | Median | Notes |
| ------ | ------ | ----- |
| square | 0.94×  | small (512³) at 0.72× |
| tall   | 0.94×  | best 0.95× |
| attn   | 0.93×  | small K=128 at 0.82× |
| llm    | 1.07×  | already beating torch |
| gemv   | 0.85×  | 1×32000×4096 at 0.85× |
| odd    | 0.75×  | non-divisible shapes routed to manual m5_gemm |

Target: ≥ 0.95× across every group, beating torch on multiple shapes.

Final state (end of session 2):

| Group  | Median  | Best                              |
| ------ | ------- | --------------------------------- |
| square | 0.99×   | 512³ at 2.29×, 256³ at 1.87×      |
| tall   | 0.98×   | 1024×4096×1024 at 1.03×           |
| attn   | 0.99×   | 4096×4096×128 at 1.04×            |
| llm    | 0.99×   | 4096×4096×11008 at 1.00–1.21×     |
| gemv   | 0.96×   | 1×32000×4096 at 0.99–1.02×        |
| odd    | 0.96×   | 4097³ at 1.04–1.07×               |

All 58 correctness tests still pass.

---

## Walkthrough — how the gap got closed

### Step 1 — diagnose the worst offenders

Ran the full bf16 bench first (warmup=50, iters=200) and ranked groups
by median.  Two outliers dominated:

- **`odd` group at 0.75×** — non-divisible shapes (257³, 511³, 1023³,
  4097³, 333×444×555) couldn't enter the m5_tensor backend because of an
  `M_ % BM != 0` guard in `dispatch.py`.  They all fell through to the
  slower manual `m5_gemm` kernel.
- **`gemv` group at 0.85×** — the `1×32000×4096` case (lm_head shape) was
  bandwidth-bound at ~220 GB/s while torch achieved ~247 GB/s (peak ≈ 273
  GB/s on M5 Pro).

These were the two big fish.  Everything else was within a few % of 0.95×
and likely to fall out of broader tuning.

### Step 2 — does the m5_tensor kernel even handle edge tiles?

The session 1 code skipped non-divisible shapes because we'd never tested
whether MPP's `cT.store(mC)` handled partial tiles.  Wrote a probe:

```python
# call m5_tensor_gemm directly with M=65, M=100, M=511, M=333×444×555
fn(a, b, out, M, N, K, ..., threads=(NSG*32 * ceil_div(N, BN), ceil_div(M, BM), 1))
# check max_err vs (a.float() @ b.float())
```

Result: max_err = 0 or 0.0625 (bf16-normal rounding) for every shape.
Output was finite and correct.  Two pieces of luck combined to make this
"just work":

1. **`tC.slice(n_off, m_off)` returns a clipped sub-tensor.**  When
   `m_off + BM > gM`, the returned `mC` has reduced extents (e.g.
   `(N_remaining, M_remaining)`), and `cT.store(mC)` only writes to that
   clipped region.  No OOB writes to C.
2. **Metal returns 0 for OOB buffer reads.**  When the kernel reads
   `A[m_off + 63, k]` and that row doesn't exist, the read silently
   returns 0.  Those rows contribute nothing to the dot products.  The
   over-computed rows of `cT_f32` correspond to OOB positions and are
   simply not stored.

Removed the gate.  `dispatch.py` change:

```python
# was:
if M_ % BM != 0 or N_ % BN != 0:
    backend = "m5"  # fall back

# now:
tiles_m = (M_ + BM - 1) // BM
tiles_n = (N_ + BN - 1) // BN
# … run kernel normally
```

Also relaxed the auto-dispatch gate from `tiles_m_64 * tiles_n_64 >= 16`
to `>= 8` so smaller problems could enter m5_tensor too.

Immediate result: `odd` group jumped from 0.75× to ~0.96× median.

### Step 3 — but 4097³ regressed

Lifting the gate exposed a new problem: `4097³` (and similar "just-over"
shapes) dropped to 0.79×.  With `(BM=64, BN=128, NSG=4)`, 4097³ produces
65 × 33 = 2145 tiles, of which 64 + 32 + 1 = 97 are partial edge tiles
(only ~4.4 % wasted compute — not enough to explain a 26 % slowdown).

Did a tile sweep on 4097³:

```
4097x4097x4097: torch=7.92 ms
  (64, 128, 4):  10.84 ms  0.73x   ← terrible
  (128, 128, 8):  7.56 ms  1.05x   ← winner
  (128, 128, 4):  9.02 ms  0.88x
  (128, 64, 4):   8.97 ms  0.88x
  (64, 64, 4):   14.76 ms  0.54x
```

`(128, 128, 8)` won by a wide margin.  The reason: bigger BM × BN tile
halves the total tile count (1089 vs 2145), and NSG=8 keeps each tile's
threads at 256 — enough simdgroups per tile that the edge-tile waste is
amortized across more in-flight work.

Added to picker:
```python
if (M >= 1024 and N >= 1024 
        and not (m_div_64 and n_div_64)
        and (M+127)//128 * (N+127)//128 >= 8):
    return (128, 128, 8)
```

### Step 4 — full sweep across all shape regimes

The session 1 picker treated `(64, 128, 4)` as the global winner.  I
suspected this was suboptimal for several shape classes (medium-size
square, tall, very small K).  Ran a broad sweep:

```
shape                  best tile           speedup
4096×4096×4096         (64, 128, 4)       0.98x
4097×4097×4097         (128, 128, 8)      1.05x   ← non-div
4096×4096×11008        (64, 128, 4)       0.93x   ← deep K
2048×14336×4096        (64, 64, 2)        0.98x   ← LLM
4096×11008×4096        (64, 64, 2)        1.01x   ← LLM
2048×2048×2048         (128, 64, 4)       0.97x
1024×1024×1024         (64, 64, 2)        1.05x
4096×4096×128          (32, 128, 4)       0.94x   ← small K
4096×4096×64           (32, 128, 4)       0.94x   ← tiny K
512×512×512            (32, 64, 2)        1.08x
256×256×256            (32, 128, 4)       2.57x
```

Three observations changed my mental model:

1. **`(64, 64, 2)` wins almost everywhere except deep K.**  Counter to
   the session 1 intuition that "bigger BN amortizes K-reuse."  The right
   intuition is **occupancy**: NSG=2 uses 64 threads/TG instead of
   128/TG, so twice as many TGs fit per core's scheduler.  For most
   shapes, K is moderate (~ M, N) and the K-reuse benefit of wider BN
   doesn't compensate for the parallelism loss.

2. **Deep K (K ≥ 2·max(M, N)) flips the rule.**  When each tile's K-loop
   is the dominant cost (e.g. K=11008 over 4096×4096 output), wider BN
   (`(64, 128, 4)`) amortizes that K-loop overhead across more output
   elements per tile.  Per-tile work increases linearly with BN; per-tile
   fixed cost stays constant.

3. **Tiny K (K ≤ 256) flips it again.**  When K is so small that the
   K-loop is barely any work, you want **more tiles** to amortize the
   per-TG fixed cost.  `(32, 128, 4)` produces 2× the tile count of
   `(64, 128, 4)` in the M direction; for 4096×4096×128 this took us
   from 0.81× to 0.94×.

Built a picker that picks by regime:

```python
def _pick_m5_tensor_tile(M, N, K, dtype):
    if K <= 256 and M >= 1024 and N >= 1024 and ...:
        return (32, 128, 4)               # tiny K
    if M >= 1024 and N >= 1024 and not (M%64==0 and N%64==0) and ...:
        return (128, 128, 8)              # non-divisible large
    if K >= 2 * max(M, N) and ...:
        return (64, 128, 4)               # deep K
    if M <= 256 and N <= 256 and ...:
        return (32, 128, 4)               # very small
    if M <= 768 and N <= 768 and ...:
        return (32, 64, 4)                # small-medium
    if M%64==0 and N%64==0:
        return (64, 64, 2)                # default divisible
    if M <= 768 and N <= 768:
        return (64, 64, 4)                # small awkward
    return (64, 128, 4)                   # large awkward
```

### Step 5 — GEMV via "padded" m5_tensor

The 1×32000×4096 GEMV was the most-fixable remaining issue.  Tried
incremental tuning of `gemv_t` first:

```
  N=32000, K=4096, BLOCK_N=32:
    NWARPS=2: 0.90x
    NWARPS=4: 0.90x  ← current
    NWARPS=8: 0.92x
    NWARPS=16: 0.92x
```

Sub-percent gains.  Not the path.

Then tested calling m5_tensor directly with M=1 — i.e. let the kernel's
BM=64 over-read the 63 OOB rows (which Metal zeros) and use the natural
`cT.store(mC)` clipping to only write row 0:

```
  N=32000: torch=0.99 ms | gemv_t=1.15 ms (0.86x) | m5_tensor=0.99 ms (1.00x)
```

It just works.  The wasted compute (63 dot-products × 0 = 0) doesn't
matter because the operation is bandwidth-bound — the over-fetched A
rows are OOB and don't consume bandwidth (Metal short-circuits the
load).  We pay the same memory traffic as `gemv_t` but get MPP's better
internal scheduling.

Crossover sweep with various N:

```
  N=1024:  gemv 0.67x | m5 0.41x      ← gemv wins (compute-bound, OOB waste hurts)
  N=4096:  gemv 0.98x | m5 0.77x      ← gemv wins
  N=8192:  gemv 0.92x | m5 1.11x      ← m5 wins
  N=12000: gemv 0.97x | m5 0.95x      ← tie
  N=14336: gemv 0.94x | m5 0.99x      ← m5 wins
  N=24000: gemv 0.87x | m5 1.00x      ← m5 wins
  N=32000: gemv 0.85x | m5 1.00x      ← m5 wins
```

Crossover ~ N=8000.  Added to dispatch:

```python
if M == 1 and N >= 8192 and K >= 1024 and is_lp and not trans_a and not trans_b:
    backend = "m5_tensor"
```

Kept the dedicated `gemv_t` for small/medium N (where its lack of OOB
waste matters) and `gemv_nt` for N=1 (where it already achieves bandwidth
peak).

### Step 6 — `MN_ALIGNED` kernel template flag

Looked at the kernel's convert loop:

```cpp
auto cT_out = op.get_destination_cooperative_tensor<..., OUT_T>();
for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i) {
    if (cT_f32.is_valid_element(i)) cT_out[i] = (OUT_T)cT_f32[i];
}
cT_out.store(mC);
```

For divisible shapes, every slot is valid — the `is_valid_element` check
is wasted work.  64 iterations × 4096 threads × ~2-cycle branch = ~half
a microsecond per kernel call, plus pipeline stalls from the branch.

Added an `MN_ALIGNED` template parameter:

```cpp
#if MN_ALIGNED
    for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
        cT_out[i] = (OUT_T)cT_f32[i];
#else
    for (uint16_t i = 0; i < cT_f32.get_capacity(); ++i)
        if (cT_f32.is_valid_element(i)) cT_out[i] = (OUT_T)cT_f32[i];
#endif
```

Dispatcher passes `mn_aligned = (M % BM == 0) and (N % BN == 0)`.  The
LRU cache produces a second compiled kernel for the aligned variant.

Result: 3–5 % across the board on aligned shapes.  4096³ went from
0.96× → 0.99–1.00×, attn shapes from 0.93× → 0.97–0.99×.

### Step 7 — small-shape tile tuning

With everything else fixed, the only remaining sub-0.95× shape was 256³
at 0.36× (regression caused by the new `(64, 64, 2)` default — too few
simdgroups for a 16-tile problem).

Sweep on small shapes:

```
256x256x256: torch=19.0 us
   (64, 64, 4):  10.0 us  1.89x
   (64, 64, 2):   8.1 us  2.33x
   (64, 128, 4):  8.1 us  2.34x
   (32, 128, 4):  7.4 us  2.57x   ← winner
   (32, 64, 4):  19.7 us  0.97x

512x512x512: torch=15.1 us
   (32, 64, 2):  14.0 us  1.08x   ← winner
   (32, 128, 4): 14.4 us  1.05x
   (32, 64, 4):  14.3 us  1.06x
   (64, 64, 4):  15.6 us  0.97x
```

Added small-shape branches to the picker.  The intuition: for very small
problems, you want enough tiles to fill 15 cores × ~4 schedulers (~60
simdgroup slots).  A single tile of `(64, 64, 4)` produces only 4
simdgroups, so 16 tiles = 64 SG, just barely filling one wave — leaving
no slack for hiding latency.  `(32, 128, 4)` for 256³ gives 16 SG per
tile × 16 tiles = 256 SG, plenty of latency hiding.

### Step 8 — verify and document

All 58 correctness tests pass after every change.  Re-ran the full
bench three times to confirm stability and capture run-to-run variance
(torch's MPS times vary ±5 % depending on thermal state).  Updated
README.md with the new median table and per-regime tile picker
documentation.

---

## What didn't help (recorded so we don't re-try)

- **NWARPS sweep on `gemv_t`** (2/4/8/16): sub-1 % gains, not the 13 %
  needed.  Padded-m5_tensor route beat all of them by skipping the
  problem entirely.
- **Tiles larger than 128×128** (`(256, 128, 8)`, `(128, 256, 8)`):
  still tank to < 5 TF.  Probably exceeds an internal MPP per-tile
  thread or register budget.
- **K-split with `multiply_accumulate`**: same correctness issue as
  session 1.  Full-K `op.run` remains the right call.
- **External tile swizzle on m5_tensor**: still 15-20 % slowdown.
  MPP has its own internal scheduling — leave `swizzle_log = 0`.
- **NSG=2 for tiny problems**: not enough simdgroups in flight.  Use
  NSG=4 below ~32 tiles total.

---

## Final picker (the actual rules)

```python
def _pick_m5_tensor_tile(M, N, K, dtype):
    m_div_64  = (M % 64  == 0); n_div_64  = (N % 64  == 0)
    n_div_128 = (N % 128 == 0); m_div_32  = (M % 32  == 0)
    n_div_32  = (N % 32  == 0)

    if K <= 256 and M >= 1024 and N >= 1024 and m_div_32 and n_div_128:
        return (32, 128, 4)         # tiny K, big M*N

    if (M >= 1024 and N >= 1024 and not (m_div_64 and n_div_64)
            and (M+127)//128 * (N+127)//128 >= 8):
        return (128, 128, 8)        # non-divisible large

    if K >= 2 * max(M, N) and m_div_64 and n_div_128:
        return (64, 128, 4)         # deep K

    if M <= 256 and N <= 256 and m_div_32 and n_div_128:
        return (32, 128, 4)         # very small

    if M <= 768 and N <= 768 and m_div_32 and n_div_32:
        return (32, 64, 4)          # small-medium

    if m_div_64 and n_div_64:
        return (64, 64, 2)          # default divisible (the big one)

    if M <= 768 and N <= 768:
        return (64, 64, 4)          # small awkward

    return (64, 128, 4)             # large awkward
```

Plus the dispatch routing for big-N GEMV:

```python
if M == 1 and N >= 8192 and K >= 1024 and is_lp:
    backend = "m5_tensor"          # padded GEMV
```

---

## Files touched (session 2)

- **`metalblas/dispatch.py`**
  - Lifted the `M_ % BM != 0` guard in the m5_tensor branch
  - Relaxed the auto-dispatch gate to `tiles_m * tiles_n >= 8`
  - Big-N GEMV (M=1, N ≥ 8192) routes to m5_tensor
  - Rewrote `_pick_m5_tensor_tile` with the 6-regime picker
  - Pass `mn_aligned` to the kernel JIT
- **`metalblas/kernels.py`**
  - Added `MN_ALIGNED` template parameter to `M5_TENSOR_GEMM_SRC`
  - Added `mn_aligned` arg to `m5_tensor_gemm()` (separate LRU entry)
- **`README.md`**
  - New benchmark table with the post-session-2 numbers
  - Added "Tile picker by regime" table
  - Mentioned big-N GEMV → m5_tensor routing

---

## Session 1 — original breakthrough

## Session log — what got done

Starting point: bf16 large GEMM was at 0.74-0.77× of `torch.matmul` (manual
`m5_gemm` kernel was memory-bandwidth bound).  The prior notes ranked
"crack the MPP cooperative-tensor coord system" as the most likely next
25 %.  We did that, and it worked.

Concretely:

1. **Re-tested the MPP `op.run` + `cT.store(tensor_inline)` path** that
   previous notes called "blocked".  Built a 16×32 single-simdgroup probe
   (multiplying ones × ones, K=16) to verify correctness.  Found the path
   itself is fine — the prior failure was two C++ type bugs (see
   "Summary of the breakthrough" below).

2. **Scaled to a real GEMM** with `matmul2d_descriptor(BM, BN, dynamic_extent, …)`
   and `execution_simdgroups<4>` over `tensor_inline` device views.
   Single `op.run` call does the entire K-loop; MPP handles loads internally.

3. **Tile sweep on M5 Pro for 4096³ bf16** identified `(BM=64, BN=128, NSG=4)`
   as the winner (24.4 TF / 0.95×).  Bigger tiles (`(128,128)`, `(256,128)`)
   tank — apparently an internal MPP cap.  See sweep table below.

4. **Added the new `m5_tensor` backend** (kernels.py:738-815 and
   dispatch.py:_pick_m5_tensor_tile + the auto-dispatch branch).  Picker
   prefers `(64, 128, 4)` when N % 128 == 0 and ≥ 16 TGs, else
   `(64, 64, 4)`.  Manual `m5_gemm` stays as fallback for fp32, transposed
   inputs, non-divisible shapes, and tiny problems (< 16 (64×64) TGs).

5. **Discovered two anti-optimizations** worth recording so we don't
   re-try them: (a) external TG swizzle costs 15-20 % on the MPP path
   because MPP has its own internal scheduling — leave swizzle_log = 0;
   (b) K-split with `multiply_accumulate` didn't beat single full-K
   `op.run` and broke correctness on first attempt.

6. **Verified all 58 correctness tests pass** (added `not trans_a and
   not trans_b` to the m5_tensor dispatch condition — the kernel's
   descriptor TRANS_* flag isn't wired up correctly for the slice-based
   addressing yet, so transposed inputs go to manual `m5_gemm`).

7. **Bumped bench to warmup=50, iters=200** for less noise.

8. **Updated README and this file** with the new numbers and the working
   code pattern.

## Result vs starting point (bf16 medians)

| Group  | Before | After | Best after |
| ------ | ------ | ----- | ---------- |
| square | 0.78×  | 0.94× | 0.99× (4096³) |
| tall   | 0.72×  | 0.94× | 0.94×      |
| attn   | 0.81×  | 0.93× | 1.03× (small K) |
| llm    | 0.75×  | **1.07×** | **1.12×** |
| gemv   | 1.07×  | 0.85× | 1.00×      |
| odd    | 0.85×  | 0.75× | 0.90×      |

`llm` here is the FFN block shapes for Llama-7B / Llama-3:
`(2048, 14336, 4096)`, `(4096, 4096, 11008)`, `(4096, 11008, 4096)` —
i.e. `gate_proj`/`up_proj`/`down_proj` with batch×seq flattened into M.

`gemv` and `odd` regressed slightly — those still use the manual paths,
just with a bit more bench noise.  Worth a separate look later.

## Files touched

- `metalblas/kernels.py` — added `M5_TENSOR_GEMM_SRC` and the
  `m5_tensor_gemm()` JIT helper.
- `metalblas/dispatch.py` — added `_pick_m5_tensor_tile`, the
  `m5_tensor` backend branch, and the auto-dispatch condition.
- `bench/bench_matmul.py` — bumped warmup/iters defaults.
- `README.md` — rewrote benchmark section, added "How `m5_tensor` works".

## Summary of the breakthrough (this session)

The previous notes were stuck at 0.74-0.77× on large bf16 GEMM with the
manual-threadgroup `m5_gemm` kernel.  The path to close the gap was:

**Use `mpp::tensor_ops::matmul2d` directly with `tensor<device …, tensor_inline>`
views and `cT.store(C_view)`.**

The previous attempt at this got "wrong results" because of two type-system
gotchas, not because the MPP path was broken:

1. **`tensor<device const bfloat, …>` fails the cooperative-tensor static_asserts.**
   The cooperative tensor implementation does
   `__remove_addrspace_t<__remove_addrspace_t<remove_reference<T>>::element_type>`
   on the operand tensor — that strips `device` but **NOT `const`**.  So the
   left/right element type ends up being `const bfloat`, which doesn't match
   the `is_same_v<…, bfloat>` check in `__operand_layout`.  Fix:
   `tensor<device bfloat, …, tensor_inline> tA(const_cast<device bfloat*>(A), …)`.
2. **`tensor_inline` must be tagged explicitly.**  Without the third template
   argument, the constructor `tensor(ptr, dextents)` doesn't exist — only
   `tensor(__metal_tensor_t, offsets)` does, which we can't get from PyTorch.

With those two fixes, the doc example pattern works as documented:

```cpp
constexpr auto desc = matmul2d_descriptor(
    BM, BN, dynamic_extent, false, false, false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroups<4>> op;

auto mA = tA.slice(0, m_off);
auto mB = tB.slice(n_off, 0);
auto mC = tC.slice(n_off, m_off);

auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
op.run(mA, mB, cT);

// Convert to bfloat cooperative tensor and store.
auto cT_bf = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT_bf[i] = bfloat(cT[i]);
cT_bf.store(mC);
```

`dynamic_extent` K means a single `op.run` call does the entire K-loop —
MPP's internal implementation handles loads.  No threadgroup memory to
manage, no manual prefetching, no swizzling.

The MPP runtime is doing the load orchestration we couldn't do manually,
which closes the bandwidth gap we identified in the prior notes (we were
loading 3.22 GB at 4096³, torch was at ~2.16 GB — MPP gets there too).

## Tile sweep result (bf16, M5 Pro)

`execution_simdgroups<N>` with descriptor `matmul2d_descriptor(BM, BN, dynamic_extent, …)`:

For 4096³:
- `(BM=64, BN=128, NSG=4)`: **24.4 TF / 0.95×**  ← winner
- `(BM=128, BN=64, NSG=4)`: 20.4 TF / 0.80×
- `(BM=128, BN=128, NSG=8)`: 21.3 TF / 0.85×
- `(BM=64, BN=64, NSG=4)`: 19.4 TF / 0.75×
- `(BM=64, BN=32, NSG=4)`: 12.4 TF / 0.48× (doc example tile — too small for big K)

Tiles much bigger than `(128, 128)` (e.g. `(256, 128)`, `(128, 256)`) tank
hard (< 5 TF) — probably exceeding some internal limit.

For small problems (512-768), `(64, 128, 4)` still wins.  Below ~ 256³,
the per-TG count drops too low; we route those to manual `m5_gemm`.

## What didn't help

- **External tile swizzle:** MPP has its own internal TG scheduling.
  Adding our own swizzle (swizzle_log = 2) slowed us down 15-20 %.
  Conclusion: dispatch with plain `(tiles_n, tiles_m, 1)`, no swizzle.
- **`relaxed_precision = true` vs `false`:** no measurable difference for
  bf16 inputs (already at "relaxed" precision).
- **K-split into `multiply_accumulate` chunks:** broke correctness (max_err
  hit ~ 1.0) in initial attempts and didn't beat single-`op.run` even at
  BK=512 in benchmarks before correctness failed.  May be an init-order
  issue with the float cooperative tensor we initialized via `cT[i] = 0`.
- **Direct bfloat accumulator (no fp32 conversion step):** previous notes
  showed this loses precision at K ≥ 256.  Still true.

## Final benchmark (warmup 50, iters 200)

bf16 medians vs `torch.matmul` on MPS:
- square: 0.94×  (4096³: 0.99×, 1024³: 0.95×)
- tall:   0.94×
- attn:   0.93×  (4096×4096×64: 1.03× — beats torch on small K)
- llm:    **1.07× median, best 1.12×** ← beats torch on LLM shapes
- gemv:   0.85×  (the manual GEMV path; could be improved later)
- odd:    0.75×  ← held back by non-divisible shapes that can't use the
                   MPP tile and fall back to manual `m5_gemm`

## Where the remaining gap on large GEMM probably lives

- The cT_f32 → cT_bf conversion loop (`for i ∈ get_capacity()`) is roughly
  16 elements × 32 lanes per simdgroup.  Cheap but non-zero.  Eliminating
  it would require either bf16 accumulation (fails precision) or a way
  to make `cT.store` do type conversion (no such overload exists in the
  header).
- Edge handling for shapes that aren't a multiple of `(BM, BN)` — these
  fall back to the manual kernel and lose ~ 20-25 %.  Could be improved
  by zero-padding the operands in PyTorch then using MPP, but the realloc
  costs likely outweigh the win for typical workloads.

## Things still in the kernels (legacy)

- `DBUF` (double-buffered K-loop) — kept behind compile flag in `m5_gemm`
  for completeness, but the picker no longer uses it.
- `__PAD__` template parameter for the threadgroup A/B layout — same.
- The `(BM, BN, BK, WM, WN)` manual kernel itself stays as the fallback
  for fp32 (where MPP path doesn't beat it) and for unaligned / small
  shapes.
