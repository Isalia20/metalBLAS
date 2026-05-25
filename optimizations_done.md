# bf16 GEMM optimization notes

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
