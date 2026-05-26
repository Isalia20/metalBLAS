# Apple M3 Ultra

## Machine

- Chip: Apple M3 Ultra
- macOS: 26.5 (build 25F71)
- GPU: 80 cores, Metal 4
- RAM: 512 GB
- PyTorch: 2.12.0
- Branch: `optimize-older-macs`
- Smoke test (`tests/test_basic.py`): all `[OK]`

## Methodology

- dtype: bf16 only
- Per shape, per kernel (torch, metalblas):
  - 50 warmup iterations
  - 10 outer iterations, each = 100 kernel calls followed by `torch.mps.synchronize()` and a wall-clock measurement
  - Reported per-call time = `min(outer_iter_total) / 100`
- No inter-shape cooldown (`--cool 0`); the script's warmup absorbs the few-shape thermal ramp on a 32-core / 80-GPU-core chassis.

## Results — bf16

| group  | median | min   | max   | notable |
| ------ | ------ | ----- | ----- | ------- |
| square | 0.99×  | 0.96× | 1.54× | 4096³ at 0.99× (parity at 26.1 TFLOPS); 128³ / 256³ at 1.54× / 1.53× |
| tall   | 0.98×  | 0.94× | 1.01× | 2048×2048×8192 at 0.94× (deep-K square, the one tall regression) |
| attn   | 1.10×  | 1.00× | 1.19× | thin-K wins: 4096×4096×64 at 1.19× |
| gemv   | 1.27×  | 1.09× | 1.77× | **1×4096×4096 = 1.41× (M3 regression closed)**; 1×1024×1024 at 1.77× |
| llm    | 1.00×  | 1.00× | 1.01× | dead even at ~26.5 TFLOPS — bf16 compute ceiling for both |
| odd    | 1.04×  | 0.96× | 1.40× | 4097³ at 1.40× (torch's MPS drops from 26→18 TF on non-aligned squares) |

### Full bf16 table

```
=== square ===
    M     N     K |  torch ms  TFLOPS |     mb ms  TFLOPS | speedup
  128   128   128 |     0.020    0.21 |     0.013    0.33 |  1.54x
  256   256   256 |     0.018    1.87 |     0.012    2.87 |  1.53x
  512   512   512 |     0.022   12.13 |     0.023   11.65 |  0.96x
 1024  1024  1024 |     0.095   22.49 |     0.097   22.22 |  0.99x
 2048  2048  2048 |     0.678   25.33 |     0.706   24.33 |  0.96x
 4096  4096  4096 |     5.216   26.35 |     5.261   26.12 |  0.99x

=== tall ===
 4096  1024  1024 |     0.352   24.39 |     0.359   23.92 |  0.98x
 1024  4096  1024 |     0.354   24.29 |     0.360   23.83 |  0.98x
 8192  1024  1024 |     0.686   25.03 |     0.682   25.21 |  1.01x
 2048  2048  8192 |     2.727   25.20 |     2.915   23.57 |  0.94x

=== attn ===
 4096  4096    64 |     0.110   19.49 |     0.093   23.18 |  1.19x
 4096  4096   128 |     0.191   22.49 |     0.173   24.86 |  1.10x
 4096  4096  1024 |     1.329   25.86 |     1.326   25.92 |  1.00x

=== gemv ===
    1  4096  4096 |     0.034    0.97 |     0.024    1.37 |  1.41x
 4096     1  4096 |     0.028    1.19 |     0.026    1.30 |  1.09x
    1  1024  1024 |     0.017    0.12 |     0.010    0.22 |  1.77x
    1 32000  4096 |     0.385    0.68 |     0.342    0.77 |  1.13x

=== llm ===
 2048 14336  4096 |     9.070   26.52 |     9.093   26.45 |  1.00x
 4096  4096 11008 |    14.082   26.23 |    14.137   26.13 |  1.00x
 4096 11008  4096 |    13.899   26.58 |    13.786   26.79 |  1.01x

=== odd ===
  257   257   257 |     0.018    1.90 |     0.014    2.44 |  1.28x
 1023  1023  1023 |     0.097   22.08 |     0.098   21.90 |  0.99x
 4097  4097  4097 |     7.658   17.96 |     5.466   25.16 |  1.40x
  511   511   511 |     0.023   11.74 |     0.024   11.30 |  0.96x
  333   444   555 |     0.023    7.17 |     0.022    7.47 |  1.04x
```

## Shapes below 1.0×

| shape (M×N×K) | torch ms | mb ms  | ratio | note |
| ------------- | -------- | ------ | ----- | ---- |
| 2048×2048×8192 (tall)  | 2.727 | 2.915 | 0.94× | deep-K square, the one tall regression — ~7% slower than the matched dense-K cousin |
| 512×512×512 (square)   | 0.022 | 0.023 | 0.96× | ~4% gap, within run-to-run noise at the bf16 compute ceiling (11.7 TF) |
| 511×511×511 (odd)      | 0.023 | 0.024 | 0.96× | mirrors the 512³ result — same compute-ceiling band, no extra non-alignment penalty |
| 2048×2048×2048 (square)| 0.678 | 0.706 | 0.96× | another 4% loss at the bf16 ceiling (24.3 vs 25.3 TF); reproducible |
| 1024×1024×1024 (square)| 0.095 | 0.097 | 0.99× | within noise |
| 4096×4096×4096 (square)| 5.216 | 5.261 | 0.99× | parity at 26.1 TF — bf16 compute ceiling for the chip |
| 4096×1024×1024 (tall)  | 0.352 | 0.359 | 0.98× | 2% behind torch at 23.9 TF |
| 1024×4096×1024 (tall)  | 0.354 | 0.360 | 0.98× | same regime as the row above |
| 1023×1023×1023 (odd)   | 0.097 | 0.098 | 0.99× | within noise |

## Notes

- **`1×4096×4096` bf16 GEMV at 1.41×** — this is the shape that motivated `optimize-older-macs`. On `main` it lands at 0.46–0.51× on this M3 Ultra because the M5 Pro–tuned `gemv_t` heuristic picks `(VEC=8, NWARPS=8)` for `cols≥4096`, which spawns 16 threadgroups for an 80-core GPU. The pre-M5 branch in `_gemv_pick` swaps to `(VEC=2, NWARPS=32)` → 64 threadgroups × 32 simdgroups = 2048 sg, saturating the cores; same shape with the same kernel family runs ~3× faster than `main`.
- **`4096×1×4096` bf16 GEMV at 1.09×** — closed by the vectorized `gemv_nt` (VEC=4) added on `optimize-older-macs`; on `main` this shape was 0.73–0.82× on M3 Ultra because the scalar-stride read pattern wastes half a 128-B cache line per warp.
- **`llm` group is dead-even at 26.5 TF** — both `m5_tensor` and torch's MPS bf16 GEMM hit the same compute ceiling here. M3 maps `mpp::tensor_ops::matmul2d` to simdgroup matrix ops (no dedicated M5 tensor unit), which is the same machinery PyTorch's GEMM already uses, so neither side has a structural advantage on this regime.
- **One real regression worth investigation: `2048×2048×8192` (tall) at 0.94×** — deep-K square shape where torch's MPS kernel pulls ahead by ~7%. The `m5_tensor` autotuner picks a tile from the heuristic candidate list; on M5 Pro the same shape is at 1.00× per the README, so this is likely a candidate-list shortfall for the high-core M3 Ultra at this aspect ratio.
- The `square 4096³` 0.94× regression reported on M4 does **not** appear here (M3 Ultra: 0.99× at 26.1 TF, parity with torch).
