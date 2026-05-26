# Apple M5 Pro

## Machine

- Chip: Apple M5 Pro
- macOS: 26.4.1 (build 25E253)
- GPU: 16 cores, Metal 4
- RAM: 24 GB
- PyTorch: 2.12.0
- Branch: `optimize-older-macs`
- Smoke test (`tests/test_basic.py`): 58/58 `[OK]`, 0 `[FAIL]`

## Methodology

- dtype: bf16 only (the most-used dtype; matches the M2/M3-Ultra/M4 reports)
- `bench/bench_matmul.py --dtype bf16`, default timing config — per shape, per kernel (torch, metalblas):
  - 50 warmup iterations
  - 10 outer trials, each = 100 kernel calls followed by `torch.mps.synchronize()` and a wall-clock measurement
  - Reported per-call time = `min(trial_total) / 100`
- Two full runs (`--cool 0`); the 50-iter warmup absorbs the thermal ramp. Numbers below are run 1; run-to-run deltas are called out where they matter.

## Results — bf16

| group  | median | min   | max   | notable |
| ------ | ------ | ----- | ----- | ------- |
| square | 1.01×  | 0.96× | 2.21× | 4096³ at 1.00× / 25.5 TF (**no M4-style regression**); 512³ at 0.96–0.97× (the one consistent loss); 128³/256³ swing 1.1–2.2× (launch-overhead floor) |
| tall   | 0.99×  | 0.99× | 1.00× | dead even at ~25 TF, including 2048×2048×8192 |
| attn   | 1.00×  | 0.99× | 1.02× | parity; thin-K 4096×4096×128 the only win |
| gemv   | 1.07×  | 1.01× | 2.25× | **1×4096×4096 = 1.07× (no M3 regression — this is the M5-path target chip)**; 1×1024×1024 1.9–2.25× |
| llm    | 1.00×  | 0.97× | 1.02× | parity within thermal jitter on the ~10–16 ms shapes |
| odd    | 1.15×  | 1.01× | 1.51× | 257³ at 1.51×; 4097³ 1.15× and 1023³ 1.16× (torch's MPS drops to ~18 TF on non-aligned squares) |

### Full bf16 table (run 1)

```
=== square ===
    M     N     K |  torch ms  TFLOPS |     mb ms  TFLOPS | speedup
  128   128   128 |     0.015    0.29 |     0.013    0.31 |  1.09x
  256   256   256 |     0.013    2.55 |     0.006    5.25 |  2.06x
  512   512   512 |     0.015   17.88 |     0.015   17.42 |  0.97x
 1024  1024  1024 |     0.091   23.61 |     0.089   24.03 |  1.02x
 2048  2048  2048 |     0.673   25.51 |     0.688   24.97 |  0.98x
 4096  4096  4096 |     5.379   25.55 |     5.376   25.57 |  1.00x

=== tall ===
 4096  1024  1024 |     0.338   25.38 |     0.340   25.23 |  0.99x
 1024  4096  1024 |     0.339   25.34 |     0.338   25.41 |  1.00x
 8192  1024  1024 |     0.664   25.86 |     0.670   25.66 |  0.99x
 2048  2048  8192 |     2.762   24.88 |     2.782   24.70 |  0.99x

=== attn ===
 4096  4096    64 |     0.146   14.70 |     0.146   14.66 |  1.00x
 4096  4096   128 |     0.178   24.11 |     0.175   24.48 |  1.02x
 4096  4096  1024 |     1.318   26.06 |     1.328   25.87 |  0.99x

=== gemv ===
    1  4096  4096 |     0.117    0.29 |     0.109    0.31 |  1.07x
 4096     1  4096 |     0.115    0.29 |     0.113    0.30 |  1.01x
    1  1024  1024 |     0.016    0.13 |     0.007    0.29 |  2.25x
    1 32000  4096 |     0.984    0.27 |     0.925    0.28 |  1.06x

=== llm ===
 2048 14336  4096 |     9.257   25.98 |     9.335   25.77 |  0.99x
 4096  4096 11008 |    15.827   23.34 |    15.852   23.30 |  1.00x
 4096 11008  4096 |    15.593   23.69 |    15.237   24.24 |  1.02x

=== odd ===
  257   257   257 |     0.013    2.61 |     0.009    3.96 |  1.51x
 1023  1023  1023 |     0.114   18.73 |     0.099   21.65 |  1.16x
 4097  4097  4097 |     7.591   18.12 |     6.611   20.81 |  1.15x
  511   511   511 |     0.017   15.39 |     0.017   15.55 |  1.01x
  333   444   555 |     0.014   11.40 |     0.014   11.89 |  1.04x
```

## Shapes below 1.0×

| shape (M×N×K) | torch ms | mb ms  | ratio | note |
| ------------- | -------- | ------ | ----- | ---- |
| 512×512×512 (square)     | 0.015 | 0.015 | 0.96–0.97× | the only reproducible loss; ~3–4% at the bf16 ceiling (17.4 vs 17.9 TF). Same shape is 0.96–0.97× on M3 Ultra and M4 too. |
| 2048×2048×8192 (tall)    | 2.762 | 2.782 | 0.99× | deep-K square, within noise here (was 0.94× on M3 Ultra) |
| 8192×1024×1024 (tall)    | 0.664 | 0.670 | 0.99× | ~1%, noise floor at 25.7 TF |
| 4096×1024×1024 (tall)    | 0.338 | 0.340 | 0.99× | ~1%; flips to 1.00× in run 2 |
| 4096×4096×1024 (attn)    | 1.318 | 1.328 | 0.99× | ~1% at 25.9 TF |
| 2048×14336×4096 (llm)    | 9.257 | 9.335 | 0.99× | parity; dropped to 0.97× in run 2 as the chassis warmed (torch 9.74 / mb 10.00) — thermal jitter on a ~10 ms shape, not a kernel loss |
| 2048×2048×2048 (square)  | 0.673 | 0.688 | 0.98× | borderline; 1.00× in run 2 |

## Notes

- **This is the chip the kernels were originally tuned on** (`main` is "tuned on M5 Pro" per `reproduce.md`). `optimize-older-macs` preserves that: every group is parity-or-win, and the only shape that consistently sits below 0.97× is 512³. This is the cleanest of the four perf reports — expected, since the pre-M5 GEMV/`gemv_nt` paths added on this branch are gated behind `_HAS_TENSOR_UNIT` and don't fire here, so M5 Pro takes the same code path as `main`.
- **`1×4096×4096` bf16 GEMV = 1.07×** — no M3-style regression. On the M3 Ultra this shape collapses to ~0.46× on `main` (the bug `optimize-older-macs` exists to fix), but M5 Pro is the target the `(VEC=8, NWARPS=8)` heuristic was tuned for, so it lands the M5 path correctly. Rock-solid across both runs (1.07× / 1.07×).
- **`square 4096³` is parity at 25.5 TF** — the 0.94× large-square regression seen on the base M4 does **not** appear here (matches M3 Ultra).
- **`odd` is the strongest group (median 1.15×).** torch's MPS GEMM drops to ~18 TF on non-power-of-two squares (4097³, 1023³) while `m5_tensor` holds ~20–22 TF, a structural ~15% win on misaligned shapes.
- **Tiny-shape ratios (128³, 256³, 1×1024×1024) are launch-overhead-dominated and noisy** — they swing 1.6×–2.25× between the two runs because torch and mb are both sub-15 µs there. Real win (mb has lower per-call dispatch overhead) but don't read the exact multiplier as stable.
- **llm group is parity within thermal noise.** The three ~10–16 ms shapes read 0.97–1.02× and shift run-to-run with chassis temperature; both `m5_tensor` and torch's MPS GEMM are at the same ~23–26 TF compute ceiling here, so neither has a structural edge.
