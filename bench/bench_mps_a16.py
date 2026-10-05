"""Validate recovered A16/NNA14 schedules against MPSGraph; never select by timing.

Measures prepared kernel launches with preallocated outputs. This excludes Python
planning overhead and only covers GEMM shapes, not the separate vector families.
M1/M2 use NNA14; M3/M4 use A16 for all three dtypes; M5 uses A16 for float32.
"""
import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metalblas import mps_a16, mps_nna14
from metalblas.mps_heuristics import device_info
from metalblas.dispatch import _resolve_inputs

SHAPES = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
          (4096, 1024, 1024), (128, 4096, 4096), (4096, 128, 4096), (64, 64, 8192),
          (257, 257, 257), (1023, 1023, 1023), (333, 444, 555)]
DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def elapsed(fn, iterations):
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--dtypes", help="Comma-separated f32,f16,bf16; defaults to the device's recovered GEMM dtypes")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--baseline-root", type=Path,
                        help="Optional isolated package snapshot for a prepared-launch comparison")
    args = parser.parse_args()
    if min(args.iterations, args.trials) <= 0:
        parser.error("iterations and trials must be positive")
    dev = device_info()
    family = "nna14" if dev.generation < 18 else "a16"
    _launch_plan = (mps_nna14 if family == "nna14" else mps_a16)._launch_plan
    names = args.dtypes.split(",") if args.dtypes else (["f32"] if dev.generation >= 22 else list(DTYPES))
    if any(name not in DTYPES for name in names):
        parser.error("Unknown dtype")
    paths = (f"metalblas/shaders/{family}_gemm.h", f"metalblas/mps_{family}.py", "metalblas/mps_a14.py")
    result = {"device": dataclasses.asdict(dev), "macos": platform.mac_ver()[0], "family": family,
              "torch": torch.__version__, "iterations": args.iterations, "trials": args.trials,
              "method": "median of interleaved synchronized wall-clock trials; prepared launches; preallocated outputs",
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
              "cases": []}
    previous_launch = None
    if args.baseline_root is not None:
        from bench.bench_mps_dispatch import load_baseline, source_hashes
        previous = load_baseline(args.baseline_root, autotune=False)
        previous_launch = importlib.import_module(f"{previous.__name__}.mps_{family}")._launch_plan
        result["baseline_root"] = str(args.baseline_root.resolve())
        result["baseline_autotune"] = False
        result["baseline_source_sha256"] = source_hashes(args.baseline_root)
    torch.manual_seed(77)
    for name in names:
        dtype = DTYPES[name]
        for layout in ("nn", "nt", "tn", "tt"):
            for m, n, k in SHAPES:
                a = torch.randn((k, m) if layout[0] == "t" else (m, k), device="mps", dtype=dtype)
                b = torch.randn((n, k) if layout[1] == "t" else (k, n), device="mps", dtype=dtype)
                if layout[0] == "t":
                    a = a.t()
                if layout[1] == "t":
                    b = b.t()
                a, b, _, _, _, lda, ldb, ta, tb = _resolve_inputs(a, b)
                out = torch.empty((m, n), device="mps", dtype=dtype)
                ref = torch.empty_like(out)
                run, plan = _launch_plan(dtype, m, n, k, lda, ldb, n, ta, tb,
                                         lda % 4 == 0, ldb % 2 == 0, 1, 0, 0, 0, dev, None)
                own = lambda: run(a, b, out)
                apple = lambda: torch.mm(a, b, out=ref)
                functions = {"torch": apple, family: own}
                if previous_launch is not None:
                    prior_out = torch.empty_like(out)
                    prior_run, prior_plan = previous_launch(
                        dtype, m, n, k, lda, ldb, n, ta, tb, lda % 4 == 0, ldb % 2 == 0,
                        1, 0, 0, 0, dev, None)
                    if dataclasses.asdict(prior_plan) != dataclasses.asdict(plan):
                        raise ValueError("The baseline selected a different schedule")
                    functions["previous"] = lambda: prior_run(a, b, prior_out)
                for _ in range(10):
                    for fn in functions.values():
                        fn()
                error = (out.float() - ref.float()).abs().max().item()
                previous_error = ((out.float() - prior_out.float()).abs().max().item()
                                  if previous_launch is not None else None)
                times = {name: [] for name in functions}
                for trial in range(args.trials):
                    pair = list(functions.items())
                    if previous_launch is not None:
                        pair = pair[trial % 3:] + pair[:trial % 3]
                    for label, fn in pair[::1 if trial % 2 == 0 else -1]:
                        times[label].append(elapsed(fn, args.iterations))
                t, v = (statistics.median(times[label]) for label in ("torch", family))
                row = {"dtype": name, "layout": layout, "shape": [m, n, k],
                                         "plan": dataclasses.asdict(plan), "max_error": error,
                                         "torch_us": t, f"{family}_us": v, "speedup": t / v,
                                         "trials_us": times}
                if previous_launch is not None:
                    row["previous_us"] = statistics.median(times["previous"])
                    row["speedup_previous"] = row["previous_us"] / v
                    row["max_error_previous"] = previous_error
                result["cases"].append(row)
                Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
                print(f"{name} {layout} {m}x{n}x{k}: MPS {t:.1f} us, {family.upper()} {v:.1f} us, "
                      f"{t/v:.3f}x, error {error:g}", flush=True)


if __name__ == "__main__":
    main()
