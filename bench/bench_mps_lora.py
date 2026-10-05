"""Validate the recovered LoRA schedule against MPSGraph, without selecting by timing."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.bench_mps_a16 import elapsed
from metalblas.mps_a14 import vector_preferred
from metalblas.mps_heuristics import device_info
from metalblas.mps_lora import _launch_plan

SHAPES = ([(128, n, k) for n in range(2, 17) for k in (1024, 8192, 8193)]
          + [(32, n, k) for n in (2, 8, 16) for k in (17, 4096, 16384)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.trials) <= 0:
        parser.error("iterations and trials must be positive")
    dev = device_info()
    major = int(platform.mac_ver()[0].split('.')[0])
    paths = ("metalblas/shaders/mps_lora.h", "metalblas/mps_lora.py", "metalblas/mps_a14.py")
    result = {"device": dataclasses.asdict(dev), "macos": platform.mac_ver()[0],
              "torch": torch.__version__, "iterations": args.iterations, "trials": args.trials,
              "method": "median of interleaved synchronized wall-clock trials; prepared launches; preallocated outputs",
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
              "cases": [], "skipped": []}
    torch.manual_seed(78)
    for m, n, k in SHAPES:
        if not vector_preferred(m, n, k, torch.float16, dev=dev, trans_b=True, macos_major=major):
            result["skipped"].append({"shape": [m, n, k], "reason": "Source routes to GEMM"})
            continue
        a = torch.randn((m, k), device="mps", dtype=torch.float16)
        b = torch.randn((n, k), device="mps", dtype=torch.float16).t()
        out = torch.empty((m, n), device="mps", dtype=a.dtype)
        ref = torch.empty_like(out)
        run, plan = _launch_plan(m, n, k, k, k, n, k % 4 == 0, dev, major)
        own = lambda: run(a, b, out)
        apple = lambda: torch.mm(a, b, out=ref)
        for _ in range(10):
            own()
            apple()
        error = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=1e-3)
        times = {"torch": [], "lora": []}
        for trial in range(args.trials):
            pair = (("torch", apple), ("lora", own))
            for label, fn in pair[::1 if trial % 2 == 0 else -1]:
                times[label].append(elapsed(fn, args.iterations))
        t, v = (statistics.median(times[label]) for label in ("torch", "lora"))
        result["cases"].append({"dtype": "f16", "layout": "nt", "shape": [m, n, k],
                                 "plan": dataclasses.asdict(plan), "max_error": error,
                                 "torch_us": t, "lora_us": v, "speedup": t / v,
                                 "trials_us": times})
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"{m}x{n}x{k}: MPS {t:.1f} us, LoRA {v:.1f} us, {t/v:.3f}x, error {error:g}", flush=True)
    Path(args.json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
