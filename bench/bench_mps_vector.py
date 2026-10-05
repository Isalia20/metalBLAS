"""Compare recovered ordinary vector schedules with traced MPSGraph launches.

Write the candidate cases with --write-cases, capture them with
tools/mps_trace_sweep.py, then use --trace and --json to measure. Timings
validate this implementation and never select or modify its schedules.
"""
import argparse
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.bench_mps_a16 import DTYPES, elapsed
from metalblas.dispatch import _resolve_inputs
from metalblas.mps_heuristics import device_info
from metalblas.mps_vector import _launch_plan
from tools.mps_check_sweep import check_sweep, dispatches

SHAPES = [(w, r, k) for w, r, k in
          ((129, 1, 255), (4096, 1, 4096), (129, 4, 257), (4096, 4, 4096),
           (129, 7, 2048), (129, 8, 2176), (129, 12, 8192),
           (129, 15, 8193), (129, 16, 1024))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--write-cases", help="Write candidate cases for a separate dispatch capture and exit")
    source.add_argument("--trace", help="Dispatch trace captured on this machine")
    parser.add_argument("--json")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()
    if args.write_cases:
        cases = [[d, layout, m, n, k] for d in DTYPES for layout in ("nn", "nt", "tn", "tt")
                 for w, r, k in SHAPES for m, n in ((w, r), (r, w))]
        Path(args.write_cases).write_text(json.dumps(cases) + "\n")
        return
    if not args.json or min(args.iterations, args.trials) <= 0:
        parser.error("--json and positive iterations/trials are required for a comparison")
    dev = device_info()
    major = int(platform.mac_ver()[0].split('.')[0])
    raw = Path(args.trace).read_bytes()
    trace = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    if trace["device"] != dataclasses.asdict(dev):
        parser.error("Trace device does not match this GPU")
    mismatches = check_sweep(trace, major)["mismatches"]
    if mismatches:
        parser.error(f"Source selection disagrees with trace: {mismatches[:1]}")
    paths = ("metalblas/shaders/mps_vector.h", "metalblas/mps_vector.py", "metalblas/mps_a14.py")
    result = {"device": dataclasses.asdict(dev), "macos": platform.mac_ver()[0],
              "torch": torch.__version__, "iterations": args.iterations, "trials": args.trials,
              "method": "median of interleaved synchronized wall-clock trials; prepared launches; preallocated outputs",
              "trace_sha256": hashlib.sha256(raw).hexdigest(),
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
              "cases": [], "skipped": []}
    torch.manual_seed(79)
    for row in trace["rows"]:
        name, layout, m, n, k = (row[key] for key in ("dtype", "layout", "M", "N", "K"))
        ds = dispatches(row)
        if (row.get("batch", 1) != 1 or len(ds) != 1
                or not ds[0]["kernel"].startswith("NDArrayVectorMatrixMultiply_")):
            result["skipped"].append({"dtype": name, "layout": layout, "shape": [m, n, k],
                                      "kernels": [d["kernel"] for d in ds]})
            continue
        dtype = DTYPES[name]
        a = torch.randn((k, m) if layout[0] == "t" else (m, k), device="mps", dtype=dtype)
        b = torch.randn((n, k) if layout[1] == "t" else (k, n), device="mps", dtype=dtype)
        if layout[0] == "t": a = a.t()
        if layout[1] == "t": b = b.t()
        a, b, _, _, _, lda, ldb, ta, tb = _resolve_inputs(a, b)
        if a.is_contiguous(): ta, lda = False, k
        if b.is_contiguous(): tb, ldb = False, n
        out = torch.empty((m, n), device="mps", dtype=dtype)
        ref = torch.empty_like(out)
        run, plan = _launch_plan(dtype, m, n, k, lda, ldb, n, ta, tb, 1, 0, 0, 0, dev, major)
        own = lambda: run(a, b, out)
        apple = lambda: torch.mm(a, b, out=ref)
        for _ in range(10):
            own()
            apple()
        error = (out.float() - ref.float()).abs().max().item()
        atol, rtol = {"f32": (2e-4, 2e-5), "f16": (2e-3, 1e-3), "bf16": (2e-2, 8e-3)}[name]
        torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
        times = {"torch": [], "vector": []}
        for trial in range(args.trials):
            pair = (("torch", apple), ("vector", own))
            for label, fn in pair[::1 if trial % 2 == 0 else -1]:
                times[label].append(elapsed(fn, args.iterations))
        t, v = (statistics.median(times[label]) for label in ("torch", "vector"))
        result["cases"].append({"dtype": name, "layout": layout, "shape": [m, n, k],
                                 "plan": dataclasses.asdict(plan), "max_error": error,
                                 "reference_kernel": ds[0]["kernel"],
                                 "torch_us": t, "vector_us": v, "speedup": t / v,
                                 "trials_us": times})
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name} {layout} {m}x{n}x{k}: MPS {t:.1f} us, vector {v:.1f} us, {t/v:.3f}x, error {error:g}", flush=True)
    Path(args.json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
