"""Probe what MPSGraph dispatches for matmul on this Mac and check the port.

Runs ``torch.matmul`` over a shape sweep with MPS's own parameter logging
enabled (``MPS_LOG_INFO=1 MPS_LOG_AUTO_TUNE=1``), parses the
``MatMul A18 kernel running key with params`` lines that MPSNDArray prints,
and compares them with ``metalblas.mps_heuristics.select_params`` for the
detected device.  Use it on a new GPU (for example a 60+ core Ultra) to
confirm the heuristic port before trusting it there.

    python tools/mps_probe.py            # default sweep
    python tools/mps_probe.py --quick    # a few dozen shapes
    python tools/mps_probe.py --json out.json

Shapes that MPS routes to its vector-matrix kernels print no parameter line;
those are checked against ``vector_kernel_preferred`` instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DIMS = [1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 100, 127, 128, 129, 192, 200,
        255, 256, 257, 384, 500, 512, 513, 768, 1000, 1024, 1025, 1536, 2000, 2048, 2049, 3000,
        4096, 4097, 8192]
KS = [16, 32, 48, 64, 65, 96, 128, 192, 256, 500, 512, 768, 1024, 2048, 4096, 8192, 16384, 65536]

CHILD = r'''
import sys, torch
dts = {"f16": torch.float16, "bf16": torch.bfloat16}
for line in sys.stdin:
    dt, lay, M, N, K = line.split()
    M, N, K = int(M), int(N), int(K)
    t = dts[dt]
    a = torch.randn(M, K, device="mps", dtype=t) if lay[0] == "n" else torch.randn(K, M, device="mps", dtype=t).t()
    b = torch.randn(K, N, device="mps", dtype=t) if lay[1] == "n" else torch.randn(N, K, device="mps", dtype=t).t()
    torch.mps.synchronize()
    print(f"CASE {dt} {lay} {M} {N} {K}", file=sys.stderr, flush=True)
    c = a @ b
    torch.mps.synchronize()
    del a, b, c
print("DONE", file=sys.stderr, flush=True)
'''


def default_cases(quick: bool):
    cases = []
    if quick:
        for (M, N, K) in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192), (256, 256, 256),
                          (128, 128, 128), (64, 64, 64), (16, 16, 16), (2048, 14336, 4096), (4097, 4097, 4097),
                          (333, 444, 555), (1, 4096, 4096), (8, 4096, 4096), (16, 4096, 4096), (64, 4096, 4096),
                          (4096, 8, 4096), (4096, 16, 4096), (4096, 32, 4096), (4096, 64, 4096), (4096, 128, 4096),
                          (128, 128, 16384), (512, 512, 16384), (256, 256, 65536), (64, 64, 65536),
                          (96, 4096, 32768), (1024, 1024, 8192), (2000, 768, 2304)]:
            for lay in ("nn", "nt", "tn", "tt"):
                cases.append(("bf16", lay, M, N, K))
        return cases
    for M in DIMS:
        for N in DIMS:
            cases.append(("bf16", "nn", M, N, 1024))
    for K in KS:
        for (M, N) in [(1024, 1024), (4096, 4096), (128, 128), (64, 64), (16, 4096), (4096, 16), (1, 4096),
                       (4096, 64), (2048, 2048), (300, 700), (8192, 8192), (512, 512), (256, 256)]:
            cases.append(("bf16", "nn", M, N, K))
    for lay in ("nt", "tn", "tt"):
        for (M, N, K) in [(1024, 1024, 1024), (2000, 768, 2304), (4096, 4096, 4096), (8, 4096, 4096),
                          (1, 4096, 4096), (16, 4096, 4096), (32, 4096, 4096), (4096, 32, 4096),
                          (4096, 128, 4096), (128, 128, 16384), (512, 512, 512), (64, 64, 64), (200, 300, 400),
                          (4097, 4097, 4097), (2048, 14336, 4096), (1024, 1024, 8192)]:
            cases.append(("bf16", lay, M, N, K))
    for (M, N, K) in [(1024, 1024, 1024), (4096, 4096, 4096), (16, 4096, 4096), (4096, 16, 4096),
                      (200, 300, 400), (4097, 4097, 4097)]:
        cases.append(("f16", "nn", M, N, K))
    seen, out = set(), []
    for c in cases:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def run_probe(cases):
    env = dict(os.environ, MPS_LOG_INFO="1", MPS_LOG_AUTO_TUNE="1")
    proc = subprocess.run([sys.executable, "-c", CHILD], input="\n".join(" ".join(map(str, c)) for c in cases) + "\n",
                          capture_output=True, text=True, env=env)
    rows, cur = [], None
    for line in proc.stderr.splitlines():
        if line.startswith("CASE "):
            _, dt, lay, M, N, K = line.split()
            cur = {"dtype": dt, "layout": lay, "M": int(M), "N": int(N), "K": int(K), "key": None, "params": None}
            rows.append(cur)
        elif "running key with params" in line and cur is not None and cur["params"] is None:
            m = re.search(r"\{\{([^}]*)\}, \{([^}]*)\}\}", line)
            cur["key"] = [int(x) for x in m.group(1).split(",")]
            cur["params"] = [int(x) for x in m.group(2).split(",")]
    if not rows or "DONE" not in proc.stderr:
        sys.exit("probe child failed:\n" + proc.stderr[-3000:])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", help="write raw observations here")
    args = ap.parse_args()

    import torch
    from metalblas import mps_heuristics as mh

    dev = mh.device_info()
    print(f"device: {dev.chip}, {dev.cores} GPU cores, MPS generation {dev.generation}, table {dev.table}")
    cases = default_cases(args.quick)
    print(f"probing {len(cases)} shapes through torch.matmul ...", flush=True)
    rows = run_probe(cases)
    if args.json:
        json.dump(rows, open(args.json, "w"))

    dts = {"f16": torch.float16, "bf16": torch.bfloat16}
    ok = bad = vok = vbad = 0
    mism = []
    for r in rows:
        ta, tb = r["layout"][0] == "t", r["layout"][1] == "t"
        dtype = dts[r["dtype"]]
        vec_pred = mh.vector_kernel_preferred(r["M"], r["N"], r["K"], dtype, dev, ta, tb)
        vec_obs = r["params"] is None
        if vec_pred == vec_obs:
            vok += 1
        else:
            vbad += 1
            mism.append((r, "vector", vec_pred, vec_obs))
        if r["params"] is None:
            continue
        try:
            src, p = mh.select_params(r["M"], r["N"], r["K"], ta, tb, dtype, dev)
        except NotImplementedError as e:
            bad += 1
            mism.append((r, "unsupported", str(e), r["params"]))
            continue
        if list(p) == r["params"]:
            ok += 1
        else:
            bad += 1
            mism.append((r, src, list(p), r["params"]))
    gemm_rows = sum(1 for r in rows if r["params"] is not None)
    print(f"GEMM params: {ok}/{gemm_rows} match;  vector-vs-GEMM routing: {vok}/{len(rows)} match")
    if gemm_rows == 0:
        print("no 'MatMul A18 kernel' lines were logged: this GPU does not use the A18 "
              "device behavior (older than M5-class, or MPS changed). The port does not apply here.")
    for r, src, pred, obs in mism[:40]:
        print(f"  MISMATCH {r['dtype']} {r['layout']} {r['M']}x{r['N']}x{r['K']} [{src}] predicted={pred} observed={obs}")
    if len(mism) > 40:
        print(f"  ... {len(mism) - 40} more")
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
