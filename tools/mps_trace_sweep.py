"""Record which Metal kernels MPSGraph's matmul dispatches on this Mac, per shape.

This tool validates selection recovered from MPSNDArray's disassembly. It
runs ``torch.matmul`` over a shape sweep with ``tools/mbtrace.m`` loaded,
which records public Metal API calls (pipeline functions and their function
constants, ``setBytes`` contents, threadgroup memory lengths, dispatch sizes),
and writes one JSON row per case with the matmul dispatches it produced.

    python tools/mps_trace_sweep.py --out sweep_m4.json            # default sweep
    python tools/mps_trace_sweep.py --quick --out quick.json
    python tools/mps_trace_sweep.py --cases "bf16 nn 1024 1024 1024; f16 nt 33 65 129"

``tools/mps_check_sweep.py`` compares the trace with the recovered selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACER_SRC = os.path.join(REPO, "tools", "mbtrace.m")

DIMS = [1, 2, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 255, 256,
        257, 384, 500, 512, 513, 768, 1000, 1024, 1025, 1536, 2047, 2048, 2049, 4096]
KS = [1, 15, 16, 17, 32, 33, 64, 65, 128, 129, 256, 500, 512, 1000, 1024, 2048, 2049, 3000,
      4096, 8192, 8193, 16384, 32768, 65536]

CHILD = r'''
import ctypes, json, os, sys
lib = ctypes.CDLL(os.environ["MBTRACE_LIB"])
lib.mbtrace_mark.argtypes = [ctypes.c_char_p]
import torch
dts = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
for line in sys.stdin:
    fields = line.split()
    dt, lay, M, N, K = fields[:5]
    M, N, K = int(M), int(N), int(K)
    prefix = (int(fields[5]),) if len(fields) == 6 else ()
    t = dts[dt]
    a = torch.randn(*prefix, M, K, device="mps", dtype=t) if lay[0] == "n" else torch.randn(*prefix, K, M, device="mps", dtype=t).transpose(-2, -1)
    b = torch.randn(*prefix, K, N, device="mps", dtype=t) if lay[1] == "n" else torch.randn(*prefix, N, K, device="mps", dtype=t).transpose(-2, -1)
    torch.mps.synchronize()
    lib.mbtrace_mark(("begin " + " ".join(fields)).encode())
    c = a @ b
    torch.mps.synchronize()
    lib.mbtrace_mark(b"end")
    del a, b, c
print("DONE", flush=True)
'''

GEMM_HINTS = ("mmul", "MatrixMultiply", "gemm", "gemv", "Reduction", "matmul")


def default_cases(quick: bool):
    cases = []
    if quick:
        for (M, N, K) in [(1024, 1024, 1024), (4096, 4096, 4096), (2048, 2048, 2048), (256, 256, 256),
                          (128, 128, 128), (64, 64, 64), (16, 16, 16), (33, 65, 129), (1023, 1023, 1023),
                          (4097, 4097, 4097), (16, 4096, 4096), (4096, 16, 4096), (128, 128, 16384),
                          (64, 64, 65536), (2048, 14336, 4096), (96, 4096, 4096), (4096, 128, 4096)]:
            for dt in ("bf16", "f16", "f32"):
                cases.append((dt, "nn", M, N, K))
        for lay in ("nt", "tn", "tt"):
            cases += [("bf16", lay, 1024, 1024, 1024), ("bf16", lay, 333, 444, 555), ("f32", lay, 512, 512, 512)]
        return cases
    for M in DIMS:
        for N in DIMS:
            cases.append(("bf16", "nn", M, N, 1024))
    for K in KS:
        for (M, N) in [(1024, 1024), (4096, 4096), (128, 128), (64, 64), (16, 4096), (4096, 16), (256, 256),
                       (512, 512), (300, 700), (2048, 2048), (32, 32), (1000, 1000)]:
            cases.append(("bf16", "nn", M, N, K))
    for dt in ("f16", "f32"):
        for M in [1, 16, 32, 33, 64, 128, 129, 256, 512, 1000, 1024, 2048, 4096]:
            for N in [16, 32, 33, 64, 128, 256, 1024, 4096]:
                for K in [64, 1024, 4096]:
                    cases.append((dt, "nn", M, N, K))
    for lay in ("nt", "tn", "tt"):
        for (M, N, K) in [(1024, 1024, 1024), (333, 444, 555), (4096, 4096, 4096), (16, 4096, 4096),
                          (4096, 16, 4096), (128, 128, 16384), (64, 64, 64), (2048, 14336, 4096),
                          (1023, 1023, 1023), (200, 300, 400), (32, 32, 4096)]:
            for dt in ("bf16", "f32"):
                cases.append((dt, lay, M, N, K))
    seen, out = set(), []
    for c in cases:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_tracer() -> str:
    cache = os.path.join(os.path.expanduser("~"), ".cache", "metalblas")
    os.makedirs(cache, exist_ok=True)
    dylib = os.path.join(cache, "libmbtrace.dylib")
    if not os.path.exists(dylib) or os.path.getmtime(dylib) < os.path.getmtime(TRACER_SRC):
        subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "Foundation",
                        "-framework", "Metal", "-o", dylib, TRACER_SRC], check=True)
    return dylib


def parse_trace(path):
    rows, cur = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("ev") == "mark":
                text = ev.get("text", "")
                if text.startswith("begin "):
                    if cur is not None:
                        raise ValueError("Trace began a case before completing the previous case")
                    fields = text.split()[1:]
                    dt, lay, M, N, K = fields[:5]
                    cur = {"dtype": dt, "layout": lay, "M": int(M), "N": int(N), "K": int(K), "dispatches": []}
                    if len(fields) == 6:
                        cur["batch"] = int(fields[5])
                elif text == "end":
                    if cur is None:
                        raise ValueError("Trace ended a case without a begin marker")
                    rows.append(cur)
                    cur = None
                continue
            if cur is None or ev.get("ev") != "dispatch":
                continue
            name = ev.get("kernel") or ev.get("label") or ""
            if not any(h in name for h in GEMM_HINTS):
                continue
            rec = {
                "kernel": name, "grid": ev.get("grid"), "tg": ev.get("tg"), "kind": ev.get("kind"),
                "tgmem": ev.get("tgmem"), "consts": ev.get("consts"),
                "bytes": {k: v.get("hex") for k, v in (ev.get("bytes") or {}).items()},
            }
            prev = cur["dispatches"][-1] if cur["dispatches"] else None
            # Each dispatch is reported twice (wrapper and concrete encoder); keep the
            # record that carries the bound bytes.
            if prev and prev["kernel"] == name and prev["grid"] == rec["grid"] and prev["tg"] == rec["tg"] \
                    and not prev["bytes"]:
                cur["dispatches"][-1] = rec
            else:
                cur["dispatches"].append(rec)
    if cur is not None:
        raise ValueError("Trace ended before the last case completed")
    return rows


def run_sweep(cases, chunk=60):
    dylib = build_tracer()
    rows = []
    for i in range(0, len(cases), chunk):
        part = cases[i:i + chunk]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            out = tf.name
        env = dict(os.environ, MBTRACE_OUT=out, MBTRACE_LIB=dylib)
        proc = subprocess.run([sys.executable, "-c", CHILD], input="\n".join(" ".join(map(str, c)) for c in part) + "\n",
                              capture_output=True, text=True, env=env)
        if proc.returncode != 0 or "DONE" not in proc.stdout:
            sys.exit("trace child failed:\n" + proc.stderr[-3000:])
        parsed = parse_trace(out)
        observed = [tuple(r[k] for k in ("dtype", "layout", "M", "N", "K"))
                    + ((r["batch"],) if "batch" in r else ()) for r in parsed]
        if observed != [tuple(c) for c in part]:
            raise ValueError(f"Trace case coverage differs from the requested cases; raw trace: {out}")
        rows += parsed
        os.unlink(out)
        print(f"  {min(i + chunk, len(cases))}/{len(cases)} cases", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    case_args = ap.add_mutually_exclusive_group()
    case_args.add_argument("--cases", help="'dt lay M N K; ...' instead of the default sweep")
    case_args.add_argument("--cases-file", help="JSON array of [dtype, layout, M, N, K, optional batch] cases")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, REPO)
    from metalblas import mps_heuristics as mh
    dev = mh.device_info()
    if args.cases_file:
        with open(args.cases_file) as f:
            cases = json.load(f)
    elif args.cases:
        cases = [tuple(int(x) if x.isdigit() else x for x in c.split()) for c in args.cases.split(";") if c.strip()]
    else:
        cases = default_cases(args.quick)
    if not cases or any(len(c) not in (5, 6) or c[0] not in ("f16", "bf16", "f32")
                        or c[1] not in ("nn", "nt", "tn", "tt")
                        or any(not isinstance(x, int) or x <= 0 for x in c[2:]) for c in cases):
        ap.error("Each case must be [f16|bf16|f32, nn|nt|tn|tt, positive M, N, K, optional positive batch]")
    print(f"device: {dev.chip}, {dev.cores} GPU cores, MPS generation {dev.generation}; {len(cases)} cases", flush=True)
    rows = run_sweep(cases)
    meta = {"chip": dev.chip, "cores": dev.cores, "generation": dev.generation}
    import torch
    with open(TRACER_SRC, "rb") as f:
        tracer_hash = hashlib.sha256(f.read()).hexdigest()
    environment = {"macos": platform.mac_ver()[0],
                   "build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
                   "torch": torch.__version__, "python": platform.python_version(),
                   "captured_at": datetime.now(timezone.utc).isoformat(), "tracer_sha256": tracer_hash}
    with open(args.out, "w") as f:
        json.dump({"device": meta, "environment": environment, "rows": rows}, f)
    kernels = {}
    for r in rows:
        for d in r["dispatches"]:
            kernels[d["kernel"]] = kernels.get(d["kernel"], 0) + 1
    print(f"wrote {len(rows)} cases to {args.out}; kernels: {kernels}")


if __name__ == "__main__":
    main()
