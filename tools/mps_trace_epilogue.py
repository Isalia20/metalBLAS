"""Capture MPSGraph's linked epilogues and their Metal dispatches; never times kernels.

    python tools/mps_trace_epilogue.py --out epilogues.json.gz

Optional --cases-file accepts JSON objects with dtype (f32/f16/bf16), layout
(nn/nt/tn/tt), shape [M,N,K], op, alpha and beta. Batched operations use `batch`
broadcast copies of the inputs (default three). The tracer's DAG SPI exposes the actual linked
operation order and scalar constants; AIR module identities are documented in
docs/mps_dispatch_sources.json.
"""
import argparse
import ctypes
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.mps_trace_sweep import build_tracer


def default_cases():
    return [dict(dtype=dtype, layout="nn", shape=shape, op=op, alpha=alpha, beta=beta)
            for dtype in ("f32", "f16", "bf16")
            for shape in ([64, 64, 128], [31, 49, 8193])
            for op in ("matmul", "addmm", "bmm", "baddbmm", "addbmm")
            for alpha, beta in ([(1., 0.)] if op in ("matmul", "bmm") else
                                [(1., 1.), (.73, .37), (1., 0.), (0., .37)])]


def parse_trace(path):
    rows, current = [], None
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["ev"] == "mark":
            marker = event["text"]
            if marker.startswith("begin "):
                if current is not None:
                    raise ValueError("Nested trace cases")
                current = {"case": json.loads(marker[6:]), "dags": [], "dispatches": []}
            elif marker == "end":
                if current is None:
                    raise ValueError("Unmatched trace end")
                rows.append(current)
                current = None
        elif current is not None and event["ev"] == "dag":
            current["dags"].append({"description": event["description"], "functions": [
                {"name": f["name"], "module": f["info"]["name"], "library": f["info"].get("lib", "")}
                for f in event["functions"]]})
        elif current is not None and event["ev"] == "dispatch":
            row = {k: event[k] for k in ("kernel", "grid", "tg", "tgmem", "bytes", "bufs", "kind")}
            row.update({k: event[k] for k in ("maxThreads", "staticTG") if k in event})
            previous = current["dispatches"][-1] if current["dispatches"] else None
            if previous and all(previous[k] == row[k] for k in ("kernel", "grid", "tg")) and not previous["bytes"]:
                current["dispatches"][-1] = row
            else:
                current["dispatches"].append(row)
    if current is not None:
        raise ValueError("Incomplete final trace case")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--reference", choices=("torch", "mpsgraph"), default="torch",
                        help="Use MPSGraph directly to avoid PyTorch's private Metal fallbacks")
    args = parser.parse_args()
    cases = json.loads(args.cases_file.read_text()) if args.cases_file else default_cases()
    if args.reference == "mpsgraph" and any(c["op"] not in ("addmm", "baddbmm", "addbmm") for c in cases):
        parser.error("The direct MPSGraph reference supports addmm, baddbmm and addbmm cases")
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as trace:
        os.environ["MBTRACE_DAG"] = "1"
        os.environ["MBTRACE_OUT"] = trace.name
        lib = ctypes.CDLL(build_tracer())
        lib.mbtrace_mark.argtypes = [ctypes.c_char_p]
        import torch
        from metalblas.mps_heuristics import device_info
        if args.reference == "mpsgraph":
            from tools.mps_graph_reference import addmm as graph_addmm

        torch.manual_seed(6416)
        dtypes = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
        for case in cases:
            dtype, layout = dtypes[case["dtype"]], case["layout"]
            m, n, k = case["shape"]
            a = torch.randn((k, m) if layout[0] == "t" else (m, k), dtype=dtype, device="mps")
            b = torch.randn((n, k) if layout[1] == "t" else (k, n), dtype=dtype, device="mps")
            if layout[0] == "t": a = a.t()
            if layout[1] == "t": b = b.t()
            bias = torch.randn(n, dtype=dtype, device="mps")
            op, alpha, beta = case["op"], case["alpha"], case["beta"]
            if op in ("bmm", "baddbmm", "addbmm"):
                batch = case.get("batch", 3)
                a, b = a[None].expand(batch, -1, -1), b[None].expand(batch, -1, -1)
            torch.mps.synchronize()
            lib.mbtrace_mark(("begin " + json.dumps(case)).encode())
            if args.reference == "mpsgraph":
                result = graph_addmm(bias, a, b, alpha=alpha, beta=beta, reduce_batch=op == "addbmm")
            elif op in ("matmul", "bmm"):
                result = getattr(torch, op)(a, b)
            else:
                result = getattr(torch, op)(bias, a, b, alpha=alpha, beta=beta)
            torch.mps.synchronize()
            lib.mbtrace_mark(b"end")
            del result
        rows = parse_trace(Path(trace.name))
    if [r["case"] for r in rows] != cases:
        raise ValueError("Trace coverage differs from the requested cases")
    sources = ["tools/mbtrace.m", "tools/mps_trace_epilogue.py"]
    if args.reference == "mpsgraph":
        sources += ["tools/mps_graph_reference.py", "tools/mps_graph_reference.m"]
    result = {"macos": platform.mac_ver()[0], "reference": args.reference,
              "build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
              "device": asdict(device_info()), "torch": torch.__version__,
              "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
                                for p in sources},
              "cases": rows}
    encoded = (json.dumps(result, separators=(",", ":")) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(encoded, mtime=0) if args.out.suffix == ".gz" else encoded)
    print(f"Captured {len(rows)} cases and {sum(len(r['dags']) for r in rows)} linked DAGs")


if __name__ == "__main__":
    main()
