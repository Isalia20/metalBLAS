"""Benchmark metalblas.matmul vs torch.matmul on MPS."""
import os
import sys
import time
import platform
import argparse
import subprocess
import statistics
from typing import List, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF_DIR = os.path.join(REPO_ROOT, "perf_benchs")
sys.path.insert(0, REPO_ROOT)
import metalblas


def _rand(M, K, dtype):
    if not (dtype.is_floating_point or dtype.is_complex):
        lo = 0 if dtype == torch.uint8 else -4
        return torch.randint(lo, 4, (M, K), device='mps', dtype=dtype)
    return torch.randn(M, K, device='mps', dtype=dtype)


def _time_call(fn, a, b, iters, trials, warmup):
    """Best per-call seconds for fn(a, b), plus total GPU-busy wall seconds."""
    t_start = time.perf_counter()
    for _ in range(warmup):
        c = fn(a, b)
    torch.mps.synchronize()
    best = float('inf')
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            c = fn(a, b)
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best, time.perf_counter() - t_start


def _cooldown(busy_s, cool):
    """Rest in proportion to how long the last bench ran (cool=ratio of busy time)."""
    if cool > 0:
        time.sleep(cool * busy_s)


def bench(M, N, K, dtype, fn, iters=100, trials=10, warmup=50):
    torch.manual_seed(0)
    a = _rand(M, K, dtype)
    b = _rand(K, N, dtype)
    best, busy = _time_call(fn, a, b, iters, trials, warmup)
    # Complex MAC is ~4 real MACs (the four ar/ai x br/bi products), so count 4x.
    flops = 2.0 * M * N * K * (4.0 if dtype.is_complex else 1.0)
    return best, flops / best / 1e12, busy


def run(shapes: List[Tuple[int, int, int]], dtype: torch.dtype, *, label: str = "", cool: float = 0.0):
    print(f"\n=== {label}  dtype={dtype} ===")
    print(f"{'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | {'mb ms':>9} {'TFLOPS':>7} | speedup")
    print("-" * 80)
    speedups, rows = [], []
    for (M, N, K) in shapes:
        torch_s, torch_t, busy = bench(M, N, K, dtype, torch.matmul)
        _cooldown(busy, cool)
        try:
            mb_s, mb_t, busy = bench(M, N, K, dtype, metalblas.matmul)
            _cooldown(busy, cool)
            ratio = torch_s / mb_s
            speedups.append(ratio)
            line = (f"{M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | "
                    f"{mb_s*1e3:>9.3f} {mb_t:>7.2f} | {ratio:>5.2f}x")
        except Exception as e:
            line = f"{M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | FAILED: {e}"
        rows.append(line)
        print(line)
    if speedups:
        print(f"Median speedup: {statistics.median(speedups):.2f}x  "
              f"Mean: {statistics.mean(speedups):.2f}x  "
              f"Min: {min(speedups):.2f}x  Max: {max(speedups):.2f}x")
    return rows


SHAPES = {
    "square": [(s, s, s) for s in [128, 256, 512, 1024, 2048, 4096]],
    "tall": [(4096, 1024, 1024), (1024, 4096, 1024), (8192, 1024, 1024),
             (2048, 2048, 8192)],
    "attn": [(4096, 4096, 64), (4096, 4096, 128), (4096, 4096, 1024)],
    "gemv": [(1, 4096, 4096), (4096, 1, 4096), (1, 1024, 1024),
             (1, 32000, 4096)],
    "llm": [(2048, 14336, 4096), (4096, 4096, 11008), (4096, 11008, 4096)],
    # thin-M (small rows, wide N): batched-decode / small-batch prefill GEMM.
    "thin_m": [(96, 4096, 4096), (128, 4096, 4096), (192, 4096, 4096),
               (256, 4096, 4096), (128, 8192, 4096)],
    # thin-N (wide M, narrow N): below MPS here
    "thin_n": [(4096, 128, 4096), (4096, 256, 4096), (8192, 128, 4096),
               (4096, 64, 4096)],
    "odd": [(257, 257, 257), (1023, 1023, 1023), (4097, 4097, 4097),
            (511, 511, 511), (333, 444, 555)],
}


def _rm(r, c, dt):     # row-major contiguous, stride (c, 1) -> packed mpp_tensor fast path
    return _rand(r, c, dt)
def _cm(r, c, dt):     # col-major view, stride (1, r) -> trans flag, mpp/simd fallback
    return _rand(c, r, dt).t()
def _rowsl(r, c, dt):  # [::2] over rows, stride (2c, 1) -> non-packed (lda=2c), read in place
    return _rand(2 * r, c, dt)[::2]
def _colsl(r, c, dt):  # [:, ::2], stride (2c, 2) -> unit stride on neither dim, contiguified
    return _rand(r, 2 * c, dt)[:, ::2]
def _off(r, c, dt):    # [1:, 1:] sub-view, stride (c+1, 1) -> nonzero storage offset, lda=c+1
    return _rand(r + 1, c + 1, dt)[1:, 1:]

# name -> (build_A, build_B); A is (M,K), B is (K,N). rm_rm is the layout-tax baseline.
LAYOUTS = {
    "rm_rm":  (_rm,    _rm),     # baseline: both row-major (packed)
    "rm_cm":  (_rm,    _cm),     # row-major A, col-major B
    "cm_rm":  (_cm,    _rm),     # col-major A, row-major B
    "cm_cm":  (_cm,    _cm),     # both col-major (A.t() @ B.t())
    "sliced": (_rowsl, _rowsl),  # both [::2] over rows (strided lda, unit inner stride)
    "colsl":  (_colsl, _colsl),  # both [:, ::2] (fully strided, forces a contiguous copy)
    "offset": (_off,   _off),    # both [1:, 1:] (nonzero storage offset, lda=K+1)
}


def bench_layout(M, N, K, dtype, build_a, build_b, fn, iters=100, trials=10, warmup=50):
    torch.manual_seed(0)
    a = build_a(M, K, dtype)
    b = build_b(K, N, dtype)
    best, busy = _time_call(fn, a, b, iters, trials, warmup)
    flops = 2.0 * M * N * K * (4.0 if dtype.is_complex else 1.0)
    return best, flops / best / 1e12, busy


def _layout_ok(M, N, K, dtype, build_a, build_b):
    """Correctness probe: mb vs torch on the identical strided views."""
    torch.manual_seed(0)
    a, b = build_a(M, K, dtype), build_b(K, N, dtype)
    ref, got = torch.matmul(a, b), metalblas.matmul(a, b)
    tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
    return torch.allclose(got.float(), ref.float(), rtol=tol, atol=tol)


def run_layouts(shapes, dtype, layout_names, *, label="", cool=0.0, check=False):
    print(f"\n=== {label}  dtype={dtype}  [layout sweep] ===")
    hdr = (f"{'layout':>7} {'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | "
           f"{'mb ms':>9} {'TFLOPS':>7} | speedup")
    print(hdr + ("  chk" if check else ""))
    print("-" * 90)
    rows, mb_tf = [], {}
    for name in layout_names:
        build_a, build_b = LAYOUTS[name]
        mb_tf[name], speedups = {}, []
        for (M, N, K) in shapes:
            torch_s, torch_t, busy = bench_layout(M, N, K, dtype, build_a, build_b, torch.matmul)
            _cooldown(busy, cool)
            try:
                mb_s, mb_t, busy = bench_layout(M, N, K, dtype, build_a, build_b, metalblas.matmul)
                _cooldown(busy, cool)
                ratio = torch_s / mb_s
                speedups.append(ratio)
                mb_tf[name][(M, N, K)] = mb_t
                chk = ("  ok" if _layout_ok(M, N, K, dtype, build_a, build_b) else "  XX") if check else ""
                line = (f"{name:>7} {M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | "
                        f"{mb_s*1e3:>9.3f} {mb_t:>7.2f} | {ratio:>5.2f}x{chk}")
            except Exception as e:
                line = (f"{name:>7} {M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | "
                        f"FAILED: {e}")
            rows.append(line)
            print(line)
        if speedups:
            print(f"  {name}: median {statistics.median(speedups):.2f}x vs torch "
                  f"(min {min(speedups):.2f}x, max {max(speedups):.2f}x)")
    # Internal layout tax: mb throughput on each layout relative to packed rm_rm.
    base = mb_tf.get("rm_rm")
    if base:
        print("  -- mb layout tax (TFLOPS / rm_rm, median over shapes) --")
        for name in layout_names:
            if name == "rm_rm":
                continue
            t = [mb_tf[name][s] / base[s] for s in base if s in mb_tf[name] and base[s] > 0]
            if t:
                print(f"     {name:>7}: {statistics.median(t):.2f}x")
    return rows


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return ""


def machine_info():
    """(chip, macOS version, RAM) for the current Mac."""
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "Unknown chip"
    macos = platform.mac_ver()[0] or _sysctl("kern.osproductversion") or "unknown"
    mem = _sysctl("hw.memsize")
    ram = f"{round(int(mem) / 1024**3)} GB" if mem.isdigit() else "unknown"
    return chip, macos, ram


def report_filename(chip: str) -> str:
    """'Apple M5 Pro' -> 'M5_Pro.md'."""
    name = chip.replace("Apple", "").strip().replace(" ", "_") or "Unknown"
    return f"{name}.md"


def _blocks(group_rows, cols) -> str:
    """Join per-group row lists into '=== group ===' blocks, header on the first."""
    blocks, first = [], True
    for g, rows in group_rows.items():
        lines = [f"=== {g} ==="]
        if first:
            lines.append(cols)
            first = False
        lines += rows
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_report(chip, macos, ram, group_rows, layout_rows=None) -> str:
    cols = (f"{'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | "
            f"{'mb ms':>9} {'TFLOPS':>7} | speedup")
    out = (f"# {chip}\n\n"
           f"- Chip: {chip}\n"
           f"- macOS: {macos}\n"
           f"- RAM: {ram}\n\n"
           f"## bf16 results vs `torch.matmul`\n\n"
           f"```\n{_blocks(group_rows, cols)}\n```\n")
    if layout_rows:
        lcols = (f"{'layout':>7} {'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | "
                 f"{'mb ms':>9} {'TFLOPS':>7} | speedup")
        out += ("\n## bf16 layout sweep vs `torch.matmul`\n\n"
                "Same shapes, awkward operand strides: rm=row-major, cm=col-major view, "
                "sliced=`[::2]` rows, colsl=`[:, ::2]`, offset=`[1:, 1:]`.\n\n"
                f"```\n{_blocks(layout_rows, lcols)}\n```\n")
    return out


def write_perf_report(*, cool: float = 0.0, layout_names=None, check: bool = False) -> str:
    """Benchmark bf16 across every shape group and write perf_benchs/<chip>.md.
    With layout_names, also run + append a same-shapes layout sweep."""
    chip, macos, ram = machine_info()
    print(f"Benchmarking bf16 on {chip}  (macOS {macos}, {ram})")
    group_rows = {g: run(shapes, torch.bfloat16, label=g, cool=cool)
                  for g, shapes in SHAPES.items()}
    layout_rows = None
    if layout_names:
        layout_rows = {g: run_layouts(shapes, torch.bfloat16, layout_names,
                                      label=g, cool=cool, check=check)
                       for g, shapes in SHAPES.items()}
    os.makedirs(PERF_DIR, exist_ok=True)
    path = os.path.join(PERF_DIR, report_filename(chip))
    with open(path, "w") as f:
        f.write(format_report(chip, macos, ram, group_rows, layout_rows))
    print(f"\nWrote {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="all",
                   choices=["fp32", "fp16", "bf16", "c64", "c32",
                            "i8", "u8", "i16", "i32", "i64", "all"])
    p.add_argument("--group", default="all",
                   choices=list(SHAPES.keys()) + ["all"])
    p.add_argument("--layout", default=None,
                   help="Run the layout sweep over --group instead of the dtype sweep. "
                        "Comma-list of {" + ",".join(LAYOUTS) + "} or 'all'. Float dtypes only "
                        "(bf16 default); combine with --report to append a layout section.")
    p.add_argument("--check", action="store_true",
                   help="Layout sweep only: verify metalblas matches torch on each strided view")
    p.add_argument("--cool", type=float, default=0.0,
                   help="Rest after each bench to avoid thermal throttling: sleep this "
                        "ratio of the time the kernel just ran (1.0 = rest as long as it "
                        "worked, scales with shape). 0 disables.")
    p.add_argument("--report", action="store_true",
                   help="Benchmark bf16 across all shapes and write perf_benchs/<chip>.md for this Mac")
    args = p.parse_args()

    layout_names = None
    if args.layout:
        layout_names = list(LAYOUTS) if args.layout == "all" else args.layout.split(",")
        bad = [n for n in layout_names if n not in LAYOUTS]
        if bad:
            p.error(f"unknown layout(s) {bad}; choose from {list(LAYOUTS)} or 'all'")

    if args.report:
        write_perf_report(cool=args.cool, layout_names=layout_names, check=args.check)
        return

    groups = list(SHAPES.keys()) if args.group == "all" else [args.group]

    if layout_names:
        # Layout sweep is float-only; map fp32/fp16/bf16, default bf16 for anything else.
        ldtype = {"fp32": torch.float32, "fp16": torch.float16,
                  "bf16": torch.bfloat16}.get(args.dtype, torch.bfloat16)
        for g in groups:
            run_layouts(SHAPES[g], ldtype, layout_names, label=g, cool=args.cool, check=args.check)
        return

    dtypes = []
    if args.dtype in ("fp32", "all"): dtypes.append(torch.float32)
    if args.dtype in ("fp16", "all"): dtypes.append(torch.float16)
    if args.dtype in ("bf16", "all"): dtypes.append(torch.bfloat16)
    if args.dtype in ("c64",): dtypes.append(torch.complex64)
    if args.dtype in ("c32",): dtypes.append(torch.complex32)
    if args.dtype in ("i8",):  dtypes.append(torch.int8)
    if args.dtype in ("u8",):  dtypes.append(torch.uint8)
    if args.dtype in ("i16",): dtypes.append(torch.int16)
    if args.dtype in ("i32",): dtypes.append(torch.int32)
    if args.dtype in ("i64",): dtypes.append(torch.int64)

    for dt in dtypes:
        for g in groups:
            run(SHAPES[g], dt, label=g, cool=args.cool)


if __name__ == "__main__":
    main()
