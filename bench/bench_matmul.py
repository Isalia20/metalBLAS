"""Benchmark metalblas.matmul vs torch.matmul on MPS."""
import os
import sys
import time
import platform
import argparse
import subprocess
from typing import List, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF_DIR = os.path.join(REPO_ROOT, "perf_benchs")
sys.path.insert(0, REPO_ROOT)
import metalblas


def bench(M, N, K, dtype, fn, iters=100, trials=10, warmup=50):
    torch.manual_seed(0)
    a = torch.randn(M, K, device='mps', dtype=dtype)
    b = torch.randn(K, N, device='mps', dtype=dtype)
    for _ in range(warmup):
        c = fn(a, b)
    torch.mps.synchronize()
    best = float('inf')
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            c = fn(a, b)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        sec = (t1 - t0) / iters
        if sec < best:
            best = sec
    flops = 2.0 * M * N * K
    tflops = flops / best / 1e12
    return best, tflops


def run(shapes: List[Tuple[int, int, int]], dtype: torch.dtype, *, label: str = "", cool: float = 0.0):
    print(f"\n=== {label}  dtype={dtype} ===")
    print(f"{'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | {'mb ms':>9} {'TFLOPS':>7} | speedup")
    print("-" * 80)
    speedups, rows = [], []
    for (M, N, K) in shapes:
        if cool > 0:
            time.sleep(cool)
        torch_s, torch_t = bench(M, N, K, dtype, torch.matmul)
        try:
            if cool > 0:
                time.sleep(cool)
            mb_s, mb_t = bench(M, N, K, dtype, metalblas.matmul)
            ratio = torch_s / mb_s
            speedups.append(ratio)
            line = (f"{M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | "
                    f"{mb_s*1e3:>9.3f} {mb_t:>7.2f} | {ratio:>5.2f}x")
        except Exception as e:
            line = f"{M:>5} {N:>5} {K:>5} | {torch_s*1e3:>9.3f} {torch_t:>7.2f} | FAILED: {e}"
        rows.append(line)
        print(line)
    if speedups:
        import statistics
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


def format_report(chip, macos, ram, group_rows) -> str:
    cols = (f"{'M':>5} {'N':>5} {'K':>5} | {'torch ms':>9} {'TFLOPS':>7} | "
            f"{'mb ms':>9} {'TFLOPS':>7} | speedup")
    blocks, first = [], True
    for g, rows in group_rows.items():
        lines = [f"=== {g} ==="]
        if first:
            lines.append(cols)
            first = False
        lines += rows
        blocks.append("\n".join(lines))
    table = "\n\n".join(blocks)
    return (f"# {chip}\n\n"
            f"- Chip: {chip}\n"
            f"- macOS: {macos}\n"
            f"- RAM: {ram}\n\n"
            f"## bf16 results vs `torch.matmul`\n\n"
            f"```\n{table}\n```\n")


def write_perf_report(*, cool: float = 0.0) -> str:
    """Benchmark bf16 across every shape group and write perf_benchs/<chip>.md."""
    chip, macos, ram = machine_info()
    print(f"Benchmarking bf16 on {chip}  (macOS {macos}, {ram})")
    group_rows = {g: run(shapes, torch.bfloat16, label=g, cool=cool)
                  for g, shapes in SHAPES.items()}
    os.makedirs(PERF_DIR, exist_ok=True)
    path = os.path.join(PERF_DIR, report_filename(chip))
    with open(path, "w") as f:
        f.write(format_report(chip, macos, ram, group_rows))
    print(f"\nWrote {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="all", choices=["fp32", "fp16", "bf16", "all"])
    p.add_argument("--group", default="all",
                   choices=list(SHAPES.keys()) + ["all"])
    p.add_argument("--cool", type=float, default=0.0,
                   help="Seconds to sleep before each bench() call to let the GPU cool")
    p.add_argument("--report", action="store_true",
                   help="Benchmark bf16 across all shapes and write perf_benchs/<chip>.md for this Mac")
    args = p.parse_args()

    if args.report:
        write_perf_report(cool=args.cool)
        return

    dtypes = []
    if args.dtype in ("fp32", "all"): dtypes.append(torch.float32)
    if args.dtype in ("fp16", "all"): dtypes.append(torch.float16)
    if args.dtype in ("bf16", "all"): dtypes.append(torch.bfloat16)

    groups = list(SHAPES.keys()) if args.group == "all" else [args.group]

    for dt in dtypes:
        for g in groups:
            run(SHAPES[g], dt, label=g, cool=args.cool)


if __name__ == "__main__":
    main()
