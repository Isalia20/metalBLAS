"""Tabulate the recovered MPS matmul dispatch numbers for every M1–M5 GPU.

    python tools/mps_dispatch_numbers.py --out docs/mps_dispatch_numbers.md

Every value comes from the selection code that metalBLAS dispatches with
(``metalblas/mps_a14.py``, ``mps_heuristics.py``, ``mps_chunk.py``,
``mps_reduction.py``, ``mps_submission.py``), which transcribes MPSCore,
MPSNDArray and MPSGraph. Nothing here is measured or tuned; the tool only
evaluates those rules for each (MPS generation, GPU core count).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from metalblas import mps_a14, mps_heuristics as mh

# (label, GPU cores). Core counts are the shipped GPU configurations; the
# generation code is derived from them exactly as MPSCore does.
CHIPS = (
    ("M1", 7), ("M1", 8), ("M1 Pro", 14), ("M1 Pro", 16), ("M1 Max", 24), ("M1 Max", 32),
    ("M1 Ultra", 48), ("M1 Ultra", 64),
    ("M2", 8), ("M2", 10), ("M2 Pro", 16), ("M2 Pro", 19), ("M2 Max", 30), ("M2 Max", 38),
    ("M2 Ultra", 60), ("M2 Ultra", 76),
    ("M3", 8), ("M3", 10), ("M3 Pro", 14), ("M3 Pro", 18), ("M3 Max", 30), ("M3 Max", 40),
    ("M3 Ultra", 60), ("M3 Ultra", 80),
    ("M4", 8), ("M4", 10), ("M4 Pro", 16), ("M4 Pro", 20), ("M4 Max", 32), ("M4 Max", 40),
    ("M5", 10), ("M5 Pro", 16), ("M5 Pro", 20), ("M5 Max", 32), ("M5 Max", 40),
)
# Configurations whose dispatches were traced (see tests/fixtures/mps).
VALIDATED = {("M2", 10): "M2 / macOS 26.3", ("M3 Ultra", 80): "M3 Ultra / macOS 26.5.2",
             ("M4", 10): "M4 / macOS 26.6", ("M5 Pro", 16): "M5 Pro / macOS 27.2"}

DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
EXAMPLES = (
    ("bf16", 4096, 4096, 4096), ("bf16", 1024, 1024, 1024), ("bf16", 1025, 1025, 1024),
    ("bf16", 2048, 14336, 4096), ("bf16", 4096, 16, 4096), ("bf16", 16, 4096, 4096),
    ("bf16", 128, 128, 16384), ("bf16", 8, 4096, 1024), ("f16", 1024, 1024, 1024),
    ("f32", 1024, 1024, 1024), ("f32", 4096, 4096, 4096),
)


def device(chip, cores):
    return mh.DeviceInfo(f"Apple {chip}", cores, mh._generation_from_name(f"Apple {chip}", cores))


def macos_for(dev):
    # M5 16-bit products use the A18 entry point, recovered from macOS 27.
    return 27 if dev.generation >= 22 else 26


def families(dev):
    g = dev.generation
    a14 = "NNA14" if g <= 17 else "a16"
    half = "gemm_a18" if g >= 22 else a14
    if g >= 22:
        branch = "low-core" if dev.cores <= 12 else ("high-core" if dev.cores < 60 else "high-core, 60+ branch")
        a18 = f"{dev.table} table, then {branch}"
    else:
        a18 = "—"
    return half, a14, a18


def submission(dev):
    effective = dev.cores * (3 if dev.generation >= 22 else 1)
    if effective >= 20:
        factor = min(7 * effective, 512)
        return f"11 / {factor * 3125:,}", f"{factor * 300_000 >> 6:,}"
    return "8 / 200,000", "300,000"


def summary_rows():
    rows = []
    for chip, cores in CHIPS:
        dev = device(chip, cores)
        half, a14, a18 = families(dev)
        g, c = dev.generation, cores
        if g <= 17:
            cubic = "≤64-core rules" if c <= 64 else ">64-core rules"
        else:
            cubic = "fixed (32,32,16) f16/bf16, (32,64,32) f32"
        first, later = submission(dev)
        rows.append([
            f"{chip} {cores}-core", str(g), half, a14, a18,
            "32" if g >= 18 else "16", f"{4 * c}", cubic,
            f"{4096 * c:,}", f"{8 * c}/{16 * c}/{32 * c}", f"{16 * c} (26), {32 * c}/{16 * c} (27)",
            "1024" if c < 9 else "8192", "yes" if c > 63 else "no", first, later,
            VALIDATED.get((chip, cores), "binary only"),
        ])
    return rows


def describe_a14(dev, dtype, M, N, K, major):
    if mps_a14.vector_preferred(M, N, K, dtype, dev=dev, macos_major=major):
        p = mps_a14.vector_plan(M, N, K, dtype, dev=dev, macos_major=major)
        return f"{p.kernel.replace('NDArrayVectorMatrixMultiply', 'vector')} grid {p.grid[0]}×{p.grid[1]}"
    p = mps_a14.gemm_plan(M, N, K, dtype, dev=dev, macos_major=major)
    name = "NNA14" + ("·edge" if p.kernel.endswith("_EdgeCase") else "") if "NNA14" in p.kernel else "a16"
    stage = "".join(x for x, s in (("A", p.stage_a), ("B", p.stage_b)) if s) or "none"
    split = f", {p.splits}×{p.k_per_split} K" if p.splits > 1 else ""
    return f"{name} {p.tile_m}×{p.tile_n}×{p.tile_k}, {p.threads} thr, stage {stage}{split}"


def describe_a18(dev, dtype, M, N, K):
    if mh.vector_kernel_preferred(M, N, K, dtype, dev):
        p = mps_a14.vector_plan(M, N, K, dtype, dev=dev, macos_major=27)
        return f"{p.kernel.replace('NDArrayVectorMatrixMultiply', 'vector')} grid {p.grid[0]}×{p.grid[1]}"
    source, q = mh.select_params(M, N, K, False, False, dtype, dev)
    plan = mh.MPSPlan(source, q)
    tm, tn = plan.tile
    split = ""
    if q[5] > 1:
        split = f", {q[5]} {'global' if q[6] else 'local'} K splits"
    return f"a18 {tm}×{tn}×{plan.tile_k}, {plan.simdgroups} SG, uber {q[8]}×{q[9]}{split} ({source})"


def example_rows(chips=CHIPS, major=None):
    rows = []
    for chip, cores in chips:
        dev = device(chip, cores)
        cells = [f"{chip} {cores}"]
        for dt, M, N, K in EXAMPLES:
            dtype = DT[dt]
            if dev.generation >= 22 and dtype is not torch.float32:
                cells.append(describe_a18(dev, dtype, M, N, K))
            else:
                cells.append(describe_a14(dev, dtype, M, N, K, major or macos_for(dev)))
        rows.append(cells)
    return rows


# One configuration per pre-M5 branch, for the macOS 27 encoder profile.
MACOS27_CHIPS = (("M1", 8), ("M1 Max", 32), ("M2", 10), ("M2 Ultra", 76), ("M3 Ultra", 80), ("M4", 10))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render():
    head = ["GPU", "MPS gen", "f16/bf16 GEMM", "fp32 GEMM", "A18 selection",
            "split-K quantum", "A14 shrink capacity (tiles)", "A14 cubic tiles",
            "vector T=9 when M·N·w ≥", "vector launch W steps", "vector coverage target",
            "host chunk", "addbmm >63-core path", "submission ops/values (idle)",
            "submission values (busy)", "traced"]
    ex_head = ["GPU"] + [f"{dt} {M}×{N}×{K}" for dt, M, N, K in EXAMPLES]
    return "\n".join([
        "# MPS matmul dispatch numbers for every M1–M5 GPU",
        "",
        "Generated by `python tools/mps_dispatch_numbers.py --out docs/mps_dispatch_numbers.md`.",
        "Each value is the recovered MPSCore/MPSNDArray/MPSGraph rule evaluated for that",
        "GPU's core count and MPS generation code; nothing is measured. The rules and their",
        "binary provenance are described in [mps_matmuls.md](mps_matmuls.md),",
        "[mps_matmul_heuristics.md](mps_matmul_heuristics.md) and",
        "[mps_dispatch_sources.json](mps_dispatch_sources.json).",
        "",
        "The `traced` column marks configurations whose MPS dispatches were captured and",
        "match these rules in `tests/fixtures`. Others use the same binary branches with",
        "their own core count. M1–M4 rows use the macOS 26 encoder profile; M5 rows use",
        "macOS 27, which the A18 entry point requires. `w` is 4 for fp32 and 3 for f16/bf16.",
        "",
        "## Per-GPU constants",
        "",
        table(head, summary_rows()),
        "",
        "Generation codes come from MPSCore's device constructor: Apple7 (M1) 11/12/13/14/15",
        "below 5/9/17/33/65 cores, Apple8 (M2) 16 below 10 cores else 17, Apple9 (M3) 18",
        "below 11 else 19, M4 20 below 11 else 21, Apple10 (M5) 22 below 11 else 23.",
        "Generations 11–17 run `NDArrayMatrixMultiplyNNA14[_EdgeCase]`; 18–21 and M5 fp32",
        "run `mmul_kernel_a16_<T>_<T>_float`; M5 f16/bf16 run `gemm_a18` (exact-key table",
        "G below 11 cores, S below 21, C otherwise, then the core-count heuristic).",
        "",
        "## Selected schedules for sample products",
        "",
        "Row-major (`nn`) operands, batch 1. A14-family cells give tile M×N×K, threads,",
        "staged operands and split K; A18 cells give the threadgroup tile, SIMD groups,",
        "uber tile and K splits with the parameter source.",
        "",
        table(ex_head, example_rows()),
        "",
        "### The same products under the macOS 27 encoder",
        "",
        "The 27.2 A14 encoder splits K with rounded-up divisions and an 8192 cap, and adds",
        "a branch for generation 17 above 64 cores (the 76-core M2 Ultra) that flattens the",
        "grid onto x. These rows evaluate that profile; its pre-M5 branches were recovered",
        "from the 27.2 binary and are exercised on hardware only through the M5 fp32 path.",
        "",
        table(ex_head, example_rows(MACOS27_CHIPS, 27)),
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render()
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
